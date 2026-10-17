"""Official IP102 splits and straightforward image classification data loading.

No images are copied and no new random train/val/test split is created.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import csv
import json

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = False
SPLITS = ("train", "val", "test")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Record:
    filename: str
    label: int


def find_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if not (root / "train.txt").exists() and (root / "ip102_v1.1" / "train.txt").exists():
        root = root / "ip102_v1.1"
    if not (root / "train.txt").is_file():
        raise FileNotFoundError(
            f"Could not locate IP102 classification train.txt in {root}. "
            "Extract the official classification archive, not the detection archive."
        )
    return root


def parse_split(path: Path) -> list[Record]:
    records: list[Record] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Official format is "image.jpg integer_label"; also allow CSV files.
        parts = [x.strip().strip('"') for x in
                 (next(csv.reader([line])) if "," in line else line.split())]
        if len(parts) != 2:
            raise ValueError(f"{path.name}:{number}: expected filename and label, got {line!r}")
        filename, label = parts
        try:
            label_id = int(label)
        except ValueError:
            if number == 1 and filename.lower() in ("image", "filename", "path"):
                continue
            raise ValueError(f"{path.name}:{number}: invalid numeric label {label!r}")
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError(f"{path.name}:{number}: unsafe image path {filename!r}")
        records.append(Record(filename.replace("\\", "/"), label_id))
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def load_splits(root: str | Path, num_classes: int = 102, check_files: bool = True):
    root = find_root(root)
    raw = {split: parse_split(root / f"{split}.txt") for split in SPLITS}
    label_set = {item.label for entries in raw.values() for item in entries}
    if min(label_set) == 0 and max(label_set) < num_classes:
        offset = 0
    elif min(label_set) >= 1 and max(label_set) == num_classes:
        offset = 1
    else:
        raise ValueError(
            f"Unexpected label range {min(label_set)}..{max(label_set)} for {num_classes} classes. "
            "Expected zero-based 0..101 or one-based 1..102."
        )
    splits = {
        s: [Record(r.filename, r.label - offset) for r in records]
        for s, records in raw.items()
    }
    seen: dict[str, str] = {}
    missing: list[str] = []
    for split, entries in splits.items():
        for r in entries:
            if not 0 <= r.label < num_classes:
                raise ValueError(f"{split}: label outside 0..{num_classes - 1}: {r}")
            if r.filename in seen:
                raise ValueError(f"Image {r.filename!r} appears in both {seen[r.filename]} and {split}.")
            seen[r.filename] = split
            if check_files and not (root / "images" / r.filename).is_file():
                if len(missing) < 10:
                    missing.append(r.filename)
    if missing:
        raise FileNotFoundError(
            f"Images missing from {root / 'images'}, examples: {missing}. "
            "Check your extracted Classification archive."
        )
    if set(range(num_classes)) - {r.label for r in splits["train"]}:
        raise ValueError("Some IP102 classes have no training images. Check the split and label files.")
    return root, splits, offset


def dataset_summary(splits: dict[str, list[Record]], offset: int, num_classes: int):
    report = {"original_label_offset": offset, "splits": {}}
    fingerprints = {}
    for split, entries in splits.items():
        counts = np.bincount([r.label for r in entries], minlength=num_classes)
        digest = sha256()
        for r in entries:
            digest.update(f"{r.filename}\t{r.label}\n".encode())
        fingerprints[split] = digest.hexdigest()
        report["splits"][split] = {
            "total": len(entries), "present_classes": int((counts > 0).sum()),
            "min_class_count": int(counts.min()), "max_class_count": int(counts.max()),
            "per_class": {str(i): int(count) for i, count in enumerate(counts)}
        }
    return report, fingerprints


class IP102Dataset(Dataset):
    def __init__(self, root: Path, records: list[Record], transform):
        self.root = root / "images"
        self.records = records
        self.targets = [r.label for r in records]
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record.filename) as src:
            img = src.convert("RGB")
        return self.transform(img), record.label, index



def build_transform(size: int, training: bool = False, aug: dict | None = None):
    """No on-the-fly augmentation: augmented images were saved during preparation.

    Resize/crop and ImageNet normalization are deterministic for all splits.
    """
    return T.Compose([
        T.Resize(int(size * 256 / 224)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def make_loader(dataset: IP102Dataset, batch_size: int, workers: int,
                training: bool, seed: int = 42, device: str = "cpu"):
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be >=1 and workers must be >=0")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=device == "cuda",
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
