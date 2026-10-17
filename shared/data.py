"""Official IP102 classification splits, strict integrity checks and train-only balancing.

No images are copied and no new random train/val/test split is created.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import csv
import json
import random
import re

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
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


class EpochWeightedSampler(WeightedRandomSampler):
    """Weighted draws reproducible by epoch (including after checkpoint resume)."""
    def __init__(self, weights, num_samples: int, seed: int):
        super().__init__(weights, num_samples=num_samples, replacement=True)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.multinomial(self.weights, self.num_samples,
                                      self.replacement, generator=generator).tolist())


class ExactBalancedSampler(Sampler[int]):
    """Exactly K sampled indices per class per epoch, with replacement if needed."""
    def __init__(self, labels: list[int], samples_per_class: int, seed: int = 0):
        self.groups: dict[int, list[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            self.groups[label].append(i)
        self.samples_per_class = samples_per_class
        self.seed = seed
        self.epoch = 0
        if not self.groups or samples_per_class < 1:
            raise ValueError("Need nonempty classes and samples_per_class >= 1")

    def __len__(self):
        return len(self.groups) * self.samples_per_class

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = []
        for group in self.groups.values():
            if len(group) >= self.samples_per_class:
                indices.extend(rng.sample(group, self.samples_per_class))
            else:
                indices.extend(rng.choices(group, k=self.samples_per_class))
        rng.shuffle(indices)
        return iter(indices)


def build_transform(size: int, training: bool, aug: dict):
    if training:
        steps = []
        if aug.get("random_resized_crop", True):
            steps.append(T.RandomResizedCrop(size, scale=(0.75, 1.0)))
        else:
            steps += [T.Resize(int(size * 1.14)), T.RandomCrop(size)]
        if aug.get("horizontal_flip", True):
            steps.append(T.RandomHorizontalFlip())
        if aug.get("rotation_degrees", 0):
            steps.append(T.RandomRotation(aug["rotation_degrees"]))
        if aug.get("color_jitter", 0):
            steps.append(T.ColorJitter(brightness=aug["color_jitter"],
                                       contrast=aug["color_jitter"],
                                       saturation=aug["color_jitter"]))
    else:
        steps = [T.Resize(int(size * 256 / 224)), T.CenterCrop(size)]
    steps.extend([T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose(steps)


def make_loader(dataset: IP102Dataset, batch_size: int, workers: int, training: bool,
                balance: str = "none", seed: int = 42, equal_k: int = 100,
                device: str = "cpu"):
    sampler = None
    if training and balance == "weighted_sampler":
        counts = Counter(dataset.targets)
        weights = torch.tensor([1.0 / counts[t] for t in dataset.targets], dtype=torch.double)
        sampler = EpochWeightedSampler(weights, num_samples=len(dataset), seed=seed)
    elif training and balance == "equal_epoch":
        sampler = ExactBalancedSampler(dataset.targets, equal_k, seed)
    elif balance not in ("none", "weighted_loss"):
        raise ValueError(f"Unknown balance mode: {balance}")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=training and sampler is None,
        sampler=sampler, num_workers=workers, pin_memory=device == "cuda",
        persistent_workers=workers > 0, generator=generator
    )


def class_weights(labels: list[int], n: int, device):
    counts = np.bincount(labels, minlength=n).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Cannot weight class with zero training examples")
    weights = counts.sum() / (n * counts)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)
