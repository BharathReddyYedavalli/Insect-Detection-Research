"""Create a class-balanced image classification dataset *before* training.

Only training images are augmented. Validation and test images are copied/
hard-linked unchanged and never used to synthesize training data.

Examples:
  python -m shared.prepare_balanced --target-per-class 600
  python -m shared.prepare_balanced             # all 3,444/class for IP102
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import os
import random
import shutil

from PIL import Image, ImageEnhance, ImageOps
import numpy as np

from shared.config import PROJECT
from shared.data import dataset_summary, load_splits


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if target.stat().st_size == source.stat().st_size:
            return
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _make_augmented(source: Path, destination: Path, rng: random.Random, size: int):
    """Simple crop, optional horizontal flip, small rotation and color jitter."""
    with Image.open(source) as original:
        img = ImageOps.exif_transpose(original).convert("RGB")
    width, height = img.size
    fraction = rng.uniform(0.80, 1.00)
    crop_w = max(1, int(width * fraction))
    crop_h = max(1, int(height * fraction))
    left = rng.randint(0, width - crop_w)
    top = rng.randint(0, height - crop_h)
    img = img.crop((left, top, left + crop_w, top + crop_h))
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    angle = rng.uniform(-16, 16)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.90, 1.10))
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".incomplete")
    try:
        img.save(temporary, format="JPEG", quality=90, subsampling=0)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare(source_root: Path, destination_root: Path, target_per_class: int | None = None,
            seed: int = 42, size: int = 224, num_classes: int = 102,
            force: bool = False) -> dict:
    src, splits, offset = load_splits(source_root, num_classes=num_classes)
    dest = destination_root.expanduser().resolve()
    if dest == src or src in dest.parents:
        # A sibling data directory avoids accidentally parsing prepared images as input.
        raise ValueError("Destination must not be the source directory or inside it")
    counts = np.bincount([x.label for x in splits["train"]], minlength=num_classes)
    maximum = int(counts.max())
    target = maximum if target_per_class is None else target_per_class
    if target < 1:
        raise ValueError("--target-per-class must be a positive integer")
    if size < 32:
        raise ValueError("--size must be at least 32")
    initial_summary, fingerprints = dataset_summary(splits, offset, num_classes)
    expected = {
        "source_root": str(src), "source_split_fingerprints": fingerprints,
        "target_per_class": target, "seed": seed, "size": size,
        "classes": num_classes, "preprocessing": "offline_crop_flip_rotation_color_v1",
    }
    manifest_path = dest / "augmentation_manifest.json"
    if manifest_path.exists() and not force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(previous.get(key) == value for key, value in expected.items()):
            # Check the prepared metadata; do not just trust the manifest.
            _, loaded, _ = load_splits(dest, num_classes)
            output_counts = np.bincount([r.label for r in loaded["train"]],
                                        minlength=num_classes)
            if (output_counts == target).all():
                print(f"Already prepared and verified: {dest}")
                return previous
        raise FileExistsError("Existing prepared dataset doesn't match current settings. "
                              "Use a different --output or --force to rebuild deliberately.")
    if manifest_path.exists() and force:
        # Avoid a mixed old/new dataset when changing target counts or transforms.
        shutil.rmtree(dest)
    elif dest.exists() and (dest / "train.txt").exists():
        raise FileExistsError(f"{dest} has an incomplete or unrecognized dataset. "
                              "Remove it manually or use a new output path.")
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for record in splits["train"]:
        grouped[record.label].append(record)
    rows = {"train": [], "val": [], "test": []}
    generated = 0
    kept_original = 0
    removed_from_training = 0
    print(f"Balancing {num_classes} classes to EXACTLY {target:,} training images each.",
          flush=True)
    print(f"Prepared training size: {target * num_classes:,} images. "
          "Validation/test remain unmodified.", flush=True)
    print(f"Original largest class: {maximum:,}. "
          "Targets below this downsample large classes.", flush=True)
    for class_id in range(num_classes):
        images = sorted(grouped[class_id], key=lambda rec: rec.filename)
        class_rng = random.Random(seed + class_id * 1000003)
        class_rng.shuffle(images)
        originals = images[:target]
        removed_from_training += len(images) - len(originals)
        # Every original image retained when target >= the largest class.
        for record in originals:
            relative = record.filename
            _link_or_copy(src / "images" / record.filename, dest / "images" / relative)
            rows["train"].append(f"{relative} {class_id}\n")
            kept_original += 1
        need = target - len(originals)
        if need:
            # Cycle through different originals; seeds control transforms.
            for i in range(need):
                record = images[i % len(images)]
                relative = f"train/{class_id}/__aug_{seed}_{i:06d}.jpg"
                output_file = dest / "images" / relative
                if output_file.exists():
                    raise FileExistsError(f"Augmented image would overwrite {output_file}")
                _make_augmented(src / "images" / record.filename,
                                output_file, class_rng, size)
                rows["train"].append(f"{relative} {class_id}\n")
                generated += 1
        if (class_id + 1) % 10 == 0 or class_id == num_classes - 1:
            print(f"Prepared {class_id + 1}/{num_classes} classes "
                  f"({generated:,} augmented images)", flush=True)
    # No evaluation augmentation. Keep the original partition contents.
    for split in ("val", "test"):
        for record in splits[split]:
            _link_or_copy(src / "images" / record.filename,
                          dest / "images" / record.filename)
            rows[split].append(f"{record.filename} {record.label}\n")
    for split in ("train", "val", "test"):
        tmp = dest / (split + ".txt.partial")
        tmp.write_text("".join(sorted(rows[split])), encoding="utf-8")
        tmp.replace(dest / (split + ".txt"))
    prepared_root, prepared_splits, prepared_offset = load_splits(dest, num_classes)
    actual = np.bincount([r.label for r in prepared_splits["train"]],
                         minlength=num_classes)
    if not (actual == target).all():
        raise AssertionError(f"Class balancing failed: {actual}")
    summary, prepared_fingerprints = dataset_summary(prepared_splits, prepared_offset, num_classes)
    result = {
        **expected, "output_root": str(prepared_root),
        "original_train_count": int(counts.sum()),
        "largest_original_class": maximum,
        "kept_original_training_images": kept_original,
        "augmented_training_images": generated,
        "excluded_original_training_images": removed_from_training,
        "total_balanced_training_images": int(actual.sum()),
        "split_fingerprints": prepared_fingerprints,
        "prepared_split_summary": summary,
        "original_split_summary": initial_summary,
        "evaluation_images_unchanged": True,
        "augmentation_only_on_train": True,
    }
    tmp = manifest_path.with_suffix(".partial")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)
    print(f"Verified {int(actual.sum()):,} balanced training images. "
          f"Saved manifest: {manifest_path}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=PROJECT / "data" / "ip102_v1.1")
    p.add_argument("--output", type=Path, default=PROJECT / "data" / "ip102_balanced")
    p.add_argument("--target-per-class", type=int, default=None,
                   help="Default largest class size; lower means downsampling largest classes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--force", action="store_true",
                   help="Delete and rebuild existing prepared dataset with new parameters")
    args = p.parse_args()
    prepare(args.source, args.output, args.target_per_class, args.seed, args.size,
            force=args.force)


if __name__ == "__main__":
    main()
