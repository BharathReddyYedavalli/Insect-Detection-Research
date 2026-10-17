"""Safely convert Kaggle IP102's classification/<split>/<class>/*.jpg ZIP layout.

Preserves the archive's existing train/val/test membership. This tool DOES NOT
randomly split images, resample evaluation data, or move examples between splits.

Usage:
  python -m shared.import_kaggle --zip "$HOME/Downloads/archive.zip" --inspect-only
  python -m shared.import_kaggle --zip "$HOME/Downloads/archive.zip"
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

from shared.config import PROJECT

# Expected counts from the official IP102 classification benchmark.
OFFICIAL_SPLIT_SIZES = {"train": 45095, "val": 7508, "test": 22619}
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test",
}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
ZIP_MEMBER = re.compile(
    r"(?:^|/)classification/(train|training|val|valid|validation|test)/(\d+)/([^/]+)$",
    re.IGNORECASE,
)


def inspect_archive(archive: Path, num_classes: int = 102):
    """Return verified zip items and counts without extracting a single image."""
    found = []
    class_ids = set()
    names = set()
    member_digest = sha256()
    class_info = None
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            filename = item.filename.replace("\\", "/")
            if filename.lower().endswith("/classes.txt") or filename.lower() == "classes.txt":
                if class_info is None:
                    class_info = filename
            match = ZIP_MEMBER.search(filename)
            if match is None or item.is_dir():
                continue
            split_name, class_str, basename = match.groups()
            if Path(basename).suffix.lower() not in IMAGE_TYPES:
                continue
            if basename in (".", "..") or "\\" in basename or ":" in basename:
                raise ValueError(f"Unsafe filename within ZIP: {filename!r}")
            split = SPLIT_ALIASES[split_name.lower()]
            label = int(class_str)
            class_ids.add(label)
            relative = f"{split}/{label}/{basename}"
            if relative.casefold() in names:
                raise ValueError(f"Conflicting ZIP image path: {filename!r}")
            names.add(relative.casefold())
            found.append((item, split, label, basename))
            member_digest.update(
                f"{relative}\t{item.file_size}\t{item.CRC}\n".encode("utf-8")
            )
    if not found:
        raise ValueError(
            "No classification/<split>/<class>/<image> files found in this ZIP. "
            "Confirm the archive is the 102-class IP102 classification dataset."
        )
    zero_based = set(range(num_classes))
    one_based = set(range(1, num_classes + 1))
    if class_ids == zero_based:
        offset = 0
    elif class_ids == one_based:
        offset = 1
    else:
        missing = sorted(zero_based - class_ids)
        raise ValueError(
            f"Expected all {num_classes} class folders numbered 0..{num_classes - 1} "
            f"or 1..{num_classes}; found {len(class_ids)} unique labels "
            f"({min(class_ids)}..{max(class_ids)}), missing zero-based examples {missing[:10]}."
        )
    by_split = Counter(entry[1] for entry in found)
    if any(by_split[s] == 0 for s in ("train", "val", "test")):
        raise ValueError(
            f"Archive is missing at least one existing train/val/test partition: {dict(by_split)}. "
            "Do not generate random splits to cover for missing official partitions."
        )
    train_classes = {label - offset for _, split, label, _ in found if split == "train"}
    if train_classes != zero_based:
        raise ValueError(
            f"Training split only contains {len(train_classes)} of {num_classes} classes."
        )
    summary = {
        "images_total": len(found),
        "zip_entries_used": len(found),
        "split_counts": {s: by_split[s] for s in ("train", "val", "test")},
        "classes": num_classes,
        "original_label_offset": offset,
        "official_split_counts_match": (
            num_classes == 102 and dict(by_split) == OFFICIAL_SPLIT_SIZES
        ),
        "uncompressed_image_bytes": sum(item.file_size for item, _, _, _ in found),
        "archive_manifest_sha256": member_digest.hexdigest(),
        "classes_file_in_archive": class_info,
    }
    return found, summary


def import_archive(archive: Path, root: Path, num_classes: int = 102,
                   allow_nonstandard: bool = False):
    archive = Path(archive).expanduser().resolve()
    root = Path(root).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"ZIP not found: {archive}")
    entries, summary = inspect_archive(archive, num_classes)
    print(json.dumps(summary, indent=2), flush=True)
    if num_classes == 102 and not summary["official_split_counts_match"] and not allow_nonstandard:
        raise ValueError(
            "Archive split counts differ from the published IP102 benchmark. "
            "STOP and verify your Kaggle source/version before training. "
            "Use --allow-nonstandard-splits only if you knowingly want a non-comparable study."
        )

    disk_parent = root
    while not disk_parent.exists():
        disk_parent = disk_parent.parent
    free_bytes = shutil.disk_usage(disk_parent).free
    needed = summary["uncompressed_image_bytes"]
    if free_bytes < needed + 512 * 1024 * 1024:
        raise OSError(
            f"Insufficient free disk space for extraction. Need about "
            f"{(needed + 512 * 1024 * 1024)/2**30:.2f} GiB free, "
            f"available {free_bytes/2**30:.2f} GiB on {disk_parent}"
        )
    root.mkdir(parents=True, exist_ok=True)
    records = {"train": [], "val": [], "test": []}
    extracted = skipped = 0
    with zipfile.ZipFile(archive) as zf:
        for i, (item, split, label, basename) in enumerate(entries, 1):
            label -= summary["original_label_offset"]
            relative = f"{split}/{label}/{basename}"
            dest = root / "images" / split / str(label) / basename
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == item.file_size:
                skipped += 1
            else:
                part = dest.with_name(dest.name + ".partial")
                try:
                    with zf.open(item) as src, part.open("wb") as target:
                        shutil.copyfileobj(src, target, length=1024 * 1024)
                    if part.stat().st_size != item.file_size:
                        raise OSError(f"Extracted size mismatch for {item.filename!r}")
                    part.replace(dest)
                finally:
                    if part.exists():
                        part.unlink()
                extracted += 1
            records[split].append(f"{relative} {label}\n")
            if i % 5000 == 0:
                print(f"Processed {i:,}/{len(entries):,} images...", flush=True)

        if summary["classes_file_in_archive"]:
            class_dest = root / "classes.txt"
            class_dest.write_bytes(zf.read(summary["classes_file_in_archive"]))
    # Don't write split indices until all images were successfully processed.
    for split in ("train", "val", "test"):
        path = root / f"{split}.txt"
        partial = path.with_suffix(".tmp")
        partial.write_text("".join(sorted(records[split])), encoding="utf-8")
        partial.replace(path)
    summary.update({
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive": str(archive), "output_root": str(root),
        "newly_extracted_images": extracted, "existing_size_matched_images": skipped,
        "split_manifest_format": "relative_image_path zero_based_integer_label",
    })
    (root / "import_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from shared.data import load_splits
    load_splits(root, num_classes=num_classes, check_files=True)
    print(f"Import and dataset integrity checks passed. Data directory: {root}", flush=True)
    print("Next: python -m shared.audit --data-root data/ip102_v1.1", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path, help="Kaggle archive.zip path")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data" / "ip102_v1.1")
    parser.add_argument("--num-classes", type=int, default=102)
    parser.add_argument("--inspect-only", action="store_true",
                        help="Show split counts, classes and image sizes without extracting")
    parser.add_argument("--allow-nonstandard-splits", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        _, summary = inspect_archive(args.zip, args.num_classes)
        print(json.dumps(summary, indent=2))
        if args.num_classes == 102 and not summary["official_split_counts_match"]:
            print("WARNING: Counts differ from official IP102 benchmark; verify dataset source.")
        return
    import_archive(args.zip, args.data_root, args.num_classes,
                   allow_nonstandard=args.allow_nonstandard_splits)


if __name__ == "__main__":
    main()
