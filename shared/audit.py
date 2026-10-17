"""Check IP102 image paths, label ranges, cross-split overlap and distribution."""
from __future__ import annotations
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.data import dataset_summary, load_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/ip102_v1.1")
    parser.add_argument("--num-classes", type=int, default=102)
    parser.add_argument("--content-hashes", action="store_true",
                        help="Slow but detects byte-identical images named differently across splits")
    args = parser.parse_args()
    root, splits, offset = load_splits(args.data_root, args.num_classes, check_files=True)
    summary, digests = dataset_summary(splits, offset, args.num_classes)
    output = root / "audit"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "split_fingerprints.json").write_text(json.dumps(digests, indent=2), encoding="utf-8")
    frame = pd.DataFrame({split: [summary["splits"][split]["per_class"][str(i)]
                                 for i in range(args.num_classes)] for split in splits})
    frame.index.name = "class_index"
    frame.to_csv(output / "class_counts.csv")
    frame.plot.bar(figsize=(19, 6), width=.8)
    plt.title("Official IP102 class counts by split")
    plt.ylabel("Number of images")
    plt.xticks(ticks=range(0, args.num_classes, 5), labels=range(0, args.num_classes, 5), rotation=0)
    plt.tight_layout()
    plt.savefig(output / "class_distribution.png", dpi=200)
    plt.close()

    if args.content_hashes:
        hashes = defaultdict(list)
        for split, records in splits.items():
            for rec in records:
                digest = sha256((root / "images" / rec.filename).read_bytes()).hexdigest()
                hashes[digest].append({"split": split, "filename": rec.filename})
        clashes = [group for group in hashes.values()
                   if len({item["split"] for item in group}) > 1]
        (output / "byte_identical_cross_split.json").write_text(
            json.dumps(clashes[:100], indent=2), encoding="utf-8")
        if clashes:
            raise ValueError(f"Detected {len(clashes)} groups of byte-identical images across splits")
    print(json.dumps(summary, indent=2))
    print(f"Dataset audit passed; outputs: {output}")


if __name__ == "__main__":
    main()
