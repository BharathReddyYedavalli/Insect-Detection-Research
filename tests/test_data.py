"""Synthetic-data integrity and simple loader tests."""
from pathlib import Path

import pytest
from PIL import Image
from shared.data import IP102Dataset, build_transform, dataset_summary, load_splits, make_loader


def make_fake(root: Path, one_based: bool = False):
    images = root / "images"
    images.mkdir()
    for split in ("train", "val", "test"):
        lines = []
        for cls in range(3):
            count = cls + 1 if split == "train" else 1
            for j in range(count):
                name = f"{split}_{cls}_{j}.jpg"
                Image.new("RGB", (32, 32), (cls * 30, j * 10, 120)).save(images / name)
                lines.append(f"{name} {cls + (1 if one_based else 0)}")
        (root / f"{split}.txt").write_text("\n".join(lines) + "\n")
    return root


@pytest.mark.parametrize("one_based", [False, True])
def test_load_and_simple_batches(tmp_path, one_based):
    root, splits, offset = load_splits(make_fake(tmp_path, one_based), num_classes=3)
    assert offset == int(one_based)
    assert [len(splits[x]) for x in ("train", "val", "test")] == [6, 3, 3]
    summary, hashes = dataset_summary(splits, offset, 3)
    assert len(set(hashes.values())) == 3
    assert summary["splits"]["train"]["per_class"] == {"0": 1, "1": 2, "2": 3}
    ds = IP102Dataset(root, splits["train"], build_transform(32))
    assert len(ds) == 6
    loader = make_loader(ds, 3, 0, training=False)
    assert sum(x.shape[0] for x, _, _ in loader) == 6


def test_cross_split_leakage_rejected(tmp_path):
    root = make_fake(tmp_path)
    with (root / "val.txt").open("a") as f:
        f.write("train_0_0.jpg 0\n")
    with pytest.raises(ValueError, match="appears in both"):
        load_splits(root, num_classes=3)
