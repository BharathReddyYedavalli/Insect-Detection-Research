"""Verify offline image augmentation yields exact class balance and no data leakage."""
from collections import Counter
from pathlib import Path

from PIL import Image
import pytest

from shared.prepare_balanced import prepare
from shared.data import load_splits, dataset_summary
from shared.train import assert_prepared_balanced


def fake_source(root: Path, n_classes: int = 3):
    (root / "images").mkdir(parents=True)
    counts = {}
    for split in ("train", "val", "test"):
        lines = []
        for label in range(n_classes):
            count = (label + 1) if split == "train" else 1
            counts[split, label] = count
            for index in range(count):
                relative = f"{split}/{label}/image_{index}.png"
                photo = root / "images" / relative
                photo.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (48, 48),
                          (20 + label * 70, 40 + index * 20, 115)).save(photo)
                lines.append(f"{relative} {label}\n")
        (root / (split + ".txt")).write_text("".join(lines), encoding="utf-8")
    return counts


def test_balancing_by_augmentations_and_untouched_eval(tmp_path):
    source = tmp_path / "original"
    counts = fake_source(source)
    output = tmp_path / "balanced"
    result = prepare(source, output, target_per_class=4,
                     seed=42, size=48, num_classes=3)
    assert result["kept_original_training_images"] == 6
    assert result["augmented_training_images"] == 6
    assert result["excluded_original_training_images"] == 0
    assert result["total_balanced_training_images"] == 12
    root, splits, offset = load_splits(output, num_classes=3)
    counts = Counter(item.label for item in splits["train"])
    assert counts == {0: 4, 1: 4, 2: 4}
    assert len(splits["val"]) == len(splits["test"]) == 3
    for part in ("val", "test"):
        for record in splits[part]:
            assert (output / "images" / record.filename).read_bytes() == (
                source / "images" / record.filename).read_bytes()
    _, fingerprints = dataset_summary(splits, offset, num_classes=3)
    metadata = assert_prepared_balanced(root, splits, fingerprints, n_classes=3)
    assert metadata["target_per_class"] == 4
    again = prepare(source, output, 4, 42, 48, 3)
    assert again["split_fingerprints"] == result["split_fingerprints"]


def test_optional_downsample_majority(tmp_path):
    source = tmp_path / "original"
    fake_source(source)
    output = tmp_path / "balanced_small"
    result = prepare(source, output, target_per_class=2,
                     seed=42, size=48, num_classes=3)
    assert result["kept_original_training_images"] == 5
    assert result["excluded_original_training_images"] == 1
    assert result["augmented_training_images"] == 1
    _, splits, _ = load_splits(output, 3)
    assert Counter(r.label for r in splits["train"]) == {0: 2, 1: 2, 2: 2}


def test_invalid_preparation_is_rejected(tmp_path):
    source = tmp_path / "original"
    fake_source(source)
    with pytest.raises(ValueError):
        prepare(source, source / "wrong", target_per_class=4, num_classes=3)
    with pytest.raises(ValueError):
        prepare(source, tmp_path / "other", target_per_class=0, num_classes=3)
