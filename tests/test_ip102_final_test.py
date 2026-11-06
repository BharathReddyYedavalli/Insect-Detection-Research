"""Offline tests for frozen official-IP102 test evaluation safeguards.

Never download a checkpoint or touch real test data in GitHub CI.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

pytest.importorskip("transformers")

from shared.data import IP102Dataset, Record, dataset_summary, load_splits
from experiments.ip102_hf_study import final_evaluate as final_test


def write_tiny_official_structure(root: Path):
    # 102 labels present in train, but no actual test images are created:
    # preflight must not open test pixels.
    (root / "train.txt").write_text(
        "".join(f"train/image_{i}.jpg {i}\n" for i in range(102)), encoding="utf-8"
    )
    (root / "val.txt").write_text(
        "".join(f"val/image_{i}.jpg {i}\n" for i in range(3)), encoding="utf-8"
    )
    (root / "test.txt").write_text(
        "".join(f"test/image_{i}.jpg {i}\n" for i in range(4)), encoding="utf-8"
    )


def make_fake_runs(original: Path, runs_root: Path):
    _, splits, offset = load_splits(original, num_classes=102, check_files=False)
    _, fingerprints = dataset_summary(splits, offset, 102)
    for name, model_name, seed, last_n in final_test.LOCKED_RUNS:
        target = runs_root / name
        target.mkdir(parents=True)
        cfg = dict(
            model=model_name, seed=seed, unfreeze_last_n=last_n,
            head="linear", epochs=20, image_size=224, classes=102,
            dataset="original", augmentation="basic", sampler="original",
            split_fingerprints=fingerprints
        )
        (target / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        (target / "validation_metrics.json").write_text(
            json.dumps({
                "n_samples": 3, "split": "validation", "accuracy": 0.76,
                "macro_f1": 0.70, "best_epoch": 16,
            }), encoding="utf-8"
        )
        (target / "best_model.pth").write_bytes(b"fake; preflight must not load this")
    return fingerprints


def test_preflight_only_checks_metadata_and_never_opens_test_images(tmp_path, monkeypatch, capsys):
    original = tmp_path / "ip102_v1.1"
    original.mkdir()
    write_tiny_official_structure(original)
    runs = tmp_path / "runs"
    make_fake_runs(original, runs)
    monkeypatch.setattr(final_test, "OFFICIAL_SIZES",
                        {"train": 102, "val": 3, "test": 4})
    def no_images(*args, **kwargs):
        raise AssertionError("Preflight must not open image pixel data")
    monkeypatch.setattr(Image, "open", no_images)
    final_test.main(["--original-root", str(original), "--runs-root", str(runs),
                     "--output-root", str(tmp_path / "results")])
    assert "READY: 4 FROZEN checkpoints" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


def test_preflight_refuses_modified_split_and_model_config(tmp_path, monkeypatch):
    original = tmp_path / "ip102_v1.1"
    original.mkdir()
    write_tiny_official_structure(original)
    runs = tmp_path / "runs"
    make_fake_runs(original, runs)
    monkeypatch.setattr(final_test, "OFFICIAL_SIZES",
                        {"train": 102, "val": 3, "test": 4})
    file = original / "test.txt"
    file.write_text(file.read_text().replace("test/image_0.jpg 0", "test/image_changed.jpg 0"))
    with pytest.raises(ValueError, match="fingerprints"):
        final_test.preflight(original, runs)


def test_inference_uses_deterministic_inputs_and_probabilities(tmp_path):
    image_root = tmp_path / "ip102_v1.1" / "images"
    entries = []
    for i in range(3):
        path = image_root / "test" / f"{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (40 * i, 100, 120)).save(path)
        entries.append(Record(f"test/{i}.jpg", i))
    dataset = IP102Dataset(
        image_root.parent, entries,
        final_test.make_transforms(56, False, "basic",
                                   [0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    class Model(nn.Module):
        def forward(self, pixel_values):
            x = pixel_values.mean(dim=(1, 2, 3))
            return SimpleNamespace(logits=torch.stack([x, x + 1, x - 1], dim=1))
    names, labels, probs = final_test.test_predict(
        Model(), loader, torch.device("cpu"), "off", "dinov2_base", 56
    )
    assert names.tolist() == [f"test/{i}.jpg" for i in range(3)]
    assert labels.tolist() == [0, 1, 2]
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
