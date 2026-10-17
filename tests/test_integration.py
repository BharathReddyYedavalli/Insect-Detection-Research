"""Integration smoke test for validation, per-class metrics, and artifact generation."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn

from shared.data import IP102Dataset, build_transform, load_splits, make_loader
from shared.train import infer
from shared.metrics import write_results


class TinyClassifier(nn.Module):
    def forward(self, x):
        avg = x.mean(dim=(2, 3))
        return avg[:, :3]


def test_infer_and_report(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for split in ("train", "val", "test"):
        records = []
        for cls in range(3):
            name = f"{split}_{cls}.png"
            rgb = [10, 10, 10]
            rgb[cls] = 240
            Image.new("RGB", (32, 32), tuple(rgb)).save(images / name)
            records.append(f"{name} {cls}")
        (tmp_path / f"{split}.txt").write_text("\n".join(records))
    root, splits, _ = load_splits(tmp_path, num_classes=3)
    ds = IP102Dataset(root, splits["val"], build_transform(32, False, {}))
    loader = make_loader(ds, 2, 0, False)
    model = TinyClassifier()
    scores, truth, probs, indices = infer(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), False)
    assert probs.shape == (3, 3)
    assert scores["accuracy"] == 1.0
    assert len(indices) == 3
    names = [ds.records[i].filename for i in indices]
    out = tmp_path / "reports"
    result = write_results(out, truth, names, probs, n_classes=3)
    assert result["macro_f1"] == 1.0
    assert (out / "metrics.json").is_file()
    assert (out / "graphs" / "confusion_row_normalized.png").is_file()
    assert (out / "graphs" / "reliability.png").is_file()
    assert (out / "graphs" / "prediction_probability_correlation.png").is_file()
