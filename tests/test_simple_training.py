"""CPU end-to-end training test: augmented balanced images -> checkpoint -> plots."""
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import torch
from torch import nn

import shared.train as module
from shared.prepare_balanced import prepare


def test_full_one_epoch_training(tmp_path, monkeypatch):
    source = tmp_path / "original"
    (source / "images").mkdir(parents=True)
    for split in ("train", "val", "test"):
        lines = []
        for label in range(3):
            for index in range(label + 1 if split == "train" else 1):
                filename = f"{split}/{label}/{index}.png"
                location = source / "images" / filename
                location.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (48, 48), (label * 90, 100, index * 30)).save(location)
                lines.append(f"{filename} {label}\n")
        (source / (split + ".txt")).write_text("".join(lines), encoding="utf-8")
    balanced = tmp_path / "balanced"
    prepare(source, balanced, target_per_class=3, num_classes=3, size=48)
    config = {
        "model": "resnet18",
        "dataset": {
            "root": str(balanced), "num_classes": 3, "image_size": 32, "workers": 0,
        },
        "training": {
            "epochs": 1, "batch_size": 3, "learning_rate": 0.001,
            "weight_decay": 0.0, "seed": 42, "pretrained": False, "amp": False,
        },
    }
    monkeypatch.setattr(module, "config_for", lambda name: config)
    monkeypatch.setattr(
        module, "build_model",
        lambda name, num_classes, pretrained: nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 32 * 32, num_classes))
    )
    def run_directory(_):
        folder = tmp_path / "run"
        folder.mkdir(exist_ok=True)
        return folder
    monkeypatch.setattr(module, "new_run", run_directory)
    args = SimpleNamespace(
        data_root=str(balanced), epochs=1, batch_size=3, lr=None,
        seed=42, workers=0, no_pretrained=True,
    )
    module.train("resnet18", args)
    run = tmp_path / "run"
    assert (run / "last_model.pth").is_file()
    assert (run / "best_model.pth").is_file()
    assert (run / "history.csv").is_file()
    assert (run / "validation_evaluation" / "metrics.json").is_file()
    assert (run / "validation_evaluation" / "graphs" / "confusion_counts.png").is_file()
    state = torch.load(run / "best_model.pth", map_location="cpu", weights_only=False)
    assert state["epoch"] == state["best_epoch"] == 1
