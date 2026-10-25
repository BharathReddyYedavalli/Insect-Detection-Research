"""CPU smoke tests for new standalone IP102 head study."""
import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

from experiments.ip102_head_study.train import make_model, validate_dataset
from shared.prepare_balanced import prepare


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "experiments/ip102_head_study/train.py"


def create_data(root: Path):
    for split in ("train", "val", "test"):
        lines = []
        for label in range(3):
            n = label + 2 if split == "train" else 1
            for idx in range(n):
                rel = f"{split}/{label}/picture_{idx}.jpg"
                dst = root / "images" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 64), (60 + label * 40, 70 + idx * 10, 120)).save(dst)
                lines.append(f"{rel} {label}\n")
        (root / f"{split}.txt").write_text("".join(lines), encoding="utf-8")


def test_heads_and_architectures_produce_all_class_logits():
    x = torch.randn(2, 3, 32, 32)
    for name in ("resnet50", "convnext_tiny"):
        for head in ("linear", "mlp512", "mlp1024_256"):
            model = make_model(name, head, classes=3, dropout=0.2, pretrained=False)
            model.eval()
            with torch.no_grad():
                y = model(x)
            assert y.shape == (2, 3)
            del model


def test_split_validation_and_epoch_resume(tmp_path):
    original = tmp_path / "original"
    create_data(original)
    balanced = tmp_path / "balanced"
    prepare(original, balanced, target_per_class=4, size=64, num_classes=3)
    root, splits, _, _ = validate_dataset(original, balanced, 3, "balanced")
    assert root == balanced.resolve()
    assert len(splits["train"]) == 12

    output = tmp_path / "experiment"
    common = [
        sys.executable, str(TRAIN), "--model", "resnet50",
        "--head", "mlp512", "--no-pretrained",
        "--original-root", str(original), "--dataset", "original",
        "--classes", "3", "--batch-size", "3", "--image-size", "32",
        "--workers", "0", "--epochs", "2", "--lr", "0.0001",
    ]
    paused = subprocess.run(
        [*common, "--output-dir", str(output), "--stop-after-epoch", "1"],
        cwd=ROOT, text=True, capture_output=True, timeout=240
    )
    assert paused.returncode == 0, paused.stdout[-3000:] + paused.stderr[-3000:]
    assert "PAUSED at epoch 1/2" in paused.stdout
    assert not (output / "validation_metrics.json").exists()
    state = torch.load(output / "last_model.pth", weights_only=False, map_location="cpu")
    assert state["epoch"] == 1

    complete = subprocess.run(
        [*common, "--resume", str(output / "last_model.pth")],
        cwd=ROOT, text=True, capture_output=True, timeout=240
    )
    assert complete.returncode == 0, complete.stdout[-3000:] + complete.stderr[-3000:]
    state = torch.load(output / "last_model.pth", weights_only=False, map_location="cpu")
    assert state["epoch"] == 2
    assert len(state["history"]) == 2
    metrics = json.loads((output / "validation_metrics.json").read_text())
    assert metrics["n_samples"] == 3
    assert (output / "validation_evaluation/per_class.csv").exists()
    assert not (output / "test_evaluation").exists()
