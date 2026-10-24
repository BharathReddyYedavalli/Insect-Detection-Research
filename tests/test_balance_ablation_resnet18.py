"""Small CPU tests for the separate matched-update class-balancing experiment."""
import csv
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

from experiments.balance_ablation_resnet18 import MatchedStepBatchSampler, verify_datasets
from shared.prepare_balanced import prepare


ROOT = Path(__file__).resolve().parents[1]


def create_tiny_original(root):
    for split in ("train", "val", "test"):
        rows = []
        for class_id in range(3):
            count = class_id + 2 if split == "train" else 1
            for j in range(count):
                relative = f"{split}/{class_id}/{j}.jpg"
                path = root / "images" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (48, 48), (30 + class_id * 55, 60 + j * 20, 100)).save(path)
                rows.append(f"{relative} {class_id}\n")
        (root / f"{split}.txt").write_text("".join(rows), encoding="utf-8")


def test_sampler_has_equal_batch_size_and_stable_resume():
    full = list(MatchedStepBatchSampler(9, 2, 42, 0, 13))
    prefix = list(MatchedStepBatchSampler(9, 2, 42, 0, 6))
    suffix = list(MatchedStepBatchSampler(9, 2, 42, 6, 13))
    assert len(full) == 13
    assert all(len(batch) == 2 for batch in full)
    assert full == prefix + suffix


def test_two_conditions_and_resume(tmp_path):
    original = tmp_path / "original"
    create_tiny_original(original)
    balanced = tmp_path / "balanced"
    prepare(original, balanced, target_per_class=4, size=48, num_classes=3)
    verify_datasets(original, balanced, 3)

    experiment = ROOT / "experiments" / "balance_ablation_resnet18.py"
    common = [
        "--original-root", str(original), "--balanced-root", str(balanced),
        "--classes", "3", "--no-pretrained", "--image-size", "32",
        "--batch-size", "2", "--workers", "0", "--steps", "4",
        "--eval-every", "2", "--checkpoint-every", "1",
    ]

    def run(condition, folder, extras=()):
        cmd = [
            sys.executable, str(experiment), "--condition", condition,
            *common, *extras
        ]
        completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=240)
        assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
        return completed

    first = tmp_path / "first"
    run("original", first, ["--output-dir", str(first), "--run-until", "2"])
    first_state = torch.load(first / "last_model.pth", map_location="cpu", weights_only=False)
    assert first_state["step"] == 2
    assert first_state["history"][0]["step"] == 2
    assert not (first / "validation_metrics.json").exists()

    run("original", first, ["--resume", str(first / "last_model.pth")])
    second = tmp_path / "balanced_run"
    run("balanced", second, ["--output-dir", str(second)])

    for folder, condition in ((first, "original"), (second, "balanced")):
        state = torch.load(folder / "last_model.pth", map_location="cpu", weights_only=False)
        assert state["step"] == 4
        assert state["config"]["condition"] == condition
        with (folder / "history.csv").open(newline="") as handle:
            history = list(csv.DictReader(handle))
        assert [int(row["step"]) for row in history] == [2, 4]
        assert (folder / "best_model.pth").exists()
        assert (folder / "validation_metrics.json").exists()
        assert (folder / "validation_evaluation" / "per_class.csv").exists()
