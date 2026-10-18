"""Regression test for the readable single-file ResNet18 training entrypoint."""
import ast
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

from shared.prepare_balanced import prepare


ROOT = Path(__file__).resolve().parents[1]


def create_dataset(path):
    (path / "images").mkdir(parents=True)
    for split in ("train", "val", "test"):
        records = []
        for label in range(3):
            number = label + 1 if split == "train" else 1
            for index in range(number):
                relative = f"{split}/{label}/image_{index}.jpg"
                location = path / "images" / relative
                location.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (48, 48),
                          (30 + label * 60, 80 + index * 20, 120)).save(location)
                records.append(f"{relative} {label}\n")
        (path / f"{split}.txt").write_text("".join(records), encoding="utf-8")


def test_all_four_model_scripts_are_selfcontained():
    for name in ("ResNet18", "ResNet50", "VGG16", "VGG19"):
        script = ROOT / "models" / name / "train.py"
        text = script.read_text(encoding="utf-8")
        ast.parse(text, filename=str(script))
        assert "from shared.train import main" not in text
        for function in ("get_model", "validate", "save_graphs", "main"):
            assert f"def {function}(" in text
        assert "nn.CrossEntropyLoss()" in text
        assert "--extra-augmentation" in text


def test_resnet18_readable_script_trains_and_reports(tmp_path):
    original = tmp_path / "original"
    create_dataset(original)
    balanced = tmp_path / "balanced"
    prepare(original, balanced, target_per_class=3, seed=42, size=48, num_classes=3)
    run = tmp_path / "training_result"
    cmd = [
        sys.executable,
        str(ROOT / "models" / "ResNet18" / "train.py"),
        "--data-root", str(balanced),
        "--output-dir", str(run),
        "--classes", "3", "--image-size", "32",
        "--epochs", "1", "--batch-size", "2",
        "--workers", "0", "--no-pretrained",
    ]
    finished = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=240)
    assert finished.returncode == 0, finished.stdout[-4000:] + finished.stderr[-4000:]
    assert "Finished. Best epoch=1" in finished.stdout
    for filename in ("best_model.pth", "last_model.pth", "history.csv",
                     "validation_metrics.json", "config.yaml", "splits_sha256.json"):
        assert (run / filename).is_file()
    assert (run / "graphs" / "accuracy_by_epoch.png").is_file()
    assert (run / "validation_evaluation" / "graphs" / "confusion_counts.png").is_file()
    result = json.loads((run / "validation_metrics.json").read_text())
    assert result["n_samples"] == 3
    assert result["classes_present"] == 3
    assert result["best_epoch"] == 1
