"""Small OFFLINE Hugging Face training smoke tests, never download pretrained weights."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("transformers")

from transformers import (
    ConvNextConfig, ConvNextForImageClassification,
    SwinConfig, SwinForImageClassification,
    Dinov2Config, Dinov2ForImageClassification,
)
from experiments.ip102_hf_study import train


def tiny_model(name):
    if name == "convnext_small":
        cfg = ConvNextConfig(num_labels=3, num_stages=2,
                             hidden_sizes=[8, 16], depths=[1, 1])
        return ConvNextForImageClassification(cfg)
    if name == "swin_base":
        cfg = SwinConfig(num_labels=3, image_size=56, patch_size=4,
                         depths=[1, 1], embed_dim=16, num_heads=[2, 4],
                         window_size=4)
        return SwinForImageClassification(cfg)
    cfg = Dinov2Config(num_labels=3, image_size=56, patch_size=14,
                       hidden_size=32, intermediate_size=64,
                       num_hidden_layers=2, num_attention_heads=4)
    return Dinov2ForImageClassification(cfg)


@pytest.mark.parametrize("name", list(train.MODELS))
@pytest.mark.parametrize("head", ["linear", "mlp512", "mlp1024_256"])
def test_shapes_for_all_three_backbones_and_heads(name, head):
    model = tiny_model(name)
    train.replace_classifier(model, head, classes=3, dropout=0.2)
    model.eval()
    with torch.inference_mode():
        logits = model(pixel_values=torch.rand(1, 3, 56, 56)).logits
    assert logits.shape == (1, 3)


def test_dinov2_partial_unfreeze():
    model = tiny_model("dinov2_base")
    train.replace_classifier(model, "linear", classes=3, dropout=0.2)
    backbone = train.apply_trainability(model, "dinov2_base", False, 1)
    assert not any(p.requires_grad for p in backbone.encoder.layer[0].parameters())
    assert any(p.requires_grad for p in backbone.encoder.layer[-1].parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_augmentations_do_not_change_labels():
    x = torch.randn(4, 3, 56, 56)
    y = torch.tensor([0, 1, 2, 1])
    for mixup, cutmix in [(0, 0), (0.2, 0), (0, 0.2)]:
        mixed, a, b, lam = train.mix_batch(x.clone(), y, mixup, cutmix)
        assert mixed.shape == x.shape and a.shape == b.shape == y.shape
        assert 0 <= lam <= 1
    assert train.make_schedule(20, 2)(0) == .5


def test_reject_invalid_resolution():
    with pytest.raises(SystemExit):
        train.parse_args(["--model", "dinov2_base", "--image-size", "225"])
    with pytest.raises(SystemExit):
        train.parse_args(["--model", "swin_base", "--image-size", "336"])


def create_tiny_dataset(root: Path):
    for split in ("train", "val", "test"):
        lines = []
        for class_id in range(3):
            for i in range(2 if split == "train" else 1):
                rel = f"{split}/{class_id}/{i}.jpg"
                path = root / "images" / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (80, 80), color=(50+class_id*30, 100, 120)).save(path)
                lines.append(f"{rel} {class_id}\n")
        (root / (split + ".txt")).write_text("".join(lines), encoding="utf-8")


def test_one_epoch_pause_resume_and_no_test_leakage(tmp_path, monkeypatch):
    original = tmp_path / "original"
    create_tiny_dataset(original)
    output = tmp_path / "run"
    monkeypatch.setattr(
        train.AutoImageProcessor, "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            image_mean=[.485, .456, .406], image_std=[.229, .224, .225])
    )
    def mock_from_pretrained(*args, **kwargs):
        model = tiny_model("dinov2_base")
        return model, {"missing_keys": []}
    monkeypatch.setattr(train.AutoModelForImageClassification, "from_pretrained",
                        mock_from_pretrained)
    common = [
        "--model", "dinov2_base", "--classes", "3",
        "--original-root", str(original),
        "--image-size", "56", "--batch-size", "2",
        "--accum-steps", "2", "--workers", "0",
        "--epochs", "2", "--warmup-epochs", "0",
        "--freeze-backbone", "--amp", "off",
    ]
    train.main([*common, "--output-dir", str(output), "--stop-after-epoch", "1"])
    assert (output / "last_model.pth").is_file()
    assert not (output / "validation_metrics.json").exists()
    train.main([*common, "--resume", str(output / "last_model.pth")])
    scores = json.loads((output / "validation_metrics.json").read_text())
    assert scores["n_samples"] == 3
    assert (output / "validation_probabilities.npz").is_file()
    assert not (output / "test_evaluation").exists()
    with np.load(output / "validation_probabilities.npz", allow_pickle=False) as arr:
        assert arr["probabilities"].shape == (3, 3)
