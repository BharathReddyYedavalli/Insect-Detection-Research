"""Full-IP102 ResNet50 / ConvNeXt-Tiny classifier-head experiments.

Run from repository root:
  python experiments/ip102_head_study/train.py --model resnet50 --head linear --epochs 5
  python experiments/ip102_head_study/train.py --model convnext_tiny --head mlp512 --lr 0.00003 --epochs 10
Resume an interrupted run with its own last_model.pth and IDENTICAL settings.
Only the official validation split is used to select checkpoints. Test is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models, transforms as T
from tqdm.auto import tqdm

from shared.data import IP102Dataset, dataset_summary, load_splits
from shared.metrics import summarize, write_results

STUDY = Path(__file__).resolve().parent
CLASSES = 102
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
HEADS = ("linear", "mlp512", "mlp1024_256")
MODELS = ("resnet50", "convnext_tiny")


def build_head(features: int, classes: int, head: str, dropout: float):
    """Replace ONLY the classification head, fully fine-tune the backbone."""
    if head == "linear":
        return nn.Linear(features, classes)
    if head == "mlp512":
        return nn.Sequential(
            nn.Linear(features, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(512, classes)
        )
    if head == "mlp1024_256":
        return nn.Sequential(
            nn.Linear(features, 1024), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(1024, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, classes)
        )
    raise ValueError(f"Unknown head: {head}")


def make_model(name: str, head: str, classes: int, dropout: float, pretrained=True):
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        features = model.fc.in_features
        model.fc = build_head(features, classes, head, dropout)
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        # Keep torchvision's LayerNorm2d and Flatten; replace final linear layer.
        features = model.classifier[2].in_features
        model.classifier[2] = build_head(features, classes, head, dropout)
    else:
        raise ValueError(f"Unknown architecture: {name}")
    return model


def validate_dataset(original: Path, selected: Path, classes: int, treatment: str):
    source, source_splits, original_offset = load_splits(original, classes)
    if original_offset != 0:
        raise ValueError("Expected 0-based class labels 0..101 (or 0..N-1 in tests).")
    _, source_hashes = dataset_summary(source_splits, original_offset, classes)
    if treatment == "original":
        if selected.resolve() != source.resolve():
            raise ValueError("The original treatment must use the original dataset root.")
        return source, source_splits, source_hashes, source_hashes

    root, splits, offset = load_splits(selected, classes)
    if offset != 0:
        raise ValueError("Balanced labels must be 0-based.")
    _, hashes = dataset_summary(splits, offset, classes)
    manifest = json.loads((root / "augmentation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source_split_fingerprints") != source_hashes:
        raise ValueError("Balanced dataset was not prepared from this official original split.")
    if manifest.get("split_fingerprints") != hashes or not manifest.get("augmentation_only_on_train"):
        raise ValueError("Balanced dataset manifest or split hashes do not match.")
    if not manifest.get("evaluation_images_unchanged"):
        raise ValueError("Balanced dataset does not certify unchanged validation/test images.")
    for split in ("val", "test"):
        x = [(r.filename, r.label) for r in source_splits[split]]
        y = [(r.filename, r.label) for r in splits[split]]
        if x != y:
            raise ValueError(f"Official {split} split has been modified.")
    return root, splits, source_hashes, hashes


def transforms_for(size: int, train: bool):
    if train:
        steps = [
            T.RandomResizedCrop(size, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
        ]
    else:
        steps = [T.Resize(int(size * 256 / 224)), T.CenterCrop(size)]
    return T.Compose([*steps, T.ToTensor(), T.Normalize(MEAN, STD)])


def evaluate(model, loader, criterion, device, classes, amp):
    model.eval()
    targets, probs, filenames = [], [], []
    loss_sum, total = 0., 0
    with torch.inference_mode():
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=amp):
                logits = model(images)
            loss_sum += float(criterion(logits.float(), labels).item()) * len(labels)
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            targets.extend(labels.cpu().tolist())
            filenames.extend(loader.dataset.records[int(i)].filename for i in indices)
            total += len(labels)
    probabilities = np.concatenate(probs)
    scores, *_ = summarize(targets, probabilities.argmax(1), probabilities, classes)
    scores["loss"] = loss_sum / total
    return scores, targets, probabilities, filenames


def save_checkpoint(path: Path, state: dict):
    pending = path.with_suffix(".writing")
    torch.save(state, pending)
    pending.replace(path)


def draw_curves(history, folder: Path):
    plots = folder / "graphs"
    plots.mkdir(exist_ok=True)
    for fields, filename, ylabel in (
        (("train_accuracy", "val_accuracy"), "accuracy.png", "Accuracy"),
        (("val_macro_f1",), "macro_f1.png", "Macro-F1"),
        (("train_loss", "val_loss"), "loss.png", "Cross-entropy"),
    ):
        plt.figure(figsize=(8, 4.5))
        for field in fields:
            plt.plot([r["epoch"] for r in history], [r[field] for r in history], marker=".", label=field)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(plots / filename, dpi=160)
        plt.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--head", choices=HEADS, required=True)
    p.add_argument("--epochs", type=int, default=10, help="TOTAL epochs; fixed when resuming.")
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--dataset", choices=("original", "balanced"), default="original")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Optional dataset root for the selected treatment.")
    p.add_argument("--original-root", type=Path, default=ROOT / "data" / "ip102_v1.1")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--min-lr", type=float, default=0.000001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--classes", type=int, default=CLASSES)
    p.add_argument("--no-pretrained", action="store_true", help="For synthetic CPU smoke tests.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    args = p.parse_args(argv)
    if args.epochs < 1 or args.epochs > 20:
        p.error("--epochs must be between 1 and 20 for this study.")
    if args.batch_size < 1 or args.image_size < 32 or args.workers < 0 or args.classes < 2:
        p.error("Invalid batch/image/workers/classes.")
    if not (args.lr > 0 and args.min_lr >= 0 and args.min_lr <= args.lr):
        p.error("Require learning rate > 0 and 0 <= min-lr <= lr.")
    if not (0 <= args.dropout < 1 and 0 <= args.label_smoothing < 1):
        p.error("Dropout/label smoothing must be between 0 and 1.")
    if args.resume and args.output_dir:
        p.error("Do not specify --output-dir with --resume.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    original = args.original_root.expanduser().resolve()
    selected = args.data_root if args.data_root else (
        original if args.dataset == "original" else ROOT / "data" / "ip102_balanced"
    )
    train_root, splits, source_hash, selected_hash = validate_dataset(
        original, Path(selected).expanduser().resolve(), args.classes, args.dataset
    )
    # For balanced treatment, validation MUST load from original, never synthetic.
    original_root, original_splits, _ = load_splits(original, args.classes)
    train_set = IP102Dataset(train_root, splits["train"], transforms_for(args.image_size, True))
    val_set = IP102Dataset(original_root, original_splits["val"],
                           transforms_for(args.image_size, False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    model = make_model(args.model, args.head, args.classes, args.dropout, not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    scaler = GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    config = dict(
        study="full_IP102_ResNet50_ConvNeXt_head_study_v1",
        model=args.model, head=args.head, epochs=args.epochs, lr=args.lr,
        dataset=args.dataset, original_root=str(original),
        selected_root=str(train_root), original_split_hashes=source_hash,
        selected_split_hashes=selected_hash, batch_size=args.batch_size,
        image_size=args.image_size, weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing, dropout=args.dropout,
        min_lr=args.min_lr, seed=args.seed, workers=args.workers,
        classes=args.classes, pretrained=not args.no_pretrained,
        optimizer="AdamW", scheduler="CosineAnnealingLR",
        selection="best official validation accuracy",
        train_augmentation="RandomResizedCrop 0.75-1; flip; rotation 10; mild color jitter",
    )
    generator = torch.Generator().manual_seed(args.seed)
    history, best_acc, best_epoch, start_epoch = [], -1., 0, 1

    if args.resume:
        checkpoint = args.resume.expanduser().resolve()
        if checkpoint.name != "last_model.pth" or not checkpoint.is_file():
            p.error("Resume requires an existing last_model.pth from this study.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("config") != config:
            p.error("All experimental settings must match the checkpoint; resume only identical runs.")
        run = checkpoint.parent
        if int(state["epoch"]) >= args.epochs:
            p.error("Run has already reached its total epochs.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        history = state["history"]
        best_acc = float(state["best_acc"])
        best_epoch = int(state["best_epoch"])
        start_epoch = int(state["epoch"]) + 1
        if len(history) != start_epoch - 1:
            p.error("Checkpoint history is inconsistent.")
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if amp and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        generator.set_state(state["loader_rng"])
        if not (run / "best_model.pth").is_file():
            p.error("Previous best_model.pth is missing.")
        print(f"Resuming at epoch {start_epoch}/{args.epochs}", flush=True)
    else:
        if args.output_dir is None:
            name = (f"{args.model}_{args.head}_{args.dataset}_"
                    f"lr{args.lr:g}_e{args.epochs}_s{args.seed}_"
                    f"{datetime.now():%Y%m%d_%H%M%S}")
            run = STUDY / "runs" / name
        else:
            run = args.output_dir
        run = run.expanduser().resolve()
        run.mkdir(parents=True, exist_ok=False)
        (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=args.workers,
        pin_memory=amp, persistent_workers=args.workers > 0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=amp
    )
    print(f"{args.model} / {args.head} / {args.dataset} | "
          f"train={len(train_set):,} val={len(val_set):,} | "
          f"epochs={args.epochs} batch={args.batch_size} lr={args.lr:g} "
          f"device={device} | {run}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        began = time.perf_counter()
        model.train()
        total_loss, correct, seen = 0., 0, 0
        lr_this_epoch = optimizer.param_groups[0]["lr"]
        for images, labels, _ in tqdm(train_loader, desc=f"{args.model} epoch {epoch}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp):
                logits = model(images)
            loss = criterion(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum().item())
            seen += len(labels)
        val, _, _, _ = evaluate(model, val_loader, criterion, device, args.classes, amp)
        row = dict(epoch=epoch, lr=lr_this_epoch, train_accuracy=correct / seen,
                   train_loss=total_loss / seen, val_accuracy=val["accuracy"],
                   val_macro_f1=val["macro_f1"],
                   val_balanced_accuracy=val["balanced_accuracy"],
                   val_loss=val["loss"], val_top5_accuracy=val["top5_accuracy"],
                   seconds=time.perf_counter() - began)
        history.append(row)
        improved = val["accuracy"] > best_acc
        if improved:
            best_acc, best_epoch = val["accuracy"], epoch
        scheduler.step()
        state = dict(
            config=config, epoch=epoch, history=history, best_epoch=best_epoch,
            best_acc=best_acc, model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(), scaler_state=scaler.state_dict(),
            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if amp else None,
            loader_rng=generator.get_state()
        )
        save_checkpoint(run / "last_model.pth", state)
        if improved:
            save_checkpoint(run / "best_model.pth", state)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        draw_curves(history, run)
        print(f"Epoch {epoch}: accuracy={val['accuracy']*100:.2f}% "
              f"macro-F1={val['macro_f1']*100:.2f}% best_epoch={best_epoch}", flush=True)

    best = torch.load(run / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"])
    _, targets, probs, names = evaluate(model, val_loader, criterion, device, args.classes, amp)
    summary = write_results(run / "validation_evaluation", targets, names, probs, args.classes)
    (run / "validation_metrics.json").write_text(
        json.dumps({"best_epoch": best_epoch, "model": args.model,
                    "head": args.head, "condition": args.dataset, **summary}, indent=2),
        encoding="utf-8",
    )
    print(f"FINISHED | best epoch {best_epoch} | "
          f"val accuracy {summary['accuracy']*100:.2f}% | "
          f"macro-F1 {summary['macro_f1']*100:.2f}% | {run}", flush=True)


if __name__ == "__main__":
    main()
