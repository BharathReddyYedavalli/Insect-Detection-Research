"""Simple image classification training on a preprocessed balanced IP102 dataset.

This contains no weighted loss, weighted sampling, online augmentation,
class rebalancing, early stopping or scheduler. Change epochs using --epochs.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from shared.config import capture_environment, config_for, new_run, save_config, seed_everything
from shared.data import IP102Dataset, build_transform, dataset_summary, load_splits, make_loader
from shared.metrics import summarize, write_results
from shared.models import build_model
from shared.plots import learning_curves


def infer(model, loader, criterion, device, amp: bool):
    """Evaluate without augmentation, gradient updates or shuffled samples."""
    model.eval()
    all_y, all_p, all_indices = [], [], []
    total_loss, total = 0.0, 0
    with torch.inference_mode():
        for x, y, indices in tqdm(loader, desc="  validate", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            all_y.extend(y.cpu().tolist())
            all_p.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            all_indices.extend(indices.tolist())
            total_loss += float(loss.item()) * len(y)
            total += len(y)
    probs = np.concatenate(all_p)
    result, *_ = summarize(all_y, probs.argmax(axis=1), probs, probs.shape[1])
    result["loss"] = total_loss / total
    return result, all_y, probs, all_indices


def assert_prepared_balanced(root: Path, splits, fingerprints, n_classes: int):
    manifest = root / "augmentation_manifest.json"
    if not manifest.is_file():
        raise ValueError(
            "This training path requires a preprocessed dataset. First run "
            "'python -m shared.prepare_balanced' and point --data-root to its output."
        )
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    counts = np.bincount([record.label for record in splits["train"]], minlength=n_classes)
    if len(counts) != n_classes or counts.min() != counts.max():
        raise ValueError("Training dataset is not balanced to equal images per class.")
    if counts[0] != metadata.get("target_per_class"):
        raise ValueError("Prepared dataset count differs from augmentation manifest.")
    if metadata.get("split_fingerprints") != fingerprints:
        raise ValueError("Prepared split files differ from augmentation manifest.")
    if not metadata.get("augmentation_only_on_train"):
        raise ValueError("Manifest does not guarantee train-only augmentation.")
    return metadata


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_accuracy: float,
                    best_epoch: int, config: dict, fingerprints: dict):
    state = {
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "epoch": epoch, "best_accuracy": best_accuracy,
        "best_epoch": best_epoch, "config": config,
        "split_fingerprints": fingerprints,
    }
    partial = path.with_suffix(".writing")
    torch.save(state, partial)
    partial.replace(path)


def train(name: str, args):
    config = config_for(name)
    settings = config["training"]
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.lr is not None:
        settings["learning_rate"] = args.lr
    if args.seed is not None:
        settings["seed"] = args.seed
    if args.workers is not None:
        config["dataset"]["workers"] = args.workers
    if args.no_pretrained:
        settings["pretrained"] = False
    if settings["epochs"] <= 0 or settings["batch_size"] <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if settings["learning_rate"] <= 0:
        raise ValueError("learning rate must be positive")
    seed_everything(settings["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and settings.get("amp", True)

    root, splits, offset = load_splits(
        args.data_root or config["dataset"]["root"], config["dataset"]["num_classes"])
    config["dataset"]["root"] = str(root)
    summary, fingerprints = dataset_summary(
        splits, offset, config["dataset"]["num_classes"])
    metadata = assert_prepared_balanced(root, splits, fingerprints,
                                      config["dataset"]["num_classes"])
    run = new_run(name)
    save_config(config, run)
    capture_environment(run)
    (run / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run / "splits_sha256.json").write_text(json.dumps(fingerprints, indent=2), encoding="utf-8")
    (run / "augmentation_manifest.json").write_text(json.dumps(metadata, indent=2),
                                                      encoding="utf-8")

    size = config["dataset"]["image_size"]
    train_ds = IP102Dataset(root, splits["train"], build_transform(size, training=True))
    val_ds = IP102Dataset(root, splits["val"], build_transform(size, training=False))
    train_loader = make_loader(train_ds, settings["batch_size"], config["dataset"]["workers"],
                               True, seed=settings["seed"], device=device.type)
    val_loader = make_loader(val_ds, settings["batch_size"], config["dataset"]["workers"],
                             False, device=device.type)
    model = build_model(name, config["dataset"]["num_classes"],
                        pretrained=settings.get("pretrained", True)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"],
        weight_decay=settings.get("weight_decay", 0.0001))
    scaler = GradScaler("cuda", enabled=amp)
    history = []
    best_accuracy = -math.inf
    best_epoch = 0
    print(
        f"Model {name} | device={device} | "
        f"balanced train={len(train_ds):,} ({metadata['target_per_class']} per class) | "
        f"original val={len(val_ds):,} | output={run}", flush=True
    )
    if device.type != "cuda":
        print("WARNING: CUDA unavailable; training may be very slow.", flush=True)
    for epoch in range(1, settings["epochs"] + 1):
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()
        total_loss = 0.0
        correct = total = 0
        for x, y, _ in tqdm(train_loader, desc=f"{name} epoch {epoch}/{settings['epochs']}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            total += len(y)
        scores, _, _, _ = infer(model, val_loader, criterion, device, amp)
        row = {
            "epoch": epoch, "train_loss": total_loss / total,
            "train_accuracy": correct / total,
            "val_loss": scores["loss"], "val_accuracy": scores["accuracy"],
            "val_macro_f1": scores["macro_f1"],
            "val_balanced_accuracy": scores["balanced_accuracy"],
            "val_top5_accuracy": scores["top5_accuracy"],
            "val_ece": scores["ece_10_bins"],
            "learning_rate": settings["learning_rate"],
            "epoch_seconds": time.perf_counter() - start,
            "gpu_peak_memory_gb": (
                torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0
            ),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        improved = scores["accuracy"] > best_accuracy
        if improved:
            best_accuracy, best_epoch = scores["accuracy"], epoch
        save_checkpoint(run / "last_model.pth", model, optimizer, epoch,
                        best_accuracy, best_epoch, config, fingerprints)
        if improved:
            save_checkpoint(run / "best_model.pth", model, optimizer, epoch,
                            best_accuracy, best_epoch, config, fingerprints)
        learning_curves(run)
        print(
            f"Epoch {epoch}: train={100*row['train_accuracy']:.2f}% | "
            f"val={100*scores['accuracy']:.2f}% | "
            f"macro-F1={100*scores['macro_f1']:.2f}% | best epoch={best_epoch}",
            flush=True
        )

    # Selection uses validation accuracy; test remains untouched.
    checkpoint = torch.load(run / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    _, ys, probabilities, indices = infer(model, val_loader, criterion, device, amp)
    filenames = [val_ds.records[i].filename for i in indices]
    metrics = write_results(run / "validation_evaluation", ys, filenames, probabilities,
                            config["dataset"]["num_classes"])
    (run / "validation_metrics.json").write_text(
        json.dumps({"best_epoch": best_epoch, **metrics}, indent=2), encoding="utf-8")
    print(f"Completed. Best epoch={best_epoch}; "
          f"validation accuracy={100*metrics['accuracy']:.2f}%")
    print(f"Saved to {run}")


def main(name: str):
    p = argparse.ArgumentParser(description=f"Simple {name} training on prepared balanced IP102")
    p.add_argument("--data-root", help="Defaults to data/ip102_balanced")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()
    train(name, args)
