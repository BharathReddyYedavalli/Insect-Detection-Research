"""Experiment runner: CUDA AMP, checkpoints, resume, logs and validation-only selection."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from shared.config import PROJECT, DISPLAY, capture_environment, config_for, new_run, save_config, seed_everything
from shared.data import (IP102Dataset, build_transform, class_weights, dataset_summary,
                         load_splits, make_loader)
from shared.metrics import summarize, write_results
from shared.models import build_model
from shared.plots import learning_curves


def infer(model, loader, criterion, device, amp: bool):
    model.eval()
    all_y, all_p, all_indices, losses = [], [], [], []
    with torch.inference_mode():
        for x, y, indices in tqdm(loader, desc="  validate", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
            probs = torch.softmax(logits.float(), dim=1)
            losses.append((float(loss.item()), len(y)))
            all_y.extend(y.cpu().tolist())
            all_p.append(probs.cpu().numpy())
            all_indices.extend(indices.tolist())
    probabilities = np.concatenate(all_p)
    predictions = probabilities.argmax(axis=1)
    scores, *_ = summarize(all_y, predictions, probabilities, probabilities.shape[1])
    scores["loss"] = sum(v * n for v, n in losses) / sum(n for _, n in losses)
    return scores, all_y, probabilities, all_indices


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch,
                    best_score, best_epoch, bad_epochs, config, fingerprints):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(), "epoch": epoch,
        "best_score": best_score, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
        "config": config, "split_fingerprints": fingerprints,
        "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    # Atomic write avoids corrupting a successful checkpoint if the process is interrupted.
    partial = path.with_suffix(".writing")
    torch.save(state, partial)
    partial.replace(path)


def restore_checkpoint(path, model, optimizer, scheduler, scaler, fingerprints, device):
    # Only load self-generated/trusted checkpoints: pickle-backed checkpoints are executable.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["split_fingerprints"] != fingerprints:
        raise ValueError("Dataset split fingerprint differs from original run. Refusing resume.")
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    scaler.load_state_dict(state["scaler_state"])
    torch.set_rng_state(state["rng_torch"])
    np.random.set_state(state["rng_numpy"])
    random.setstate(state["rng_python"])
    if device.type == "cuda" and state.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state


def train(name: str, args):
    config = config_for(name)
    if args.resume:
        source = torch.load(args.resume, map_location="cpu", weights_only=False)
        if source["config"]["model"] != name:
            raise ValueError("Resumed checkpoint architecture does not match this launcher")
        config = source["config"]
    else:
        settings = config["training"]
        for arg, key in [
            ("epochs", "epochs"), ("batch_size", "batch_size"),
            ("lr", "learning_rate"), ("balance", "balance"), ("seed", "seed"),
            ("workers", None), ("image_size", None), ("equal_k", "equal_samples_per_class"),
        ]:
            value = getattr(args, arg)
            if value is None:
                continue
            if arg == "workers":
                config["dataset"]["workers"] = value
            elif arg == "image_size":
                config["dataset"]["image_size"] = value
            else:
                settings[key] = value
        if args.no_pretrained:
            settings["pretrained"] = False
        if args.no_amp:
            settings["amp"] = False
        if args.deterministic:
            settings["deterministic"] = True
    settings = config["training"]
    if args.resume and args.epochs is not None:
        settings["epochs"] = args.epochs
    if settings["epochs"] < 1 or settings["batch_size"] < 1:
        raise ValueError("epochs and batch_size must both be positive")
    if settings["balance"] not in ("none", "weighted_sampler", "equal_epoch", "weighted_loss"):
        raise ValueError("Invalid balancing mode")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and settings["amp"]
    seed_everything(settings["seed"], settings["deterministic"])
    data_root = args.data_root or config["dataset"]["root"]
    root, splits, offset = load_splits(data_root, config["dataset"]["num_classes"])
    config["dataset"]["root"] = str(root)
    summary, fingerprints = dataset_summary(splits, offset, config["dataset"]["num_classes"])
    if args.resume:
        run = Path(args.resume).resolve().parent
        if source["split_fingerprints"] != fingerprints:
            raise ValueError("Dataset split fingerprint changed, cannot resume")
    else:
        run = new_run(name)
        save_config(config, run)
        capture_environment(run)
        (run / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (run / "splits_sha256.json").write_text(json.dumps(fingerprints, indent=2), encoding="utf-8")

    img_size = config["dataset"]["image_size"]
    training = IP102Dataset(root, splits["train"], build_transform(img_size, True, config["augmentation"]))
    validation = IP102Dataset(root, splits["val"], build_transform(img_size, False, config["augmentation"]))
    workers = config["dataset"]["workers"]
    train_loader = make_loader(training, settings["batch_size"], workers, True,
                               settings["balance"], settings["seed"], settings["equal_samples_per_class"],
                               device.type)
    val_loader = make_loader(validation, settings["batch_size"], workers, False, device=device.type)
    print(f"Model {name} | device={device} | train={len(training)} | val={len(validation)} | "
          f"balance={settings['balance']} | output={run}")
    if device.type != "cuda":
        print("WARNING: CUDA unavailable. Training this dataset on CPU will be very slow.")

    # Resume restores weights so avoid downloading pretrained weights unnecessarily.
    model = build_model(name, config["dataset"]["num_classes"],
                        pretrained=settings["pretrained"] and not bool(args.resume)).to(device)
    loss_weights = (class_weights(training.targets, config["dataset"]["num_classes"], device)
                    if settings["balance"] == "weighted_loss" else None)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=settings["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    if settings["scheduler"] != "cosine":
        raise ValueError("Only scheduler='cosine' is currently supported")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["epochs"], eta_min=settings["min_lr"])
    scaler = GradScaler("cuda", enabled=amp)
    best_score, best_epoch, bad_epochs, start = -math.inf, 0, 0, 1
    history_path = run / "history.csv"
    history = pd.read_csv(history_path).to_dict("records") if args.resume and history_path.exists() else []
    if args.resume:
        state = restore_checkpoint(args.resume, model, optimizer, scheduler, scaler, fingerprints, device)
        best_score, best_epoch, bad_epochs = state["best_score"], state["best_epoch"], state["bad_epochs"]
        start = state["epoch"] + 1
        if settings["epochs"] > state["config"]["training"]["epochs"]:
            # A longer resumed run has a newly defined cosine horizon; record as protocol change.
            scheduler.T_max = settings["epochs"]

    for epoch in range(start, settings["epochs"] + 1):
        begin = time.perf_counter()
        sampler = train_loader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        total_loss = total_correct = n_images = 0
        lr = optimizer.param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for x, y, _ in tqdm(train_loader, desc=f"{name} epoch {epoch}/{settings['epochs']}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            if settings["grad_clip"] and settings["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            n_images += y.numel()
            total_loss += float(loss.detach()) * y.numel()
            total_correct += int((logits.argmax(1) == y).sum().item())
        val, *_ = infer(model, val_loader, criterion, device, amp)
        metric = settings["monitor"]
        if metric not in ("loss", "accuracy", "macro_f1", "balanced_accuracy"):
            raise ValueError(f"Invalid metric to monitor: {metric}")
        score = -val["loss"] if metric == "loss" else val[metric]
        improved = score > best_score + 1e-10
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
        else:
            bad_epochs += 1
        scheduler.step()
        row = {
            "epoch": epoch, "train_loss": total_loss / n_images,
            "train_accuracy": total_correct / n_images,
            "val_loss": val["loss"], "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_top5_accuracy": val["top5_accuracy"], "val_ece": val["ece_10_bins"],
            "learning_rate": lr, "epoch_seconds": time.perf_counter() - begin,
            "gpu_peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        save_checkpoint(run / "last_model.pth", model, optimizer, scheduler, scaler, epoch,
                        best_score, best_epoch, bad_epochs, config, fingerprints)
        if improved:
            save_checkpoint(run / "best_model.pth", model, optimizer, scheduler, scaler, epoch,
                            best_score, best_epoch, bad_epochs, config, fingerprints)
        print(f"Epoch {epoch}: train acc={row['train_accuracy']:.4f}, "
              f"val acc={val['accuracy']:.4f}, val macro-F1={val['macro_f1']:.4f}, "
              f"best={best_epoch}, {row['epoch_seconds']:.0f}s")
        learning_curves(run)
        if bad_epochs >= settings["patience"]:
            print(f"Early stopping: {bad_epochs} epochs without validation improvement")
            break

    # Produce detailed validation artifacts from the BEST model, not the last model.
    best_path = run / "best_model.pth"
    if best_path.exists():
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        val_metrics, ys, probs, indices = infer(model, val_loader, criterion, device, amp)
        filenames = [validation.records[i].filename for i in indices]
        scores = write_results(run / "validation_evaluation", ys, filenames, probs,
                               config["dataset"]["num_classes"])
        (run / "validation_metrics.json").write_text(
            json.dumps({"best_epoch": best_epoch, **scores}, indent=2), encoding="utf-8")
        print(f"Finished. Best validation epoch={best_epoch}. Validation metrics: {scores}")
    print(f"Experiment artifacts: {run}")


def main(name: str):
    parser = argparse.ArgumentParser(description=f"Train {name} on official IP102 splits")
    parser.add_argument("--data-root", help="Directory containing images/ and the 3 split TXT files")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--balance", choices=["none", "weighted_sampler", "equal_epoch", "weighted_loss"])
    parser.add_argument("--equal-k", type=int, help="Exact per-class training draws each epoch")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", help="Resume the same run from last_model.pth")
    args = parser.parse_args()
    train(name, args)
