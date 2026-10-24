"""Matched-update ResNet18 ablation: original IP102 vs offline-balanced IP102.

Uses the EXACT model, dataset transforms, optimizer, and validation routine from
models/ResNet18/train.py, but without modifying the original training scripts.

Examples, run separately:
    python experiments/balance_ablation_resnet18.py --condition original
    python experiments/balance_ablation_resnet18.py --condition balanced

Default: 55,000 optimizer updates, batch 32, validation every 11,000 updates
for both conditions. Validation/test images NEVER enter training.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from models.ResNet18.train import (
    InsectDataset, check_balanced_dataset, get_model, validate,
)
from shared.data import load_splits, dataset_summary
from shared.metrics import write_results


class MatchedStepBatchSampler(Sampler[list[int]]):
    """Deterministic shuffled batches addressable by optimizer step.

    The last incomplete batch in each dataset traversal is dropped, so every
    optimizer update sees exactly batch_size images in both conditions.
    The order at each step can be reconstructed after restarting.
    """

    def __init__(self, length: int, batch_size: int, seed: int,
                 start_step: int, stop_step: int):
        if length < batch_size:
            raise ValueError("Training dataset must contain at least one full batch.")
        if start_step < 0 or stop_step < start_step:
            raise ValueError("Invalid step interval.")
        self.length = length
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.stop_step = stop_step
        self.steps_per_pass = length // batch_size

    def __len__(self):
        return self.stop_step - self.start_step

    def __iter__(self):
        step = self.start_step
        while step < self.stop_step:
            pass_index, offset = divmod(step, self.steps_per_pass)
            generator = torch.Generator().manual_seed(self.seed + pass_index)
            order = torch.randperm(self.length, generator=generator)
            while offset < self.steps_per_pass and step < self.stop_step:
                begin = offset * self.batch_size
                yield order[begin:begin + self.batch_size].tolist()
                offset += 1
                step += 1


def verify_datasets(original: Path, balanced: Path, classes: int):
    """Verify official originals and manifest; use same ORIGINAL validation images."""
    source, src_splits, source_offset = load_splits(
        original, num_classes=classes, check_files=False
    )
    if source_offset != 0:
        raise ValueError("This experiment expects normalized 0-based IP102 labels.")
    _, source_hashes = dataset_summary(src_splits, source_offset, classes)
    prepared, manifest, balanced_hashes = check_balanced_dataset(balanced, classes)
    if manifest.get("source_split_fingerprints") != source_hashes:
        raise ValueError("Balanced augmentation was NOT prepared from this original dataset.")
    _, prepared_splits, prepared_offset = load_splits(
        prepared, num_classes=classes, check_files=False
    )
    if prepared_offset != 0:
        raise ValueError("Prepared dataset label indexing changed.")
    for split in ("val", "test"):
        original_items = {(item.filename, item.label) for item in src_splits[split]}
        prepared_items = {(item.filename, item.label) for item in prepared_splits[split]}
        if original_items != prepared_items or len(src_splits[split]) != len(prepared_splits[split]):
            raise ValueError(f"{split} differs between original and prepared datasets.")
    return source, prepared, source_hashes, balanced_hashes


def atomic_save(state, path: Path):
    temp = path.with_suffix(".writing")
    torch.save(state, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", choices=("original", "balanced"), required=True)
    p.add_argument("--original-root", type=Path, default=ROOT / "data/ip102_v1.1")
    p.add_argument("--balanced-root", type=Path, default=ROOT / "data/ip102_balanced")
    p.add_argument("--steps", type=int, default=55000,
                   help="TOTAL optimizer updates in either condition.")
    p.add_argument("--eval-every", type=int, default=11000)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--run-until", type=int,
                   help="Optionally stop this invocation at an optimizer step and resume later.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--classes", type=int, default=102)
    p.add_argument("--no-pretrained", action="store_true", help="CPU synthetic tests only.")
    p.add_argument("--resume", type=Path,
                   help="Resume this experiment from its own last_model.pth.")
    p.add_argument("--output-dir", type=Path,
                   help="New output folder; do not use with --resume.")
    args = p.parse_args()

    if args.steps < 1 or args.eval_every < 1 or args.checkpoint_every < 1:
        p.error("Steps and checkpoint/evaluation intervals must be positive.")
    if args.batch_size < 1 or args.lr <= 0 or args.workers < 0 or args.classes < 2:
        p.error("Invalid batch size, learning rate, workers, or classes.")
    if args.image_size < 32:
        p.error("Image size must be at least 32.")
    if args.run_until is not None and not 1 <= args.run_until <= args.steps:
        p.error("--run-until must be between 1 and --steps.")
    if args.resume is not None and args.output_dir is not None:
        p.error("Do not use --output-dir with --resume.")

    source, prepared, source_hashes, balanced_hashes = verify_datasets(
        args.original_root, args.balanced_root, args.classes
    )
    training_root = source if args.condition == "original" else prepared
    config = {
        "experiment": "matched_step_resnet18_balance_ablation_v1",
        "condition": args.condition,
        "original_root": str(source),
        "balanced_root": str(prepared),
        "source_split_fingerprints": source_hashes,
        "balanced_split_fingerprints": balanced_hashes,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "checkpoint_every": args.checkpoint_every,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "workers": args.workers,
        "image_size": args.image_size,
        "classes": args.classes,
        "pretrained": not args.no_pretrained,
        "online_training_augmentation": "horizontal_flip_and_rotation_8_degrees",
        "validation": "original_official_IP102_split",
        "selection": "highest_validation_accuracy_at_matched_steps",
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    train_data = InsectDataset(training_root, "train", args.image_size, extra_augmentation=True)
    # Evaluate BOTH experiments on the exact same ORIGINAL validation images.
    val_data = InsectDataset(source, "val", args.image_size, extra_augmentation=False)
    model = get_model(args.classes, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    step, history, best_acc, best_step = 0, [], -1.0, 0
    window = {"loss_sum": 0.0, "correct": 0, "seen": 0}
    if args.resume:
        checkpoint_file = args.resume.expanduser().resolve()
        if checkpoint_file.name != "last_model.pth" or not checkpoint_file.is_file():
            p.error("Resume requires this experiment's existing last_model.pth.")
        state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if state.get("config") != config:
            p.error("Resume settings do not match the saved run (except --resume).")
        out = checkpoint_file.parent
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        step = int(state["step"])
        history = state["history"]
        best_acc, best_step = float(state["best_accuracy"]), int(state["best_step"])
        window = state["train_window"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if step >= args.steps and (out / "validation_metrics.json").is_file():
            print("Run is already complete:", out)
            return
        if step > args.steps:
            p.error("Checkpoint exceeds --steps.")
        print(f"Resuming {args.condition} after step {step:,}/{args.steps:,}", flush=True)
    else:
        out = args.output_dir or (
            ROOT / "experiments" / "runs" /
            f"resnet18_{args.condition}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        )
        out = out.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=False)
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    stop_step = args.run_until if args.run_until is not None else args.steps
    if stop_step < step:
        p.error("--run-until is before the saved checkpoint step.")
    sampler = MatchedStepBatchSampler(
        len(train_data), args.batch_size, args.seed, step, stop_step
    )
    train_loader = DataLoader(
        train_data, batch_sampler=sampler, num_workers=args.workers,
        pin_memory=use_amp, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=use_amp,
        persistent_workers=args.workers > 0,
    )

    def save_state(destination: Path):
        state = {
            "config": config, "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history, "best_accuracy": best_acc, "best_step": best_step,
            "train_window": window.copy(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if use_amp else None,
        }
        atomic_save(state, destination)

    print(f"{args.condition.upper()} | train={len(train_data):,} | "
          f"val={len(val_data):,} original images | batch={args.batch_size} | "
          f"steps={args.steps:,} | output={out}", flush=True)
    started = time.perf_counter()
    progress = tqdm(train_loader, total=stop_step - step, desc=args.condition)
    for images, labels, _ in progress:
        model.train()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(images)
        loss = criterion(logits.float(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step += 1
        window["loss_sum"] += float(loss.item()) * len(labels)
        window["correct"] += int((logits.argmax(1) == labels).sum().item())
        window["seen"] += len(labels)

        needs_eval = step % args.eval_every == 0 or step == args.steps
        if needs_eval:
            val, _, _, _ = validate(
                model, val_loader, criterion, device, use_amp, args.classes
            )
            row = {
                "step": step,
                "images_seen": step * args.batch_size,
                "train_accuracy": window["correct"] / window["seen"],
                "train_loss": window["loss_sum"] / window["seen"],
                "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
                "val_balanced_accuracy": val["balanced_accuracy"],
                "val_loss": val["loss"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            window = {"loss_sum": 0.0, "correct": 0, "seen": 0}
            if val["accuracy"] > best_acc:
                best_acc, best_step = val["accuracy"], step
                save_state(out / "best_model.pth")
            print(f"Step {step:,}: validation accuracy={val['accuracy']*100:.2f}% "
                  f"macro-F1={val['macro_f1']*100:.2f}% "
                  f"best_step={best_step:,}", flush=True)

        if needs_eval or step % args.checkpoint_every == 0:
            save_state(out / "last_model.pth")
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    if step < args.steps:
        save_state(out / "last_model.pth")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print("Paused at step " + str(step) + ". Resume from " +
              str(out / "last_model.pth"), flush=True)
        return

    # Produce paper-ready class-level output for the selected validation checkpoint.
    best = torch.load(out / "best_model.pth", weights_only=False, map_location="cpu")
    model.load_state_dict(best["model_state"])
    scores, y_true, probs, names = validate(
        model, val_loader, criterion, device, use_amp, args.classes
    )
    report = write_results(out / "validation_evaluation", y_true, names, probs, args.classes)
    (out / "validation_metrics.json").write_text(
        json.dumps({"best_step": best_step, "condition": args.condition, **report}, indent=2),
        encoding="utf-8",
    )
    print(f"FINISHED {args.condition}: best at {best_step:,} steps | "
          f"accuracy={scores['accuracy']*100:.2f}% | "
          f"macro-F1={scores['macro_f1']*100:.2f}% | {out}", flush=True)


if __name__ == "__main__":
    main()
