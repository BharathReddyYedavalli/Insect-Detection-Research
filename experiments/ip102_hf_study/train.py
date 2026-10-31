"""Fine-tune three Hugging Face pretrained backbones on OFFICIAL IP102 splits.

The pretrained models are cached by Hugging Face on the local computer.
No IP102 test examples enter fitting, selection, evaluation, or ensembling.

Quick example (from repo root):
 python experiments/ip102_hf_study/train.py --model convnext_small --epochs 20 --batch-size 4 --accum-steps 4 --stop-after-epoch 1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

from shared.data import IP102Dataset, dataset_summary, load_splits
from shared.metrics import summarize, write_results
from experiments.ip102_head_study.train import build_head, draw_curves, save_checkpoint

STUDY = Path(__file__).resolve().parent
MODELS = {
    "convnext_small": "facebook/convnext-small-224",
    "swin_base": "microsoft/swin-base-patch4-window7-224-in22k",
    "dinov2_base": "facebook/dinov2-base",
}
BACKBONES = {
    "convnext_small": "convnext",
    "swin_base": "swin",
    "dinov2_base": "dinov2",
}
CLASSES = 102


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--head", choices=("linear", "mlp512", "mlp1024_256"), default="linear")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--backbone-lr", type=float, default=0.00002)
    p.add_argument("--head-lr", type=float, default=0.0001)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--augmentation", choices=("basic", "strong"), default="basic")
    p.add_argument("--mixup-alpha", type=float, default=0.0)
    p.add_argument("--cutmix-alpha", type=float, default=0.0)
    p.add_argument("--sampler", choices=("original", "sqrt"), default="original")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Train only the classifier (useful for DINOv2 linear probing).")
    p.add_argument("--unfreeze-last-n", type=int, default=0,
                   help="DINOv2 only: train its final N transformer blocks and classifier.")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--amp", choices=("fp16", "bf16", "off"), default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--original-root", type=Path, default=ROOT / "data" / "ip102_v1.1")
    p.add_argument("--classes", type=int, default=CLASSES, help="Only override in tiny synthetic smoke tests.")
    p.add_argument("--revision", default="main", help="HF revision; resolved hash is saved for resume.")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after-epoch", type=int)
    args = p.parse_args(argv)
    if not 1 <= args.epochs <= 100:
        p.error("--epochs must be between 1 and 100.")
    if args.batch_size < 1 or args.accum_steps < 1 or args.workers < 0 or args.classes < 2:
        p.error("Invalid batch/accumulation/workers/classes.")
    if args.image_size < 56:
        p.error("--image-size must be at least 56.")
    if args.model == "dinov2_base" and args.image_size % 14:
        p.error("DINOv2 uses 14-pixel patches: image size must be divisible by 14 (224, 336, 378...).")
    if args.model == "swin_base" and args.image_size % 32:
        p.error("Swin-Base requires image size divisible by 32 (224, 384...).")
    if args.model == "convnext_small" and args.image_size % 32:
        p.error("ConvNeXt-Small requires image size divisible by 32 (224, 384...).")
    if args.backbone_lr <= 0 or args.head_lr <= 0 or args.weight_decay < 0:
        p.error("Learning rates must be positive, weight decay must be nonnegative.")
    if not 0 <= args.dropout < 1 or not 0 <= args.label_smoothing < 1:
        p.error("Dropout/label smoothing must be in [0,1).")
    if args.mixup_alpha < 0 or args.cutmix_alpha < 0:
        p.error("Mixup/CutMix strengths must be nonnegative.")
    if not 0 <= args.warmup_epochs < args.epochs:
        p.error("Warmup must be >=0 and strictly fewer than total epochs.")
    if args.freeze_backbone and args.unfreeze_last_n:
        p.error("Use either --freeze-backbone or --unfreeze-last-n, not both.")
    if args.unfreeze_last_n and args.model != "dinov2_base":
        p.error("--unfreeze-last-n currently supports only DINOv2.")
    if args.unfreeze_last_n < 0 or args.unfreeze_last_n > 12:
        p.error("DINOv2 unfreeze-last-n must be between 0 and 12.")
    if args.resume and args.output_dir:
        p.error("Do not pass --output-dir alongside --resume.")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        p.error("stop-after-epoch must be within training budget.")
    return args


def make_transforms(size, training, augmentation, mean, std):
    if training:
        steps = [
            T.RandomResizedCrop(size, scale=((0.6 if augmentation == "strong" else 0.75), 1.0)),
            T.RandomHorizontalFlip(),
        ]
        if augmentation == "strong":
            steps += [T.RandAugment(num_ops=2, magnitude=7)]
        else:
            steps += [T.RandomRotation(10), T.ColorJitter(0.1, 0.1, 0.1)]
    else:
        steps = [T.Resize(round(size * 256 / 224)), T.CenterCrop(size)]
    steps += [T.ToTensor(), T.Normalize(mean, std)]
    if training and augmentation == "strong":
        steps.append(T.RandomErasing(p=0.25))
    return T.Compose(steps)


def replace_classifier(model, head, classes, dropout):
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise TypeError("Expected Hugging Face image-classification model with a linear .classifier.")
    features = classifier.in_features
    model.classifier = build_head(features, classes, head, dropout)
    return features


def apply_trainability(model, name, freeze, unfreeze_last_n):
    backbone = getattr(model, BACKBONES[name])
    if freeze or unfreeze_last_n:
        backbone.requires_grad_(False)
    if unfreeze_last_n:
        blocks = backbone.encoder.layer
        if unfreeze_last_n > len(blocks):
            raise ValueError("Requested more transformer blocks than present.")
        for layer in blocks[-unfreeze_last_n:]:
            layer.requires_grad_(True)
        if hasattr(backbone, "layernorm"):
            backbone.layernorm.requires_grad_(True)
    return backbone


def mix_batch(images, labels, mixup_alpha, cutmix_alpha):
    """Mixup/CutMix only inside the training loop; returns two label sets and lambda."""
    enabled = [method for method, alpha in (("mixup", mixup_alpha), ("cutmix", cutmix_alpha)) if alpha > 0]
    if not enabled:
        return images, labels, labels, 1.0
    method = random.choice(enabled)
    alpha = mixup_alpha if method == "mixup" else cutmix_alpha
    lam = float(np.random.beta(alpha, alpha))
    order = torch.randperm(len(labels), device=images.device)
    if method == "mixup":
        images = lam * images + (1.0 - lam) * images[order]
    else:
        _, _, h, w = images.shape
        cut_ratio = math.sqrt(1.0 - lam)
        ch, cw = int(h * cut_ratio), int(w * cut_ratio)
        cy, cx = random.randrange(h), random.randrange(w)
        y1, y2 = max(0, cy - ch // 2), min(h, cy + ch // 2)
        x1, x2 = max(0, cx - cw // 2), min(w, cx + cw // 2)
        original = images.clone()
        images[:, :, y1:y2, x1:x2] = original[order, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    return images, labels, labels[order], lam


def evaluate(model, loader, criterion, device, amp_mode, name, image_size, classes):
    model.eval()
    targets, probabilities, filenames = [], [], []
    total_loss, seen = 0., 0
    use_amp = device.type == "cuda" and amp_mode != "off"
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    kwargs = {"interpolate_pos_encoding": True} if name == "swin_base" and image_size != 224 else {}
    with torch.inference_mode():
        for images, labels, indices in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                logits = model(pixel_values=images, **kwargs).logits
            total_loss += criterion(logits.float(), labels).item() * len(labels)
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            targets.extend(labels.cpu().tolist())
            filenames.extend(loader.dataset.records[int(i)].filename for i in indices)
            seen += len(labels)
    probs = np.concatenate(probabilities)
    scores, *_ = summarize(targets, probs.argmax(axis=1), probs, classes)
    scores["loss"] = total_loss / seen
    return scores, np.asarray(targets, dtype=np.int32), probs, np.asarray(filenames, dtype=str)


def make_schedule(epochs, warmup):
    def scale(index):
        if warmup and index < warmup:
            return (index + 1) / warmup
        fraction = (index - warmup) / max(1, epochs - warmup)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * fraction)))
    return scale


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    original, splits, offset = load_splits(args.original_root, args.classes)
    _, hashes = dataset_summary(splits, offset, args.classes)
    if args.classes == CLASSES and (len(splits["train"]), len(splits["val"]), len(splits["test"])) != (45095, 7508, 22619):
        raise ValueError("Official full-IP102 split sizes do not match expected 45,095/7,508/22,619.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU doesn't support BF16. Use --amp fp16.")
    if args.resume:
        checkpoint_path = args.resume.expanduser().resolve()
        if checkpoint_path.name != "last_model.pth" or not checkpoint_path.is_file():
            raise FileNotFoundError("--resume must point to an existing last_model.pth from this study.")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        revision = state["config"]["resolved_revision"] or state["config"]["requested_revision"]
        run = checkpoint_path.parent
    else:
        revision = args.revision
        state = None
        name = (f"{args.model}_{args.head}_lr{args.backbone_lr:g}_"
                f"s{args.seed}_e{args.epochs}_{datetime.now():%Y%m%d_%H%M%S}")
        run = (args.output_dir if args.output_dir else STUDY / "runs" / name).expanduser().resolve()
        if run.exists():
            raise FileExistsError(f"Refusing to overwrite existing run directory {run}")

    repo = MODELS[args.model]
    print(f"Loading {repo} from Hugging Face (revision {revision}). First run downloads weights to HF cache.",
          flush=True)
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision)
    model, loading_info = AutoModelForImageClassification.from_pretrained(
        repo, revision=revision, num_labels=args.classes, ignore_mismatched_sizes=True,
        use_safetensors=True, output_loading_info=True, trust_remote_code=False
    )
    # A task-specific 102-way classification head must be new; all other weights should load.
    unknown_missing = [
        x for x in loading_info.get("missing_keys", [])
        if "classifier" not in x
    ]
    if unknown_missing:
        raise RuntimeError(f"Unexpected missing pretrained backbone parameters: {unknown_missing[:12]}")
    features = replace_classifier(model, args.head, args.classes, args.dropout)
    backbone = apply_trainability(model, args.model, args.freeze_backbone, args.unfreeze_last_n)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)
    mean = tuple(processor.image_mean)
    std = tuple(processor.image_std)
    train_set = IP102Dataset(original, splits["train"], make_transforms(
        args.image_size, True, args.augmentation, mean, std
    ))
    val_set = IP102Dataset(original, splits["val"], make_transforms(
        args.image_size, False, args.augmentation, mean, std
    ))
    shuffle_gen = torch.Generator().manual_seed(args.seed)
    worker_gen = torch.Generator().manual_seed(args.seed + 1000)
    counts = Counter(r.label for r in splits["train"])
    sampler = None
    if args.sampler == "sqrt":
        weights = torch.DoubleTensor([1.0 / math.sqrt(counts[y]) for y in train_set.targets])
        sampler = WeightedRandomSampler(weights, num_samples=len(train_set),
                                        replacement=True, generator=shuffle_gen)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler, generator=shuffle_gen if sampler is None else worker_gen,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            num_workers=args.workers, shuffle=False,
                            pin_memory=device.type == "cuda")
    parameters = [
        {"params": [p for p in backbone.parameters() if p.requires_grad],
         "lr": args.backbone_lr, "initial_lr": args.backbone_lr},
        {"params": list(model.classifier.parameters()), "lr": args.head_lr,
         "initial_lr": args.head_lr},
    ]
    parameters = [group for group in parameters if group["params"]]
    optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_schedule(args.epochs, args.warmup_epochs)
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda" and args.amp == "fp16")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    config = dict(
        model=args.model, repo=repo, requested_revision=args.revision,
        resolved_revision=getattr(model.config, "_commit_hash", None),
        head=args.head, features=features, classes=args.classes, dataset="original",
        split_fingerprints=hashes, data_root=str(original), epochs=args.epochs,
        batch_size=args.batch_size, accum_steps=args.accum_steps, image_size=args.image_size,
        backbone_lr=args.backbone_lr, head_lr=args.head_lr, weight_decay=args.weight_decay,
        dropout=args.dropout, label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs, augmentation=args.augmentation,
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, sampler=args.sampler,
        freeze_backbone=args.freeze_backbone, unfreeze_last_n=args.unfreeze_last_n,
        gradient_checkpointing=args.gradient_checkpointing, amp=args.amp,
        seed=args.seed, workers=args.workers, image_mean=mean, image_std=std,
        criterion="soft-target mixture cross entropy" if args.mixup_alpha or args.cutmix_alpha else "cross entropy",
        optimizer="AdamW", scheduler="warmup then cosine",
        selection="highest official validation top-1 accuracy",
    )
    best, best_epoch, history, start = -1., 0, [], 1
    if state is not None:
        if config != state["config"]:
            raise ValueError("Resume arguments, dataset or pretrained revision differ from original run.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        best, best_epoch = state["best_acc"], state["best_epoch"]
        history = state["history"]
        start = int(state["epoch"]) + 1
        if start > args.epochs or len(history) != start - 1:
            raise ValueError("Invalid/already finished checkpoint.")
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        shuffle_gen.set_state(state["shuffle_rng"])
        worker_gen.set_state(state["worker_rng"])
        if not (run / "best_model.pth").exists():
            raise FileNotFoundError("Missing best_model.pth in resumed run.")
    else:
        run.mkdir(parents=True, exist_ok=False)
        (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"{args.model} | {len(train_set):,} train / {len(val_set):,} val / "
          f"{len(splits['test']):,} untouched test | device={device}, "
          f"batch={args.batch_size}, accumulation={args.accum_steps}, "
          f"resolution={args.image_size}, head={args.head} | {run}", flush=True)
    kwargs = {"interpolate_pos_encoding": True} if args.model == "swin_base" and args.image_size != 224 else {}
    use_amp = device.type == "cuda" and args.amp != "off"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    trainable = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(start, args.epochs + 1):
        began = time.perf_counter()
        model.train()
        if args.freeze_backbone:
            backbone.eval()
        loss_sum, seen, correct = 0., 0, 0
        mixed = args.mixup_alpha > 0 or args.cutmix_alpha > 0
        optimizer.zero_grad(set_to_none=True)
        batches = len(train_loader)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        for step, (images, labels, _) in enumerate(
            tqdm(train_loader, desc=f"{args.model} epoch {epoch}/{args.epochs}")
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images, first, second, lam = mix_batch(
                images, labels, args.mixup_alpha, args.cutmix_alpha
            )
            # Last gradient accumulation window may be smaller.
            window_start = (step // args.accum_steps) * args.accum_steps
            window = min(args.accum_steps, batches - window_start)
            with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                logits = model(pixel_values=images, **kwargs).logits
                loss = lam * criterion(logits.float(), first)
                if lam != 1.0:
                    loss = loss + (1.0 - lam) * criterion(logits.float(), second)
            scaler.scale(loss / window).backward()
            if (step + 1) % args.accum_steps == 0 or (step + 1) == batches:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.item()) * len(labels)
            if not mixed:
                correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += len(labels)
        scores, _, _, _ = evaluate(model, val_loader, criterion, device,
                                   args.amp, args.model, args.image_size, args.classes)
        row = {
            "epoch": epoch, "backbone_lr": current_lrs[0],
            "head_lr": current_lrs[-1],
            "train_loss": loss_sum / seen,
            "train_accuracy": float("nan") if mixed else correct / seen,
            "val_accuracy": scores["accuracy"], "val_macro_f1": scores["macro_f1"],
            "val_balanced_accuracy": scores["balanced_accuracy"],
            "val_top5_accuracy": scores["top5_accuracy"],
            "val_loss": scores["loss"], "seconds": time.perf_counter() - began
        }
        history.append(row)
        improved = scores["accuracy"] > best
        if improved:
            best, best_epoch = scores["accuracy"], epoch
        scheduler.step()
        state = {
            "config": config, "epoch": epoch, "history": history,
            "best_acc": best, "best_epoch": best_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "shuffle_rng": shuffle_gen.get_state(),
            "worker_rng": worker_gen.get_state(),
        }
        save_checkpoint(run / "last_model.pth", state)
        if improved:
            save_checkpoint(run / "best_model.pth", state)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        # Existing plot helper expects these exact keys:
        plot_history = [{**r, "lr": r["backbone_lr"]} for r in history]
        draw_curves(plot_history, run)
        print(f"Epoch {epoch}/{args.epochs}: validation={scores['accuracy']*100:.2f}% "
              f"macro-F1={scores['macro_f1']*100:.2f}% best_epoch={best_epoch}",
              flush=True)
        if args.stop_after_epoch == epoch and epoch < args.epochs:
            print(f"PAUSED. Resume from: {run / 'last_model.pth'}", flush=True)
            return
    # Validate only the BEST checkpoint, and save all per-image scores for later ensembling.
    best_state = torch.load(run / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model_state"])
    scores, targets, probs, names = evaluate(
        model, val_loader, criterion, device, args.amp, args.model, args.image_size, args.classes
    )
    write_results(run / "validation_evaluation", targets, names.tolist(), probs, args.classes)
    np.savez_compressed(
        run / "validation_probabilities.npz", filenames=names,
        labels=targets, probabilities=probs.astype(np.float32)
    )
    (run / "validation_metrics.json").write_text(
        json.dumps({"model": args.model, "head": args.head, "best_epoch": best_epoch,
                    "split": "validation", **scores}, indent=2), encoding="utf-8"
    )
    print(f"FINISHED | best epoch {best_epoch} | "
          f"accuracy {scores['accuracy']*100:.2f}% | macro-F1 "
          f"{scores['macro_f1']*100:.2f}% | {run}", flush=True)


if __name__ == "__main__":
    main()
