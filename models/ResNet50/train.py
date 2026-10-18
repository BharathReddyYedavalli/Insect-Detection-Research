"""Train ResNet50 on the offline augmentation-balanced IP102 dataset.

One file contains the dataset, model, training, validation, checkpoints and
learning curves. The balanced images are prepared separately, once, by:
    python -m shared.prepare_balanced
Run from the repository directory:
    python models/ResNet50/train.py --epochs 12 --batch-size 16
Optional simple regularization experiment:
    python models/ResNet50/train.py --epochs 12 --batch-size 16 --extra-augmentation
"""
import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Allow imports from the repository root when called as a script from Windows PowerShell.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm

# Only evaluation statistics and paper plots are reused. Training lives in this file.
from shared.metrics import summarize, write_results


MODEL_NAME = "resnet50"
MODEL_TITLE = "ResNet50"
CLASS_COUNT = 102
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class InsectDataset(Dataset):
    """Read filenames/labels from an EXISTING split file. Never resplit IP102."""

    def __init__(self, dataset_root, split, image_size=224, extra_augmentation=False):
        self.root = Path(dataset_root)
        self.split = split
        self.records = []
        for line in (self.root / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                filename, class_id = line.rsplit(" ", 1)
                self.records.append((filename, int(class_id)))
        if not self.records:
            raise ValueError(f"No {split} images in {self.root}")
        # Data was already augmented and balanced on disk during preprocessing.
        # This extra, mild augmentation is optional and applied ONLY while training.
        if split == "train" and extra_augmentation:
            steps = [
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=8),
            ]
        else:
            steps = [
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
            ]
        self.transform = transforms.Compose([
            *steps,
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        filename, label = self.records[index]
        with Image.open(self.root / "images" / filename) as source:
            image = source.convert("RGB")
        return self.transform(image), label, filename


def check_balanced_dataset(dataset_root, classes):
    """Reject accidental training on raw/incorrectly prepared or changed splits."""
    root = Path(dataset_root).expanduser().resolve()
    manifest_file = root / "augmentation_manifest.json"
    if not manifest_file.is_file():
        raise ValueError("First prepare the balanced dataset: python -m shared.prepare_balanced")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("classes") != classes:
        raise ValueError("The number of classes does not match the preprocessing manifest.")
    expected_hashes = manifest.get("split_fingerprints", {})
    actual_hashes = {}
    for split in ("train", "val", "test"):
        items = (root / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        digest = hashlib.sha256()
        for line in items:
            if line.strip():
                filename, label = line.rsplit(" ", 1)
                digest.update(f"{filename}\t{int(label)}\n".encode("utf-8"))
        actual_hashes[split] = digest.hexdigest()
    if actual_hashes != expected_hashes:
        raise ValueError("Prepared split lists have changed since augmentation.")
    records = (root / "train.txt").read_text(encoding="utf-8").splitlines()
    class_counts = Counter(int(line.rsplit(" ", 1)[1]) for line in records if line.strip())
    target = manifest.get("target_per_class")
    if set(class_counts) != set(range(classes)) or set(class_counts.values()) != {target}:
        raise ValueError("Training images are not exactly balanced across all classes.")
    if not manifest.get("augmentation_only_on_train") or not manifest.get("evaluation_images_unchanged"):
        raise ValueError("Only training images may be augmented; original validation/test required.")
    return root, manifest, actual_hashes


def get_model(classes, pretrained=True):
    """Pretrained ResNet50 with a fresh classification head for all IP102 classes."""
    model = models.resnet50(weights="DEFAULT" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, classes)
    return model


def validate(model, loader, criterion, device, use_amp, classes):
    """Evaluate on the original validation set; never update model weights."""
    model.eval()
    labels, probabilities, filenames = [], [], []
    total_loss, total = 0.0, 0
    with torch.inference_mode():
        for images, correct, names in loader:
            images = images.to(device, non_blocking=True)
            correct = correct.to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp):
                logits = model(images)
            loss = criterion(logits.float(), correct)
            total_loss += float(loss.item()) * len(correct)
            total += len(correct)
            labels.extend(correct.cpu().tolist())
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            filenames.extend(names)
    probs = np.concatenate(probabilities)
    scores, _, _, _ = summarize(labels, probs.argmax(axis=1), probs, classes)
    scores["loss"] = total_loss / total
    return scores, labels, probs, filenames


def save_graphs(history, output):
    """Save the familiar loss/accuracy and macro-F1 learning curves."""
    graph_folder = output / "graphs"
    graph_folder.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    for series, ylabel, filename in [
        (("train_loss", "val_loss"), "Cross-entropy loss", "loss_by_epoch.png"),
        (("train_accuracy", "val_accuracy"), "Accuracy", "accuracy_by_epoch.png"),
        (("val_macro_f1",), "Macro-F1", "validation_macro_f1.png"),
    ]:
        plt.figure(figsize=(9, 5))
        for key in series:
            plt.plot(epochs, [item[key] for item in history], marker=".", label=key)
        plt.title(f"{MODEL_TITLE} {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(graph_folder / filename, dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=f"Simple {MODEL_TITLE} IP102 training")
    parser.add_argument("--data-root", default=str(ROOT / "data" / "ip102_balanced"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--extra-augmentation", action="store_true",
                        help="Try mild new flips and rotations on each training pass")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--classes", type=int, default=CLASS_COUNT,
                        help="102 for IP102; small values only for synthetic tests")
    parser.add_argument("--output-dir", type=Path,
                        help="Optional new experiment directory")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.image_size < 32:
        parser.error("epochs/batch-size must be positive, workers >=0 and image-size >=32")
    if args.lr <= 0 or args.classes < 2:
        parser.error("learning rate must be positive and classes >=2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    dataset_root, manifest, hashes = check_balanced_dataset(args.data_root, args.classes)
    train_set = InsectDataset(dataset_root, "train", args.image_size, args.extra_augmentation)
    val_set = InsectDataset(dataset_root, "val", args.image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=use_amp, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=use_amp, persistent_workers=args.workers > 0,
    )

    # Create a new folder for every run; never overwrite earlier checkpoints.
    run = args.output_dir or (ROOT / "models" / MODEL_TITLE / "runs" /
                               f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}")
    run = run.expanduser().resolve()
    run.mkdir(parents=True, exist_ok=False)

    config = {
        "model": MODEL_NAME,
        "dataset": {
            "root": str(dataset_root), "num_classes": args.classes,
            "image_size": args.image_size, "workers": args.workers,
        },
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.lr, "weight_decay": args.weight_decay,
            "seed": args.seed, "pretrained": not args.no_pretrained,
            "amp": use_amp, "balance": "offline_augmented",
        },
        "augmentation": {"extra_training_augmentation": args.extra_augmentation},
    }
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run / "augmentation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run / "splits_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    model = get_model(args.classes, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=use_amp)
    history = []
    best_accuracy, best_epoch = -1.0, 0

    print(f"{MODEL_TITLE} | {device} | train={len(train_set):,} "
          f"({manifest['target_per_class']:,}/class) | val={len(val_set):,} | "
          f"lr={args.lr} | extra_aug={args.extra_augmentation} | output={run}", flush=True)

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        running_loss, correct_predictions, seen = 0.0, 0, 0
        if use_amp:
            torch.cuda.reset_peak_memory_stats()
        for images, labels, _ in tqdm(train_loader, desc=f"{MODEL_NAME} epoch {epoch}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                outputs = model(images)
            loss = criterion(outputs.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * len(labels)
            correct_predictions += int((outputs.argmax(1) == labels).sum().item())
            seen += len(labels)

        val, _, _, _ = validate(model, val_loader, criterion, device, use_amp, args.classes)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_accuracy": correct_predictions / seen,
            "val_loss": val["loss"], "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "val_balanced_accuracy": val["balanced_accuracy"],
            "val_top5_accuracy": val["top5_accuracy"],
            "val_ece": val["ece_10_bins"],
            "learning_rate": args.lr,
            "epoch_seconds": time.perf_counter() - started,
            "gpu_peak_memory_gb": (
                torch.cuda.max_memory_allocated() / (1024**3) if use_amp else 0
            ),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        save_graphs(history, run)

        if val["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = val["accuracy"], epoch
            selected = True
        else:
            selected = False
        state = {
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "best_accuracy": best_accuracy, "best_epoch": best_epoch,
            "config": config, "split_fingerprints": hashes,
        }
        # Save both the most recent checkpoint and the highest validation accuracy.
        temp = run / "last_model.writing"
        torch.save(state, temp)
        temp.replace(run / "last_model.pth")
        if selected:
            temp = run / "best_model.writing"
            torch.save(state, temp)
            temp.replace(run / "best_model.pth")

        print(f"Epoch {epoch}: train={100*row['train_accuracy']:.2f}% "
              f"val={100*val['accuracy']:.2f}% "
              f"macro-F1={100*val['macro_f1']:.2f}% best={best_epoch}", flush=True)

    # Generate full paper-ready validation metrics for the best saved epoch.
    state = torch.load(run / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    _, y_true, probs, names = validate(model, val_loader, criterion, device, use_amp, args.classes)
    summary = write_results(run / "validation_evaluation", y_true, names, probs, args.classes)
    (run / "validation_metrics.json").write_text(
        json.dumps({"best_epoch": best_epoch, **summary}, indent=2), encoding="utf-8")
    print(f"Finished. Best epoch={best_epoch} | "
          f"validation accuracy={100*summary['accuracy']:.2f}% | "
          f"macro-F1={100*summary['macro_f1']:.2f}%")
    print(f"Saved checkpoints, CSVs and graphs to {run}", flush=True)


if __name__ == "__main__":
    main()
