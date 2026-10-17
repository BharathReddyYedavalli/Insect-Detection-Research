"""Explicit held-out evaluation of a locked checkpoint. Never trains or picks epochs."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import torch
from torch import nn

from shared.data import IP102Dataset, build_transform, dataset_summary, load_splits, make_loader
from shared.metrics import write_results
from shared.models import build_model
from shared.train import infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to your run's best_model.pth")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root", help="Override saved dataset location after moving to new computer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Explicitly replace prior evaluation output")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    # Only load trusted checkpoints that this project saved.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    root, splits, offset = load_splits(args.data_root or config["dataset"]["root"],
                                      config["dataset"]["num_classes"])
    _, fingerprints = dataset_summary(splits, offset, config["dataset"]["num_classes"])
    if fingerprints != state["split_fingerprints"]:
        raise ValueError("The split files changed since training. Evaluation aborted.")
    out = checkpoint.parent / ("test_evaluation" if args.split == "test" else "manual_val_evaluation")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise FileExistsError(f"{out} already exists; use --force to rerun deliberately")
    out.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], config["dataset"]["num_classes"], pretrained=False).to(device)
    model.load_state_dict(state["model_state"])
    size = config["dataset"]["image_size"]
    ds = IP102Dataset(root, splits[args.split], build_transform(size, False))
    loader = make_loader(ds, args.batch_size, args.workers, False, device=device.type)
    criterion = nn.CrossEntropyLoss()
    scores, ys, probs, indices = infer(model, loader, criterion, device,
                                      device.type == "cuda" and config["training"]["amp"])
    names = [ds.records[i].filename for i in indices]
    results = write_results(out, ys, names, probs, config["dataset"]["num_classes"])
    results["loss"] = scores["loss"]
    results["checkpoint_epoch"] = state["epoch"]
    results["split"] = args.split
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved {args.split} evaluation and graphs to {out}")


if __name__ == "__main__":
    main()
