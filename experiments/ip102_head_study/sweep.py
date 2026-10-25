"""Run a deliberately small, staged IP102 head-hyperparameter sweep.

Pilot: 2 models x linear head x 2 learning rates x 5 epochs = 4 runs.
Heads: 2 models x 2 MLP heads x 1 learning rate x 10 epochs = 4 runs.
Afterward manually select final 20-epoch configurations, trained from scratch.

Run ONE configuration at a time; finished runs are skipped and interrupted ones resumed:
  python experiments/ip102_head_study/sweep.py --preset pilot --max-runs 1
"""
from __future__ import annotations
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]


def grid(preset):
    if preset == "pilot":
        return list(itertools.product(
            ("resnet50", "convnext_tiny"), ("linear",),
            (0.00003, 0.0001), (5,)
        ))
    if preset == "heads":
        return list(itertools.product(
            ("resnet50", "convnext_tiny"),
            ("mlp512", "mlp1024_256"), (0.0001,), (10,)
        ))
    raise ValueError(preset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("pilot", "heads"), default="pilot")
    parser.add_argument("--dataset", choices=("original", "balanced"), default="original")
    parser.add_argument("--max-runs", type=int, default=1,
                        help="Limit new/unfinished runs launched per invocation.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_runs < 1:
        parser.error("--max-runs must be >= 1")
    pending = []
    for model, head, lr, epochs in grid(args.preset):
        name = (f"{args.dataset}_{args.preset}_{model}_{head}_"
                f"lr{lr:g}_e{epochs}_s{args.seed}_b{args.batch_size}")
        out = STUDY / "runs" / name
        if (out / "validation_metrics.json").exists():
            print(f"SKIP complete: {name}")
            continue
        command = [
            sys.executable, str(STUDY / "train.py"),
            "--model", model, "--head", head, "--lr", str(lr),
            "--epochs", str(epochs), "--dataset", args.dataset,
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers), "--seed", str(args.seed)
        ]
        checkpoint = out / "last_model.pth"
        if checkpoint.is_file():
            command += ["--resume", str(checkpoint)]
        elif out.exists():
            raise RuntimeError(f"Incomplete folder exists without checkpoint: {out}")
        else:
            command += ["--output-dir", str(out)]
        pending.append((name, command))
    if not pending:
        print("All configurations in this preset are complete.")
        return
    if args.dry_run:
        for name, command in pending:
            print(" ".join(command))
        return
    for name, command in pending[:args.max_runs]:
        print(f"\nRUN: {name}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("\nCompleted launched runs. Re-run the same sweep command for the next experiment.")


if __name__ == "__main__":
    main()
