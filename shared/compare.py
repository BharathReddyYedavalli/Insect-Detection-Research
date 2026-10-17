"""Compare model runs by validation metrics, keeping old runs readable."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.config import PROJECT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="Model runs/ folders, or individual run folders")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or PROJECT / "comparisons" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for folder in args.runs:
        if folder.is_dir():
            for file in folder.rglob("validation_metrics.json"):
                files[file.resolve()] = file
    rows = []
    for file in files.values():
        run = file.parent
        config_file = run / "config.yaml"
        if not config_file.exists():
            continue
        import yaml
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        result = json.loads(file.read_text(encoding="utf-8"))
        row = {"run": str(run), "model": config["model"],
               "balance": config["training"].get("balance", "offline_augmented"),
               "seed": config["training"]["seed"],
               "epochs_requested": config["training"]["epochs"],
               "best_epoch": result["best_epoch"],
               "lr": config["training"]["learning_rate"],
               "image_size": config["dataset"]["image_size"]}
        row.update({k: result.get(k) for k in
                    ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision",
                     "macro_recall", "weighted_f1", "top5_accuracy", "ece_10_bins")})
        history = run / "history.csv"
        if history.exists():
            data = pd.read_csv(history)
            row["training_minutes"] = float(data["epoch_seconds"].sum() / 60)
            row["gpu_memory_gb_max"] = float(data["gpu_peak_memory_gb"].max())
        rows.append(row)
    if not rows:
        raise SystemExit("No completed runs found. Train a model first.")
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    df.to_csv(output / "runs.csv", index=False)
    agg = df.groupby(["model", "balance"], dropna=False).agg(
        runs=("run", "count"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        minutes_mean=("training_minutes", "mean") if "training_minutes" in df.columns else ("accuracy", "size")
    ).reset_index()
    agg.to_csv(output / "grouped_mean_std.csv", index=False)
    for key, ylabel, name in [
        ("accuracy", "Validation accuracy", "validation_accuracy_comparison.png"),
        ("macro_f1", "Validation macro-F1", "validation_f1_comparison.png"),
        ("balanced_accuracy", "Validation balanced accuracy", "validation_balanced_accuracy_comparison.png")
    ]:
        labels = df["model"] + " / " + df["balance"] + " / seed " + df["seed"].astype(str)
        plt.figure(figsize=(max(9, len(df) * .8), 6))
        plt.bar(range(len(df)), df[key])
        plt.xticks(range(len(df)), labels, rotation=70, ha="right")
        plt.ylabel(ylabel)
        plt.ylim(0, 1)
        plt.title(ylabel + " by run (validation split only)")
        plt.tight_layout()
        plt.savefig(output / name, dpi=200, bbox_inches="tight")
        plt.close()
    print("Compared", len(df), "runs. Saved to", output)


if __name__ == "__main__":
    main()
