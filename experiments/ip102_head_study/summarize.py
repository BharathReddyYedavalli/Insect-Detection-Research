"""Summarize completed full-IP102 head experiments; validation data only."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parent


def main():
    records = []
    groups = []
    for run in sorted((STUDY / "runs").glob("*")):
        metrics_path = run / "validation_metrics.json"
        config_path = run / "config.json"
        report_path = run / "validation_evaluation" / "per_class.csv"
        if not all(path.is_file() for path in (metrics_path, config_path, report_path)):
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        record = {
            "run": run.name, "model": config["model"], "head": config["head"],
            "dataset": config["dataset"], "epochs_budget": config["epochs"],
            "best_epoch": metrics["best_epoch"], "learning_rate": config["lr"],
            "seed": config["seed"], "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_top5_accuracy": metrics["top5_accuracy"],
        }
        records.append(record)
        original = Path(config["original_root"]) / "train.txt"
        counts = Counter(int(line.split()[-1]) for line in original.read_text(encoding="utf-8").splitlines() if line.strip())
        perclass = pd.read_csv(report_path)
        perclass["original_train_count"] = perclass["class_index"].map(counts)
        if perclass["original_train_count"].isna().any():
            raise ValueError(f"Missing original training count in {run}")
        perclass["quartile"] = pd.qcut(
            perclass["original_train_count"].rank(method="first"),
            q=4, labels=("Q1 rarest", "Q2", "Q3", "Q4 most common")
        )
        for quartile, subset in perclass.groupby("quartile", observed=True):
            groups.append({
                "run": run.name, "model": config["model"], "head": config["head"],
                "quartile": quartile, "classes": len(subset),
                "mean_f1": subset["f1"].mean(), "mean_recall": subset["recall"].mean(),
            })
    if not records:
        print("No completed runs yet. Train a model first.")
        return
    frame = pd.DataFrame(records).sort_values("validation_accuracy", ascending=False)
    frame.to_csv(STUDY / "summary.csv", index=False)
    pd.DataFrame(groups).to_csv(STUDY / "class_frequency_quartiles.csv", index=False)
    readable = frame.copy()
    for key in ("validation_accuracy", "validation_macro_f1", "validation_top5_accuracy"):
        readable[key] = (readable[key] * 100).round(2)
    print(readable.to_string(index=False))
    print("\nSaved summary.csv and class_frequency_quartiles.csv")


if __name__ == "__main__":
    main()
