"""Aggregate completed Hugging Face IP102 runs (validation metrics only)."""
import json
from pathlib import Path

import pandas as pd

STUDY = Path(__file__).resolve().parent


def main():
    rows = []
    for metrics_file in sorted((STUDY / "runs").glob("*/validation_metrics.json")):
        run = metrics_file.parent
        config_file = run / "config.json"
        if not config_file.exists():
            continue
        config = json.loads(config_file.read_text(encoding="utf-8"))
        scores = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows.append({
            "run": run.name,
            "model": config["model"],
            "hf_repo": config["repo"],
            "head": config["head"],
            "epochs": config["epochs"],
            "best_epoch": scores["best_epoch"],
            "image_size": config["image_size"],
            "backbone_lr": config["backbone_lr"],
            "head_lr": config["head_lr"],
            "augmentation": config["augmentation"],
            "sampler": config["sampler"],
            "mixup_alpha": config["mixup_alpha"],
            "cutmix_alpha": config["cutmix_alpha"],
            "seed": config["seed"],
            "top1_accuracy": scores["accuracy"],
            "macro_f1": scores["macro_f1"],
            "top5_accuracy": scores["top5_accuracy"],
        })
    if not rows:
        print("No completed models. Train a backbone first.")
        return
    df = pd.DataFrame(rows).sort_values("top1_accuracy", ascending=False)
    df.to_csv(STUDY / "summary.csv", index=False)
    pretty = df.copy()
    for field in ("top1_accuracy", "macro_f1", "top5_accuracy"):
        pretty[field] = (pretty[field] * 100).round(2)
    print(pretty.to_string(index=False))
    print(f"\nSaved: {STUDY / 'summary.csv'}")


if __name__ == "__main__":
    main()
