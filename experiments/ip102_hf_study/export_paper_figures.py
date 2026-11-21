"""Export publication figures 2-5 using existing frozen IP102 research results.

From the repository root:
    python experiments/ip102_hf_study/export_paper_figures.py

Requires only already-saved history CSVs and frozen test-evaluation reports.
Does NOT train models, access test image pixels, or alter test predictions.
Figure 1 is the methodology flowchart already prepared separately.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.data import load_splits

STUDY = Path(__file__).resolve().parent
FROZEN = "frozen_dinov2_s42_s43_s44_swin_s42_equal4_v1"
RUNS = {
    "DINOv2 (seed 44)": "dinov2_base_linear_lr5e-06_s44_e20_20261105_173316",
    "Swin-Base (seed 42)": "swin_base_linear_lr1e-05_s42_e20_20261101_111552",
}
TEST_MODELS = (
    ("DINOv2\nseed 42", "model_1_dinov2_base_s42"),
    ("DINOv2\nseed 43", "model_2_dinov2_base_s43"),
    ("DINOv2\nseed 44", "model_3_dinov2_base_s44"),
    ("Swin-Base\nseed 42", "model_4_swin_base_s42"),
    ("Four-model\nensemble", "ensemble_test_evaluation"),
)
FILENAMES = (
    "Fig2_training_validation_accuracy.png",
    "Fig3_test_accuracy_comparison.png",
    "Fig4_confusion_matrix.png",
    "Fig5_training_frequency_vs_F1.png",
)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required research output not found: {path}\n"
            "Confirm you are in the repository with your local training and final test results."
        )
    return path


def _percent(values):
    vals = np.asarray(values, dtype=float)
    if not np.isfinite(vals).all() or (vals < 0).any() or (vals > 1).any():
        raise ValueError("Accuracy and F1 columns must contain fractions between 0 and 1.")
    return vals * 100


def export_figures(
    runs_root: Path, final_root: Path, data_root: Path, output_dir: Path
) -> list[Path]:
    final_root = final_root / FROZEN
    manifest = json.loads(require_file(final_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("test_images") != 22619:
        raise ValueError("Expected a completed frozen 22,619-image official test evaluation.")

    histories = {}
    for label, run_folder in RUNS.items():
        history = pd.read_csv(require_file(runs_root / run_folder / "history.csv"))
        needed = {"epoch", "train_accuracy", "val_accuracy"}
        if not needed.issubset(history.columns) or len(history) != 20:
            raise ValueError(f"Incomplete 20-epoch training history for {label}")
        if not np.array_equal(history["epoch"].to_numpy(), np.arange(1, 21)):
            raise ValueError(f"Unexpected epoch indices for {label}")
        histories[label] = history

    metrics = []
    for label, folder in TEST_MODELS:
        record = json.loads(require_file(final_root / folder / "metrics.json").read_text(encoding="utf-8"))
        if record.get("n_samples") != 22619:
            raise ValueError(f"Unexpected test-set size for {label}")
        metrics.append((label, 100 * float(record["accuracy"])))

    confusion_png = require_file(
        final_root / "ensemble_test_evaluation" / "graphs" / "confusion_row_normalized.png"
    )
    per_class = pd.read_csv(
        require_file(final_root / "ensemble_test_evaluation" / "per_class.csv")
    )
    if len(per_class) != 102 or set(per_class["class_index"]) != set(range(102)):
        raise ValueError("Expected metrics for all 102 official IP102 classes.")
    _, splits, _ = load_splits(data_root, check_files=False)
    if len(splits["train"]) != 45095 or len(splits["test"]) != 22619:
        raise ValueError("IP102 official split counts differ from the expected splits.")
    train_counts = Counter(record.label for record in splits["train"])
    per_class["training_images"] = per_class["class_index"].map(train_counts)
    if per_class["training_images"].isna().any() or (per_class["training_images"] <= 0).any():
        raise ValueError("Missing training counts for one or more IP102 classes.")
    rho = per_class["training_images"].corr(per_class["f1"], method="spearman")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / filename for filename in FILENAMES]

    # Figure 2. Exactly what is described in manuscript section V.A.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for axis, (label, history) in zip(axes, histories.items()):
        axis.plot(history["epoch"], _percent(history["train_accuracy"]),
                  linewidth=1.8, label="Training accuracy")
        axis.plot(history["epoch"], _percent(history["val_accuracy"]),
                  linewidth=1.8, label="Validation accuracy")
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.set_xlim(1, 20)
        axis.set_ylim(0, 100)
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3. Only OFFICIAL test results, without mixing validation and test.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    names, values = zip(*metrics)
    bars = ax.bar(range(len(values)), values, width=0.68, edgecolor="black", linewidth=0.5)
    for rect, value in zip(bars, values):
        ax.annotate(f"{value:.2f}%", (rect.get_x() + rect.get_width()/2, value),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Official IP102 test top-1 accuracy (%)")
    ax.set_title("Individual models and frozen four-model ensemble")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(paths[1], dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 4. Reuse, do not alter, the audited normalized 102x102 confusion matrix.
    if confusion_png.resolve() != paths[2].resolve():
        shutil.copy2(confusion_png, paths[2])

    # Figure 5. The paper reports training sample frequency vs. test F1,
    # not test support vs. recall (which is a different existing graph).
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(per_class["training_images"], _percent(per_class["f1"]), alpha=0.72)
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Training images per class (log scale)")
    ax.set_ylabel("Official test per-class F1 (%)")
    ax.set_title(f"Training class frequency vs. test F1 (Spearman rho = {rho:.3f})")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(paths[3], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=STUDY / "runs")
    parser.add_argument("--final-root", type=Path, default=STUDY / "final_test_results")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "ip102_v1.1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_figures")
    args = parser.parse_args(argv)
    files = export_figures(args.runs_root, args.final_root, args.data_root, args.output_dir)
    print("\nREADY: Figures 2-5 exported from YOUR original research data:")
    for path in files:
        print(f"  {path.resolve()}")
    print("\nFigure 1 is your methodology flowchart already inserted in the paper.")
    print("Open the figure folder in Windows PowerShell: explorer .\\paper_figures")


if __name__ == "__main__":
    main()
