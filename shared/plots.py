"""Generate manuscript-quality, labeled plots from a fixed set of predictions."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def learning_curves(run: Path):
    path = run / "history.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    folder = run / "graphs"
    folder.mkdir(exist_ok=True)
    for columns, ylabel, filename in [
        (["train_loss", "val_loss"], "Cross entropy", "loss_by_epoch.png"),
        (["train_accuracy", "val_accuracy"], "Accuracy", "accuracy_by_epoch.png"),
        (["val_macro_f1", "val_balanced_accuracy"], "Score", "validation_macro_scores.png"),
        (["learning_rate"], "Learning rate", "learning_rate.png"),
        (["epoch_seconds"], "Seconds", "epoch_runtime.png"),
        (["gpu_peak_memory_gb"], "GPU memory (GB)", "gpu_peak_memory.png"),
    ]:
        columns = [c for c in columns if c in df.columns and df[c].notna().any()]
        if not columns:
            continue
        plt.figure(figsize=(9, 5))
        for c in columns:
            plt.plot(df["epoch"], df[c], marker=".", label=c.replace("_", " "))
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(ylabel + " by training epoch")
        plt.legend()
        plt.grid(alpha=.2)
        _save(folder / filename)


def create_plots(outdir: Path, metrics: dict, per_class: pd.DataFrame,
                 cm: np.ndarray, calibration: pd.DataFrame,
                 actual: np.ndarray, predicted: np.ndarray, probs: np.ndarray):
    plots = outdir / "graphs"
    plots.mkdir(exist_ok=True)
    # All classes appear by numeric index; no misleading truncated class matrix.
    for normalize, filename in [(False, "confusion_counts.png"),
                                (True, "confusion_row_normalized.png")]:
        values = cm.astype(float)
        if normalize:
            denom = values.sum(axis=1, keepdims=True)
            values = np.divide(values, denom, out=np.zeros_like(values), where=denom > 0)
        plt.figure(figsize=(17, 14))
        plt.imshow(values, cmap="Blues", aspect="auto", interpolation="nearest")
        plt.colorbar(label="Recall fraction" if normalize else "Images")
        ticks = np.arange(0, len(cm), 10)
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.title("IP102 confusion matrix " + ("(row normalized)" if normalize else "(counts)"))
        plt.xlabel("Predicted class index")
        plt.ylabel("True class index")
        _save(plots / filename)

    for column, name in [("recall", "recall_by_class.png"), ("f1", "f1_by_class.png")]:
        plt.figure(figsize=(19, 5))
        plt.bar(per_class["class_index"], per_class[column], width=0.85)
        plt.xlabel("IP102 class index (0-based)")
        plt.ylabel(column.title())
        plt.title(column.title() + " for every class (102 classes)")
        plt.ylim(0, 1)
        _save(plots / name)

    plt.figure(figsize=(19, 5))
    plt.bar(per_class["class_index"], per_class["support"])
    plt.yscale("symlog", linthresh=1)
    plt.xlabel("IP102 class index (0-based)")
    plt.ylabel("Test/validation support (log-like scale)")
    plt.title("Class frequency in evaluated split")
    _save(plots / "support_by_class.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(per_class["support"], per_class["recall"], alpha=.65)
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Samples in this evaluation split (log-like scale)")
    plt.ylabel("Per-class recall")
    plt.ylim(0, 1)
    plt.title("Class support vs recall (association, not causation)")
    _save(plots / "class_support_vs_recall.png")

    pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs = sorted(pairs, reverse=True)[:20]
    if pairs:
        plt.figure(figsize=(11, 8))
        names = [f"{i} to {j}" for count, i, j in pairs][::-1]
        values = [count for count, i, j in pairs][::-1]
        plt.barh(names, values)
        plt.title("20 most frequent confusion directions (true to predicted)")
        plt.xlabel("Misclassified images")
        _save(plots / "top_confusion_pairs.png")
        pd.DataFrame(pairs, columns=["count", "true_class", "predicted_class"]).to_csv(
            outdir / "top_confusion_pairs.csv", index=False)

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    valid = calibration["count"] > 0
    plt.plot(calibration.loc[valid, "confidence"], calibration.loc[valid, "accuracy"],
             marker="o", label="Empirical confidence vs accuracy")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Fraction correct")
    plt.title(f"Reliability / calibration (ECE={metrics['ece_10_bins']:.3f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    _save(plots / "reliability.png")

    confidence = probs.max(axis=1)
    plt.figure(figsize=(9, 5))
    plt.hist(confidence[actual == predicted], bins=30, alpha=.6, label="Correct")
    plt.hist(confidence[actual != predicted], bins=30, alpha=.6, label="Incorrect")
    plt.xlabel("Top-class predicted probability")
    plt.ylabel("Number of predictions")
    plt.title("Prediction confidence: correct vs incorrect")
    plt.legend()
    _save(plots / "confidence_histogram.png")

    # Correlation of predictive probabilities across images; this is NOT label correlation.
    top = np.argsort(per_class["support"].to_numpy())[-min(25, probs.shape[1]):]
    corr = np.corrcoef(probs[:, top], rowvar=False)
    corr = np.nan_to_num(corr, nan=0., posinf=0., neginf=0.)
    pd.DataFrame(corr, index=top, columns=top).to_csv(outdir / "prediction_probability_correlation.csv")
    plt.figure(figsize=(11, 9))
    plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(top)), top, rotation=90)
    plt.yticks(range(len(top)), top)
    plt.title("Prediction-probability correlation (top 25 supported classes)")
    plt.xlabel("Predicted probability for class")
    plt.ylabel("Predicted probability for class")
    _save(plots / "prediction_probability_correlation.png")
