"""Class-wise and aggregate classification statistics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def summarize(y_true, y_pred, probs, n_classes: int = 102):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probs = np.asarray(probs, dtype=np.float32)
    if probs.shape != (len(y_true), n_classes):
        raise ValueError(f"Probabilities must have shape (N, {n_classes}), got {probs.shape}")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    correct = np.diag(cm)
    precision = np.divide(correct, pred_count, out=np.zeros(n_classes), where=pred_count > 0)
    recall = np.divide(correct, support, out=np.zeros(n_classes), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n_classes), where=(precision + recall) > 0)
    present = support > 0
    confidence = probs.max(axis=1)
    accuracy = float((y_true == y_pred).mean())
    top_k = min(5, n_classes)
    top5 = float(np.mean([y_true[i] in np.argpartition(probs[i], -top_k)[-top_k:]
                           for i in range(len(y_true))]))
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    calib = []
    for i in range(10):
        mask = (confidence >= bins[i]) & (confidence <= bins[i + 1] if i == 9 else confidence < bins[i + 1])
        count = int(mask.sum())
        acc = float((y_true[mask] == y_pred[mask]).mean()) if count else 0.
        conf = float(confidence[mask].mean()) if count else 0.
        ece += count / len(y_true) * abs(acc - conf)
        calib.append({"bin_lower": bins[i], "bin_upper": bins[i + 1],
                      "count": count, "accuracy": acc, "confidence": conf})
    result = {
        "accuracy": accuracy, "top5_accuracy": top5, "balanced_accuracy": float(recall[present].mean()),
        "macro_f1": float(f1.mean()), "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()), "weighted_f1": float(np.average(f1, weights=support)),
        "ece_10_bins": float(ece), "n_samples": len(y_true),
        "classes_present": int(present.sum()),
    }
    per_class = pd.DataFrame({
        "class_index": np.arange(n_classes), "support": support, "predicted_count": pred_count,
        "true_positive": correct, "false_positive": pred_count - correct,
        "false_negative": support - correct, "precision": precision, "recall": recall, "f1": f1
    })
    return result, per_class, cm, pd.DataFrame(calib)


def write_results(outdir: Path, labels, filenames, probs, n_classes: int = 102):
    outdir.mkdir(parents=True, exist_ok=True)
    actual = np.array(labels, dtype=int)
    probs = np.array(probs, dtype=np.float32)
    predicted = np.argmax(probs, axis=1)
    scores, per_class, cm, calib = summarize(actual, predicted, probs, n_classes)
    (outdir / "metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    per_class.to_csv(outdir / "per_class.csv", index=False)
    calib.to_csv(outdir / "calibration.csv", index=False)
    pd.DataFrame({
        "filename": filenames, "true_class": actual, "predicted_class": predicted,
        "confidence": probs.max(axis=1), "correct": actual == predicted,
        "true_class_probability": probs[np.arange(len(actual)), actual],
    }).to_csv(outdir / "predictions.csv", index=False)
    np.save(outdir / "confusion_matrix.npy", cm)
    from shared.plots import create_plots
    create_plots(outdir, scores, per_class, cm, calib, actual, predicted, probs)
    return scores
