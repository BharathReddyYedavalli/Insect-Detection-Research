"""Evaluate a *fixed equal-weight* validation ensemble of 2+ completed runs.

The equal weights are declared before seeing the ensemble performance. Results
are exploratory validation estimates, NOT held-out test performance.

Usage (paths are exact run folders printed by train.py):
 python experiments/ip102_hf_study/ensemble.py runs/path1 runs/path2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.metrics import write_results

STUDY = Path(__file__).resolve().parent


def load_and_align(folder, reference_names=None, reference_labels=None):
    folder = Path(folder).expanduser().resolve()
    path = folder / "validation_probabilities.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Completed run is missing validation probabilities: {path}")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config.get("dataset") != "original":
        raise ValueError("Only original-split validation ensembles are supported.")
    with np.load(path, allow_pickle=False) as data:
        names = np.asarray(data["filenames"], dtype=str)
        labels = np.asarray(data["labels"], dtype=int)
        probs = np.asarray(data["probabilities"], dtype=np.float32)
    if len(set(names.tolist())) != len(names):
        raise ValueError(f"Duplicate validation filenames in {folder}")
    if probs.shape != (len(labels), int(config["classes"])):
        raise ValueError(f"Unexpected probability dimensions for {folder}")
    if not np.isfinite(probs).all():
        raise ValueError(f"Non-finite probabilities in {folder}")
    if reference_names is not None:
        positions = {name: i for i, name in enumerate(names)}
        if set(positions) != set(reference_names.tolist()):
            raise ValueError("Ensemble runs use different validation image sets.")
        ordered = [positions[name] for name in reference_names]
        probs, labels = probs[ordered], labels[ordered]
        if not np.array_equal(labels, reference_labels):
            raise ValueError("Ensemble runs disagree on labels.")
        names = reference_names
    return names, labels, probs, config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+", help="At least two completed HF-study run folders.")
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    if len(args.runs) < 2 or len(set(str(x.resolve()) for x in args.runs)) != len(args.runs):
        p.error("Provide at least two DISTINCT completed run folders.")
    names, labels, first, base_cfg = load_and_align(args.runs[0])
    probability_sum = first.astype(np.float64)
    configs = [base_cfg]
    for folder in args.runs[1:]:
        _, _, probs, config = load_and_align(folder, names, labels)
        if config["classes"] != base_cfg["classes"]:
            raise ValueError("Models were trained with different class counts.")
        if config["split_fingerprints"] != base_cfg["split_fingerprints"]:
            raise ValueError("Models do not share identical official IP102 splits.")
        probability_sum += probs
        configs.append(config)
    probabilities = (probability_sum / len(args.runs)).astype(np.float32)
    out = args.output_dir or STUDY / "ensembles" / ("equal_" + "_".join(
        Path(x).name for x in args.runs
    )[:150])
    out = out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Ensemble folder already exists: {out}")
    out.mkdir(parents=True)
    scores = write_results(out / "validation_evaluation",
                           labels, names.tolist(), probabilities, int(base_cfg["classes"]))
    (out / "manifest.json").write_text(
        json.dumps({
            "split": "validation", "method": "fixed equal-weight probability averaging",
            "note": "NOT official held-out test accuracy",
            "runs": [str(x.resolve()) for x in args.runs],
            "hf_repos": [c["repo"] for c in configs],
            "num_models": len(args.runs),
            "metrics": scores,
        }, indent=2), encoding="utf-8"
    )
    print(f"Validation equal ensemble top-1={100*scores['accuracy']:.2f}% "
          f"macro-F1={100*scores['macro_f1']:.2f}% | {out}")


if __name__ == "__main__":
    main()
