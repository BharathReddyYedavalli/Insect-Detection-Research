"""Synthetic offline test of paper figure exporting (no model weights or test pixels)."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image

from experiments.ip102_hf_study.export_paper_figures import (
    FROZEN, RUNS, TEST_MODELS, FILENAMES, export_figures,
)


def test_export_figures_offline_synthetic(tmp_path):
    runs = tmp_path / "runs"
    final = tmp_path / "results"
    data = tmp_path / "ip102_v1.1"
    out = tmp_path / "paper_figures"
    data.mkdir()
    # Metadata only. No actual images are created or read.
    for split, count in (("train", 45095), ("val", 7508), ("test", 22619)):
        (data / f"{split}.txt").write_text(
            "".join(f"{split}/{i}.jpg {i % 102}\n" for i in range(count)), encoding="utf-8"
        )
    for folder in RUNS.values():
        path = runs / folder
        path.mkdir(parents=True)
        pd.DataFrame({
            "epoch": np.arange(1, 21),
            "train_accuracy": np.linspace(.2, .8, 20),
            "val_accuracy": np.linspace(.3, .77, 20),
        }).to_csv(path / "history.csv", index=False)
    frozen = final / FROZEN
    frozen.mkdir(parents=True)
    (frozen / "manifest.json").write_text(
        json.dumps({"status": "complete", "test_images": 22619}), encoding="utf-8"
    )
    for i, (_, folder) in enumerate(TEST_MODELS):
        target = frozen / folder
        target.mkdir()
        (target / "metrics.json").write_text(
            json.dumps({"n_samples": 22619, "accuracy": .75 + .01*i}), encoding="utf-8"
        )
    evalu = frozen / "ensemble_test_evaluation"
    pd.DataFrame({
        "class_index": list(range(102)),
        "f1": np.linspace(.2, .9, 102),
    }).to_csv(evalu / "per_class.csv", index=False)
    graph = evalu / "graphs" / "confusion_row_normalized.png"
    graph.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(graph)
    exported = export_figures(runs, final, data, out)
    assert [p.name for p in exported] == list(FILENAMES)
    assert all(p.exists() and p.stat().st_size > 0 for p in exported)
    assert exported[2].read_bytes() == graph.read_bytes()
    assert not (data / "images").exists()
