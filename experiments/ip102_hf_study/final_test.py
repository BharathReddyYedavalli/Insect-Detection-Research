"""ONE-TIME held-out official IP102 evaluation of the PRESELECTED four-model ensemble.

The four model identities and equal weights are frozen in this source file.
All four checkpoint files must exist locally; no training occurs.

From repository root:
  python experiments/ip102_hf_study/final_test.py           # metadata preflight ONLY
  python experiments/ip102_hf_study/final_test.py --execute # official test, ONCE

Do not change the four checkpoint names, weights, model selection, or
evaluation code based on observed test results.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForImageClassification

from shared.data import IP102Dataset, dataset_summary, load_splits
from shared.metrics import summarize, write_results
from experiments.ip102_hf_study.train import make_transforms, replace_classifier

STUDY = Path(__file__).resolve().parent
OFFICIAL_SIZES = {"train": 45095, "val": 7508, "test": 22619}
N_CLASSES = 102

# FROZEN before any access to official test image pixels. Equal 1/4 weight.
LOCKED_RUNS = (
    ("dinov2_base_linear_lr5e-06_s42_e20_20261104_172628", "dinov2_base", 42, 3),
    ("dinov2_base_linear_lr5e-06_s43_e20_20261105_074137", "dinov2_base", 43, 3),
    ("dinov2_base_linear_lr5e-06_s44_e20_20261105_173316", "dinov2_base", 44, 3),
    ("swin_base_linear_lr1e-05_s42_e20_20261101_111552", "swin_base", 42, 0),
)
ENSEMBLE_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
FINAL_NAME = "frozen_dinov2_s42_s43_s44_swin_s42_equal4_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(data_root: Path, runs_root: Path):
    original, splits, offset = load_splits(data_root, num_classes=N_CLASSES, check_files=False)
    if offset != 0:
        raise ValueError("Expected the official 0-based class IDs, as used in training.")
    sizes = {name: len(entries) for name, entries in splits.items()}
    if sizes != OFFICIAL_SIZES:
        raise ValueError(f"Expected official split counts {OFFICIAL_SIZES}, found {sizes}")
    _, fingerprints = dataset_summary(splits, offset, N_CLASSES)
    details = []
    if len(set(x[0] for x in LOCKED_RUNS)) != len(LOCKED_RUNS):
        raise ValueError("Ensemble contains duplicate run names.")
    if abs(sum(ENSEMBLE_WEIGHTS) - 1.0) > 1e-12 or len(ENSEMBLE_WEIGHTS) != len(LOCKED_RUNS):
        raise ValueError("The locked ensemble weights are not valid.")
    for (folder_name, model_name, seed, last_n), weight in zip(LOCKED_RUNS, ENSEMBLE_WEIGHTS):
        run = runs_root / folder_name
        config_path, checkpoint_path = run / "config.json", run / "best_model.pth"
        metrics_path = run / "validation_metrics.json"
        for required in (config_path, checkpoint_path, metrics_path):
            if not required.is_file():
                raise FileNotFoundError(f"Selected model is incomplete: {required}")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected = {
            "model": model_name, "seed": seed, "unfreeze_last_n": last_n,
            "head": "linear", "epochs": 20, "image_size": 224,
            "classes": 102, "dataset": "original",
            "augmentation": "basic", "sampler": "original",
        }
        for field, value in expected.items():
            if cfg.get(field) != value:
                raise ValueError(f"{folder_name}: expected {field}={value!r}; got {cfg.get(field)!r}")
        if cfg.get("split_fingerprints") != fingerprints:
            raise ValueError(f"{folder_name}: train/val/test split fingerprints differ from training.")
        if metrics.get("n_samples") != OFFICIAL_SIZES["val"] or metrics.get("split") != "validation":
            raise ValueError(f"{folder_name}: validation metrics missing or inconsistent.")
        if not 0 <= metrics.get("accuracy", -1) <= 1:
            raise ValueError(f"{folder_name}: invalid validation accuracy.")
        details.append({
            "run": folder_name, "path": str(run.resolve()),
            "model": model_name, "seed": seed, "weight": weight,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "config": cfg,
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_best_epoch": metrics["best_epoch"],
        })
    return original, splits, fingerprints, details


def create_model(cfg: dict, checkpoint: dict):
    """Rebuild the trained architecture; load ONLY the local checkpoint weights.

    Never use the pretrained task classifier, and never retrain on the test set.
    """
    revision = cfg.get("resolved_revision") or cfg.get("requested_revision", "main")
    hf_config = AutoConfig.from_pretrained(
        cfg["repo"], revision=revision, trust_remote_code=False
    )
    hf_config.num_labels = int(cfg["classes"])
    model = AutoModelForImageClassification.from_config(
        hf_config, trust_remote_code=False
    )
    features = replace_classifier(
        model, cfg["head"], int(cfg["classes"]), float(cfg["dropout"])
    )
    if features != int(cfg["features"]):
        raise ValueError("Model feature size doesn't match training checkpoint.")
    # STRICT: any weight mismatch aborts instead of silently testing wrong weights.
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def test_predict(model, loader, device, amp_mode, model_name):
    model.eval()
    names, labels, probs = [], [], []
    kwargs = {}
    if model_name == "swin_base" and loader.dataset.transform.transforms[0].size != 224:
        kwargs["interpolate_pos_encoding"] = True
    use_amp = device.type == "cuda" and amp_mode != "off"
    dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    with torch.inference_mode():
        for images, targets, indices in tqdm(loader, desc=f"Test inference {model_name}"):
            images = images.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp, dtype=dtype):
                logits = model(pixel_values=images, **kwargs).logits
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
            labels.extend(targets.tolist())
            names.extend(loader.dataset.records[int(i)].filename for i in indices)
    probabilities = np.concatenate(probs, axis=0).astype(np.float32)
    return np.asarray(names, dtype=str), np.asarray(labels, dtype=np.int32), probabilities


def run_once(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Final evaluation requires CUDA to avoid unexpected CPU performance problems.")
    original, splits, fingerprints, details = preflight(args.original_root, args.runs_root)
    folder = args.output_root / FINAL_NAME
    if folder.exists():
        raise FileExistsError(
            f"Final test output already exists: {folder}\n"
            "To preserve the held-out test protocol, do not rerun or tune based on prior results."
        )
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "evaluation": "official IP102 held-out test",
        "precommitted_ensemble": "equal 0.25 probability average of four frozen checkpoints",
        "important": "Do NOT tune hyperparameters, checkpoint selection, or ensemble membership using test scores.",
        "selection_split": "validation", "test_images": OFFICIAL_SIZES["test"],
        "split_fingerprints": fingerprints,
        "run_order": [d["run"] for d in details],
        "weights": list(ENSEMBLE_WEIGHTS),
        "status": "in_progress",
        "models": [],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    expected_names = np.asarray([r.filename for r in splits["test"]], dtype=str)
    expected_labels = np.asarray([r.label for r in splits["test"]], dtype=np.int32)
    combined = np.zeros((len(expected_names), N_CLASSES), dtype=np.float64)
    print(f"LOCKED evaluation of {len(expected_names):,} OFFICIAL test images, "
          f"{len(details)} frozen checkpoints, equal 0.25 weights", flush=True)
    for index, detail in enumerate(details):
        cfg = detail["config"]
        print(f"\nModel {index + 1}/4: {detail['run']}", flush=True)
        path = Path(detail["checkpoint"])
        checksum = file_sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # The model checkpoint must be the selected best-accuracy epoch.
        checkpoint_cfg = json.loads(json.dumps(checkpoint["config"]))
        if checkpoint_cfg != cfg:
            raise ValueError(f"Checkpoint configuration mismatch: {path}")
        if checkpoint["epoch"] != detail["validation_best_epoch"] or (
            abs(float(checkpoint["best_acc"]) - float(detail["validation_accuracy"])) > 1e-8
        ):
            raise ValueError(f"Best checkpoint metadata disagrees with validation results: {path}")
        model = create_model(cfg, checkpoint).to(device)
        del checkpoint
        dataset = IP102Dataset(original, splits["test"], make_transforms(
            int(cfg["image_size"]), False, cfg["augmentation"],
            cfg["image_mean"], cfg["image_std"]
        ))
        batch_size = args.swin_batch_size if cfg["model"] == "swin_base" else args.dino_batch_size
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
        names, labels, probs = test_predict(
            model, loader, device, cfg["amp"], cfg["model"]
        )
        if not np.array_equal(names, expected_names) or not np.array_equal(labels, expected_labels):
            raise ValueError(f"Official test example order or labels don't match: {detail['run']}")
        if probs.shape != combined.shape or not np.isfinite(probs).all():
            raise ValueError(f"Invalid probability matrix: {detail['run']}")
        combined += ENSEMBLE_WEIGHTS[index] * probs
        model_dir = folder / f"model_{index + 1}_{cfg['model']}_s{cfg['seed']}"
        model_scores = write_results(
            model_dir, labels.tolist(), names.tolist(), probs, N_CLASSES
        )
        detail_record = {
            "run": detail["run"], "hf_repo": cfg["repo"],
            "revision": cfg.get("resolved_revision") or cfg.get("requested_revision"),
            "seed": cfg["seed"], "best_epoch": checkpoint_cfg["epochs"],
            "weight": ENSEMBLE_WEIGHTS[index], "checkpoint_sha256": checksum,
            "test_scores": model_scores,
        }
        # Report actual selected checkpoint epoch, not overall training horizon.
        detail_record["best_epoch"] = detail["validation_best_epoch"]
        manifest["models"].append(detail_record)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Frozen model {index + 1}: test accuracy={model_scores['accuracy']*100:.2f}%, "
              f"macro-F1={model_scores['macro_f1']*100:.2f}%", flush=True)
        del model, dataset, loader, probs
        gc.collect()
        torch.cuda.empty_cache()
    final_probs = combined.astype(np.float32)
    final_scores = write_results(
        folder / "ensemble_test_evaluation",
        expected_labels.tolist(), expected_names.tolist(), final_probs, N_CLASSES
    )
    np.savez_compressed(
        folder / "ensemble_test_probabilities.npz",
        filenames=expected_names, labels=expected_labels, probabilities=final_probs
    )
    manifest["status"] = "complete"
    manifest["ensemble_test_scores"] = final_scores
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nFINAL HELD-OUT OFFICIAL IP102 TEST: top-1={100*final_scores['accuracy']:.2f}% "
          f"macro-F1={100*final_scores['macro_f1']:.2f}% "
          f"top-5={100*final_scores['top5_accuracy']:.2f}%", flush=True)
    print(f"Frozen result directory: {folder}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true",
                   help="Run the one-time official test inference (without this, metadata preflight only).")
    p.add_argument("--original-root", type=Path, default=ROOT / "data" / "ip102_v1.1")
    p.add_argument("--runs-root", type=Path, default=STUDY / "runs")
    p.add_argument("--output-root", type=Path, default=STUDY / "final_test_results")
    p.add_argument("--dino-batch-size", type=int, default=4)
    p.add_argument("--swin-batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    if min(args.dino_batch_size, args.swin_batch_size) < 1 or args.workers < 0:
        p.error("Batch sizes must be positive and workers must be nonnegative.")
    if not args.execute:
        _, splits, _, details = preflight(args.original_root, args.runs_root)
        print(f"READY: {len(details)} FROZEN checkpoints, "
              f"{len(splits['test']):,} official test entries.")
        for item in details:
            print(f"  {item['model']} seed={item['seed']} "
                  f"validation={item['validation_accuracy']*100:.2f}% "
                  f"weight={item['weight']:.2f}: {item['checkpoint']}")
        print("No test images have been loaded or scored.")
        print("To perform the final held-out evaluation ONCE: "
              "python experiments/ip102_hf_study/final_test.py --execute")
        return
    run_once(args)


if __name__ == "__main__":
    main()
