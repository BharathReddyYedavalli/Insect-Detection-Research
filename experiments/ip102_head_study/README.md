# Full IP102: ResNet50 versus ConvNeXt-Tiny, classifier heads and learning rates

**This is a separate experiment folder.** Existing `models/` training scripts, checkpoints and balancing study are unchanged.

## Aim and methodological limits

Use **all 102 classes** of the official IP102 classification dataset. Compare ResNet50 (ImageNet-1K V2 pretraining) and ConvNeXt-Tiny (ImageNet-1K V1 pretraining), with identical external preprocessing, optimizer and run configurations. We alter **only each model's classifier head**, not the internal convolutional blocks:

- `linear`: pretrained backbone → 102-way linear classifier
- `mlp512`: pretrained backbone → 512-unit hidden layer, ReLU, dropout 0.2 → 102 classes
- `mlp1024_256`: pretrained backbone → 1024 then 256 hidden units, ReLU and dropout → 102 classes

These are controlled, common architectural variations, **not a new architecture or novel method by themselves**. Checkpoint selection uses accuracy on original official validation images; report macro-F1, rare-class quartile F1 and computational cost as supporting results. **Never train/tune on the test images**. Only after selecting and freezing the final method should a held-out test evaluation be implemented.

Note: ImageNet initialization differs between the two official pretrained architectures; this is a practical transfer-learning comparison, **not an isolating causal comparison of architecture only**. When studying head effects, make within-backbone comparisons. Different models may train at different speeds. All seeds, dataset splits and settings are logged in each run.

## Dataset

- Original: `data/ip102_v1.1`: 45,095 train; 7,508 val; 22,619 test.
- Optional augmented-balanced: `data/ip102_balanced`: 351,288 train; original val/test.
- Default is **original, unbalanced IP102** to avoid automatically changing the data distribution.
- ImageNet channel normalization; training-only random resized crop (scale 0.75–1), horizontal flip, rotation ±10°, gentle color jitter. Validation is deterministic resize + center crop at 224.
- Full backbone fine-tuning, AdamW, cosine learning rate decay, label smoothing 0.1, weight decay 0.01, AMP on CUDA, seed 42, batch size 16. No weighted sampler, class weights or early stopping.

## Install and test on Windows

In PowerShell:
```powershell
cd C:\Users\bhara\Insect-Detection-Research
.\.venv\Scripts\Activate.ps1
git pull
python -m pytest -q
```

## Quick start: one ResNet50 baseline

```powershell
python experiments/ip102_head_study/train.py --model resnet50 --head linear --lr 0.0001 --epochs 5 --batch-size 16
```

And one ConvNeXt-Tiny baseline with identical external settings:

```powershell
python experiments/ip102_head_study/train.py --model convnext_tiny --head linear --lr 0.0001 --epochs 5 --batch-size 16
```

First-time pretrained weights download automatically. For limited CUDA memory, try `--batch-size 8` **for both models** and record that in the comparison.

## Reproducible staged sweep: don't launch dozens of expensive jobs

A manageable sequence is:

**Phase 1 — pilot (5 epochs each)**: each backbone, linear head, two initial learning rates `3e-5` and `1e-4` (4 total runs). One new run per invocation:
```powershell
python experiments/ip102_head_study/sweep.py --preset pilot --dry-run
python experiments/ip102_head_study/sweep.py --preset pilot --max-runs 1
```

Re-running the last command skips completed runs and runs the next. It resumes interrupted runs with saved checkpoints.

**Phase 2 — classifier depth (10 epochs each)**: both backbones, linear baseline, one hidden layer, and two hidden layers at the same `lr=1e-4` (6 runs). The linear control is repeated at 10 epochs so head-depth comparisons have equal training budgets:
```powershell
python experiments/ip102_head_study/sweep.py --preset heads --max-runs 1
```

**Phase 3 — shortlisted 20-epoch studies**: after reading Phase 1/2 validation results, run a selected head and learning rate *from scratch* at 20 epochs. Example (not an assertion that this is the winner):
```powershell
python experiments/ip102_head_study/train.py --model convnext_tiny --head mlp512 --lr 0.00003 --epochs 20
```
We intentionally do **not** simply resume a 5-epoch cosine-scheduler experiment to 20 epochs. Its learning-rate history would differ from a 20-epoch schedule. All experiments log their initial LR, epoch-by-epoch accuracy, F1 and run duration.

To compare results after finishing any runs:
```powershell
python experiments/ip102_head_study/summarize.py
```
This writes `summary.csv` and `class_frequency_quartiles.csv` within this folder.

## Saving progress and recovering after a restart

Every epoch writes `history.csv`, `last_model.pth`, `best_model.pth`, `config.json`, and graphs. Finished runs also save `validation_metrics.json` and per-class confusion matrices/CSV reports. `--resume` restores model, optimizer, scheduler, AMP scaler and RNG state from an **owned trusted checkpoint**; all settings, including `--epochs`, must be identical. A safe staged first epoch of a 20-epoch plan:

```powershell
python experiments/ip102_head_study/train.py --model resnet50 --head linear --lr 0.0001 --epochs 20 --stop-after-epoch 1
```
Then use the exact output path printed by the first run:
```powershell
python experiments/ip102_head_study/train.py --model resnet50 --head linear --lr 0.0001 --epochs 20 --resume "experiments/ip102_head_study/runs/<ACTUAL_RUN_FOLDER>/last_model.pth"
```
A restart during an epoch loses only that unfinished epoch. Worker-level stochastic augmentation may not replay bit-for-bit even when an epoch-boundary checkpoint resumes successfully.

## Research reporting / publication discipline

- Main independent variables: **backbone**, head depth/width and **initial learning rate**; epoch budget is a separate variable.
- Main outcomes: official validation top-1 accuracy, macro-F1, top-5, rare-class quartile F1, training time, parameters.
- Avoid picking the best number from many trials without reporting all other trials and selection rules.
- Lock a final method, repeat selected settings with at least three seeds (`--seed 42`, `43`, `44`) if compute allows, and report mean ± standard deviation.
- Pre-specify any final held-out test use; don't tune after viewing test results.
- A hyperparameter sweep can support a comparative study but is **not automatically a novel research contribution**. A defensible original method and ablations may be needed for a stronger publication.
