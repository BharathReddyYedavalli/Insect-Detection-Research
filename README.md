# IP102 Pest Classification — Simple Balanced Training

A PyTorch image-classification research project for the **official IP102 dataset**, using **102 output classes** and four architectures: ResNet18, ResNet50, VGG16 and VGG19.

## Training workflow (simple and reproducible)

1. **Keep the original** 45,095 training, 7,508 validation and 22,619 test images unchanged at `data/ip102_v1.1/`.
2. **Prepare a separate balanced dataset offline.** Copy/hard-link unmodified originals, augment minority-class *training* images with ordinary rotations, horizontal flips, crops and slight color variation, and save the transformed JPG images. Every class gets exactly the same number of training images. Evaluation images are copied/hard-linked byte-for-byte without augmentation.
3. **Train normally:** ImageNet-initialized torchvision model, 102-way classifier, ordinary CrossEntropyLoss, AdamW optimizer, fixed learning rate, shuffle the *already-balanced* training set, validate every epoch, run for all requested epochs.
4. Pick **best checkpoint by validation accuracy**; save history, `.pth` weights, per-class scores, confusion matrices, reliability plots, and comparison CSVs.
5. **After** selecting the model and settings on validation, run the locked checkpoint on the held-out test split. Never pick hyperparameters using test results.

Each model now has a **complete, readable training program** in `models/<Model>/train.py` (dataset class, model, training, validation, saving, and learning curves). The original import/audit/balancing utilities remain separate because those run only when preparing the dataset. Detailed paper-grade confusion matrices and calibration plots reuse `shared/metrics.py` and `shared/plots.py`.

There are **no class-weighted losses, weighted samplers, LR schedulers or early stopping**. Simple random flips and small rotations can be enabled independently with `--extra-augmentation` for a controlled anti-overfitting experiment; they are not enabled by default. Earlier research outputs under `models/*/runs/` stay untouched.

## Windows RTX 5070 environment

Open PowerShell inside the repo. If the existing virtual environment is already installed, just activate it:

```powershell
cd "$HOME\Insect-Detection-Research"
.\.venv\Scripts\Activate.ps1
git pull
python -m pytest -q
```

If setting up from scratch, install Python 3.10 or 3.11 and:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Dataset import (already completed on your PC)

Your Kaggle archive with `classification/{train,val,test}/{class}/{image}.jpg` was imported to `data/ip102_v1.1/` using:

```powershell
python -m shared.import_kaggle --zip "$HOME\Downloads\archive.zip" --inspect-only
python -m shared.import_kaggle --zip "$HOME\Downloads\archive.zip"
python -m shared.audit --data-root data/ip102_v1.1
```

The original split sizes match the IP102 classification benchmark. The *fixed class count alone does not prove file-level correspondence with the authors' releases*; verify if later publication comparison requires it.

## Generate equal-sized classes with saved augmented images

**Research default: preserve EVERY original training image.** The largest original class has **3,444 images**; to get equal-sized classes without discarding originals, the target must be at least 3,444 per class. This makes **351,288 training images** total (45,095 original + **306,193 saved augmented images**). Expect substantial preparation time, multi-GB extra disk space, and about **7.8× as many training examples per epoch** versus the earlier baseline:

```powershell
python -m shared.prepare_balanced
python -m shared.audit --data-root data/ip102_balanced
```

The default prepared output is:

```text
data/ip102_balanced/
    images/
        train/0/        # original and __aug_*.jpg
        ...
        train/101/
        val/            # original images only, unchanged
        test/           # original images only, unchanged
    train.txt
    val.txt
    test.txt
    augmentation_manifest.json
```

**Quick balanced pilot (smaller data):** set a target of 600 images per class for **61,200** training images, saved separately. This also *downsamples* classes originally above 600; the source remains intact. This is a separate experiment, not equivalent to retaining all original images.

```powershell
python -m shared.prepare_balanced --target-per-class 600 --output data/ip102_balanced_600
python -m shared.audit --data-root data/ip102_balanced_600
```

The metadata records how many original images were retained, augmented and excluded. **Never put augmented images into validation/test**. Rerunning the same preparation settings simply verifies and reuses the prepared dataset. To regenerate with different settings, use a different `--output` directory; do not mix prepared versions.

## Train on a prepared dataset

The default learning rate for the new readable trainers is **0.0001**, based on the ResNet18 results you observed (65.70% best val at 1e-4, versus 64.85% at 2e-4; both 12 epochs). That does **not** guarantee improvement in a future run or in other architectures.

For the full all-originals-preserved augmented dataset:

```powershell
# First: test simple additional train-only augmentation on the balanced dataset.
python models/ResNet18/train.py --epochs 12 --batch-size 32 --lr 0.0001 --extra-augmentation

# Other models, after comparing results and deciding on protocol:
python models/ResNet50/train.py --epochs 12 --batch-size 16
python models/VGG16/train.py --epochs 12 --batch-size 8
python models/VGG19/train.py --epochs 12 --batch-size 4
```

The `--extra-augmentation` option only transforms training images on each epoch using mild horizontal flips and rotations. It does not rebalance classes again or change evaluation images. Run the same command **without** `--extra-augmentation` to compare; you already have an earlier low-LR unaugmented balanced run from the previous implementation.

For the 600-per-class pilot, explicitly set the root on every model invocation:

```powershell
python models/ResNet18/train.py --data-root data/ip102_balanced_600 --epochs 3 --batch-size 32
```

Training automatically verifies that the `augmentation_manifest.json`, split fingerprints and per-class counts match; if not, it refuses to start. If you run out of GPU memory, retry with a smaller `--batch-size`. These runs are independent and can be compared only after documenting their differing sample counts and training budgets.

Each `models/<Model>/train.py` holds its own training code and CLI defaults; the earlier `config.yaml` files are preserved for older runs but are **not read** by the new trainers. Set `--epochs`, `--batch-size`, `--lr`, `--seed`, `--workers`, `--image-size` or `--extra-augmentation` in PowerShell. The script has no balancing choices at training time: it reads preprocessed images from disk.

## Outputs and evaluation

Each experiment automatically creates a separate `models/<Model>/runs/<run_id>/` containing:

- `config.yaml`, `augmentation_manifest.json`, `splits_sha256.json`
- `history.csv`, `best_model.pth`, `last_model.pth`, `validation_metrics.json`
- `graphs/` — loss, accuracy and validation macro-F1 learning curves
- `validation_evaluation/` — confusion matrices (including normalized), class recall/F1, support plots, calibrated confidence and probability-correlation matrices, prediction CSVs

The **test data stays untouched during training**. After fixing a final training protocol based on validation, evaluate the selected checkpoint once:

```powershell
python -m shared.evaluate --checkpoint "models/ResNet18/runs/YOUR_RUN/best_model.pth" --split test
```

Compare all validation runs (including your earlier baselines; their training recipes will be labeled separately):

```powershell
python -m shared.compare --runs models/ResNet18/runs models/ResNet50/runs models/VGG16/runs models/VGG19/runs
```

## Research notes

The offline balanced dataset gives *equal training image counts*, **not** 3,444 independent real photos for every class. Augmented derivatives remain correlated with their originals. Equal image counts do not guarantee equal class recall or fair predictions. Do not report validation accuracy as test accuracy, and do not claim the new method beats a paper before the same benchmark's test evaluation. Compare against the original unbalanced baseline, use multiple random seeds, and document augmentation, resolution, pretrained weights and computational budget.

Original dataset: [IP102 (Wu et al., CVPR 2019)](https://github.com/xpwu95/IP102). Follow its academic-use restrictions and cite the authors.
