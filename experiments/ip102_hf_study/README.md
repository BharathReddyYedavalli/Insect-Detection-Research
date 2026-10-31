# IP102 pretrained Hugging Face backbone study

**Isolated from the existing 14-run ResNet50 / ConvNeXt-Tiny study.**
Do not delete old checkpoints or overwrite old results. The pretrained checkpoints
download to the local Hugging Face cache automatically when first used; **model
weights are not committed to GitHub**. The three model cards are:

| Experiment name | Hugging Face ID | Initialization |
| --- | --- | --- |
| convnext_small | https://huggingface.co/facebook/convnext-small-224 | ImageNet-1K supervised |
| swin_base | https://huggingface.co/microsoft/swin-base-patch4-window7-224-in22k | ImageNet-21K supervised |
| dinov2_base | https://huggingface.co/facebook/dinov2-base | Self-supervised DINOv2 (no pretrained 102-way classification head) |

For **all three**, the original classification head is discarded and a new
102-class head is initialized. Choices: linear, mlp512, mlp1024_256. The loaded
model's native classifier input dimension is used automatically, which matters
because DINOv2's internal feature representation differs.

## Accuracy target and dataset guardrails

The target of 77.39% in Chen et al. (2024) is **held-out IP102 test accuracy**,
while the project's current 74.07% is **validation accuracy**. They are not
directly comparable. Do NOT train or tune on the official test set.

- Official IP102 original train: **45,095 images**, validation: **7,508**,
  test: **22,619**, categories: **102**.
- This study only loads *train* and *validation* pixels; test *metadata* is
  checked solely for matching split counts / fingerprints.
- Validation is deterministic; **test evaluation is intentionally absent**.
- The pretrained models can be large; 224px is the practical first setting on
  an RTX 5070 before trying higher resolutions.
- Each training run saves its pretrained HF checkpoint reference and resolved
  revision when available, all hyperparameters, dataset-split hashes and seed.
- Model architectures have DIFFERENT pretraining datasets, so this is a
  practical transfer-learning comparison, not a controlled isolation of
  backbone architecture.
- A new MLP head **does not change hidden layers within the backbone**.

## Install once on your Windows PC

In PowerShell:

    cd C:\Users\bhara\Insect-Detection-Research
    .\.venv\Scripts\Activate.ps1
    git pull
    python -m pip install -r experiments/ip102_hf_study/requirements.txt
    python -m pytest -q

Do NOT replace your existing CUDA-enabled PyTorch wheels with CPU-only versions.
You don't need to download Hugging Face weights manually.
To **prefetch all three pretrained checkpoints into the HF cache** before any
training, optionally run:

    python experiments/ip102_hf_study/prefetch.py

This downloads each model's config, image processor and safetensors weights,
verifies their local sizes, and does not start IP102 training.

## Step 1: ConvNeXt-Small, low-memory baseline (recommended first)

Start a **20-epoch scheduled training run, safely pausing after epoch 1**:

    python experiments/ip102_hf_study/train.py --model convnext_small --head linear --epochs 20 --batch-size 4 --accum-steps 4 --backbone-lr 0.00002 --head-lr 0.0001 --stop-after-epoch 1

If the first epoch and validation work, resume it by copying its *actual*
last_model.pth path (printed by the script), using exactly the same training
arguments and replacing the stop flag with the resume flag:

    python experiments/ip102_hf_study/train.py --model convnext_small --head linear --epochs 20 --batch-size 4 --accum-steps 4 --backbone-lr 0.00002 --head-lr 0.0001 --resume "experiments/ip102_hf_study/runs/ACTUAL_FOLDER/last_model.pth"

If the GPU runs out of memory on 224px, start a **new** run with
--batch-size 2 --accum-steps 8. Changing batch/accumulation on a resume is
not supported to maintain experiment provenance.

## Step 2: Swin-Base from ImageNet-21K

    python experiments/ip102_hf_study/train.py --model swin_base --head linear --epochs 20 --batch-size 2 --accum-steps 8 --backbone-lr 0.00001 --head-lr 0.0001 --gradient-checkpointing --stop-after-epoch 1

Swin may need more VRAM than ConvNeXt-Small. For larger images test
--image-size 384 *in a separate new run*; the model forward explicitly
interpolates positional encoding. Do not begin with 384.

## Step 3: DINOv2 ViT-Base/14

Begin with an inexpensive **frozen-backbone** probe to check whether
DINOv2 features transfer well:

    python experiments/ip102_hf_study/train.py --model dinov2_base --head linear --epochs 10 --warmup-epochs 1 --freeze-backbone --batch-size 8 --accum-steps 2 --stop-after-epoch 1

Then a **separate new run** to update the last three DINOv2 transformer blocks,
rather than training the whole backbone immediately:

    python experiments/ip102_hf_study/train.py --model dinov2_base --head linear --epochs 20 --batch-size 4 --accum-steps 4 --backbone-lr 0.000005 --head-lr 0.0001 --unfreeze-last-n 3 --stop-after-epoch 1

When capacity permits, experiment with full fine-tuning and 336px separately.
DINOv2 patch size is 14; valid sizes include 224, 336 and 378. The
frozen probe and the partially-unfrozen run do **not** automatically transfer
the probe's learned classifier weights; both start with original HF
pretraining and a fresh IP102 head.

## Controlled ablations, not one giant experiment

Only vary **one setting at a time** to attribute improvements correctly:

1. Baseline each architecture at resolution 224 with basic augmentation.
2. Compare a 512-unit or 1024→256-unit MLP head at the SAME epoch budget.
3. Try stronger augmentation and, separately, Mixup/CutMix. Example extra
   options: --augmentation strong --mixup-alpha 0.2 --cutmix-alpha 1.0
4. Try mild sqrt-inverse-frequency class sampling with --sampler sqrt (this
   preserves the 45,095-example training budget via replacement).
5. Try a larger input size as a new run: ConvNeXt 384, Swin 384, DINOv2 336.
6. Use seed 42 for a pilot, then seeds 43 and 44 for the shortlisted best
   settings to estimate variation before a final choice.

Remember that one seed and one validation dataset cannot establish whether
a 0.1-percentage-point difference is repeatable. Per-class macro-F1 and
frequency quartiles are important because IP102 has a long-tailed class
distribution. Strong augmentation, CutMix and sampler balancing are optional
hypotheses; they can make fine-grained insect recognition *worse*.

## What the code saves

Each new run gets a unique local folder:

- config.json — source model ID, original/revision, settings and data hashes
- history.csv — all epoch train/validation metrics (train_accuracy is NaN
  when labels are mixed, not incorrectly reported as ordinary accuracy)
- graphs/ — validation accuracy, macro-F1 and training/validation loss
- best_model.pth — full best-checkpoint state (selected on val accuracy)
- last_model.pth — optimizer, scheduler, scaler and RNG for resuming
- validation_evaluation/ — predictions, confusion matrix and per-class metrics
- validation_probabilities.npz — per-image **102-class** probabilities for
  later equal-weight model ensembling
- validation_metrics.json — aggregate metrics for the selected checkpoint

With only two or more **completed HF study runs**, compare a fixed
probability-average ensemble (use your exact completed run paths):

    python experiments/ip102_hf_study/ensemble.py "experiments/ip102_hf_study/runs/RUN_ONE" "experiments/ip102_hf_study/runs/RUN_TWO"

This uses only the same official validation images and verifies matching
files, labels and split fingerprints. No test images are used.

Get a tabular leaderboard anytime:

    python experiments/ip102_hf_study/summarize.py

**Checkpoint safety:** Only resume a checkpoint produced by this study on your
own machine. PyTorch optimizer checkpoints are Python-pickle based. Do not
load arbitrary untrusted .pth files from the internet.

## Paper methodology

Publish *complete*, not selectively cherry-picked, experimental tables:
backbone, initialization dataset, classifier head, optimizer, effective batch
size, augmentation, resolution, learning-rate schedule, best epoch, validation
accuracy, macro-F1, top-5, training time and seed variation.

A single held-out test evaluation belongs **only after** choosing the final
method on validation data. Our next objective is >77.39% **test** top-1
accuracy, but the pretrained checkpoints and experimental changes do not
guarantee that outcome.
