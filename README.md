# IP102 Insect Pest Classification Research

Research-ready PyTorch training/evaluation pipelines for **ResNet18, ResNet50, VGG16 and VGG19** on the official **IP102 102-class classification dataset**.

> **Status:** Codebase provided for local training. No model has been trained and no accuracy improvement is claimed.

## Windows / RTX 5070 quick start

Prerequisites: Python 3.11, recent NVIDIA graphics driver, Git, enough storage for IP102 and checkpoints.

```powershell
git clone https://github.com/BharathReddyYedavalli/Insect-Detection-Research.git
cd Insect-Detection-Research
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

If your installed NVIDIA driver cannot support this build, update it and consult [PyTorch's official selector](https://pytorch.org/get-started/locally/) for a compatible Windows CUDA install. Installing `requirements.txt` alone does **not** install the GPU-enabled PyTorch wheels.

### Dataset setup

Download **Classification** from [IP102 official dataset](https://github.com/xpwu95/IP102) ([Google Drive](https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing)). Extract the **classification** archive into `data/ip102_v1.1/`, ensuring the folder contains:
```text
data/ip102_v1.1/
    images/       # JPG files
    train.txt     # official training split
    val.txt       # official validation split
    test.txt      # official test split
```
`train.txt` entries are assumed to contain an image filename and an integer class label separated by whitespace/comma, consistent with the IP102 classification archive. The dataset checker supports 0-based and 1-based labels *if consistent*. If your download differs, run the checker and inspect the first few rows; never generate random new splits.

```powershell
python -m shared.audit --data-root data/ip102_v1.1
python models/ResNet18/train.py --epochs 5
python models/ResNet50/train.py --epochs 40 --batch-size 32 --balance weighted_sampler
python models/VGG16/train.py --epochs 35 --batch-size 8
python models/VGG19/train.py --epochs 35 --batch-size 8
```

Each model gets a separate `models/<Name>/runs/<timestamp>_<shortid>/` with:
- `parameters.txt`, `config.yaml`, `environment.json`, `split_summary.json` and `splits_sha256.json`
- `best_model.pth` and `last_model.pth` (full state; checkpoints may be large)
- `history.csv`, `best_validation_predictions.csv`, `validation_metrics.json`, `per_class_validation.csv`
- `graphs/` with learning curves, confusion heatmaps, normalized heatmap, per-class recall/F1, support charts, reliability plot, confidence histogram, top-confusion pairs and support-vs-recall scatter

**Do not evaluate on the test split during model development.** After selecting configuration/epochs based only on validation data, evaluate the locked best checkpoint on test **once**:
```powershell
python -m shared.evaluate --checkpoint "models/ResNet18/runs/YOUR_RUN/best_model.pth" --split test
```
Test results go under the same run's `test_evaluation/` directory. To compare different runs:
```powershell
python -m shared.compare --runs models/ResNet18/runs models/ResNet50/runs models/VGG16/runs models/VGG19/runs
```

### Design and scientific safeguards

- **Fixed official splits,** audited for label validity, missing images, duplicates across splits and class counts. Never reshuffle the test set.
- Train balancing choices: `none` (natural distribution), `weighted_sampler` (approximately balanced in expectation), `equal_epoch` (same number of sampled items per class per epoch, with replacement as needed), `weighted_loss`. Compare methods in ablations rather than assuming balancing guarantees unbiased predictions.
- Early stopping / best-checkpoint selection is based on a **validation** metric, never test.
- Explicit seed, data/metadata fingerprints, dependency version info, logged LR and GPU memory, best/last checkpoints, resume support.
- Pretrained weights download on the *first run* (internet required) unless `--no-pretrained` is set.
- Accuracy must be reported alongside macro-F1 and balanced accuracy on the natural IP102 test distribution. No results are fabricated.

**Class-balancing caveat:** Giving every class equal weight in training is not equivalent to achieving equal predictive accuracy. Balanced sampling may hurt overall real-distribution accuracy; compare fairly against unbalanced baselines.

### Research plan
Run 3+ seeds for each architecture at the same resolution/splits, first natural sampling, then class-weighted loss and balanced sampling. Fix the protocol *before* opening the test results. Compare training times and confidence/calibration in addition to accuracy. Distinguish published IP102 classification benchmarks from 9-class/24-class results using other datasets.

### Official citation and license
IP102 is released for academic usage. For other uses, contact the dataset authors. Cite: Wu et al., **IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition**, CVPR 2019. This repository is an independent experimental implementation; it is not affiliated with the IP102 authors.


### Useful commands and troubleshooting

```powershell
# inspect training options
python models/ResNet18/train.py --help

# compare baseline vs exact balanced sampling
python models/ResNet18/train.py --epochs 40 --balance none --seed 42
python models/ResNet18/train.py --epochs 40 --balance equal_epoch --equal-k 100 --seed 42

# resume an interrupted training run (you may extend the horizon with --epochs)
python models/ResNet18/train.py --resume "models/ResNet18/runs/YOUR_RUN/last_model.pth" --epochs 50

# run small synthetic-data CPU tests before downloading IP102
python -m pytest -q
```

If VGG reports CUDA out-of-memory, retry in a **new** run with `--batch-size 4` or `--batch-size 2`; do not assume that a particular batch size is guaranteed on every 5070 configuration. The sampler's exact balance applies to **draws per epoch**, not the native distribution or eventual model performance. Model outputs are stored locally and deliberately ignored by Git.
