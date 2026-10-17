# Research protocol: IP102 102-class image classification

## Goal and hypothesis

Determine whether modern fine-tuning of ResNet18, ResNet50, VGG16 and VGG19 improves classification performance on the **official 102-class IP102 benchmark**. Our primary outcome is **top-1 accuracy on the official test split**; macro-F1 and balanced accuracy are complementary fairness/imbalance indicators. No comparison with 9-class or 24-class results from other datasets should be made as if directly equivalent.

## Strict separation of data

- Train: fit parameters and class weighting/balanced sampling.
- Validation: choose architecture, augmentation, learning rate, hyperparameters, epoch and checkpoint.
- Test: held out; run once after the experiment protocol is frozen.
- Verify filename overlap and optionally hash image files to catch byte-identical cross-split duplicates. Visually similar images and duplicate crops still need manual review.
- Do not balance validation or test: preserve natural prevalence for benchmark comparison.

## Controlled experimental matrix (suggested)

1. Baseline: all four architectures, ImageNet initialization, `balance=none`.
2. Weighted cross entropy with all training images.
3. Weighted random sampling with replacement, equal expected class probability.
4. Exact-K-per-class sampling per epoch, carefully reporting K and training images/updates.

For each, run at least **three independent seeds** (e.g. 42, 123, 456). Compare **per-step** training budgets as well as epochs: exact-K sampling can make epochs much shorter or longer than the natural data distribution. Treat these as distinct sampling conditions, not identical computational budgets.

Use the same test transform, input size (224x224 for baseline), checkpoint selection criterion and identical official splits for all models. GPU memory availability may force different batch sizes; disclose them. Hyperparameter tuning should be restricted to validation.

## Reporting

Primary: test top-1 accuracy, macro-F1, balanced accuracy. Secondary: top-5, class precision/recall/F1 and confusion matrix, ECE calibration, inference runtime if separately measured, training runtime and GPU peak memory. Include class-count-versus-recall plots and confidence histograms. Use grouped mean and standard deviation across independent seeds.

A model being trained with equal class frequency does **not** mathematically eliminate bias or ensure equal accuracy. In particular, oversampling can overfit minority-class images, while weighting can lower majority-class accuracy. Discuss both improvements and regressions.

## Practical limits

- IP102 author description: >75,000 images across 102 categories with a long-tailed distribution.
- ImageNet pretrained model weights require a first-time download; offline training should use pre-cached weights or `--no-pretrained` and must disclose initialization.
- CUDA 12.8 PyTorch wheel is recommended for Windows RTX 5070. Update the NVIDIA driver as needed. CUDA toolkit itself is not typically required for pip wheels.
- Dataset is for academic usage; check licensing before redistribution or commercial use. Do not commit dataset files or model weights to this repository.
- Training outputs should include local `.pth` files, train/validation history, graphs, manifest fingerprints and dependency versions.

## Publication integrity

A working pipeline does **not** guarantee a target accuracy, paper acceptance, or improved results. Report complete results and code with seeds; avoid tuning on the test set or omitting negative results.

## References

- Wu et al., IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition, CVPR 2019. https://github.com/xpwu95/IP102
- Torchvision model library. https://pytorch.org/vision/stable/models.html
