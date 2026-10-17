# IP102 research protocol: simple training after offline augmentation

## Objective

Classify the 102 insect pest classes in the official IP102 classification dataset using ResNet18, ResNet50, VGG16 and VGG19. First establish a simple, reproducible experimental procedure. Subsequently make **one controlled change at a time**, tracking top-1 accuracy, macro-F1 and balanced accuracy.

## Fixed data partitions

- **Training:** 45,095 original images from the existing official split. Augmented variations of training images are permitted.
- **Validation:** 7,508 original images with no augmentation, for model/epoch/hyperparameter selection.
- **Test:** 22,619 original images with no augmentation. Use only for final locked evaluation after model selection. Never train on test data or tune using its accuracy.

## Balanced preprocessing

Run `python -m shared.prepare_balanced` to create a **separate** dataset at `data/ip102_balanced`. The procedure uses small random crops, horizontal flips, rotations and color/brightness/contrast variation, saving transformed images as JPEG. It includes all originals and augments smaller classes to match the largest class's **3,444** images, creating exactly **351,288 training images**, of which 306,193 are augmented derivatives.

An optional smaller target (e.g., `--target-per-class 600 --output data/ip102_balanced_600`) reduces GPU demands; classes with more originals than the target are then downsampled. This must be clearly reported, as it is *not* the same protocol as keeping all originals.

Augmented derivatives are not independent observations. Class frequency is balanced but per-class predictive performance is not guaranteed to be balanced. A source image may have numerous correlated augmented variants; this is a limitation to acknowledge.

## Simple model pipeline

1. Load the previously saved balanced training images and the untouched validation images.
2. Resize/center-crop to the same model resolution and normalize using ImageNet channel statistics.
3. Initialize a torchvision ResNet or VGG (ImageNet-pretrained weights, unless specified otherwise) and replace its last layer with **102 logits**.
4. Train using standard **CrossEntropyLoss** and **AdamW** with a fixed learning rate for a predetermined number of epochs. No class weighting, weighted sampling, stochastic augmentation during training, learning-rate scheduling, or early stopping.
5. Save `best_model.pth` by **validation top-1 accuracy**, and `last_model.pth` for diagnostics, with a full config, seed, data fingerprints, results and visualizations.
6. Keep test results separate, produced only by explicit final evaluation.

## Initial comparison

- Preserve your three *earlier* ResNet18 baseline runs (unbalanced, weighted loss, weighted sampling) as historical experiments, without renaming them.
- Train a short balanced-pilot sanity check for 3 epochs.
- Train ResNet18 with the full prepared dataset using fixed settings. Compare against the earlier unbalanced baseline, explicitly noting differences in the number of training examples/updates.
- Train ResNet50, VGG16 and VGG19 with the same balanced dataset and appropriate memory-limited batch sizes.
- Later test small changes independently (epochs, augment intensity, learning rate, input size) instead of mixing modifications.

## Research integrity

Every result should identify data version, target images per class, whether classes were downsampled, augmentation types, seed, architecture, pretrained initialization, training epochs, batch size, learning rate, checkpoint-selection metric, total training updates, and hardware/time. Report validation metrics separately from held-out test metrics. Multiple independent seeds are needed before attributing small differences to a method. Similar benchmark image counts do not alone prove that third-party ZIP membership matches the original author's exact files.

Reference: Wu et al., *IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition*, CVPR 2019; https://github.com/xpwu95/IP102
