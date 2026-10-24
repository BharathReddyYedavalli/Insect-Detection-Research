# Controlled ResNet18 IP102 class-balancing ablation

This experiment is deliberately **separate** from `models/ResNet18/train.py`. It does not change the four original trainer scripts, the prepared dataset, or any prior checkpoints.

## Research question

Does the offline class-balancing augmentation strategy (3,444 training images per class) improve minority-class F1 compared with the original imbalanced IP102 training images, under an approximately matched computational budget?

**Conditions:**
- `original`: official 45,095-image imbalanced training set.
- `balanced`: prepared 351,288-image training set with all 45,095 originals retained plus 306,193 offline augmentations.

Both use the same ResNet18 code, ImageNet pretrained backbone, 102-class head, AdamW, CE loss, batch size 32, learning rate 1e-4, weight decay 1e-4, seed 42, horizontal flip and ±8° online rotation. Neither uses a scheduler or class weights. Both evaluate on the exact same original 7,508-image validation set; the test set is left untouched.

## Run

In an activated Windows PowerShell virtual environment from the repository root:

```powershell
git pull
python -m pytest -q

# Sanity-check first 1,000 optimizer steps (no validation until 11,000):
python experiments/balance_ablation_resnet18.py --condition original --run-until 1000

# The runner displays the new experiment's output folder and last_model.pth.
# Resume it, keeping the default settings:
python experiments/balance_ablation_resnet18.py --condition original --resume "experiments/runs/<exact original run folder>/last_model.pth"

# Run the balanced counterpart as a separate independent experiment:
python experiments/balance_ablation_resnet18.py --condition balanced
```

A full run executes **55,000 optimizer updates** (1,760,000 training-image presentations) in each condition, validating at steps 11,000, 22,000, 33,000, 44,000 and 55,000. Both drop the last incomplete batch in each traversal for constant batch size. Periodic `last_model.pth` checkpoints are saved every 1,000 steps, and the best model is selected by validation accuracy at matched evaluation checkpoints. `--run-until` stops safely at any specified step, even before validation, and `--resume` continues the same output folder. Use actual printed folder paths, not the angle-bracket placeholder.

Each completed run produces `history.csv`, `config.json`, `best_model.pth`, `last_model.pth`, `validation_metrics.json` and `validation_evaluation/per_class.csv` alongside other metrics and plots. Report quartile F1/recall by **original** class frequencies, plus overall accuracy and macro-F1.

## Limitations and integrity

- This compares the **offline balancing approach as a whole** with original training; it does not separately isolate reweighting, duplicated exposures, and the kinds of offline augmentation. Follow-up ablations are needed for those claims.
- Matching optimizer updates, image presentations, evaluation opportunities, architecture, hyperparameters, and online augmentation does not mean identical per-class exposures. That difference is the balancing treatment.
- Offline augmentations of a minority class derive from relatively few original photographs, so nominal image counts do not represent independent samples.
- After a mid-run restart the deterministic **training sample order** is recovered; stochastic online augmentation workers might not replay identically, particularly when `--workers > 0`. Preserve this caveat and repeat with additional seeds for confidence.
- Accuracy-selected checkpoints are for fair model selection; F1 should also be compared at matched checkpoints, not selectively cherry-picked.
- Do not touch test until experiments, model selection, and analysis protocol are locked.
