# checkpoints/

All 40 trained model checkpoints (4 architectures × 10 seeds), plus their cached training
history, predictions, and selected thresholds, belong here — so every number in the paper can be
recomputed without retraining or re-running inference.

## Naming convention

The pipeline writes checkpoints as `{safe_model_name}_seed{seed}.pth`, where spaces and slashes
in the model name are normalised to underscores/hyphens:

| Model name (paper)      | `safe_model_name`         |
|--------------------------|----------------------------|
| Baseline 1D-CNN           | `Baseline_1D-CNN`           |
| Mitigated 1D-CNN         | `Mitigated_1D-CNN`         |
| Large 1D-CNN               | `Large_1D-CNN`               |
| Hybrid CNN-Transformer | `Hybrid_CNN-Transformer` |

Each `{safe_model_name}_seed{seed}` prefix has six associated files:

| Suffix | Contents |
|---|---|
| `.pth` | Model weights (best-validation-loss checkpoint) |
| `_history.json` | Per-epoch train/val loss and epoch count (used for Figure 6) |
| `_thresholds.npy` | Per-class threshold selected on the validation fold (F1-maximising grid search, Section III-F) |
| `_val_predictions.npz` | Cached raw predictions on the PTB-XL validation fold |
| `_test_predictions.npz` | Cached raw predictions on the PTB-XL held-out test fold |
| `_external_predictions.npz` | Cached raw predictions on the Georgia external cohort (zero-shot) |

Caching every split's predictions means the lettered appendix blocks (A–I) in the notebook — the
disparity-CV, matched-comparison, threshold-policy, and validation-selection analyses in Sections
IV-E through IV-I — re-run in seconds by loading these files, rather than re-running inference on
every architecture/seed pair from scratch. `checkpoints/analysis_cache_v2/` is an internal
duplicate of the `*_predictions.npz` files kept by one of the notebook's caching layers; it is not
a separate set of results.

Seeds: `42, 123, 2026, 7, 99, 256, 512, 777, 2024, 31415` (see `configs/seeds.json`). The
seed-42 checkpoint for each architecture is reused, not retrained, as both the primary-seed
single-run model (Section IV-A) and the seed-42 member of the ten-seed stability suite
(Section IV-E) — 40 checkpoints total, not 44, and no model was trained more than once across the
entire study.

Example: `checkpoints/Mitigated_1D-CNN_seed777.pth`

This folder is a direct copy of `{PROJECT_DIR}/checkpoints` from the Colab run (`CHECKPOINT_DIR`
in Block 10 of the notebook). Placing it at that same path inside your Google Drive lets
`train_or_load` (Block 10) find every checkpoint already trained and skip straight to evaluation —
see the root [`README.md`](../README.md) for the exact Colab setup steps.

## Loading a checkpoint

The architecture classes (`ECG_1D_CNN`, `ECG_Large_CNN`, `Hybrid_ECG_Transformer`) are defined in
Block 6 of `notebooks/main.ipynb`, not in a standalone `.py` module. To load a checkpoint, run
Block 6 first (in Colab, or any Python session where you've pasted that cell), then:

```python
import torch

model = ECG_1D_CNN(num_classes=5)  # or ECG_Large_CNN / Hybrid_ECG_Transformer, matching the file
model.load_state_dict(torch.load("checkpoints/Mitigated_1D-CNN_seed777.pth", map_location="cpu"))
model.eval()
```

## Storage

All 40 checkpoints, together with their cached predictions, thresholds, and training history, are
**~78 MB** (47K–479K parameters per model), so they're committed directly to this git repository —
no Git LFS needed. If a future revision of this study adds larger architectures or many more
seeds, switch this directory to [Git LFS](https://git-lfs.com/) or attach the checkpoints to a
GitHub Release / Zenodo archive instead, and update this file accordingly.
