# Reproducibility notes

## Environment (paper Section III-D)

Training was run on a single NVIDIA Tesla T4 GPU (15,360 MiB, Google Colab):

| Component | Version |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| cuDNN | 91900 (deterministic mode at its PyTorch default) |
| NumPy | 2.1.3 |
| scikit-learn | 1.6.1 |
| DataLoader settings | `num_workers=0, pin_memory=False` |

`requirements.txt` pins the packages where the paper specifies a version. Loading and evaluating
released checkpoints does not require a GPU or these exact versions; retraining from scratch to
exactly reproduce every reported digit does (see "Known non-determinism" below).

## Notebook block → paper section map

The notebook (`notebooks/ECG_JOURNAL.ipynb`) is one linear pipeline. This table maps each block
to what it produces in the paper.

| Block | Contents | Paper section(s) | Output |
|---|---|---|---|
| 1 | Dependencies | III-D | — |
| 2 | Imports, global config, seed list | III-D, I-A | `configs/seeds.json` mirrors this |
| 3 | PTB-XL acquisition | III-A | `data/` |
| 4 | SCP → superclass mapping, official 10-fold split | III-A | Table I |
| 5 | `Dataset`/`DataLoader`, class-frequency weights (Eq. 1) | III-B, III-C (Mitigated CNN) | — |
| 6 | Model architectures (Baseline, Mitigated, Large, Transformer) | III-C | Table II |
| 7 | Early stopping, training loops (standard + reweighted + transformer schedule) | III-D | — |
| 8 | Subgroup definitions, inference, threshold selection | III-F, III-G | — |
| 9 | FNR/FPR/equalized-odds computation, bootstrap CIs, DeLong test, Brown-Forsythe test | III-F, III-G, III-H | — |
| 10 | Checkpoint caching (`train_or_load`) | III-D (persisted weights) | `checkpoints/` |
| 11 | Single-run training + evaluation, seed 42 | IV-A, IV-B, IV-C, IV-D | Tables III-VII |
| 12 | Multi-run stability, 10 seeds, all four architectures | IV-E | Table VIII |
| — | Grad-CAM + clinical alignment scoring | III-I, IV-F | Table XIV, Fig. 4/7 |
| 14 | Robustness to clinical artifacts (Baseline CNN only) | III-J, IV-F | Fig. 5 |
| 15 | Paper figures (internal) | IV | Figures 1-3, 6, 10, 11 |
| 16 | Georgia database acquisition | III-K | `data/georgia/` |
| 17 | SNOMED-CT → superclass crosswalk | III-K | — |
| 18 | External preprocessing, dataset, dataloader | III-K | — |
| 19 | Zero-shot external inference, overall performance | IV-G | Table IX |
| 20 | External fairness audit (FNR/FPR/equalized odds) | IV-G | — |
| 21 | Cross-domain stability (reuses Block 12's checkpoints, no retraining) | IV-G | Tables X, XI, XIII |
| 22 | External validation figures | IV-G | Figures 8, 9 |

## Known non-determinism (paper Section V-C)

A fixed seed does not guarantee a bitwise-reproducible result on this stack: GPU floating-point
reduction order and kernel selection are not guaranteed identical across runs at fixed seed unless
determinism is explicitly enforced end-to-end. The paper reports a same-seed, independently
executed replicate of the seed-42 models (identical hyperparameters, data splits, architecture
code, and nominal seed, same GPU type) that produced a smaller MI DeLong effect for Mitigated vs.
Baseline (z = +2.06, p = 0.039) than the original training pass at the nominally identical seed
(z = +3.39, p = 0.001). Both values are reported in the paper rather than one being silently
preferred; the checkpoints released here correspond to the run used throughout the paper.
Consequently, retraining from `notebooks/ECG_JOURNAL.ipynb` from scratch may reproduce the paper's
qualitative conclusions (macro-AUROC stability vs. subgroup-FNR instability) without reproducing
every reported digit exactly — this is why the checkpoints themselves, not just the training code,
are released in `checkpoints/`.

## Three-part fairness taxonomy (paper Section III-E)

All fairness results in this repository are organized under three headings, applied consistently
across the notebook's audit functions:

1. **Error-rate fairness** — FNR/FPR/equalized-odds parity across subgroups (Sections III-F, IV-C).
2. **Calibration fairness** — whether predicted probability carries the same meaning across
   subgroups; assessed qualitatively here via reliability diagrams and Brier scores (open item,
   Section VIII — AUPRC/PPV/NPV are listed as future work, not computed).
3. **Stability fairness** — how much (1) and (2) vary across independently trained instances
   differing only in seed; this study's central axis (Sections III-G–IV-E).
