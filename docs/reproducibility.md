# Reproducibility notes

## Environment (paper Section III-D)

Training was run on a single NVIDIA Tesla T4 GPU (15,360 MiB, Google Colab):

| Component | Version |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| cuDNN | 91900, determinism explicitly enforced (`torch.backends.cudnn.deterministic=True`, `benchmark=False`), rather than left at PyTorch's default |
| NumPy | 2.1.3 |
| scikit-learn | 1.6.1 |
| DataLoader settings | `num_workers=0, pin_memory=False` |

`requirements.txt` pins the packages where the paper specifies a version. Loading and evaluating
released checkpoints does not require a GPU or these exact versions; retraining from scratch to
exactly reproduce every reported digit does (see "Known non-determinism" below).

## Notebook block → paper section map

The notebook (`notebooks/main.ipynb`) is one linear pipeline of 22 numbered blocks, followed by
a set of lettered appendix blocks (A–I) that extend the multi-seed and threshold-policy analyses.
This table maps each block to what it produces in the paper.

| Block | Contents | Paper section(s) | Output |
|---|---|---|---|
| 1 | Dependencies | III-D | — |
| 2 | Imports, global config, seed list | III-D, Section III | `configs/seeds.json` mirrors this |
| 3 | PTB-XL acquisition | III-A | `data/` |
| 4 | SCP → superclass mapping, official 10-fold split | III-A, III-B | Table I |
| 5 | `Dataset`/`DataLoader`, class-frequency weights (Eq. 1) | III-B, III-D (Mitigated CNN) | — |
| 6 | Model architectures (Baseline, Mitigated, Large, Transformer) | III-C | Table II |
| 7 | Early stopping, training loops (standard + reweighted + Transformer schedule) | III-D | — |
| 8 | Subgroup definitions, inference | III-E | — |
| 9 | FNR/FPR/equalized-odds computation, bootstrap CIs, DeLong test, Brown-Forsythe test | III-E, III-G | — |
| 10 | Checkpoint & prediction caching (`train_or_load`) | III-D (persisted weights) | `checkpoints/` |
| 11 | Single-run training + evaluation, seed 42 | IV-A, IV-B, IV-C | Tables III, IV, V; DeLong results |
| 12 | Multi-run stability, 10 seeds, all four architectures, fixed vs. validation-F1 threshold | III-F, IV-D, IV-E | Table VI, Table XIII, Fig. 1, Appendix A (Table XIV) |
| — | Supplementary, disabled by default (`RUN_OPTIONAL_DIAGNOSTICS`): Grad-CAM clinical-alignment scoring | not reported in the paper | — |
| 14 | Supplementary, disabled by default: robustness to clinical signal artifacts (Baseline CNN only) | not reported in the paper | — |
| 15 | Paper figures (internal) | IV-E | `Figure3_Stability` = paper Fig. 1; other files are supplementary, not embedded in the paper |
| 16 | Georgia database acquisition | III-H | — |
| 17 | SNOMED-CT → superclass crosswalk (main + conservative) | III-H, VII | — |
| 18 | External preprocessing, dataset, dataloader; crosswalk-sensitivity check | III-H, VII | — |
| 19 | Zero-shot external inference, overall performance | IV-J | Table X |
| 20 | External fairness audit (FNR/FPR/equalized odds) | IV-J | Table XI, Appendix B (Table XV) |
| 21 | Cross-domain stability (reuses Block 12's checkpoints, no retraining) | IV-J | Table XII |
| 22 | External validation figures (supplementary, not embedded in the paper) | IV-J | `Figure8_External_STTC_FNR`, `Figure9_CrossDomain_Stability` |
| A | Patient count per split | III-A | Table I (patient counts) |
| B | Per-seed, per-subgroup MI FNR for all four architectures; disparity-based CV | IV-E | Disparity CV figures (23.13%/28.40%/14.52%/27.58%) |
| C | Subgroup-ranking instability across seeds | IV-E, VII | "Young Female worst in all ten seeds" finding |
| D | Age-cutoff sensitivity (55/60/65/70) | III-E (post-hoc check) | — |
| E | AUPRC / PPV / NPV per subgroup, MI class, across ten seeds | Appendix C | Table XVI |
| F | Matched same-class, same-subgroup AUROC vs. FNR (Older Female MI) | IV-E | Matched-comparison ratios, checkpoint-bootstrap CIs |
| G | Paired seed-level and patient-cluster bootstrap CIs on the CV and on paired FNR/disparity differences | IV-F | Table VII, patient-cluster results in IV-F |
| H | Matched-sensitivity threshold policy (80% MI-sensitivity target) | IV-G | Table VIII, third row of Table IX |
| I | Does validation-AUROC checkpoint selection give a misleading fairness conclusion? | IV-I | — |

## Known non-determinism (paper Sections III-D, VII)

A fixed seed does not guarantee a bitwise-reproducible result on this stack: GPU floating-point
reduction order and kernel selection are not guaranteed identical across runs at a fixed seed
unless determinism is explicitly enforced end-to-end, and even then are not guaranteed identical
across different hardware or library versions. The paper is explicit that "seed" in this study
identifies an observably distinct training run under a fixed, documented software and hardware
environment, not a guarantee of exact bitwise reproducibility elsewhere. Consistent with this, no
model in this study was trained more than once: all forty checkpoints (four architectures × ten
seeds) come from a single training pass each, and the seed-42 checkpoint from that same set is
reused — not retrained — as the primary-seed model for the single-run analyses of Section IV-A.
A same-seed, independently executed replicate is proposed as future work (Section VII) and has
not been run for this version of the study; the checkpoints released here correspond to the one
training pass used throughout the paper.

## What "fairness variability" means here (paper Section I, III-G)

The paper defines **fairness variability** as the sensitivity of a subgroup-stratified fairness
metric — subgroup FNR, FPR, or the equalized-odds gap — to the stochastic elements of
independently executed training runs, when training data, architecture, hyperparameters, and
evaluation protocol are otherwise held fixed. This is distinct from, and reported alongside:

1. **Discrimination stability** — how much macro-AUROC (threshold-free) varies across the same
   ten seeds; found to be very small (CV 0.17–0.41%) for every architecture.
2. **Error-rate fairness** — FNR/FPR/equalized-odds parity across the four sex×age subgroups at a
   given seed and threshold (Sections III-E, IV-B, IV-C).
3. **Fairness variability itself** — how much (2) varies across seeds (Section IV-E), across
   threshold policies (Sections III-F, IV-D, IV-G), and across institutions (Section IV-J).

Three distinct resampling procedures are used, and should not be conflated (Section III-G):
record-level bootstrap (1,000 iterations, for the confidence intervals in Tables III–IV), paired
seed-level bootstrap (2,000 iterations, resampling which of the ten seeds were observed, for
training-run variability in Section IV-F and Table VII), and patient-cluster bootstrap (resampling
over patient IDs, for the primary-seed checkpoint comparison in Section IV-F). The paper reports
overall Brier score as an aggregate performance measure throughout, but does not empirically
assess calibration fairness — that remains an open item (Section VII).
