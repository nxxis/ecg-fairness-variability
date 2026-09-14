# Discrimination Stability and Fairness Variability in Multi-Label ECG Classification

Code and artifacts for **"Discrimination Stability and Fairness Variability in Multi-Label ECG
Classification: A Multi-Seed, External-Cohort Analysis"** (submitted to *IEEE Journal of
Biomedical and Health Informatics*).

Sudip Sharma, Bto Bhatta, Kenzi Merchant, Stephanie Egwuchukwu, Kayvon Sheared, Blessing Ojeme,
and Md Mahmudur Rahman — Morgan State University / Xavier University of Louisiana, CEAMLS.

## What this study does

Single-model ECG fairness audits report subgroup error rates from one trained model, treating
that estimate as a stable property of the pipeline rather than one draw from a distribution
induced by stochastic training. This project quantifies that variation — **fairness
variability**, defined as the sensitivity of a subgroup-stratified fairness metric to the
stochastic elements of independently executed training runs — for automated 12-lead ECG
interpretation:

- **Four architectures** — a Baseline 1D-CNN, an identical network with class-frequency-reweighted
  loss ("Mitigated"), a larger-capacity 1D-CNN, and a Hybrid CNN-Transformer — each trained from
  **ten independent random seeds** on **PTB-XL** (n = 21,799 recordings, 18,869 patients).
- **Fairness audited two ways**: FNR parity and the stronger equalized-odds criterion (the larger
  of the FNR and FPR disparities), across four intersectional sex × age subgroups, for all five
  PTB-XL diagnostic superclasses (NORM, MI, STTC, CD, HYP).
- **Three threshold policies**: a fixed τ = 0.5 cutoff, a per-seed validation F1-maximizing
  threshold, and a per-seed validation threshold matched to a fixed 80% MI sensitivity target —
  testing whether cross-seed fairness instability, and the apparent benefit of reweighting, is an
  artifact of any one operating point.
- **Statistical testing**: DeLong tests for per-class AUROC between architectures (Bonferroni-
  corrected), an exploratory Brown-Forsythe variance test comparing the relative dispersion of
  macro-AUROC to subgroup FNR, and three distinct bootstrap procedures — record-level, paired
  seed-level, and patient-cluster-level — to separate sampling uncertainty from training-run
  variability.
- **Zero-shot external validation** on the **Georgia 12-Lead ECG Challenge Database**
  (n = 8,144, after excluding recordings outside the five-superclass taxonomy and two with
  non-finite signal values), with no retraining or adaptation, testing whether architecture
  ranking and fairness-instability magnitude generalize across institutions.
- **A validation-based model-selection simulation**: for each architecture, picking the seed with
  the highest validation macro-AUROC (as a practitioner ordinarily would) and checking where that
  checkpoint's test-set fairness falls within the full ten-seed distribution.
- **Supplementary, disabled-by-default diagnostics** in the notebook (1D Grad-CAM clinical-
  alignment scoring, robustness to clinical signal artifacts) that evaluate only a single Baseline
  checkpoint and are **not** part of the paper's central analysis or its reported tables.

**Central findings**: macro-AUROC coefficient of variation (CV) across seeds is 0.17–0.41% for
every architecture, while fixed-threshold FNR for MI in an older-female subgroup has CV of
11.4–15.1% — 11–52× larger under a matched same-subgroup AUROC comparison, with bootstrap
confidence intervals excluding a ratio of 1 for every architecture. Re-selecting the threshold
does not resolve this uniformly (it *increases* FNR CV for two architectures and *decreases* it
for the other two). A joint FNR/FPR audit surfaces a 23.7 pp false-positive-rate disparity that
FNR-only auditing misses. Class-frequency reweighting reduces older-female MI FNR by 46.4% at a
fixed threshold, but that advantage is not statistically established once thresholds are
independently re-selected per model or matched to a common sensitivity target. External
validation on the Georgia database preserves architecture ranking at the extremes while the two
middle-ranked models swap places. See the paper for the complete tables (I–XVI) and discussion.

## Repository structure

Everything here was trained and run in **Google Colab**, not locally — the folder layout
deliberately mirrors the notebook's own Google Drive project folder
(`/content/drive/MyDrive/ecg_fairness_project`), so you can drop this repo straight into Drive and
the notebook's hardcoded paths just work. See "Running this in Google Colab" below.

```
ecg-fairness-instability/
├── notebooks/main.ipynb            Main pipeline notebook (22 numbered blocks + lettered
│                                     appendix blocks A-I, Colab-native)
├── configs/seeds.json              The 10 fixed seeds
├── data/README.md                  Dataset acquisition notes (no raw data committed)
├── checkpoints/                    All 40 trained checkpoints plus cached predictions,
│                                     thresholds, and training history (4 architectures x
│                                     10 seeds, ~78 MB)
├── figures/                        Generated figures (only one appears in the paper body;
│                                     the rest are supplementary notebook output)
├── docs/reproducibility.md         Notebook-block-to-paper-section map, environment record
├── requirements.txt                Only needed for loading checkpoints outside Colab
├── CITATION.cff
└── LICENSE
```

`checkpoints/` and `figures/` are exact copies of what the notebook writes to
`{PROJECT_DIR}/checkpoints` and `{PROJECT_DIR}/figures` respectively — see
[`docs/reproducibility.md`](docs/reproducibility.md) for the full block-by-block map, and
[`checkpoints/README.md`](checkpoints/README.md) / [`figures/README.md`](figures/README.md) for
what's in each.

## Running this in Google Colab (recommended)

This is how the study was actually built and run, and by far the easiest way to reproduce it —
no local Python environment, no GPU of your own required.

1. **Get the notebook into Colab.** Either push this repo to GitHub and open
   `notebooks/main.ipynb` via Colab's *File → Open notebook → GitHub* tab, or upload the file
   directly via *File → Upload notebook*.

2. **Put the pretrained artifacts in your Google Drive** so the notebook loads them instead of
   retraining all 40 models from scratch. In your Drive, create:

   ```
   My Drive/ecg_fairness_project/checkpoints/   ← copy the contents of this repo's checkpoints/ here
   ```

   (Optional but saves a multi-GB download: also copy the PTB-XL zip into
   `My Drive/ecg_fairness_project/data/` once you've downloaded it once — Block 3 will find and
   reuse it. Not required; if it's missing, Block 3 downloads it automatically.)

3. **Set the runtime.** *Runtime → Change runtime type*. A GPU (T4 is what the paper used) is only
   needed if you intend to retrain something; if every checkpoint above is already in place, a
   CPU-only runtime is enough to reload everything and regenerate all tables/figures.

4. **Run all cells top to bottom** (*Runtime → Run all*). Block 2 mounts your Drive
   (`drive.mount('/content/drive')`) — approve the auth popup. From there:
   - Block 3 downloads/extracts PTB-XL (skipped if already cached, see `data/README.md`).
   - Block 10 (`train_or_load`, used by Blocks 11–12) checks `{PROJECT_DIR}/checkpoints` for
     each of the 40 `{model}_seed{seed}.pth` files before training anything — with Step 2 done,
     every one is found and loaded, so the whole notebook re-runs in minutes instead of retraining.
   - Blocks 16–21 download the Georgia database fresh each session (not cached to Drive, see
     `data/README.md`) and run zero-shot external validation on the loaded checkpoints.
   - Appendix Blocks A–I (near the end of the notebook) re-run the multi-run, threshold-policy,
     and validation-selection analyses reported in Sections IV-E through IV-I and Appendices A–C,
     reusing cached predictions rather than retraining.
   - Every paper table is printed directly to cell output; the one figure that appears in the
     paper body, plus several supplementary plots, are saved to `{PROJECT_DIR}/figures`
     (Blocks 15 and 22).

5. **Retrieve results.** Copy `{PROJECT_DIR}/figures` back out of Drive if you want the
   regenerated figures versioned in this repo, and copy any table output you need from the cell
   logs — see [`docs/reproducibility.md`](docs/reproducibility.md) for the full block-to-table
   mapping.

## Optional: loading a checkpoint outside Colab

Not the primary workflow, but useful for a quick local sanity check or downstream analysis
without opening Colab:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then, since the architecture classes live in Block 6 of the notebook rather than a standalone
module, see [`checkpoints/README.md`](checkpoints/README.md) for the loading snippet.

## Reproducibility

- **Seeds**: `configs/seeds.json` holds the ten seeds, fixed before any model was trained
  (`42, 123, 2026, 7, 99, 256, 512, 777, 2024, 31415`), to preclude post-hoc seed selection.
- **Checkpoints**: all 40 trained checkpoints (4 architectures × 10 seeds) are provided in
  `checkpoints/`, along with their cached test/validation/external predictions and selected
  thresholds, so every number in the paper can be recomputed without retraining or re-running
  inference.
- **Known non-determinism**: as the paper notes (Sections III-D, VII), a fixed seed identifies an
  observably distinct training run under a documented software/hardware environment, not a
  guarantee of exact bitwise reproducibility elsewhere — GPU floating-point reduction order and
  kernel selection are not guaranteed identical across hardware or library versions even with
  `torch.backends.cudnn.deterministic=True` enforced. No same-seed retraining pass was executed
  in this study (a same-seed replicate is proposed as future work); no checkpoint here was trained
  more than once. See `docs/reproducibility.md` for the full environment record.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this code, please cite the paper once published;
in the meantime, cite this repository directly.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). PTB-XL and the Georgia
12-Lead ECG Challenge Database remain governed by their own PhysioNet licenses.
