# Predictive Stability vs. Fairness Instability in Multi-Label ECG Classification

Code and artifacts for **"Predictive Stability Versus Fairness Instability in Multi-Label ECG
Classification: A Multi-Seed, Cross-Institutional Analysis"** (submitted to *IEEE Journal of
Biomedical and Health Informatics*).

Sudip Sharma, Bto Bhatta, Kenzi Merchant, Stephanie Egwuchukwu, Kayvon Sheared, Blessing Ojeme
— Morgan State University / Xavier University of Louisiana, CEAMLS.

## What this study does

Clinical AI fairness audits typically report subgroup error rates from a single trained model,
treating that estimate as a stable property of the pipeline rather than one draw from a
distribution induced by stochastic training. This project tests that assumption for automated
12-lead ECG interpretation:

- **Four architectures** — Baseline 1D-CNN, class-frequency-reweighted "Mitigated" 1D-CNN, a
  larger-capacity 1D-CNN, and a Hybrid CNN-Transformer — each trained from **ten independent
  random seeds** on **PTB-XL** (n = 21,799).
- **Fairness audited two ways**: FNR parity (equality of opportunity) and the stronger
  equalized-odds criterion (joint FNR/FPR), across four intersectional sex × age subgroups, for
  all five PTB-XL diagnostic superclasses (NORM, MI, STTC, CD, HYP).
- **Operating-point sensitivity**: every fairness number is recomputed under a per-seed
  validation-calibrated threshold, not just the fixed τ = 0.5 cutoff, to test whether instability
  is a threshold artifact.
- **Statistical testing**: Brown-Forsythe variance tests compare the relative dispersion of
  macro-AUROC to that of subgroup FNR across the ten seeds; DeLong tests compare per-class AUROC
  between architectures.
- **Zero-shot external validation** on the **Georgia 12-Lead ECG Challenge Database**
  (n = 10,338), with no retraining, testing whether architecture ranking and fairness-instability
  magnitude generalize across institutions and populations.
- **Supporting diagnostics** (single-checkpoint, not central claims): 1D Grad-CAM clinical
  alignment scoring and robustness to baseline wander / Gaussian / powerline-interference
  artifacts.

The central finding: macro-AUROC is highly reproducible across seeds (CV 0.16–0.35%), while
subgroup FNR for MI in an older-female subgroup is not (CV 3.6–14.7%, 17–92× larger), and this
gap survives threshold recalibration and replicates externally. See the paper for full results.

## Repository structure

Everything here was trained and run in **Google Colab**, not locally — the folder layout
deliberately mirrors the notebook's own Google Drive project folder
(`/content/drive/MyDrive/ecg_fairness_project`), so you can drop this repo straight into Drive and
the notebook's hardcoded paths just work. See "Running this in Google Colab" below.

```
ecg-fairness-instability/
├── notebooks/ECG_JOURNAL.ipynb   Main pipeline notebook (22 blocks, Colab-native)
├── configs/seeds.json              The 10 fixed seeds
├── data/README.md                    Dataset acquisition notes (no raw data committed)
├── checkpoints/                      All 40 trained checkpoints (4 architectures x 10 seeds, ~46 MB)
├── figures/                            All generated paper figures (Fig. 1-11)
├── docs/reproducibility.md         Notebook-block-to-paper-section map, environment record
├── requirements.txt                   Only needed for loading checkpoints outside Colab
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
   `notebooks/ECG_JOURNAL.ipynb` via Colab's *File → Open notebook → GitHub* tab, or upload the
   file directly via *File → Upload notebook*.

2. **Put the pretrained artifacts in your Google Drive** so the notebook loads them instead of
   retraining all 40 models from scratch. In your Drive, create:

   ```
   My Drive/ecg_fairness_project/checkpoints/   ← copy the contents of this repo's checkpoints/ here
   ```

   (Optional but saves a ~2 GB download: also copy the PTB-XL zip into
   `My Drive/ecg_fairness_project/data/` once you've downloaded it once — Block 3 will find and
   reuse it. Not required; if it's missing, Block 3 downloads it automatically.)

3. **Set the runtime.** *Runtime → Change runtime type*. A GPU (T4 is what the paper used) is only
   needed if you intend to retrain something; if every checkpoint above is already in place, a
   CPU-only runtime is enough to reload everything and regenerate all figures/tables.

4. **Run all cells top to bottom** (*Runtime → Run all*). Block 2 mounts your Drive
   (`drive.mount('/content/drive')`) — approve the auth popup. From there:
   - Block 3 downloads/extracts PTB-XL (skipped if already cached, see `data/README.md`).
   - Blocks 10–12 (`train_or_load`) check `{PROJECT_DIR}/checkpoints` for each of the 40
     `{model}_seed{seed}.pth` files before training anything — with Step 2 done, every one is
     found and loaded, so the whole notebook re-runs in minutes instead of retraining.
   - Blocks 16–21 download the Georgia database fresh each session (not cached to Drive, see
     `data/README.md`) and run zero-shot external validation on the loaded checkpoints.
   - Every paper table is printed directly to cell output (Blocks 9, 11, 12, 19–21); every paper
     figure is saved to `{PROJECT_DIR}/figures` (Blocks 13, 15, 22).

5. **Retrieve results.** Copy `{PROJECT_DIR}/figures` back out of Drive if you want the
   regenerated figures versioned in this repo, and copy any table output you need from the cell
   logs — see [`figures/README.md`](figures/README.md) for the full figure-to-table mapping.

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
  `checkpoints/`, so every number in the paper can be recomputed without retraining.
- **Known non-determinism**: Section V-C of the paper reports that a same-seed, independently
  executed retraining pass did *not* reproduce bitwise-identical results on this hardware/software
  stack (consistent with documented GPU floating-point non-determinism). This is a limitation of
  the underlying stack, not of the released code, and is documented in
  `docs/reproducibility.md`.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this code, please cite the paper once published;
in the meantime, cite this repository directly.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). PTB-XL and the Georgia
12-Lead ECG Challenge Database remain governed by their own PhysioNet licenses.
