# notebooks/

`ECG_JOURNAL.ipynb` is the single, linear Colab notebook that runs the entire study end to end: PTB-XL
acquisition and preprocessing, all four architectures trained across the ten fixed seeds
(`configs/seeds.json`), the fairness audits (FNR parity, equalized odds, threshold recalibration,
Brown-Forsythe variance testing, DeLong AUROC comparisons), Grad-CAM and robustness diagnostics,
and zero-shot external validation on the Georgia 12-Lead ECG Challenge Database.

The notebook is written for Google Colab — it mounts Google Drive in Block 2 and persists
checkpoints/figures to a Drive folder so a dropped session never loses progress. See the root
[`README.md`](../README.md) for the recommended step-by-step Colab setup, including how to seed
your Drive with the checkpoints already in `../checkpoints/` so the notebook loads instead of
retrains. (To run it outside Colab instead, replace the Drive-mount cell with a local
`PROJECT_DIR` path — every downstream cell reads/writes through that one variable — but this is
not the workflow the study was built or tested on.)

See [`../docs/reproducibility.md`](../docs/reproducibility.md) for a block-by-block map from the
notebook to the paper's sections, tables, and figures.
