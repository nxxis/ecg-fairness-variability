# data/

No raw signal data is committed to this repository. Both datasets are publicly available on
PhysioNet and are downloaded by the notebook itself, inside Colab. This file documents where they
come from and exactly what the notebook does and does not persist — see the root
[`README.md`](../README.md) for the full Colab walkthrough.

## PTB-XL (primary dataset, training + internal test)

- Source: [PTB-XL, a large publicly available electrocardiography dataset (v1.0.3)](https://physionet.org/content/ptb-xl/1.0.3/)
- 21,799 12-lead recordings from 18,869 patients, 100 Hz downsampled signals used throughout.
- SCP codes are mapped to five diagnostic superclasses (NORM, MI, STTC, CD, HYP) following
  Strodthoff et al. (2021).
- The official stratified 10-fold split is used unmodified: folds 1-8 train (n=17,418, 15,023
  patients), fold 9 validation (n=2,183, 1,942 patients), fold 10 held-out test (n=2,198, 1,904
  patients), ensuring patient-level disjointness. Because this is a multi-label problem, the five
  per-class counts in each split can and do sum to more than that split's total recording count.

**How the notebook actually handles this (Block 3):** the zip archive is downloaded once to
Google Drive at `{PROJECT_DIR}/data/` (persisted across sessions, so it is never re-downloaded),
then extracted to local, ephemeral Colab disk at `/content/fast_ptbxl/...` (`DATA_DIR`) for fast
per-file read throughput during training — the extracted files themselves are **not** persisted to
Drive and are re-extracted from the cached zip at the start of every fresh Colab runtime (fast;
seconds to a couple of minutes, not a re-download).

You do not need to do anything manually — running Block 3 in Colab handles both the download and
extraction. This section is here only so the paths make sense if you're reading the notebook or
adapting it outside Colab.

## Georgia 12-Lead ECG Challenge Database (external validation, zero-shot)

- Source: [PhysioNet/CinC 2020 Challenge training data, Georgia subset](https://physionet.org/content/challenge-2020/1.0.2/training/georgia/)
- 10,344 recordings from the Southeastern United States, collected independently of PTB-XL's
  German cohort, with age, sex, and SNOMED-CT diagnosis codes per recording.
- Native 500 Hz; resampled to 100 Hz and truncated/zero-padded to 1000 samples before inference,
  with per-lead z-score normalisation applied identically to PTB-XL.
- SNOMED-CT diagnosis codes are mapped onto the same five PTB-XL superclasses via keyword rules
  grounded in the PTB-XL category definitions (see `docs/reproducibility.md`); 34 of the 111
  distinct SNOMED-CT codes present in the database map onto this five-superclass taxonomy.
- **Two exclusion steps reduce the usable cohort from 10,344 to n = 8,144**, the number reported
  in the paper's external-validation results: (1) recordings whose *only* diagnoses fall outside
  the five-superclass taxonomy (e.g. atrial fibrillation, ectopy, pacing artefacts) are excluded,
  reducing the population to 8,146; (2) a further two recordings (0.02%) with non-finite raw
  signal values (a WFDB sentinel for missing samples) are excluded, leaving n = 8,144. Resulting
  class prevalence: NORM 2,205; MI 7; STTC 4,171; CD 2,230; HYP 1,368; 46.8% female; 57.8% aged
  ≥60. MI is too sparse (0–3 positives per subgroup) for meaningful subgroup-level FNR estimation,
  so STTC (773–1,357 positives per subgroup) is used as the headline metric for cross-domain
  fairness-stability comparison; Georgia's MI results are reported separately with an explicit
  low-power caveat.
- The notebook also builds a second, more conservative crosswalk retaining only unambiguous
  SNOMED-CT diagnoses, as a mapping-sensitivity check (not a like-for-like replication): it reduces
  the eligible cohort to 6,259 recordings, predominantly from STTC, with demographic composition
  shifting only slightly (Section VII of the paper).

**How the notebook actually handles this (Block 16):** unlike PTB-XL, the Georgia data is
downloaded directly to local, ephemeral Colab disk at `/content/georgia` (`EXTERNAL_DIR`) and is
**not** cached to Drive at all — it is re-downloaded from PhysioNet at the start of every fresh
Colab runtime. Again, Block 16 handles this automatically; nothing to do manually.

## No demographic model inputs

Sex and age are used exclusively for post-hoc subgroup fairness auditing in both datasets — never
as model inputs — consistent with the Ethical Considerations in Section VI of the paper. Both
datasets are de-identified and distributed under PhysioNet's terms; no new patient data was
collected for this study.
