# PrediCT — CAC Segmentation Pipeline
### GSoC 2026 @ ML4Sci | Building and Comparing Segmentation Strategies for Coronary Artery Calcium

**Contributor:** Soham Jadhav &nbsp;|&nbsp; **Mentors:** Katy, Anna &nbsp;|&nbsp; **Co-contributor:** Rajat  
**Midterm:** July 24, 2026 &nbsp;|&nbsp; **Final:** September 21, 2026

---

## What This Branch Contains

This branch (`soham_segmentation`) covers **Stages 1–10** of the PrediCT pipeline:

```
Stage 1  DICOM + XML → NIfTI + binary masks     [COMPLETE]
Stage 2  Exploratory data analysis               [COMPLETE]
Stage 3  Dataset cleaning + train/val/test split [COMPLETE]
Stage 4  3D UNet baseline training               [COMPLETE]
Stage 5  ROI Cropping & TotalSegmentator Masking [COMPLETE]
Stage 6  Approach 3 (Soft Coverage) Training     [COMPLETE]
Stage 7  Final Test Set Evaluation (Volume MAE)  [COMPLETE]
Stage 8  Corrupted-cohort hunt (14 patients) + A3 retrain (v2) [COMPLETE]
Stage 9  Agatston scoring — XML GT, A1 & A3 scorers, risk stratification [COMPLETE — core research contribution]
Stage 10 Hybrid + Custom CNN approach (Results Impr.) [PENDING — time-permitting]
Stage 11 Software deployment automation plan     [IN PROGRESS — see `predict_software` branch]
```

**Core outcome of this branch:** `A3 Coverage v2` — trained on the fully-cleaned 441-patient cohort after removing 14 corrupted-label scans, val Dice 0.7227 (median 0.7865). This is the model that ships in PrediCT Studio's segmentation pipeline alongside `A1 ROI`. See **Models** below for the full comparison; see `docs/progress_report.md` for the complete writeup with source-file references.

---

## Models

Four models were trained across this project. Two (**bold**) are the ones actually deployed in the PrediCT Studio clinical tool (`predict_software` branch, `models/a1-roi/` and `models/a3-coverage-v2/`); the other two are reference/superseded runs kept for the ablation story.

| Model | Status | Val Dice (mean / median) | Test Dice (mean / median) | Test Vol. MAE (mm³) | Test Vol. Bias (mm³) |
|---|---|---|---|---|---|
| A1 Full-Volume | Reference baseline (pre-ROI) | 0.6097 / 0.6916 | 0.640 / 0.707 | 249.46 | −199.89 |
| **A1 ROI-Cropped** | **Deployed** | **0.6524 / 0.7467** | **0.669 / 0.747** | **171.30** | **−32.46** |
| A3 Coverage v1 | Superseded by v2 | 0.6156 / 0.6900 | 0.654 / 0.742 | 164.23 | −0.09 |
| **A3 Coverage v2** | **Deployed — core project outcome** | **0.7227 / 0.7865** | **0.655** | **174.53†** | **−56.27†** |

† A3 v2's test-set volumetric MAE/bias are carried from earlier reporting and haven't yet been re-verified against a raw per-patient CSV in this repo (unlike the other three rows, which were independently recomputed from `Results/.../bland_altman_comparison.csv`). Scheduled for re-confirmation once the doc-update pass is done — see `docs/progress_report.md` Open Decisions.

All Dice figures above are read at each run's actual best-checkpoint epoch (`summary.json` → `best_epoch`, cross-checked against `train_log.csv`), not mixed across epochs. Full validation and training-run detail, plus the Agatston/clinical-risk results (the project's core research finding), is in `docs/progress_report.md`.

---

## Dataset

**Stanford COCA** — Coronary Artery Calcium and Chest CTs  
Download: https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa

> **Do not push dataset files.** The `.gitignore` excludes all DICOM, NIfTI, and parquet files.  
> After downloading, place the dataset at the path configured in `configs/default_config.yaml`.

Expected structure after download:
```
<data_root>/
  Gated_release_final/
    patient/          ← DICOM series (444 patients, ~789 series)
    calcium_xml/      ← XML plist annotation files (451 files)
  deidentified_nongated/
```

---

## Installation

```bash
# Clone and enter the repo
git clone <repo_url>
cd PrediCT-main
git checkout soham_segmentation

# Create virtual environment
python -m venv vmenv
source vmenv/bin/activate        # Linux/Mac
vmenv\Scripts\Activate.ps1      # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt covers:**
`SimpleITK`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`,
`pyarrow`, `scipy`, `scikit-image`, `matplotlib`, `tqdm`, `plistlib`

---

## Quick Start — Run The Full Pipeline

> Set your paths in `configs/default_config.yaml` before running anything.

```bash
# Step 1: Preprocess all DICOM series → NIfTI + masks (~60 min)
python src/preprocessing/COCA_processor_main.py

# Step 2: Filter to 447 clean patients (XML-annotated, deduped, P263 removed)
python src/preprocessing/cleanup_patient.py

# Step 3: EDA — distribution plots, spacing, calcium burden
python src/analysis/eda.py

# Step 4: Train/val/test split (313 / 67 / 67)
python src/preprocessing/split_dataset.py

# Step 5: Verify masks visually
python src/visualization/visualize_masks.py

# Step 6: Generate comparative limitation slides (meeting figures)
python src/visualization/xml_vs_mask_comparison_v3.py
```

All figures save to `docs/figures/`.

---

## Pipeline Details

### Stage 1 — Preprocessing (`src/preprocessing/COCA_processor.py`)

Converts raw DICOM cardiac CT scans + XML polygon annotations into standardized NIfTI volumes with binary calcium masks.

**Key parameters** (set in `configs/default_config.yaml`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target spacing | `0.37 × 0.37 × 3.0 mm` | Native x/y, standardised z |
| CT interpolator | `sitkLinear` | For image volumes |
| Mask interpolator | `sitkNearestNeighbor` | Preserves binary labels |
| Mask renderer | `cv2.fillPoly` | Integer pixel grid |
| HU window | **EXPERIMENTAL** | See open decisions below |
| TotalSegmentator | Enabled | Cardiac ROI masking via `generate_roi_cropped_dataset.py` |

**Output per patient:**
```
data_canonical/images/<scan_id>/
  <id>_img.nii.gz    ← Resampled CT volume
  <id>_seg.nii.gz    ← Binary calcium mask (0/1)
  <id>_meta.json     ← Spacing, size, voxel count, slice indices
```

### Stage 2 — EDA (`src/analysis/eda.py`)

Generates a 6-panel figure covering:
- Class distribution (positive vs negative patients)
- Calcium burden histogram (log scale, voxels per patient)
- Slice spread distribution
- Voxels vs slices scatter
- Original spacing distribution
- Key statistics table (percentiles)

**Key EDA findings:**

```
Raw dataset        787 scans processed
XML-annotated      448 scans (valid gated cardiac)
Non-gated excluded 339 scans (patient IDs 451+, no XML)
After cleaning     447 patients

Calcium burden (positive patients):
  Median  357 voxels   (~0.001% of scan volume)
  Mean    1135 voxels  (right-skewed, heavy outliers)
  Max     13093 voxels

Spacing range: 0.246mm – 0.715mm across 787 scans
  → Per-patient scale correction was essential

⚠ Voxel-level imbalance: ~1:27,000 foreground:background
  → Foreground-biased patch sampling is NON-NEGOTIABLE for training
  → Use MONAI RandCropByPosNegLabeld(pos=1, neg=1)
```

---

### Stage 3 — Dataset Cleaning (`src/preprocessing/cleanup_patient.py`)

```
787 scans
  → Filter: keep XML-annotated only          = 448 scans
  → Remove Patient 263 (bad mask, Rajat README) = 447 scans
  → Deduplicate P700 and P726 (2 series each)  = 447 unique patients
```

**Split** (`src/preprocessing/split_dataset.py`):
```
Train : 313 patients  (70%)
Val   :  67 patients  (15%)
Test  :  67 patients  (15%)
```
Split at **patient level**, `random_state=42`. All splits are ~100% positive  
(446/447 gated patients have detectable calcium — this is expected for COCA).

**⚠ Superseded by the corrupted-cohort exclusion.** All ROI-cropped, A3, and Agatston results reported in `docs/progress_report.md` use a further-cleaned **441-patient cohort (Train 310 / Val 65 / Test 66)**, after excluding 14 patients found to have DICOM/XML slice-misalignment defects (see `docs/Analysis/Corrupted_Datasets.md`). Before rebuilding splits from scratch, confirm which exclusion list is on disk — this is tracked as an open reconciliation item.

---

### Stage 6 — Limitation Analysis (`src/visualization/xml_vs_mask_comparison_v3.py`)

**Finding:** `cv2.fillPoly` rounds subpixel XML polygon vertices to integer grid.  
For small calcium deposits, this creates 7–79% area error depending on original pixel spacing.

| Patient | Scale Factor | Area Error | Severity |
|---------|-------------|------------|----------|
| P0 (tiny deposit, 34px) | 1.28× | 62.5% | High |
| P1 z=19 (larger deposit) | 1.03× | 7.0% | Low |
| P10 z=10 | 1.18× | 79.4% | High |

**Rule:** Larger original pixel spacing → larger rescaling → higher boundary error.  
This is a **label-quality ceiling**, not a model ceiling.

**Figure layout (5 columns per slice):**

| Col | Panel | Purpose |
|-----|-------|---------|
| ① | CT only | Anatomical context |
| ② | fillPoly mask | What the model trains on |
| ③ | XML outline | Radiologist annotation (subpixel) |
| ④ | Overlay | Green XML on top, red boundary scatter below |
| ⑤ | Error map | Yellow=correct, Orange=over-seg FP, Cyan=under-seg FN |

---

## Open Experimental Decisions

These are unresolved and require ablation runs. **None block the baseline training run.**

| # | Decision | Status |
|---|----------|--------|
| 1 | **HU window** | ✅ Resolved — settled on `[0, 1200]`. The narrower candidates `[-150, 350]` and `[-100, 1000]` both clipped the calcium density range and plateaued near Dice 0.25; `[0, 1200]` reached ~0.61 mean / ~0.69 median. Every trained model (A1 and A3, all variants) uses this window. |
| 2 | Patient 263 — fixable XML edge case or corrupt DICOM? | Inspect manually |
| 3 | 2 additional error patients (IDs unknown) — confirm with Rajat | Pending |
| 4 | Patch sampling ratio: `pos=1, neg=1` vs `pos=1, neg=3` | Tune during training |
| 5 | TotalSegmentator cardiac ROI masking — enable after baseline | ✅ Completed |

---

## Folder Structure

```
PrediCT-main/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default_config.yaml       ← All paths and hyperparameters here
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── COCA_processor.py     ← COCAProcessor class
│   │   ├── COCA_processor_main.py← Entry point, runs preprocessing
│   │   ├── cleanup_patient.py    ← Filter 787→447 clean patients
│   │   ├── split_dataset.py      ← 70/15/15 patient-level split
│   │   └── generate_roi_cropped_dataset.py ← TotalSegmentator ROI masking
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── eda.py                ← 6-panel EDA figure
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── visualize_masks.py              ← Mask overlay verification
│   │   └── xml_vs_mask_comparison_v3.py    ← 5-col limitation slides
│   └── training/
│       ├── __init__.py
│       ├── Train_3D_Unet_Binery.py         ← Baseline Training
│       └── Train_3D_Unet_ROI.py            ← ROI Cropped Training
├── src/testing/
│       └── evaluate_models.py              ← Test set evaluation script
├── Results/                      ← Training logs, config, best models
│   ├── approach1_roi_cropped/    ← A1 ROI-cropped (deployed)
│   ├── approach3_coverage_v2/    ← A3 Coverage v2 (deployed, core outcome)
│   ├── Archives/
│   │   ├── approach1_binary/     ← A1 full-volume (reference baseline)
│   │   ├── approach3_coverage/   ← A3 Coverage v1 (superseded by v2)
│   │   └── testing_models_A1_binary_A1_ROI_corpped/  ← per-patient test CSV, A1 models
│   └── Agaston_results/
│       ├── Agaston Results (Unseen Data)/  ← A1 & A3 Agatston vs XML GT, 66-patient test set
│       └── TrainVal_Experiment/            ← 374-patient train+val replication (McNemar significance)
├── docs/
│   ├── progress_report.md        ← Full written report
│   ├── Final_Testing_Report.md   ← Test generalization & volumetric MAE
│   ├── Approach3_Analysis.md     ← Approach 3 design and Soft Agatston
│   ├── Approach2_Analysis.md     ← Approach 2 comparative analysis
│   ├── FAQ.md                    ← Frequently Asked Questions (Data & Vis)
│   ├── Artifacts/                ← Interactive HTML artifacts
│   ├── Analysis/                 ← fillpoly vs xml real comparisons
│   └── figures/
│       ├── eda_full_dataset.png
│       ├── Mask vs XML/          ← XML boundary vs fillPoly visualisations
│       └── ROI Crop/             ← TotalSegmentator overlay examples
└── data_canonical/               ← NOT pushed (see .gitignore)
    ├── images/<scan_id>/         ← NIfTI volumes + masks
    └── tables/                   ← CSV + parquet splits
```

---

## Next Steps (Post-Midterm & Final)

- [x] `src/training/Train_3D_Unet_Binery.py` — 3D UNet baseline on GCP L4 VM
- [x] `src/training/Train_3D_Unet_ROI.py` — 3D UNet with TotalSegmentator ROI Cropping
- [x] Approach 3 Training and Testing
- [x] Final Test Set Evaluation (Volume MAE & Bias Analysis)
- [x] Comparative results & HTML Artifacts added
- [x] Foreground-biased patch sampling (`RandCropByPosNegLabeld`)
- [x] HU window — resolved, settled on `[0, 1200]`
- [x] Corrupted-cohort hunt — all 14 mislabeled scans found and excluded; A3 retrained as `Coverage v2` on the clean 441-patient cohort
- [x] **Agatston Score Calculation:** XML Shoelace area × peak-HU density weight, computed for GT, A1, and A3.
- [x] **Agatston Score Comparison:** A1 vs A3 vs XML ground truth, on both the 66-patient test set and a 374-patient train+val replication. Core finding: A3's coverage-fraction labels improve clinical risk-category accuracy (92.4% vs 86.4% on test; 83.7% vs 79.7% on the replication, p=0.038). Full results and caveats in `docs/progress_report.md`.
- [ ] **Known defect — scorer re-run pending:** both Agatston scorers currently normalize HU with `[100,1000]` (models were trained on `[0,1200]`), and the A3 scorer loads the superseded v1 checkpoint, not v2. Numbers above are correct for what they measure but are provisional until this is fixed and re-run.
- [ ] **Hybrid + Custom CNN Architecture:** Research and experiment with advanced models for further segmentation improvements (time permitting).
- [ ] **Software Application Automation Plan:** Formulate a full deployment plan to automate the end-to-end clinical pipeline (see `predict_software` branch for current state).

---

## Citation / Acknowledgements

**Dataset:** Gräni et al. (2021). COCA — Coronary Artery Calcium and Chest CTs. PhysioNet.  
**Project:** Google Summer of Code 2026 — ML4Sci  
**Mentors:** Katy Butler,