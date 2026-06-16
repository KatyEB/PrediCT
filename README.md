# PrediCT — CAC Segmentation Pipeline
### GSoC 2026 @ ML4Sci | Building and Comparing Segmentation Strategies for Coronary Artery Calcium

**Contributor:** Soham Jadhav &nbsp;|&nbsp; **Mentors:** Katy, Anna &nbsp;|&nbsp; **Co-contributor:** Rajat  
**Midterm:** July 10, 2026 &nbsp;|&nbsp; **Final:** August 24, 2026

---

## What This Branch Contains

This branch (`soham/preprocessing-pipeline`) covers **Stages 1–3** of the PrediCT pipeline:

```
Stage 1  DICOM + XML → NIfTI + binary masks     [COMPLETE]
Stage 2  Exploratory data analysis               [COMPLETE]
Stage 3  Dataset cleaning + train/val/test split [COMPLETE]
Stage 4  3D UNet baseline training               [IN PROGRESS]
Stage 5  nnU-Net + Hybrid Attention comparison   [PENDING]
Stage 6  Soft Agatston evaluation                [PENDING]
```

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
# Clone and enter repo
git clone <repo_url>
cd PrediCT-main
git checkout soham/preprocessing-pipeline

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
| TotalSegmentator | Disabled | Optional cardiac ROI masking |

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
| 1 | **HU window**: `[-150, 350]` (Rajat's cardiac window, preserves soft tissue) vs `[100, 1000]` (calcium-salient, cuts anatomy) | Ablation required |
| 2 | Patient 263 — fixable XML edge case or corrupt DICOM? | Inspect manually |
| 3 | 2 additional error patients (IDs unknown) — confirm with Rajat | Pending |
| 4 | Patch sampling ratio: `pos=1, neg=1` vs `pos=1, neg=3` | Tune during training |
| 5 | TotalSegmentator cardiac ROI masking — enable after baseline | Post-baseline |

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
│   │   └── split_dataset.py      ← 70/15/15 patient-level split
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── eda.py                ← 6-panel EDA figure
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── visualize_masks.py              ← Mask overlay verification
│   │   └── xml_vs_mask_comparison_v3.py    ← 5-col limitation slides
│   └── training/
│       ├── __init__.py
│       └── train_unet.py         ← [IN PROGRESS]
├── docs/
│   ├── progress_report.md        ← Full written report
│   └── figures/
│       ├── eda_full_dataset.png
│       ├── v3_P0_z034.png
│       ├── v3_P1_z012.png
│       ├── v3_P1_z019.png
│       ├── v3_P10_z009.png
│       ├── v3_P10_z010.png
│       ├── 2740d96a230c_mask_check.png
│       ├── fd14b377bebc_mask_check.png
│       └── c3be56167c58_mask_check.png
└── data_canonical/               ← NOT pushed (see .gitignore)
    ├── images/<scan_id>/         ← NIfTI volumes + masks
    └── tables/                   ← CSV + parquet splits
```

---

## Next Steps (Midterm — July 10)

- [ ] `src/training/train_unet.py` — 3D UNet baseline on GCP L4 VM
- [ ] Foreground-biased patch sampling (`RandCropByPosNegLabeld`)
- [ ] HU window ablation — `[-150, 350]` vs `[100, 1000]`
- [ ] Validation Dice score ≥ 0.65 (acceptable) / ≥ 0.75 (strong)
- [ ] `src/evaluation/soft_agatston.py` — soft scorer implementation
- [ ] XML Area extraction → `xml_agatston_gt.csv`
- [ ] Agatston comparison table: binary vs soft vs XML ground truth

---

## Citation / Acknowledgements

**Dataset:** Gräni et al. (2021). COCA — Coronary Artery Calcium and Chest CTs. PhysioNet.  
**Project:** Google Summer of Code 2026 — ML4Sci  
**Mentors:** Katy (primary),
