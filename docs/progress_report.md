# Progress Report — PrediCT CAC Segmentation Pipeline
**GSoC 2026 @ ML4Sci** | Soham Jadhav | Updated: June 2026

---

## Pipeline Status

| Stage | Task | Status | Output |
|-------|------|--------|--------|
| 1 | DICOM + XML → NIfTI + masks | ✅ Complete | 787 NIfTI pairs |
| 2 | Exploratory Data Analysis | ✅ Complete | `docs/figures/eda_full_dataset.png` |
| 3 | Dataset cleaning + splitting | ✅ Complete | 313 / 67 / 67 |
| 4 | Limitation analysis (fillPoly vs XML) | ✅ Complete | 5 comparison figures, CSV analysis |
| 5 | 3D UNet baseline training | ✅ Complete | Mean Dice 0.61, Median 0.69 |
| 6 | Cardiac ROI Cropping Simulation & Training | ✅ Complete | Results in `approach1_roi_cropped/` |
| 7 | Approach 3 (Soft Coverage) Training | ✅ Complete | Results in `Final_Testing_Report.md` |
| 8 | Final Test Set Evaluation (Volume MAE) | ✅ Complete | Results in `Final_Testing_Report.md` |

---

## Final Model Performance & Volumetric Analysis (Unseen Test Set)

The models were evaluated on the `test_split.parquet` dataset consisting of 66 unseen patients to measure generalization and absolute calcium volume accuracy.

| Model | Test Set Dice | Volumetric MAE (mm³) | Volume Bias (Mean Error) mm³ |
|-------|---------------|-----------------------|------------------------------|
| **A1 Full Volume** | 0.640 | 249.46 | -199.89 |
| **A1 ROI Cropped** | 0.669 | 171.30 | -32.46 |
| **A3 Coverage (Soft Labels)** | 0.654 | **164.23** | **-0.087** |

**Key Takeaways:**
- **Generalization:** All models generalized exceptionally well, scoring equal to or higher on the test set than in validation (A1 ROI Cropped reached 0.669 Test Dice).
- **ROI Cropping Impact:** Cropping to the heart reduced the extreme under-prediction bias of the full volume model (improving bias from -199.89 mm³ to -32.46 mm³).
- **A3 Soft Labels Win:** Modeling sub-pixel partial volume directly (`A3_Coverage`) yielded the lowest Mean Absolute Error (164.23 mm³) and achieved an incredible near-zero volume bias of -0.087 mm³.

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Raw DICOM series found | 789 |
| Successfully processed | 787 |
| XML-annotated (valid gated) | 448 |
| Non-gated chest CTs excluded | 339 |
| After removing P263 + dedup | **447 clean patients** |
| Positive (calcium detected) | 446 (99.8%) |
| Negative | 1 (0.2%) |

**Calcium burden (positive patients):**

| Stat | Voxels | Slices with CAC |
|------|--------|-----------------|
| Min | 7 | 1 |
| 25th pct | 98 | 2 |
| Median | 357 | 6 |
| Mean | 1,135 | 8.2 |
| 75th pct | 1,276 | 13 |
| Max | 13,093 | 35 |

**Original spacing range:** 0.246mm – 0.715mm across 787 scans.  
Per-patient scale correction was applied before XML polygon overlay.

---

## Dataset Split

| Split | Patients | Positive | Negative |
|-------|----------|----------|----------|
| Train | 313 | 312 (99.7%) | 1 |
| Val | 67 | 67 (100%) | 0 |
| Test | 67 | 67 (100%) | 0 |

Split at patient level. `random_state=42`.

---

## Key Finding — fillPoly Boundary Limitation

XML polygon annotations are stored at floating-point (subpixel) precision.  
`cv2.fillPoly` rounds vertices to the integer pixel grid before filling.  
This creates boundary quantisation error that scales with original pixel spacing mismatch.

| Patient | Original Spacing | Scale Factor | Area Error |
|---------|-----------------|--------------|------------|
| P0 (34px deposit) | 0.47mm | 1.28× | **62.5%** |
| P1 z=19 | 0.38mm | 1.03× | **7.0%** |
| P10 z=10 | 0.44mm | 1.18× | **79.4%** |

**Implication:** For small calcium deposits (<50 voxels), the majority of boundary pixels  
are uncertain. This is a **label-quality ceiling**, not a model performance ceiling.

---

## Approach 3 — Continuous Coverage Fraction (Soft Labels)

To eliminate the `fillPoly` boundary quantisation error, a new labeling strategy was implemented using exact analytic polygon-pixel clipping (Sutherland-Hodgman) to compute the exact fractional coverage `[0.0 - 1.0]` of each voxel.

**Validation Results (15-patient sample):**
A verification script (`src/analysis/verify_a3_coverage_area.py`) compared the raw subpixel XML area (Shoelace formula) directly against the sum of the coverage mask fractions.

| Label Type | Mean Area Error (vs XML) | Notes |
|------------|-------------------------|-------|
| Approach 1 (fillPoly) | 10.19% (full dataset) | Integer snapping introduces systematic +6.33% over-counting bias. |
| **Approach 3 (Coverage)** | **0.03% (15-pt sample)** | Boundary-inclusion bias completely eliminated. Error reduced to float rounding noise. |

**Sample Patient Results:**

| Patient ID | XML Area (Shoelace) | A3 Coverage Mask Area | Area Error % |
|------------|---------------------|-----------------------|--------------|
| 411 (Tiny) | 6.47 px² | 6.51 px² | 0.59% |
| 316 (Small) | 99.06 px² | 99.25 px² | 0.19% |
| 354 (Large) | 1228.16 px² | 1229.14 px² | 0.08% |
| 321 (Massive)| 12788.52 px² | 12790.81 px² | 0.02% |

**Implication:** The soft coverage labels are geometrically exact. When evaluated using a Soft Agatston Scorer (sum of predicted probability × voxel volume), Approach 3 is highly expected to reduce the 171.30 mm³ Volumetric MAE baseline established by Approach 1.

---

## Critical Training Note — Voxel-Level Imbalance

| Metric | Value |
|--------|-------|
| Voxels per scan (approx.) | ~31 million |
| Median calcium voxels | 357 |
| Foreground:background | ~1 : 27,000 |

**Random patch sampling will produce Dice = 0.**  
Model converges to predicting all-zero with near-perfect BCE loss.  
**Required:** `RandCropByPosNegLabeld(pos=1, neg=1)` in MONAI.

---

## Open Experimental Decisions

| # | Decision | Priority |
|---|----------|----------|
| 1 | HU window: `[-150, 350]` vs `[100, 1000]` — ablation required | HIGH |
| 2 | Patient 263 — fixable or permanently exclude? | MEDIUM |
| 3 | 2 unknown error patients — confirm IDs with Rajat | MEDIUM |
| 4 | Patch sampling ratio `pos:neg` — tune during training | LOW |
| 5 | TotalSegmentator ROI masking — enable after baseline | ✅ COMPLETED |

---

## Figures

| Figure | Description |
|--------|-------------|
| `eda_full_dataset.png` | 6-panel EDA: distribution, burden, spacing |
| `Mask vs XML/*.png` | 5-column comparison: CT / fillPoly / XML / Overlay / Error map |
| `ROI Crop/*.png` | TotalSegmentator ROI Cropping Overlay examples |
| `docs/Artifacts/*.html` | Interactive HTML visualisations (Simulation, Metrics) |
