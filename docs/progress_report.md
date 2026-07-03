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
| 7 | nnU-Net + Hybrid Attention | ⏳ Pending | Comparison table |
| 8 | Soft Agatston evaluation | ⏳ Pending | Core research contribution |

---

## Model Performance & Improvements

Transitioning from full CT scans to cardiac ROI-cropped scans (along with the removal of the anomaly patient `fc53a04c4dd5`) yielded substantial improvements in segmentation accuracy and training efficiency.

| Approach | Setup | Mean Dice | Median Dice | Total Training Time |
|----------|-------|-----------|-------------|---------------------|
| **Approach 1 Baseline** | Full CT, native masks | 0.6097 | 0.6916 | 284.6 min |
| **Approach 1 ROI Cropped** | TotalSegmentator Heart ROI + anomaly excluded | 0.6524 | 0.7466 | 105.4 min |

**Key Takeaways:**
- **Accuracy Boost:** Mean Dice improved by ~0.043 and Median Dice by ~0.055, driven by eliminating false positives in non-cardiac structures.
- **Efficiency:** Total training time dropped by over 60% despite running more epochs, due to significantly smaller input volumes.

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
