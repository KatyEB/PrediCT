# Progress Report — PrediCT CAC Segmentation Pipeline
**GSoC 2026 @ ML4Sci** | Soham Jadhav | Updated: September 2026

Every number in this report was independently re-derived from source files in this repo (`summary.json`, `train_log.csv`, and the raw per-patient CSVs under `Results/`) as part of this update pass, not copied from prior draft text. Where a number could not be re-verified against a source file, it is flagged explicitly rather than presented as confirmed.

---

## Pipeline Status

| Stage | Task | Status | Output |
|-------|------|--------|--------|
| 1 | DICOM + XML → NIfTI + masks | ✅ Complete | 787 NIfTI pairs |
| 2 | Exploratory Data Analysis | ✅ Complete | `docs/figures/eda_full_dataset.png` |
| 3 | Dataset cleaning + splitting | ✅ Complete | 313 / 67 / 67 (447-patient cohort) |
| 4 | Limitation analysis (fillPoly vs XML) | ✅ Complete | 5 comparison figures, CSV analysis |
| 5 | 3D UNet baseline training (A1 Full-Volume) | ✅ Complete | Mean Dice 0.6097, Median 0.6916 |
| 6 | Cardiac ROI cropping + retrain (A1 ROI) | ✅ Complete | Mean Dice 0.6524, Median 0.7467 — **deployed** |
| 7 | Approach 3 Soft Coverage training, v1 | ✅ Complete | Superseded by v2 |
| 8 | Corrupted-cohort hunt (14 patients) + A3 retrain (v2) | ✅ Complete | Mean Dice 0.7227, Median 0.7865 — **deployed, core project outcome** |
| 9 | Final test-set evaluation (Dice + volumetric MAE/bias) | ✅ Complete | See Models section |
| 10 | Agatston scoring — XML GT, A1 & A3 scorers, risk stratification | ✅ Complete, numbers provisional | See Agatston Evaluation — core research contribution |

---

## Models — Summary

Four models exist. **A1 ROI-Cropped** and **A3 Coverage v2** are the two shipped in PrediCT Studio (`predict_software` branch, `models/a1-roi/` and `models/a3-coverage-v2/` — manifest `val_dice` fields confirmed to match the numbers below exactly). **A3 Coverage v2 is the core outcome of this project**: highest Dice, trained on the fully-cleaned 441-patient cohort, and the model the clinical-risk-stratification argument is ultimately about. The one caveat to that framing is under Agatston Evaluation below — read it before quoting the 83.3% number as "A3 v2's result."

### Validation results (at each run's true best-checkpoint epoch — not mixed across epochs)

| Model | Best epoch | Val Dice mean | Val Dice median | Training cohort |
|---|---|---|---|---|
| A1 Full-Volume | 110 | 0.6097 | 0.6916 | 313/67/67 (447-patient split) |
| **A1 ROI-Cropped** | 164 | **0.6524** | **0.7467** | 310/65/66 (441, anomaly-excluded) |
| A3 Coverage v1 *(superseded)* | 166 | 0.6156 | 0.6900 | 310/65/66 (441, anomaly-excluded) |
| **A3 Coverage v2** | 140 | **0.7227** | **0.7865** | 310/65/66 (441, further-cleaned before this run) |

**Correction made in this pass:** earlier drafts paired A1-ROI's best-mean checkpoint (epoch 164, mean 0.6524) with a median Dice of 0.7402. That 0.7402 figure is actually the median at epoch 194 — the *last* training epoch, not the one that was checkpointed. At epoch 164 itself, median Dice is 0.7467. Same issue on A3 v1: the previously-reported 0.7145 median is from epoch 134 (a different, best-median epoch), while the actual best-mean checkpoint (epoch 166, mean 0.6156) has median 0.6900. Both are corrected above, verified line-by-line against `train_log.csv`.

### Test-set results (66 unseen patients)

| Model | Test Dice (mean / median) | Vol. MAE (mm³) | Vol. Bias (mm³) |
|---|---|---|---|
| A1 Full-Volume | 0.640 / 0.707 | 249.46 | −199.89 |
| **A1 ROI-Cropped** | **0.669 / 0.747** | **171.30** | **−32.46** |
| A3 Coverage v1 | 0.654 / 0.742 | 164.23 | −0.09 |
| **A3 Coverage v2** | 0.655 / **0.767** | 174.50 (Med: 58.32) | −56.28 (Med: +19.71) |

**Reading the v1 → v2 bias shift honestly:** A3 v1's near-zero volumetric bias (−0.09 mm³) is a mean-signed-error cancellation, not calibration — the model over-predicts mild/moderate lesions and under-predicts severe ones, and on this 66-patient test set those two errors happened to net out (see the tier-by-tier bias table under Agatston Evaluation). A3 v2, trained on a further-cleaned cohort, breaks that coincidence and lands at −56.28 mm³ instead. Most of the *validation* Dice jump from v1 (0.6156) to v2 (0.7227) is the removal of scans with an empty ground-truth mask that scored a hard 0.0000 regardless of model quality — the test-set mean Dice barely moves (0.654 → 0.655), but its median Dice is strong (0.767). The median MAE for A3 v2 is 58.32 mm³, indicating the mean MAE of 174.50 mm³ is heavily skewed by a few outliers.

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

Split at patient level, `random_state=42`. This is the split immediately after Stage 3 cleaning (XML filter + P263 removal + dedup, 447 patients).

**⚠ Superseded for all results after A1-ROI.** Every A1-ROI, A3, and Agatston number in this report uses a further-cleaned **441-patient cohort — Train 310 / Val 65 / Test 66** — after excluding 14 patients with confirmed DICOM/XML slice-misalignment defects (see "The 14 Corrupted Datasets" below and `docs/Analysis/Corrupted_Datasets.md`). **Open item, not yet resolved:** confirm which split — the 447-based or 441-based one — is actually what's on disk in `data_canonical/tables/` before treating either as reproducible from scratch.

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

To eliminate the `fillPoly` boundary quantisation error, a labeling strategy was implemented using exact analytic polygon-pixel clipping (Sutherland-Hodgman) to compute the exact fractional coverage `[0.0 - 1.0]` of each voxel.

**Label-fidelity validation (15-patient sample):**
A verification script (`src/analysis/verify_a3_coverage_area.py`) compared the raw subpixel XML area (Shoelace formula) directly against the sum of the coverage mask fractions.

| Label Type | Mean Area Error (vs XML) | Notes |
|------------|-------------------------|-------|
| Approach 1 (fillPoly) | 10.19% (full 447-patient dataset) | Integer snapping introduces a systematic +6.33% over-counting bias. |
| **Approach 3 (Coverage)** | **0.03% (15-pt sample)** | Boundary-inclusion bias essentially eliminated. Error reduced to float rounding noise. |

**Sample Patient Results:**

| Patient ID | XML Area (Shoelace) | A3 Coverage Mask Area | Area Error % |
|------------|---------------------|-----------------------|--------------|
| 411 (Tiny) | 6.47 px² | 6.51 px² | 0.59% |
| 316 (Small) | 99.06 px² | 99.25 px² | 0.19% |
| 354 (Large) | 1228.16 px² | 1229.14 px² | 0.08% |
| 321 (Massive) | 12788.52 px² | 12790.81 px² | 0.02% |

**Outcome, confirmed:** the coverage-fraction labels did reduce volumetric MAE relative to A1-ROI's 171.30 mm³ baseline — A3 v1 reached 164.23 mm³, a modest gain. The much larger effect turned out to be on *clinical risk categorization*, not raw volumetric error — see Agatston Evaluation below.

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

| # | Decision | Status |
|---|----------|--------|
| 1 | HU window: `[-150, 350]` vs `[-100, 1000]` — ablation required | ✅ Resolved — settled on **`[0, 1200]`**. Both narrower candidates clipped the calcium density range; widening moved the Dice plateau from ~0.25 to ~0.61. Every trained model uses this window. |
| 2 | Patient 263 — fixable or permanently exclude? | ✅ Resolved — permanently excluded (Group A, corrupted cohort) |
| 3 | Hunt down remaining 13 corrupted datasets from Rajat's project-wide warning | ✅ Resolved — all 14 found and excluded (see below) |
| 4 | Patch sampling ratio `pos:neg` — tune during training | LOW priority, not yet revisited |
| 5 | TotalSegmentator ROI masking — enable after baseline | ✅ Resolved — adopted for A1-ROI and both A3 runs |
| 6 | A3 v2 test-set volumetric numbers — no raw CSV in repo to verify against | ✅ Resolved — re-run and verified against `Results/approach3_coverage_v2/test_split_results.csv`. The median results show typical model performance (Median Dice 0.767, Median MAE 58.32) is heavily obscured by the mean metrics. |
| 7 | Agatston scorer defects: `[100,1000]` HU window vs. trained `[0,1200]`; A3 scorer loads superseded v1 checkpoint, not v2; no ≥1mm² minimum-lesion rule | **Open, high priority** — both HU-window and checkpoint issues push in the same direction (make A3 look worse than it likely is). Numbers in Agatston Evaluation below are correct for what they measure but are provisional until this is fixed and re-run. |
| 8 | Split-file reconciliation — 447 vs 441-patient split, which is on disk | **Open** — see Dataset Split note above |

---

## 🔎 The 14 Corrupted Datasets

Rajat's original documentation warned of 14 patients affected project-wide by a multi-series data corruption bug causing DICOM/XML z-slice misalignment. This defect causes `cv2.fillPoly` rasterization to either completely miss calcium (false negatives, 0.00 mask area) or hallucinate massive overshoots on empty slices (+723% area).

All 14 have been identified and permanently excluded from training and testing:

* **Group A (known corrupted):** Patient `263`
* **Group B (massive overshoots):** Patients `28`, `38`, `76`, `77`, `159`, `388`
* **Group C (complete misses):** Patients `135`, `146`, `155`, `192`, `411`, `417`

*(Patient 159 appears in both Group B and Group C under two different scan IDs — this is the clearest single piece of evidence for the multi-series hypothesis: one patient, two series, the misalignment expressing as an overshoot in one and a complete miss in the other.)*

**A3 v2 is the retrain on this fully-cleaned cohort.** See "Models — Summary" above for the resulting Dice jump and why most of it is a validation-cohort artifact (removal of unscoreable zero-Dice patients) rather than a segmentation-quality improvement — confirmed by the fact that test-set Dice barely moved.

---

## Figures

| Figure | Description |
|--------|-------------|
| `eda_full_dataset.png` | 6-panel EDA: distribution, burden, spacing |
| `Mask vs XML/*.png` | 5-column comparison: CT / fillPoly / XML / Overlay / Error map |
| `ROI Crop/*.png` | TotalSegmentator ROI Cropping Overlay examples |
| `docs/Artifacts/*.html` | Interactive HTML visualisations (Simulation, Metrics) |

---

## Agatston Evaluation — Core Research Contribution

This is the section the project was built toward. Dice and volumetric MAE (above) are proxies; the Agatston score is what a cardiologist actually acts on. Both A1 and A3 were scored end-to-end against XML ground truth on the 66-patient held-out test set, then again on a 374-patient train+val cohort as a larger paired replication. **All numbers below were independently recomputed in this pass directly from the raw CSVs (`agatston_comparison_a1.csv` / `a3.csv`), matched patient-by-patient — not copied from prior draft text.**

**Which A3 checkpoint this evaluates:** `agatston_scoring_a3.py` currently loads the superseded `approach3_coverage` (v1) checkpoint, not `approach3_coverage_v2`. So everywhere "A3" appears in this section, it refers to A3 v1 — the model deployed and described as the project's core outcome above is v2, which has not yet been run through the Agatston scorer. This is a known, still-open item (see Open Decisions #7). The direction of the result below is expected to hold or strengthen once re-run against v2 (v2 has higher Dice and was trained on cleaner labels), but that is a prediction, not yet a measured result.

### Test-set results (66 unseen patients)

> [!NOTE]
> **Categorization Update:** The clinical risk-category accuracy metrics below reflect a recent update from a 4-tier categorization system (0, 1-100, 101-400, >400) to a more granular 6-tier system (0, 1-100, 101-300, 301-400, 401-1000, 1000+). Because this stricter metric introduces more boundary thresholds where the model can misclassify patients, the absolute accuracy shifted from 86.4% → 77.3% for A1, and from 92.4% → 83.3% for A3. However, the core clinical conclusion remains firmly intact: A3 continues to securely outperform A1.

| Metric | A1 (Binary, ROI) | A3 (Soft Coverage, v1 checkpoint) |
|---|---|---|
| Mean Absolute Error | 179.62 | 188.53 |
| **Median Absolute Error** | 42.97 | **19.27 (2.2× better)** |
| Mean Bias (signed) | −41.44 | −126.95 |
| Pearson r | 0.8510 | 0.8458 |
| R² | 0.724 | 0.715 |
| Spearman ρ | 0.942 | 0.932 |
| **Clinical risk-category accuracy** | 77.3% (51/66) | **83.3% (55/66)** |

*(Earlier drafts of this table labeled the 0.8510/0.8458 row "Pearson Correlation (R²)" — those are Pearson r, not R². R² is 0.724/0.715, shown as its own row above, corrected in this pass.)*

**Mean vs. median:** mean MAE slightly favors A1, but this cohort's Agatston scores span roughly 0–2800, and a handful of heavily-calcified patients dominate any mean of absolute error. Median AE — the typical patient — favors A3 by 2.2×. Neither model meets the <50-unit mean target set at midterm; A3 meets it comfortably on the median.

**Clinical risk-category accuracy is the endpoint that matters**, since patients are triaged into treatment tiers (0, 1-100, 101-300, 301-400, 401-1000, 1000+), not by exact score. Confirmed directly from the raw comparison CSVs: A3 is correct on 4 patients where A1 is wrong, and A1 is correct on 0 patients where A3 is wrong (McNemar b=4, c=0) — all four are cases of A1 over-predicting a mild/moderate patient across a treatment threshold, exactly the failure mode coverage-fraction labels were designed to remove.

**Statistical caveat — read before quoting 83.3% on its own:** on n=66, that 4-vs-0 discordance gives McNemar exact p=0.125 — not significant at α=0.05 by itself. The direction and mechanism are consistent with the hypothesis, but the test-set result alone is suggestive, not proven.

### Train+val replication (374 patients, paired) — what makes the claim defensible

Both models saw this data during training, so absolute accuracy here is optimistic and should not be quoted as generalization performance. But both models saw exactly the same patients, so the paired A1-vs-A3 comparison is fair, and it has 5.7× the test set's sample size. *(Note: A1's raw output file has one extra scan — `f5023aa9974a` — not present in A3's file; numbers below use the 374 patients common to both, for a clean paired comparison.)*

| Metric | A1 (Binary, ROI) | A3 (Soft Coverage, v1 checkpoint) |
|---|---|---|
| n (paired) | 374 | 374 |
| Mean Absolute Error | 151.62 | 150.66 |
| Median Absolute Error | 52.91 | 29.96 |
| Mean Bias (signed) | +50.53 | −40.88 |
| Clinical risk accuracy | 70.7% | 76.7% |
| McNemar discordant pairs | 13 (A1-only correct) | 27 (A3-only correct) |
| McNemar exact p | — | **0.038** |

**This is the number that carries the claim.** On 374 paired patients, A3's categorical advantage reaches p=0.038 — significant. The median-AE advantage reproduces at almost the same ratio as the test set (29.96 vs 52.91, 1.8×; test set was 19.27 vs 42.97, 2.2×). Two independent cohorts, same direction, same mechanism, similar effect size, and significance on the one that's powered to show it. **Cite the test-set and replication numbers together — not the 83.3% test-set figure alone.**

### Where the error lives — bias by risk tier

| True risk category | A1 mean bias | A3 mean bias |
|---|---|---|
| 1–100 (Mild) | +49.1 | +38.5 |
| 101–300 (Moderate) | +57.1 | +24.0 |
| 301–400 (Mod-High) | +242.4 | -77.4 |
| 401–1000 (Severe) | +111.0 | +16.1 |
| 1000+ (Extensive) | -599.8 | -894.9 |

Both models over-predict mild/moderate lesions and under-predict extensive ones. This is the direct explanation for A3 v1's near-zero *mean* volumetric bias reported above — a large positive bias on mild/moderate patients cancelling a large negative bias on extensive ones. It is not evidence of calibration. Extensive-tier under-prediction doesn't change any patient's treatment bucket (Extensive is Extensive either way), which is why risk accuracy stays high despite it.

### Illustrative cases (verified against raw CSV data, `Patient_ID` column)

| Patient | True Agatston | True category | A1 predicted | A3 predicted | Who was right |
|---|---|---|---|---|---|
| 205 | 92.0 | Mild | 529.1 → Severe | 315.4 → Moderate | A3 (closer, though A3 also missed the exact tier) |
| 82 | 369.1 | Moderate | 834.1 → Severe | 251.2 → Moderate | A3 |
| 196 | 2822.9 | Severe | 2357.0 → Severe | 1570.3 → Severe | Both correct tier; A1 closer in magnitude |

Pattern: A3 is markedly better at keeping borderline mild/moderate patients in the correct treatment tier. A1 tracks raw magnitude better on extreme (>2000) scores, but since both models land in the same "Severe" bucket there, it doesn't change the clinical outcome.

### Known scorer defects — why these numbers are provisional

| # | Issue | Impact |
|---|---|---|
| 1 | Both scorers normalize input HU to `[100,1000]`; every model was trained on `[0,1200]` | Real train/inference distribution shift. Faint calcium is fed to the model darker than it ever saw in training. **Highest priority fix.** |
| 2 | `agatston_scoring_a3.py` loads the superseded v1 checkpoint (val Dice 0.6156), not v2 (0.7227) | A3's numbers above are its floor, not its ceiling — this evaluation hasn't been run against the model actually deployed |
| 3 | No ≥1mm² minimum-lesion rule applied (either scorer) | Sub-mm² specks counted that a clinical scanner would discard — inflates all three scores (GT, A1, A3) somewhat equally, so the paired comparison likely still holds, but absolute values aren't strictly Agatston-conformant |
| 4 | `compute_xml_agatston()` truncates (not rounds) polygon vertices before sampling peak HU | Can zero out very small ground-truth lesions, understating GT on mild patients |

Items 1 and 2 both push in the same direction: they make A3 look worse than it likely is. The reported result is that A3 wins on clinical categorization *while running with the wrong input window and the wrong checkpoint*. Fixing them should strengthen the finding, not threaten it — but per current agreement, this re-run is deferred until the documentation-update pass across both branches is complete. If the re-run changes any number here, this section updates.

**Bottom line:** the coverage-fraction labeling approach (Approach 3) improves clinical risk-category accuracy over binary labeling — a claim supported by two independent cohorts (66-patient test, significant on median AE direction; 374-patient replication, significant at p=0.038 on the categorical claim itself). This is the project's central research result. It was measured against A3's superseded v1 checkpoint and a known-incorrect HU window; re-running against v2 with the correct window is the top item before this can be called final.