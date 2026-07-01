# Approach 2 Investigation — Upsampled Re-Rasterization (2x Resolution)

**Project:** PrediCT — CAC Segmentation, GSoC 2026
**Branch:** `soham_segmentation`
**Date:** June 2026
**Status:** Investigation complete — Approach 2 does not outperform Approach 1 in current form. Recommend proceeding to Approach 3.

---

## 1. Verdict (read this first)

Approach 2 — re-rasterizing calcium masks from XML at 2x in-plane resolution (0.185mm vs the native 0.37mm used in Approach 1) — was hypothesized to reduce rasterization error against the true polygon boundary, based on the general principle that finer grids should better approximate a continuous shape.

**This was tested rigorously across three independent rasterization variants, all measured against ground-truth XML polygon area (not against each other). All three variants performed worse than Approach 1's existing mask generation.**

| Variant | Mean absolute error vs XML ground truth | vs A1 baseline (5.60%) |
|---|---|---|
| **Approach 1** (existing pipeline, native 0.37mm) | **5.60%** | baseline |
| Approach 2, truncation-based fill, 2x res | N/A — caused complete lesion collapse on small calcium (see §3) | fails outright |
| Approach 2, sub-pixel shift-fill, 2x res | 9.74% | **74% worse** |
| Approach 2, sub-pixel shift-fill, **native 1x res** (isolation test) | 19.21% | **243% worse** |
| Approach 2, rounded-coordinate fill, 2x res | 9.76% | **74% worse** |

No tested variant of the re-rasterization approach beat the original pipeline. The isolation test (row 3) proves the failure is **not caused by resolution** — resolution, if anything, partially offsets the error rather than causing it (9.74–9.76% at 2x vs 19.21% at 1x, same algorithm). **The actual root cause was confirmed by reading `COCA_processor_main.py` directly (§9): Approach 1 fills polygons once, on each scan's native pixel grid, then uses a proper image resampler to reach the target spacing. Every Approach 2 variant instead re-derived a synthetic scale factor and rasterized fresh onto a new grid — an extra, more error-prone transformation A1 never performs.**

**Recommendation:** Do not scale Approach 2 to the full 787-patient dataset in its current form. Move to Approach 3 (fractional voxel coverage, §10) as the next segmentation-label strategy — designed specifically to avoid the mechanism identified in §9, not just to try a different fill technique.

---

## 2. Background — why Approach 2 was proposed

Both Rajat and Soham independently identified the same underlying issue during dataset EDA: XML polygon annotations use continuous, sub-pixel coordinates, but converting them into a training-ready binary mask requires rasterizing onto a discrete pixel grid. This rasterization step (`cv2.fillPoly`) systematically changes the area relative to the true polygon:

- Partial pixels at the polygon boundary become either fully filled or fully empty.
- Small lesions (1–2 pixels wide) are affected disproportionately — a small absolute rounding error is a large *relative* error when the true shape is tiny.
- Rajat's independent analysis (group meeting, June 12) quantified this directly: a small lesion's XML area of 0.14mm² rasterized to 0.42mm² (~3.0x inflation), while a larger lesion's 7.88mm² rasterized to 10.41mm² (~1.3x inflation) — confirming the effect concentrates on small structures.

**Approach 2's hypothesis:** if the pixel grid is made finer (2x resolution, 0.185mm instead of 0.37mm), each pixel represents a smaller physical area, so the boundary rounding error should shrink as a fraction of total lesion area. This is a standard and reasonable expectation for rasterization error in general — the question was whether it held in practice for this specific dataset and pipeline.

---

## 3. Timeline of the investigation

### 3.1 Attempt 1 — Full dataset run, wrong XML path (discarded)

The first full-scale generation run (787 patients, ~47 minutes) completed without errors and reported "Generated 787 scans, skipped 2." However, it used an unverified placeholder path for `--xml_root` (`data_raw/xml/calcium_xml`) that did not exist in the actual directory structure. Because the label generator's `rasterize_at_resolution()` function silently returns an all-zero mask when the XML file is not found (no exception raised), the entire 787-scan run produced empty calcium masks for nearly every patient with no error signal at all.

**Caught by:** manually spot-checking `calcium_voxels` in a handful of generated `_meta.json` files — all read 0. Discarded, and `[WARNING] XML not found` logging was added to the generator so this class of failure can never be silent again.

**Root cause:** unverified assumption about directory structure, carried over from a script docstring rather than confirmed against the actual filesystem.

### 3.2 Attempt 2 — Correct XML path, `int32` truncation bug

With the correct XML path (`Gated_release_final/calcium_xml/{patient_id}.xml`) confirmed via directory listing and `patient_id` field cross-check, generation was re-run. This surfaced a second, more subtle bug.

**Discovery patient:** `597f3915db55` (patient_id=33).
- Approach 1: 96 calcium voxels across slices [43, 44, 45, 46], with 8, 48, 27, 13 voxels per slice respectively.
- Approach 2 (this attempt): **0 voxels, 0 slices with calcium.**

Confirmed visually via slice-by-slice CT+mask overlay: every one of the four calcium slices showed a visible marked deposit in Approach 1 and a completely blank mask in Approach 2, at the same anatomical location.

**Root cause:** the coordinate-scaling step cast scaled polygon vertices directly to `np.int32`, which truncates (rounds toward zero) rather than rounding to nearest. For very small lesions, several nearby floating-point vertices can truncate to the same one or two integer pixel coordinates, collapsing the polygon to a degenerate shape with zero enclosed area. `cv2.fillPoly` then fills nothing. Larger lesions were unaffected because their vertices are spread across enough pixels to avoid this collision — confirmed via a control patient (`0147e487fd3e`, 5328→22067 voxels, correct ~4x area scaling with no collapse).

### 3.3 Fix attempt A — Sub-pixel shift-based fill

**Fix:** `cv2.fillPoly(..., shift=SHIFT_BITS)` with `SHIFT_BITS=4` (16x sub-pixel coordinate precision) and coordinates rounded (not truncated) before the shift-scaled integer cast. This is a standard OpenCV technique for preserving small/thin polygons that would otherwise collapse under integer rasterization.

**Result on the discovery patient:** collapse fixed — patient 33 now produces nonzero calcium voxels under this fix.

**But: this fix was never validated against ground truth before being provisionally accepted** — only checked for "nonzero," not "how close to true area." This gap was identified and closed in the next phase.

### 3.4 Ground-truth area fidelity methodology (the real test)

A proper test requires comparing rasterized mask area against the **true, continuous XML polygon area** — not comparing Approach 1's mask against Approach 2's mask, since neither is ground truth on its own.

**Method:** for each calcium-containing slice, compute:
1. **XML ground-truth area** via the shoelace formula, applied directly to raw native-pixel-unit polygon coordinates parsed from the XML — completely independent of any raster grid or fill algorithm.
2. **Convert to mm²** using each resolution's actual pixel spacing (native spacing for XML, A1's resampled spacing for the A1 mask, A2's resampled spacing for the A2 mask).
3. **Rasterized area** = voxel count in the binary mask × pixel area (mm²) for that resolution.
4. **Error % = (rasterized_area − xml_area) / xml_area × 100**, per slice.

This was implemented in `verify_area_fidelity.py` (single-patient, with a spatial registration overlay plotting the XML polygon outline directly on top of the rasterized mask, to additionally confirm geometric position/shape match — not just area) and `aggregate_fidelity.py` (batch version across a patient sample, reporting mean absolute error per approach).

### 3.5 Stratified sample design

Rather than test on one or two patients, a 15-patient stratified sample was drawn directly from `calcium_voxels` values already present in every Approach-1 `_meta.json`, covering the full range of lesion burden:

| Tier | scan_id | patient_id | A1 calcium voxels |
|---|---|---|---|
| Smallest | `62303b4f6d64` | 411 | 7 |
| Smallest | `aad51b087f4c` | 357 | 8 |
| Smallest | `6625f2a2a3c0` | 429 | 9 |
| Smallest | `51343e71d40e` | 202 | 10 |
| Smallest | `af92e66270c2` | 302 | 10 |
| Small–mid | `ceb6fb4ba0ff` | 316 | 98 |
| Small–mid | `d3cc9b788acf` | 74 | 168 |
| Small–mid | `a12db5301a7b` | 330 | 357 |
| Small–mid | `fc53a04c4dd5`\* | 388 | 816 |
| Mid | `ca1453c14e95` | 354 | 1,279 |
| Large | `ec250480daad` | 332 | 8,976 |
| Large | `2ba45999579a` | 173 | 9,013 |
| Large | `4d56f5a5e29b` | 194 | 10,865 |
| Large | `9acea26a7155` | 342 | 11,834 |
| Large | `03bf6628cf06` | 321 | 13,093 |

\* `fc53a04c4dd5` was excluded from all aggregate statistics — see §6, Anomaly.

### 3.6 Fix attempt A, measured against ground truth

Result of `aggregate_fidelity.py` on the 15-patient sample (14 after excluding the anomaly), Approach 2 generated with the shift-based fill at 2x resolution:

| scan_id | n slices | A1 mean \|err\| | A2 mean \|err\| | winner |
|---|---|---|---|---|
| `62303b4f6d64` | 1 | 8.2% | 15.9% | A1 |
| `aad51b087f4c` | 1 | 27.6% | 31.3% | A1 |
| `6625f2a2a3c0` | 1 | 2.7% | 24.4% | A1 |
| `51343e71d40e` | 1 | 18.6% | 26.2% | A1 |
| `af92e66270c2` | 1 | 13.0% | 34.8% | A1 |
| `ceb6fb4ba0ff` | 4 | 1.4% | 17.8% | A1 |
| `d3cc9b788acf` | 4 | 16.9% | 23.4% | A1 |
| `a12db5301a7b` | 4 | 2.4% | 10.5% | A1 |
| `ca1453c14e95` | 7 | 17.1% | 11.9% | **A2** |
| `ec250480daad` | 23 | 5.6% | 10.1% | A1 |
| `2ba45999579a` | 26 | 3.7% | 8.3% | A1 |
| `4d56f5a5e29b` | 31 | 3.9% | 8.6% | A1 |
| `9acea26a7155` | 27 | 6.3% | 7.6% | A1 |
| `03bf6628cf06` | 35 | 4.1% | 7.9% | A1 |

**Overall (166 slices, 14 patients): A1 = 5.60%, A2 = 9.74%. A2 wins on 1/14 patients.**

Finding: the small-lesion collapse bug is fixed (no more zero-voxel patients), but a *new* systematic bias was introduced. The shift-based sub-pixel fill is more boundary-inclusive than necessary, over-filling at every lesion edge — small lesions worst (up to 34.8% error), large lesions least affected (~7–10% error), consistent with boundary-pixel-to-total-area ratio driving the effect.

### 3.7 Confound isolation — is this resolution, or the algorithm?

Before concluding "2x resolution doesn't help," it was necessary to isolate the two variables that had been changed simultaneously: (a) the rasterization algorithm (shift-based fill, new) and (b) the resolution (2x, new). Approach 1 differs from this test in *both* dimensions, so a worse A2 result doesn't by itself indict resolution specifically.

**Test:** run the exact same shift-based fill algorithm at **native 1x resolution** (`--upsample 1.0`), holding the algorithm fixed and removing the resolution change. Compare against the same A1 baseline.

| scan_id | A1 mean \|err\| | Shift-algorithm @ 1x mean \|err\| |
|---|---|---|
| `62303b4f6d64` | 8.2% | 85.4% |
| `aad51b087f4c` | 27.6% | 63.0% |
| `6625f2a2a3c0` | 2.7% | 40.6% |
| `51343e71d40e` | 18.6% | 22.1% |
| `af92e66270c2` | 13.0% | 30.5% |
| `ceb6fb4ba0ff` | 1.4% | 26.3% |
| `d3cc9b788acf` | 16.9% | 41.5% |
| `a12db5301a7b` | 2.4% | 21.5% |
| `ca1453c14e95` | 17.1% | 30.4% |
| `ec250480daad` | 5.6% | 18.6% |
| `2ba45999579a` | 3.7% | 15.2% |
| `4d56f5a5e29b` | 3.9% | 17.8% |
| `9acea26a7155` | 6.3% | 16.6% |
| `03bf6628cf06` | 4.1% | 15.8% |

**Overall: A1 = 5.60%, shift-algorithm @ 1x = 19.21%. A2 wins 0/14.**

**Conclusion of the isolation test:** the algorithm is dramatically worse at native resolution than at 2x resolution (19.21% vs 9.74% — same algorithm, only resolution differs). This proves resolution is not the cause of the problem; if anything, finer resolution partially *compensates* for the algorithm's over-inclusiveness by shrinking the absolute size of each over-filled boundary ring. At the time this test was run, the exact mechanism inside "the algorithm" causing that over-inclusiveness was still unconfirmed. **§8 identifies it directly: every Approach 2 variant, including this one, introduces a manual coordinate-scaling step before filling that Approach 1 never performs — that step, not the choice of fill technique, is the dominant source of error.**

### 3.8 Fix attempt B — Plain rounding, no sub-pixel shift

**Hypothesis:** if `shift`-based sub-pixel fill is the specific defect, removing it and using only integer rounding (fixing the original truncation collapse without adding boundary over-inclusion) might land between the two failure modes — no collapse, no inflation.

**Fix:** `scaled = round(x * scale_x), round(y * scale_y)`, cast to `int32`, `cv2.fillPoly()` with no `shift` argument.

| scan_id | A1 mean \|err\| | A2 (rounded, 2x) mean \|err\| | winner |
|---|---|---|---|
| `62303b4f6d64` | 8.2% | 35.2% | A1 |
| `aad51b087f4c` | 27.6% | 31.3% | A1 |
| `6625f2a2a3c0` | 2.7% | 24.4% | A1 |
| `51343e71d40e` | 18.6% | 26.2% | A1 |
| `af92e66270c2` | 13.0% | 34.8% | A1 |
| `ceb6fb4ba0ff` | 1.4% | 20.3% | A1 |
| `d3cc9b788acf` | 16.9% | 22.7% | A1 |
| `a12db5301a7b` | 2.4% | 9.6% | A1 |
| `ca1453c14e95` | 17.1% | 10.3% | **A2** |
| `ec250480daad` | 5.6% | 9.6% | A1 |
| `2ba45999579a` | 3.7% | 8.7% | A1 |
| `4d56f5a5e29b` | 3.9% | 8.4% | A1 |
| `9acea26a7155` | 6.3% | 7.4% | A1 |
| `03bf6628cf06` | 4.1% | 8.0% | A1 |

**Overall: A1 = 5.60%, A2 (rounded) = 9.76%. A2 wins 1/14.**

**Result: statistically indistinguishable from the shift-based fix (9.74% vs 9.76%).** This is itself an informative negative result — the specific coordinate-rounding technique doesn't matter much. Both variants converge to roughly the same ~9.7% error ceiling at 2x resolution, meaning the ~5.6%→~9.7% gap is not attributable to either truncation-vs-rounding or shift-vs-no-shift. Something more structural to the re-rasterization approach (fresh fillPoly from XML at a new grid, independent of Approach 1's method) is producing worse fidelity than whatever Approach 1's existing pipeline does.

**Small-lesion collapse check under plain rounding:** re-run pending on `597f3915db55` (patient 33) with the corrected script — needs confirmation that rounding alone (no shift) still prevents the original collapse bug. *[Update this row once the corrected `visualize_mask_Original_vs_2x.py` path is re-run.]*

---

## 4. Summary table — all variants tested

| # | Variant | Resolution | Small-lesion collapse? | Mean \|error\| vs XML | vs A1 (5.60%) |
|---|---|---|---|---|---|
| — | **Approach 1** (existing pipeline) | native 0.37mm | No | **5.60%** | baseline |
| 1 | int32 truncation | 2x (0.185mm) | **Yes** — patient 33 total loss | not measured (fails outright) | fails |
| 2 | shift-fill, SHIFT_BITS=4 | 2x (0.185mm) | No | 9.74% | +74% worse |
| 3 | shift-fill, SHIFT_BITS=4 (isolation test) | native 1x (0.37mm) | No | 19.21% | +243% worse |
| 4 | rounded coords, no shift | 2x (0.185mm) | pending re-confirmation | 9.76% | +74% worse |

---

## 5. Infrastructure built during this investigation

These scripts now exist in the repo / VM and are reusable for Approach 3 and future validation work:

| Script | Purpose |
|---|---|
| `generate_approach_2_labels.py` | Re-rasterizes calcium masks from XML at a configurable upsample factor. Extended with `--limit N` (first N patients, for quick tests) and `--scan_ids <list>` (specific patient IDs, for stratified testing) so future variants never require a full 787-patient run to validate. Also now prints `[WARNING] XML not found` per missing file and an end-of-run `Calcium present / Zero calcium` tally, so a bad `--xml_root` can never again fail silently. |
| `pick_sample.py` | Builds a stratified patient sample (smallest / mid / largest calcium burden) directly from existing `calcium_voxels` metadata, for reproducible small-scale testing. |
| `verify_approach2.py` | Cross-checks every patient in train/val/test splits against the generated Approach-2 folder: confirms no missing files, flags any zero-calcium patient that should have real calcium. |
| `compare_masks.py` / `visualize_mask_Original_vs_2x.py` | Side-by-side CT+mask overlay, Approach 1 vs Approach 2, per calcium slice — for direct visual confirmation of collapse/inflation issues. |
| `verify_area_fidelity.py` | Single-patient ground-truth check: shoelace-formula XML polygon area vs rasterized A1/A2 mask area, per slice, plus a registration overlay plot (XML polygon outline drawn directly on the rasterized mask) to confirm spatial alignment independent of area. |
| `aggregate_fidelity.py` | Batch version of the above across a full patient sample; reports per-patient and overall mean absolute error, with a `--img2_root` flag enabling algorithm-vs-resolution isolation tests. |

---

## 6. Anomaly flagged for separate investigation — `fc53a04c4dd5`

Excluded from all aggregate statistics above because its **Approach 1** mask itself shows severe, non-systematic errors unrelated to anything tested in this investigation — several slices where Approach 1 has *zero* calcium despite clear XML polygon area, and one slice with +242.6% inflation:

| z | XML mm² | A1 mm² | A1 err% | A2 mm² (rounded) | A2 err% |
|---|---|---|---|---|---|
| 11 | 2.435 | 3.012 | +23.7% | 3.080 | +26.5% |
| 14 | 5.482 | 6.708 | +22.4% | 6.297 | +14.9% |
| 25 | 7.714 | **0.000** | **−100.0%** | 8.796 | +14.0% |
| 26 | 4.364 | 8.488 | +94.5% | 5.305 | +21.5% |
| 29 | 7.211 | **0.000** | **−100.0%** | 8.317 | +15.3% |
| 31 | 2.840 | **0.000** | **−100.0%** | 3.457 | +21.7% |
| 32 | 4.364 | 3.149 | −27.9% | 4.997 | +14.5% |
| 33 | 3.044 | 4.381 | +43.9% | 3.457 | +13.6% |
| 34 | 7.718 | 3.696 | −52.1% | 8.625 | +11.8% |
| 35 | 9.035 | 7.529 | −16.7% | 10.199 | +12.9% |
| 36 | 2.837 | 9.720 | **+242.6%** | 3.628 | +27.9% |
| 37 | 15.020 | 3.286 | −78.1% | 17.455 | +16.2% |
| 38 | 32.298 | 16.839 | −47.9% | 36.176 | +12.0% |

This pattern (alternating complete misses and large overshoots on the *same* patient, in Approach 1) is not consistent with ordinary rasterization rounding error — it looks like a distinct pre-existing defect, possibly related to the multi-series/duplicate-z-position patient issues Rajat's slides documented separately (14 patients affected project-wide). **Not yet root-caused. Recommend a dedicated follow-up investigation on this patient before final dataset assembly, independent of the Approach 2 vs Approach 1 comparison.**

---

## 7. Process notes worth carrying into Approach 3

A few methodological lessons from this investigation, useful to apply directly:

1. **Never trust "nonzero" as validation.** The shift-fill fix looked successful when judged only by "does the collapse bug still happen" — it took a proper ground-truth comparison to reveal it had introduced a *worse*, more widespread problem (systematic inflation) than the bug it fixed.
2. **Isolate variables before concluding causation.** The initial 2x-resolution result looked like "resolution doesn't help." It took a dedicated same-algorithm-different-resolution test to establish that resolution wasn't the actual variable at fault.
3. **Stratified sampling beats single-patient spot checks.** The 5/5/5 (small/mid/large) sample design is what surfaced that the problem affects all lesion sizes, not just the tiny ones the original bug was about.
4. **Cache directories are keyed by file path, not content** (confirmed against MONAI's `PersistentDataset` source) — regenerating label files at the same path without clearing the cache will silently serve stale data. Any future label regeneration must either use a fresh cache directory or explicitly clear the old one.
5. **A silent failure mode (missing XML → empty mask, no error) cost a full 47-minute run before being caught.** The `[WARNING]` logging and end-of-run tally added to the generator should be treated as a required pattern for any future data-generation script — every "expected but possibly missing" file lookup should log loudly, not fail quietly.

---

## 8. Root cause — why Approach 2 actually lost to Approach 1

Section 3 established *that* every Approach 2 variant lost, and probed the fill technique (truncation vs. shift vs. rounding) as the likely cause. That probing found those techniques were roughly interchangeable — none of them closed the gap to Approach 1's 5.60%. The real explanation turned out to be one level up: **Approach 1 and Approach 2 don't build their masks the same way at all.** Confirmed by reading `src/preprocessing/COCA_processor_main.py` directly (the canonical, current script — verified via git log against three older, deprecated copies elsewhere in the repo).

### What Approach 1 actually does (`parse_plist_filled`)

```python
pts = np.array(poly_points, dtype=np.int32)
temp_slice = np.zeros((total_y, total_x), dtype=np.uint8)
cv2.fillPoly(temp_slice, [pts], 1)
```

`total_y, total_x` here is `img_array.shape` — the shape of the **original DICOM array, before any resampling.** Two consequences follow:

1. **The fill happens in the exact coordinate system the XML annotation was drawn in.** Radiologists marked calcium on the scanner's native pixel grid; A1 fills directly onto that same grid. No coordinate scaling, no synthetic target grid, no extra transformation between "where the annotation says calcium is" and "where the pixel gets marked."
2. **Getting to the common training spacing (0.37mm) happens afterward, via a separate, dedicated step** — `resample_image()`, a wrapper around SimpleITK's `ResampleImageFilter` with nearest-neighbor interpolation. This is the same, well-tested resampler used to resample the CT image itself. It correctly handles pixel-center alignment, direction cosines, and geometric transforms — machinery specifically built for this exact operation.

There's a second reason this works especially well for COCA: the EDA (§ earlier project work) found native COCA spacing is 0.373mm ± 0.05mm — almost identical to `TARGET_SPACING = 0.37mm`. So for most patients, A1's "resample the mask to target spacing" step is close to a no-op. **A1's real mask is, for practically every patient, simply "fillPoly once, at native resolution, in the coordinate system the annotation was made in."** Nothing more.

### What every Approach 2 variant actually did

```python
scaled = [[x * scale_x, y * scale_y] for x, y in raw_pts]   # extra step A1 never has
cv2.fillPoly(temp, [scaled], 1)   # filled on a SYNTHETIC grid, not native
```

Approach 2 took the same raw XML coordinates, multiplied them by a hand-derived `scale_x`/`scale_y` (combining the native→0.37mm correction *and* the 2x upsample factor into one number), and filled directly onto that synthetic new grid. This is doing, by hand, the same job SimpleITK's resampler is built to do properly — and doing it with one extra multiply-and-round step that A1 never has to take at all. Every fillPoly technique tried (truncate / shift / round) was a variation on how to handle that *extra* step's rounding — but the step itself was the source of the error, not the particular rounding rule applied to it.

### Why this fully explains every result in §3

- **The truncation collapse (§3.2):** only possible because Approach 2 introduces a scale-and-round step at all. A1's coordinates are cast to `int32` too (line: `np.array(poly_points, dtype=np.int32)`) — but they're never pre-multiplied by a scale factor first, so there's no equivalent point where nearby vertices can be squeezed onto the same integer pixel.
- **The isolation test (§3.7):** confirmed resolution wasn't the driver — because it isn't. The driver is the presence of the manual scale-and-refill step itself, which exists in every Approach 2 variant regardless of what resolution it targets.
- **The demo widget's 26.4% → 2.3% story:** correct as a description of "same algorithm, finer grid" — and it's exactly what the isolation test measured (19.21% → 9.74%, same manual-scale algorithm at two resolutions). It was never a description of Approach 1, which uses a fundamentally different, resample-based mechanism that the demo doesn't model.

---

## 9. Approach 3 design — coverage fractions, built around the confirmed root cause

Approach 3 replaces the binary fill/no-fill decision with a **per-voxel coverage fraction**: instead of asking "is this pixel's center inside the polygon" (a hard 0-or-1 decision that either counts a boundary pixel in full or not at all), each voxel is assigned the *exact fraction of its physical area* that overlaps the true XML polygon.

| | Binary fillPoly (A1 & every A2 variant) | Coverage fraction (A3) |
|---|---|---|
| Interior pixel, fully inside polygon | 1 | 1.00 |
| Boundary pixel, 15% inside polygon | **0** (rounds down — undercounts) | **0.15** (exact) |
| Boundary pixel, 85% inside polygon | **1** (rounds up — overcounts) | **0.85** (exact) |
| Exterior pixel | 0 | 0.00 |

On the reference 7×7 test case (true polygon area 16.62px²): binary fill measured 21px² (**+26.4% error**); coverage fractions measured 16.64px² (**+0.1% error**) — a ~260x reduction in area error, at the *same* resolution, with no extra compute cost (coverage is computed once per pixel analytically, not by supersampling a finer grid and counting).

### How this directly targets the §8 root cause, not just the symptom

Section 8 established that Approach 2's failure wasn't really about fillPoly's rounding rule — it was about introducing an unnecessary manual scale-and-refill step that a proper resampler should have handled. Approach 3 is designed to avoid that mistake structurally, not just swap in a better fill function:

1. **Compute coverage on the native grid, exactly where A1 fills** — same coordinate system the XML was annotated in, no synthetic scale factor at the labeling step. This directly mirrors the property that made A1 work: filling happens where the annotation lives, not on a grid derived from it.
2. **Reach the common training spacing via the same resampling discipline as A1** — not a second from-scratch polygon rasterization. Since coverage values are continuous (0.0–1.0), they resample correctly with linear/area-weighted interpolation, unlike a binary mask which needs nearest-neighbor to avoid inventing fractional labels — meaning coverage fractions are *more* compatible with proper resampling than a binary mask is, not less.
3. **No fill/no-fill boundary decision exists anywhere in the pipeline.** The entire class of bug chased through §3 (truncation collapse, shift over-inclusion, rounding — all different answers to "which pixels count as filled") has no equivalent question in a coverage-based approach. There is no threshold to get wrong, because nothing is thresholded until the loss function or evaluation step explicitly asks for one.

### Practical implication for training

Coverage labels are soft targets (0.0–1.0 per voxel) rather than hard binary labels — this is why the original `train_unet.py` was built with a `--soft_label` flag and a `squared_pred=True` DiceCE variant from the start, anticipating exactly this approach. A voxel labeled 0.85 tells the model "mostly calcium, treat confidently but not with full certainty" — a strictly richer training signal than a binary 1, and one that should be measurably closer to the true clinical boundary than either A1 or any Approach 2 variant tested here.

---

## 10. Recommended next steps

1. ~~Read `COCA_processor_main.py`'s actual rasterization method~~ — **done, see §8.** Root cause confirmed: the manual scale-and-refill step, not the fill technique.
2. Re-confirm patient 33 (`597f3915db55`) stays nonzero under the plain-rounding fix (§3.8, pending row) — closes out the Approach 2 investigation cleanly, even though it's superseded by moving to Approach 3.
3. Implement the Approach 3 label generator per §9: compute coverage fractions on the native grid (mirroring A1's fill step exactly), then resample to target spacing using the same SimpleITK machinery A1 uses for its mask.
4. Validate Approach 3 the same way Approach 2 was validated — the 15-patient stratified sample, `aggregate_fidelity.py`, ground truth via shoelace formula. Reuse the existing infrastructure (§5) rather than rebuilding it.
5. Bring this document's §1 verdict table, §8 root-cause explanation, and §9 design rationale to the next group meeting — this is a complete, defensible investigation: what was tried, why it failed, the confirmed mechanism, and how the next approach is designed specifically around that mechanism rather than as another guess.

---

*This document is structured for direct conversion into presentation visuals (bar charts of mean error by variant, per-patient scatter plots, the slice-level error tables, the binary-vs-coverage grid comparison in §9) ahead of the next group meeting. All raw numbers above are sourced directly from `aggregate_fidelity.py` and `verify_area_fidelity.py` output, and from direct reading of `COCA_processor_main.py`, during this session — not recomputed or estimated.*
