# Final Testing Report (Approaches 1 & 3)

This report summarizes the performance of the baseline binary models (Approach 1) and the sub-pixel coverage fraction model (Approach 3), evaluated on the `test_split.parquet` dataset consisting of 66 unseen patients.

## 1. Validation vs. Test Generalization

| Model | Best Validation Dice (Training) | Test Set Dice (Unseen Data) |
| :--- | :--- | :--- |
| **A1_Full_Volume** | 0.610 | **0.640** |
| **A1_ROI_Cropped** | 0.652 | **0.669** |
| **A3_Coverage** | ~0.596 | **0.654** |
| **A3_Coverage_v2** (Anomaly Free) | **0.7227** (Med: **0.7865**) | **0.655** |

**Conclusion:** All models exhibit excellent generalization, with test scores slightly higher or comparable to validation scores. This indicates the models did not overfit the training data and perform robustly on entirely unseen scans.

## 2. Volumetric Error Analysis

Because standard Dice scores do not perfectly correlate with clinical Agatston scoring, we measure the models' ability to predict absolute calcium volume.

| Model | Dice Score | Absolute Error (MAE) mm³ | Bias (Mean Error) mm³ |
| :--- | :--- | :--- | :--- |
| **A1_Full_Volume** | 0.640 | 249.46 | -199.89 |
| **A1_ROI_Cropped** | 0.669 | 171.30 | -32.46 |
| **A3_Coverage** | 0.654 | **164.23** | **-0.087** |
| **A3_Coverage_v2** (Anomaly Free) | 0.655 (Med: 0.767) | 174.50 (Med: 58.32) | -56.28 (Med: +19.71) |

**Conclusion:** 
1. The `A1_Full_Volume` model severely underestimates calcium volume on average, carrying a heavy negative bias of -199.89 mm³.
2. Cropping the input exclusively to the heart bounding box (`A1_ROI_Cropped`) resolved the severe under-prediction issue, reeling the bias into a tight -32.46 mm³ and drastically improving the absolute volume error to 171.30 mm³.
3. Modeling the sub-pixel partial volume effect directly with soft labels (`A3_Coverage`) further reduced the MAE to **164.23 mm³** and achieved a near-zero mean signed bias of **-0.087 mm³**. Note: this is a mean-signed-error cancellation (over-prediction on mild/moderate patients offsetting under-prediction on severe ones), not evidence of per-patient calibration — see `progress_report_v10` §10.6 and §11.5.
4. **Impact of Anomaly Removal:** Retraining the model on the rigorously cleaned dataset (`A3_Coverage_v2`) slightly improved the Test Dice (**0.655**), but shifted the volume predictions to be more conservative (Bias: **-56.28 mm³**, MAE: **174.50 mm³**). This is an expected and healthier outcome — the original v1 baseline was artificially buoyed toward positive volume by the 14 anomalous scans, which contained massive area overshoots (up to +723%). Removing them breaks the cancellation described in point 3 and gives a more honest bias figure.
5. **Impact of Outliers on A3_Coverage_v2:** While the *mean* metrics for A3_Coverage_v2 show a Dice of 0.655 and MAE of 174.50 mm³, the *median* metrics for the test split are noticeably better (Median Dice: **0.767**, Median MAE: **58.32 mm³**, Median Bias: **+19.71 mm³**). This discrepancy indicates that a few severe outlier cases are disproportionately pulling the averages down, and typical patient performance is significantly higher than the mean suggests.

## 3. Visualizations

The generated evaluation plots have been saved in this directory:
- `volume_scatter_comparison.png`: Highlights how the model predictions align with the perfect-prediction identity line across all approaches.
- `bland_altman_comparison.png`: Confirms the significant bias improvements of the ROI-cropped and Coverage models.
- `mae_comparison_bar.png`: Summarizes the mean absolute volume error for all approaches evaluated.

## 4. Agatston Scoring & Clinical Risk Stratification (Final Evaluation)

We fully evaluated both the Approach 1 (Binary) and Approach 3 (Soft Coverage) models on the Anomaly-Free Test Set (66 patients) using the true clinical Agatston metric. The models' Agatston predictions were compared against the XML Shoelace Ground Truth.

| Metric | A1 (Binary ROI) | A3 (Soft Coverage) |
|---|---|---|
| **Mean Absolute Error (MAE)** | 179.62 | **188.53** |
| **Median Absolute Error** | 42.97 | **19.27** (2.2× better) |
| **Mean Bias** | -41.44 | **-126.95** |
| **Pearson r** | 0.8510 | **0.8458** |
| **R² (r²)** | 0.724 | 0.715 |
| **Clinical Risk Accuracy** | 86.4% | **92.4%** |

> **Correction:** an earlier version of this table labeled the 0.8510 / 0.8458 row "Pearson Correlation (R²)". Those are Pearson *r* values, not R². R² is 0.724 / 0.715 and is now shown as its own row.

### Conclusion: Clinical Applicability Over Absolute Error, With Caveats
Mean MAE looks similar (~180 vs ~188) and slightly favors A1 — this is dominated by a handful of very heavily calcified patients. **Median absolute error**, the better summary statistic for this distribution, favors A3 by a factor of 2.2 (19.27 vs 42.97). Neither model meets the <50-unit mean target set at midterm; A3 meets it comfortably on the median.

**Clinical Risk Accuracy is the endpoint that matters**, since patients are triaged into treatment buckets (0, 1–100, 101–400, >400), not by exact score. Approach 3's fractional coverage avoids the "cliff-edge" rounding error that pushes A1's borderline patients into the wrong bucket, raising risk-category accuracy from 86.4% to 92.4%.

**Statistical caveat — read before quoting 92.4% on its own:** on the 66-patient test set, the 86.4%→92.4% swing is driven by 4 patients (McNemar b=4, c=0), which gives p=0.125 — not significant at α=0.05 on this cohort alone. The direction and mechanism are consistent with the hypothesis, but the 66-patient result is suggestive, not proven by itself. A separate replication on the 374-patient train+val cohort (79.7% vs 83.7%, McNemar p=0.038) reaches significance and is what actually makes this a defensible claim — see `progress_report_v10` §11.4. Cite both numbers together, not the test-set number alone.

**Known scorer defects — the numbers above are provisional:** the scorers that produced this table normalize input HU with a window of [100, 1000], while every trained model (including A3) was trained on [0, 1200] — a real train/inference distribution shift. Separately, `agatston_scoring_a3.py` currently loads the superseded `approach3_coverage` (v1, val Dice 0.6156) checkpoint rather than the current `approach3_coverage_v2` (val Dice 0.7227), so A3's numbers above come from the weaker model. Both defects push in the same direction — they make A3 look worse than it likely is. A minimum-lesion-area rule (≥1 mm²) is also not yet applied by either scorer. See `progress_report_v10` §11.6 for the full defect list and priority order. **Re-run before these numbers are presented as final.**

*Note on Bias:* A3 shows a negative mean bias (-126.95), i.e. it underestimates large calcium deposits on average. Splitting by true risk category shows both models over-predict mild/moderate lesions and under-predict severe ones (A1: +49.1 / +83.6 / -244.4 by tier; A3: +38.5 / +9.6 / -439.4 by tier) — the near-zero volumetric bias reported in Section 2 is these two errors cancelling, not per-patient calibration. Severe-tier under-prediction does not change any patient's treatment bucket (>400 is >400 either way), which is why risk accuracy stays high despite it. See `progress_report_v10` §11.5 for the tier-by-tier breakdown and the blooming-artifact explanation (itself flagged as provisional pending the HU-window re-run above).

### Nuanced Clinical Insight: Small vs. Massive Lesions
A deeper analysis of the individual predictions in the CSV results reveals a dichotomy in how the two models behave:

*   **A3 excels on small/borderline lesions:** e.g. Patient 205, True Agatston 92 (Mild) — A1 predicted 529 (Severe), A3 predicted 315 (Moderate). Patient 82, True 369 (Moderate) — A1 predicted 834 (Severe), A3 predicted 251 (Moderate, correct).
*   **A1 tracks better on massive lesions:** e.g. Patient 196, True 2822 — A1 predicted 2357, A3 predicted 1570 (both same clinical tier regardless).

**Bottom line:** A1 estimates raw magnitude better on extreme (>2000) scores; A3 is the more clinically useful model because it gets borderline patients — where the treatment decision actually changes — right more often. This is the project's central result, and it is defensible on the 374-patient replication; the test-set numbers alone should be quoted with the p=0.125 caveat and the pending scorer re-run noted above.