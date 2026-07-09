# Final Testing Report (Approaches 1 & 3)

This report summarizes the performance of the baseline binary models (Approach 1) and the sub-pixel coverage fraction model (Approach 3), evaluated on the `test_split.parquet` dataset consisting of 66 unseen patients.

## 1. Validation vs. Test Generalization

| Model | Best Validation Dice (Training) | Test Set Dice (Unseen Data) |
| :--- | :--- | :--- |
| **A1_Full_Volume** | 0.610 | **0.640** |
| **A1_ROI_Cropped** | 0.652 | **0.669** |
| **A3_Coverage** | ~0.596 | **0.654** |

**Conclusion:** All models exhibit excellent generalization, with test scores slightly higher or comparable to validation scores. This indicates the models did not overfit the training data and perform robustly on entirely unseen scans.

## 2. Volumetric Error Analysis

Because standard Dice scores do not perfectly correlate with clinical Agatston scoring, we measure the models' ability to predict absolute calcium volume.

| Model | Dice Score | Absolute Error (MAE) mm³ | Bias (Mean Error) mm³ |
| :--- | :--- | :--- | :--- |
| **A1_Full_Volume** | 0.640 | 249.46 | -199.89 |
| **A1_ROI_Cropped** | 0.669 | 171.30 | -32.46 |
| **A3_Coverage** | 0.654 | **164.23** | **-0.087** |

**Conclusion:** 
1. The `A1_Full_Volume` model severely underestimates calcium volume on average, carrying a heavy negative bias of -199.89 mm³.
2. Cropping the input exclusively to the heart bounding box (`A1_ROI_Cropped`) resolved the severe under-prediction issue, reeling the bias into a tight -32.46 mm³ and drastically improving the absolute volume error to 171.30 mm³.
3. Modeling the sub-pixel partial volume effect directly with soft labels (`A3_Coverage`) further reduced the MAE to **164.23 mm³** and achieved an incredible near-zero bias of **-0.087 mm³**.

## 3. Visualizations

The generated evaluation plots have been saved in this directory:
- `volume_scatter_comparison.png`: Highlights how the model predictions align with the perfect-prediction identity line across all approaches.
- `bland_altman_comparison.png`: Confirms the significant bias improvements of the ROI-cropped and Coverage models.
- `mae_comparison_bar.png`: Summarizes the mean absolute volume error for all approaches evaluated.
