# Approach 1: Final Testing Report

This report summarizes the performance of the two binary (Approach 1) baseline models evaluated on the `test_split.parquet` dataset consisting of 66 unseen patients.

## 1. Validation vs. Test Generalization

| Model | Best Validation Dice (Training) | Test Set Dice (Unseen Data) |
| :--- | :--- | :--- |
| **A1_Full_Volume** | 0.610 | **0.640** |
| **A1_ROI_Cropped** | 0.652 | **0.669** |

**Conclusion:** The models exhibit excellent generalization. The test scores being slightly higher than validation scores indicates the models did not overfit the training data and perform robustly on entirely unseen scans.

## 2. Volumetric Error Analysis

Because standard Dice scores do not perfectly correlate with clinical Agatston scoring, we measure the models' ability to predict absolute calcium volume.

| Model | Dice Score | Absolute Error (MAE) mm³ | Bias (Mean Error) mm³ |
| :--- | :--- | :--- | :--- |
| **A1_Full_Volume** | 0.640 | 249.46 | -199.89 |
| **A1_ROI_Cropped** | **0.669** | **171.30** | **-32.46** |

**Conclusion:** 
1. The `A1_Full_Volume` model severely underestimates calcium volume on average, carrying a heavy negative bias of -199.89 mm³.
2. Cropping the input exclusively to the heart bounding box (`A1_ROI_Cropped`) resolved the severe under-prediction issue, reigning the bias into a very tight -32.46 mm³ and drastically improving the absolute volume error.

## 3. Visualizations

The generated evaluation plots have been saved in this directory:
- `volume_scatter_comparison.png`: Highlights that ROI-cropped predictions sit much closer to the perfect-prediction identity line.
- `bland_altman_comparison.png`: Confirms the bias improvement of the ROI-cropped model.
- `mae_comparison_bar.png`: Summarizes the mean absolute volume error for both approaches.

## Next Steps

Once the **Approach 3** (Continuous Coverage Fraction) model finishes training, it will be evaluated using the identical `evaluate_models.py` script. Its metrics and plots will be appended to this directory to directly verify if sub-pixel modeling further reduces the Volumetric MAE below the 171 mm³ baseline established here.
