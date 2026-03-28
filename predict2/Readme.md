# PREDICT2: Radiomics Feature Extraction and Calcium Phenotype Discovery

## Overview
This project develops a feature extraction and phenotyping framework for coronary 
artery calcium (CAC) using the Stanford COCA dataset. We extract radiomic features 
from calcium segmentation masks, perform statistical analysis, and discover calcium 
phenotypes via unsupervised clustering.

## Dataset
- **Source**: Stanford COCA (Coronary Calcium and Chest CT)
- **Scans used**: 24 gated coronary CT scans with calcium segmentation masks
- **Format**: DICOM → converted to NIfTI (.nii.gz) via COCA pipeline

## Pipeline

### 1. Preprocessing
- DICOM series loaded using SimpleITK
- Calcium segmentation masks parsed from XML annotation files
- Agatston scores calculated per patient

### 2. Feature Extraction (PyRadiomics)
Extracted **107 radiomic features** per patient across 6 categories:

| Category | Features |
|----------|---------|
| Shape | 14 |
| First Order | 18 |
| GLCM (texture) | 24 |
| GLSZM | 16 |
| GLRLM | 16 |
| NGTDM | 5 |

### 3. Statistical Analysis

#### Spearman Correlation
- **81 out of 107 features** significantly correlated with Agatston score (p<0.05)
- Top feature: `glcm_JointEnergy` (r = -0.9885, p < 0.001)
- Strong texture features dominate correlation with calcium burden

#### Kruskal-Wallis Test
- **76 out of 107 features** significantly different across Agatston categories (p<0.05)
- Shape and intensity features most discriminative between Minimal/Mild groups

### 4. Calcium Phenotype Discovery (Clustering)
- Applied K-Means clustering (K=2 to 5) on PCA-reduced features
- **Best K=2** (Silhouette Score = 0.439)
- Cluster 0 (n=21): Minimal calcium burden
- Cluster 1 (n=3): Mild calcium burden
- Clusters perfectly align with Agatston categories

### 5. Feature Importance (Random Forest + SHAP)
Top discriminating features:
1. `gldm_LargeDependenceEmphasis`
2. `gldm_LargeDependenceHighGrayLevelEmphasis`
3. `ngtdm_Complexity`
4. `firstorder_Energy`
5. `glszm_SizeZoneNonUniformity`

## Key Results
- 107 radiomic features extracted from 24 patients
- 81/107 features show significant Spearman correlation with Agatston score
- 76/107 features pass Kruskal-Wallis significance test
- K=2 clustering perfectly separates calcium burden categories
- GLCM texture features most strongly associated with calcium severity

## Visualizations
| File | Description |
|------|-------------|
| `pca_clustering.png` | PCA plot colored by Agatston category and cluster |
| `spearman_plot.png` | Top 15 features by Spearman correlation |
| `agatston_distribution.png` | Score distribution and category counts |
| `correlation_heatmap.png` | Feature correlation heatmap |
| `feature_importance.png` | Random Forest feature importance |

## Output Files
| File | Description |
|------|-------------|
| `features.csv` | 107 radiomic features for 24 patients |
| `agatston_correct.csv` | Agatston scores per patient |
| `final_merged.csv` | Features merged with Agatston scores |
| `spearman_results.csv` | Full Spearman correlation results |
| `kruskal_results.csv` | Full Kruskal-Wallis test results |
| `top_features.csv` | Top 20 features by RF importance |

## Requirements
```
pip install pyradiomics SimpleITK pydicom scikit-learn 
pip install matplotlib seaborn pandas numpy scipy
```

## How to Run
```bash
# 1. Preprocess DICOM data
python COCA_scripts/COCA_scripts/COCA_pipeline.py

# 2. Run full analysis
python predict2/complete_analysis.py
```

## Author
Prince Bhadania
GSoC 2026 Applicant — ML4SCI / PREDICT Project