## Last Updated on: 17/06/2026 5:30PM IST

### Branch: `PrediCT/segmentation_rajat`

**CAC Scores** were not present in the DICOM or XML files directly, so calculated them from area and HU values using a custom script. Resampling will also shift scores slightly so need to decide whether to recalculate from binary masks post-resample.

**Masks**: each patient now has a binary mask and a multilabel mask (RCA, LCA, etc. from XML labels). Unknown artery labels are marked as `5`. Found a bug where `fillPoly` inflates mask area slightly beyond the actual annotation, needs fixing.

**Resampling**: going with `0.375 × 0.375 × 3` mm voxel spacing. Having uniform spacing for the long term goal of the PrediCT Project is important. Analysis from Aditya's Radiomics project confirms results are invariant to spacing, but we will still maintain this throughout the project.

**789 patients total**: only a subset has segmentation masks and/or calcium scores. Open question: patients without segmentation files — true negatives or just unannotated? Marking them as 0 could mislead the model if deposits exist but weren't annotated. Needs investigation — other papers do not include unannotated samples at all.

**Old COCA scripts** fail for 2 patients (`b41d81f0bd53`, `ca1a9ce04bbd`) — both have 2 DICOM series that combine into just 1 Z-slice (internal IDs 763 and 135). XML exists but no calcium score for patient IDs 268 and 135.

**pre_process.py** has been configured to generate ROI masks, which can be cached for faster inference. These masks will be used in the pipeline for ROI cropping and as an input channel. Also splits the dataset into train (0.7) / val (0.15) / test (0.15). Ablation studies planned, stay tuned.

**dataset.py** has been configured to include CoordConv channels, Heart ROI masks, Dual HU Windowing for both Calcium and Soft Tissue, and Persistent Caching to speed up training.

**All global variables and hyperparameters can be tuned from `config.py**

Ablation flags: `HEART_MASK_FLAG`, `ADD_HEART_MASK_CHANNEL`, `ADD_COORD_CHANNELS`, 'DUAL_HU_WINDOWING'

**Cited Papers** — Refer to `cited_papers.md` for more info.
- [Standardization of Coronary Artery Calcification Scoring](https://www.jacc.org/doi/10.1016/j.jcmg.2022.02.026)
- [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://papers.nips.cc/paper_files/paper/2018/file/60106888f8977b71e1f15db7bc9a88d1-Paper.pdf)
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/pdf/2010.08895)

---

*Built as part of [PrediCT](https://ml4sci.org/gsoc/2026/proposal_PREDICT1.html) — ML4SCI x GSoC 2026. All rights reserved.*
