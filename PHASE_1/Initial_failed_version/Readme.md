## Initial Registration Pipeline Failure Analysis

The initial registration pipeline produced visually plausible results in some cases but failed to preserve anatomically consistent coronary geometry. The primary limitations identified during development were:

- **Limited Preprocessing:** The initial preprocessing lacked robust anatomical normalization, intensity standardization and adaptive cardiac ROI extraction, reducing registration robustness across patients.

- **Global Registration Bias:** Registration was performed over the full thoracic volume, allowing non-cardiac anatomy (lungs, ribs and spine) to influence the optimization rather than focusing on the coronary region.

- **Affine Initialization Instability:** Several registrations failed due to insufficient overlap during the affine stage, producing the SimpleITK *"All samples map outside moving image buffer"* error.

- **Transform Management:** Early versions required refinement of rigid-to-affine transform initialization and composition to maintain a consistent physical coordinate system.

- **Elastix Registration Artifacts:** Early experiments using Elastix produced anatomically unrealistic deformations, including stretched and distorted coronary vessel masks. This configuration proved unsuitable for the dataset and was replaced with a native SimpleITK registration workflow.

- **Distorted Vessel Geometry:** Registration inaccuracies propagated through the atlas warping process, producing elongated, disconnected and anatomically implausible coronary masks despite correct source segmentations.

- **Single-Atlas Limitation:** The original pipeline relied on a single atlas, limiting robustness to variations in coronary anatomy and patient morphology.

