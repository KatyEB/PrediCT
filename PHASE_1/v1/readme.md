## Version 1 Improvements

### 1. Hierarchical Rigid → Affine Registration

The registration pipeline was redesigned into a two-stage hierarchical optimization consisting of an initial rigid alignment followed by affine refinement. This improved optimization stability by allowing coarse global alignment before estimating local geometric transformations. 

---

### 2. Standardized Image Preprocessing

A dedicated preprocessing pipeline was introduced to perform LPS orientation normalization, isotropic resampling and Hounsfield Unit windowing before registration. This reduced variability between scans and provided consistent inputs for the optimizer.

---

### 3. Separation of Registration and Evaluation Resolution

The implementation introduced two independent processing paths:- a lower-resolution volume for registration (1.5 mm) and a high-resolution volume (1.0 mm) for final label transformation and evaluation. This reduced computational cost while preserving evaluation accuracy. 

---

### 4. Deterministic Multi-Resolution Registration

The registration process adopted a multi-resolution image pyramid together with fixed regular sampling and deterministic random seeds. This improved optimization reproducibility and reduced sensitivity to stochastic sampling behaviour. 

---

### 5. Modular Pipeline Architecture

The implementation was reorganized into modular components including preprocessing, registration, label transformation, validation and visualization. This significantly improved maintainability and simplified debugging of individual pipeline stages. 

---

### 6. Improved Label Propagation

Atlas vessel labels were transformed using nearest-neighbour interpolation after image registration, preserving binary vessel topology during resampling and reducing interpolation artefacts.

---

### 7. Quantitative Registration Validation

Instead of relying solely on visual inspection, the pipeline introduced EDT-based calcium overlap validation to quantitatively evaluate anatomical registration quality against COCA annotations. This enabled objective performance assessment across patients. 

---

### 8. Diagnostic Visualisation and Logging

Comprehensive diagnostic overlays, optimizer logging, execution timing, and failure reporting were incorporated to facilitate systematic debugging and performance monitoring during development. 






##Confirmed that the Vessels are not being Morphed into an Ugly Bunch like last time. 😭❤️✌️
