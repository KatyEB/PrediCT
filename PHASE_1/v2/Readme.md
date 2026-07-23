# Version 2 Progress Report

## Overview

Version 2 focused on improving the robustness of the atlas registration pipeline by refining affine optimization, registration stability, ROI handling and quantitative validation. These changes increased the registration success rate to approximately **40%**, while significantly improving anatomical consistency compared to Version 1.

---

## Improvements over Version 1

- **Metric Mask-Based Affine Registration:** Replaced crop-based affine optimization with a cardiac ROI metric mask, allowing affine registration to operate in the full physical coordinate system while focusing optimization on the heart. 

- **Improved Affine Optimizer Stability:** Increased the affine learning rate, minimum step size and introduced relaxation to prevent premature convergence and allow the optimizer to reach better solutions. 

- **Physically Consistent Optimizer Scaling:** Replaced Jacobian-based optimizer scaling with physical-shift scaling, improving the balance between rotational and translational parameter updates during rigid registration. 

- **Improved Mutual Information Estimation:** Increased the Mattes Mutual Information histogram bins from 32 to 50, providing a more stable similarity metric during multi-resolution optimization. 

- **Random Sampling for Affine Registration:** Switched from regular to random metric sampling during the affine stage, improving optimization over sparse coronary structures and reducing sampling bias. 

- **Adaptive Cardiac ROI Generation:** Replaced a fixed voxel margin with a spacing-aware physical margin, producing more consistent cardiac ROIs across patients with varying heart sizes and scan resolutions. 

- **Reduced Computational Overhead:** Removed redundant intermediate resampling operations, simplifying the registration workflow and reducing unnecessary processing time. 

- **Improved Registration Validation:** Validation was refined using COCA ground-truth calcium annotations together with Euclidean Distance Transform (EDT) overlap analysis, providing a more reliable assessment of anatomical registration quality. 

---

## Outcome

- **Registration Success Rate:** ~40% of evaluated scans achieved successful anatomical registration.
- **Primary Improvement:** Vessel masks remained anatomically coherent without the severe stretching and deformation observed during earlier development.
- **Remaining Limitation:** Although registration stability improved substantially, anatomical alignment remained inconsistent across a significant portion of the cohort, motivating further refinement in subsequent versions.
