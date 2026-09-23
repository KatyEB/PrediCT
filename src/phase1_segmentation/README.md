# Phase 1: Segmentation & Masking

## Purpose
Phase 1 is responsible for isolating the coronary arteries from the raw CT scan. By generating a highly accurate 3D binary mask of the vessel tree, we establish the bounding geometry where hemodynamics (Phase 2) will be simulated and calcium (Phase 3) will physically grow.

## Methodology
The vessel mask is extracted to delineate the **intima** and **lumen**. This serves two critical biological functions in the pipeline:
1. **Fluid Dynamics Boundary:** It defines the exact shape of the pipe through which blood flows, necessary for calculating wall shear stress.
2. **Growth Constraint:** It acts as the absolute barrier ensuring that synthetic calcium deposits only grow on the vessel wall and do not bleed into the surrounding heart muscle or open lumen.

## Key Files
* `masking.py`: Contains the core logic for loading, thresholding, and extracting the 3D vessel geometry from the non-contrast or contrast-enhanced DICOM/NIfTI files.
