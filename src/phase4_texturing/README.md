# Phase 4: Texturing & Alpha Blending

## Purpose
Phase 4 transforms the raw, blocky binary calcium mask from Phase 3 into photorealistic, sub-voxel synthetic CT anatomy.

## Methodology
Raw voxel generation suffers from harsh "staircase" pixelation on a CT grid. We resolve this using a two-step texturing and blending process:

1. **CT Texture Generation (Normal Distributions):**
   - Each patient draws a baseline Hounsfield Unit (HU) density from a Normal distribution ($\mu=850, \sigma=100$) calibrated for Contrast-Enhanced CT (CCTA) visibility.
   - Within each plaque, every individual voxel draws a variance from a secondary Normal distribution ($\mu=150, \sigma=30$) to perfectly simulate internal CT quantum noise and tissue heterogeneity.

2. **Dual-Stage Alpha Blending (Blooming Artifact):**
   - We apply a coarse Gaussian blur ($\sigma=0.8$mm) on the 1mm grid to establish a fading intensity gradient.
   - We apply an ultra-fine Gaussian blur ($\sigma=0.6$mm) on the resampled 0.375mm native CT grid to act as a flawless anti-aliasing filter.
   - This blurred spatial gradient is then used as an **Alpha Channel** to mathematically mix the intense calcium brightness with the underlying heart tissue.

## Result
The hard edges vanish. The solid calcium core degrades organically outward into the surrounding tissue, perfectly replicating the natural blooming artifact produced by the Point Spread Function (PSF) of clinical CT scanners.

## Key Files
* `run_phase4_texture.py`: The execution script that applies the dual-stage Gaussian blurs, generates the statistical HU noise, and blends the final synthetic calcium directly into the patient's original CT scan.
