# Phase 2: Hemodynamics (PINN)

## Purpose
Phase 2 simulates the blood flow through the patient's specific coronary arteries to calculate **Endothelial Shear Stress (ESS)**. ESS is the primary biological driver of atherosclerosis; low-ESS zones are where plaque and calcium naturally develop.

## Methodology
Instead of using computationally prohibitive Computational Fluid Dynamics (CFD), we use **Physics-Informed Neural Networks (PINNs)**. 
1. The vessel geometry from Phase 1 is sampled for boundary coordinates (`geometry.py`).
2. The PINN (`network.py`) is trained to solve the Navier-Stokes equations (`physics.py`) governing blood flow.
3. The neural network learns the velocity field and pressure gradients inside the specific artery.
4. The spatial derivatives of this velocity field at the vessel wall are extracted to compute the ESS scalar field (`ess.py`).

## Output
A 3D spatial probability map. Regions of low ESS (< 1.0 Pa) are marked as highly atherogenic, telling Phase 3 exactly where calcium is biologically permitted to grow.

## Key Files
* `physics.py`: Navier-Stokes PDE implementation.
* `network.py`: The PyTorch PINN model.
* `ess.py`: The shear stress extraction logic.
* `run_phase2.py`: The execution script for training the PINN and outputting the ESS map.
