# Phase 3: Stochastic Plaque Growth

## Purpose
Phase 3 takes the low-ESS locations identified in Phase 2 and stochastically grows 3D calcium lesions onto the vessel wall.

## Methodology
To ensure infinite, biologically realistic variability, the pipeline avoids hardcoded shapes and instead relies on clinical statistical distributions derived from the COCA dataset:
1. **Lesion Count (Negative Binomial):** We sample the number of independent seeds to place on the vessel wall using a right-skewed Negative Binomial distribution ($n=4, p=1/3$).
2. **Plaque Size (Log-Normal):** Each individual seed is assigned a unique "voxel budget" drawn from a Log-Normal distribution ($\mu=4.3, \sigma=1.2$). This ensures plaques within the same patient are highly heterogeneous in size.
3. **Anisotropic Growth (Breadth-First Search):** The plaques physically grow voxel-by-voxel using a BFS algorithm governed by an exponential decay probability ($P = e^{-d/\lambda}$). 
   - We use a **High $\lambda$ (Slow decay)** for longitudinal/circumferential growth so the plaque spreads wide along the wall.
   - We use a **Low $\lambda$ (Fast decay)** for radial growth to strictly prevent the plaque from growing unnaturally inward and blocking the blood lumen.

## Key Files
* `run_phase3_sde.py`: The core script that samples the distributions, seeds the wall, and executes the anisotropic BFS growth to output the raw 1mm binary calcium mask.
