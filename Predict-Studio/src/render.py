"""
render.py — Visualization of prediction overlays.

Generates PNG slices showing the raw CT alongside the model's prediction.
Supports both flat coloring (for binary masks) and alpha mapping (for soft coverage).

Does NOT: load files, calculate scores, or do inference.
Called by: run.py (only if save_png=True).

Usage:
    save_slices(ct_array, prob_array, out_dir, mode="coverage")
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy.ma as ma

def save_slices(ct_array: np.ndarray, prob: np.ndarray, out_dir: Path, mode: str):
    """Save PNG overlays using Matplotlib to match the visualizer reference.
    
    Args:
        ct_array: (Z, Y, X)
        prob:     (Z, Y, X)
        out_dir:  destination directory for PNGs
        mode:     "binary" or "coverage"
    """
    print(f"Saving ALL PNG slices to {out_dir} (mode={mode})...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for z in range(prob.shape[0]):
        # The visualizer clips from -100 to 400 for display
        ct_slice = np.clip(ct_array[z], -100, 400)
        p_slice = prob[z]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
        
        for ax in axes:
            ax.set_facecolor("black")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            
        # Panel 1: Original CT (origin='lower' aligns with radiology view)
        axes[0].imshow(ct_slice, cmap='gray', origin='lower', vmin=-100, vmax=400)
        
        # Panel 2: Soft Probability Overlay
        axes[1].imshow(ct_slice, cmap='gray', origin='lower', vmin=-100, vmax=400)
        
        if mode == "binary":
            # For binary mode, plot a flat overlay of everything > 0.5
            p_masked = ma.masked_where(p_slice < 0.5, p_slice)
            axes[1].imshow(p_masked, cmap='hsv', origin='lower', alpha=0.5, vmin=0, vmax=1)
        elif mode == "coverage":
            # Mask out probabilities near zero so they are transparent
            p_masked = ma.masked_where(p_slice < 0.01, p_slice)
            axes[1].imshow(p_masked, cmap='inferno', origin='lower', alpha=0.5, vmin=0, vmax=1)
        else:
            raise ValueError(f"unknown mode: {mode}")
            
        plt.tight_layout()
        plt.savefig(out_dir / f"slice_{z:03d}.png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
