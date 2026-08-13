"""
render.py — Visualization of prediction overlays.

Generates separate PNG slices for the raw CT and the model's prediction mask.
Outputs raw sizes without interpolation using PIL directly.

Does NOT: load files, calculate scores, or do inference.
Called by: run.py (only if save_png=True).

Usage:
    save_slices(ct_array, prob_array, out_dir, manifest, lesion_rows)
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Window for HUMAN VIEWING, distinct from model HU window in the manifest
DISPLAY_WINDOW_HU = (-100, 400)

def save_slices(ct_array: np.ndarray, prob: np.ndarray, out_dir: Path, manifest: dict, lesion_rows: list[dict]):
    """Save PNG images using PIL directly for exact pixel alignment and speed."""
    print(f"Saving ALL PNG slices to {out_dir} ...")
    
    ct_dir = out_dir / "ct"
    mask_dir = out_dir / "mask"
    ct_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    num_slices = prob.shape[0]
    spacing_z = manifest["spacing"][2]
    mode = manifest["output"]
    threshold = manifest["threshold"]
    
    slices_meta = []
    
    # Pre-group lesions by slice_idx for O(1) lookup
    lesions_by_slice = {}
    for r in lesion_rows:
        lesions_by_slice.setdefault(r["slice_idx"], []).append(r)
    
    for z in range(num_slices):
        # 1. Process CT Image (Mode "L")
        # render.py — display only. diag(-1,-1,1) means increasing y is anterior and
        # increasing x is patient-right. flipud puts anterior at the top; fliplr puts
        # patient right on the viewer's left (radiological convention, how cardiac CT
        # is read). Neither affects any number: scoring.py reads the unflipped array
        # and pred.nii.gz is saved unflipped. lesions.csv stays in array coordinates.
        ct_slice = np.fliplr(np.flipud(ct_array[z]))
        
        # Window and normalize to 0-255 uint8
        ct_windowed = np.clip(ct_slice, DISPLAY_WINDOW_HU[0], DISPLAY_WINDOW_HU[1])
        ct_norm = ((ct_windowed - DISPLAY_WINDOW_HU[0]) / (DISPLAY_WINDOW_HU[1] - DISPLAY_WINDOW_HU[0]) * 255).astype(np.uint8)
        
        Image.fromarray(ct_norm, mode="L").save(ct_dir / f"slice_{z:03d}.png")
        
        # 2. Process Mask Image (RGBA)
        p_slice = np.fliplr(np.flipud(prob[z]))
        
        # Create RGB channels (201, 84, 31)
        r = np.full_like(p_slice, 201, dtype=np.uint8)
        g = np.full_like(p_slice, 84, dtype=np.uint8)
        b = np.full_like(p_slice, 31, dtype=np.uint8)
        
        # Alpha channel based on mode
        if mode == "coverage":
            alpha = np.round(p_slice * 255).astype(np.uint8)
        elif mode == "binary":
            alpha = np.where(p_slice > threshold, 255, 0).astype(np.uint8)
        else:
            raise ValueError(f"unknown mode: {mode}")
            
        rgba = np.stack([r, g, b, alpha], axis=-1)
        Image.fromarray(rgba, mode="RGBA").save(mask_dir / f"slice_{z:03d}.png")
        
        # 3. Collect metadata
        slice_rows = lesions_by_slice.get(z, [])
        included_rows = [r for r in slice_rows if r["included"]]
        
        z_mm = slice_rows[0]["z_mm"] if slice_rows else z * spacing_z
        
        # Bins for the COVERAGE ON THIS SLICE panel. Lower edge is the component
        # threshold: voxels below it are not part of any lesion.
        COVERAGE_BINS = [0.1, 0.25, 0.5, 0.75, 1.01]
        
        slices_meta.append({
            "idx": z,
            "z_mm": round(z_mm, 2),
            "n_lesions": len(included_rows),
            "n_lesions_all": len(slice_rows),
            "slice_score": sum(r["agatston"] for r in included_rows),
            "has_calcium": len(included_rows) > 0,
            "coverage_hist": [int(n) for n in np.histogram(p_slice[p_slice > threshold], bins=COVERAGE_BINS)[0]]
        })
        
    with open(out_dir.parent / "slices.json", "w") as f:
        json.dump(slices_meta, f, indent=2)
