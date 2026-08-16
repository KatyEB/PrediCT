"""
scoring.py — Agatston scoring from a probability volume.

Takes a model's output probability map plus the raw HU volume and produces one
row per detected lesion. Patient-level totals are a sum over those rows, never
a separate calculation.

The A1 vs A3 difference lives here and nowhere else:
    binary   (A1): lesion area = voxel count x pixel_area
    coverage (A3): lesion area = sum(probabilities) x pixel_area  (never thresholded)
That single line is the scientific claim of this project.

Components are found per slice, which is how Agatston is defined. grouping.py
then labels which of those components belong to the same 3D lesion. That
labelling is descriptive only — no area, weight or total depends on it.

Does NOT: load models, read files, or know about HU windows.
Called by: run.py, after predict().

Usage:
    rows    = score(prob, hu, spacing, mode="coverage", threshold=0.1)
    summary = totals(rows)
    groups  = lesion_3d_table(rows, spacing[2])   # from grouping.py
"""
import numpy as np
import scipy.ndimage as ndimage

from src.backend.grouping import link_lesions

WEIGHTS = [(130, 0), (200, 1), (300, 2), (400, 3)]

def density_weight(peak_hu: float) -> int:
    for lim, w in WEIGHTS:
        if peak_hu < lim: 
            return w
    return 4

def score(prob: np.ndarray, hu: np.ndarray, spacing: tuple, mode: str, threshold: float,
          min_area_mm2: float = 1.0, max_gap_slices: int = 0) -> list[dict]:
    """Score every lesion in a probability volume.

    Args:
        prob:    (Z, Y, X) float32 in [0, 1]. Model output, already activated.
        hu:      (Z, Y, X) raw Hounsfield Units — NOT the normalised array.
                 Density weights are defined on true HU values.
        spacing: (sx, sy, sz) in mm — SimpleITK order, the REVERSE of the
                 array axis order above.
        mode:    "binary" | "coverage"
        threshold: minimum probability to trigger connected component building.
        min_area_mm2: minimum area to flag as included.
        max_gap_slices: how far a 3D lesion link may jump in z. 0 = strictly
                 adjacent slices. See grouping.py. Affects lesion_3d_id only,
                 never any score.

    Returns:
        list[dict], one row per PER-SLICE component. Sub-threshold components
        are included and flagged included=False, so conformant and
        non-conformant totals both come from one inference run.

        Two identities per row:
          slice_lesion_key   z024_c01  — this component, on this slice
          lesion_3d_key      L004      — the 3D lesion it belongs to
        lesion_id stays a flat running counter, unchanged, for compatibility.
    """
    sx, sy, sz = spacing
    
    # Agatston is defined per 3 mm slice with no thickness term. Because we resample
    # to exactly 3.0 mm that factor is 1.0 and is omitted. If spacing ever changes,
    # every score silently scales by sz/3 — hence the assert.
    assert abs(sz - 3.0) < 1e-6, f"Agatston assumes 3.0 mm slices, got {sz}"
    
    pixel_area = sx * sy
    struct = ndimage.generate_binary_structure(2, 1)
    rows = []
    planes = {}          # slice_idx -> labelled array, kept for 3D linking below
    lid = 0
    for z in range(prob.shape[0]):
        plane = prob[z] > threshold
        if not plane.any():
            continue
            
        labelled, n = ndimage.label(plane, structure=struct)
        planes[z] = labelled
        for lab in range(1, n + 1):
            ys, xs = np.nonzero(labelled == lab)
            
            if mode == "binary":
                area_mm2 = ys.size * pixel_area
            elif mode == "coverage":
                area_mm2 = float(prob[z][ys, xs].sum()) * pixel_area
            else:
                raise ValueError(f"unknown mode: {mode}")
                
            peak_hu = float(hu[z][ys, xs].max())
            w = density_weight(peak_hu)
            lid += 1
            
            rows.append(dict(
                slice_idx=z, 
                z_mm=round(z * sz, 2), 
                lesion_id=lid,
                label_2d=lab,                             # component index within this slice
                slice_lesion_key=f"z{z:03d}_c{lab:02d}",  # stable per-slice identity
                area_mm2=area_mm2, 
                peak_hu=peak_hu, 
                density_weight=w,
                agatston=area_mm2 * w,
                centroid_x=float(xs.mean()), 
                centroid_y=float(ys.mean()),
                bbox_x0=int(xs.min()), 
                bbox_y0=int(ys.min()),
                bbox_x1=int(xs.max()), 
                bbox_y1=int(ys.max()),
                n_voxels=int(ys.size),
                mean_coverage=float(prob[z][ys, xs].mean()),
                included=bool(area_mm2 >= min_area_mm2 and w > 0),
            ))

    # Adds lesion_3d_id / lesion_3d_key to every row. Nothing above this line is
    # touched, so the total cannot move; run.py asserts that anyway.
    link_lesions(planes, rows, max_gap_slices)
    return rows

def totals(rows: list[dict]) -> dict:
    """Compute patient-level totals from a list of lesion rows."""
    kept = [r for r in rows if r["included"]]
    total = sum(r["agatston"] for r in kept)
    
    cat = ("zero" if total == 0 else "mild" if total <= 100
           else "moderate" if total <= 400 else "severe")
           
    return dict(
        agatston_total=total, 
        risk_category=cat,
        n_lesions=len(kept), 
        n_lesions_all=len(rows),
        n_lesions_3d=len({r["lesion_3d_key"] for r in kept}),
        n_lesions_3d_all=len({r["lesion_3d_key"] for r in rows}),
        slices_with_calcium=sorted({r["slice_idx"] for r in kept})
    )
