# %% [markdown]
# # Experiment 005: SAM2 Pericardium Bootstrapping
#
# **Started:** 2026-05-15
# **Status:** Active
#
# Uses SAM2 (prompted with the heart bounding box) to trace the pericardial
# sac region on every axial slice of each score-1 COCA patient. The resulting
# mask separates epicardial fat (inside pericardium) from paracardial fat
# (outside pericardium in the cardiac ROI). These pseudo-labels will train nnU-Net.

# %% [markdown]
# ## Question
#
# Does SAM2.1 (base+), prompted with a heart bounding box, reliably segment
# the pericardial sac in COCA non-contrast CT? What does the Dice against manual
# or visual check look like on 3 pilot patients?

# %% [markdown]
# ## Hypothesis
#
# SAM2 with a heart bounding box should segment the largest enclosed structure —
# the heart + epicardial fat bounded by the pericardium — on most slices.
# We expect it to work well on clear slices (score-1 patients) and fail at basal
# and apical slices where the pericardium is harder to distinguish.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import (
    crop_to_roi,
    fat_mask_from_hu,
    get_cardiac_roi_bbox,
    load_coca_patient,
    window_hu,
)
from src.inference.sam2_predictor import load_sam2_predictor, predict_pericardium_volume

# %%
EXPERIMENT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
METRICS_PATH = EXPERIMENT_DIR / "metrics.json"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DATA_ROOT = REPO_ROOT / "data" / "COCA_dataset"
RESAMPLED_DIR = DATA_ROOT / "data_resampled"
HEART_MASKS_DIR = DATA_ROOT / "heart_masks"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

PERI_MASKS_DIR = REPO_ROOT / config["output"]["masks_dir"]
PERI_MASKS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Experiment: {config['experiment']['id']} — {config['experiment']['name']}")
print(f"SAM2 model: {config['sam2']['model_id']}")
print(f"Output: {PERI_MASKS_DIR}")

# %% [markdown]
# ## Load SAM2 Model

# %%
print("Loading SAM2...")
t0 = time.time()
predictor = load_sam2_predictor(
    model_id=config["sam2"]["model_id"],
    device=config["sam2"]["device"],
)
print(f"SAM2 loaded in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Pilot — 3 Patients
#
# Run the full pipeline on 3 patients first. Visually inspect the results
# before running all 14. This is the Decision Point 1 preview.

# %%
pilot_ids = config["data"]["score1_patients"][:config["data"]["pilot_patients"]]
print(f"Pilot patients: {pilot_ids}")

dil = (29, 29, 7)
win_lo = config["windowing"]["lo"]
win_hi = config["windowing"]["hi"]
box_pad = config["sam2"]["box_pad"]
min_heart_vox = config["sam2"]["min_heart_vox"]

pilot_results = []

for pid in pilot_ids:
    print(f"\nProcessing {pid}...")
    p = load_coca_patient(pid, resampled_dir=RESAMPLED_DIR, heart_masks_dir=HEART_MASKS_DIR)
    roi = get_cardiac_roi_bbox(p["heart_mask"], dilation_vox=dil, array_shape=p["ct"].shape)
    crop = crop_to_roi(p["ct"], roi, heart_mask=p["heart_mask"])
    ct_c = crop["ct_crop"]
    hm_c = crop["heart_mask_crop"]

    t1 = time.time()
    peri_mask = predict_pericardium_volume(
        ct_c, hm_c, predictor,
        box_pad=box_pad,
        min_heart_vox=min_heart_vox,
        win_lo=win_lo,
        win_hi=win_hi,
    )
    elapsed = time.time() - t1

    # Save mask as NIfTI (in ROI space; store ROI for reconstruction)
    out_dir = PERI_MASKS_DIR / pid
    out_dir.mkdir(exist_ok=True)
    nib_img = nib.Nifti1Image(peri_mask, affine=np.eye(4))
    nib.save(nib_img, out_dir / "pericardium_roi.nii.gz")

    # Quick stats
    n_slices = int((hm_c.sum(axis=(0, 1)) >= min_heart_vox).sum())
    n_pred = int((peri_mask.sum(axis=(0, 1)) > 0).sum())
    vol_ml = float(peri_mask.sum()) * np.prod(p["spacing"]) / 1000

    result = {
        "scan_id": pid,
        "time_s": round(elapsed, 1),
        "heart_slices": n_slices,
        "predicted_slices": n_pred,
        "pericardium_vol_ml": round(vol_ml, 1),
    }
    pilot_results.append(result)
    print(f"  {n_pred}/{n_slices} slices predicted | vol={vol_ml:.0f}mL | {elapsed:.1f}s")

# %% [markdown]
# ## Visualization — Pilot QC
#
# Show 3 slices per patient: SAM2 mask (blue) + fat mask (red) + mediastinal CT.
# This lets us see whether the pericardium boundary is captured correctly.

# %%
def qc_figure(pid, ct_c, hm_c, peri_mask, win_lo, win_hi, n_slices=3):
    """QC figure: CT background + pericardium mask overlay."""
    nz = ct_c.shape[2]
    z_indices = [max(0, nz//4), nz//2, min(nz-1, 3*nz//4)]

    fig, axes = plt.subplots(1, n_slices, figsize=(n_slices * 4, 4))
    fig.suptitle(f"{pid}  — SAM2 pericardium mask (blue) + fat voxels (red)", fontsize=9)

    for ax, z in zip(axes, z_indices):
        bg = window_hu(ct_c[:, :, z], lo=win_lo, hi=win_hi)
        fat = fat_mask_from_hu(ct_c[:, :, z])
        peri = peri_mask[:, :, z]

        # Compose overlay
        rgb = np.stack([bg, bg, bg], axis=-1)  # grayscale base
        # Pericardium region → blue tint
        rgb[:, :, 2] = np.clip(rgb[:, :, 2] + 0.4 * peri, 0, 1)
        # Fat pixels inside pericardium → red (epicardial)
        epi = fat & peri
        rgb[:, :, 0] = np.clip(rgb[:, :, 0] + 0.6 * epi, 0, 1)
        rgb[:, :, 1] = np.clip(rgb[:, :, 1] - 0.3 * epi, 0, 1)

        ax.imshow(rgb.transpose(1, 0, 2), origin="lower")
        n_epi = int(epi.sum())
        ax.set_title(f"z={z}  epi_fat={n_epi}px", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    return fig


# %%
for i, (pid, res) in enumerate(zip(pilot_ids, pilot_results)):
    p = load_coca_patient(pid, resampled_dir=RESAMPLED_DIR, heart_masks_dir=HEART_MASKS_DIR)
    roi = get_cardiac_roi_bbox(p["heart_mask"], dilation_vox=dil, array_shape=p["ct"].shape)
    crop = crop_to_roi(p["ct"], roi, heart_mask=p["heart_mask"])
    ct_c = crop["ct_crop"]
    hm_c = crop["heart_mask_crop"]
    peri_mask = nib.load(PERI_MASKS_DIR / pid / "pericardium_roi.nii.gz").get_fdata().astype(np.uint8)

    fig = qc_figure(pid, ct_c, hm_c, peri_mask, win_lo, win_hi)
    fname = FIGURES_DIR / f"fig01_qc_pilot_{i+1:02d}_{pid[:8]}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname.name}")

# %% [markdown]
# ## Evaluation — Pilot Stats

# %%
print("\nPilot results:")
for r in pilot_results:
    print(f"  {r['scan_id']}: {r['predicted_slices']}/{r['heart_slices']} slices, "
          f"vol={r['pericardium_vol_ml']}mL, {r['time_s']}s")

avg_time = sum(r["time_s"] for r in pilot_results) / len(pilot_results)
n_full = len(config["data"]["score1_patients"])
print(f"\nAvg time/patient: {avg_time:.1f}s")
print(f"Estimated full run ({n_full} patients): {avg_time * n_full / 60:.1f}min")

# %%
results = {
    "pilot_patients": len(pilot_ids),
    "full_score1_patients": n_full,
    "avg_time_s": round(avg_time, 1),
    "estimated_full_run_min": round(avg_time * n_full / 60, 1),
    "pilot_results": pilot_results,
}

with open(METRICS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Metrics saved: {METRICS_PATH}")

# %% [markdown]
# ## Interpretation
#
# **Check the QC figures:**
# - Is the blue region (pericardium mask) a reasonable outline of the heart?
# - Does it include the epicardial fat layer (red pixels inside blue)?
# - Does it exclude the paracardial fat (outside blue boundary)?
# - Are there slices where it completely fails (no mask, or mask covers wrong region)?
#
# **Decision gate:**
# - If ≥80% of heart slices have a plausible mask → proceed to all 14 patients
# - If 50–80% → tweak box_pad or add point prompts; re-run
# - If <50% → need a different prompting strategy (e.g., negative points inside heart)
#
# **Run all 14 patients:** Change `pilot_patients: 14` in config.yaml, then re-run.
