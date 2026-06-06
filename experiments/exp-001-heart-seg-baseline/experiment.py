# %% [markdown]
# # Experiment 001: Heart Segmentation Baseline — TotalSegmentator
#
# **Started:** 2026-05-14
# **Status:** Active
#
# We need a heart ROI mask for each COCA patient to constrain EAT segmentation
# to the cardiac region. This experiment validates TotalSegmentator as our
# heart segmentation tool on 3 pilot patients, measures inference time, and
# confirms the masks are suitable for the EAT pipeline.

# %% [markdown]
# ## Question
#
# Does TotalSegmentator (fast mode, heart class, MPS device) reliably segment
# the heart in COCA NIfTI scans in a reasonable time? Are the masks
# adequate ROIs for constraining pericardial fat segmentation?

# %% [markdown]
# ## Hypothesis
#
# TotalSegmentator's heart segmentation should work well on COCA non-gated CTs.
# Fast mode should be sufficient for ROI generation (we don't need precise chamber boundaries).
# Inference time should be under 60s/scan on M4 MPS.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

# %%
SEED = 42
np.random.seed(SEED)

EXPERIMENT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
REPO_ROOT = EXPERIMENT_DIR.parent.parent
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
METRICS_PATH = EXPERIMENT_DIR / "metrics.json"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

print(f"Experiment: {config['experiment']['id']} — {config['experiment']['name']}")

# %%
RESAMPLED_DIR = REPO_ROOT / config["data"]["resampled_dir"]
HEART_MASKS_DIR = REPO_ROOT / config["data"]["heart_masks_dir"]
SCAN_INDEX_PATH = REPO_ROOT / config["data"]["scan_index"]
HEART_MASKS_DIR.mkdir(parents=True, exist_ok=True)

scan_df = pd.read_csv(SCAN_INDEX_PATH)
print(f"COCA scan index: {len(scan_df)} patients")
print(scan_df.head(3))

# %%
# Verify TotalSegmentator + MPS
import torch
from totalsegmentator.python_api import totalsegmentator

device = config["totalsegmentator"]["device"]
if device == "mps" and not torch.backends.mps.is_available():
    device = "cpu"
    print("WARNING: MPS not available, falling back to CPU")
print(f"Device: {device}")
print(f"TotalSegmentator fast={config['totalsegmentator']['fast']}, "
      f"roi_subset={config['totalsegmentator']['roi_subset']}")

# %% [markdown]
# ## Data — Select Test Patients

# %%
# Pick 3 patients that exist in data_resampled/ for our pilot run
available = sorted([p.name for p in RESAMPLED_DIR.iterdir() if p.is_dir()])
test_patients = available[:config["data"]["test_n_patients"]]
print(f"Test patients: {test_patients}")

# Confirm input files exist
for pid in test_patients:
    img_path = RESAMPLED_DIR / pid / f"{pid}_img.nii.gz"
    assert img_path.exists(), f"Missing: {img_path}"
    print(f"  {pid}: {img_path.stat().st_size / 1e6:.1f} MB")

# %% [markdown]
# ## Run — TotalSegmentator on 3 Pilot Patients

# %%
def run_totalseg_heart(scan_id: str, resampled_dir: Path, output_base: Path,
                        task: str, fast: bool, roi_subset: list[str], device: str):
    """Run TotalSegmentator and return the heart mask + inference time."""
    img_path = resampled_dir / scan_id / f"{scan_id}_img.nii.gz"
    out_dir = output_base / scan_id
    out_dir.mkdir(parents=True, exist_ok=True)
    heart_out = out_dir / "heart.nii.gz"

    if heart_out.exists():
        print(f"  {scan_id}: cached")
        t = 0.0
    else:
        t0 = time.time()
        totalsegmentator(
            input=str(img_path),
            output=str(out_dir),
            task=task,
            fast=fast,
            roi_subset=roi_subset,
            device=device,
            quiet=True,
        )
        t = time.time() - t0
        print(f"  {scan_id}: {t:.1f}s")

    heart_img = nib.load(heart_out)
    return heart_img, t

# Download models on first call (progress shown in stdout)
print("Running TotalSegmentator on pilot patients (first run downloads models ~400MB)...")
pilot_results = {}
total_time = 0
for pid in test_patients:
    heart_img, elapsed = run_totalseg_heart(
        pid, RESAMPLED_DIR, HEART_MASKS_DIR,
        task=config["totalsegmentator"]["task"],
        fast=config["totalsegmentator"]["fast"],
        roi_subset=config["totalsegmentator"]["roi_subset"],
        device=device,
    )
    pilot_results[pid] = {"heart_img": heart_img, "time_s": elapsed}
    total_time += elapsed

avg_time = total_time / len(test_patients) if test_patients else 0
print(f"\nAverage inference time: {avg_time:.1f}s per patient")
print(f"Estimated time for all 787 patients: {avg_time * 787 / 3600:.1f} hours")

# %% [markdown]
# ## Evaluation — Sanity Checks on Heart Masks

# %%
def heart_mask_stats(heart_img: nib.Nifti1Image, scan_id: str) -> dict:
    """Compute basic stats on the heart mask."""
    arr = heart_img.get_fdata()
    vox_vol_ml = np.prod(heart_img.header.get_zooms()) / 1000.0  # mm³ → mL
    heart_voxels = (arr > 0).sum()
    heart_vol_ml = heart_voxels * vox_vol_ml

    # Bounding box
    coords = np.argwhere(arr > 0)
    if len(coords) > 0:
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        bbox_size_vox = bbox_max - bbox_min + 1
    else:
        bbox_min = bbox_max = bbox_size_vox = np.zeros(3)

    return {
        "scan_id": scan_id,
        "heart_vol_ml": float(heart_vol_ml),
        "heart_voxels": int(heart_voxels),
        "bbox_z_slices": int(bbox_size_vox[0]),
        "bbox_y_px": int(bbox_size_vox[1]),
        "bbox_x_px": int(bbox_size_vox[2]),
    }

stats_rows = []
for pid, res in pilot_results.items():
    s = heart_mask_stats(res["heart_img"], pid)
    stats_rows.append(s)
    print(f"  {pid}: {s['heart_vol_ml']:.0f} mL, bbox z={s['bbox_z_slices']} y={s['bbox_y_px']} x={s['bbox_x_px']}")

stats_df = pd.DataFrame(stats_rows)
print(f"\nPublished cardiac volume reference: ~400-900 mL total heart")
print(f"Our measurements: {stats_df['heart_vol_ml'].mean():.0f} mL (mean)")

# %% [markdown]
# ## Visualization

# %%
# Show 3 axial slices per patient: CT + heart mask overlay
fig, axes = plt.subplots(len(test_patients), 3, figsize=(13, 4 * len(test_patients)))
if len(test_patients) == 1:
    axes = axes[np.newaxis, :]

for row_i, pid in enumerate(test_patients):
    ct_img = nib.load(RESAMPLED_DIR / pid / f"{pid}_img.nii.gz")
    ct = ct_img.get_fdata()
    heart = pilot_results[pid]["heart_img"].get_fdata()

    # Find z-slices with heart tissue
    heart_slices = np.where(heart.sum(axis=(1, 2)) > 0)[0]
    if len(heart_slices) == 0:
        continue
    lo, mid, hi = heart_slices[0], heart_slices[len(heart_slices) // 2], heart_slices[-1]

    for col_i, z in enumerate([lo, mid, hi]):
        ax = axes[row_i, col_i]
        ct_slice = ct[z]
        heart_slice = heart[z]

        # Window for soft tissue
        vmin, vmax = -200, 300
        ct_norm = np.clip((ct_slice - vmin) / (vmax - vmin), 0, 1)
        overlay = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)
        overlay[heart_slice > 0] = [1.0, 0.4, 0.1]  # orange = heart

        ax.imshow(overlay, origin="lower")
        label = {lo: "inf", mid: "mid", hi: "sup"}[z]
        ax.set_title(f"{pid} z={z} ({label})", fontsize=8)
        ax.axis("off")

plt.suptitle("Heart masks (orange) — CT soft-tissue window", y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig01_heart_mask_overlays.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved fig01_heart_mask_overlays.png")

# %%
# Dilated ROI — show what the cardiac ROI looks like after dilation
dilation_mm = config["heart_roi"]["dilation_mm"]

def dilate_mask_mm(heart_img: nib.Nifti1Image, dilation_mm: float) -> nib.Nifti1Image:
    """Dilate the heart mask by dilation_mm using SimpleITK."""
    sitk_img = sitk.GetImageFromArray(heart_img.get_fdata().astype(np.uint8))
    zooms = [float(z) for z in heart_img.header.get_zooms()]
    sitk_img.SetSpacing(zooms)
    dil_radius_px = [int(np.ceil(dilation_mm / sp)) for sp in zooms]
    dilated = sitk.BinaryDilate(sitk_img, dil_radius_px)
    arr_dil = sitk.GetArrayFromImage(dilated)
    return nib.Nifti1Image(arr_dil, heart_img.affine, heart_img.header)

pid = test_patients[0]
roi_img = dilate_mask_mm(pilot_results[pid]["heart_img"], dilation_mm)
roi_arr = roi_img.get_fdata()
heart_arr = pilot_results[pid]["heart_img"].get_fdata()
ct = nib.load(RESAMPLED_DIR / pid / f"{pid}_img.nii.gz").get_fdata()

heart_slices = np.where(heart_arr.sum(axis=(1, 2)) > 0)[0]
z_mid = heart_slices[len(heart_slices) // 2]

vmin, vmax = -200, 300
ct_norm = np.clip((ct[z_mid] - vmin) / (vmax - vmin), 0, 1)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(f"{pid} — z={z_mid} — ROI dilation effect ({dilation_mm}mm)")

for ax, (label, data, color) in zip(axes, [
    ("CT only", None, None),
    ("Heart mask (orange)", heart_arr[z_mid], [1.0, 0.4, 0.1]),
    (f"ROI = heart + {dilation_mm}mm dilation (green)", roi_arr[z_mid], [0.2, 0.9, 0.2]),
]):
    overlay = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)
    if data is not None:
        overlay[data > 0] = color
    ax.imshow(overlay, origin="lower")
    ax.set_title(label, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig02_roi_dilation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved fig02_roi_dilation.png")

# %% [markdown]
# ## Evaluation

# %%
roi_vol_ml = float(roi_img.get_fdata().sum() * np.prod(roi_img.header.get_zooms()) / 1000.0)
heart_vol_ml_pilot = float(stats_df["heart_vol_ml"].mean())

results = {
    "device": device,
    "fast_mode": config["totalsegmentator"]["fast"],
    "pilot_patients": len(test_patients),
    "avg_inference_time_s": float(avg_time),
    "estimated_total_time_h": float(avg_time * 787 / 3600),
    "heart_vol_ml_mean": heart_vol_ml_pilot,
    "heart_vol_ml_min": float(stats_df["heart_vol_ml"].min()),
    "heart_vol_ml_max": float(stats_df["heart_vol_ml"].max()),
    "roi_vol_ml_pilot0": roi_vol_ml,
    "dilation_mm": dilation_mm,
    "masks_saved_to": str(HEART_MASKS_DIR),
}

with open(METRICS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("=== Key metrics ===")
for k, v in results.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## Interpretation
#
# **What to check:**
# - Heart volume in range 400–900 mL? Published reference for healthy adults ~400-600g (~400-600 mL).
#   Slightly larger is expected since TotalSegmentator includes blood pool.
# - Inference time per patient < 60s on MPS? If > 60s, need to think about overnight batch run.
# - Overlays: does the mask cover all 4 chambers in mid-slices?
# - After dilation: does the 20mm ROI capture the pericardium + paracardial fat region?
#
# **Decisions triggered:**
# - If heart vol or mask shape looks correct → proceed to batch all 787 via `src/inference/generate_heart_masks.py`
# - If fast mode produces artifacts → retry with fast=False on subset
# - If inference too slow for local machine → run batch job on cluster or Colab
#
# **Next step:** Run `python src/inference/generate_heart_masks.py` to generate all 787 masks.
# That script can be left running overnight. The notebook just validates the first 3.
