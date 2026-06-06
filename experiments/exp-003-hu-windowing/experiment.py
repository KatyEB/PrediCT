# %% [markdown]
# # Experiment 003: HU Windowing and Preprocessing for Fat Visibility
#
# **Started:** 2026-05-15
# **Status:** Active
#
# This experiment establishes the optimal input preprocessing for EAT segmentation.
# We compare several HU windowing strategies on COCA patients and evaluate which
# best exposes: (1) fat compartments (epi vs para), (2) the pericardial line — the
# critical boundary for compartment separation. The output informs what representation
# we feed to SAM2 in exp-004.

# %% [markdown]
# ## Question
#
# Which HU windowing strategy (or multi-channel combination) best reveals
# (a) fat pixels [-200, -30] and (b) the pericardial sac boundary? And does the
# cardiac ROI crop correctly isolate the region of interest?

# %% [markdown]
# ## Hypothesis
#
# The standard fat window [-200, -30] (ADR-001) clearly shows fat but washes out
# the pericardial line. A mediastinal window [-160, +240] should reveal the pericardium
# more clearly. A 2-channel input (fat + mediastinal) likely gives SAM2 the best of
# both. We expect the cardiac ROI crop to already remove most non-cardiac tissue.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add repo root to path
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

print(f"Experiment: {config['experiment']['id']} — {config['experiment']['name']}")
print(f"Repo: {REPO_ROOT}")

# %% [markdown]
# ## Data
#
# Load the 3 pilot COCA patients (those with heart masks from exp-001).

# %%
pilot_ids = config["data"]["pilot_scan_ids"]
patients = {}
for pid in pilot_ids:
    p = load_coca_patient(
        pid,
        resampled_dir=RESAMPLED_DIR,
        heart_masks_dir=HEART_MASKS_DIR,
    )
    patients[pid] = p
    print(
        f"{pid}: CT shape={p['ct'].shape}, spacing={p['spacing']}, "
        f"heart_mask={'yes' if p['heart_mask'] is not None else 'NO'}"
    )

# %%
# Get cardiac ROI crops for each patient
dil = tuple(config["cardiac_roi"]["dilation_vox"])
for pid, p in patients.items():
    if p["heart_mask"] is None:
        print(f"WARNING: {pid} has no heart mask — skipping ROI crop")
        p["roi"] = None
        continue
    roi = get_cardiac_roi_bbox(p["heart_mask"], dilation_vox=dil, array_shape=p["ct"].shape)
    crop = crop_to_roi(p["ct"], roi, heart_mask=p["heart_mask"])
    p.update(crop)
    p["roi"] = roi
    p["roi_shape"] = crop["ct_crop"].shape
    print(f"{pid}: full={p['ct'].shape} → ROI crop={p['roi_shape']}")

# %% [markdown]
# ## Run — Windowing Comparison
#
# Apply each window to the cropped CT and compare side-by-side.

# %%
windows = {
    "fat [-200,-30]":       {"lo": -200, "hi": -30},
    "fat_wide [-250,0]":    {"lo": -250, "hi": 0},
    "mediastinal [-160,240]": {"lo": -160, "hi": 240},
    "soft_tissue [-150,250]": {"lo": -150, "hi": 250},
}


def get_mid_slice(ct_crop: np.ndarray) -> int:
    """Return the through-plane index of the largest heart cross-section."""
    return ct_crop.shape[2] // 2


# %%
# Fig 1: 4-panel windowing comparison for one patient, one slice
pid0 = pilot_ids[0]
p0 = patients[pid0]
ct_c = p0["ct_crop"]
z_mid = get_mid_slice(ct_c)

fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4))
fig.suptitle(f"Patient {pid0[:8]} — Windowing comparison (slice {z_mid})", fontsize=11)

for ax, (label, wargs) in zip(axes, windows.items()):
    img = window_hu(ct_c[:, :, z_mid], **wargs)
    ax.imshow(img.T, cmap="gray", origin="lower")
    ax.set_title(label, fontsize=8)
    ax.axis("off")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig01_windowing_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig01_windowing_comparison.png")

# %%
# Fig 2: fat mask overlay on mediastinal window — 3 slices (basal, mid, apical)
fig, axes = plt.subplots(len(pilot_ids), 3, figsize=(9, 3 * len(pilot_ids)))
fig.suptitle("Fat mask overlay on mediastinal window", fontsize=11)

for row, pid in enumerate(pilot_ids):
    p = patients[pid]
    if p.get("ct_crop") is None:
        continue
    ct_c = p["ct_crop"]
    nz = ct_c.shape[2]
    slices_z = [max(0, nz // 4), nz // 2, min(nz - 1, 3 * nz // 4)]

    for col, z in enumerate(slices_z):
        bg = window_hu(ct_c[:, :, z], lo=-160, hi=240)
        fat = fat_mask_from_hu(ct_c[:, :, z])
        ax = axes[row, col] if len(pilot_ids) > 1 else axes[col]
        ax.imshow(bg.T, cmap="gray", origin="lower")
        ax.imshow(fat.T, cmap="Reds", alpha=0.5, origin="lower", vmin=0, vmax=1)
        ax.set_title(f"{pid[:8]} z={z}", fontsize=7)
        ax.axis("off")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig02_fat_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig02_fat_overlay.png")

# %%
# Fig 3: ROI shape statistics and fat fraction per patient
fat_stats = []
for pid, p in patients.items():
    if p.get("ct_crop") is None:
        continue
    ct_c = p["ct_crop"]
    fat = fat_mask_from_hu(ct_c)
    roi_vox = int(np.prod(ct_c.shape))
    fat_vox = int(fat.sum())
    fat_stats.append({
        "pid": pid[:8],
        "roi_shape": ct_c.shape,
        "roi_voxels": roi_vox,
        "fat_voxels": fat_vox,
        "fat_fraction": fat_vox / roi_vox,
    })
    print(
        f"{pid[:8]}: ROI {ct_c.shape}, fat_vox={fat_vox:,} "
        f"({100 * fat_vox / roi_vox:.1f}%)"
    )

# %% [markdown]
# ## Evaluation

# %%
# Fat fraction stats across patients
fractions = [s["fat_fraction"] for s in fat_stats]
print(f"Fat fraction in cardiac ROI: mean={np.mean(fractions):.3f}, "
      f"min={min(fractions):.3f}, max={max(fractions):.3f}")

# Window range sanity: fraction of voxels in ADR-001 fat range vs wider ranges
for pid, p in patients.items():
    if p.get("ct_crop") is None:
        continue
    ct_c = p["ct_crop"]
    n = ct_c.size
    n_fat_strict = int(((ct_c >= -200) & (ct_c <= -30)).sum())
    n_fat_wide = int(((ct_c >= -250) & (ct_c <= 0)).sum())
    gain = (n_fat_wide - n_fat_strict) / n_fat_strict * 100 if n_fat_strict > 0 else 0
    print(
        f"{pid[:8]}: strict fat={n_fat_strict / n:.3%}, "
        f"wide fat={n_fat_wide / n:.3%} (+{gain:.1f}%)"
    )

# %%
results = {
    "n_patients": len(fat_stats),
    "fat_fraction_mean": float(np.mean(fractions)),
    "fat_fraction_min": float(min(fractions)),
    "fat_fraction_max": float(max(fractions)),
    "fat_stats": fat_stats,
}

with open(METRICS_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"Metrics saved to {METRICS_PATH}")

# %% [markdown]
# ## Visualization

# %%
# Fig 3: bar chart of fat fraction per patient
fig, ax = plt.subplots(figsize=(6, 3))
pids = [s["pid"] for s in fat_stats]
fracs = [s["fat_fraction"] * 100 for s in fat_stats]
ax.bar(pids, fracs, color="steelblue", edgecolor="black", linewidth=0.5)
ax.set_ylabel("Fat fraction in cardiac ROI (%)")
ax.set_title("Fat voxel fraction per patient (ADR-001: [-200,-30] HU)")
ax.set_ylim(0, max(fracs) * 1.3 if fracs else 1)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fig03_fat_fraction.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig03_fat_fraction.png")

# %% [markdown]
# ## Interpretation
#
# Fill in after running:
#
# - Which window best reveals the pericardial line?
# - Is the cardiac ROI crop tight enough / too tight?
# - Does the fat overlay look anatomically correct?
# - Does the fat fraction (%) match expectations (~20-30% of ROI)?
#
# Decisions and results to document:
# - If mediastinal window is clearly better for pericardium → inform exp-004 SAM2 input
# - If fat fraction distribution looks off → revisit heart mask dilation (ADR-009)
# - If fat overlay on mediastinal window clearly separates epi/para → note for pseudo-label strategy
