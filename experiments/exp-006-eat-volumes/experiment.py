# %% [markdown]
# # Experiment 006: EAT Volume Extraction and Pseudo-Label QC
#
# **Started:** 2026-06-06
# **Status:** Active
#
# Extracts epicardial (EAT) and paracardial (PAT) fat volumes from the SAM2
# pericardium pseudo-labels produced in exp-005. Compares against Commandeur 2018
# expected range (~78 mL median EAT for suspected-CAD patients) to validate that
# the pseudo-labels are anatomically plausible. This is the quantitative component
# of Decision Point 1.

# %% [markdown]
# ## Question
#
# Do the SAM2 pericardium pseudo-labels produce epicardial fat volumes consistent
# with the Commandeur 2018 reference population? Are any patients clearly
# over- or under-segmented?

# %% [markdown]
# ## Hypothesis
#
# Given 100% slice coverage on all 26 scored patients, we expect volumes in the
# expected range (~30–200 mL EAT). Patients with volumes outside [15, 250] mL
# are flagged for manual review.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import json
import sys
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
)
from src.data.splits import load_quality_scores

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

PERI_MASKS_DIR = REPO_ROOT / config["data"]["masks_dir"]
FAT_LO = config["fat_hu"]["lo"]
FAT_HI = config["fat_hu"]["hi"]
EXP_LO = config["data"]["expected_lo_ml"]
EXP_HI = config["data"]["expected_hi_ml"]
DIL = (29, 29, 7)

print(f"Experiment: {config['experiment']['id']} — {config['experiment']['name']}")
print(f"Fat HU range: [{FAT_LO}, {FAT_HI}]")
print(f"Expected EAT range: [{EXP_LO}, {EXP_HI}] mL")

# %% [markdown]
# ## Discover Available Pseudo-Labels

# %%
quality_scores = load_quality_scores()
available = sorted(
    pid for pid in quality_scores
    if (PERI_MASKS_DIR / pid / "pericardium_roi.nii.gz").exists()
)
print(f"Patients with pseudo-labels: {len(available)}")
print(f"  Score-1: {sum(1 for p in available if quality_scores[p] == 1)}")
print(f"  Score-2: {sum(1 for p in available if quality_scores[p] == 2)}")

# %% [markdown]
# ## EAT and PAT Volume Extraction
#
# For each patient:
# - **EAT** (epicardial): pericardium_mask ∩ fat_HU_mask
# - **PAT** (paracardial): cardiac_ROI ∩ fat_HU_mask ∩ NOT pericardium_mask
# - **Pericardium sac** total: pericardium_mask (includes non-fat tissue too)

# %%
results = []

for pid in available:
    p = load_coca_patient(pid, resampled_dir=RESAMPLED_DIR, heart_masks_dir=HEART_MASKS_DIR)
    roi = get_cardiac_roi_bbox(p["heart_mask"], dilation_vox=DIL, array_shape=p["ct"].shape)
    crop = crop_to_roi(p["ct"], roi, heart_mask=p["heart_mask"])
    ct_c = crop["ct_crop"]

    peri_mask = nib.load(PERI_MASKS_DIR / pid / "pericardium_roi.nii.gz").get_fdata().astype(np.uint8)
    fat_mask = fat_mask_from_hu(ct_c, lo=FAT_LO, hi=FAT_HI)

    vox_vol_ml = np.prod(p["spacing"]) / 1000.0  # mm³ per voxel → mL

    eat_vox = int((peri_mask & fat_mask).sum())
    pat_vox = int(((1 - peri_mask) & fat_mask).sum())  # fat in ROI but outside pericardium
    peri_vox = int(peri_mask.sum())

    eat_ml = eat_vox * vox_vol_ml
    pat_ml = pat_vox * vox_vol_ml
    peri_ml = peri_vox * vox_vol_ml
    total_fat_ml = (eat_vox + pat_vox) * vox_vol_ml

    eat_fraction = eat_ml / total_fat_ml if total_fat_ml > 0 else 0.0
    flagged = not (EXP_LO <= eat_ml <= EXP_HI)

    results.append({
        "scan_id": pid,
        "quality_score": quality_scores[pid],
        "eat_ml": round(eat_ml, 1),
        "pat_ml": round(pat_ml, 1),
        "total_fat_ml": round(total_fat_ml, 1),
        "pericardium_sac_ml": round(peri_ml, 1),
        "eat_fraction": round(eat_fraction, 3),
        "flagged": flagged,
    })
    flag_str = " *** FLAGGED ***" if flagged else ""
    print(f"  {pid} (q{quality_scores[pid]}): EAT={eat_ml:.0f}mL  PAT={pat_ml:.0f}mL  "
          f"total={total_fat_ml:.0f}mL  epi_frac={eat_fraction:.1%}{flag_str}")

# %% [markdown]
# ## Summary Statistics

# %%
eat_vols = [r["eat_ml"] for r in results]
pat_vols = [r["pat_ml"] for r in results]
flagged = [r for r in results if r["flagged"]]

print(f"\nEAT volume (n={len(eat_vols)}):")
print(f"  Mean:   {np.mean(eat_vols):.1f} mL")
print(f"  Median: {np.median(eat_vols):.1f} mL")
print(f"  Std:    {np.std(eat_vols):.1f} mL")
print(f"  Range:  [{np.min(eat_vols):.1f}, {np.max(eat_vols):.1f}] mL")
print(f"\nPAT volume (n={len(pat_vols)}):")
print(f"  Mean:   {np.mean(pat_vols):.1f} mL")
print(f"  Median: {np.median(pat_vols):.1f} mL")
print(f"\nCommandeur 2018 reference: median ~78 mL (IQR ~48–117 mL)")
print(f"\nFlagged outliers (outside [{EXP_LO}, {EXP_HI}] mL): {len(flagged)}")
for r in flagged:
    print(f"  {r['scan_id']} (q{r['quality_score']}): EAT={r['eat_ml']} mL")

# %% [markdown]
# ## Visualization
#
# Figure 1: EAT volume distribution vs Commandeur 2018 reference
# Figure 2: EAT vs PAT scatter (colored by quality score)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("exp-006: EAT + PAT volumes from SAM2 pseudo-labels", fontsize=10)

# Panel 1: EAT histogram with literature reference
ax = axes[0]
ax.hist(eat_vols, bins=12, color="#4878d0", edgecolor="white", alpha=0.85)
ax.axvline(78, color="red", linestyle="--", linewidth=1.5, label="Commandeur median (78 mL)")
ax.axvspan(48, 117, color="red", alpha=0.08, label="Commandeur IQR (48–117 mL)")
ax.axvline(np.median(eat_vols), color="#4878d0", linestyle="-", linewidth=1.5,
           label=f"This cohort median ({np.median(eat_vols):.0f} mL)")
ax.set_xlabel("Epicardial fat volume (mL)", fontsize=9)
ax.set_ylabel("Patients", fontsize=9)
ax.set_title("EAT volume distribution", fontsize=9)
ax.legend(fontsize=7)

# Panel 2: EAT vs PAT scatter
ax = axes[1]
colors = {1: "#2196F3", 2: "#FF9800"}
for r in results:
    c = colors[r["quality_score"]]
    ax.scatter(r["eat_ml"], r["pat_ml"], color=c, s=50, alpha=0.8,
               label=f"Score-{r['quality_score']}")

# Remove duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=7)
ax.set_xlabel("EAT (mL)", fontsize=9)
ax.set_ylabel("PAT (mL)", fontsize=9)
ax.set_title("EAT vs PAT by quality score", fontsize=9)

plt.tight_layout()
fname = FIGURES_DIR / "fig01_eat_pat_volumes.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname.name}")

# %% [markdown]
# ## Figure 2: EAT fraction (epi vs total cardiac fat)

# %%
fig, ax = plt.subplots(figsize=(8, 4))
fractions = sorted([r["eat_fraction"] for r in results])
ax.bar(range(len(fractions)), [f * 100 for f in fractions], color="#4878d0", alpha=0.85)
ax.axhline(np.mean([r["eat_fraction"] for r in results]) * 100,
           color="red", linestyle="--", linewidth=1.2,
           label=f"Mean {np.mean([r['eat_fraction'] for r in results]):.1%}")
ax.set_xlabel("Patients (sorted by EAT fraction)", fontsize=9)
ax.set_ylabel("EAT / (EAT + PAT) (%)", fontsize=9)
ax.set_title("Epicardial fat fraction of total cardiac ROI fat", fontsize=9)
ax.legend(fontsize=8)
plt.tight_layout()
fname = FIGURES_DIR / "fig02_eat_fraction.png"
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname.name}")

# %% [markdown]
# ## Save Metrics

# %%
summary = {
    "n_patients": len(results),
    "n_score1": sum(1 for r in results if r["quality_score"] == 1),
    "n_score2": sum(1 for r in results if r["quality_score"] == 2),
    "eat_mean_ml": round(float(np.mean(eat_vols)), 1),
    "eat_median_ml": round(float(np.median(eat_vols)), 1),
    "eat_std_ml": round(float(np.std(eat_vols)), 1),
    "eat_min_ml": round(float(np.min(eat_vols)), 1),
    "eat_max_ml": round(float(np.max(eat_vols)), 1),
    "pat_mean_ml": round(float(np.mean(pat_vols)), 1),
    "pat_median_ml": round(float(np.median(pat_vols)), 1),
    "n_flagged": len(flagged),
    "flagged_ids": [r["scan_id"] for r in flagged],
    "commandeur_2018_median_ml": 78,
    "commandeur_2018_iqr": [48, 117],
    "per_patient": results,
}

with open(METRICS_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Metrics saved: {METRICS_PATH}")

# %% [markdown]
# ## Interpretation
#
# **Compare against Commandeur 2018:**
# - Commandeur 2018 (n=140, suspected CAD): EAT median 78 mL (IQR 48–117 mL)
# - COCA is a similar population (suspected CAD, non-contrast CT)
# - Expect our median to be within ~20 mL of 78 mL
#
# **EAT fraction:**
# - Literature: EAT is typically 30–50% of total cardiac ROI fat
# - Very high fractions (>70%) may indicate the ROI is too tight
# - Very low fractions (<10%) may indicate SAM2 is under-segmenting the pericardium
#
# **Decision gate (Decision Point 1 preview):**
# - If median EAT ∈ [40, 120] mL and <15% of patients flagged → proceed to nnU-Net
# - If median is off or many outliers → review QC figures, consider manual correction
