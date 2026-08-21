# %% [markdown]
# # Experiment 007: nnU-Net Training Dataset Creation
#
# **Started:** 2026-06-09
# **Status:** Active
#
# Converts the 153 SAM2 pericardium pseudo-labels into nnU-Net v2 Task format:
# cardiac ROI CT crops as images, 3-class label maps (0=bg, 1=epicardial fat,
# 2=paracardial fat). Dataset is then pushed to Kaggle for GPU training.

# %% [markdown]
# ## What we're building
#
# nnU-Net expects:
# ```
# nnunet/raw/Dataset001_EAT/
#   dataset.json
#   imagesTr/EAT_XXXX_0000.nii.gz   ← CT cardiac ROI crop
#   labelsTr/EAT_XXXX.nii.gz        ← 3-class label
#   imagesVal/EAT_XXXX_0000.nii.gz  ← val set (optional, nnU-Net does internal CV)
#   labelsVal/EAT_XXXX.nii.gz
# ```
#
# Labels:
# - 0: background (non-fat tissue in cardiac ROI)
# - 1: epicardial fat  = pericardium_mask & fat_HU[-200,-30]
# - 2: paracardial fat = NOT pericardium_mask & fat_HU[-200,-30]

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import json
import sys
import shutil
from pathlib import Path

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
from src.data.splits import load_quality_scores, make_splits

# %%
EXPERIMENT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
METRICS_PATH = EXPERIMENT_DIR / "metrics.json"

DATA_ROOT = REPO_ROOT / "data" / "COCA_dataset"
RESAMPLED_DIR = DATA_ROOT / "data_resampled"
HEART_MASKS_DIR = DATA_ROOT / "heart_masks"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

PERI_MASKS_DIR = REPO_ROOT / config["data"]["masks_dir"]
FAT_LO = config["fat_hu"]["lo"]
FAT_HI = config["fat_hu"]["hi"]
DIL = tuple(config["cardiac_roi"]["dilation_vox"])
DATASET_ID = config["nnunet"]["dataset_id"]
DATASET_NAME = config["nnunet"]["dataset_name"]

NNUNET_RAW = REPO_ROOT / config["output"]["nnunet_raw"]
DATASET_DIR = NNUNET_RAW / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"

print(f"Output: {DATASET_DIR}")

# %% [markdown]
# ## Load splits

# %%
quality_scores = load_quality_scores()
splits_path = REPO_ROOT / config["data"]["splits_file"]
with open(splits_path) as f:
    splits_data = json.load(f)

splits = splits_data["splits"]
train_ids = splits["train"]
val_ids = splits["val"]

# Only keep patients that have pericardium masks
train_ids = [p for p in train_ids if (PERI_MASKS_DIR / p / "pericardium_roi.nii.gz").exists()]
val_ids = [p for p in val_ids if (PERI_MASKS_DIR / p / "pericardium_roi.nii.gz").exists()]

print(f"Train: {len(train_ids)} patients")
print(f"Val:   {len(val_ids)} patients")
print(f"Total: {len(train_ids) + len(val_ids)} patients")

# %% [markdown]
# ## Build nnU-Net directory structure

# %%
for subdir in ["imagesTr", "labelsTr", "imagesVal", "labelsVal"]:
    (DATASET_DIR / subdir).mkdir(parents=True, exist_ok=True)

print(f"Created dataset dirs under {DATASET_DIR}")

# %% [markdown]
# ## Label generation function
#
# For each patient:
# 1. Load CT, heart mask, pericardium mask
# 2. Crop CT to cardiac ROI
# 3. Compute 3-class label: 0=bg, 1=epi fat, 2=para fat
# 4. Save crop as image, label as mask

# %%
def build_patient(pid: str, case_id: int, img_dir: Path, lbl_dir: Path) -> dict:
    """Build one nnU-Net case. Returns stats dict."""
    p = load_coca_patient(pid, resampled_dir=RESAMPLED_DIR, heart_masks_dir=HEART_MASKS_DIR)
    roi = get_cardiac_roi_bbox(p["heart_mask"], dilation_vox=DIL, array_shape=p["ct"].shape)
    crop = crop_to_roi(p["ct"], roi, heart_mask=p["heart_mask"])
    ct_c = crop["ct_crop"]

    peri_mask = nib.load(PERI_MASKS_DIR / pid / "pericardium_roi.nii.gz").get_fdata().astype(np.uint8)
    fat_mask = fat_mask_from_hu(ct_c, lo=FAT_LO, hi=FAT_HI)

    # 3-class label
    label = np.zeros(ct_c.shape, dtype=np.uint8)
    label[(peri_mask == 1) & (fat_mask == 1)] = 1   # epicardial fat
    label[(peri_mask == 0) & (fat_mask == 1)] = 2   # paracardial fat

    # nnU-Net affine: use patient spacing (identity orientation, spacing in affine diagonal)
    sx, sy, sz = p["spacing"]
    affine = np.diag([sx, sy, sz, 1.0])

    case_name = f"{DATASET_NAME}_{case_id:04d}"
    img_path = img_dir / f"{case_name}_0000.nii.gz"
    lbl_path = lbl_dir / f"{case_name}.nii.gz"

    nib.save(nib.Nifti1Image(ct_c, affine), img_path)
    nib.save(nib.Nifti1Image(label, affine), lbl_path)

    vox_ml = np.prod(p["spacing"]) / 1000
    return {
        "scan_id": pid,
        "case_name": case_name,
        "shape": list(ct_c.shape),
        "spacing": list(p["spacing"]),
        "epi_fat_vox": int((label == 1).sum()),
        "para_fat_vox": int((label == 2).sum()),
        "epi_fat_ml": round(float((label == 1).sum()) * vox_ml, 1),
        "para_fat_ml": round(float((label == 2).sum()) * vox_ml, 1),
    }

# %% [markdown]
# ## Build training set

# %%
train_stats = []
for i, pid in enumerate(train_ids):
    s = build_patient(pid, case_id=i+1,
                      img_dir=DATASET_DIR / "imagesTr",
                      lbl_dir=DATASET_DIR / "labelsTr")
    train_stats.append(s)
    if (i+1) % 10 == 0 or i == 0:
        print(f"  [{i+1}/{len(train_ids)}] {pid}: shape={s['shape']} "
              f"epi={s['epi_fat_ml']}mL para={s['para_fat_ml']}mL")

print(f"\nTrain set built: {len(train_stats)} cases")

# %% [markdown]
# ## Build validation set

# %%
val_stats = []
for i, pid in enumerate(val_ids):
    s = build_patient(pid, case_id=len(train_ids)+i+1,
                      img_dir=DATASET_DIR / "imagesVal",
                      lbl_dir=DATASET_DIR / "labelsVal")
    val_stats.append(s)
    print(f"  [{i+1}/{len(val_ids)}] {pid}: epi={s['epi_fat_ml']}mL para={s['para_fat_ml']}mL")

print(f"Val set built: {len(val_stats)} cases")

# %% [markdown]
# ## Write dataset.json (nnU-Net v2 format)

# %%
all_cases = train_stats + val_stats

dataset_json = {
    "channel_names": {"0": "CT"},
    "labels": {
        "background": 0,
        "epicardial_fat": 1,
        "paracardial_fat": 2,
    },
    "numTraining": len(train_stats),
    "file_ending": ".nii.gz",
    "name": f"Dataset{DATASET_ID:03d}_{DATASET_NAME}",
    "description": config["nnunet"]["description"],
    "reference": "COCA dataset (Stanford) + SAM2 pseudo-labels",
    "licence": "research only",
    "overwrite_image_reader_writer": "NibabelIOWithReorient",
}

with open(DATASET_DIR / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)
print(f"Wrote dataset.json")
print(f"  labels: {dataset_json['labels']}")
print(f"  numTraining: {dataset_json['numTraining']}")

# %% [markdown]
# ## Summary statistics

# %%
epi_vols = [s["epi_fat_ml"] for s in all_cases]
para_vols = [s["para_fat_ml"] for s in all_cases]

print(f"\nDataset summary ({len(all_cases)} patients):")
print(f"  EAT: mean={np.mean(epi_vols):.0f} mL  median={np.median(epi_vols):.0f} mL  range=[{np.min(epi_vols):.0f},{np.max(epi_vols):.0f}]")
print(f"  PAT: mean={np.mean(para_vols):.0f} mL  median={np.median(para_vols):.0f} mL  range=[{np.min(para_vols):.0f},{np.max(para_vols):.0f}]")

# Dataset size on disk
total_size_mb = sum(
    f.stat().st_size for d in [DATASET_DIR/"imagesTr", DATASET_DIR/"labelsTr",
                                DATASET_DIR/"imagesVal", DATASET_DIR/"labelsVal"]
    for f in d.iterdir()
) / 1e6
print(f"\nDataset size on disk: {total_size_mb:.0f} MB")
print(f"Kaggle dataset limit: 100,000 MB — using {total_size_mb/1000:.2f}%")

# %%
metrics = {
    "n_train": len(train_stats),
    "n_val": len(val_stats),
    "dataset_dir": str(DATASET_DIR),
    "dataset_size_mb": round(total_size_mb, 1),
    "eat_mean_ml": round(float(np.mean(epi_vols)), 1),
    "eat_median_ml": round(float(np.median(epi_vols)), 1),
    "pat_mean_ml": round(float(np.mean(para_vols)), 1),
    "train_cases": train_stats,
    "val_cases": val_stats,
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved: {METRICS_PATH}")
print(f"\nNext: run nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 2d 3d_fullres")
print(f"Then: push to Kaggle with python kaggle/scripts/push_dataset.py")
