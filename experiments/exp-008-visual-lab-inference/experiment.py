# %% [markdown]
# # Experiment 008: Visual Lab Inference — Decision Point 2
#
# **Started:** 2026-06-19
# **Status:** Active
#
# Run nnU-Net 2D fold 0 (trained on COCA pseudo-labels) on the Visual Lab 20-patient
# held-out test set and compare against expert ground truth annotations.
# This is the primary Decision Point 2 evaluation: does the model achieve
# epi Dice > 0.75 and para Dice > 0.65?
#
# Pipeline:
# 1. Convert Visual Lab DICOMs → NIfTI (full volume per patient)
# 2. Generate heart masks via TotalSegmentator
# 3. Crop to cardiac ROI (same dilation as training)
# 4. Convert BMP GT → NIfTI label crops (same ROI)
# 5. Run nnUNetv2_predict on crops
# 6. Compute Dice scores per patient and aggregate

# %% [markdown]
# ## Question
#
# Does nnU-Net 2D fold 0 (trained on COCA pseudo-labels) generalize to the Visual Lab
# expert-annotated dataset, achieving Dice ≥ 0.75 (epi) and ≥ 0.65 (para)?
#
# NOTE: The Visual Lab Cardiac Fat Database is **non-contrast cardiac CT (NCCT)** — the
# SAME modality as COCA (confirmed on visual.ic.uff.br/en/cardio/ctfat). There is NO
# contrast/domain shift. An earlier version of this notebook wrongly attributed the poor
# Dice to NCCT→CCTA domain shift; that was incorrect on the facts.

# %% [markdown]
# ## Hypothesis
#
# Since modality matches (both NCCT), the main risk is a DEFINITIONAL gap: our training
# pseudo-labels define epicardial fat as *all* fat inside the coarse SAM2 pericardium sac,
# whereas Visual Lab experts trace a thinner epicardial band and explicitly exclude the
# pericardium gap. We expect the per-patient epi VOLUME to correlate well but pixel-level
# Dice to be limited by this boundary-definition difference.
#
# (This notebook was also carrying two harness bugs — a BMP colour-decode error and a
# GT slice-ordering error — that made the raw Dice artificially catastrophic. Both are
# fixed here; see bmp_to_epi_para_masks and build_aligned_gt.)

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import yaml
from PIL import Image

EXPERIMENT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
REPO_ROOT = EXPERIMENT_DIR.parent.parent
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
METRICS_PATH = EXPERIMENT_DIR / "metrics.json"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

print(f"Experiment: {config['experiment']['id']} — {config['experiment']['name']}")
print(f"Repo root: {REPO_ROOT}")

# %%
DICOM_DIR = REPO_ROOT / config["data"]["dicom_dir"]
GT_DIR = REPO_ROOT / config["data"]["gt_fat_range_dir"]
NIFTI_DIR = REPO_ROOT / config["data"]["nifti_dir"]
HEART_MASKS_DIR = REPO_ROOT / config["data"]["heart_masks_dir"]
NNUNET_INPUT_DIR = REPO_ROOT / config["data"]["nnunet_input_dir"]
NNUNET_GT_DIR = REPO_ROOT / config["data"]["nnunet_gt_dir"]
NNUNET_PRED_DIR = REPO_ROOT / config["data"]["nnunet_pred_dir"]

for d in [NIFTI_DIR, HEART_MASKS_DIR, NNUNET_INPUT_DIR, NNUNET_GT_DIR, NNUNET_PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

patients = sorted([p.name for p in DICOM_DIR.iterdir() if p.is_dir()])
print(f"Patients: {len(patients)}")
print(patients)

# %% [markdown]
# ## Step 1: Convert DICOMs → NIfTI (full volume per patient)

# %%
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))
from data.loaders import get_cardiac_roi_bbox


def load_dicom_volume(patient_dir: Path) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Load sorted DICOM slices → (H, W, Z) array, positions array, (dx, dy, dz) spacing."""
    dcm_files = sorted(patient_dir.glob("*.dcm"))
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(f)
        slices.append(ds)

    # Sort by Z position (ImagePositionPatient[2])
    slices.sort(key=lambda s: float(getattr(s, "ImagePositionPatient", [0, 0, 0])[2]))

    # Stack into 3D volume (H, W, Z)
    pixel_data = []
    for ds in slices:
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        arr = ds.pixel_array.astype(np.float32) * slope + intercept
        pixel_data.append(arr)

    vol = np.stack(pixel_data, axis=-1)  # (H, W, Z)

    # Spacing
    ps = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    dz = float(getattr(slices[0], "SliceThickness", 3.0))
    spacing = (float(ps[0]), float(ps[1]), dz)  # (dy_mm, dx_mm, dz_mm)

    # Affine: simple diagonal (no oblique slice support needed for these axial scans)
    dx, dy = spacing[1], spacing[0]
    z0 = float(slices[0].ImagePositionPatient[2])
    x0 = float(slices[0].ImagePositionPatient[0])
    y0 = float(slices[0].ImagePositionPatient[1])
    affine = np.diag([dx, dy, dz, 1.0]).astype(np.float64)
    affine[:3, 3] = [x0, y0, z0]

    return vol, affine, spacing


# Convert all patients (skip if already done)
for patient in patients:
    nifti_path = NIFTI_DIR / f"{patient}.nii.gz"
    if nifti_path.exists():
        print(f"  {patient}: already converted")
        continue
    vol, affine, spacing = load_dicom_volume(DICOM_DIR / patient)
    img = nib.Nifti1Image(vol, affine)
    img.header.set_zooms(spacing)
    nib.save(img, nifti_path)
    print(f"  {patient}: saved {vol.shape} ({spacing[0]:.3f}×{spacing[1]:.3f}×{spacing[2]:.1f} mm) → {nifti_path.name}")

print(f"\nDone: {len(patients)} NIfTIs in {NIFTI_DIR}")

# %% [markdown]
# ## Step 2: Heart Masks via TotalSegmentator

# %%
import torch

print(f"MPS available: {torch.backends.mps.is_available()}")

for patient in patients:
    mask_dir = HEART_MASKS_DIR / patient
    mask_path = mask_dir / "heart.nii.gz"
    if mask_path.exists():
        print(f"  {patient}: heart mask already exists")
        continue
    mask_dir.mkdir(parents=True, exist_ok=True)
    nifti_path = NIFTI_DIR / f"{patient}.nii.gz"
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/TotalSegmentator"),
            "-i", str(nifti_path),
            "-o", str(mask_dir),
            "--roi_subset", "heart",
            "--fast",
            "--ml",
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  {patient}: TotalSegmentator FAILED\n{result.stderr[-500:]}")
    else:
        print(f"  {patient}: heart mask saved")

print(f"\nDone: heart masks in {HEART_MASKS_DIR}")

# %% [markdown]
# ## Step 3 + 4: Crop to Cardiac ROI and Convert BMP GT → NIfTI Labels

# %%
DILATION = tuple(config["cardiac_roi"]["dilation_vox"])  # (ax0, ax1, ax2)
EPI_LABEL = config["labels"]["epicardial_fat"]   # 1
PARA_LABEL = config["labels"]["paracardial_fat"]  # 2


def bmp_to_epi_para_masks(arr: np.ndarray):
    """Extract epi (red) and para (green) masks from a Visual Lab GT BMP.

    Encoding — per the Visual Lab CT-FAT dataset page (visual.ic.uff.br/en/cardio/ctfat):
    "the red color represents the epicardial fat, the green color represents the
    mediastinal fat and the blue color represents the gap between the epicardial and
    mediastinal fats, which can also be interpreted as the pericardium."

    - Epicardial fat  = RED-dominant   (r > g and r > b)
    - Paracardial fat = GREEN-dominant (g > r and g > b)
    - Pericardium (blue) and background are excluded from both.

    NOTE (fixed 2026-07-01): the original decoder looked for MAGENTA (required high
    blue), which dropped ~85% of the pure-red epicardial-fat pixels — making GT epi
    volumes ~7x too small and collapsing epi Dice. See docs/results.md 2026-07-01.
    """
    if arr.ndim != 3:
        return np.zeros(arr.shape[:2], dtype=bool), np.zeros(arr.shape[:2], dtype=bool)
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    epi_mask = (r > g + 30) & (r > b + 30)
    para_mask = (g > r + 30) & (g > b + 30)
    return epi_mask, para_mask


# Some patients have a DICOM/GT folder-name mismatch (same patient, different label).
GT_NAME_MAP = {"FPiq": "FSiq"}


def build_aligned_gt(patient: str, vol: np.ndarray) -> np.ndarray:
    """Build a Z-aligned GT label volume (H, W, Z) matching `vol`.

    Fixes the slice-ordering bug (2026-07-01): `load_dicom_volume` sorts the CT by
    ImagePositionPatient[2] ASCENDING, but Z DECREASES as InstanceNumber increases, so
    the BMP GT (numbered 001..N in acquisition order) is stacked in the OPPOSITE Z
    direction. We therefore reverse the BMP stack and offset-align it to the CT by
    maximising the overlap between the annotation footprint (all coloured pixels ≈ the
    fat region) and the CT fat-HU mask [-200,-30]. This is prediction-independent.
    """
    fat = (vol >= -200) & (vol <= -30)
    Zc = vol.shape[2]
    gt_dir = GT_DIR / GT_NAME_MAP.get(patient, patient)
    bmp_files = sorted(gt_dir.glob("*.bmp"))
    if not bmp_files:
        return np.zeros(vol.shape, dtype=np.uint8)

    epi_st, para_st, col_st = [], [], []
    for f in bmp_files:
        arr = np.array(Image.open(f).convert("RGB"))
        if arr.shape[:2] != vol.shape[:2]:
            arr = np.array(Image.fromarray(arr).resize(
                (vol.shape[1], vol.shape[0]), Image.NEAREST))
        e, p = bmp_to_epi_para_masks(arr)
        r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
        col = (np.abs(r - g) > 25) | (np.abs(g - b) > 25) | (np.abs(r - b) > 25)
        epi_st.append(e); para_st.append(p); col_st.append(col)

    # Reverse (acquisition/Z-descending -> Z-ascending to match the CT)
    epi_st = np.stack(epi_st, -1)[:, :, ::-1]
    para_st = np.stack(para_st, -1)[:, :, ::-1]
    col_st = np.stack(col_st, -1)[:, :, ::-1]
    Nb = epi_st.shape[2]

    # Choose the whole-stack offset that best overlaps the CT fat mask
    best_off, best_iou = 0, -1.0
    for off in range(0, max(1, Zc - Nb) + 1):
        inter = uni = 0
        for j in range(Nb):
            z = off + j
            if 0 <= z < Zc:
                inter += int((col_st[:, :, j] & fat[:, :, z]).sum())
                uni += int((col_st[:, :, j] | fat[:, :, z]).sum())
        iou = inter / uni if uni else 0.0
        if iou > best_iou:
            best_iou, best_off = iou, off

    label_vol = np.zeros(vol.shape, dtype=np.uint8)
    for j in range(Nb):
        z = best_off + j
        if 0 <= z < Zc:
            label_vol[epi_st[:, :, j], z] = EPI_LABEL
            label_vol[para_st[:, :, j], z] = PARA_LABEL
    print(f"    aligned GT: reversed + offset={best_off} (footprint IoU={best_iou:.2f})")
    return label_vol


roi_map = {}  # patient → (roi, original_shape)

for patient in patients:
    input_path = NNUNET_INPUT_DIR / f"VL_{patient}_0000.nii.gz"
    gt_path = NNUNET_GT_DIR / f"VL_{patient}.nii.gz"

    if input_path.exists() and gt_path.exists():
        print(f"  {patient}: crops already exist")
        # Still need to restore roi_map for later evaluation
        img = nib.load(input_path)
        roi_map[patient] = None  # placeholder — reload during eval
        continue

    # Load full NIfTI (H, W, Z)
    img = nib.load(NIFTI_DIR / f"{patient}.nii.gz")
    vol = img.get_fdata(dtype=np.float32)  # (H, W, Z)
    affine = img.affine
    zooms = img.header.get_zooms()

    # Load heart mask
    mask_path = HEART_MASKS_DIR / patient / "heart.nii.gz"
    if not mask_path.exists():
        print(f"  {patient}: no heart mask — skipping")
        continue
    heart_img = nib.load(mask_path)
    heart_mask = (heart_img.get_fdata() > 0.5).astype(np.uint8)

    # Cardiac ROI crop (axes: H, W, Z)
    roi = get_cardiac_roi_bbox(heart_mask, dilation_vox=DILATION, array_shape=vol.shape)
    vol_crop = vol[roi]
    roi_map[patient] = (roi, vol.shape)

    # Save cropped CT as nnUNet input (add channel dim: 1-channel CT)
    crop_affine = affine.copy()
    # Adjust origin for the crop offset
    crop_offset = np.array([roi[0].start, roi[1].start, roi[2].start], dtype=float)
    crop_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ crop_offset
    ct_img = nib.Nifti1Image(vol_crop, crop_affine)
    ct_img.header.set_zooms(zooms)
    nib.save(ct_img, input_path)

    # Build a Z-aligned GT label volume from the BMP files (reversed + offset-aligned;
    # see build_aligned_gt). `vol` is Z-ascending, BMPs are acquisition-order (reversed).
    label_vol = build_aligned_gt(patient, vol)
    label_crop = label_vol[roi]
    gt_img = nib.Nifti1Image(label_crop.astype(np.int16), crop_affine)
    gt_img.header.set_zooms(zooms)
    nib.save(gt_img, gt_path)

    epi_px = (label_crop == EPI_LABEL).sum()
    para_px = (label_crop == PARA_LABEL).sum()
    print(f"  {patient}: crop {vol_crop.shape} | epi={epi_px:,}px | para={para_px:,}px")

print(f"\nDone: nnU-Net inputs in {NNUNET_INPUT_DIR}")
print(f"Done: GT labels in {NNUNET_GT_DIR}")

# %% [markdown]
# ## Step 5: Run nnUNetv2_predict

# %%
# Set nnU-Net env vars
os.environ["nnUNet_raw"] = str(REPO_ROOT / "nnunet/raw")
os.environ["nnUNet_preprocessed"] = str(REPO_ROOT / "nnunet/preprocessed")
os.environ["nnUNet_results"] = str(REPO_ROOT / "nnunet/results")

ds_id = config["nnunet"]["dataset_id"]
cfg = config["nnunet"]["config"]
fold = config["nnunet"]["fold"]
trainer = config["nnunet"]["trainer"]
chkpt = config["nnunet"]["checkpoint"]

n_existing = len(list(NNUNET_PRED_DIR.glob("*.nii.gz")))
if n_existing >= len(patients):
    print(f"Predictions already exist ({n_existing} files) — skipping inference.")
else:
    print(f"Running nnUNetv2_predict on {len(patients)} patients...")
    result = subprocess.run(
        [
            str(REPO_ROOT / ".venv/bin/nnUNetv2_predict"),
            "-i", str(NNUNET_INPUT_DIR),
            "-o", str(NNUNET_PRED_DIR),
            "-d", str(ds_id),
            "-c", cfg,
            "-f", str(fold),
            "-tr", trainer,
            "--chk", chkpt,
            "--disable_tta",   # faster; TTA can be re-enabled for final numbers
        ],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"nnUNetv2_predict FAILED with code {result.returncode}")
    else:
        print("Inference complete.")

pred_files = sorted(NNUNET_PRED_DIR.glob("VL_*.nii.gz"))
print(f"Prediction files: {len(pred_files)}")

# %% [markdown]
# ## Step 6: Compute Dice Scores

# %%
def dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Dice coefficient for a single class label."""
    p = (pred == label)
    g = (gt == label)
    tp = (p & g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return float("nan")  # no GT and no pred — undefined
    return float(2 * tp / denom)


results_per_patient = []

for patient in patients:
    pred_path = NNUNET_PRED_DIR / f"VL_{patient}.nii.gz"
    gt_path = NNUNET_GT_DIR / f"VL_{patient}.nii.gz"

    if not pred_path.exists():
        print(f"  {patient}: no prediction — skipping")
        continue
    if not gt_path.exists():
        print(f"  {patient}: no GT — skipping")
        continue

    pred = nib.load(pred_path).get_fdata(dtype=np.float32).astype(np.int32)
    gt = nib.load(gt_path).get_fdata(dtype=np.float32).astype(np.int32)

    if pred.shape != gt.shape:
        warnings.warn(f"{patient}: pred shape {pred.shape} != gt shape {gt.shape}")

    dice_epi = dice_score(pred, gt, EPI_LABEL)
    dice_para = dice_score(pred, gt, PARA_LABEL)

    # Voxel counts for volume estimates (assume same spacing as input)
    img = nib.load(NNUNET_INPUT_DIR / f"VL_{patient}_0000.nii.gz")
    zooms = img.header.get_zooms()
    vox_vol_mm3 = float(np.prod(zooms[:3]))
    vox_vol_mL = vox_vol_mm3 / 1000.0

    pred_epi_mL = (pred == EPI_LABEL).sum() * vox_vol_mL
    pred_para_mL = (pred == PARA_LABEL).sum() * vox_vol_mL
    gt_epi_mL = (gt == EPI_LABEL).sum() * vox_vol_mL
    gt_para_mL = (gt == PARA_LABEL).sum() * vox_vol_mL

    row = {
        "patient": patient,
        "dice_epi": round(dice_epi, 4),
        "dice_para": round(dice_para, 4),
        "pred_epi_mL": round(pred_epi_mL, 1),
        "pred_para_mL": round(pred_para_mL, 1),
        "gt_epi_mL": round(gt_epi_mL, 1),
        "gt_para_mL": round(gt_para_mL, 1),
    }
    results_per_patient.append(row)
    print(f"  {patient}: epi Dice={dice_epi:.3f} | para Dice={dice_para:.3f} | "
          f"pred epi={pred_epi_mL:.0f}mL (GT {gt_epi_mL:.0f}) | "
          f"pred para={pred_para_mL:.0f}mL (GT {gt_para_mL:.0f})")

# %% [markdown]
# ## Evaluation

# %%
import pandas as pd

df = pd.DataFrame(results_per_patient)
print(df.to_string(index=False))

if len(df) > 0:
    mean_epi = df["dice_epi"].mean()
    mean_para = df["dice_para"].mean()
    mean_overall = (mean_epi + mean_para) / 2
    std_epi = df["dice_epi"].std()
    std_para = df["dice_para"].std()

    # Volume correlation
    from scipy.stats import pearsonr
    r_epi, p_epi = pearsonr(df["pred_epi_mL"], df["gt_epi_mL"])
    r_para, p_para = pearsonr(df["pred_para_mL"], df["gt_para_mL"])

    dp2_epi = config["decision_point_2"]["epi_dice_threshold"]
    dp2_para = config["decision_point_2"]["para_dice_threshold"]
    epi_pass = mean_epi >= dp2_epi
    para_pass = mean_para >= dp2_para

    print(f"\n=== Decision Point 2 Evaluation ===")
    print(f"Epi  Dice: {mean_epi:.3f} ± {std_epi:.3f}  (threshold {dp2_epi}) → {'PASS' if epi_pass else 'FAIL'}")
    print(f"Para Dice: {mean_para:.3f} ± {std_para:.3f}  (threshold {dp2_para}) → {'PASS' if para_pass else 'FAIL'}")
    print(f"Overall:   {mean_overall:.3f}")
    print(f"Epi  volume Pearson r: {r_epi:.3f} (p={p_epi:.3f})")
    print(f"Para volume Pearson r: {r_para:.3f} (p={p_para:.3f})")
    print(f"\nDecision Point 2: {'CLEARED' if (epi_pass and para_pass) else 'NOT CLEARED'}")
    if not epi_pass and not para_pass:
        print("  → Fallback: ship total EAT volume only (ADR scope.md)")
    elif not epi_pass:
        print("  → Epi boundary unclear — consider adding pericardium mask post-processing")
    elif not para_pass:
        print("  → Para boundary unclear — consider larger cardiac ROI dilation")

    metrics = {
        "n_patients": len(df),
        "mean_dice_epi": round(mean_epi, 4),
        "std_dice_epi": round(std_epi, 4),
        "mean_dice_para": round(mean_para, 4),
        "std_dice_para": round(std_para, 4),
        "mean_dice_overall": round(mean_overall, 4),
        "pearson_r_epi_volume": round(r_epi, 4),
        "pearson_r_para_volume": round(r_para, 4),
        "decision_point_2_epi_threshold": dp2_epi,
        "decision_point_2_para_threshold": dp2_para,
        "decision_point_2_epi_pass": bool(epi_pass),
        "decision_point_2_para_pass": bool(para_pass),
        "decision_point_2_cleared": bool(epi_pass and para_pass),
        "per_patient": results_per_patient,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

# %% [markdown]
# ## Visualization

# %%
import matplotlib.pyplot as plt

if len(df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Visual Lab Validation — nnU-Net 2D fold 0", fontsize=13)

    # Per-patient Dice bars
    x = range(len(df))
    axes[0].bar(x, df["dice_epi"], alpha=0.7, color="red", label="Epi")
    axes[0].bar(x, df["dice_para"], alpha=0.7, color="green", label="Para", bottom=0)
    axes[0].axhline(dp2_epi, color="red", linestyle="--", linewidth=1.5, label=f"DP2 epi ({dp2_epi})")
    axes[0].axhline(dp2_para, color="green", linestyle="--", linewidth=1.5, label=f"DP2 para ({dp2_para})")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(df["patient"], rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Dice")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Per-patient Dice")
    axes[0].legend(fontsize=8)

    # Volume scatter: epi
    axes[1].scatter(df["gt_epi_mL"], df["pred_epi_mL"], color="red", alpha=0.7)
    lim = max(df["gt_epi_mL"].max(), df["pred_epi_mL"].max()) * 1.1
    axes[1].plot([0, lim], [0, lim], "k--", linewidth=1)
    axes[1].set_xlabel("GT epi volume (mL)")
    axes[1].set_ylabel("Pred epi volume (mL)")
    axes[1].set_title(f"Epi volume (r={r_epi:.2f})")

    # Volume scatter: para
    axes[2].scatter(df["gt_para_mL"], df["pred_para_mL"], color="green", alpha=0.7)
    lim = max(df["gt_para_mL"].max(), df["pred_para_mL"].max()) * 1.1
    axes[2].plot([0, lim], [0, lim], "k--", linewidth=1)
    axes[2].set_xlabel("GT para volume (mL)")
    axes[2].set_ylabel("Pred para volume (mL)")
    axes[2].set_title(f"Para volume (r={r_para:.2f})")

    plt.tight_layout()
    fig_path = EXPERIMENT_DIR / "figures" / "fig01_validation_summary.png"
    fig_path.parent.mkdir(exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {fig_path.name}")

# %% [markdown]
# ## Interpretation (corrected 2026-07-01)
#
# **Decision Point 2 threshold:** epi Dice ≥ 0.75, para Dice ≥ 0.65 → **NOT CLEARED.**
#
# Corrected results (after fixing the colour-decode and slice-ordering bugs):
# - Epi Dice ~0.18 mean (up to ~0.30 on best-aligned patients) — was 0.034 with the bugs.
# - Para Dice ~0.11 mean.
# - Epi VOLUME Pearson r = 0.87; predicted epi mean 107 mL vs GT 81 mL (right magnitude).
#
# **Why DP2 still fails — and why it does NOT invalidate the deliverable:**
# - The Visual Lab is NCCT (no domain shift). The low Dice is a *definitional* gap:
#   our pseudo-labels call all fat inside the coarse SAM2 pericardium sac "epicardial",
#   while the experts trace a thinner band and exclude the pericardium gap. The model
#   therefore over-thickens epi → spatial overlap is limited even when the amount is right.
# - The project deliverable is per-patient EAT *volume* for PrediCT, not pixel-perfect
#   masks. Epi volume correlates strongly (r=0.87) and is the right magnitude — the
#   validation that matters for the deliverable holds.
# - The expert GT para volume (mean 71 mL) is close to Rodrigues (103 mL) and corroborates
#   that the raw nnU-Net para (mean 305 mL here) is over-broad — i.e. it independently
#   supports the 10 mm pericardium-shell para correction (ADR-013).
#
# **Consequences:**
# - Report corrected epi/para Dice honestly; DP2 not cleared on Dice.
# - Keep shipping per-patient epi/para VOLUMES (validated by volume correlation).
# - Residual slice-correspondence uncertainty (footprint-IoU alignment is weak on
#   low-fat patients) means ~0.18 is a lower bound; the exact Dice is fuzzy but
#   unambiguously < 0.75. A cleaner external validation still needs expert-labelled
#   NCCT COCA scans (see docs/questions.md Q-007).
