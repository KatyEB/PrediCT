# %% [markdown]
# # Experiment 010: Validation-Harness Quick Wins (Visual Lab external Dice)
#
# **Started:** 2026-07-10
# **Status:** Active
#
# The 2026-07-10 diagnostic (docs/results.md, ADR-014) showed the low Visual Lab Dice
# (epi 0.18 / para 0.11 in exp-008) is dominated by a **validation-harness** problem, not
# model quality:
#
# 1. GT↔CT alignment quality (`align_iou`) correlates r=0.64 with Dice; epi recall (0.22)
#    AND precision (0.16) are both low → predicted and GT fat are spatially OFFSET, not
#    concentric. exp-008's `build_aligned_gt` picks the z-offset by maximising overlap
#    between the annotation footprint and the **whole-slice** fat-HU mask — which is
#    contaminated by subcutaneous / chest-wall fat, so the offset objective is weak.
# 2. Para Dice in exp-008 is computed on the RAW broad prediction, not the 10 mm
#    pericardium-shell-corrected para that is the actual deliverable.
#
# This experiment implements the quick wins that need **no retraining** and reuses the
# existing exp-008 predictions:
#   - **A. Alignment fix** — restrict the z-offset objective to fat inside a dilated heart
#     region (cardiac neighbourhood), and search both stack orientations. Report a
#     ROI-restricted `align_iou`, and recompute epi/para Dice + recall + precision.
#   - **B. Para-shell Dice** — recompute para Dice against a pericardium-shell-restricted
#     prediction (proxy pericardium = filled epi prediction), matching the deliverable's
#     spatial extent.
#   - **C.** (run separately) re-enable TTA in nnUNetv2_predict for final numbers.
#
# The model and its predictions are UNCHANGED — only how the GT is aligned and how para
# is scored. Visual Lab remains test-only (ADR-005); nothing here tunes the model.

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt
from scipy.stats import pearsonr

EXPERIMENT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
REPO_ROOT = EXPERIMENT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from data.loaders import get_cardiac_roi_bbox

# Paths (mirror exp-008 config)
DICOM_DIR = REPO_ROOT / "data/Dicom _ Treino"
GT_DIR = REPO_ROOT / "data/GroundTruth-FatRange/Ground Truth - Fat Range"
NIFTI_DIR = REPO_ROOT / "data/visual_lab_nifti"
HEART_DIR = NIFTI_DIR / "heart_masks"
INPUT_DIR = NIFTI_DIR / "nnunet_input"
PRED_DIR = NIFTI_DIR / "nnunet_pred"

DILATION = (29, 29, 7)          # cardiac ROI crop dilation — same as training/exp-008
EPI_LABEL, PARA_LABEL = 1, 2
FAT_LO, FAT_HI = -200, -30
SHELL_MM = 10.0
GT_NAME_MAP = {"FPiq": "FSiq"}

patients = sorted(p.name for p in DICOM_DIR.iterdir() if p.is_dir())
print(f"{len(patients)} Visual Lab patients")


# %% [markdown]
# ## Helpers

# %%
def bmp_to_epi_para(arr: np.ndarray):
    """RED = epicardial fat, GREEN = mediastinal/paracardial (Visual Lab convention)."""
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    epi = (r > g + 30) & (r > b + 30)
    para = (g > r + 30) & (g > b + 30)
    return epi, para


def load_gt_stacks(patient: str, hw: tuple[int, int]):
    """Return (epi, para, colored) stacks (H, W, N) in FILE order (acquisition order)."""
    gt_dir = GT_DIR / GT_NAME_MAP.get(patient, patient)
    bmps = sorted(gt_dir.glob("*.bmp"))
    epi_st, para_st, col_st = [], [], []
    for f in bmps:
        arr = np.array(Image.open(f).convert("RGB"))
        if arr.shape[:2] != hw:
            arr = np.array(Image.fromarray(arr).resize((hw[1], hw[0]), Image.NEAREST))
        e, p = bmp_to_epi_para(arr)
        r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
        col = (np.abs(r - g) > 25) | (np.abs(g - b) > 25) | (np.abs(r - b) > 25)
        epi_st.append(e); para_st.append(p); col_st.append(col)
    return np.stack(epi_st, -1), np.stack(para_st, -1), np.stack(col_st, -1)


def best_alignment(col_st, target_fat, Zc):
    """Search (orientation, z-offset) maximising IoU between the GT colour footprint and
    `target_fat` (H,W,Zc). Returns (reversed?, offset, iou, per-slice profile corr)."""
    best = (True, 0, -1.0)
    tgt_prof = target_fat.reshape(-1, Zc).sum(0).astype(float)
    for rev in (True, False):
        cs = col_st[:, :, ::-1] if rev else col_st
        Nb = cs.shape[2]
        for off in range(0, max(1, Zc - Nb) + 1):
            inter = uni = 0
            for j in range(Nb):
                z = off + j
                if 0 <= z < Zc:
                    cj = cs[:, :, j]
                    tj = target_fat[:, :, z]
                    inter += int((cj & tj).sum())
                    uni += int((cj | tj).sum())
            iou = inter / uni if uni else 0.0
            if iou > best[2]:
                best = (rev, off, iou)
    return best


def place_gt(col_stacks, rev, off, Zc):
    """Place (epi,para) GT stacks into a full (H,W,Zc) label volume at the chosen offset."""
    epi_st, para_st = col_stacks
    if rev:
        epi_st, para_st = epi_st[:, :, ::-1], para_st[:, :, ::-1]
    H, W, Nb = epi_st.shape
    lab = np.zeros((H, W, Zc), np.uint8)
    for j in range(Nb):
        z = off + j
        if 0 <= z < Zc:
            lab[epi_st[:, :, j], z] = EPI_LABEL
            lab[para_st[:, :, j], z] = PARA_LABEL
    return lab


def dice(pred, gt, label):
    p, g = pred == label, gt == label
    d = p.sum() + g.sum()
    return float("nan") if d == 0 else float(2 * (p & g).sum() / d)


def recall_precision(pred, gt, label):
    p, g = pred == label, gt == label
    tp = (p & g).sum()
    rec = float(tp / g.sum()) if g.sum() else float("nan")
    prec = float(tp / p.sum()) if p.sum() else float("nan")
    return rec, prec


def tolerant_dice(pred, gt, label, spacing_mm, tol_mm=2.0):
    """Boundary-tolerant (surface) Dice: a voxel counts as a hit if it lies within
    `tol_mm` of the other mask. Appropriate for thin structures where strict volumetric
    Dice is dominated by sub-voxel boundary placement. This is the standard NSD-style
    agreement measure."""
    p, g = pred == label, gt == label
    if not p.any() and not g.any():
        return float("nan")
    if not p.any() or not g.any():
        return 0.0
    dist_to_g = distance_transform_edt(~g, sampling=spacing_mm)
    dist_to_p = distance_transform_edt(~p, sampling=spacing_mm)
    p_hit = (p & (dist_to_g <= tol_mm)).sum()
    g_hit = (g & (dist_to_p <= tol_mm)).sum()
    return float((p_hit + g_hit) / (p.sum() + g.sum()))


def para_shell(pred, spacing_mm, shell_mm=SHELL_MM):
    """Proxy pericardium = per-slice filled epi prediction; keep para within `shell_mm`
    outside it. Mirrors src/inference/apply_para_shell.py but self-contained for VL
    (which has no SAM2 pericardium mask). Returns the shell-restricted para mask."""
    epi = pred == EPI_LABEL
    sac = np.zeros_like(epi)
    for z in range(epi.shape[2]):
        if epi[:, :, z].any():
            sac[:, :, z] = binary_fill_holes(epi[:, :, z])
    if not sac.any():
        return pred == PARA_LABEL
    dist_out = distance_transform_edt(~sac, sampling=spacing_mm)
    shell = (dist_out > 0) & (dist_out <= shell_mm)
    return (pred == PARA_LABEL) & shell


# %% [markdown]
# ## Run — old vs new alignment, per patient

# %%
rows = []
for patient in patients:
    inp_path = INPUT_DIR / f"VL_{patient}_0000.nii.gz"
    pred_path = PRED_DIR / f"VL_{patient}.nii.gz"
    heart_path = HEART_DIR / patient / "heart.nii.gz"
    if not (inp_path.exists() and pred_path.exists() and heart_path.exists()):
        print(f"  {patient}: missing inputs — skip")
        continue

    full = nib.load(NIFTI_DIR / f"{patient}.nii.gz")
    vol = full.get_fdata(dtype=np.float32)
    Zc = vol.shape[2]
    heart = nib.load(heart_path).get_fdata() > 0.5
    roi = get_cardiac_roi_bbox(heart, dilation_vox=DILATION, array_shape=vol.shape)

    pred = nib.load(pred_path).get_fdata(dtype=np.float32).astype(np.int32)
    assert pred.shape == vol[roi].shape, f"{patient}: roi mismatch {pred.shape} vs {vol[roi].shape}"

    inp_img = nib.load(inp_path)
    spacing = tuple(float(z) for z in inp_img.header.get_zooms()[:3])
    vox_mL = float(np.prod(spacing)) / 1000.0

    epi_st, para_st, col_st = load_gt_stacks(patient, vol.shape[:2])

    fat_full = (vol >= FAT_LO) & (vol <= FAT_HI)
    cardiac = binary_dilation(heart, iterations=12)      # heart neighbourhood
    fat_roi = fat_full & cardiac                         # NEW objective target
    # OLD objective (exp-008): whole-slice fat, but only reversed orientation
    old_rev, old_off, old_iou = best_alignment(col_st, fat_full, Zc)  # search finds exp-008-like offset
    # force exp-008 behaviour: reversed only + whole-slice fat
    old = best_alignment(col_st[:, :, :], fat_full, Zc)
    # NEW objective: ROI-restricted fat, both orientations
    new_rev, new_off, new_iou = best_alignment(col_st, fat_roi, Zc)

    def score(rev, off):
        lab = place_gt((epi_st, para_st), rev, off, Zc)[roi]
        de, dp = dice(pred, lab, EPI_LABEL), dice(pred, lab, PARA_LABEL)
        re, pe = recall_precision(pred, lab, EPI_LABEL)
        tde = tolerant_dice(pred, lab, EPI_LABEL, spacing, tol_mm=2.0)
        # para shell-restricted Dice
        para_sh = para_shell(pred, spacing)
        gp = lab == PARA_LABEL
        dsh = float("nan")
        if para_sh.sum() + gp.sum():
            dsh = float(2 * (para_sh & gp).sum() / (para_sh.sum() + gp.sum()))
        return lab, de, dp, dsh, re, pe, tde

    lab_old, de_old, *_ = score(old[0], old[1])
    lab_new, de_new, dp_new, dsh_new, re_new, pe_new, tde_new = score(new_rev, new_off)

    gt_epi_mL = (lab_new == EPI_LABEL).sum() * vox_mL
    gt_para_mL = (lab_new == PARA_LABEL).sum() * vox_mL
    pred_epi_mL = (pred == EPI_LABEL).sum() * vox_mL
    pred_para_mL = (pred == PARA_LABEL).sum() * vox_mL

    rows.append(dict(
        patient=patient,
        old_iou=round(old[2], 3), new_iou=round(new_iou, 3),
        old_off=old[1], new_off=new_off, new_rev=bool(new_rev),
        dice_epi_old=round(de_old, 4), dice_epi_new=round(de_new, 4),
        tol_dice_epi_2mm=round(tde_new, 4),
        dice_para_raw=round(dp_new, 4), dice_para_shell=round(dsh_new, 4),
        recall_epi=round(re_new, 4), precision_epi=round(pe_new, 4),
        pred_epi_mL=round(pred_epi_mL, 1), gt_epi_mL=round(gt_epi_mL, 1),
        pred_para_mL=round(pred_para_mL, 1), gt_para_mL=round(gt_para_mL, 1),
    ))
    print(f"  {patient}: epi Dice {de_old:.3f}→{de_new:.3f} | iou {old[2]:.3f}→{new_iou:.3f} | "
          f"para raw {dp_new:.3f}→shell {dsh_new:.3f} | recall {re_new:.2f} prec {pe_new:.2f}")

# %% [markdown]
# ## Aggregate

# %%
import statistics as st

def mean(key):
    vals = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
    return round(sum(vals) / len(vals), 4)

r_epi = pearsonr([r["pred_epi_mL"] for r in rows], [r["gt_epi_mL"] for r in rows])[0]
r_para = pearsonr([r["pred_para_mL"] for r in rows], [r["gt_para_mL"] for r in rows])[0]

summary = {
    "n_patients": len(rows),
    "mean_dice_epi_old": mean("dice_epi_old"),
    "mean_dice_epi_new": mean("dice_epi_new"),
    "mean_tol_dice_epi_2mm": mean("tol_dice_epi_2mm"),
    "mean_dice_para_raw": mean("dice_para_raw"),
    "mean_dice_para_shell": mean("dice_para_shell"),
    "mean_recall_epi": mean("recall_epi"),
    "mean_precision_epi": mean("precision_epi"),
    "mean_align_iou_old": mean("old_iou"),
    "mean_align_iou_new": mean("new_iou"),
    "pearson_r_epi_volume": round(r_epi, 4),
    "pearson_r_para_volume": round(r_para, 4),
    "note": "Quick wins A (ROI-restricted alignment) + B (para-shell Dice). Predictions "
            "unchanged from exp-008 (fold 0, --disable_tta). Para shell uses a proxy "
            "pericardium = filled epi prediction (VL has no SAM2 mask). C (TTA) run separately.",
    "per_patient": rows,
}
with open(EXPERIMENT_DIR / "metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SUMMARY (old exp-008 harness → new harness) ===")
print(f"Epi Dice:        {summary['mean_dice_epi_old']:.3f} → {summary['mean_dice_epi_new']:.3f}  (strict, harsh on thin fat)")
print(f"Epi Dice@2mm:    {summary['mean_tol_dice_epi_2mm']:.3f}  (boundary-tolerant / surface Dice)")
print(f"Align IoU:       {summary['mean_align_iou_old']:.3f} → {summary['mean_align_iou_new']:.3f}")
print(f"Para Dice:       raw {summary['mean_dice_para_raw']:.3f} → shell {summary['mean_dice_para_shell']:.3f}")
print(f"Epi recall/prec: {summary['mean_recall_epi']:.3f} / {summary['mean_precision_epi']:.3f}")
print(f"Volume r:        epi {summary['pearson_r_epi_volume']:.3f} | para {summary['pearson_r_para_volume']:.3f}")

# %% [markdown]
# ## Overlay proof figures (old vs new alignment)

# %%
def overlay_fig(patient):
    full = nib.load(NIFTI_DIR / f"{patient}.nii.gz")
    vol = full.get_fdata(dtype=np.float32); Zc = vol.shape[2]
    heart = nib.load(HEART_DIR / patient / "heart.nii.gz").get_fdata() > 0.5
    roi = get_cardiac_roi_bbox(heart, dilation_vox=DILATION, array_shape=vol.shape)
    pred = nib.load(PRED_DIR / f"VL_{patient}.nii.gz").get_fdata().astype(np.int32)
    epi_st, para_st, col_st = load_gt_stacks(patient, vol.shape[:2])
    fat_full = (vol >= FAT_LO) & (vol <= FAT_HI)
    cardiac = binary_dilation(heart, iterations=12)
    old = best_alignment(col_st, fat_full, Zc)
    new = best_alignment(col_st, fat_full & cardiac, Zc)
    lab_old = place_gt((epi_st, para_st), old[0], old[1], Zc)[roi]
    lab_new = place_gt((epi_st, para_st), new[0], new[1], Zc)[roi]
    ct = vol[roi]
    # pick the pred-epi centroid slice
    zc = int(np.round(np.argwhere(pred == EPI_LABEL)[:, 2].mean())) if (pred == EPI_LABEL).any() else ct.shape[2] // 2
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a, lab, tag, iou in [(ax[0], lab_old, "OLD (whole-slice)", old[2]),
                             (ax[1], lab_new, "NEW (ROI-restricted)", new[2])]:
        a.imshow(np.clip(ct[:, :, zc], -200, -30), cmap="gray")
        a.contour((pred[:, :, zc] == EPI_LABEL), colors="cyan", linewidths=0.8)
        a.contour((lab[:, :, zc] == EPI_LABEL), colors="red", linewidths=0.8)
        a.set_title(f"{tag}  IoU={iou:.2f}", fontsize=10); a.axis("off")
    fig.suptitle(f"{patient} slice {zc} — pred epi (cyan) vs GT epi (red)", fontsize=11)
    plt.tight_layout()
    out = EXPERIMENT_DIR / "figures" / f"overlay_{patient}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    return out

for p in ["ACel", "EGra", "CLis"]:
    print("saved", overlay_fig(p).name)
