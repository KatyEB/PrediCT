# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: eat-seg
#     language: python
#     name: eat-seg
# ---

# # exp-009: Qualitative visualization of ensemble EAT predictions
#
# **Goal:** Visual sanity check of the fold 0+1 ensemble predictions on 9 representative
# COCA patients spanning the full epi fat distribution (P10 → P90).
#
# Each patient gets a 4-panel figure showing axial slices evenly spaced across the heart,
# with semi-transparent overlays: red = epicardial fat, blue = paracardial fat.

# +
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import yaml

REPO = Path("../..").resolve()
sys.path.insert(0, str(REPO))

cfg = yaml.safe_load(open("config.yaml"))

INPUT_DIR = REPO / cfg["paths"]["nnunet_input"]
PRED_DIR  = REPO / cfg["paths"]["nnunet_pred"]
PERI_DIR  = REPO / cfg["paths"]["peri_masks"]
OUT_DIR   = REPO / cfg["paths"]["output_dir"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

HU_LO, HU_HI = cfg["visualization"]["hu_window"]
ALPHA = cfg["visualization"]["alpha"]
N_SLICES = cfg["visualization"]["n_slices"]
EPI_COLOR  = np.array(cfg["visualization"]["epi_color"])
PARA_COLOR = np.array(cfg["visualization"]["para_color"])
PERI_COLOR = np.array(cfg["visualization"]["peri_color"])
# -

def load_patient(pid):
    inp  = nib.load(INPUT_DIR / f"{pid}_0000.nii.gz").get_fdata(dtype=np.float32)
    pred = nib.load(PRED_DIR  / f"{pid}.nii.gz").get_fdata(dtype=np.float32).astype(np.int32)
    peri_path = PERI_DIR / pid / "pericardium_roi.nii.gz"
    peri = nib.load(peri_path).get_fdata(dtype=np.float32) > 0.5 if peri_path.exists() else None
    return inp, pred, peri


def window_ct(arr):
    return np.clip((arr - HU_LO) / (HU_HI - HU_LO), 0, 1)


def pick_slices(pred, n):
    """Pick n axial slices evenly spaced where the heart has predictions."""
    has_pred = (pred > 0).sum(axis=(0, 1)) > 0
    active = np.where(has_pred)[0]
    if len(active) == 0:
        return list(range(0, pred.shape[2], max(1, pred.shape[2] // n)))[:n]
    idx = np.linspace(active[0], active[-1], n + 2, dtype=int)[1:-1]
    return idx.tolist()


def make_overlay(ct_slice, pred_slice, peri_slice):
    """Return H×W×3 RGB image with semi-transparent label overlays."""
    base = np.stack([window_ct(ct_slice)] * 3, axis=-1)
    out = base.copy()

    for label, color in [(1, EPI_COLOR), (2, PARA_COLOR)]:
        mask = pred_slice == label
        if mask.any():
            out[mask] = (1 - ALPHA) * base[mask] + ALPHA * color

    if peri_slice is not None:
        boundary = _boundary(peri_slice)
        out[boundary] = PERI_COLOR

    return np.clip(out, 0, 1)


def _boundary(mask):
    """Thin boundary of a binary mask."""
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask)


# +
patients = cfg["patients"]
print(f"Generating figures for {len(patients)} patients...\n")

for p in patients:
    pid = p["id"]
    print(f"  {pid}  epi={p['epi_mL']} mL  para={p['para_mL']} mL  [{p['group']}]")

    inp, pred, peri = load_patient(pid)
    slices = pick_slices(pred, N_SLICES)

    fig, axes = plt.subplots(1, N_SLICES, figsize=(4 * N_SLICES, 4.5))
    fig.suptitle(
        f"{pid}  |  epi = {p['epi_mL']} mL,  para = {p['para_mL']} mL  [{p['group']}]",
        fontsize=11, y=1.01
    )

    for ax, z in zip(axes, slices):
        ct_sl   = inp[:, :, z].T
        pred_sl = pred[:, :, z].T
        peri_sl = peri[:, :, z].T if peri is not None else None
        img = make_overlay(ct_sl, pred_sl, peri_sl)
        ax.imshow(img, origin="lower", aspect="equal")
        ax.set_title(f"z={z}", fontsize=9)
        ax.axis("off")

    # Legend
    legend = [
        mpatches.Patch(color=EPI_COLOR,  label="Epicardial fat"),
        mpatches.Patch(color=PARA_COLOR, label="Paracardial fat (10mm)"),
        mpatches.Patch(color=PERI_COLOR, label="Pericardium boundary"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    out_path = OUT_DIR / f"{pid}_{p['group']}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    saved → {out_path.relative_to(REPO)}")

print(f"\nDone. {len(patients)} figures in {OUT_DIR.relative_to(REPO)}/")
# -

# ## Summary panel — all 9 patients on one page

# +
fig, axes = plt.subplots(len(patients), N_SLICES,
                         figsize=(4 * N_SLICES, 3.5 * len(patients)))
fig.suptitle("EAT Ensemble Predictions — COCA NCCT (9 representative patients)",
             fontsize=13, y=1.002)

for row, p in enumerate(patients):
    pid = p["id"]
    inp, pred, peri = load_patient(pid)
    slices = pick_slices(pred, N_SLICES)

    for col, z in enumerate(slices):
        ax = axes[row][col]
        ct_sl   = inp[:, :, z].T
        pred_sl = pred[:, :, z].T
        peri_sl = peri[:, :, z].T if peri is not None else None
        ax.imshow(make_overlay(ct_sl, pred_sl, peri_sl), origin="lower", aspect="equal")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(f"{p['epi_mL']}mL epi\n{p['para_mL']}mL para",
                          fontsize=8, rotation=0, labelpad=60, va="center")

legend = [
    mpatches.Patch(color=EPI_COLOR,  label="Epicardial fat"),
    mpatches.Patch(color=PARA_COLOR, label="Paracardial fat (10mm)"),
    mpatches.Patch(color=PERI_COLOR, label="Pericardium boundary"),
]
fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.01))

plt.tight_layout()
summary_path = OUT_DIR / "summary_all_patients.png"
plt.savefig(summary_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Summary panel saved → {summary_path.relative_to(REPO)}")
# -
