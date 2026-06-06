# %% [markdown]
# # Experiment 002: Visual Lab Annotation Pixel-Level Study
#
# **Started:** 2026-05-14
# **Status:** Active
#
# The Visual Lab Cardiac Fat Database (Rodrigues 2016) is our only source of expert-labeled
# epicardial + paracardial fat annotations. Before any model work, we need to fully understand
# the annotation format, color encoding, DICOM structure, and HU distributions. This experiment
# is purely exploratory — no model training, just careful documentation of the data.

# %% [markdown]
# ## Question
#
# What does the Visual Lab ground truth encode, and how do we reliably extract separate
# epicardial (epi) and paracardial (para) labels from the BMP mask files?
# Specifically: are epi and para encoded as distinct colors (red vs green), or as separate files?

# %% [markdown]
# ## Hypothesis
#
# Based on Rodrigues 2016 (Section 3.2) and scope.md, the BMP masks use color to distinguish
# compartments: red channel = epicardial fat, green channel = paracardial fat.
# If this is correct, we can extract binary masks by thresholding each color channel.
# We also expect HU values at labeled pixels to fall within [-200, -30] (ADR-001).

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
print(f"Repo root: {REPO_ROOT}")

# %%
DICOM_DIR = REPO_ROOT / config["data"]["dicom_dir"]
GT_FAT_DIR = REPO_ROOT / config["data"]["gt_fat_range_dir"]
GT_COMBINED_DIR = REPO_ROOT / config["data"]["gt_combined_range_dir"]
GT_HIGHER_DIR = REPO_ROOT / config["data"]["gt_higher_range_dir"]
FAT_IMAGES_DIR = REPO_ROOT / config["data"]["fat_images_dir"]

patients = sorted([p.name for p in DICOM_DIR.iterdir() if p.is_dir()])
print(f"Patients in DICOM dir: {len(patients)}")
print(patients)

# %% [markdown]
# ## Data — DICOM Structure

# %%
import pydicom

# Examine one patient
sample_patient = patients[0]
dicom_patient_dir = DICOM_DIR / sample_patient
dicom_files = sorted(dicom_patient_dir.glob("*.dcm"))
print(f"Patient: {sample_patient}")
print(f"DICOM files: {len(dicom_files)}")
print(f"Sample file name: {dicom_files[0].name}")

# %%
# Load one DICOM slice to inspect metadata
ds = pydicom.dcmread(dicom_files[0])
print(f"Modality:          {getattr(ds, 'Modality', 'N/A')}")
print(f"Rows x Cols:       {ds.Rows} x {ds.Columns}")
print(f"Pixel Spacing:     {getattr(ds, 'PixelSpacing', 'N/A')}")
print(f"Slice Thickness:   {getattr(ds, 'SliceThickness', 'N/A')}")
print(f"Bits Allocated:    {getattr(ds, 'BitsAllocated', 'N/A')}")
print(f"Rescale Intercept: {getattr(ds, 'RescaleIntercept', 'N/A')}")
print(f"Rescale Slope:     {getattr(ds, 'RescaleSlope', 'N/A')}")

# %%
# Convert pixel array to HU values
def dcm_to_hu(ds):
    """Convert DICOM pixel array to Hounsfield Units."""
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    slope = float(getattr(ds, "RescaleSlope", 1))
    return ds.pixel_array.astype(np.float32) * slope + intercept

ct_slice = dcm_to_hu(ds)
print(f"CT slice HU range: [{ct_slice.min():.0f}, {ct_slice.max():.0f}]")
print(f"CT slice shape:    {ct_slice.shape}")

# %% [markdown]
# ## Data — BMP Ground Truth Structure

# %%
from PIL import Image

# Examine ground truth for the same patient
gt_patient_dir = GT_FAT_DIR / sample_patient
gt_files = sorted(gt_patient_dir.glob("*.bmp"))
print(f"Patient: {sample_patient}")
print(f"GT BMP files: {len(gt_files)}")
print(f"Sample GT file name: {gt_files[0].name}")

# %%
# Load one BMP and inspect channels
bmp_img = np.array(Image.open(gt_files[0]))
print(f"BMP shape:  {bmp_img.shape}")  # expecting (H, W, 3) or (H, W)
print(f"BMP dtype:  {bmp_img.dtype}")
print(f"BMP range:  [{bmp_img.min()}, {bmp_img.max()}]")

if bmp_img.ndim == 3:
    r, g, b = bmp_img[:, :, 0], bmp_img[:, :, 1], bmp_img[:, :, 2]
    print(f"\nChannel max — R: {r.max()}, G: {g.max()}, B: {b.max()}")
    print(f"R nonzero pixels: {(r > 0).sum()}")
    print(f"G nonzero pixels: {(g > 0).sum()}")
    print(f"B nonzero pixels: {(b > 0).sum()}")

# %%
# Survey ALL BMP files for a patient to understand red vs green encoding
def bmp_to_epi_para_masks(arr: np.ndarray):
    """Extract epicardial and paracardial masks from a Visual Lab GT BMP array.

    Encoding discovered in exp-002:
    - Background / CT tissue: grayscale (R=G=B)
    - Epicardial fat: MAGENTA (R>>G, B>>G, R>B) — annotation tool "red" renders as magenta
    - Paracardial fat: GREEN (G>>R, G>>B)

    Uses channel dominance rather than simple thresholding.
    """
    if arr.ndim != 3:
        return np.zeros(arr.shape, dtype=bool), np.zeros(arr.shape, dtype=bool)
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    # Para: green dominant — G much higher than both R and B
    para_mask = (g > r + 20) & (g > b + 20)
    # Epi: magenta/red dominant — R and B are both much higher than G (and R>B)
    epi_mask = (r > g + 50) & (b > g + 30) & (r > b)
    return epi_mask, para_mask


def count_bmp_labels(gt_dir: Path, patient: str) -> dict:
    """Count epi (magenta) and para (green) labeled pixels across all slices."""
    pat_dir = gt_dir / patient
    bmp_files = sorted(pat_dir.glob("*.bmp"))
    epi_pixels, para_pixels, both_pixels, slices_with_epi, slices_with_para = 0, 0, 0, 0, 0
    for f in bmp_files:
        arr = np.array(Image.open(f))
        epi_mask, para_mask = bmp_to_epi_para_masks(arr)
        epi_pixels += epi_mask.sum()
        para_pixels += para_mask.sum()
        both_pixels += (epi_mask & para_mask).sum()
        slices_with_epi += int(epi_mask.any())
        slices_with_para += int(para_mask.any())
    return {
        "patient": patient,
        "num_slices": len(bmp_files),
        "epi_pixels": int(epi_pixels),
        "para_pixels": int(para_pixels),
        "both_pixels": int(both_pixels),
        "slices_with_epi": slices_with_epi,
        "slices_with_para": slices_with_para,
    }

sample_counts = count_bmp_labels(GT_FAT_DIR, sample_patient)
print(json.dumps(sample_counts, indent=2))

# %% [markdown]
# ## Run — Survey All 20 Patients

# %%
# Run for all 20 patients
rows = []
for pat in patients:
    gt_pat_dir = GT_FAT_DIR / pat
    if not gt_pat_dir.exists():
        print(f"WARNING: No GT for {pat}")
        continue
    row = count_bmp_labels(GT_FAT_DIR, pat)
    rows.append(row)
    print(f"  {pat}: {row['num_slices']} slices | epi={row['epi_pixels']:6d}px | para={row['para_pixels']:6d}px")

df = pd.DataFrame(rows)
print(f"\nTotal patients with GT: {len(df)}")
print(f"\n{df[['patient','num_slices','epi_pixels','para_pixels','slices_with_epi','slices_with_para']].to_string(index=False)}")

# %%
# Summary statistics
print("=== Summary ===")
print(f"Patients:          {len(df)}")
print(f"Slices/patient:    {df['num_slices'].min()}–{df['num_slices'].max()} (mean {df['num_slices'].mean():.1f})")
print(f"Epi pixels/pat:    {df['epi_pixels'].min():,}–{df['epi_pixels'].max():,} (mean {df['epi_pixels'].mean():,.0f})")
print(f"Para pixels/pat:   {df['para_pixels'].min():,}–{df['para_pixels'].max():,} (mean {df['para_pixels'].mean():,.0f})")
print(f"Overlap pixels:    {df['both_pixels'].sum()} total (should be 0 if epi/para are exclusive)")
epi_frac = df["epi_pixels"].sum() / (df["epi_pixels"].sum() + df["para_pixels"].sum())
print(f"Epi fraction:      {epi_frac:.2%} of total fat pixels")

# %% [markdown]
# ## Run — HU Distribution at Labeled Pixels

# %%
def extract_hu_at_labels(dicom_dir: Path, gt_dir: Path, patient: str, hu_range=(-200, -30)):
    """Extract HU values at epi and para labeled pixels for one patient.

    Matches DICOM slices to BMP masks by sort order (both are sorted by filename).
    """
    dcm_files = sorted((dicom_dir / patient).glob("*.dcm"))
    bmp_files = sorted((gt_dir / patient).glob("*.bmp"))

    if len(dcm_files) != len(bmp_files):
        warnings.warn(f"{patient}: {len(dcm_files)} DICOMs vs {len(bmp_files)} BMPs — skipping")
        return None

    hu_epi, hu_para = [], []
    for dcm_f, bmp_f in zip(dcm_files, bmp_files):
        ds = pydicom.dcmread(dcm_f)
        ct = dcm_to_hu(ds)
        arr = np.array(Image.open(bmp_f))
        epi_mask, para_mask = bmp_to_epi_para_masks(arr)
        if ct.shape != epi_mask.shape:
            # Resize mask if needed
            from PIL import Image as PILImage
            epi_mask = np.array(PILImage.fromarray(epi_mask).resize(
                (ct.shape[1], ct.shape[0]), PILImage.NEAREST))
            para_mask = np.array(PILImage.fromarray(para_mask).resize(
                (ct.shape[1], ct.shape[0]), PILImage.NEAREST))
        hu_epi.extend(ct[epi_mask].tolist())
        hu_para.extend(ct[para_mask].tolist())
    return np.array(hu_epi, dtype=np.float32), np.array(hu_para, dtype=np.float32)

# Run on a few patients
hu_results = {}
for pat in patients[:5]:
    gt_pat_dir = GT_FAT_DIR / pat
    if not gt_pat_dir.exists():
        continue
    result = extract_hu_at_labels(DICOM_DIR, GT_FAT_DIR, pat)
    if result is not None:
        hu_epi, hu_para = result
        hu_results[pat] = {"epi": hu_epi, "para": hu_para}
        print(f"{pat}: epi HU [{hu_epi.min():.0f}, {hu_epi.max():.0f}] n={len(hu_epi):,} | "
              f"para HU [{hu_para.min():.0f}, {hu_para.max():.0f}] n={len(hu_para):,}")

# %% [markdown]
# ## Visualization

# %%
# 1. Overlay: CT slice + epi (red) + para (green) for one patient/slice

pat = patients[0]
dcm_files = sorted((DICOM_DIR / pat).glob("*.dcm"))
bmp_files = sorted((GT_FAT_DIR / pat).glob("*.bmp"))

# Find a slice with both epi and para labels
for i, (dcm_f, bmp_f) in enumerate(zip(dcm_files, bmp_files)):
    arr = np.array(Image.open(bmp_f))
    epi_m, para_m = bmp_to_epi_para_masks(arr)
    if epi_m.any() and para_m.any():
        chosen_dcm, chosen_bmp, chosen_i = dcm_f, bmp_f, i
        break
else:
    chosen_dcm, chosen_bmp, chosen_i = dcm_files[len(dcm_files)//2], bmp_files[len(bmp_files)//2], len(dcm_files)//2

ds = pydicom.dcmread(chosen_dcm)
ct = dcm_to_hu(ds)
gt_arr = np.array(Image.open(chosen_bmp))
epi_mask_vis, para_mask_vis = bmp_to_epi_para_masks(gt_arr)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Patient {pat} — slice {chosen_i}", fontsize=13)

# CT windowed for soft tissue
vmin, vmax = -200, 300
axes[0].imshow(ct, cmap="gray", vmin=vmin, vmax=vmax)
axes[0].set_title("CT (soft tissue window)")
axes[0].axis("off")

# Ground truth BMP (raw annotated image)
axes[1].imshow(gt_arr)
axes[1].set_title(f"GT BMP (raw) — epi=magenta, para=green")
axes[1].axis("off")

# Overlay: extracted binary masks on CT
ct_norm = np.clip((ct - vmin) / (vmax - vmin), 0, 1)
overlay = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)
overlay[epi_mask_vis] = [1.0, 0.2, 0.2]   # red = epi
overlay[para_mask_vis] = [0.2, 1.0, 0.2]  # green = para
axes[2].imshow(overlay)
axes[2].set_title(f"Extracted masks: red=epi({epi_mask_vis.sum()}px), green=para({para_mask_vis.sum()}px)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig01_annotation_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved fig01_annotation_overlay.png")

# %%
# 2. HU distributions: epi vs para for first 5 patients
if hu_results:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-250, 50, 80)
    all_epi = np.concatenate([v["epi"] for v in hu_results.values()])
    all_para = np.concatenate([v["para"] for v in hu_results.values()])
    ax.hist(all_epi[np.isfinite(all_epi)], bins=bins, alpha=0.6, color="red", label=f"Epicardial (n={len(all_epi):,})", density=True)
    ax.hist(all_para[np.isfinite(all_para)], bins=bins, alpha=0.6, color="green", label=f"Paracardial (n={len(all_para):,})", density=True)
    ax.axvline(-200, color="k", linestyle="--", linewidth=1, label="HU range boundary (-200)")
    ax.axvline(-30, color="k", linestyle=":", linewidth=1, label="HU range boundary (-30)")
    ax.set_xlabel("HU value")
    ax.set_ylabel("Density")
    ax.set_title("HU distribution at labeled pixels (5 patients)")
    ax.legend()
    ax.set_xlim(-260, 60)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig02_hu_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved fig02_hu_distributions.png")

# %%
# 3. Pixel counts per patient
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = range(len(df))
axes[0].bar(x, df["epi_pixels"], color="red", alpha=0.7, label="Epi")
axes[0].bar(x, df["para_pixels"], bottom=df["epi_pixels"], color="green", alpha=0.7, label="Para")
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(df["patient"], rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("Labeled pixels")
axes[0].set_title("Epi + Para pixels per patient (fat range GT)")
axes[0].legend()

axes[1].bar(x, df["num_slices"], color="steelblue", alpha=0.7)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels(df["patient"], rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("Number of slices")
axes[1].set_title("Slices per patient")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig03_per_patient_stats.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved fig03_per_patient_stats.png")

# %% [markdown]
# ## Evaluation

# %%
# Key findings to document
overlap_ok = df["both_pixels"].sum() == 0
total_epi = df["epi_pixels"].sum()
total_para = df["para_pixels"].sum()

if hu_results:
    all_epi_hu = np.concatenate([v["epi"] for v in hu_results.values()])
    all_para_hu = np.concatenate([v["para"] for v in hu_results.values()])
    epi_in_range = ((all_epi_hu >= -200) & (all_epi_hu <= -30)).mean()
    para_in_range = ((all_para_hu >= -200) & (all_para_hu <= -30)).mean()
else:
    epi_in_range = para_in_range = float("nan")

results = {
    "num_patients": int(len(df)),
    "slices_per_patient_min": int(df["num_slices"].min()),
    "slices_per_patient_max": int(df["num_slices"].max()),
    "slices_per_patient_mean": float(df["num_slices"].mean()),
    "total_epi_pixels": int(total_epi),
    "total_para_pixels": int(total_para),
    "epi_para_overlap_pixels": int(df["both_pixels"].sum()),
    "epi_para_exclusive": bool(overlap_ok),
    "epi_fraction_of_fat": float(total_epi / (total_epi + total_para)) if (total_epi + total_para) > 0 else 0.0,
    "epi_hu_in_fat_range_frac": float(epi_in_range) if not np.isnan(epi_in_range) else None,
    "para_hu_in_fat_range_frac": float(para_in_range) if not np.isnan(para_in_range) else None,
    "color_encoding": "R=epi, G=para (hypothesis to verify from above)",
}

with open(METRICS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print("=== Key findings ===")
for k, v in results.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## Interpretation
#
# **What did we learn?**
#
# - The color encoding: if `epi_para_exclusive=True` and both R and G channels have nonzero
#   pixels, the hypothesis is confirmed: red channel = epicardial, green channel = paracardial.
#
# - HU distribution: if `epi_hu_in_fat_range_frac` and `para_hu_in_fat_range_frac` are both
#   near 1.0, the labeled pixels align well with our [-200, -30] ADR-001 threshold.
#   Any pixels outside range tell us where to expect annotation noise.
#
# - Dataset size: we have 20 patients, NOT the 5 mentioned in prior planning. This is better
#   for validation (wider CI on Dice, more representative). No change to ADR-005 (all 20 stay
#   held-out). Update project docs accordingly.
#
# - DICOM slice count: variable across patients. Note this for nnU-Net planning —
#   we need to understand whether each scan covers the full heart or is cardiac-gated.
#
# **Decisions triggered:**
# - If BMP encoding confirmed → document in `docs/visual-lab-convention.md`
# - If HU outside range > 5% → add sensitivity note in docs/decisions.md
# - Update "5 scans" → "20 patients" everywhere in docs
#
# **Questions for Katy:**
# - Confirm that all 20 Visual Lab patients are safe to use as held-out test
#   (licensing, prior use in PrediCT group, etc.)
