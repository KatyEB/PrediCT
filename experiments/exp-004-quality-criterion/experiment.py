# %% [markdown]
# # Experiment 004: Pericardial Line Visibility — Quality Criterion
#
# **Started:** 2026-05-15
# **Status:** Active
#
# Defines and applies an operational quality criterion for COCA patients:
# can the pericardial sac boundary be reliably traced in this scan?
# Scans that fail this criterion are held out as "hard cases" (ADR-003).
# This determines which patients are eligible for SAM2 pseudo-labeling in exp-005.

# %% [markdown]
# ## Question
#
# Which COCA patients have a clearly visible pericardial line (score 1), partial
# visibility (score 2), or poor/no visibility (score 3)?
# How does the distribution look across 785 patients?

# %% [markdown]
# ## Hypothesis
#
# Based on Wang 2025 §4 (the major EAT failure mode is pericardial invisibility),
# we expect ~15-25% of COCA scans to have poor pericardial visibility (score 3),
# leaving ~580–665 "usable" patients for SAM2 bootstrapping.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import (
    crop_to_roi,
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
SCORES_CSV = REPO_ROOT / "data" / "quality_scores.csv"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

SEED = config["data"]["random_seed"]
N_SAMPLE = config["data"]["n_sample"]
WIN_LO = config["windowing"]["lo"]
WIN_HI = config["windowing"]["hi"]
DIL = tuple(config["cardiac_roi"]["dilation_vox"])

print(f"Experiment: {config['experiment']['id']} — {config['experiment']['name']}")

# %% [markdown]
# ## Data — Sample 30 patients for scoring
#
# We score a random sample of 30 patients to calibrate the criterion and get an
# initial quality distribution. After this session, score another ~50–100 to finalize
# the training split.

# %%
available = sorted(
    p.name for p in RESAMPLED_DIR.iterdir()
    if p.is_dir() and (HEART_MASKS_DIR / p.name / "heart.nii.gz").exists()
)
print(f"Available patients: {len(available)}")

random.seed(SEED)
sample_ids = sorted(random.sample(available, min(N_SAMPLE, len(available))))
print(f"Scoring sample: {len(sample_ids)} patients")

# %%
# Load the sample
patients = {}
for pid in sample_ids:
    p = load_coca_patient(pid, resampled_dir=RESAMPLED_DIR, heart_masks_dir=HEART_MASKS_DIR)
    roi = get_cardiac_roi_bbox(p["heart_mask"], dilation_vox=DIL, array_shape=p["ct"].shape)
    crop = crop_to_roi(p["ct"], roi, heart_mask=p["heart_mask"])
    p.update(crop)
    patients[pid] = p

print(f"Loaded {len(patients)} patients")

# %% [markdown]
# ## Visualization — Quality Scoring Grid
#
# For each patient: 4 axial slices (basal → apical) in mediastinal window.
# After viewing these figures, enter scores in the `quality_scores.csv` template.

# %%
# Scoring rubric (print for reference)
rubric = """
PERICARDIAL LINE VISIBILITY SCORING RUBRIC
==========================================
Score 1 — CLEAR: Pericardial sac outline visible as a continuous bright line in
          ≥3 consecutive mid-heart axial slices. Both inner boundary (epicardium)
          and outer pericardium are traceable. Suitable for SAM2 prompting.

Score 2 — PARTIAL: Pericardium visible on some slices or as a partial arc.
          May still be usable with careful prompting. Include unless borderline.

Score 3 — POOR / HOLD OUT: Pericardium invisible, FOV truncated, severe motion
          artifact, or anatomical variant. Never use for training. Document reason.
"""
print(rubric)

# %%
# Helper: get 4 representative slice indices for a cropped volume
def get_slice_indices(ct_crop: np.ndarray) -> list[int]:
    nz = ct_crop.shape[2]
    return [max(0, nz//5), max(0, 2*nz//5), min(nz-1, 3*nz//5), min(nz-1, 4*nz//5)]


# %%
# Fig 1: 6-patient grid (page 1 of 5), mediastinal window
PATIENTS_PER_PAGE = 6
n_pages = (len(sample_ids) + PATIENTS_PER_PAGE - 1) // PATIENTS_PER_PAGE

for page in range(n_pages):
    page_ids = sample_ids[page * PATIENTS_PER_PAGE : (page + 1) * PATIENTS_PER_PAGE]
    n_rows = len(page_ids)
    n_cols = 4  # 4 slices per patient

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        f"Pericardial visibility scoring — patients {page*PATIENTS_PER_PAGE+1}–"
        f"{min((page+1)*PATIENTS_PER_PAGE, len(sample_ids))} of {len(sample_ids)}\n"
        "Mediastinal window [-160, +240] HU  |  Score 1=clear  2=partial  3=poor",
        fontsize=9, y=1.01
    )

    for row, pid in enumerate(page_ids):
        p = patients[pid]
        ct_c = p["ct_crop"]
        slices = get_slice_indices(ct_c)
        global_idx = page * PATIENTS_PER_PAGE + row + 1  # 1-based row number for CSV
        for col, z in enumerate(slices):
            img = window_hu(ct_c[:, :, z], lo=WIN_LO, hi=WIN_HI)
            ax = axes[row, col]
            ax.imshow(img.T, cmap="gray", origin="lower")
            ax.axis("off")
            if col == 0:
                # Scan ID + row number as text overlay on the first slice
                ax.text(
                    0.02, 0.97, f"#{global_idx}  {pid}",
                    transform=ax.transAxes,
                    fontsize=6.5, color="yellow", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, lw=0),
                )
            ax.text(
                0.98, 0.03, f"z={z}",
                transform=ax.transAxes,
                fontsize=5.5, color="white", va="bottom", ha="right",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname = FIGURES_DIR / f"fig01_scoring_grid_page{page+1:02d}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname.name}")

# %% [markdown]
# ## Generate Scoring Template CSV
#
# Open `data/quality_scores.csv` and fill in the `quality_score` column (1/2/3)
# and optionally a `notes` field (e.g. "FOV truncated", "motion blur", "clear").

# %%
# Check if scores CSV already exists — don't overwrite existing scores
if SCORES_CSV.exists():
    print(f"Scores CSV already exists: {SCORES_CSV} — not overwriting")
    print("To add new patients, append rows manually.")
else:
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scan_id", "quality_score", "notes"])
        writer.writeheader()
        for pid in sample_ids:
            writer.writerow({"scan_id": pid, "quality_score": "", "notes": ""})
    print(f"Created scoring template: {SCORES_CSV}")
    print(f"Open it and fill in quality_score (1/2/3) for each patient.")

# %% [markdown]
# ## Evaluation — Read back scores (run after filling in CSV)
#
# Re-run this section after entering scores in quality_scores.csv.

# %%
def load_scores(csv_path: Path) -> dict[str, int]:
    """Load quality_scores.csv → {scan_id: score} for scored patients."""
    scores = {}
    if not csv_path.exists():
        return scores
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["quality_score"].strip().isdigit():
                scores[row["scan_id"]] = int(row["quality_score"])
    return scores

scores = load_scores(SCORES_CSV)
print(f"Scored so far: {len(scores)} patients")

if scores:
    from collections import Counter
    dist = Counter(scores.values())
    total = sum(dist.values())
    print("Score distribution:")
    for s in [1, 2, 3]:
        n = dist.get(s, 0)
        pct = 100 * n / total
        label = {1: "clear", 2: "partial", 3: "poor"}[s]
        print(f"  Score {s} ({label}): {n} ({pct:.0f}%)")

    # Extrapolate to full 785-patient cohort
    if total >= 10:
        print("\nExtrapolated to 785 patients:")
        for s in [1, 2, 3]:
            n = dist.get(s, 0)
            est = int(785 * n / total)
            print(f"  Score {s}: ~{est} patients")

# %%
results = {
    "n_available": len(available),
    "n_sampled": len(sample_ids),
    "n_scored": len(scores),
    "score_distribution": {str(k): v for k, v in Counter(scores.values()).items()} if scores else {},
}

with open(METRICS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Metrics: {METRICS_PATH}")

# %% [markdown]
# ## Interpretation
#
# **After scoring:**
# - Score 1+2 patients → eligible for SAM2 bootstrapping (exp-005)
# - Score 3 patients → hard cases, hold out, document
# - Target: ≥500 score-1 patients for a robust training set
#
# If fewer than 400 score-1 patients exist → raise with Katy;
# may need to include score-2 or pivot manual annotation strategy.
#
# **Next:** Run `src/data/splits.py` to generate stratified train/val splits
# from the quality-filtered set.
