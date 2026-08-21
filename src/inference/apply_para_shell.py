"""
Apply 10mm pericardium shell post-processing to fix para fat volumes.

For patients with a pericardium mask, restrict label-2 (para fat) to voxels
within 10 mm outside the pericardial boundary (distance transform approach).

Updates data/COCA_dataset/eat_volumes.csv in-place.

Usage:
    python src/inference/apply_para_shell.py [--shell-mm 10]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "COCA_dataset"
PRED_DIR = DATA / "nnunet_pred"
INP_DIR = DATA / "nnunet_input"
PERI_DIR = DATA / "pericardium_masks"
CSV_PATH = DATA / "eat_volumes.csv"

EPI_LABEL = 1
PARA_LABEL = 2


def apply_shell(scan_id_csv: str, shell_mm: float) -> dict | None:
    """Return updated volume fields for one patient, or None if no pericardium mask."""
    bare_id = scan_id_csv.replace(".nii", "")
    peri_path = PERI_DIR / bare_id / "pericardium_roi.nii.gz"
    if not peri_path.exists():
        return None

    pred_path = PRED_DIR / f"{bare_id}.nii.gz"
    inp_path = INP_DIR / f"{bare_id}_0000.nii.gz"
    if not pred_path.exists() or not inp_path.exists():
        return None

    pred = nib.load(pred_path).get_fdata(dtype=np.float32).astype(np.int32)
    peri = nib.load(peri_path).get_fdata(dtype=np.float32) > 0.5
    inp_img = nib.load(inp_path)
    # Use input file zooms (real physical spacing), NOT peri zooms (stored as 1mm identity)
    zooms = inp_img.header.get_zooms()
    vox_mL = float(np.prod(zooms[:3])) / 1000.0
    spacing = tuple(float(z) for z in zooms[:3])

    # Distance (mm) from each voxel to the pericardium surface, measured outside only
    # distance_transform_edt returns 0 inside the mask, positive outside
    dist_outside = distance_transform_edt(~peri, sampling=spacing)
    shell_mask = (dist_outside > 0) & (dist_outside <= shell_mm)

    para_mask = (pred == PARA_LABEL) & shell_mask
    epi_vox = int((pred == EPI_LABEL).sum())
    para_vox = int(para_mask.sum())
    total_vox = epi_vox + para_vox

    return {
        "epi_vox": epi_vox,
        "para_vox": para_vox,
        "total_vox": total_vox,
        "epi_mL": round(epi_vox * vox_mL, 2),
        "para_mL": round(para_vox * vox_mL, 2),
        "total_mL": round(total_vox * vox_mL, 2),
        "para_note": f"corrected_{int(shell_mm)}mm_shell",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell-mm", type=float, default=10.0)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    args = parser.parse_args()

    with open(args.csv) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Ensure annotation columns exist
    if "para_note" not in fieldnames:
        fieldnames.append("para_note")
        for row in rows:
            row.setdefault("para_note", "no_pericardium_mask")
    if "epi_flag" not in fieldnames:
        fieldnames.append("epi_flag")
        for row in rows:
            row["epi_flag"] = "high_outlier" if float(row["epi_mL"]) > 350 else "ok"

    updated = 0
    for row in tqdm(rows, desc="Applying shell correction"):
        result = apply_shell(row["scan_id"], args.shell_mm)
        if result is None:
            continue
        row.update(result)
        updated += 1

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    corrected = [r for r in rows if "corrected" in r.get("para_note", "")]
    para_vals = [float(r["para_mL"]) for r in corrected]
    epi_vals = [float(r["epi_mL"]) for r in corrected]
    all_para = [float(r["para_mL"]) for r in rows]

    print(f"\nUpdated {updated} patients")
    if corrected:
        print(f"\n=== Para fat — corrected subset ({len(corrected)} patients) ===")
        print(f"  mean={np.mean(para_vals):.1f}  median={np.median(para_vals):.1f}  "
              f"range=[{min(para_vals):.1f}, {max(para_vals):.1f}]")
        print(f"\n=== Epi fat — corrected subset ===")
        print(f"  mean={np.mean(epi_vals):.1f}  median={np.median(epi_vals):.1f}  "
              f"range=[{min(epi_vals):.1f}, {max(epi_vals):.1f}]")
    print(f"\n=== Para fat — full cohort ({len(rows)} patients, uncorrected still present) ===")
    print(f"  mean={np.mean(all_para):.1f}  median={np.median(all_para):.1f}")


if __name__ == "__main__":
    main()
