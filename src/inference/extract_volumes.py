"""
Extract per-patient epi/para fat volumes from nnUNet predictions.

Usage:
    python src/inference/extract_volumes.py \
        --pred-dir data/COCA_dataset/nnunet_pred \
        --input-dir data/COCA_dataset/nnunet_input \
        --out data/COCA_dataset/eat_volumes.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

EPI_LABEL = 1
PARA_LABEL = 2


def extract_volumes(pred_path: Path, input_path: Path) -> dict:
    pred = nib.load(pred_path).get_fdata(dtype=np.float32).astype(np.int32)
    inp = nib.load(input_path)
    zooms = inp.header.get_zooms()
    vox_mL = float(np.prod(zooms[:3])) / 1000.0

    epi_vox = int((pred == EPI_LABEL).sum())
    para_vox = int((pred == PARA_LABEL).sum())
    total_vox = epi_vox + para_vox

    return {
        "scan_id": pred_path.stem,
        "epi_vox": epi_vox,
        "para_vox": para_vox,
        "total_vox": total_vox,
        "epi_mL": round(epi_vox * vox_mL, 2),
        "para_mL": round(para_vox * vox_mL, 2),
        "total_mL": round(total_vox * vox_mL, 2),
        "vox_size_mm3": round(float(np.prod(zooms[:3])), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    pred_files = sorted([
        f for f in args.pred_dir.glob("*.nii.gz")
        if not f.name.startswith(("dataset", "plans", "predict"))
    ])
    print(f"Found {len(pred_files)} predictions")

    rows = []
    missing_input = []
    for pred_path in tqdm(pred_files):
        scan_id = pred_path.stem.replace(".nii", "")
        input_path = args.input_dir / f"{scan_id}_0000.nii.gz"
        if not input_path.exists():
            missing_input.append(scan_id)
            continue
        rows.append(extract_volumes(pred_path, input_path))

    rows.sort(key=lambda r: r["scan_id"])

    fieldnames = ["scan_id", "epi_mL", "para_mL", "total_mL", "epi_vox", "para_vox", "total_vox", "vox_size_mm3"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    epi_vals = [r["epi_mL"] for r in rows]
    para_vals = [r["para_mL"] for r in rows]
    total_vals = [r["total_mL"] for r in rows]
    print(f"\nWrote {len(rows)} rows to {args.out}")
    if missing_input:
        print(f"Missing input files for: {missing_input}")
    print(f"\n=== Volume summary ===")
    print(f"Epi   mL:   mean={np.mean(epi_vals):.1f}  median={np.median(epi_vals):.1f}  range=[{min(epi_vals):.1f}, {max(epi_vals):.1f}]")
    print(f"Para  mL:   mean={np.mean(para_vals):.1f}  median={np.median(para_vals):.1f}  range=[{min(para_vals):.1f}, {max(para_vals):.1f}]")
    print(f"Total mL:   mean={np.mean(total_vals):.1f}  median={np.median(total_vals):.1f}  range=[{min(total_vals):.1f}, {max(total_vals):.1f}]")


if __name__ == "__main__":
    main()
