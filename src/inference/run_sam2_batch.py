
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loaders import crop_to_roi, get_cardiac_roi_bbox, load_coca_patient
from src.data.splits import load_quality_scores
from src.inference.sam2_predictor import load_sam2_predictor, predict_pericardium_volume

DATA_ROOT = REPO_ROOT / "data" / "COCA_dataset"
RESAMPLED_DIR = DATA_ROOT / "data_resampled"
HEART_MASKS_DIR = DATA_ROOT / "heart_masks"
PERI_MASKS_DIR = DATA_ROOT / "pericardium_masks"

DIL = (29, 29, 7)
BOX_PAD = 8
MIN_HEART_VOX = 200
WIN_LO = -160.0
WIN_HI = 240.0
MODEL_ID = "facebook/sam2.1-hiera-base-plus"
DEVICE = "mps"


def process_patient(pid: str, predictor) -> dict:
    out_path = PERI_MASKS_DIR / pid / "pericardium_roi.nii.gz"
    if out_path.exists():
        print(f"  {pid}: already done, skipping")
        return None

    p = load_coca_patient(pid, resampled_dir=RESAMPLED_DIR, heart_masks_dir=HEART_MASKS_DIR)
    if p["heart_mask"] is None:
        print(f"  {pid}: no heart mask, skipping")
        return None

    roi = get_cardiac_roi_bbox(p["heart_mask"], dilation_vox=DIL, array_shape=p["ct"].shape)
    crop = crop_to_roi(p["ct"], roi, heart_mask=p["heart_mask"])

    t0 = time.time()
    peri_mask = predict_pericardium_volume(
        crop["ct_crop"], crop["heart_mask_crop"], predictor,
        box_pad=BOX_PAD, min_heart_vox=MIN_HEART_VOX, win_lo=WIN_LO, win_hi=WIN_HI,
    )
    elapsed = time.time() - t0

    (PERI_MASKS_DIR / pid).mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(peri_mask, affine=np.eye(4)), out_path)

    n_slices = int((crop["heart_mask_crop"].sum(axis=(0, 1)) >= MIN_HEART_VOX).sum())
    n_pred = int((peri_mask.sum(axis=(0, 1)) > 0).sum())
    vol_ml = float(peri_mask.sum()) * np.prod(p["spacing"]) / 1000

    print(f"  {pid}: {n_pred}/{n_slices} slices | vol={vol_ml:.0f}mL | {elapsed:.1f}s", flush=True)
    return {
        "scan_id": pid,
        "heart_slices": n_slices,
        "predicted_slices": n_pred,
        "pericardium_vol_ml": round(vol_ml, 1),
        "time_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch SAM2 pericardium mask generation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pids", nargs="+", help="Explicit patient IDs to process")
    group.add_argument("--from-file", type=Path, metavar="FILE",
                       help="Text file with one patient ID per line")
    group.add_argument("--score", nargs="+", type=int, metavar="N",
                       help="Process all patients with quality score(s) N (1, 2, or 3)")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional path to write per-patient results JSON")
    args = parser.parse_args()

    if args.score is not None:
        scores = load_quality_scores()
        pids = [pid for pid, s in scores.items() if s in args.score]
        print(f"Score-{args.score} patients: {len(pids)} found")
    elif args.from_file is not None:
        pids = [l.strip() for l in args.from_file.read_text().splitlines() if l.strip()]
        print(f"Loaded {len(pids)} patient IDs from {args.from_file}")
    else:
        pids = args.pids

    if not pids:
        print("No patients to process.")
        return

    PERI_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading SAM2 ({MODEL_ID})...")
    predictor = load_sam2_predictor(model_id=MODEL_ID, device=DEVICE)
    print(f"Processing {len(pids)} patients...\n")

    results = []
    for pid in pids:
        r = process_patient(pid, predictor)
        if r is not None:
            results.append(r)

    if results:
        avg_t = sum(r["time_s"] for r in results) / len(results)
        total_slices = sum(r["heart_slices"] for r in results)
        total_pred = sum(r["predicted_slices"] for r in results)
        print(f"\nDone: {len(results)} new patients processed")
        print(f"Slice coverage: {total_pred}/{total_slices} ({100*total_pred/total_slices:.1f}%)")
        print(f"Avg time: {avg_t:.1f}s/patient")

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results written to {args.output_json}")
    else:
        print("No new patients processed (all already done or skipped).")


if __name__ == "__main__":
    main()
