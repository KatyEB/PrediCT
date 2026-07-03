r"""
generate_roi_cropped_dataset.py — Full pipeline: TotalSegmentator heart
detection -> crop box -> crop existing CT + calcium label together.

Order matters and is fixed by design:
  1. Label already exists (_seg.nii.gz, built from XML once, at native
     resolution — untouched here).
  2. TotalSegmentator finds the heart on the RAW CT.
  3. Box computed from that heart mask, expanded by a physical margin.
  4. CT and the EXISTING label are cropped TOGETHER, same index, same
     size, via SimpleITK.RegionOfInterest — metadata stays correct,
     nothing is re-derived from XML.

Resumable: any patient whose output meta.json already exists is skipped,
so an interrupted run can just be restarted with the same command.

USAGE:
  python generate_roi_cropped_dataset.py \
      --images_root "C:\...\data_canonical\images" \
      --splits_dir  "C:\...\data_canonical\tables" \
      --out_root    "C:\...\data_canonical\images_roi" \
      --margin_mm 8
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# ── Step 1: TotalSegmentator ──────────────────────────────────────
def run_totalsegmentator(ct_path, tmp_out_dir):
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    heart_path = tmp_out_dir / "heart.nii.gz"
    if heart_path.exists():
        return heart_path

    cmd = ["TotalSegmentator", "-i", str(ct_path), "-o", str(tmp_out_dir),
        "--roi_subset", "heart"]
    # No capture_output -- let TotalSegmentator's own device-detection
    # messages print live to the console, so we can actually see them.
    result = subprocess.run(cmd)
    if result.returncode != 0 or not heart_path.exists():
        raise RuntimeError(f"TotalSegmentator failed (exit code {result.returncode})")
    return heart_path


# ── Step 2: box from heart mask, physical margin per axis ────────
def compute_crop_box(heart_mask_sitk, margin_mm):
    arr = sitk.GetArrayFromImage(heart_mask_sitk)  # (z, y, x)
    nz, ny, nx = np.where(arr > 0)
    if len(nz) == 0:
        return None, None

    spacing = heart_mask_sitk.GetSpacing()  # (x, y, z)
    margin_vox = [int(round(margin_mm / spacing[i])) for i in range(3)]

    size = heart_mask_sitk.GetSize()  # (x, y, z)
    x0 = max(0, nx.min() - margin_vox[0]); x1 = min(size[0], nx.max()+1+margin_vox[0])
    y0 = max(0, ny.min() - margin_vox[1]); y1 = min(size[1], ny.max()+1+margin_vox[1])
    z0 = max(0, nz.min() - margin_vox[2]); z1 = min(size[2], nz.max()+1+margin_vox[2])

    index = [int(x0), int(y0), int(z0)]
    box_size = [int(x1-x0), int(y1-y0), int(z1-z0)]
    return index, box_size


# ── Step 3: crop CT + existing label together, metadata-correct ──
def crop_with_metadata(image_sitk, index, box_size):
    return sitk.RegionOfInterest(image_sitk, box_size, index)


def process_patient(scan_id, images_root, out_root, margin_mm):
    out_dir = out_root / scan_id
    meta_path = out_dir / f"{scan_id}_meta.json"
    if meta_path.exists():
        return "skipped_done"

    ct_path = images_root / scan_id / f"{scan_id}_img.nii.gz"
    label_path = images_root / scan_id / f"{scan_id}_seg.nii.gz"
    if not ct_path.exists() or not label_path.exists():
        return "missing_input"

    tmp_dir = out_root / "_tmp_totalseg_raw" / scan_id
    heart_path = run_totalsegmentator(ct_path, tmp_dir)

    heart_sitk = sitk.ReadImage(str(heart_path))
    index, box_size = compute_crop_box(heart_sitk, margin_mm)
    if index is None:
        return "no_heart_found"

    ct = sitk.ReadImage(str(ct_path))
    label = sitk.ReadImage(str(label_path))

    calcium_before = int(sitk.GetArrayFromImage(label).sum())
    cropped_ct = crop_with_metadata(ct, index, box_size)
    cropped_label = crop_with_metadata(label, index, box_size)
    calcium_after = int(sitk.GetArrayFromImage(cropped_label).sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(cropped_ct, str(out_dir / f"{scan_id}_img.nii.gz"), useCompression=True)
    sitk.WriteImage(cropped_label, str(out_dir / f"{scan_id}_seg.nii.gz"), useCompression=True)

    meta = {
        "scan_id": scan_id,
        "crop_index_xyz": index,
        "crop_size_xyz": box_size,
        "original_size_xyz": list(ct.GetSize()),
        "spacing_xyz": list(ct.GetSpacing()),
        "margin_mm": margin_mm,
        "calcium_voxels_before": calcium_before,
        "calcium_voxels_after": calcium_after,
        "calcium_preserved": calcium_after >= calcium_before,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    if calcium_after < calcium_before:
        return "calcium_loss"
    return "ok"


def resolve_patient_list(args):
    if args.splits_dir:
        ids, seen = [], set()
        for name in ["train_split", "val_split", "test_split"]:
            df = pd.read_parquet(Path(args.splits_dir) / f"{name}.parquet")
            id_col = next(c for c in ["scan_id", "patient_id", "id"] if c in df.columns)
            for sid in df[id_col].astype(str):
                if sid not in seen:
                    ids.append(sid); seen.add(sid)
        return ids
    else:
        return sorted(p.name for p in Path(args.images_root).iterdir() if p.is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--splits_dir", default=None,
                    help="if given, process exactly the patients in train/val/test_split.parquet")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--margin_mm", type=float, default=8.0)
    args = ap.parse_args()

    images_root = Path(args.images_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    patient_ids = resolve_patient_list(args)
    print(f"About to process {len(patient_ids)} patients.")
    print(f"(If this number looks wrong, stop now and check --splits_dir vs --images_root.)\n")

    counts = {}
    calcium_loss_patients = []

    for sid in tqdm(patient_ids, desc="Patients"):
        try:
            status = process_patient(sid, images_root, out_root, args.margin_mm)
        except Exception as e:
            status = "error"
            print(f"\n  [ERROR] {sid}: {e}")
        counts[status] = counts.get(status, 0) + 1
        if status == "calcium_loss":
            calcium_loss_patients.append(sid)

    print(f"\n{'='*60}")
    print(f"Processed: {len(patient_ids)} total")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if calcium_loss_patients:
        print(f"\n[WARNING] calcium lost on crop for {len(calcium_loss_patients)} patients "
              f"— margin_mm={args.margin_mm} may be too small for these:")
        for sid in calcium_loss_patients:
            print(f"    {sid}")
        print("  Increase --margin_mm and rerun (already-good patients will be skipped).")
    print(f"\nOutput -> {out_root}")


if __name__ == "__main__":
    main()