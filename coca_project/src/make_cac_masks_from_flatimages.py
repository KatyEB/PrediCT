#!/usr/bin/env python3
"""
make_cac_masks_from_flat_images.py

Build per-case folders with:
  out_root/
    case_<patient_folder>__<seriesuid-short>/
      ct.nii.gz
      seg.nii.gz

Inputs:
- ROI table: Parquet/CSV with columns:
    xml_path, image_index, point_px
  Optional: series_uid column (recommended if multiple series per patient)
- Flat images directory containing paired files:
    <id>.nii.gz and <id>.json
  JSON contains patient folder and SeriesInstanceUID (key names configurable)

Rasterizes per-slice polygons (Point_px) into a 3D segmentation NIfTI aligned to CT.
"""

import argparse
import ast
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import nibabel as nib

try:
    from skimage.draw import polygon as sk_polygon
except ImportError as e:
    raise SystemExit("Missing scikit-image. Install: pip install scikit-image") from e


_NUM_PAIR_RE = re.compile(r"\(?\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)?")

def parse_points_field(val) -> List[Tuple[float, float]]:
    """Parse polygon points into list of (x,y) floats from many encodings."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple)):
        pts: List[Tuple[float, float]] = []
        for item in val:
            pts.extend(parse_points_field(item))
        if len(pts) == 0 and len(val) > 0 and isinstance(val[0], (list, tuple)) and len(val[0]) == 2:
            return [(float(p[0]), float(p[1])) for p in val]
        return pts

    s = str(val).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return []

    # Try literal eval first
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            pts: List[Tuple[float, float]] = []
            for item in obj:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    pts.append((float(item[0]), float(item[1])))
                else:
                    pts.extend(parse_points_field(item))
            return pts
        if isinstance(obj, str):
            s = obj.strip()
    except Exception:
        pass

    matches = _NUM_PAIR_RE.findall(s)
    return [(float(x), float(y)) for x, y in matches]


def safe_case_name(s: str) -> str:
    s = str(s).strip().replace("\\", "/").strip("/")
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s if s else "unknown"


def copy_or_link(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if suf == ".tsv" else ",")
    raise SystemExit("ROI table must be .parquet, .csv, or .tsv")


def get_nested(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    """Return first matching value for any dotted key path in keys."""
    for k in keys:
        cur: Any = d
        ok = True
        for part in k.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def build_image_index(images_dir: Path,
                      json_patient_keys: List[str],
                      json_series_keys: List[str]) -> Tuple[
                          Dict[Tuple[str, str], Path],
                          Dict[str, List[Tuple[str, Path]]]
                      ]:
    """
    Index JSON+NIfTI pairs.
    Returns:
      by_patient_series[(patient_folder, series_uid)] = nii_path
      by_patient[patient_folder] = [(series_uid, nii_path), ...]
    """
    by_patient_series: Dict[Tuple[str, str], Path] = {}
    by_patient: Dict[str, List[Tuple[str, Path]]] = {}

    json_paths = sorted(images_dir.glob("*.json"))
    for jp in json_paths:
        stem = jp.stem
        nii_candidates = []
        # support .nii.gz primarily, but accept .nii
        p1 = images_dir / f"{stem}.nii.gz"
        p2 = images_dir / f"{stem}.nii"
        if p1.exists():
            nii_candidates.append(p1)
        if p2.exists():
            nii_candidates.append(p2)
        if len(nii_candidates) == 0:
            continue
        nii_path = nii_candidates[0]

        try:
            meta = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue

        patient_folder = get_nested(meta, json_patient_keys)
        series_uid = get_nested(meta, json_series_keys)

        if patient_folder is None or series_uid is None:
            # can't index reliably
            continue

        patient_folder = str(patient_folder)
        series_uid = str(series_uid)

        by_patient_series[(patient_folder, series_uid)] = nii_path
        by_patient.setdefault(patient_folder, []).append((series_uid, nii_path))

    # make deterministic order in by_patient
    for pf in by_patient:
        by_patient[pf] = sorted(by_patient[pf], key=lambda t: (t[0], str(t[1])))

    return by_patient_series, by_patient


def detect_z_axis(ct_shape: Tuple[int, int, int], max_image_index: int) -> int:
    """Choose axis whose length best matches max_image_index+1."""
    target = max_image_index + 1
    diffs = [abs(ct_shape[a] - target) for a in range(3)]
    return int(np.argmin(diffs))


def fill_polygon(mask_rc: np.ndarray, pts_xy: List[Tuple[float, float]]) -> None:
    """mask_rc shape (rows, cols), pts_xy are (x,y)."""
    if len(pts_xy) < 3:
        return
    xs = np.array([p[0] for p in pts_xy], dtype=np.float32)
    ys = np.array([p[1] for p in pts_xy], dtype=np.float32)
    rr, cc = sk_polygon(ys, xs, shape=mask_rc.shape)
    mask_rc[rr, cc] = 1


def make_slice_mask(ct_shape: Tuple[int, int, int], z_axis: int, pts_xy: List[Tuple[float, float]]) -> np.ndarray:
    """
    Create slice mask in the plane orthogonal to z_axis.
    Assumes Point_px is defined on the axial slice plane in pixel coords.
    """
    a0, a1, a2 = ct_shape
    if z_axis == 2:
        X, Y = a0, a1
        mask_yx = np.zeros((Y, X), dtype=np.uint8)
        fill_polygon(mask_yx, pts_xy)
        return mask_yx.T  # (X,Y)
    if z_axis == 0:
        Y, X = a1, a2
        mask_yx = np.zeros((Y, X), dtype=np.uint8)
        fill_polygon(mask_yx, pts_xy)
        return mask_yx  # (Y,X)
    if z_axis == 1:
        R, C = a0, a2
        mask_rc = np.zeros((R, C), dtype=np.uint8)
        fill_polygon(mask_rc, pts_xy)
        return mask_rc  # (axis0, axis2)
    raise ValueError("z_axis must be 0,1,2")


def assign_slice(seg: np.ndarray, z_axis: int, z: int, slice_mask: np.ndarray) -> None:
    """Write slice_mask into seg at slice index z along z_axis."""
    if z_axis == 2:
        seg[..., z] = np.maximum(seg[..., z], slice_mask.astype(seg.dtype))
    elif z_axis == 0:
        seg[z, :, :] = np.maximum(seg[z, :, :], slice_mask.astype(seg.dtype))
    elif z_axis == 1:
        seg[:, z, :] = np.maximum(seg[:, z, :], slice_mask.astype(seg.dtype))
    else:
        raise ValueError("z_axis must be 0,1,2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-table", required=True, help="Parquet/CSV/TSV with ROIs (xml_path, image_index, point_px).")
    ap.add_argument("--images-dir", required=True, help="Flat folder containing <id>.nii(.gz) and <id>.json pairs.")
    ap.add_argument("--out-root", required=True, help="Output folder for case_*/ct.nii.gz and seg.nii.gz")

    ap.add_argument("--xml-path-col", default="xml_path")
    ap.add_argument("--image-index-col", default="image_index")
    ap.add_argument("--points-col", default="point_px")

    ap.add_argument("--series-uid-col", default=None,
                    help="Optional series UID column in ROI table. Strongly recommended if multiple series per patient.")

    ap.add_argument("--xml-to-patient-regex", default=None,
                    help="Optional regex with one capture group to extract patient_folder from xml_path. "
                         "If omitted, uses parent folder name of xml_path.")

    ap.add_argument("--json-patient-keys", default="patient_folder,patientFolder,patient.folder,PatientFolder",
                    help="Comma-separated JSON key paths for patient folder (supports dotted paths).")
    ap.add_argument("--json-series-keys", default="series_uid,SeriesInstanceUID,seriesInstanceUID,series.uid,SeriesUID",
                    help="Comma-separated JSON key paths for series UID (supports dotted paths).")

    ap.add_argument("--flip-z", action="store_true", help="If ImageIndex is reversed relative to NIfTI z order.")
    ap.add_argument("--z-offset", type=int, default=0, help="Constant offset added to ImageIndex.")
    ap.add_argument("--z-axis", type=int, default=-1, help="Force z-axis 0/1/2. Default auto.")
    ap.add_argument("--limit-cases", type=int, default=20, help="Process first N cases (default 20).")
    ap.add_argument("--link-ct", action="store_true", help="Hardlink CT into output (same filesystem only).")

    args = ap.parse_args()

    roi_table = Path(args.roi_table)
    images_dir = Path(args.images_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df = read_table(roi_table)

    for col in [args.xml_path_col, args.image_index_col, args.points_col]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}'. Available: {list(df.columns)}")

    if args.series_uid_col is not None and args.series_uid_col not in df.columns:
        raise SystemExit(f"series-uid-col '{args.series_uid_col}' not found. Available: {list(df.columns)}")

    # Derive patient folder key from xml_path
    if args.xml_to_patient_regex:
        rx = re.compile(args.xml_to_patient_regex)
        def extract_patient(p: str) -> str:
            m = rx.search(str(p).replace("\\", "/"))
            return m.group(1) if m else str(Path(str(p)).parent)
        df["_patient_folder"] = df[args.xml_path_col].apply(extract_patient).astype(str)
    else:
        def parent_name(p: str) -> str:
            p = str(p).replace("\\", "/")
            return Path(p).parent.name if Path(p).parent.name else p
        df["_patient_folder"] = df[args.xml_path_col].apply(parent_name).astype(str)

    # Index images
    json_patient_keys = [k.strip() for k in args.json_patient_keys.split(",") if k.strip()]
    json_series_keys = [k.strip() for k in args.json_series_keys.split(",") if k.strip()]
    by_ps, by_p = build_image_index(images_dir, json_patient_keys, json_series_keys)

    # Group cases by patient (and series if available)
    if args.series_uid_col:
        df["_series_uid"] = df[args.series_uid_col].astype(str)
        group_cols = ["_patient_folder", "_series_uid"]
    else:
        group_cols = ["_patient_folder"]

    grouped = df.groupby(group_cols, dropna=False)

    # Select up to N groups
    group_keys = list(grouped.groups.keys())[: max(0, int(args.limit_cases))]

    skipped_ambiguous = 0
    skipped_missing = 0
    processed = 0

    for gk in group_keys:
        if args.series_uid_col:
            patient_folder, series_uid = gk
            series_uid = str(series_uid)
        else:
            patient_folder = gk
            series_uid = None

        patient_folder = str(patient_folder)

        # Resolve CT nii
        ct_path: Optional[Path] = None
        if series_uid is not None:
            ct_path = by_ps.get((patient_folder, series_uid))
            if ct_path is None:
                skipped_missing += 1
                print(f"[SKIP missing] No CT for patient='{patient_folder}' series='{series_uid}'")
                continue
        else:
            cands = by_p.get(patient_folder, [])
            if len(cands) == 0:
                skipped_missing += 1
                print(f"[SKIP missing] No CT candidates for patient='{patient_folder}'")
                continue
            if len(cands) > 1:
                skipped_ambiguous += 1
                print(f"[SKIP ambiguous] patient='{patient_folder}' has {len(cands)} series; "
                      f"provide --series-uid-col to disambiguate.")
                continue
            series_uid, ct_path = cands[0]

        assert ct_path is not None
        case_df = grouped.get_group(gk).copy()

        # Load CT
        ct_img = nib.load(str(ct_path))
        ct_data = ct_img.get_fdata(dtype=np.float32)
        if ct_data.ndim != 3:
            print(f"[SKIP] CT not 3D: {ct_path} shape={ct_data.shape}")
            continue
        ct_shape = ct_data.shape

        max_idx = int(case_df[args.image_index_col].max())
        z_axis = args.z_axis if args.z_axis in (0, 1, 2) else detect_z_axis(ct_shape, max_idx)
        Z = ct_shape[z_axis]

        seg = np.zeros(ct_shape, dtype=np.uint8)

        for _, row in case_df.iterrows():
            pts = parse_points_field(row[args.points_col])
            if len(pts) < 3:
                continue

            idx = int(row[args.image_index_col]) + int(args.z_offset)
            if args.flip_z:
                idx = (Z - 1) - idx
            if idx < 0 or idx >= Z:
                continue

            slice_mask = make_slice_mask(ct_shape, z_axis, pts)
            assign_slice(seg, z_axis, idx, slice_mask)

        # Write output case folder
        pf_safe = safe_case_name(patient_folder)
        su_short = safe_case_name(str(series_uid))[:12] if series_uid else "noseries"
        case_dir = out_root / f"case_{pf_safe}__{su_short}"
        case_dir.mkdir(parents=True, exist_ok=True)

        ct_out = case_dir / "ct.nii.gz"
        if not ct_out.exists():
            copy_or_link(ct_path, ct_out, link=args.link_ct)

        seg_img = nib.Nifti1Image(seg.astype(np.uint8), affine=ct_img.affine, header=ct_img.header)
        seg_img.set_data_dtype(np.uint8)
        seg_out = case_dir / "seg.nii.gz"
        nib.save(seg_img, str(seg_out))

        processed += 1
        print(f"[OK] {case_dir.name} z_axis={z_axis} flip_z={args.flip_z} seg_voxels={int(seg.sum())}")

    print(f"Done. processed={processed} skipped_missing={skipped_missing} skipped_ambiguous={skipped_ambiguous}")
    print(f"Output: {out_root}")

if __name__ == "__main__":
    main()
