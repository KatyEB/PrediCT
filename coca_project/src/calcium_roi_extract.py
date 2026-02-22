# calcium_roi_extract.py
#
# Extract calcium ROI annotations from plist-XML files (saved with .xml extension)
# into a single Parquet file with a stable, normalized schema.
#
# Expected schema per file:
#   top-level dict with key "Images" -> list[dict]
#   each image dict: "ImageIndex", "ROIs" -> list[dict]
#   each ROI dict: "Name", "Area", "Mean", "Max", "Point_px", "Point_mm", etc.

from __future__ import annotations

import json
import plistlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Configuration (edit as needed)
# -----------------------------
XML_DIR = Path(r"C:\coca_project\data_raw\xml\calcium_xml")
OUT_PARQUET = Path(r"C:\coca_project\derived\calcium_rois_raw.parquet")

# If True, removes placeholder ROIs where area==0 or n_points==0
DROP_EMPTY_ROIS = False


# -----------------------------
# Parsing helpers
# -----------------------------
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_point_tuple(s: str) -> Tuple[float, ...]:
    """
    Parses strings like:
      "(269.003922, 377.000000)" -> (269.003922, 377.0)
      "(38.165314, -122.242188, -151.250000)" -> (38.165314, -122.242188, -151.25)
    """
    nums = [float(x) for x in _NUM_RE.findall(s)]
    if not nums:
        raise ValueError(f"Could not parse point from: {s!r}")
    return tuple(nums)


def load_plist_xml(path: Path) -> Dict[str, Any]:
    """
    Loads an Apple plist stored as XML (even if the extension is .xml).
    """
    with open(path, "rb") as f:
        obj = plistlib.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Top-level plist is not a dict in {path}")
    return obj


def safe_points(arr: Any) -> Optional[List[Tuple[float, ...]]]:
    """
    Converts a list of "(x,y)" or "(x,y,z)" strings into a list of float tuples.
    Returns None if empty or not parseable.
    """
    if not isinstance(arr, list) or len(arr) == 0:
        return None

    pts: List[Tuple[float, ...]] = []
    for item in arr:
        if not isinstance(item, str):
            continue
        pts.append(parse_point_tuple(item))

    return pts if pts else None


def extract_rows_from_file(xml_path: Path) -> List[Dict[str, Any]]:
    """
    Extract one row per ROI, with normalized column names.
    """
    data = load_plist_xml(xml_path)
    images = data.get("Images", [])

    if not isinstance(images, list):
        return [{
            "xml_path": str(xml_path),
            "image_index": None,
            "vessel": None,
            "area": None,
            "mean_hu": None,
            "max_hu": None,
            "point_px": None,
            "point_mm": None,
            "roi_idx": None,
            "n_points": None,
            "roi_type": None,
            "total_hu": None,
            "center_mm": None,
            "raw_roi": None,
            "note": "Top-level 'Images' is not a list."
        }]

    rows: List[Dict[str, Any]] = []

    for img in images:
        if not isinstance(img, dict):
            continue

        image_index = img.get("ImageIndex", None)
        rois = img.get("ROIs", [])
        if not isinstance(rois, list):
            rois = []

        for roi in rois:
            if not isinstance(roi, dict):
                continue

            # Normalized fields
            vessel = roi.get("Name", None)
            area = roi.get("Area", None)
            mean_hu = roi.get("Mean", None)
            max_hu = roi.get("Max", None)

            point_px = safe_points(roi.get("Point_px", None))
            point_mm = safe_points(roi.get("Point_mm", None))

            roi_idx = roi.get("IndexInImage", None)
            n_points = roi.get("NumberOfPoints", None)
            roi_type = roi.get("Type", None)
            total_hu = roi.get("Total", None)
            center_mm = roi.get("Center", None)

            rows.append({
                "xml_path": str(xml_path),
                "image_index": image_index,
                "vessel": vessel,
                "area": area,
                "mean_hu": mean_hu,
                "max_hu": max_hu,
                "point_px": point_px,
                "point_mm": point_mm,
                "roi_idx": roi_idx,
                "n_points": n_points,
                "roi_type": roi_type,
                "total_hu": total_hu,
                "center_mm": center_mm,
                # Keep raw ROI dict for debugging/traceability
                "raw_roi": json.dumps(roi, default=str),
            })

    # Optional: keep a QC marker row if file contains no ROIs at all
    if not rows:
        rows.append({
            "xml_path": str(xml_path),
            "image_index": None,
            "vessel": None,
            "area": None,
            "mean_hu": None,
            "max_hu": None,
            "point_px": None,
            "point_mm": None,
            "roi_idx": None,
            "n_points": None,
            "roi_type": None,
            "total_hu": None,
            "center_mm": None,
            "raw_roi": None,
            "note": "No ROIs found in this file."
        })

    return rows


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce consistent dtypes.
    """
    int_cols = ["image_index", "roi_idx", "n_points", "roi_type"]
    float_cols = ["area", "mean_hu", "max_hu", "total_hu"]

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def main() -> None:
    if not XML_DIR.exists():
        raise FileNotFoundError(f"XML_DIR does not exist: {XML_DIR}")

    xml_files = sorted([p for p in XML_DIR.rglob("*.xml") if p.is_file()])
    if not xml_files:
        raise FileNotFoundError(f"No .xml files found under: {XML_DIR}")

    all_rows: List[Dict[str, Any]] = []
    for xp in xml_files:
        try:
            all_rows.extend(extract_rows_from_file(xp))
        except Exception as e:
            # Keep a row so failures are visible in the output parquet
            all_rows.append({
                "xml_path": str(xp),
                "image_index": None,
                "vessel": None,
                "area": None,
                "mean_hu": None,
                "max_hu": None,
                "point_px": None,
                "point_mm": None,
                "roi_idx": None,
                "n_points": None,
                "roi_type": None,
                "total_hu": None,
                "center_mm": None,
                "raw_roi": None,
                "note": f"Failed to parse file: {e}"
            })

    df = pd.DataFrame(all_rows)
    df = coerce_types(df)

    if DROP_EMPTY_ROIS:
        # Drop placeholders: either area==0 or n_points==0 (or missing)
        df = df[(df["area"].fillna(0) > 0) & (df["n_points"].fillna(0) > 0)].copy()
        df.reset_index(drop=True, inplace=True)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    # Requires pyarrow or fastparquet installed; pyarrow recommended
    df.to_parquet(OUT_PARQUET, index=False)

    print(f"Wrote {len(df):,} rows to: {OUT_PARQUET}")
    print("Columns:", df.columns.tolist())
    print(df[["vessel", "image_index", "area", "mean_hu", "max_hu"]].head(10))


if __name__ == "__main__":
    main()
