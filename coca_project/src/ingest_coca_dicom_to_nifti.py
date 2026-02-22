from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import SimpleITK as sitk




def stable_id(*parts: str, n: int = 12) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return h[:n]


def safe_get_tag(reader: sitk.ImageSeriesReader, tag: str, file_index: int = 0) -> Optional[str]:
    try:
        if reader.HasMetaDataKey(file_index, tag):
            return reader.GetMetaData(file_index, tag)
    except Exception:
        pass
    return None


def extract_geometry(image: sitk.Image) -> Dict[str, Any]:
    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
        "origin_xyz": list(image.GetOrigin()),
        "direction_3x3_flat": list(image.GetDirection()),
    }


def ingest_series(series_dir: Path, series_uid: str, out_images: Path, cohort: str, patient_folder: str) -> Optional[Dict[str, Any]]:
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_dir), series_uid)
    if not file_names:
        return None

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    image = reader.Execute()

    modality = safe_get_tag(reader, "0008|0060")  # Modality
    if modality and modality.strip().upper() != "CT":
        return None

    series_description = safe_get_tag(reader, "0008|103e")  # SeriesDescription
    protocol_name = safe_get_tag(reader, "0018|1030")       # ProtocolName
    slice_thickness = safe_get_tag(reader, "0018|0050")     # SliceThickness
    kernel = safe_get_tag(reader, "0018|1210")              # ConvolutionKernel (often helpful)

    scan_id = stable_id(str(series_dir.resolve()), series_uid)

    nifti_path = out_images / f"{scan_id}.nii.gz"
    sitk.WriteImage(image, str(nifti_path), useCompression=True)

    meta = extract_geometry(image)
    meta.update({
        "scan_id": scan_id,
        "cohort": cohort,  # gated / nongated (or whatever you call it)
        "patient_folder": patient_folder,
        "series_uid": series_uid,
        "source_series_dir": str(series_dir.resolve()),
        "num_files_in_series": len(file_names),
        "modality": modality,
        "series_description": series_description,
        "protocol_name": protocol_name,
        "slice_thickness_tag_mm": slice_thickness,
        "convolution_kernel": kernel,
        "num_slices_z": int(image.GetSize()[2]),
        "spacing_z_mm": float(image.GetSpacing()[2]),
    })

    json_path = out_images / f"{scan_id}.json"
    json_path.write_text(json.dumps(meta, indent=2))

    return {
        "scan_id": scan_id,
        "cohort": cohort,
        "patient_folder": patient_folder,
        "series_uid": series_uid,
        "source_series_dir": str(series_dir.resolve()),
        "nifti_path": str(nifti_path.resolve()),
        "json_path": str(json_path.resolve()),
        "modality": modality,
        "series_description": series_description,
        "protocol_name": protocol_name,
        "slice_thickness_tag_mm": slice_thickness,
        "convolution_kernel": kernel,
        "num_slices_z": meta["num_slices_z"],
        "spacing_x_mm": meta["spacing_xyz"][0],
        "spacing_y_mm": meta["spacing_xyz"][1],
        "spacing_z_mm": meta["spacing_xyz"][2],
        "size_x": meta["size_xyz"][0],
        "size_y": meta["size_xyz"][1],
        "size_z": meta["size_xyz"][2],
    }



def _infer_patient_folder(dicom_root: Path, series_dir: Path) -> str:
    """
    Heuristics to infer patient folder across different layouts.

    Priority:
      1) If there's an ancestor literally named 'patient', use the next part after it.
         e.g., .../patient/<patient_id>/<series_dir>
      2) Otherwise, use numeric directory names near the tail of the path.
         e.g., .../deidentified_nongated/deidentified_nongated/9/9  -> patient_folder='9'
    """
    try:
        rel_parts = series_dir.relative_to(dicom_root).parts
    except Exception:
        rel_parts = series_dir.parts

    # Rule 1: COCA-like /patient/<id>/...
    lower_parts = [p.lower() for p in rel_parts]
    if "patient" in lower_parts:
        i = lower_parts.index("patient")
        if i + 1 < len(rel_parts):
            return rel_parts[i + 1]

    # Rule 2: fallback: last numeric-ish dir in the relative path
    numeric_parts = [p for p in rel_parts if p.isdigit()]
    if numeric_parts:
        # Usually the first numeric in the trailing chain corresponds to patient id,
        # but layouts vary; this is a reasonable default.
        return numeric_parts[-2] if len(numeric_parts) >= 2 else numeric_parts[-1]

    return "unknown"


def _infer_series_name(series_dir: Path) -> str:
    return series_dir.name


def find_series_candidate_dirs(
    dicom_root: Path,
    *,
    min_dcm_files: int = 5,
) -> List[Dict[str, Any]]:
    """
    Layout-agnostic DICOM series discovery.

    Strategy:
      - Walk all *.dcm files under dicom_root (case-insensitive by checking suffix).
      - Group by parent directory (series_dir).
      - Keep directories with >= min_dcm_files.
      - Infer cohort/patient metadata from the path using heuristics.

    Returns list of dicts:
      {
        "cohort": str,
        "patient_folder": str,
        "series_dir": Path,
        "series_name": str,
        "dcm_count": int,
        "example_dcm": Path
      }
    """
    dicom_root = Path(dicom_root)

    # Group DICOM files by their parent directory
    counts: Dict[Path, int] = defaultdict(int)
    example_file: Dict[Path, Path] = {}

    # Use rglob("*") and suffix check to catch .dcm, .DCM, etc.
    SKIP_DIRS = {"gated_release_final"}

    for p in dicom_root.rglob("*"):
        # Skip gated data entirely (already processed)
        if any(part.lower() in SKIP_DIRS for part in p.parts):
            continue

        if p.is_file() and p.suffix.lower() == ".dcm":
            parent = p.parent
            counts[parent] += 1
            if parent not in example_file:
                example_file[parent] = p

    candidates: List[Dict[str, Any]] = []

    for series_dir, n in counts.items():
        if n < min_dcm_files:
            continue

        # cohort = first path component under dicom_root (if possible)
        try:
            cohort = series_dir.relative_to(dicom_root).parts[0]
        except Exception:
            cohort = "unknown"

        patient_folder = _infer_patient_folder(dicom_root, series_dir)
        
        candidates.append({
            "cohort": cohort,
            "patient_folder": patient_folder,
            "series_dir": series_dir,
            "series_name": _infer_series_name(series_dir),
            "dcm_count": n,
            "example_dcm": example_file.get(series_dir),
        })

    # Optional: sort for stable output
    candidates.sort(key=lambda d: (str(d["cohort"]), str(d["patient_folder"]), str(d["series_dir"])))
    return candidates


def main():
    project_root = Path(r"C:\coca_project")  # <- change if needed
    dicom_root = project_root / "data_raw" / "dicom"

    out_images = project_root / "data_canonical" / "images"
    out_tables = project_root / "data_canonical" / "tables"
    out_images.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    if not dicom_root.exists():
        raise FileNotFoundError(f"Missing: {dicom_root}. Copy DICOM folders there first.")

    candidates = find_series_candidate_dirs(dicom_root)
    print(f"Found {len(candidates)} series candidate directories under {dicom_root}")

    rows: List[Dict[str, Any]] = []
    for item in candidates:
        cohort = item["cohort"]
        patient_folder = item["patient_folder"]
        series_dir: Path = item["series_dir"]

        # Skip obvious non-image label folders if any slipped in
        if series_dir.name.lower() in {"calcium_xml", "xml", "labels"}:
            continue

        try:
            series_uids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(series_dir))
        except Exception:
            series_uids = None

        if not series_uids:
            continue

        for uid in series_uids:
            try:
                row = ingest_series(series_dir, uid, out_images, cohort=cohort, patient_folder=patient_folder)
                if row:
                    rows.append(row)
                    print(f"[OK] {row['scan_id']} cohort={cohort} patient={patient_folder} z={row['num_slices_z']} dir={series_dir.name}")
            except Exception as e:
                print(f"[SKIP] cohort={cohort} patient={patient_folder} dir={series_dir.name} uid={uid}: {e}")

    if not rows:
        print("No CT series ingested. If you expected some, tell me ONE example folder path that has .dcm files.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["scan_id"]).reset_index(drop=True)

    csv_path = out_tables / "scan_index.csv"
    df.to_csv(csv_path, index=False)

    parquet_path = out_tables / "scan_index.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Wrote: {parquet_path}")
    except Exception as e:
        print(f"[WARN] Parquet failed ({e}). CSV written: {csv_path}")

    print(f"Wrote: {csv_path}")
    print(f"Total ingested CT series: {len(df)}")


if __name__ == "__main__":
    main()
