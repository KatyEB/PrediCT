import re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import nibabel as nib

# ----------------------------
# Parsing helpers
# ----------------------------

def parse_triplet(s):
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if len(nums) < 3:
        return None
    return tuple(float(x) for x in nums[:3])

def extract_roi_z_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    z_vals = []

    for d in root.findall(".//dict"):
        children = list(d)
        kv = {}
        for i in range(0, len(children) - 1, 2):
            if children[i].tag == "key":
                kv[children[i].text] = children[i + 1]

        if "Center" in kv and kv["Center"].text:
            t = parse_triplet(kv["Center"].text)
            if t:
                z_vals.append(t[2])

        if "Point_mm" in kv and kv["Point_mm"].tag == "array":
            for s in kv["Point_mm"].findall("string"):
                if s.text:
                    t = parse_triplet(s.text)
                    if t:
                        z_vals.append(t[2])

    z_vals = np.array(z_vals, dtype=float)
    # downweight polygon density by using unique-ish z values
    z_vals = np.unique(np.round(z_vals, 3))
    return z_vals

def slice_z_positions_from_nifti(nifti_path, slice_axis=2):
    img = nib.load(nifti_path)
    aff = img.affine
    shape = img.shape

    if slice_axis == 2:
        ks = np.arange(shape[2])
        pts = np.stack([np.zeros_like(ks), np.zeros_like(ks), ks, np.ones_like(ks)], axis=0)
    elif slice_axis == 0:
        ks = np.arange(shape[0])
        pts = np.stack([ks, np.zeros_like(ks), np.zeros_like(ks), np.ones_like(ks)], axis=0)
    elif slice_axis == 1:
        ks = np.arange(shape[1])
        pts = np.stack([np.zeros_like(ks), ks, np.zeros_like(ks), np.ones_like(ks)], axis=0)
    else:
        raise ValueError("slice_axis must be 0, 1, or 2")

    world = aff @ pts
    return world[2, :]

def score_scan(z_roi, z_slices, spacing_z_mm=None):
    z_roi = np.asarray(z_roi)
    z_slices = np.asarray(z_slices)

    zs = np.sort(z_slices)
    idx = np.searchsorted(zs, z_roi)
    idx0 = np.clip(idx - 1, 0, len(zs) - 1)
    idx1 = np.clip(idx, 0, len(zs) - 1)
    d = np.minimum(np.abs(z_roi - zs[idx0]), np.abs(z_roi - zs[idx1]))

    tol = 2.0
    if spacing_z_mm is not None and float(spacing_z_mm) > 0:
        tol = 0.6 * float(spacing_z_mm)

    return (d <= tol).mean(), d.mean(), tol

# ----------------------------
# Main: patient-folder-only matching
# ----------------------------

def assign_xml_to_scan_per_patient(
    scans_df: pd.DataFrame,
    xml_index_df: pd.DataFrame,
    patient_col: str = "patient_folder",
    xml_path_col: str = "xml_path",
    scan_id_col: str = "scan_id",
    nifti_col: str = "nifti_path",
    spacing_col: str = "spacing_z_mm",
):
    """
    xml_index_df: one row per patient_folder containing xml_path
    scans_df: scan table with patient_folder and nifti paths
    """
    results = []

    for _, xr in xml_index_df.iterrows():
        pf = xr[patient_col]
        xml_path = xr[xml_path_col]

        # Filter to ONLY scans for this patient_folder
        cand = scans_df[scans_df[patient_col] == pf].copy()

        # If no scans or only 1 scan, skip heavy work
        if len(cand) == 0:
            results.append({
                patient_col: pf,
                "xml_path": xml_path,
                "status": "no_scans_found"
            })
            continue

        if len(cand) == 1:
            results.append({
                patient_col: pf,
                "xml_path": xml_path,
                "best_scan_id": cand.iloc[0][scan_id_col],
                "status": "single_scan_only"
            })
            continue

        # Only consider candidates with valid nifti paths
        cand = cand[cand[nifti_col].astype(str).str.len() > 0]
        if len(cand) == 0:
            results.append({
                patient_col: pf,
                "xml_path": xml_path,
                "status": "no_valid_nifti_paths"
            })
            continue

        z_roi = extract_roi_z_from_xml(xml_path)
        if len(z_roi) == 0:
            results.append({
                patient_col: pf,
                "xml_path": xml_path,
                "status": "no_roi_z_found"
            })
            continue

        scored = []
        for _, sr in cand.iterrows():
            nifti_path = sr[nifti_col]
            z_slices = slice_z_positions_from_nifti(nifti_path, slice_axis=2)
            in_tol, mae, tol = score_scan(z_roi, z_slices, spacing_z_mm=sr.get(spacing_col, None))
            scored.append({
                patient_col: pf,
                "xml_path": xml_path,
                "scan_id": sr[scan_id_col],
                "in_tol_frac": in_tol,
                "mae_mm": mae,
                "tol_mm": tol,
            })

        scored_df = pd.DataFrame(scored).sort_values(["in_tol_frac", "mae_mm"], ascending=[False, True]).reset_index(drop=True)
        best = scored_df.iloc[0]
        second = scored_df.iloc[1] if len(scored_df) > 1 else None

        out = {
            patient_col: pf,
            "xml_path": xml_path,
            "best_scan_id": best["scan_id"],
            "best_in_tol_frac": float(best["in_tol_frac"]),
            "best_mae_mm": float(best["mae_mm"]),
            "tol_mm": float(best["tol_mm"]),
            "status": "matched"
        }
        if second is not None:
            out.update({
                "second_scan_id": second["scan_id"],
                "second_in_tol_frac": float(second["in_tol_frac"]),
                "second_mae_mm": float(second["mae_mm"]),
            })

        results.append(out)

    return pd.DataFrame(results)

# Example usage:
# xml_index_df = rois_df[['patient_folder','xml_path']].drop_duplicates()
# assignments = assign_xml_to_scan_per_patient(scans_df, xml_index_df)
# assignments.to_csv("roi_xml_scan_assignments.csv", index=False)

if __name__ == "__main__":
    import pandas as pd

    # ---- EDIT THESE PATHS ----
    SCANS_CSV = "data_canonical/tables/Gated_scan_index.csv"
    ROIS_CSV = "data_canonical/tables/calcium_rois_sorted_by_patient_folder.csv"
    OUTPUT_CSV = "data_canonical/tables/roi_xml_scan_assignments.csv"

    # ---- LOAD TABLES ----
    scans_df = pd.read_csv(SCANS_CSV)

    rois_df = pd.read_csv(ROIS_CSV)
    rois_df["patient_folder"] = (
        rois_df["xml_path"]
        .str.extract(r'\\calcium_xml\\(\d+)\.xml', expand=False)
        .astype(int)
    )

    # one XML per patient_folder
    xml_index_df = rois_df[["patient_folder", "xml_path"]].drop_duplicates()

    # ---- RUN MATCHING (patient-folder–restricted) ----
    assignments = assign_xml_to_scan_per_patient(
        scans_df=scans_df,
        xml_index_df=xml_index_df,
        patient_col="patient_folder",
        scan_id_col="scan_id",
        nifti_col="nifti_path",
        spacing_col="spacing_z_mm",
    )

    assignments.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}")






for /l %i in (1,1,20) do robocopy ^
  C:\coca_project\data_raw\dicom\Gated_release_final\Gated_release_final\patient\%i ^
  C:\Users\Jagdf\PrediCT\coca_project\sample_scans\%i ^
  /E
