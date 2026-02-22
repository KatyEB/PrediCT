import json
import shutil
from pathlib import Path

# ---- ADJUST THESE PATHS TO MATCH YOUR SETUP ----

# Folder that contains *.json and *.nii / *.nii.gz pairs (the folder in your screenshot)
SRC_ROOT = Path(r"C:\coca_project\data_canonical\images")

# Your GitHub repo
REPO_ROOT = Path(r"C:\Users\Jagdf\PrediCT")

# Destination sample-data folders
SAMPLE_NII_DEST = REPO_ROOT / "coca_project" / "sample_data" / "NIfTI"
SAMPLE_JSON_DEST = REPO_ROOT / "coca_project" / "sample_data" / "json"

SAMPLE_NII_DEST.mkdir(parents=True, exist_ok=True)
SAMPLE_JSON_DEST.mkdir(parents=True, exist_ok=True)

# Patients to include
PATIENT_RANGE = range(0, 21)  # 0–20 inclusive

# Name of the field in JSON that holds the patient ID
PATIENT_KEY = "patient_folder"  # CHANGE THIS if your JSON uses a different key


def find_matching_nii(stem: str) -> Path | None:
    """
    Given the JSON filename stem, find the matching NIfTI.
    Tries .nii.gz first, then .nii.
    """
    nii_gz = SRC_ROOT / f"{stem}.nii.gz"
    nii = SRC_ROOT / f"{stem}.nii"
    if nii_gz.is_file():
        return nii_gz
    if nii.is_file():
        return nii
    return None


def main():
    # track how many scans per patient so we can add _scan1, _scan2 if needed
    scan_counts: dict[str, int] = {}

    for json_path in sorted(SRC_ROOT.glob("*.json")):
        stem = json_path.stem

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract patient ID from JSON
        patient_raw = data.get(PATIENT_KEY)
        if patient_raw is None:
            print(f"[WARN] {json_path.name}: no '{PATIENT_KEY}' field, skipping")
            continue

        # Normalize to string; if it's numeric, this will still work
        patient_id_str = str(patient_raw).strip()

        # Only keep patients 0–20
        try:
            pid_int = int(patient_id_str)
        except ValueError:
            print(f"[WARN] {json_path.name}: '{PATIENT_KEY}'='{patient_id_str}' is not an int, skipping")
            continue

        if pid_int not in PATIENT_RANGE:
            continue

        # Determine scan index for this patient
        scan_counts.setdefault(patient_id_str, 0)
        scan_counts[patient_id_str] += 1
        scan_idx = scan_counts[patient_id_str]

        # ---- JSON destination name ----
        if scan_idx == 1:
            json_name = f"patient_{pid_int}.json"
        else:
            json_name = f"patient_{pid_int}_scan{scan_idx}.json"
        json_dest = SAMPLE_JSON_DEST / json_name

        print(f"Copying JSON: {json_path} -> {json_dest}")
        shutil.copy2(json_path, json_dest)

        # ---- Matching NIfTI ----
        nii_src = find_matching_nii(stem)
        if nii_src is None:
            print(f"[WARN] No NIfTI found matching {stem}, skipping NIfTI")
            continue

        # preserve .nii or .nii.gz
        ext = "".join(nii_src.suffixes)
        if scan_idx == 1:
            nii_name = f"patient_{pid_int}{ext}"
        else:
            nii_name = f"patient_{pid_int}_scan{scan_idx}{ext}"

        nii_dest = SAMPLE_NII_DEST / nii_name
        print(f"Copying NIfTI: {nii_src} -> {nii_dest}")
        shutil.copy2(nii_src, nii_dest)


if __name__ == "__main__":
    main()
