import os
import shutil
from pathlib import Path
from tqdm import tqdm

def flatten_dicom_folders(root_dir):
    root_path = Path(root_dir)
    # Get all patient folders (the numeric ones)
    patient_folders = [p for p in root_path.iterdir() if p.is_dir() and p.name.isdigit()]
    
    print(f"Checking {len(patient_folders)} patient folders for intermediate nesting...")

    for patient_dir in tqdm(patient_folders, desc="Flattening"):
        # Find all .dcm files anywhere inside this patient directory
        all_dcms = list(patient_dir.rglob("*.dcm"))
        
        for dcm_path in all_dcms:
            # If the file is not already directly in the patient folder
            if dcm_path.parent != patient_dir:
                # Move to the patient_dir
                target_path = patient_dir / dcm_path.name
                
                # Handle potential filename collisions (unlikely in DICOM)
                if target_path.exists():
                    target_path = patient_dir / f"{dcm_path.parent.name}_{dcm_path.name}"
                
                shutil.move(str(dcm_path), str(target_path))

        # Cleanup: Remove now-empty subdirectories
        for subfolder in list(patient_dir.iterdir()):
            if subfolder.is_dir():
                try:
                    # Only removes if empty
                    shutil.rmtree(subfolder) 
                except Exception:
                    # If it contains non-dcm files, it stays
                    pass

if __name__ == "__main__":
    # Update this to your exact patient root
    DICOM_PATIENT_ROOT = r"C:\coca_project\data_raw\dicom\Gated_release_final\Gated_release_final\patient"
    
    # It's always a good idea to have a backup or try on one folder first!
    flatten_dicom_folders(DICOM_PATIENT_ROOT)
    print("\nFlattening complete. All slices should now be directly inside patient ID folders.")