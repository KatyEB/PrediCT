import os
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_and_flatten(source_root, dest_root):
    src_path = Path(source_root)
    dest_path = Path(dest_root)
    
    # 1. Structural Verification Check
    if not src_path.exists():
        print(f"❌ Error: Source directory {source_root} not found.")
        print("Please check that the path to your downloaded files matches precisely.")
        return

    # 2. Extract Valid Patient Folders (Numeric Layout Matches Stanford Database)
    # Reverting to your repository's original '.isdigit()' logic now that the data is uncorrupted
    patient_folders = [p for p in src_path.iterdir() if p.is_dir() and p.name.isdigit()]
    print(f"✅ Verified: Storage workspace initialized. Processing {len(patient_folders)} numeric patients...")

    # Create the isolated target folder on your local solid-state drive (SSD)
    dest_path.mkdir(parents=True, exist_ok=True)

    for patient_dir in tqdm(patient_folders, desc="Unnesting Data"):
        # Establish a clean directory named after the patient ID inside the separate destination folder
        target_patient_dir = dest_path / patient_dir.name
        target_patient_dir.mkdir(exist_ok=True)
        
        # Discover all .dcm slices regardless of subfolder nesting depth
        all_dcms = list(patient_dir.rglob("*.dcm"))
        
        for dcm_path in all_dcms:
            # Skip hidden macOS operating system metadata or hidden file streams
            if dcm_path.name.startswith("._") or dcm_path.name.startswith("."):
                continue
                
            target_file_path = target_patient_dir / dcm_path.name
            
            # Handle potential filename collisions safely by appending subfolder descriptors
            if target_file_path.exists():
                target_file_path = target_patient_dir / f"{dcm_path.parent.name}_{dcm_path.name}"
            
            # Crash-resilient file moving: strips out intermediate scanning layers out-of-place
            try:
                shutil.move(str(dcm_path), str(target_file_path))
            except FileNotFoundError:
                # Safely catches missing or hidden system items without interrupting execution
                pass
            except Exception as e:
                print(f"\n⚠️ [Warning] Could not move file {dcm_path.name}: {e}")

if __name__ == "__main__":
    # SYNCHRONIZED ARCHITECTURE CHANNELS TAILORED FOR YOUR MACBOOK PRO LOCAL SSD
    SOURCE = "/Users/karan/Desktop/PrediCT/cocacoronarycalciumandchestcts-2/Gated_release_final/patient"
    DESTINATION = "/Users/karan/Desktop/PrediCT/cocacoronarycalciumandchestcts-2/Flattened_COCA_Dataset"
    
    extract_and_flatten(SOURCE, DESTINATION)
    print(f"\n🚀 Success! Flattened files are cleanly positioned in: {DESTINATION}")