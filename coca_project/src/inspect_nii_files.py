import os
import nibabel as nib
import numpy as np
from pathlib import Path

def inspect_nifti_masks(target_dir):
    target_path = Path(target_dir)
    # Search for all files ending in _seg.nii.gz
    seg_files = list(target_path.rglob("*_seg.nii.gz"))
    
    if not seg_files:
        print(f"No segmentation files found in {target_dir}")
        return

    print(f"{'Folder/File Name':<40} | {'Total Pixels':<12} | {'Calcium Pixels':<15}")
    print("-" * 75)

    for seg_file in seg_files:
        try:
            # Load using nibabel (lightweight for quick inspection)
            img = nib.load(seg_file)
            data = img.get_fdata()
            
            total_voxels = data.size
            calcium_voxels = np.sum(data == 1)
            
            # Identify the parent folder (ScanID) for clarity
            display_name = f"{seg_file.parent.name}/{seg_file.name}"
            
            status = "✅ DATA FOUND" if calcium_voxels > 0 else "❌ EMPTY (ALL ZEROS)"
            
            print(f"{display_name[:40]:<40} | {total_voxels:<12} | {int(calcium_voxels):<15} {status}")
            
        except Exception as e:
            print(f"Error reading {seg_file.name}: {e}")

# --- RUN IT ---
PROJECT_ROOT = r"C:\coca_project" # Change if your path is different
images_dir = os.path.join(PROJECT_ROOT, "data_canonical", "images")
inspect_nifti_masks(images_dir)