import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import random

def sanity_check_load(parquet_path, num_samples=20):
    # 1. Load your metadata contract
    if not Path(parquet_path).exists():
        print(f"❌ Error: Parquet file not found at {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    
    # Check if required columns exist to avoid KeyErrors
    required_cols = ['image_path', 'mask_path', 'n_pos_voxels', 'scan_id']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"❌ Error: Parquet is missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Select random cases
    samples = df.sample(min(num_samples, len(df))).to_dict('records')
    
    print(f"--- Starting Sanity Check on {len(samples)} samples ---")
    print(f"Target Orientation: RAS")
    
    results = {"pass": 0, "fail": 0}

    for case in samples:
        case_id = case['scan_id']
        img_p = case['image_path']
        msk_p = case['mask_path']
        
        # Verify files actually exist on disk where the parquet says they are
        if not Path(img_p).exists() or not Path(msk_p).exists():
            print(f"❌ [FILE MISSING] {case_id}: Check if paths in parquet are absolute and correct.")
            results["fail"] += 1
            continue

        try:
            # 2. Load Image and Mask
            img = sitk.ReadImage(img_p)
            mask = sitk.ReadImage(msk_p)
            
            # --- Check A: Geometry/Affine Match ---
            # Using absolute tolerance (atol) for floating point comparisons
            spacing_match = np.allclose(img.GetSpacing(), mask.GetSpacing(), atol=1e-5)
            origin_match = np.allclose(img.GetOrigin(), mask.GetOrigin(), atol=1e-5)
            direction_match = np.allclose(img.GetDirection(), mask.GetDirection(), atol=1e-5)
            size_match = img.GetSize() == mask.GetSize()
            
            # --- Check B: Orientation Check (RAS) ---
            # Standard RAS direction matrix is diagonal [1, 0, 0, 0, 1, 0, 0, 0, 1]
            # SimpleITK returns directions as a flattened tuple
            is_ras = np.allclose(img.GetDirection(), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), atol=1e-3)

            if not (spacing_match and origin_match and direction_match and size_match):
                print(f"❌ [GEOMETRY ERROR] {case_id}: Image/Mask spatial mismatch.")
                results["fail"] += 1
                continue
            
            if not is_ras:
                print(f"⚠️ [ORIENTATION WARNING] {case_id}: Volume is not in standard RAS orientation.")

            # --- Check C: Voxel Count Consistency ---
            mask_array = sitk.GetArrayFromImage(mask)
            current_voxels = int(np.sum(mask_array > 0))
            expected_voxels = int(case['n_pos_voxels'])
            
            if current_voxels != expected_voxels:
                print(f"❌ [DATA ERROR] {case_id}: Voxel sum mismatch! Table: {expected_voxels}, File: {current_voxels}")
                results["fail"] += 1
            else:
                print(f"✅ {case_id}: Passed (Voxels: {current_voxels}, Size: {img.GetSize()})")
                results["pass"] += 1

        except Exception as e:
            print(f"❌ [LOAD ERROR] {case_id}: {str(e)}")
            results["fail"] += 1

    print(f"\n--- Sanity Check Complete ---")
    print(f"Passed: {results['pass']} | Failed: {results['fail']}")

if __name__ == "__main__":
    # Ensure this matches the EXACT filename of the parquet you just created
    PARQUET_FILE = r"C:\coca_project\data_canonical\tables\metadata_summary.parquet"
    sanity_check_load(PARQUET_FILE)