import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm

def resample_volume(volume, new_spacing=[1.0, 1.0, 1.0], is_mask=False):
    """Resamples a volume to a target voxel spacing."""
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    
    # Calculate new size to keep the physical extent the same
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(volume.GetDirection())
    resample.SetOutputOrigin(volume.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(volume.GetPixelIDValue())

    # Use Nearest Neighbor for masks to keep values 0 and 1
    # Use Linear for images to preserve anatomical detail
    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(volume)

def main():
    PROJECT_ROOT = Path(r"C:\coca_project")
    INPUT_CSV = PROJECT_ROOT / "data_canonical" / "tables" / "scan_index.csv"
    OUTPUT_DIR = PROJECT_ROOT / "data_resampled"
    
    # Target spacing in mm: [x, y, z]
    # Common standard is 1.0mm isotropic or native in-plane (0.5) with 1.5-3.0 slice thickness
    TARGET_SPACING = [1.0, 1.0, 1.0] 
    
    if not INPUT_CSV.exists():
        print(f"Index file not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Resampling {len(df)} scans to {TARGET_SPACING} mm...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        scan_id = row['scan_id']
        input_folder = Path(row['folder_path'])
        
        # Output folder for this specific scan
        resampled_scan_folder = OUTPUT_DIR / scan_id
        resampled_scan_folder.mkdir(parents=True, exist_ok=True)

        # 1. Load Original Files
        img_path = input_folder / f"{scan_id}_img.nii.gz"
        seg_path = input_folder / f"{scan_id}_seg.nii.gz"

        img = sitk.ReadImage(str(img_path))
        seg = sitk.ReadImage(str(seg_path))

        # 2. Resample
        res_img = resample_volume(img, new_spacing=TARGET_SPACING, is_mask=False)
        res_seg = resample_volume(seg, new_spacing=TARGET_SPACING, is_mask=True)

        # 3. Save Resampled Files
        sitk.WriteImage(res_img, str(resampled_scan_folder / f"{scan_id}_img.nii.gz"), useCompression=True)
        sitk.WriteImage(res_seg, str(resampled_scan_folder / f"{scan_id}_seg.nii.gz"), useCompression=True)

    print("\nResampling complete. Data is now ready for ML training.")

if __name__ == "__main__":
    main()