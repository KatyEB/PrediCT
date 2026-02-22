import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

class COCAResampler:
    def __init__(self, project_root: str, target_spacing: list = [0.7, 0.7, 3.0]):
        """
        Initializes the resampler.
        target_spacing: [x, y, z] in mm. [1.0, 1.0, 1.0] creates isotropic voxels.
        """
        self.project_root = Path(project_root)
        self.input_csv = self.project_root / "data_canonical" / "tables" / "scan_index.csv"
        self.output_dir = self.project_root / "data_resampled"
        self.target_spacing = target_spacing
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def resample_volume(self, volume: sitk.Image, is_mask: bool = False) -> sitk.Image:
        """
        Resamples a single SimpleITK image to the target spacing.
        Uses Linear interpolation for images and Nearest Neighbor for masks.
        """
        original_spacing = volume.GetSpacing()
        original_size = volume.GetSize()
        
        # Calculate new size to maintain physical extent
        # NewSize = OldSize * (OldSpacing / NewSpacing)
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / self.target_spacing[i])))
            for i in range(3)
        ]
        
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(self.target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(volume.GetDirection())
        resample.SetOutputOrigin(volume.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(volume.GetPixelIDValue())

        if is_mask:
            # Nearest Neighbor prevents creating new label values (keeps it 0 and 1)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            # Linear provides smoother anatomical transitions
            resample.SetInterpolator(sitk.sitkLinear)

        return resample.Execute(volume)

    def run(self):
        """Processes all scans listed in the scan_index.csv."""
        if not self.input_csv.exists():
            print(f"[ERROR] Could not find {self.input_csv}. Run the Processor first.")
            return

        df = pd.read_csv(self.input_csv)
        print(f"Starting resampling of {len(df)} scans to {self.target_spacing} mm...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Resampling"):
            scan_id = row['scan_id']
            # We use the folder_path saved in the CSV by the previous class
            input_folder = Path(row['folder_path'])
            
            resampled_folder = self.output_dir / scan_id
            resampled_folder.mkdir(parents=True, exist_ok=True)

            try:
                # 1. Load Original NIfTI files
                img_path = input_folder / f"{scan_id}_img.nii.gz"
                seg_path = input_folder / f"{scan_id}_seg.nii.gz"

                img = sitk.ReadImage(str(img_path))
                seg = sitk.ReadImage(str(seg_path))

                # 2. Perform Resampling
                res_img = self.resample_volume(img, is_mask=False)
                res_seg = self.resample_volume(seg, is_mask=True)

                # 3. Save Resampled Results
                sitk.WriteImage(res_img, str(resampled_folder / f"{scan_id}_img.nii.gz"), useCompression=True)
                sitk.WriteImage(res_seg, str(resampled_folder / f"{scan_id}_seg.nii.gz"), useCompression=True)
                
            except Exception as e:
                print(f"  [ERROR] Failed to resample {scan_id}: {e}")

        print(f"\nResampling complete. Files saved to: {self.output_dir}")

if __name__ == "__main__":
    resampler = COCAResampler(r"C:\coca_project", target_spacing=[.7, .7, 3.0])
    resampler.run()