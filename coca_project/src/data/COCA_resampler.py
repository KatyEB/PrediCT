import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

class COCAResampler:
    def __init__(self, project_root: str, target_spacing: list = [0.7, 0.7, 3.0]):
        self.project_root = Path(project_root)
        self.input_csv = self.project_root / "data_canonical" / "tables" / "scan_index.csv"
        self.output_dir = self.project_root / "data_resampled"
        self.target_spacing = target_spacing
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_volume(self, volume, is_mask=False):
        # 1. Reorient to RAS (Right, Anterior, Superior)
        # This standardizes the axis order across all scanners
        volume = sitk.DICOMOrient(volume, 'RAS')

        # 2. Resample logic
        original_spacing = volume.GetSpacing()
        original_size = volume.GetSize()
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
        resample.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)

        return resample.Execute(volume)

    def run(self):
        if not self.input_csv.exists():
            print("Index CSV not found.")
            return

        df = pd.read_csv(self.input_csv)
        updated_rows = []

        print(f"Resampling to {self.target_spacing} and orienting to RAS...")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            scan_id = row['scan_id']
            input_f = Path(row['folder_path'])
            out_f = self.output_dir / scan_id
            out_f.mkdir(parents=True, exist_ok=True)

            try:
                # Load
                img = sitk.ReadImage(str(input_f / f"{scan_id}_img.nii.gz"))
                seg = sitk.ReadImage(str(input_f / f"{scan_id}_seg.nii.gz"))

                # Process (Orient + Resample)
                res_img = self.process_volume(img, is_mask=False)
                res_seg = self.process_volume(seg, is_mask=True)

                # Save
                img_out = out_f / f"{scan_id}_img.nii.gz"
                seg_out = out_f / f"{scan_id}_seg.nii.gz"
                sitk.WriteImage(res_img, str(img_out), useCompression=True)
                sitk.WriteImage(res_seg, str(seg_out), useCompression=True)

                # Update the row for the final "Frozen" contract
                row_dict = row.to_dict()
                row_dict.update({
                    "image_path": str(img_out),
                    "mask_path": str(seg_out),
                    "spacing": str(self.target_spacing),
                    "shape": str(res_img.GetSize()),
                    "has_pos": 1 if row['voxels'] > 0 else 0
                })
                updated_rows.append(row_dict)

            except Exception as e:
                print(f"Error on {scan_id}: {e}")

        # Save the final Frozen Dataset Contract
        final_df = pd.DataFrame(updated_rows)
        final_df.to_parquet(self.project_root / "data_canonical" / "tables" / "frozen_dataset_v1.parquet")
        print("Dataset contract frozen at frozen_dataset_v1.parquet")

#if __name__ == "__main__":
    #resampler = COCAResampler(r"C:\coca_project")
    #resampler.run()