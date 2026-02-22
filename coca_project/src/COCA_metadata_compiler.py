import json
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import numpy as np

class COCAMetadataCompiler:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.resampled_dir = self.project_root / "data_resampled"
        self.canonical_dir = self.project_root / "data_canonical" / "images"
        self.output_path = self.project_root / "data_canonical" / "tables" / "metadata_summary.parquet"
        
        # Ensure table directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self):
        all_meta = []
        
        # We iterate through the resampled folder because these are the files 
        # your model will actually use for training.
        scan_folders = [f for f in self.resampled_dir.iterdir() if f.is_dir()]
        print(f"Compiling metadata for {len(scan_folders)} resampled scans...")

        for scan_folder in tqdm(scan_folders):
            scan_id = scan_folder.name
            
            # 1. Define Paths
            img_path = scan_folder / f"{scan_id}_img.nii.gz"
            seg_path = scan_folder / f"{scan_id}_seg.nii.gz"
            json_path = self.canonical_dir / scan_id / f"{scan_id}_meta.json"

            if not img_path.exists() or not seg_path.exists():
                continue

            try:
                # 2. Load the actual resampled mask to get accurate counts
                # This fixes the 'Voxel sum mismatch' error
                res_seg = sitk.ReadImage(str(seg_path))
                res_seg_array = sitk.GetArrayFromImage(res_seg)
                
                # Calculate counts based on the CURRENT file state
                current_voxels = int(np.sum(res_seg_array > 0))
                current_slices = np.where(np.any(res_seg_array > 0, axis=(1, 2)))[0].tolist()
                
                # Get geometry info
                spacing = res_seg.GetSpacing()
                shape = res_seg.GetSize()

                # 3. Pull patient info from the original JSON if it exists
                patient_id = "Unknown"
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        meta_data = json.load(f)
                        patient_id = meta_data.get("patient_id", "Unknown")

                # 4. Build the Row
                row = {
                    "scan_id": scan_id,
                    "patient_id": patient_id,
                    "image_path": str(img_path.resolve()),
                    "mask_path": str(seg_path.resolve()),
                    "has_pos": 1 if current_voxels > 0 else 0,
                    "n_pos_voxels": current_voxels,
                    "n_pos_slices": len(current_slices),
                    "spacing_x": spacing[0],
                    "spacing_y": spacing[1],
                    "spacing_z": spacing[2],
                    "shape_x": shape[0],
                    "shape_y": shape[1],
                    "shape_z": shape[2]
                }
                all_meta.append(row)
                
            except Exception as e:
                print(f"  [ERROR] Processing {scan_id}: {e}")

        if not all_meta:
            print("No data found to compile.")
            return

        # Create DataFrame and Export
        df = pd.DataFrame(all_meta)
        df.to_parquet(self.output_path, index=False, compression='snappy')
        
        print("-" * 50)
        print(f"SUCCESS: Metadata compiled at {self.output_path}")
        print(f"Positive cases: {df['has_pos'].sum()} / {len(df)}")
        print("-" * 50)

if __name__ == "__main__":
    compiler = COCAMetadataCompiler(r"C:\coca_project")
    compiler.run()