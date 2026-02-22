import os
import json
import hashlib
import plistlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from tqdm import tqdm

class COCAProcessor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.dicom_root = self.project_root / "data_raw" / "dicom" / "Gated_release_final" / "Gated_release_final" / "patient"
        self.xml_root = self.project_root / "data_raw" / "xml" / "calcium_xml"
        
        self.out_images_base = self.project_root / "data_canonical" / "images"
        self.out_tables = self.project_root / "data_canonical" / "tables"
        
        # Ensure output directories exist
        self.out_images_base.mkdir(parents=True, exist_ok=True)
        self.out_tables.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_stable_id(*parts: str, n: int = 12) -> str:
        """Generates a unique, reproducible ID for each scan."""
        h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
        return h[:n]

    def parse_plist_filled(self, xml_path: Path, image_shape: tuple):
        """Parses XML and returns a binary 3D mask and list of segmented slices."""
        mask = np.zeros(image_shape, dtype=np.uint8)
        segmented_slices = set()
        total_z, total_y, total_x = image_shape
        
        if not xml_path.exists():
            return mask, []

        try:
            with open(xml_path, 'rb') as f:
                data = plistlib.load(f)
            
            images = data.get('Images', [])
            for img_entry in images:
                z = int(img_entry.get('ImageIndex', -1))
                
                if z < 0 or z >= total_z:
                    continue
                
                rois = img_entry.get('ROIs', [])
                for roi in rois:
                    points_str = roi.get('Point_px', [])
                    if not points_str: 
                        continue
                    
                    poly_points = []
                    for p_str in points_str:
                        cleaned = p_str.replace("(", "").replace(")", "")
                        parts = cleaned.split(",")
                        if len(parts) == 2:
                            poly_points.append([float(parts[0]), float(parts[1])])
                    
                    if poly_points:
                        pts = np.array(poly_points, dtype=np.int32)
                        temp_slice = np.zeros((total_y, total_x), dtype=np.uint8)
                        
                        if len(pts) > 2:
                            cv2.fillPoly(temp_slice, [pts], 1)
                        else:
                            for p in pts:
                                if 0 <= p[0] < total_x and 0 <= p[1] < total_y:
                                    temp_slice[int(p[1]), int(p[0])] = 1
                        
                        if np.any(temp_slice):
                            mask[z, :, :] = np.logical_or(mask[z, :, :], temp_slice).astype(np.uint8)
                            segmented_slices.add(z)
                            
        except Exception as e:
            print(f"  [PARSING ERROR] {xml_path.name}: {e}")
            
        return mask, sorted(list(segmented_slices))

    def discover_series(self):
        """Scans the DICOM root for folders containing at least 5 DICOM files."""
        print(f"Scanning {self.dicom_root} for DICOM series...")
        all_series = []
        found_dirs = set()
        for p in self.dicom_root.rglob("*.dcm"):
            if p.parent not in found_dirs:
                if len(list(p.parent.glob("*.dcm"))) >= 5:
                    all_series.append(p.parent)
                    found_dirs.add(p.parent)
        return all_series

    def process_all(self):
        """Main execution loop to process all discovered DICOM series."""
        series_dirs = self.discover_series()
        print(f"Found {len(series_dirs)} valid series. Starting processing...")
        
        rows = []
        for s_dir in tqdm(series_dirs, desc="Processing Scans"):
            patient_id = s_dir.name 
            xml_path = self.xml_root / f"{patient_id}.xml"
            
            try:
                # Load DICOM Volume
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(s_dir))
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                
                img_array = sitk.GetArrayFromImage(image)
                
                # Generate Mask
                mask_array, seg_slices = self.parse_plist_filled(xml_path, img_array.shape)
                voxel_count = int(np.sum(mask_array))

                if xml_path.exists() and voxel_count == 0:
                    print(f"\n  [WARNING] Patient {patient_id}: XML exists but 0 voxels drawn. Check slice alignment.")

                # Setup output folder
                scan_id = self.generate_stable_id(str(s_dir.resolve()), patient_id)
                scan_folder = self.out_images_base / scan_id
                scan_folder.mkdir(parents=True, exist_ok=True)
                
                # Save Image
                sitk.WriteImage(image, str(scan_folder / f"{scan_id}_img.nii.gz"), useCompression=True)
                
                # Save Mask (inheriting geometry from original image)
                mask_image = sitk.GetImageFromArray(mask_array)
                mask_image.CopyInformation(image)
                sitk.WriteImage(mask_image, str(scan_folder / f"{scan_id}_seg.nii.gz"), useCompression=True)
                
                # Metadata
                meta = {
                    "scan_id": scan_id,
                    "patient_id": patient_id,
                    "calcium_voxels": voxel_count,
                    "slices_with_calcium": seg_slices,
                    "original_path": str(s_dir)
                }
                (scan_folder / f"{scan_id}_meta.json").write_text(json.dumps(meta, indent=2))
                
                rows.append({
                    "patient_id": patient_id, 
                    "scan_id": scan_id, 
                    "voxels": voxel_count,
                    "num_slices": len(seg_slices),
                    "folder_path": str(scan_folder) # Useful for the resampling script later
                })
                
            except Exception as e:
                print(f"  [ERROR] Patient {patient_id}: {e}")

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.out_tables / "scan_index.csv", index=False)
            print(f"\nProcessing complete. Check {self.out_tables}/scan_index.csv for results.")

if __name__ == "__main__":
    # This part only runs if you run COCA_processor.py DIRECTLY.
    # It will NOT run when imported by your pipeline script.
    print("Running Processor in standalone mode...")
    processor = COCAProcessor(r"C:\coca_project")
    processor.process_all()