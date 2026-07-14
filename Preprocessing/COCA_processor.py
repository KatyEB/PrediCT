import os
import json
import hashlib
import plistlib
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from tqdm import tqdm

# HU windowing constants — cardiac soft tissue / coronary artery wall
HU_WIN_MIN = -150.0
HU_WIN_MAX =  400.0


class COCAProcessor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

        self.dicom_root = self.project_root / "Flattened_COCA_Dataset"
        self.xml_root   = self.project_root / "Gated_release_final" / "calcium_xml"

        external_drive = Path("/Volumes/SanDisk/PrediCT")
        self.out_images_base = external_drive / "data_canonical" / "images"
        self.out_tables      = external_drive / "data_canonical" / "tables"

        self.out_images_base.mkdir(parents=True, exist_ok=True)
        self.out_tables.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def generate_stable_id(*parts: str, n: int = 12) -> str:
        h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
        return h[:n]

    # ------------------------------------------------------------------
    # Calcium mask parsing
    # ------------------------------------------------------------------

    def parse_plist_filled(self, xml_path: Path, image_shape: tuple):
        """
        Parse Apple plist XML annotation → binary 3-D mask.
        Returns (mask_array uint8, sorted list of annotated z-indices).
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        segmented_slices = set()
        total_z, total_y, total_x = image_shape

        if not xml_path.exists():
            return mask, []

        try:
            with open(xml_path, "rb") as f:
                data = plistlib.load(f)

            for img_entry in data.get("Images", []):
                z = int(img_entry.get("ImageIndex", -1))
                if z < 0 or z >= total_z:
                    continue

                for roi in img_entry.get("ROIs", []):
                    points_str = roi.get("Point_px", [])
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
                            mask[z] = np.logical_or(mask[z], temp_slice).astype(np.uint8)
                            segmented_slices.add(z)

        except Exception as e:
            print(f"    [WARN] plist parse error: {e}")

        return mask, sorted(segmented_slices)

    # ------------------------------------------------------------------
    # Series discovery
    # ------------------------------------------------------------------

    def discover_series(self):
        print(f"Scanning {self.dicom_root} for DICOM series...")
        all_series, found_dirs = [], set()

        if not self.dicom_root.exists():
            print(f"[ERROR] {self.dicom_root} does not exist.")
            return all_series

        for p in self.dicom_root.rglob("*.dcm"):
            if p.parent not in found_dirs:
                if len(list(p.parent.glob("*.dcm"))) >= 5:
                    all_series.append(p.parent)
                    found_dirs.add(p.parent)

        return all_series

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    def process_all(self):
        series_dirs = self.discover_series()
        print(f"Found {len(series_dirs)} valid series. Starting processing...")

        rows = []

        for s_dir in tqdm(series_dirs, desc="Processing Scans"):
            patient_id = s_dir.name
            xml_path   = self.xml_root / f"{patient_id}.xml"

            try:
                # ── Load raw DICOM volume ──────────────────────────────
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(s_dir))
                reader.SetFileNames(dicom_names)
                image = reader.Execute()   # stays in original HU

                # Geometry metadata — essential for Project 3 registration
                orig_spacing   = list(image.GetSpacing())    # (x, y, z) in mm
                orig_size      = list(image.GetSize())       # (x, y, z) in voxels
                origin         = list(image.GetOrigin())
                direction      = list(image.GetDirection())  # 9 floats, row-major

                # ── Raw HU array ───────────────────────────────────────
                raw_hu = sitk.GetArrayFromImage(image).astype(np.float32)  # (z, y, x)

                # HU statistics before any windowing (for registration QC)
                hu_stats = {
                    "hu_min":  float(raw_hu.min()),
                    "hu_max":  float(raw_hu.max()),
                    "hu_mean": float(raw_hu.mean()),
                    "hu_std":  float(raw_hu.std()),
                    "hu_p5":   float(np.percentile(raw_hu,  5)),
                    "hu_p95":  float(np.percentile(raw_hu, 95)),
                }

                # ── HU windowing for segmentation pipeline ─────────────
                # Window [-150, 400] HU isolates coronary artery wall / calcium
                img_windowed = np.clip(raw_hu, HU_WIN_MIN, HU_WIN_MAX)
                img_windowed = (img_windowed - HU_WIN_MIN) / (HU_WIN_MAX - HU_WIN_MIN)

                # ── Calcium mask ───────────────────────────────────────
                mask_array, seg_slices = self.parse_plist_filled(xml_path, raw_hu.shape)
                voxel_count = int(np.sum(mask_array))

                # ── Output directory ───────────────────────────────────
                scan_id     = self.generate_stable_id(str(s_dir.resolve()), patient_id)
                scan_folder = self.out_images_base / scan_id
                scan_folder.mkdir(parents=True, exist_ok=True)

                # ── Save raw HU image (Project 3 registration input) ───
                raw_sitk = sitk.GetImageFromArray(raw_hu)
                raw_sitk.CopyInformation(image)
                raw_img_path = scan_folder / f"{scan_id}_raw_img.nii.gz"
                sitk.WriteImage(raw_sitk, str(raw_img_path), useCompression=True)

                # ── Save windowed / normalised image ───────────────────
                windowed_sitk = sitk.GetImageFromArray(img_windowed)
                windowed_sitk.CopyInformation(image)
                windowed_img_path = scan_folder / f"{scan_id}_img.nii.gz"
                sitk.WriteImage(windowed_sitk, str(windowed_img_path), useCompression=True)

                # ── Save binary label mask ─────────────────────────────
                mask_sitk = sitk.GetImageFromArray(mask_array)
                mask_sitk.CopyInformation(image)
                seg_path = scan_folder / f"{scan_id}_seg.nii.gz"
                sitk.WriteImage(mask_sitk, str(seg_path), useCompression=True)

                # ── Write full metadata JSON ───────────────────────────
                meta = {
                    "scan_id":    scan_id,
                    "patient_id": patient_id,
                    # geometry
                    "original_spacing": orig_spacing,
                    "original_size":    orig_size,
                    "origin":           origin,
                    "direction":        direction,
                    # HU stats (pre-windowing)
                    **hu_stats,
                    # windowing parameters (to invert normalisation downstream)
                    "hu_window_min": HU_WIN_MIN,
                    "hu_window_max": HU_WIN_MAX,
                    # annotations
                    "calcium_voxels":      voxel_count,
                    "slices_with_calcium": seg_slices,
                    "has_xml":             xml_path.exists(),
                    # paths
                    "original_path":      str(s_dir),
                    "raw_img_path":       str(raw_img_path),
                    "windowed_img_path":  str(windowed_img_path),
                    "seg_path":           str(seg_path),
                }
                (scan_folder / f"{scan_id}_meta.json").write_text(
                    json.dumps(meta, indent=2)
                )

                rows.append({
                    "patient_id":       patient_id,
                    "scan_id":          scan_id,
                    "voxels":           voxel_count,
                    "num_slices":       len(seg_slices),
                    "has_calcium":      int(voxel_count > 0),
                    # geometry columns (needed for split stratification & resampler)
                    "spacing_x":        orig_spacing[0],
                    "spacing_y":        orig_spacing[1],
                    "spacing_z":        orig_spacing[2],
                    "size_x":           orig_size[0],
                    "size_y":           orig_size[1],
                    "size_z":           orig_size[2],
                    # paths
                    "folder_path":      str(scan_folder),
                    "raw_img_path":     str(raw_img_path),
                    "windowed_img_path":str(windowed_img_path),
                    "seg_path":         str(seg_path),
                })

            except Exception as e:
                print(f"  [ERROR] Patient {patient_id}: {e}")

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.out_tables / "scan_index.csv", index=False)
            n_pos = df["has_calcium"].sum()
            print(f"\nProcessing complete.")
            print(f"  Total scans : {len(df)}")
            print(f"  Calcium (+) : {n_pos}  ({100*n_pos/len(df):.1f}%)")
            print(f"  Calcium (-) : {len(df)-n_pos}")
            print(f"  Index CSV   : {self.out_tables}/scan_index.csv")


if __name__ == "__main__":
    default_path = "/Users/karan/Desktop/PrediCT/cocacoronarycalciumandchestcts-2"
    processor = COCAProcessor(default_path)
    processor.process_all()