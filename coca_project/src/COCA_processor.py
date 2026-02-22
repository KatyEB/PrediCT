import json
import hashlib
import plistlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TARGET_SPACING = (.5, .5, 3)   # mm — isotropic resampling target
USE_TOTAL_SEGMENTATOR = True        # Set False to skip heart masking
TS_DEVICE = "gpu"                   # "gpu" or "cpu"
# ---------------------------------------------------------------------------


class COCAProcessor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.dicom_root = (
            self.project_root
            / "data_raw" / "dicom"
            / "Gated_release_final" / "Gated_release_final" / "patient"
        )
        self.xml_root = self.project_root / "data_raw" / "xml" / "calcium_xml"

        self.out_images_base = self.project_root / "data_canonical" / "images"
        self.out_tables     = self.project_root / "data_canonical" / "tables"

        self.out_images_base.mkdir(parents=True, exist_ok=True)
        self.out_tables.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def generate_stable_id(*parts: str, n: int = 12) -> str:
        h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
        return h[:n]

    @staticmethod
    def resample_image(
        image: sitk.Image,
        new_spacing: tuple = TARGET_SPACING,
        interpolator=sitk.sitkLinear,
    ) -> sitk.Image:
        """Resample to isotropic voxel spacing."""
        orig_spacing = image.GetSpacing()
        orig_size    = image.GetSize()

        new_size = [
            int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
            for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(-1024)
        resampler.SetInterpolator(interpolator)
        return resampler.Execute(image)

    # ------------------------------------------------------------------
    # Label parsing
    # ------------------------------------------------------------------

    def parse_plist_filled(self, xml_path: Path, image_shape: tuple):
        """
        Parse COCA XML annotation and return a binary 3D mask.
        image_shape is (Z, Y, X) — SimpleITK array convention.
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
            print(f"  [PARSING ERROR] {xml_path.name}: {e}")

        return mask, sorted(segmented_slices)

    # ------------------------------------------------------------------
    # TotalSegmentator heart masking
    # ------------------------------------------------------------------

    def run_total_segmentator(
        self, input_nii: Path, output_dir: Path
    ) -> sitk.Image | None:
        """
        Run TotalSegmentator to generate a cardiac ROI mask.
        Returns a binary SimpleITK image, or None on failure.

        The cardiac mask is the union of: heart, aorta, pulmonary_artery.
        Keeping the aorta is important — aortic calcification is a common
        false-positive source that you still want to capture in training labels.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "TotalSegmentator",
            "-i", str(input_nii),
            "-o", str(output_dir),
            "--roi_subset", "heart", "aorta",
            # pulmonary_artery removed — not a valid class name in TS v2.x
            # heart + aorta is sufficient for cardiac calcium ROI masking
            "--device", TS_DEVICE,
            "--ml",        # multi-label output (one file per structure)
        ]

        try:
            # Capture stderr only (not stdout) so we can log errors without
            # deadlocking on the large stdout progress bar output from TS.
            result = subprocess.run(
                cmd,
                check=True,
                timeout=300,
                stdout=subprocess.DEVNULL,   # discard progress bars
                stderr=subprocess.PIPE,      # capture errors only
                text=True,
            )
        except subprocess.TimeoutExpired:
            print("  [TS ERROR] TotalSegmentator timed out (>5 min) — skipping")
            return None
        except subprocess.CalledProcessError as e:
            print(f"  [TS ERROR] TotalSegmentator returned exit code {e.returncode}")
            if e.stderr:
                print(f"  [TS STDERR] {e.stderr[:600]}")
            return None
        except FileNotFoundError:
            print("  [TS ERROR] TotalSegmentator not found — is it installed in this environment?")
            return None

        # TS v2 with --ml writes a single multi-label file named after the
        # output argument (ts_tmp.nii), sitting inside the scan folder —
        # NOT separate per-structure files inside the ts_tmp subdirectory.
        # Each structure is a different integer label; we union heart + aorta.
        ts_output = output_dir.parent / f"{output_dir.name}.nii"
        if not ts_output.exists():
            # Fallback: also check for .nii.gz variant
            ts_output_gz = output_dir.parent / f"{output_dir.name}.nii.gz"
            if ts_output_gz.exists():
                ts_output = ts_output_gz
            else:
                print(f"  [TS WARNING] Expected output not found at {ts_output}")
                return None

        seg     = sitk.ReadImage(str(ts_output))
        arr     = sitk.GetArrayFromImage(seg)

        # Print unique label values on first scan so we can verify heart/aorta IDs
        unique_labels = sorted(np.unique(arr).tolist())
        if len(unique_labels) <= 10:
            print(f"  [TS INFO] Label values in output: {unique_labels}")

        # TS v2 total task label IDs: heart=51, aorta=52
        # If these look wrong check the printed label values above and update.
        HEART_LABEL = 51
        AORTA_LABEL = 52
        combined = ((arr == HEART_LABEL) | (arr == AORTA_LABEL)).astype(np.uint8)

        if combined.max() == 0:
            print(f"  [TS WARNING] Heart/aorta labels ({HEART_LABEL},{AORTA_LABEL}) "
                  f"not found in output. Found labels: {unique_labels}. "
                  f"Update HEART_LABEL/AORTA_LABEL constants in run_total_segmentator().")
            return None

        cardiac_mask = sitk.GetImageFromArray(combined)
        cardiac_mask.CopyInformation(seg)
        return cardiac_mask

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_series(self) -> list[Path]:
        """
        Find all DICOM series under dicom_root.
        Uses the *grandparent* folder name as the patient ID to be robust
        against COCA's nested folder structure (patient/study/series).
        """
        print(f"Scanning {self.dicom_root} for DICOM series...")
        all_series, found_dirs = [], set()
        for p in self.dicom_root.rglob("*.dcm"):
            if p.parent not in found_dirs:
                dcm_files = list(p.parent.glob("*.dcm"))
                if len(dcm_files) >= 5:
                    all_series.append(p.parent)
                    found_dirs.add(p.parent)
        print(f"  → Found {len(all_series)} valid series.")
        return all_series

    @staticmethod
    def _patient_id_from_path(series_dir: Path, dicom_root: Path) -> str:
        """
        Walk up from series_dir until we hit a direct child of dicom_root.
        That child's name is the patient-level folder — much safer than
        using s_dir.name which could be a series UID.
        """
        parts = series_dir.relative_to(dicom_root).parts
        return parts[0] if parts else series_dir.name

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def process_all(self):
        series_dirs = self.discover_series()
        rows = []

        for s_dir in tqdm(series_dirs, desc="Processing Scans"):
            patient_id = self._patient_id_from_path(s_dir, self.dicom_root)
            xml_path   = self.xml_root / f"{patient_id}.xml"

            try:
                # ── 1. Load DICOM ──────────────────────────────────────
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(str(s_dir))
                reader.SetFileNames(dicom_names)
                image = reader.Execute()

                # ── 2. Resample to isotropic spacing ───────────────────
                image_iso = self.resample_image(image, TARGET_SPACING, sitk.sitkLinear)

                # ── 3. Parse calcium annotation ────────────────────────
                img_array  = sitk.GetArrayFromImage(image)      # original for mask drawing
                mask_array, seg_slices = self.parse_plist_filled(xml_path, img_array.shape)

                # Resample binary mask with nearest-neighbour to preserve labels
                mask_sitk     = sitk.GetImageFromArray(mask_array)
                mask_sitk.CopyInformation(image)
                mask_sitk_iso = self.resample_image(
                    sitk.Cast(mask_sitk, sitk.sitkFloat32),
                    TARGET_SPACING,
                    sitk.sitkNearestNeighbor,
                )
                mask_sitk_iso = sitk.Cast(mask_sitk_iso, sitk.sitkUInt8)

                voxel_count = int(
                    np.sum(sitk.GetArrayFromImage(mask_sitk_iso))
                )

                if xml_path.exists() and voxel_count == 0:
                    print(
                        f"\n  [WARNING] Patient {patient_id}: XML exists but "
                        f"0 voxels after resampling. Check slice alignment."
                    )

                # ── 4. Set up output folder ────────────────────────────
                scan_id     = self.generate_stable_id(str(s_dir.resolve()), patient_id)
                scan_folder = self.out_images_base / scan_id
                scan_folder.mkdir(parents=True, exist_ok=True)

                image_path = scan_folder / f"{scan_id}_img.nii.gz"
                mask_path  = scan_folder / f"{scan_id}_seg.nii.gz"

                sitk.WriteImage(image_iso,    str(image_path), useCompression=True)
                sitk.WriteImage(mask_sitk_iso, str(mask_path),  useCompression=True)

                # ── 5. TotalSegmentator cardiac mask (optional) ────────
                cardiac_mask_path = None
                if USE_TOTAL_SEGMENTATOR:
                    ts_tmp_dir   = scan_folder / "ts_tmp"
                    cardiac_mask = self.run_total_segmentator(image_path, ts_tmp_dir)

                    if cardiac_mask is not None:
                        ct_ref = sitk.ReadImage(str(image_path))
                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(ct_ref)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        resampler.SetDefaultPixelValue(0)
                        cardiac_mask = resampler.Execute(cardiac_mask)
                        # Fill internal holes slice by slice
                        fill = sitk.BinaryFillholeImageFilter()
                        fill.SetFullyConnected(True)
                        cardiac_mask = fill.Execute(cardiac_mask)

                        # Then dilate outward to capture epicardial surface
                        dilate = sitk.BinaryDilateImageFilter()
                        dilate.SetKernelRadius(7)  # ~7-10mm depending on your voxel spacing
                        cardiac_mask = dilate.Execute(cardiac_mask)
                        
                    if cardiac_mask is not None:
                        cardiac_mask_path = scan_folder / f"{scan_id}_cardiac_mask.nii.gz"
                        sitk.WriteImage(
                            cardiac_mask, str(cardiac_mask_path), useCompression=True
                        )

                        # Sanity check: do any calcium labels fall inside the mask?
                        ca_arr = sitk.GetArrayFromImage(mask_sitk_iso)
                        cm_arr = sitk.GetArrayFromImage(cardiac_mask)
                        overlap = int(np.sum(ca_arr & cm_arr))
                        if voxel_count > 0 and overlap == 0:
                            print(
                                f"\n  [WARNING] Patient {patient_id}: calcium labels "
                                f"do NOT overlap with cardiac mask — check registration."
                            )

                    # Clean up large intermediate TS files to save disk space
                    for f in ts_tmp_dir.glob("*.nii.gz"):
                        if f.name not in (
                            "heart.nii.gz", "aorta.nii.gz", "pulmonary_artery.nii.gz"
                        ):
                            f.unlink(missing_ok=True)

                # ── 6. Metadata ────────────────────────────────────────
                meta = {
                    "scan_id":              scan_id,
                    "patient_id":           patient_id,
                    "original_spacing":     list(image.GetSpacing()),
                    "resampled_spacing":    list(TARGET_SPACING),
                    "original_size":        list(image.GetSize()),
                    "resampled_size":       list(image_iso.GetSize()),
                    "calcium_voxels":       voxel_count,
                    "slices_with_calcium":  seg_slices,
                    "cardiac_mask_path":    str(cardiac_mask_path) if cardiac_mask_path else None,
                    "original_path":        str(s_dir),
                }
                (scan_folder / f"{scan_id}_meta.json").write_text(
                    json.dumps(meta, indent=2)
                )

                rows.append({
                    "patient_id":        patient_id,
                    "scan_id":           scan_id,
                    "image_path":        str(image_path),
                    "mask_path":         str(mask_path),
                    "cardiac_mask_path": str(cardiac_mask_path) if cardiac_mask_path else None,
                    "voxels":            voxel_count,
                    "has_calcium":       voxel_count > 0,
                    "num_slices":        len(seg_slices),
                    "folder_path":       str(scan_folder),
                })

            except Exception as e:
                print(f"  [ERROR] Patient {patient_id}: {e}")

        if rows:
            df = pd.DataFrame(rows)
            out_csv = self.out_tables / "scan_index.csv"
            df.to_csv(out_csv, index=False)
            print(f"\n✓ Processing complete.")
            print(f"  {len(df)} scans written → {out_csv}")
            print(f"  Positive (calcium > 0): {df['has_calcium'].sum()}")
            print(f"  Negative:               {(~df['has_calcium']).sum()}")


if __name__ == "__main__":
    processor = COCAProcessor(r"C:\coca_project")
    processor.process_all()