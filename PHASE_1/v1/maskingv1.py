import os
import json
import time
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: SYSTEM CONFIGURATION MATRIX
# ════════════════════════════════════════════════════════════════════════════

class CFG:
    # --- System Environment Paths ---
    IMAGECAS_DIR = Path("/Users/karan/Desktop/PrediCT/1-200")          
    COCA_DIR     = Path("/Users/karan/Desktop/PrediCT/COCA")          
    OUT_DIR      = Path("/Users/karan/Desktop/PrediCT/output")

    # --- Cohort Control Parameters ---
    MAX_TARGET_SCANS = 1             
    
    BASELINE_ATLAS_ID = "1"          

    # --- Preprocessing Grid Windowing ---
    HU_LO = -200.0
    HU_HI = 600.0
    ISO_SPACING_MM = 1.0              # Isotropic resampling processing resolution
    REG_SPACING_MM = 1.5              # Optimized downsampled tracking resolution (1.5mm)

    # --- Multi-Resolution Optimization Parameters ---
    SAMPLING_FRACTION = 0.20          # Controlled 20% regular sampling rate
    PYRAMID_SHRINK = [8, 4, 2, 1]     
    PYRAMID_SIGMAS = [3, 2, 1, 0]     # Physical unit blurring kernels in mm
    
    # --- Rigid Step Gradient Settings ---
    RIGID_LEARNING_RATE = 2.0
    RIGID_MIN_STEP = 0.001
    RIGID_ITERS = 250
    RIGID_RELAXATION = 0.5

    # --- Affine Optimizer Settings ---
    AFFINE_LEARNING_RATE = 1.0
    AFFINE_MIN_STEP = 0.0001
    AFFINE_ITERS = 200

    # --- Fluid Domain Validation Channels ---
    CARDIAC_ROI_MARGIN_VOX = 45
    CALCIUM_HU_LO = 130               # Standard Agatston step-1 plaque threshold
    CALCIUM_HU_HI = 1000.0            # Uncapped ceiling ceiling to trap dense calcium
    LABEL_DILATE_ITERS = 2
    DISTANCE_MM = 10.0                # Tolerance channel radius in millimeters->as suggested( by the Agatston scoring protocol)
    PASS_THRESHOLD_PCT = 70.0         

    RANDOM_SEED = 42


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: COHORT DATA DISCOVERY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def discover_atlas_cases(imagecas_dir: Path):
    atlas_dict = defaultdict(dict)
    for f in imagecas_dir.glob("*.nii.gz"):
        name = f.name
        if "label" in name:
            case_id = name.split(".")[0].replace("_label", "").replace(".label", "").replace("label", "")
            atlas_dict[case_id]["lbl"] = f
        else:
            case_id = name.split(".")[0].replace(".img", "")
            atlas_dict[case_id]["img"] = f

    cases = []
    for case_id in sorted(atlas_dict.keys(), key=lambda s: int(s) if s.isdigit() else s):
        d = atlas_dict[case_id]
        if "img" in d and "lbl" in d:
            cases.append({"id": case_id, "img": d["img"], "lbl": d["lbl"]})
    return cases


def discover_coca_scans(coca_dir: Path, max_scans: int):
    scans = []
    for scan_dir in sorted(p for p in coca_dir.iterdir() if p.is_dir() and p.name not in CFG.BLACKLIST_PATIENTS):
        sid = scan_dir.name
        raw_img = scan_dir / f"{sid}_raw_img.nii.gz"
        win_img = scan_dir / f"{sid}_img.nii.gz"
        seg     = scan_dir / f"{sid}_seg.nii.gz"

        image_path = raw_img if raw_img.exists() else (win_img if win_img.exists() else None)
        if image_path is None or not seg.exists():
            continue

        scans.append({
            "id": sid,                 
            "image": image_path,       
            "seg": seg
        })
    return scans[:max_scans]


def get_cardiac_roi(lbl_arr: np.ndarray, margin: int):
    coords = np.array(np.where(lbl_arr > 0))
    if coords.size == 0:
        return None
    lo = np.maximum(coords.min(axis=1) - margin, 0)
    hi = np.minimum(coords.max(axis=1) + margin + 1, np.array(lbl_arr.shape))
    return (
        int(lo[0]), int(hi[0]),
        int(lo[1]), int(hi[1]),
        int(lo[2]), int(hi[2])
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: SYSTEM PREPROCESSING PIPELINE
# ════════════════════════════════════════════════════════════════════════════

class Preprocessor:
    @staticmethod
    def orient_to_lps(image: sitk.Image) -> sitk.Image:
        """Standardizes anatomical slice directions to LPS while preserving true physical coordinates."""
        return sitk.DICOMOrient(image, "LPS")

    @staticmethod
    def resample_volume(image: sitk.Image, spacing_mm: float, is_label: bool) -> sitk.Image:
        """Normalizes volumetric scanning grids into isotropic millimeter matrices securely."""
        old_spacing = np.array(image.GetSpacing())
        old_size = np.array(image.GetSize())
        new_size = np.round(old_size * old_spacing / spacing_mm).astype(int).tolist()

        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing([spacing_mm] * 3)
        rs.SetSize(new_size)
        rs.SetOutputOrigin(image.GetOrigin())
        rs.SetOutputDirection(image.GetDirection())
        rs.SetTransform(sitk.Transform())
        rs.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        rs.SetDefaultPixelValue(-1024.0 if not is_label else 0)
        return rs.Execute(image)

    @staticmethod
    def intensity_window(image: sitk.Image, hu_lo: float, hu_hi: float) -> sitk.Image:
        """Applies a strict linear intensity transformation window without z-score distortion."""
        window_filter = sitk.IntensityWindowingImageFilter()
        window_filter.SetWindowMinimum(hu_lo)
        window_filter.SetWindowMaximum(hu_hi)
        window_filter.SetOutputMinimum(0.0)
        window_filter.SetOutputMaximum(1.0)
        return window_filter.Execute(image)

    @classmethod
    def run_memory_resample(cls, img_path: Path, lbl_path: Path = None):
        """OPTIMIZATION GATING: Reads disk files once, branching out both execution tracks in memory."""
        raw = sitk.ReadImage(str(img_path), sitk.sitkFloat32)
        oriented_img = cls.orient_to_lps(raw)
        
       
        reg_raw = cls.resample_volume(oriented_img, CFG.REG_SPACING_MM, is_label=False)
        reg_win = cls.intensity_window(reg_raw, CFG.HU_LO, CFG.HU_HI)
        
        
        eval_raw = cls.resample_volume(oriented_img, CFG.ISO_SPACING_MM, is_label=False)
        
        eval_lbl = None
        if lbl_path is not None:
            lbl = sitk.ReadImage(str(lbl_path), sitk.sitkUInt8)
            oriented_lbl = cls.orient_to_lps(lbl)
            eval_lbl = cls.resample_volume(oriented_lbl, CFG.ISO_SPACING_MM, is_label=True)

        return {
            "reg_raw": reg_raw,
            "reg_win": reg_win,
            "eval_raw": eval_raw,
            "eval_lbl": eval_lbl
        }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: MODULAR REGISTRATION ENGINE (Rigid -> Affine Cascade)
# ════════════════════════════════════════════════════════════════════════════

class RegistrationEngine:
    @staticmethod
    def registration_observer(registration_method):
        """Logs optimizer iterations cleanly at moderated 10-step intervals."""
        current_iter = registration_method.GetOptimizerIteration()
        if current_iter % 10 == 0:
            print(f"      Iteration {current_iter:3d} | Metric Value: {registration_method.GetMetricValue():.6f}")

    @classmethod
    def make_base_registration(cls, sampling_fraction: float, shrink_levels: list, sigma_levels: list) -> sitk.ImageRegistrationMethod:
        """Generates standard baseline tracking parameters using stable, reproducible REGULAR sampling."""
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        R.SetMetricSamplingStrategy(R.REGULAR)
        
        try:
            R.SetMetricSamplingPercentage(sampling_fraction, CFG.RANDOM_SEED)
        except TypeError:
            R.SetMetricSamplingPercentage(sampling_fraction)
            
        R.SetShrinkFactorsPerLevel(shrink_levels)
        R.SetSmoothingSigmasPerLevel(sigma_levels)
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInterpolator(sitk.sitkLinear)
        
        R.AddCommand(sitk.sitkIterationEvent, lambda: cls.registration_observer(R))
        return R

    @classmethod
    def run_rigid_stage(cls, fixed_win: sitk.Image, moving_win: sitk.Image) -> tuple:
        """Optimizes a 6-DOF Euler3D transform ---> geometry alignment."""
        init_tx = sitk.CenteredTransformInitializer(
            fixed_win, moving_win, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        R = cls.make_base_registration(CFG.SAMPLING_FRACTION, CFG.PYRAMID_SHRINK, CFG.PYRAMID_SIGMAS)
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=CFG.RIGID_LEARNING_RATE,
            minStep=CFG.RIGID_MIN_STEP,
            numberOfIterations=CFG.RIGID_ITERS,
            relaxationFactor=CFG.RIGID_RELAXATION
        )
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetInitialTransform(init_tx, inPlace=False)
        
        final_rigid_transform = R.Execute(fixed_win, moving_win)
        return final_rigid_transform, R.GetMetricValue()

    @classmethod
    def run_affine_stage(cls, fixed_win: sitk.Image, moving_win: sitk.Image, rigid_tx: sitk.Euler3DTransform) -> tuple:
        """
        Refines matrix scaling layers by explicitly embedding the Rigid parameters.
        FIXED: Corrected CompositeTransform LIFO sequencing to enforce Rigid(Affine(points)).
        """
        R = cls.make_base_registration(CFG.SAMPLING_FRACTION, CFG.PYRAMID_SHRINK[1:], CFG.PYRAMID_SIGMAS[1:])
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=CFG.AFFINE_LEARNING_RATE,
            minStep=CFG.AFFINE_MIN_STEP,
            numberOfIterations=CFG.AFFINE_ITERS
        )
        R.SetOptimizerScalesFromPhysicalShift()
        
        affine_tx = sitk.AffineTransform(3)
        R.SetMovingInitialTransform(rigid_tx)
        R.SetInitialTransform(affine_tx, inPlace=False)
        
        final_affine_transform = R.Execute(fixed_win, moving_win)
        
        # --- COMPOSITE TRANSFORM CONSTRUCTION ---
        composite_transform = sitk.CompositeTransform(3)
        composite_transform.AddTransform(rigid_tx)               # Added first ->evaluated last
        composite_transform.AddTransform(final_affine_transform)  # Added second -> evaluated first
        
        return composite_transform, R.GetMetricValue()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: GEOMETRICAL TRANSFORMATION LAYER
# ════════════════════════════════════════════════════════════════════════════

class LabelTransformer:
    @staticmethod
    def transform_labels(moving_lbl: sitk.Image, reference_space: sitk.Image, final_tx: sitk.Transform) -> sitk.Image:
        """Transforms structural vessel templates cleanly into the destination coordinate space."""
        return sitk.Resample(
            moving_lbl, 
            reference_space, 
            final_tx, 
            sitk.sitkNearestNeighbor, 
            0, 
            sitk.sitkUInt8
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: ANISOTROPIC PLAQUE VALIDATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

class PlaqueValidator:
    @staticmethod
    def compute_edt_overlap(fixed_raw_img: sitk.Image, transformed_lbl: sitk.Image, cfg: CFG) -> tuple:
        """Validates co-localization ratios using millimetre-accurate distance maps."""
        raw_arr = sitk.GetArrayFromImage(fixed_raw_img)
        lbl_arr = sitk.GetArrayFromImage(transformed_lbl)
        
        roi = get_cardiac_roi(lbl_arr, cfg.CARDIAC_ROI_MARGIN_VOX)
        if roi is None:
            print("      [DEBUG VALIDATOR] Label voxels found globally: 0 (No Cardiac ROI could be generated)")
            return 0.0, 0
        z0, z1, y0, y1, x0, x1 = roi
        print("\n========== ROI DEBUG ==========")
        print(f"ROI = {roi}")
        print(f"ROI shape = {(z1-z0, y1-y0, x1-x0)}")

        print("Image shape:", lbl_arr.shape)

        print("Global label voxels:", np.count_nonzero(lbl_arr))
        raw_roi = raw_arr[z0:z1, y0:y1, x0:x1]
        lbl_roi = lbl_arr[z0:z1, y0:y1, x0:x1]

        print("ROI label voxels:", np.count_nonzero(lbl_roi))
        print("ROI unique values:", np.unique(lbl_roi))

        lbl_dilated = ndi.binary_dilation(lbl_roi > 0, iterations=cfg.LABEL_DILATE_ITERS)
        
        calcium_mask = (raw_roi >= cfg.CALCIUM_HU_LO) & (raw_roi <= cfg.CALCIUM_HU_HI)
        total_calcium_voxels = int(calcium_mask.sum())
        
        # --- Diagnostic Output Prints ---
        print(f"      [DEBUG VALIDATOR] Label non-zero voxels (Full Volume): {np.count_nonzero(lbl_arr)}")
        print(f"      [DEBUG VALIDATOR] Detected Calcium voxels inside ROI: {total_calcium_voxels}")
        print(f"      [DEBUG VALIDATOR] Dilated vessel mask voxels inside ROI: {np.count_nonzero(lbl_dilated)}")
        
        if total_calcium_voxels == 0:
            return 0.0, 0
            
        spacing_zyx = fixed_raw_img.GetSpacing()[::-1]
        edt_matrix = ndi.distance_transform_edt(~lbl_dilated, sampling=spacing_zyx)
        voxels_inside_zone = int((calcium_mask & (edt_matrix <= cfg.DISTANCE_MM)).sum())
        hit_percentage = (voxels_inside_zone / total_calcium_voxels) * 100.0
        
        return hit_percentage, total_calcium_voxels


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: VISUALIZATION SUITE
# ════════════════════════════════════════════════════════════════════════════

class Visualizer:
    @staticmethod
    def generate_diagnostic_overlays(fixed_raw: sitk.Image, transformed_lbl: sitk.Image, scan_id: str, out_dir: Path):
        raw_arr = sitk.GetArrayFromImage(fixed_raw)
        
        print("\n========== HU DEBUG ==========")
        print("HU min:", raw_arr.min())
        print("HU max:", raw_arr.max())
        print("==============================")
        lbl_arr = sitk.GetArrayFromImage(transformed_lbl)
        
        coords = np.array(np.where(lbl_arr > 0))
        if coords.size == 0:
            return
            
        z_mid = int(np.median(coords[0]))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.clip(raw_arr[z_mid], -150, 450), cmap="gray")
        ax.contour(lbl_arr[z_mid] > 0, colors="red", linewidths=1.2)
        ax.set_title(f"Physio-Twin Fluid Boundary Verification: {scan_id} (z={z_mid})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{scan_id}_fluid_mask.png", dpi=120)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8: PIPELINE EXECUTIVE DRIVER
# ════════════════════════════════════════════════════════════════════════════

def main():
    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
    overlays_path = CFG.OUT_DIR / "overlays"
    overlays_path.mkdir(exist_ok=True)

    print("=" * 85)
    print("PHASE 0 -- Discovery (Modular Cross-Modal Whole-Thorax Cascades Activated)")
    print("=" * 85)
    print(f"SimpleITK Binary Release Version: {sitk.Version.VersionString()}")
    
    atlas_cases = discover_atlas_cases(CFG.IMAGECAS_DIR)
    target_scans = discover_coca_scans(CFG.COCA_DIR, CFG.MAX_TARGET_SCANS)
    
    if not atlas_cases:
        raise RuntimeError(f"Critical Asset Missing: No valid templates discovered inside directory '{CFG.IMAGECAS_DIR}'. Check file extensions.")
    if not target_scans:
        raise RuntimeError(f"Critical Cohort Missing: No valid patients discovered inside directory '{CFG.COCA_DIR}'. Check file patterns.")
        
    print(f"Cached Template Records : {len(atlas_cases)} found inside source directories.")
    print(f"Target Patient Datasets : {len(target_scans)} valid patient volumes verified.")

    base_atlas_record = None
    for case in atlas_cases:
        if case["id"] == CFG.BASELINE_ATLAS_ID:
            base_atlas_record = case
            break
    if base_atlas_record is None:
        base_atlas_record = atlas_cases[0]

    print(f"Authoritative Template Baseline Locked: Case {base_atlas_record['id']}")
    
    print("    Preloading and cache-indexing atlas template matrices...")
    atlas_data_pack = Preprocessor.run_memory_resample(base_atlas_record["img"], base_atlas_record["lbl"])
    atlas_win_reg = atlas_data_pack["reg_win"]
    atlas_lbl_eval = atlas_data_pack["eval_lbl"]
    
    atlas_img_raw_check = sitk.ReadImage(str(base_atlas_record["img"]))
    atlas_lbl_raw_check = sitk.ReadImage(str(base_atlas_record["lbl"]))
    assert atlas_lbl_raw_check.GetSize() == atlas_img_raw_check.GetSize(), "Atlas label geometry size mismatch exception."
    assert atlas_lbl_raw_check.GetSpacing() == atlas_img_raw_check.GetSpacing(), "Atlas label physical voxel resolution spacing mismatch exception."
    assert atlas_lbl_raw_check.GetOrigin() == atlas_img_raw_check.GetOrigin(), "Atlas label physical voxel origin coordinate mismatch exception."
    assert atlas_lbl_raw_check.GetDirection() == atlas_img_raw_check.GetDirection(), "Atlas label physical spatial direction matrix mismatch exception."
    print("    Atlas preloading complete. Launching target execution loops.")
    
    cohort_results = []
    failed_cases_log = []

    for idx, scan in enumerate(target_scans, 1):
        sid = scan["id"]
        print(f"\n[{idx}/{len(target_scans)}] Orchestrating Cascade Registration Suite for Volume: {sid}")
        
        current_stage = "Data_Loading"
        time_log = {}
        try:
            t_read = time.time()
            target_data_pack = Preprocessor.run_memory_resample(scan["image"], scan["seg"])
            fixed_win = target_data_pack["reg_win"]
            eval_fixed_raw = target_data_pack["eval_raw"]
            time_log["Read_and_Preprocess_Sec"] = time.time() - t_read

            if idx == 1:
                print(f"      [SANITY CHECK] Target Size: {fixed_win.GetSize()} | Atlas Size: {atlas_win_reg.GetSize()}")
                print(f"      [SANITY CHECK] Target Origin: {fixed_win.GetOrigin()} | Atlas Origin: {atlas_win_reg.GetOrigin()}")

            # ---- Phase 4: Rigid-to-Affine Intensity Alignment Cascade ----
            current_stage = "Rigid_Stage"
            t_rigid = time.time()
            try:
                rigid_tx, _ = RegistrationEngine.run_rigid_stage(fixed_win, atlas_win_reg)
            except RuntimeError as itk_err:
                raise RuntimeError(f"Rigid pass convergence failure: {itk_err}")
            time_log["Rigid_Optimization_Sec"] = time.time() - t_rigid
            
            current_stage = "Affine_Stage"
            t_affine = time.time()
            try:
                final_transform, _ = RegistrationEngine.run_affine_stage(fixed_win, atlas_win_reg, rigid_tx)
            except RuntimeError as itk_err:
                raise RuntimeError(f"Affine pass convergence failure: {itk_err}")
            time_log["Affine_Optimization_Sec"] = time.time() - t_affine

            # ---- Phase 5: Geometrical Label Transformation ----
            current_stage = "Label_Transformation"
            t_trans = time.time()
            transformed_vessels = LabelTransformer.transform_labels(atlas_lbl_eval, eval_fixed_raw, final_transform)
            sitk.WriteImage(
               transformed_vessels,
                str(CFG.OUT_DIR / "registered_vessels.nii.gz")
            )
            
            sitk.WriteImage(
                eval_fixed_raw,
                str(CFG.OUT_DIR / "patient_ct.nii.gz")
            )
            
            time_log["Label_Transformation_Sec"] = time.time() - t_trans

            # ---- Phase 6: Quantitative Validation Tracking ----
            current_stage = "Anisotropic_Validation"
            t_val = time.time()
            
            seg_img = sitk.ReadImage(str(scan["seg"]), sitk.sitkUInt8)
            seg_img = sitk.DICOMOrient(seg_img, "LPS")
            
            
            ca_pct, ca_total = PlaqueValidator.compute_edt_overlap(eval_fixed_raw, transformed_vessels, CFG)
            time_log["Anisotropic_Validation_Sec"] = time.time() - t_val

            # ---- Phase 7: Diagnostic Reporting Plots ----
            Visualizer.generate_diagnostic_overlays(eval_fixed_raw, transformed_vessels, sid, overlays_path)

            passed = (ca_pct is not None) and (ca_pct >= CFG.PASS_THRESHOLD_PCT)
            cohort_results.append({
                "scan_id": sid, "pass": passed, "ca_pct": ca_pct, "total_calcium_voxels": ca_total,
                "fused_atlas": base_atlas_record["id"], **time_log
            })
            
            status_str = "PASS" if passed else "FAIL"
            print(f"    Execution Locked | Latency: Rigid={time_log['Rigid_Optimization_Sec']:.1f}s, Affine={time_log['Affine_Optimization_Sec']:.1f}s | Match={ca_pct:.2f}% | [{status_str}]")

        except Exception as e:
            print(f"    [CRITICAL FAULT] Aborting execution track on case {sid} at stage [{current_stage}]: {e}")
            traceback.print_exc()
            
            failed_cases_log.append({
                "scan_id": sid,
                "failed_stage": current_stage,
                "exception_message": str(e)
            })
            
            cohort_results.append({
                "scan_id": sid, "pass": False, "ca_pct": None, "total_calcium_voxels": None,
                "fused_atlas": base_atlas_record["id"]
            })

    df = pd.DataFrame(cohort_results)
    csv_out = CFG.OUT_DIR / "task3_results.csv"
    df.to_csv(csv_out, index=False)

    if failed_cases_log:
        failed_df = pd.DataFrame(failed_cases_log)
        failed_csv_out = CFG.OUT_DIR / "failed_cases.csv"
        failed_df.to_csv(failed_csv_out, index=False)
        print(f"\n[ALERT] Processing exceptions caught. Debug parameters dumped to: {failed_csv_out}")

    valid_df = df[df["ca_pct"].notna()]
    print("\n" + "=" * 85)
    print("PHYSIO-TWIN INTENSITY PIPELINE RUN COMPLETE")
    print("=" * 85)
    print(f"Total Cohort Scans Evaluated : {len(df)}")
    print(f"Vascular Mask Pass Rate      : {df['pass'].sum()}/{len(df)} ({100*df['pass'].sum()/len(df):.1f}%)")
    if not valid_df.empty:
        print(f"Mean Plaque Matching Score   : {valid_df['ca_pct'].mean():.2f}%")
        print(f"Mean Rigid Loop Latency     : {valid_df['Rigid_Optimization_Sec'].mean():.2f}s")
        print(f"Mean Affine Loop Latency    : {valid_df['Affine_Optimization_Sec'].mean():.2f}s")
    print(f"Scorecard Matrix CSV Location : {csv_out}")
    print("=" * 85)


if __name__ == "__main__":
    main()