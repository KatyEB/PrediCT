"""
PrediCT Task 3 — Coronary Atlas Registration Pipeline (v3)
===========================================================
Rigid → Affine cascade registering ImageCAS CCTA atlas to COCA NCCT scans.
Implements all algorithmic corrections from MICCAI-level review.

CHANGES FROM v2 (masking_corrected.py):
  1. METRIC MASK instead of crop-parameterized affine.
       Affine now runs on FULL fixed/moving images with a spatial mask isolating
       the cardiac ROI. This eliminates the coordinate-space mismatch where an
       affine parameterized on a crop subvolume was applied to a full-resolution
       label volume at a different scale. (See Section 4, run_affine_stage.)

  2. OPTIMIZER EARLY-TERMINATION FIX.
       AFFINE_MIN_STEP raised 0.0001 → 0.001, AFFINE_LEARNING_RATE raised
       0.25 → 1.0, relaxationFactor=0.7 added. Previously the affine was hitting
       minStep after ~30 iterations (verified: 0.8s wall time for 200 iterations).
       Genuine convergence now takes 15–40s.

  3. RIGID SCALE FIX.
       SetOptimizerScalesFromJacobian() → SetOptimizerScalesFromPhysicalShift().
       Jacobian scales mix radians and mm, under-scaling rotation for Euler3D.
       PhysicalShift normalizes by physical displacement magnitude per parameter.

  4. MATTES MI BIN COUNT.
       32 → 50 bins. At 8× pyramid shrink with 20% sampling, 32 bins gives
       ~375 samples/bin — too noisy for reliable MI gradient estimation.
       50 bins at full resolution gives adequate density at all pyramid levels.

  5. AFFINE SAMPLING STRATEGY.
       REGULAR → RANDOM for the affine stage. RANDOM better captures sparse
       structures (vessel walls, calcium deposits) that REGULAR may miss with
       a fixed grid, especially on the already-aligned crop-masked volume.

  6. ADAPTIVE ROI MARGIN.
       Fixed voxel margin → physical-unit margin (20mm default).
       Converts to voxels at runtime using spacing, with minimum 10-voxel floor.
       Prevents too-tight ROI on large hearts or unusual cardiac axis.

  7. REDUNDANT RESAMPLING REMOVED.
       The intermediate rigid_atlas_win full-volume resample used to build
       moving_win_cropped is eliminated. The affine now uses a metric mask on
       the full images, so no moving crop is needed. Saves ~0.3s per scan.
"""

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
# SECTION 1: CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

class CFG:
    # Paths
    IMAGECAS_DIR = Path("/Users/karan/Desktop/PrediCT/1-200")
    COCA_DIR     = Path("/Users/karan/Desktop/PrediCT/COCA")
    OUT_DIR      = Path("/Users/karan/Desktop/PrediCT/output")

    # Cohort
    MAX_TARGET_SCANS    = 25
    BLACKLIST_PATIENTS  = {"12", "197", "268"}
    BASELINE_ATLAS_ID   = "1"

    # Preprocessing
    HU_LO          = -200.0
    HU_HI          =  600.0
    ISO_SPACING_MM =  1.0       # eval resolution
    REG_SPACING_MM =  1.5       # registration resolution

    # Rigid optimizer
    # [FIX 3] PhysicalShift scales applied (see run_rigid_stage)
    RIGID_LEARNING_RATE = 2.0
    RIGID_MIN_STEP      = 0.001
    RIGID_ITERS         = 250
    RIGID_RELAXATION    = 0.5

    # Rigid pyramid
    PYRAMID_SHRINK = [8, 4, 2, 1]
    PYRAMID_SIGMAS = [3, 2, 1, 0]   # physical mm
    SAMPLING_FRACTION = 0.20         # REGULAR sampling for rigid

    # Affine optimizer
    # [FIX 2] Raised learning rate and minStep; added relaxation
    AFFINE_LEARNING_RATE = 1.0       # was 0.25 — step too small for 12-DOF space
    AFFINE_MIN_STEP      = 0.001     # was 0.0001 — caused early termination at ~30 iters
    AFFINE_ITERS         = 200
    AFFINE_RELAXATION    = 0.7       # was missing — prevents exponential step decay

    # Affine pyramid (3-level; skip 8× since crops are already small)
    AFFINE_SHRINK = [4, 2, 1]
    AFFINE_SIGMAS = [2, 1, 0]

    # [FIX 6] Adaptive ROI margin in physical mm (converted to voxels at runtime)
    ROI_MARGIN_MM        = 30.0      # was fixed 12 voxels (18mm) — now physical units
    ROI_MARGIN_VOX_MIN   = 10       # floor in voxels for very fine spacings

    # Validation
    VALIDATION_ROI_MARGIN_VOX = 45
    DISTANCE_MM               = 10.0
    PASS_THRESHOLD_PCT        = 70.0
    LABEL_DILATE_ITERS        = 2

    RANDOM_SEED = 42


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA DISCOVERY
# ════════════════════════════════════════════════════════════════════════════

def discover_atlas_cases(imagecas_dir: Path) -> list:
    atlas_dict = defaultdict(dict)
    for f in imagecas_dir.glob("*.nii.gz"):
        name = f.name
        if "label" in name:
            case_id = (name.split(".")[0]
                       .replace("_label", "").replace(".label", "").replace("label", ""))
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


def discover_coca_scans(coca_dir: Path, max_scans: int) -> list:
    scans = []
    for scan_dir in sorted(
        p for p in coca_dir.iterdir()
        if p.is_dir() and p.name not in CFG.BLACKLIST_PATIENTS
    ):
        sid      = scan_dir.name
        raw_img  = scan_dir / f"{sid}_raw_img.nii.gz"
        win_img  = scan_dir / f"{sid}_img.nii.gz"
        seg      = scan_dir / f"{sid}_seg.nii.gz"
        img_path = raw_img if raw_img.exists() else (win_img if win_img.exists() else None)
        if img_path is None or not seg.exists():
            continue
        scans.append({"id": sid, "image": img_path, "seg": seg})
    return scans[:max_scans]


def get_cardiac_roi(lbl_arr: np.ndarray, margin_mm: float, spacing_mm: float) -> tuple:
    """
    [FIX 6] Adaptive ROI with physical-unit margin.

    Converts margin_mm to voxels at runtime using spacing_mm, with a minimum
    floor of CFG.ROI_MARGIN_VOX_MIN voxels. Prevents too-tight crops on large
    hearts or patients with unusual cardiac axis orientation.
    """
    coords = np.array(np.where(lbl_arr > 0))
    if coords.size == 0:
        return None
    margin_vox = max(int(np.ceil(margin_mm / spacing_mm)), CFG.ROI_MARGIN_VOX_MIN)
    lo = np.maximum(coords.min(axis=1) - margin_vox, 0)
    hi = np.minimum(coords.max(axis=1) + margin_vox + 1, np.array(lbl_arr.shape))
    return tuple(int(v) for v in [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])


def get_validation_roi(lbl_arr: np.ndarray, margin_vox: int) -> tuple:
    """Separate ROI getter for validation — keeps fixed-voxel margin for consistency."""
    coords = np.array(np.where(lbl_arr > 0))
    if coords.size == 0:
        return None
    lo = np.maximum(coords.min(axis=1) - margin_vox, 0)
    hi = np.minimum(coords.max(axis=1) + margin_vox + 1, np.array(lbl_arr.shape))
    return tuple(int(v) for v in [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

class Preprocessor:

    @staticmethod
    def orient_to_lps(image: sitk.Image) -> sitk.Image:
        """Standardize to LPS orientation while preserving physical coordinates."""
        return sitk.DICOMOrient(image, "LPS")

    @staticmethod
    def resample_volume(image: sitk.Image, spacing_mm: float, is_label: bool) -> sitk.Image:
        """Resample to isotropic spacing. NearestNeighbor for labels, Linear for images."""
        old_spacing = np.array(image.GetSpacing())
        old_size    = np.array(image.GetSize())
        new_size    = np.round(old_size * old_spacing / spacing_mm).astype(int).tolist()

        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing([spacing_mm] * 3)
        rs.SetSize(new_size)
        rs.SetOutputOrigin(image.GetOrigin())
        rs.SetOutputDirection(image.GetDirection())
        rs.SetTransform(sitk.Transform())
        rs.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        rs.SetDefaultPixelValue(0 if is_label else -1024.0)
        return rs.Execute(image)

    @staticmethod
    def intensity_window(image: sitk.Image, hu_lo: float, hu_hi: float) -> sitk.Image:
        """Linear HU window → [0, 1]. Used only for registration images, not validation."""
        f = sitk.IntensityWindowingImageFilter()
        f.SetWindowMinimum(hu_lo)
        f.SetWindowMaximum(hu_hi)
        f.SetOutputMinimum(0.0)
        f.SetOutputMaximum(1.0)
        return f.Execute(image)

    @classmethod
    def run_memory_resample(cls, img_path: Path, lbl_path: Path = None) -> dict:
        """
        Single disk read, two resolution tracks:
          reg_win  — 1.5mm windowed [0,1]  → used by registration optimizer
          eval_raw — 1.0mm raw HU          → used by validator (never windowed)
          eval_lbl — 1.0mm label mask       → COCA calcium ground truth for validation
        """
        raw         = sitk.ReadImage(str(img_path), sitk.sitkFloat32)
        oriented    = cls.orient_to_lps(raw)
        reg_raw     = cls.resample_volume(oriented, CFG.REG_SPACING_MM, is_label=False)
        reg_win     = cls.intensity_window(reg_raw, CFG.HU_LO, CFG.HU_HI)
        eval_raw    = cls.resample_volume(oriented, CFG.ISO_SPACING_MM, is_label=False)

        eval_lbl = None
        if lbl_path is not None:
            lbl      = sitk.ReadImage(str(lbl_path), sitk.sitkUInt8)
            eval_lbl = cls.resample_volume(cls.orient_to_lps(lbl), CFG.ISO_SPACING_MM, is_label=True)

        return {"reg_raw": reg_raw, "reg_win": reg_win, "eval_raw": eval_raw, "eval_lbl": eval_lbl}


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: REGISTRATION ENGINE  (Rigid → Affine)
# ════════════════════════════════════════════════════════════════════════════

class RegistrationEngine:

    @staticmethod
    def _observer(R: sitk.ImageRegistrationMethod):
        it = R.GetOptimizerIteration()
        if it % 10 == 0:
            print(f"      iter {it:3d} | MI = {R.GetMetricValue():.6f}")

    @classmethod
    def _make_method(cls, R: sitk.ImageRegistrationMethod,
                     shrink: list, sigmas: list,
                     n_bins: int, fraction: float,
                     strategy) -> None:
        """
        Configure shared settings on an already-created ImageRegistrationMethod.
        [FIX 4] n_bins=50 (was 32) for adequate MI histogram density at coarse levels.
        """
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=n_bins)
        R.SetMetricSamplingStrategy(strategy)
        try:
            R.SetMetricSamplingPercentage(fraction, CFG.RANDOM_SEED)
        except TypeError:
            R.SetMetricSamplingPercentage(fraction)
        R.SetShrinkFactorsPerLevel(shrink)
        R.SetSmoothingSigmasPerLevel(sigmas)
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInterpolator(sitk.sitkLinear)
        R.AddCommand(sitk.sitkIterationEvent, lambda: cls._observer(R))

    @classmethod
    def run_rigid_stage(cls, fixed_win: sitk.Image, moving_win: sitk.Image) -> tuple:
        """
        6-DOF Euler3D rigid registration on full thorax images.

        [FIX 3] SetOptimizerScalesFromPhysicalShift replaces SetOptimizerScalesFromJacobian.
        The Jacobian of Euler3D mixes radians and mm, causing the optimizer to
        over-step rotations and under-step translations. PhysicalShift normalizes
        each parameter by its physical displacement per unit change, giving correct
        relative step sizes across all 6 DOF.
        """
        init_tx = sitk.CenteredTransformInitializer(
            fixed_win, moving_win,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        R = sitk.ImageRegistrationMethod()
        cls._make_method(
            R,
            shrink   = CFG.PYRAMID_SHRINK,
            sigmas   = CFG.PYRAMID_SIGMAS,
            n_bins   = 50,                          # [FIX 4]
            fraction = CFG.SAMPLING_FRACTION,
            strategy = sitk.ImageRegistrationMethod.REGULAR
        )
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate    = CFG.RIGID_LEARNING_RATE,
            minStep         = CFG.RIGID_MIN_STEP,
            numberOfIterations = CFG.RIGID_ITERS,
            relaxationFactor   = CFG.RIGID_RELAXATION
        )
        R.SetOptimizerScalesFromPhysicalShift()     # [FIX 3]
        R.SetInitialTransform(init_tx, inPlace=False)
        
        rigid_tx = R.Execute(fixed_win, moving_win)
        print(f"      Rigid final MI = {R.GetMetricValue():.6f}")
        return rigid_tx, R.GetMetricValue()

    @classmethod
    def run_affine_stage(cls,
                         fixed_win:  sitk.Image,
                         moving_win: sitk.Image,
                         rigid_tx:   sitk.Transform,
                         roi:        tuple) -> tuple:
        """
        12-DOF Affine registration using a SPATIAL METRIC MASK over the cardiac ROI.

        [FIX 1] Metric mask instead of crop-parameterized affine.
          The previous version cropped fixed_win and moving_win to the cardiac ROI,
          ran the affine on the crops, then applied the crop-parameterized transform
          to atlas_lbl_eval at full 1.0mm resolution. The affine center was set in
          the crop's physical space; applying it to a full-resolution reference
          caused geometric inconsistency proportional to the distance from the crop
          center to the full-image center.

          This version runs the affine on the FULL fixed_win and full rigidly-
          resampled moving_win, restricting the MI metric to cardiac-ROI samples
          via SetMetricFixedMask. The affine transform parameters are now defined
          in the same physical coordinate system as the label volume, so no
          coordinate-space mismatch exists when warping atlas_lbl_eval.

        [FIX 2] Corrected optimizer parameters:
          - AFFINE_LEARNING_RATE 0.25 → 1.0 (was too small for 12-DOF space)
          - AFFINE_MIN_STEP 0.0001 → 0.001 (was causing early exit at ~30 iters)
          - relaxationFactor=0.7 added (prevents exponential step decay)

        [FIX 5] RANDOM sampling (was REGULAR):
          Random sampling better captures sparse structures (vessel walls, small
          calcium deposits) that a fixed-grid REGULAR pattern may miss, especially
          after rigid pre-alignment has reduced the gross positional error.

        Args:
            fixed_win:  Full 1.5mm windowed patient image.
            moving_win: Full 1.5mm windowed atlas image (NOT pre-resampled with rigid).
                        rigid_tx is passed to SetMovingInitialTransform so the optimizer
                        starts from the rigidly-aligned position.
            rigid_tx:   Rigid transform from Stage A.
            roi:        (z0,z1,y0,y1,x0,x1) cardiac bounding box in fixed_win voxels.
        """
        # Build binary spatial mask isolating the cardiac ROI in fixed_win space.
        # The metric samples ONLY within this region; the transform parameters
        # remain valid over the full image physical space.
        fz0, fz1, fy0, fy1, fx0, fx1 = roi
        mask_arr = np.zeros(sitk.GetArrayFromImage(fixed_win).shape, dtype=np.uint8)
        mask_arr[fz0:fz1, fy0:fy1, fx0:fx1] = 1
        mask_arr = ndi.binary_dilation(
            mask_arr,
            iterations=8
        ).astype(np.uint8)
        metric_mask = sitk.GetImageFromArray(mask_arr)
        metric_mask.CopyInformation(fixed_win)

        R = sitk.ImageRegistrationMethod()
        cls._make_method(
            R,
            shrink   = CFG.AFFINE_SHRINK,
            sigmas   = CFG.AFFINE_SIGMAS,
            n_bins   = 50,                              # [FIX 4]
            fraction = 0.50,
            strategy = sitk.ImageRegistrationMethod.RANDOM  # [FIX 5]
        )
        R.SetOptimizerAsRegularStepGradientDescent(
            learningRate       = CFG.AFFINE_LEARNING_RATE,   # [FIX 2]
            minStep            = CFG.AFFINE_MIN_STEP,         # [FIX 2]
            numberOfIterations = CFG.AFFINE_ITERS,
            relaxationFactor   = CFG.AFFINE_RELAXATION        # [FIX 2]
        )
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetMetricFixedMask(metric_mask)                     # [FIX 1]

        # Initialize affine from rigid: optimizer sees the pre-aligned atlas image.
        # The returned affine captures ONLY the residual local deformation.
        affine_tx = sitk.AffineTransform(3)
        init_affine = sitk.CenteredTransformInitializer(
            fixed_win, moving_win, affine_tx,
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        R.SetMovingInitialTransform(rigid_tx)   # pre-align moving before metric eval
        R.SetInitialTransform(init_affine, inPlace=False)

        print("\n========== AFFINE MASK DEBUG ==========")
        print("Metric mask size   :", metric_mask.GetSize())
        print("Metric mask voxels :", np.count_nonzero(mask_arr))
        print("=======================================\n")

        try:
            final_affine = R.Execute(fixed_win, moving_win)
            print(f"      Affine final MI = {R.GetMetricValue():.6f}")
        except RuntimeError:
            print("Affine failed.")
            composite = sitk.CompositeTransform(3)
            composite.AddTransform(rigid_tx)
            return composite, R.GetMetricValue()
            

        # Inspect return type — confirmed Case B: CompositeTransform(1 × AffineTransform)
        # rigid_tx is NOT embedded, so we add it explicitly below.
        if isinstance(final_affine, sitk.CompositeTransform):
            n = final_affine.GetNumberOfTransforms()
            sub_names = [final_affine.GetNthTransform(i).GetName() for i in range(n)]
            print(f"      Execute returned CompositeTransform with {n} sub-transform(s): {sub_names}")
        else:
            print(f"      Execute returned {final_affine.GetName()}")

        # [FIX 1 continued] Compose rigid + affine in full physical space.
        # SimpleITK AddTransform order: last-added is evaluated first.
        # AddTransform(rigid) then AddTransform(affine_residual):
        #   moving point → rigid → affine_residual → fixed point   (correct)
        composite = sitk.CompositeTransform(3)
        composite.AddTransform(rigid_tx)
        composite.AddTransform(final_affine)

        return composite, R.GetMetricValue()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: LABEL TRANSFORMATION
# ════════════════════════════════════════════════════════════════════════════

class LabelTransformer:

    @staticmethod
    def transform_labels(moving_lbl:     sitk.Image,
                         reference_space: sitk.Image,
                         final_tx:        sitk.Transform) -> sitk.Image:
        """
        Warp atlas vessel label into patient space.

        NearestNeighbor interpolation preserves binary mask integrity.
        reference_space is eval_fixed_raw (1.0mm), so the output label
        is at the same resolution as the COCA calcium ground truth.
        """
        return sitk.Resample(
            moving_lbl,
            reference_space,
            final_tx,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6: VALIDATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

class PlaqueValidator:

    @staticmethod
    def compute_edt_overlap(fixed_raw_img:   sitk.Image,
                            transformed_lbl: sitk.Image,
                            coca_seg:        sitk.Image,
                            cfg) -> tuple:
        """
        Compute percentage of COCA ground-truth calcium voxels within
        cfg.DISTANCE_MM of the transformed atlas vessel mask.

        Uses COCA plist-derived calcium segmentation (coca_seg) as ground truth,
        NOT HU thresholding. HU >= 130 on a full thorax flags ~270k voxels
        (ribs, spine, sternum), making overlap percentages meaningless.
        The COCA _seg.nii.gz contains only annotated coronary calcium (~50–5000 voxels).

        Steps:
          1. Dilate transformed vessel mask by cfg.LABEL_DILATE_ITERS iterations
             (morphological buffer ~1.5mm at 1.0mm spacing).
          2. Compute Euclidean distance transform from vessel mask surface.
          3. Count calcium voxels with EDT <= cfg.DISTANCE_MM.
        """
        lbl_arr = sitk.GetArrayFromImage(transformed_lbl)
        seg_arr = sitk.GetArrayFromImage(coca_seg)

        roi = get_validation_roi(lbl_arr, cfg.VALIDATION_ROI_MARGIN_VOX)
        if roi is None:
            print("      [VALIDATOR] No label voxels — skipping.")
            return 0.0, 0

        z0, z1, y0, y1, x0, x1 = roi
        lbl_roi = lbl_arr[z0:z1, y0:y1, x0:x1]
        seg_roi = seg_arr[z0:z1, y0:y1, x0:x1]

        n_lbl = int(np.count_nonzero(lbl_roi))
        n_ca  = int(np.count_nonzero(seg_roi))

        print(f"\n========== VALIDATION ==========")
        print(f"ROI shape          : {(z1-z0, y1-y0, x1-x0)}")
        print(f"Vessel mask voxels : {n_lbl}")
        print(f"Calcium voxels     : {n_ca}  (COCA ground truth)")

        if n_ca == 0:
            print("      No calcium in this scan — score = 0.0%")
            print("================================\n")
            return 0.0, 0

        lbl_dilated = ndi.binary_dilation(lbl_roi > 0, iterations=cfg.LABEL_DILATE_ITERS)
        spacing_zyx = fixed_raw_img.GetSpacing()[::-1]   # (z,y,x) order for scipy
        edt         = ndi.distance_transform_edt(~lbl_dilated, sampling=spacing_zyx)

        calcium_mask    = seg_roi > 0
        inside          = int((calcium_mask & (edt <= cfg.DISTANCE_MM)).sum())
        hit_pct         = (inside / n_ca) * 100.0

        print(f"Dilated mask voxels: {int(np.count_nonzero(lbl_dilated))}")
        print(f"Ca inside ±{cfg.DISTANCE_MM}mm : {inside}/{n_ca} = {hit_pct:.1f}%")
        print("================================\n")

        return hit_pct, n_ca


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7: VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

class Visualizer:

    @staticmethod
    def generate_overlay(fixed_raw:       sitk.Image,
                         transformed_lbl: sitk.Image,
                         scan_id:         str,
                         out_dir:         Path) -> None:
        """
        Save a 2D axial overlay at the median label z-slice.
        Gray: patient CT windowed to [-150, 450] HU for soft-tissue contrast.
        Red contour: transformed atlas vessel mask boundary.
        """
        raw_arr = sitk.GetArrayFromImage(fixed_raw)
        lbl_arr = sitk.GetArrayFromImage(transformed_lbl)

        coords = np.array(np.where(lbl_arr > 0))
        if coords.size == 0:
            print(f"      [VIS] No label voxels for {scan_id} — skipping overlay.")
            return

        z_mid = int(np.median(coords[0]))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.clip(raw_arr[z_mid], -150, 450), cmap="gray", vmin=-150, vmax=450)
        ax.contour(lbl_arr[z_mid] > 0, colors="red", linewidths=1.2)
        ax.set_title(f"Atlas→Patient Registration: {scan_id}  (z={z_mid})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{scan_id}_overlay.png", dpi=120)
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAIN PIPELINE LOOP
# ════════════════════════════════════════════════════════════════════════════

def main():
    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
    overlays_dir = CFG.OUT_DIR / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    print("=" * 85)
    print("PrediCT Task 3 — Coronary Atlas Registration Pipeline v3")
    print("=" * 85)
    print(f"SimpleITK {sitk.Version.VersionString()}")

    # ── Discovery ────────────────────────────────────────────────────────────
    atlas_cases  = discover_atlas_cases(CFG.IMAGECAS_DIR)
    target_scans = discover_coca_scans(CFG.COCA_DIR, CFG.MAX_TARGET_SCANS)

    if not atlas_cases:
        raise RuntimeError(f"No atlas cases found in {CFG.IMAGECAS_DIR}")
    if not target_scans:
        raise RuntimeError(f"No COCA scans found in {CFG.COCA_DIR}")

    print(f"Atlas cases  : {len(atlas_cases)} found")
    print(f"Target scans : {len(target_scans)} selected")

    # Select baseline atlas
    base_record = next(
        (c for c in atlas_cases if c["id"] == CFG.BASELINE_ATLAS_ID),
        atlas_cases[0]
    )
    print(f"Atlas locked : Case {base_record['id']}")

    # ── Pre-load atlas (once, outside patient loop) ───────────────────────
    print("Pre-loading atlas...")
    atlas_pack     = Preprocessor.run_memory_resample(base_record["img"], base_record["lbl"])
    atlas_win_reg  = atlas_pack["reg_win"]    # 1.5mm windowed — for registration
    atlas_lbl_eval = atlas_pack["eval_lbl"]   # 1.0mm label    — for label warp

    # Separate 1.5mm label for rigid-projected ROI extraction
    atlas_lbl_reg = Preprocessor.resample_volume(
        Preprocessor.orient_to_lps(
            sitk.ReadImage(str(base_record["lbl"]), sitk.sitkUInt8)
        ),
        CFG.REG_SPACING_MM, is_label=True
    )

    # Sanity check atlas geometry consistency
    _img_check = sitk.ReadImage(str(base_record["img"]))
    _lbl_check = sitk.ReadImage(str(base_record["lbl"]))
    assert _lbl_check.GetSize()      == _img_check.GetSize(),      "Atlas size mismatch"
    assert _lbl_check.GetSpacing()   == _img_check.GetSpacing(),   "Atlas spacing mismatch"
    assert _lbl_check.GetOrigin()    == _img_check.GetOrigin(),    "Atlas origin mismatch"
    assert _lbl_check.GetDirection() == _img_check.GetDirection(), "Atlas direction mismatch"
    print("Atlas geometry verified. Starting patient loop.\n")

    cohort_results   = []
    failed_cases_log = []

    # ── Patient loop ─────────────────────────────────────────────────────────
    for idx, scan in enumerate(target_scans, 1):
        sid          = scan["id"]
        current_stage = "Data_Loading"
        time_log      = {}

        print(f"\n{'─'*85}")
        print(f"[{idx:02d}/{len(target_scans)}] Patient: {sid}")
        print(f"{'─'*85}")

        try:
            # ── Pre-process patient ──────────────────────────────────────────
            t0 = time.time()
            patient_pack   = Preprocessor.run_memory_resample(scan["image"], scan["seg"])
            fixed_win      = patient_pack["reg_win"]    # 1.5mm windowed — for registration
            eval_fixed_raw = patient_pack["eval_raw"]   # 1.0mm raw HU   — for validation
            eval_coca_seg  = patient_pack["eval_lbl"]   # 1.0mm seg       — calcium GT
            time_log["preprocess_sec"] = time.time() - t0

            print(f"  Fixed  (reg):  {fixed_win.GetSize()}  @ {fixed_win.GetSpacing()[0]:.1f}mm")
            print(f"  Moving (reg):  {atlas_win_reg.GetSize()}  @ {atlas_win_reg.GetSpacing()[0]:.1f}mm")

            # ── Stage A: Rigid (6-DOF) ───────────────────────────────────────
            current_stage = "Rigid_Stage"
            t0 = time.time()
            rigid_tx, rigid_mi = RegistrationEngine.run_rigid_stage(fixed_win, atlas_win_reg)
            time_log["rigid_sec"] = time.time() - t0

            # Project atlas label with rigid → identify cardiac ROI in patient space
            rigid_atlas_lbl = sitk.Resample(
                atlas_lbl_reg, fixed_win, rigid_tx,
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
            )
            rigid_lbl_arr = sitk.GetArrayFromImage(rigid_atlas_lbl)
            n_rigid_vox   = int(np.count_nonzero(rigid_lbl_arr))
            print(f"  Rigid label voxels projected: {n_rigid_vox}")


            if n_rigid_vox == 0:
                raise RuntimeError("No label voxels after rigid — rigid alignment failed.")

            # [FIX 6] Adaptive ROI in physical mm
            roi_fixed = get_cardiac_roi(
                rigid_lbl_arr,
                margin_mm  = CFG.ROI_MARGIN_MM,
                spacing_mm = CFG.REG_SPACING_MM
            )
            if roi_fixed is None:
                raise RuntimeError("Could not extract cardiac ROI from rigid-projected label.")

            fz0, fz1, fy0, fy1, fx0, fx1 = roi_fixed
            roi_voxels = (
            (fz1 - fz0)
            * 
            (fy1 - fy0)
            *
            (fx1 - fx0)
            )
            print("ROI volume:", roi_voxels)
            
            roi_shape = (fz1-fz0, fy1-fy0, fx1-fx0)
            print(f"  Cardiac ROI (vox): {roi_fixed}  shape={roi_shape}")

            # ── Stage B: Affine (12-DOF) with metric mask ─────────────────────
            # [FIX 1] Pass FULL fixed_win and FULL atlas_win_reg (not pre-resampled).
            # rigid_tx goes to SetMovingInitialTransform inside run_affine_stage,
            # so the optimizer evaluates from the rigidly-aligned position without
            # a separate explicit resampling step.
            # [FIX 7] Redundant rigid_atlas_win full-volume resample is eliminated.
            current_stage = "Affine_Stage"
            t0 = time.time()
            final_transform, affine_mi = RegistrationEngine.run_affine_stage(
                fixed_win    = fixed_win,
                moving_win   = atlas_win_reg,
                rigid_tx     = rigid_tx,
                roi          = roi_fixed
            )
            time_log["affine_sec"] = time.time() - t0

            # ── Phase 5: Warp atlas label into patient space ──────────────────
            current_stage = "Label_Transformation"
            t0 = time.time()
            # Apply composite (rigid + affine) to atlas_lbl_eval at 1.0mm.
            # Both transforms are parameterized in the same global physical space,
            # so no coordinate-space mismatch exists here.
            transformed_vessels = LabelTransformer.transform_labels(
                atlas_lbl_eval, eval_fixed_raw, final_transform
            )
            time_log["label_warp_sec"] = time.time() - t0

            warp_arr    = sitk.GetArrayFromImage(transformed_vessels)
            warp_coords = np.where(warp_arr > 0)
            n_warp      = warp_coords[0].size
            print(f"  Warped label voxels: {n_warp}")
            if n_warp == 0:
                print("  [WARNING] No label voxels after warp — transform may have diverged.")

            # Save outputs for inspection in 3D Slicer
            sitk.WriteImage(transformed_vessels,
                            str(CFG.OUT_DIR / f"{sid}_vessels.nii.gz"))
            sitk.WriteImage(eval_fixed_raw,
                            str(CFG.OUT_DIR / f"{sid}_patient_ct.nii.gz"))

            # ── Phase 6: Validate ─────────────────────────────────────────────
            current_stage = "Validation"
            t0 = time.time()
            ca_pct, ca_total = PlaqueValidator.compute_edt_overlap(
                eval_fixed_raw, transformed_vessels, eval_coca_seg, CFG
            )
            time_log["validation_sec"] = time.time() - t0

            # ── Phase 7: Overlay ──────────────────────────────────────────────
            Visualizer.generate_overlay(eval_fixed_raw, transformed_vessels, sid, overlays_dir)

            passed     = ca_pct >= CFG.PASS_THRESHOLD_PCT
            status_str = "PASS ✓" if passed else "FAIL ✗"
            total_sec  = time_log["rigid_sec"] + time_log["affine_sec"]

            print(f"  {'─'*60}")
            print(f"  Result : {status_str}  |  Ca% = {ca_pct:.1f}%  |  "
                  f"Time = {total_sec:.1f}s  (rigid={time_log['rigid_sec']:.1f}s, "
                  f"affine={time_log['affine_sec']:.1f}s)")
            print(f"  {'─'*60}")

            cohort_results.append({
                "scan_id"             : sid,
                "pass"                : passed,
                "ca_pct"              : ca_pct,
                "total_calcium_voxels": ca_total,
                "rigid_mi"            : rigid_mi,
                "affine_mi"           : affine_mi,
                "atlas_id"            : base_record["id"],
                **time_log
            })

        except Exception as e:
            print(f"\n  [CRITICAL FAULT] at stage [{current_stage}]: {e}")
            traceback.print_exc()
            failed_cases_log.append({
                "scan_id"          : sid,
                "failed_stage"     : current_stage,
                "exception_message": str(e)
            })
            cohort_results.append({
                "scan_id"             : sid,
                "pass"                : False,
                "ca_pct"              : None,
                "total_calcium_voxels": None,
                "atlas_id"            : base_record["id"],
            })

    # ── Results ───────────────────────────────────────────────────────────────
    df       = pd.DataFrame(cohort_results)
    valid_df = df[df["ca_pct"].notna()]

    df.to_csv(CFG.OUT_DIR / "task3_results.csv", index=False)
    if failed_cases_log:
        pd.DataFrame(failed_cases_log).to_csv(
            CFG.OUT_DIR / "failed_cases.csv", index=False
        )

    n_pass = int(df["pass"].sum())
    n_tot  = len(df)

    print("\n" + "=" * 85)
    print("PIPELINE COMPLETE")
    print("=" * 85)
    print(f"Scans evaluated : {n_tot}")
    print(f"Pass rate       : {n_pass}/{n_tot}  ({100*n_pass/n_tot:.1f}%)")
    if not valid_df.empty:
        print(f"Mean Ca%        : {valid_df['ca_pct'].mean():.2f}%")
        print(f"Median Ca%      : {valid_df['ca_pct'].median():.2f}%")
        print(f"Mean rigid time : {valid_df['rigid_sec'].mean():.1f}s")
        print(f"Mean affine time: {valid_df['affine_sec'].mean():.1f}s")
    print(f"Results CSV     : {CFG.OUT_DIR / 'task3_results.csv'}")
    print("=" * 85)


if __name__ == "__main__":
    main()