import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import SimpleITK as sitk

class RegistrationCFG:
    # --- Paths adjusted to match your SanDisk resampler output ---
    IMAGECAS_DIR = Path("/Users/karan/Desktop/PrediCT/1-200") 
    COCA_RESAMPLED_DIR = Path("/Volumes/SanDisk/PrediCT/data_resampled")
    OUT_DIR = Path("/Volumes/SanDisk/PrediCT/registration_output")
    
    # --- Optimization Parameters ---
    SAMPLE_PERCENTAGE = 0.10       # 10% sampling is fast and highly stable for MI
    MIN_CALCIUM_VOXELS = 15
    PASS_THRESHOLD_PCT = 70.0
    DISTANCE_MM = 10.0

def build_fixed_heart_mask(fixed_image: sitk.Image) -> sitk.Image:
    """
    Creates a mathematically sound foreground mask restricted to the chest center.
    Perfectly matches the fixed image physical grid to prevent ITK crashes.
    """
    # Step 1: Otsu thresholding to separate tissue from air cavity
    otsu_mask = sitk.OtsuThreshold(fixed_image, 0, 1, 128)
    
    # Step 2: Spatial cropping to focus the metric on the mediastinum (heart region)
    # This prevents spinal bones or chest walls from pulling the registration away
    mask_arr = sitk.GetArrayFromImage(otsu_mask)
    z, y, x = mask_arr.shape
    
    spatial_bounding_box = np.zeros_like(mask_arr)
    spatial_bounding_box[:, int(y*0.25):int(y*0.85), int(x*0.25):int(x*0.75)] = 1
    
    clean_mask_arr = (mask_arr & spatial_bounding_box).astype(np.uint8)
    
    # Step 3: Reconstruct SITK Image and inject matching physical space headers
    heart_mask = sitk.GetImageFromArray(clean_mask_arr)
    heart_mask.CopyInformation(fixed_image)
    return sitk.Cast(heart_mask, sitk.sitkUInt8)

def execute_registration_cascade(fixed_img: sitk.Image, moving_img: sitk.Image, fixed_mask: sitk.Image):
    """
    Runs a multi-resolution Rigid -> Affine registration cascade using Mattes MI
    to seamlessly handle contrast differences between CCTA and NCCT.
    """
    # Initialize the transform using center of gravity alignment
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img, moving_img, 
        sitk.AffineTransform(3), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    R = sitk.ImageRegistrationMethod()
    
    # Mattes Mutual Information is critical for crossing CCTA -> NCCT contrast gaps
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(RegistrationCFG.SAMPLE_PERCENTAGE, seed=42)
    R.SetMetricFixedMask(fixed_mask)
    
    # Multi-resolution multi-stage configurations for speed and accuracy
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer tuning
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=150
    )
    R.SetOptimizerScalesFromPhysicalShift()
    
    R.SetInitialTransform(initial_transform, inPlace=False)
    
    # Execute and capture transform metadata
    final_transform = R.Execute(fixed_img, moving_img)
    metric_value = R.GetMetricValue()
    
    return final_transform, metric_value

def validate_calcium_alignment(raw_fixed_arr, transformed_vessel_arr, spacing, cfg=RegistrationCFG):
    """Calculates the % of calcium voxels falling within the transformed zone."""
    # Isolate calcium in the raw non-contrast CT volume (>130 HU)
    calcium_mask = (raw_fixed_arr >= 130) & (raw_fixed_arr <= 1000)
    total_calcium_voxels = int(calcium_mask.sum())
    
    if total_calcium_voxels == 0:
        return 0.0, 0
        
    # Dilate transformed vessel zones to create the specified tolerance channel
    dilated_vessels = ndi.binary_dilation(transformed_vessel_arr > 0, iterations=2)
    edt_vessels = ndi.distance_transform_edt(~dilated_vessels, sampling=spacing)
    
    # Calculate target localized hits
    voxels_inside_zone = int((calcium_mask & (edt_vessels <= cfg.DISTANCE_MM)).sum())
    hit_percentage = (voxels_inside_zone / total_calcium_voxels) * 100.0
    
    return hit_percentage, total_calcium_voxels

def main():
    RegistrationCFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load your pristine 0.375mm resampled database index file
    csv_path = RegistrationCFG.COCA_RESAMPLED_DIR / "resampled_index_0.375x0.375x3.000.csv"
    if not csv_path.exists():
        print(f"[ERROR] Clean resampled index not found at {csv_path}. Run resampler first.")
        return
        
    df_scans = pd.read_csv(csv_path)
    print(f"[READY] Processing {len(df_scans)} native physical scans for Project 3...")
    
    # Use Atlas Case 1 as the static template baseline
    atlas_img_path = RegistrationCFG.IMAGECAS_DIR / "1.img.nii.gz"
    atlas_lbl_path = RegistrationCFG.IMAGECAS_DIR / "1.label.nii.gz"
    
    moving_image = sitk.ReadImage(str(atlas_img_path), sitk.sitkFloat32)
    moving_label = sitk.ReadImage(str(atlas_lbl_path), sitk.sitkUInt8)
    
    results = []
    
    for idx, row in df_scans.iterrows():
        scan_id = row["scan_id"]
        # Essential change: Use the raw image path to retain pristine Hounsfield units
        raw_fixed_path = Path(row["raw_img_path"]) 
        
        print(f" -> [{idx+1}/{len(df_scans)}] Registering Scan ID: {scan_id}")
        t0 = time.time()
        
        try:
            fixed_image = sitk.ReadImage(str(raw_fixed_path), sitk.sitkFloat32)
            
            # Generate the crash-proof mask matching the exact physical matrix
            fixed_mask = build_fixed_heart_mask(fixed_image)
            
            # Run registration
            tx, metric = execute_registration_cascade(fixed_image, moving_image, fixed_mask)
            elapsed = time.time() - t0
            
            # Warp the atlas vessel tree into the patient's geometric space
            warped_label = sitk.Resample(
                moving_label, fixed_image, tx, 
                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
            )
            
            # Extract arrays for verification validation
            raw_fixed_arr = sitk.GetArrayFromImage(fixed_image)
            warped_lbl_arr = sitk.GetArrayFromImage(warped_label)
            spacing_zyx = fixed_image.GetSpacing()[::-1]
            
            hit_pct, total_ca = validate_calcium_alignment(raw_fixed_arr, warped_lbl_arr, spacing_zyx)
            passed = hit_pct >= RegistrationCFG.PASS_THRESHOLD_PCT
            
            # Save results dictionary
            results.append({
                "scan_id": scan_id, "mi_metric": metric, "execution_time_sec": elapsed,
                "calcium_hit_pct": hit_pct, "total_calcium_voxels": total_ca, "verdict": "PASS" if passed else "FAIL"
            })
            print(f"    Finished in {elapsed:.1f}s | Metric: {metric:.4f} | Calcium Hit: {hit_pct:.1f}% | {'[PASS]' if passed else '[FAIL]'}")
            
        except Exception as e:
            print(f"    [FAILED] Skipping scan {scan_id} due to error: {e}")
            
    # Save the project outputs to file
    out_df = pd.DataFrame(results)
    out_df.to_csv(RegistrationCFG.OUT_DIR / "project3_registration_metrics.csv", index=False)
    print(f"\n[SUCCESS] Registration complete. Final scorecard saved to: {RegistrationCFG.OUT_DIR}")

if __name__ == "__main__":
    main()