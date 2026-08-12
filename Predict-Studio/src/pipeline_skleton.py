"""
Standalone PrediCT Pipeline Skeleton
====================================

This is a self-contained script to test the model end-to-end. 
It does NOT depend on the `Archives` folder or any external YAML manifests.
All model parameters (HU window, spacing, architecture) are hardcoded here 
for rapid deployment and testing on the `approach1_roi_cropped` model.

Outputs:
  - pred.nii.gz
  - ct.nii.gz
  - lesions.csv (with per-lesion bounding boxes and Agatston scores)
  - run.json (summary metrics)
  - slices/ (PNG overlays)
"""
from __future__ import annotations
import sys
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image
import torch
import scipy.ndimage as ndimage
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

# ---------------------------------------------------------------------------
# HARDCODED MODEL CONFIGURATION
# ---------------------------------------------------------------------------
HU_WINDOW = (0, 1200)
SPACING_MM = (0.37, 0.37, 3.0)
MARGIN_MM = 8.0

# ---------------------------------------------------------------------------
# PIPELINE STEPS
# ---------------------------------------------------------------------------

def load(patient_folder: str | Path) -> sitk.Image:
    """Load DICOM series or NIfTI using SimpleITK."""
    patient_folder = Path(patient_folder)
    print(f"Loading {patient_folder}...")
    
    if patient_folder.is_file():
        return sitk.ReadImage(str(patient_folder))
        
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(patient_folder))
    
    if not dicom_names:
        niftis = list(patient_folder.rglob("*.nii*"))
        if niftis:
            return sitk.ReadImage(str(niftis[0]))
        raise ValueError(f"No DICOM or NIfTI found in {patient_folder}")
        
    reader.SetFileNames(dicom_names)
    return reader.Execute()

def resample(image: sitk.Image, target_spacing: tuple[float, float, float]) -> sitk.Image:
    """Resample image to target physical spacing (x, y, z) in mm."""
    print(f"Resampling to {target_spacing} mm...")
    # NOTE: The 3.0 mm z-resample ensures the slice thickness factor for Agatston is exactly 1.0.
    
    current_spacing = image.GetSpacing()
    current_size = image.GetSize()
    
    new_size = [
        int(round(current_size[i] * current_spacing[i] / target_spacing[i])) 
        for i in range(3)
    ]
    
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(target_spacing)
    rs.SetSize(new_size)
    rs.SetOutputDirection(image.GetDirection())
    rs.SetOutputOrigin(image.GetOrigin())
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(image).min()))
    
    return rs.Execute(image)

def crop_heart(image: sitk.Image, margin_mm: float) -> sitk.Image:
    """Use TotalSegmentator to locate the heart and crop the image."""
    print("Locating heart and cropping...")
    try:
        from totalsegmentator.python_api import totalsegmentator
        import nibabel as nib
        import tempfile
    except ImportError:
        print("TotalSegmentator or Nibabel not installed, skipping crop.")
        return image
        
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = Path(tmpdir) / "ct.nii.gz"
        tmp_out = Path(tmpdir) / "mask.nii.gz"
        
        # Write input for totalsegmentator
        sitk.WriteImage(image, str(tmp_in))
        
        # Run TS and save output to disk so we can read it back perfectly aligned with SITK
        result_nib = totalsegmentator(tmp_in, task="total", roi_subset=["heart"], fast=True, quiet=True)
        nib.save(result_nib, tmp_out)
        
        # Read back mask using SITK to ensure matching axis conventions (z, y, x)
        mask_img = sitk.ReadImage(str(tmp_out))
        mask_arr = sitk.GetArrayViewFromImage(mask_img) > 0
    
    zs, ys, xs = np.nonzero(mask_arr)
    if zs.size == 0:
        print("Heart not found, skipping crop.")
        return image
        
    spacing = image.GetSpacing() # x, y, z
    pad_x = int(np.ceil(margin_mm / spacing[0]))
    pad_y = int(np.ceil(margin_mm / spacing[1]))
    pad_z = int(np.ceil(margin_mm / spacing[2]))
    
    # SimpleITK indices are (x, y, z)
    x_start = max(0, int(xs.min()) - pad_x)
    x_end = min(image.GetSize()[0], int(xs.max()) + pad_x + 1)
    
    y_start = max(0, int(ys.min()) - pad_y)
    y_end = min(image.GetSize()[1], int(ys.max()) + pad_y + 1)
    
    z_start = max(0, int(zs.min()) - pad_z)
    z_end = min(image.GetSize()[2], int(zs.max()) + pad_z + 1)
    
    return image[x_start:x_end, y_start:y_end, z_start:z_end]

def normalize(array: np.ndarray, hu_window: tuple[float, float]) -> np.ndarray:
    """Map raw HU to [0, 1] range."""
    print(f"Normalizing HU {hu_window} -> [0, 1]...")
    lo, hi = float(hu_window[0]), float(hu_window[1])
    out = (array.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)

def predict(x: np.ndarray, weights_path: Path) -> np.ndarray:
    """Load PyTorch UNet and run sliding-window inference."""
    print(f"Running inference with weights: {weights_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hardcoded Architecture for A1/A3 models
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
    ).to(device)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    net.load_state_dict(state, strict=True)
    net.eval()

    tensor = torch.from_numpy(x.astype(np.float32))[None, None].to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            tensor,
            roi_size=(96, 96, 32),
            sw_batch_size=1,
            predictor=net,
            overlap=0.5,
        )
        # Activation for approach1 is sigmoid
        probs = torch.sigmoid(logits)

    return probs.squeeze().cpu().numpy().astype(np.float32)

def save_nifti(array: np.ndarray, reference_image: sitk.Image, out_path: Path):
    """Save array as NIfTI using spatial metadata from the reference image."""
    print(f"Saving NIfTI to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(reference_image.GetSpacing())
    image.SetOrigin(reference_image.GetOrigin())
    image.SetDirection(reference_image.GetDirection())
    
    sitk.WriteImage(image, str(out_path))

def score_and_save_csv(image: sitk.Image, prob: np.ndarray, out_csv: Path) -> dict:
    """Calculate Agatston score and save slice-by-slice lesion ledger."""
    print(f"Calculating Agatston score and saving to {out_csv}...")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    array = sitk.GetArrayViewFromImage(image)
    spacing = image.GetSpacing() # (x, y, z)
    pixel_area = spacing[0] * spacing[1]
    thickness_factor = spacing[2] / 3.0
    
    # Binarize threshold
    mask = prob > 0.5
    struct = ndimage.generate_binary_structure(2, 1) # 2D 4-connected
    
    total_agatston = 0.0
    calcium_vol = 0.0
    lesion_id = 1
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "slice_idx", "z_mm", "lesion_id",
            "area_mm2", "peak_hu", "density_weight", "agatston",
            "centroid_x", "centroid_y", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
            "included"
        ])
        
        for z in range(mask.shape[0]):
            plane = mask[z]
            if not plane.any():
                continue
                
            labelled, n = ndimage.label(plane, structure=struct)
            
            for lab in range(1, n + 1):
                ys, xs = np.nonzero(labelled == lab)
                n_vox = ys.size
                
                area_mm2 = n_vox * pixel_area
                peak_hu = float(array[z, ys, xs].max())
                
                # Agatston density factor
                if peak_hu < 130: factor = 0
                elif peak_hu < 200: factor = 1
                elif peak_hu < 300: factor = 2
                elif peak_hu < 400: factor = 3
                else: factor = 4
                
                score = area_mm2 * factor * thickness_factor
                
                included = True
                if area_mm2 < 1.0 or factor == 0:
                    included = False
                    
                if included:
                    total_agatston += score
                    calcium_vol += n_vox * spacing[0] * spacing[1] * spacing[2]
                
                centroid_y, centroid_x = np.mean(ys), np.mean(xs)
                
                writer.writerow([
                    z, round(z * spacing[2], 2), lesion_id,
                    round(area_mm2, 4), round(peak_hu, 2), factor, round(score, 4),
                    round(centroid_x, 2), round(centroid_y, 2),
                    xs.min(), ys.min(), xs.max(), ys.max(),
                    included
                ])
                lesion_id += 1

    return {"agatston_total": total_agatston, "calcium_volume_mm3": calcium_vol}

def save_slices(ct_array: np.ndarray, prob: np.ndarray, out_dir: Path):
    """Save PNG overlays using Matplotlib to match the visualizer reference."""
    print(f"Saving ALL PNG slices to {out_dir}...")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy.ma as ma
    
    for z in range(prob.shape[0]):
        # The visualizer clips from -100 to 400 for display
        ct_slice = np.clip(ct_array[z], -100, 400)
        p_slice = prob[z]
        
        # We only want to visualize slices that have some meaningful structure or prediction.
        # But for completeness, we can just save all slices as 1x2 grids.
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
        
        for ax in axes:
            ax.set_facecolor("black")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            
        # Panel 1: Original CT (origin='lower' aligns with radiology view)
        axes[0].imshow(ct_slice, cmap='gray', origin='lower', vmin=-100, vmax=400)
        
        # Panel 2: Soft Probability Overlay
        axes[1].imshow(ct_slice, cmap='gray', origin='lower', vmin=-100, vmax=400)
        
        # Mask out probabilities near zero so they are transparent, exactly like the visualizer
        p_masked = ma.masked_where(p_slice < 0.01, p_slice)
        axes[1].imshow(p_masked, cmap='inferno', origin='lower', alpha=0.5, vmin=0, vmax=1)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"slice_{z:03d}.png", dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

def run(patient_folder: str | Path, model_weights: str | Path, out_dir: str | Path, save_png: bool = True):
    """Execute the hardcoded testing pipeline."""
    patient_id = Path(patient_folder).name
    out_dir = Path(out_dir) / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    image = load(patient_folder)
    original_orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())
    
    print(f"Reorienting from {original_orientation} to RAS...")
    image = sitk.DICOMOrient(image, "RAS")
    image = resample(image, SPACING_MM)
    image = crop_heart(image, margin_mm=MARGIN_MM)
    
    array = sitk.GetArrayFromImage(image) # (Z, Y, X)
    
    # MONAI models expect spatial dimensions (X, Y, Z)
    # SimpleITK natively provides arrays as (Z, Y, X)
    x_monai = np.transpose(array, (2, 1, 0))
    x_monai = normalize(x_monai, HU_WINDOW)
    prob_xyz = predict(x_monai, Path(model_weights))
    
    # Transpose back to (Z, Y, X) for SimpleITK
    prob = np.transpose(prob_xyz, (2, 1, 0))
    
    # Restore the original orientation before saving
    print(f"Restoring original orientation ({original_orientation}) for outputs...")
    prob_img = sitk.GetImageFromArray(prob)
    prob_img.CopyInformation(image)
    
    image_orig = sitk.DICOMOrient(image, original_orientation)
    prob_img_orig = sitk.DICOMOrient(prob_img, original_orientation)
    
    array_orig = sitk.GetArrayFromImage(image_orig)
    prob_orig = sitk.GetArrayFromImage(prob_img_orig)
    
    save_nifti(array_orig, image_orig, out_dir / "ct.nii.gz")
    save_nifti(prob_orig, image_orig, out_dir / "pred.nii.gz")
    
    metrics = score_and_save_csv(image_orig, prob_orig, out_dir / "lesions.csv")
    
    with open(out_dir / "run.json", "w") as f:
        json.dump({
            "model_weights": str(model_weights),
            "date": datetime.now().isoformat(),
            **metrics
        }, f, indent=2)
    
    if save_png:
        save_slices(array_orig, prob_orig, out_dir / "slices")
        
    print(f"Pipeline complete! Output in {out_dir}")

if __name__ == "__main__":
    # Ensure correct weights path for hardcoded test script
    weights_path = Path(__file__).resolve().parent.parent / "models" / "approach1_roi_cropped" / "best_model.pth"
    out_path = Path(__file__).resolve().parent.parent / "out"
    
    run(
        "/pscratch/sd/s/soham95/SOHAM/coca_raw/cocacoronarycalciumandchestcts-2/deidentified_nongated/6/6", 
        model_weights=weights_path, 
        out_dir=out_path
    )
