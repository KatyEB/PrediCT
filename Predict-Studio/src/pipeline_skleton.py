"""
PrediCT Pipeline Skeleton
=========================

This script serves as the complete, end-to-end inference and scoring pipeline 
for the PrediCT coronary artery calcium (CAC) scoring project. 

It reads raw CT scans (DICOM/NIfTI), processes them to standard clinical 
specifications, runs PyTorch model inference, and calculates the Agatston score.

Pipeline Architecture:
----------------------
The pipeline is designed as a sequence of stateless, decoupled functions. 
Data flows straight down the `run()` function:
  1. `load()`        : Ingests a patient's CT folder, returning a 3D Volume object.
  2. `resample()`    : Enforces exactly 3.0mm Z-spacing to conform to Agatston rules.
  3. `crop_heart()`  : Locates the heart via TotalSegmentator and crops the volume.
  4. `normalize()`   : Maps raw Hounsfield Units (HU) to a [0, 1] range using the 
                       model's specifically declared window.
  5. `predict()`     : Runs sliding-window inference via a PyTorch UNet model.
  6. Output stage    : Saves clinical source-of-truth NIfTIs, calculates scores, 
                       and generates a CSV and visualization slices.

Output Artifacts:
-----------------
The script generates a self-describing folder for each run containing:
  - `ct.nii.gz`     : The cropped, resampled CT volume.
  - `pred.nii.gz`   : The model's raw probability mask prediction.
  - `lesions.csv`   : A flat, rectangular ledger with one row per identified lesion, 
                      containing bounding boxes, peak HU, and sub-scores. Lesions 
                      that fail the clinical >1mm^2 gate are marked `included: False`.
  - `run.json`      : Provenance data (checkpoint sha256, model ID, total Agatston).
  - `slices/`       : Native-resolution PNGs overlaying probabilities onto the CT.

Usage:
------
    run(
        patient_folder="path/to/patient/0",
        model_manifest="models/approach1_roi_cropped/config.json",
        out_dir="results/patient_0"
    )
"""
import sys
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from PIL import Image

# Ensure we can import from the existing Archives backend
backend_dir = Path(__file__).resolve().parent.parent / "Archives" / "backend"
sys.path.append(str(backend_dir))

from load import load_folder, Volume
from preprocess import resample as backend_resample
from preprocess import locate_heart, bounding_box, crop
from models import load_model, window as backend_window, Model
from scoring import score_volume, Spacing, ScoringConfig, find_components, binarise

def load(patient_folder: str | Path) -> Volume:
    """Load the first CT volume found in the given folder."""
    print(f"Loading {patient_folder}...")
    volumes = load_folder(patient_folder)
    if not volumes:
        raise ValueError(f"No volume found in {patient_folder}")
    print(f"Found {len(volumes)} series. Using {volumes[0].series_id}")
    return volumes[0]

def resample(ct: Volume, target_spacing: tuple[float, float, float]) -> Volume:
    """Resample volume to target spacing (x, y, z) in mm."""
    print(f"Resampling to {target_spacing}...")
    # NOTE: The 3.0 mm z-resample is doing real work here. 
    # Classic Agatston is defined on 3 mm slices. 
    # Because we resample to exactly 3.0 mm, the slice thickness factor is 1.0. 
    # If we ever change that spacing, every score silently scales by thickness/3.0.
    return backend_resample(ct, target_spacing)

def crop_heart(ct: Volume, margin_mm: float = 8.0) -> Volume:
    """Locate heart using TotalSegmentator and crop to its bounding box."""
    print("Locating heart and cropping...")
    heart_mask = locate_heart(ct)
    box = bounding_box(heart_mask, ct.spacing, margin_mm)
    return crop(ct, box)

def normalize(ct: Volume, hu_window: tuple[float, float]) -> np.ndarray:
    """Map raw HU to [0, 1] based on the provided window."""
    print(f"Normalizing HU {hu_window} -> [0, 1]...")
    return backend_window(ct.array, hu_window)

def predict(x: np.ndarray, model_info: Model) -> np.ndarray:
    """Load PyTorch model and run inference on normalized input."""
    import torch
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNet
    
    print(f"Running inference with {model_info.id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = model_info.architecture
    net = UNet(
        spatial_dims=3,
        in_channels=arch.get("in_channels", 1),
        out_channels=arch.get("out_channels", 1),
        channels=tuple(arch.get("channels", (16, 32, 64, 128, 256))),
        strides=tuple(arch.get("strides", (2, 2, 2, 2))),
        num_res_units=arch.get("num_res_units", 2),
        dropout=arch.get("dropout", 0.1),
    ).to(device)

    state = torch.load(model_info.weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    net.load_state_dict(state, strict=True)
    net.eval()

    tensor = torch.from_numpy(x.astype(np.float32))[None, None].to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            tensor,
            roi_size=tuple(model_info.inference.get("patch_size", (96, 96, 32))),
            sw_batch_size=model_info.inference.get("sw_batch_size", 1),
            predictor=net,
            overlap=model_info.inference.get("sw_overlap", 0.5),
        )
        
        activation = model_info.inference.get("activation", "sigmoid")
        if activation == "sigmoid":
            probs = torch.sigmoid(logits)
        else:
            probs = logits

    return probs.squeeze().cpu().numpy().astype(np.float32)

def save_nifti(array: np.ndarray, ct: Volume, out_path: str | Path):
    """Save array as a NIfTI file, matching the original CT's spatial metadata."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    image = sitk.GetImageFromArray(array)
    image.SetSpacing([float(s) for s in ct.spacing])
    
    # Copy origin and direction if present on the original Volume 
    # (requires support in Volume, but fallback safely if missing)
    if hasattr(ct, "origin"):
        image.SetOrigin(ct.origin)
    if hasattr(ct, "direction"):
        image.SetDirection(ct.direction)
        
    sitk.WriteImage(image, str(out_path))

def save_slices(ct: Volume, prob: np.ndarray, out_dir: str | Path, window: tuple[float, float]):
    """Save all slices as PNGs at native resolution."""
    print(f"Saving ALL PNG slices to {out_dir}...")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-calculate normalized CT for visualization (0-255 grayscale)
    ct_norm = np.clip((ct.array - window[0]) / (window[1] - window[0]), 0, 1)
    ct_gray = (ct_norm * 255).astype(np.uint8)
    
    z_slices = prob.shape[0]
    for z in range(z_slices):
        ct_slice = ct_gray[z]
        ct_rgb = np.stack([ct_slice, ct_slice, ct_slice], axis=-1)
        
        # Use the probability as the alpha channel over a red hue
        mask_slice = prob[z]
        
        r = (ct_slice * (1 - mask_slice) + 255 * mask_slice).astype(np.uint8)
        g = (ct_slice * (1 - mask_slice)).astype(np.uint8)
        b = (ct_slice * (1 - mask_slice)).astype(np.uint8)
        
        overlay = np.stack([r, g, b], axis=-1)
        
        # Combine side-by-side
        combined = np.concatenate([ct_rgb, overlay], axis=1)
        Image.fromarray(combined).save(out_dir / f"slice_{z:03d}.png")

def score_and_save_csv(ct: Volume, prob: np.ndarray, model_info: Model, out_csv: str | Path):
    """Calculate Agatston score and save lesion-by-lesion results to CSV."""
    print(f"Calculating Agatston score and saving to {out_csv}...")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    mask_kind = "soft" if model_info.output_type == "soft" else "binary"
    # Force 2D for Agatston per-slice definition
    config = ScoringConfig(lesion_definition="2d")
    
    # Score the volume using the raw CT array
    score = score_volume(
        mask=prob,
        hu=ct.array,
        spacing=Spacing.from_sitk(ct.spacing),
        config=config,
        mask_kind=mask_kind
    )

    # We need bounding box and centroid for each lesion. 
    # Recreate the components to calculate bbox and centroid.
    membership = binarise(prob, mask_kind, config.binary_threshold, config.soft_threshold)
    components = find_components(membership, config.lesion_definition, config.connectivity)
    
    comp_map = {c.label: c for c in components}

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "scan_id", "slice_idx", "z_mm", "lesion_id",
            "area_mm2", "peak_hu", "density_weight", "agatston",
            "centroid_x", "centroid_y", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
            "model_id", "included"
        ])
        
        for lesion in score.lesions: # Note: Iterating all lesions including excluded ones
            comp = comp_map.get(lesion.lesion_id)
            if not comp:
                continue
                
            centroid_y = np.mean(comp.ys)
            centroid_x = np.mean(comp.xs)
            bbox_y0, bbox_y1 = comp.ys.min(), comp.ys.max()
            bbox_x0, bbox_x1 = comp.xs.min(), comp.xs.max()
            
            z_mm = lesion.slice_index * ct.spacing[2] if lesion.slice_index is not None else 0.0
            slice_idx = lesion.slice_index if lesion.slice_index is not None else -1

            writer.writerow([
                ct.patient_id,
                slice_idx,
                round(z_mm, 2),
                lesion.lesion_id,
                round(lesion.area_mm2, 4),
                round(lesion.peak_hu, 2),
                lesion.density_factor,
                round(lesion.score, 4),
                round(centroid_x, 2),
                round(centroid_y, 2),
                bbox_x0, bbox_y0, bbox_x1, bbox_y1,
                model_info.id,
                lesion.included
            ])
            
    return score

def write_run_json(out_dir: Path, ct: Volume, model_info: Model, score, hu_window: tuple):
    print("Writing run.json...")
    data = {
        "scan_id": ct.patient_id,
        "model_id": model_info.id,
        "checkpoint": str(model_info.weights_path),
        "sha256": model_info.sha256,
        "hu_window": hu_window,
        "spacing": ct.spacing,
        "roi": model_info.requires.get("roi"),
        "date": datetime.now().isoformat(),
        "agatston_total": score.agatston,
        "calcium_volume_mm3": score.calcium_volume_mm3,
        "risk_category": score.risk_category
    }
    with open(out_dir / "run.json", "w") as f:
        json.dump(data, f, indent=2)

def run(patient_folder: str | Path, model_manifest: str | Path, out_dir: str | Path, save_png: bool = True):
    """Main pipeline execution."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model info once
    model_info = load_model(model_manifest)
    hu_window = tuple(model_info.requires.get("hu_window", [0, 1200]))
    
    ct = load(patient_folder)
    ct = resample(ct, (0.37, 0.37, 3.0))
    ct = crop_heart(ct, margin_mm=8)
    
    x = normalize(ct, hu_window)
    prob = predict(x, model_info)
    
    print("Saving CT and Prediction NIfTIs...")
    save_nifti(ct.array, ct, out_dir / "ct.nii.gz")
    save_nifti(prob, ct, out_dir / "pred.nii.gz")
    
    score = score_and_save_csv(ct, prob, model_info, out_dir / "lesions.csv")
    write_run_json(out_dir, ct, model_info, score, hu_window)
    
    if save_png:
        save_slices(ct, prob, out_dir / "slices", hu_window)
        
    print(f"Pipeline complete! Output in {out_dir}")

if __name__ == "__main__":
    # Example usage:
    # run("path/to/patient/folder", "models/approach1_roi_cropped/config.json", "out/dir")
    pass
