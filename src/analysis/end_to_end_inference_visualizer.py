"""
end_to_end_inference_visualizer.py

This script performs end-to-end inference and visualization for the PrediCT Coronary 
Artery Calcium (CAC) Segmentation Pipeline. It takes a raw CT scan (via patient ID), 
runs it through two different 3D U-Net models (Approach 1: Binary, and Approach 3: Soft Coverage), 
and visualizes the results alongside the original radiologist XML annotations.

It automatically:
1. Loads the cropped ROI scan.
2. Applies standard MONAI transforms and runs sliding window inference for both models.
3. Inverts the spatial transforms to map predictions back to the original image coordinate space.
4. Parses the XML ground truth and scales it to match the ROI crop.
5. Calculates the exact Agatston scores for Ground Truth, A1, and A3.
6. Generates a 1x4 Matplotlib visualization of the slice with the largest calcium deposit.

Usage:
    python end_to_end_inference_visualizer.py --patient_id <ID>

Outputs:
    A high-resolution PNG image saved in the Agaston_Results/investigation directory.
"""

import sys
import argparse
import json
import plistlib
import pandas as pd
import torch
import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage.draw import polygon as sk_polygon

# Setup sys path for imports
sys.path.append(r"C:\SOHAM\src")
from model_tests.agatston_scoring_a1 import compute_xml_agatston, compute_model_agatston_a1
from model_tests.agatston_scoring_a3 import compute_model_agatston_a3
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, EnsureTyped
)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT_ROI = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\images_roi"
DATA_ROOT_ORIG = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\images"
XML_ROOT = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\Gated_release_final\calcium_xml"
CSV_PATH = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\tables\frozen_dataset_v1.parquet"
OUT_DIR = r"C:\SOHAM\runs\Agaston_Results\investigation\all__test_patients"

A1_WEIGHTS = r"C:\SOHAM\runs\Archives\approach1_binary\best_model.pth"
A3_WEIGHTS = r"C:\SOHAM\runs\approach3_coverage_v2\best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 32)
HU_MIN, HU_MAX = 0, 1200
SW_OVERLAP = 0.5


# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_xml_polygons(xml_path: Path):
    """
    Parses a calcium scoring XML file to extract the radiologist's drawn polygons.

    Args:
        xml_path (Path): Path to the patient's XML annotation file.

    Returns:
        dict: A dictionary mapping z-slice index (int) to a list of numpy arrays, 
              where each array contains the (x, y) coordinates of a polygon.
    """
    result = {}
    if not xml_path.exists(): return result
    with open(xml_path, "rb") as f:
        data = plistlib.load(f)
    for entry in data.get("Images", []):
        z = int(entry.get("ImageIndex", -1))
        if z < 0: continue
        polys = []
        for roi in entry.get("ROIs", []):
            pts = []
            for p in roi.get("Point_px", []):
                c = p.replace("(", "").replace(")", "").split(",")
                if len(c) == 2:
                    pts.append([float(c[0]), float(c[1])])
            if len(pts) >= 3:
                polys.append(np.array(pts, dtype=np.float32))
        if polys: result[z] = polys
    return result

def build_model():
    """
    Initializes and returns the 3D U-Net model architecture used in both approaches.
    
    Returns:
        monai.networks.nets.UNet: The instantiated PyTorch model on the active device.
    """
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, dropout=0.1,
    ).to(DEVICE)

def get_transforms():
    """
    Returns the MONAI Compose transform pipeline used for preprocessing the input NIfTI scans.
    This includes loading, channel adjustment, orientation (RAS), scaling intensity (HU), 
    and spatial padding.
    """
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=PATCH_SIZE),
        EnsureTyped(keys=["image"])
    ])


# ── Main Pipeline ────────────────────────────────────────────────────────────
def run_pipeline(patient_id):
    """
    Executes the full end-to-end inference and visualization pipeline for a given patient.

    Args:
        patient_id (int or str): The ID of the patient to process.

    Process:
        1. Identifies the scan_id from the dataset splits.
        2. Computes the ground truth XML Agatston score using the original DICOMs.
        3. Loads the A1 and A3 PyTorch models and runs inference.
        4. Calculates the predicted Model Agatston scores.
        5. Aligns the XML ground truth polygons to the cropped ROI space.
        6. Finds the z-slice with the largest calcium area and plots the 1x4 comparison.
    """
    # Find scan_id
    df = pd.read_parquet(CSV_PATH)
    row = df[df["patient_id"].astype(str) == str(patient_id)]
    if len(row) == 0:
        print(f"Patient {patient_id} not found in {CSV_PATH}")
        return
    scan_id = str(row.iloc[0]["scan_id"])
    print(f"[*] Running End-to-End Pipeline for Patient {patient_id} (Scan: {scan_id})")

    # Load images and metadata
    img_path = Path(DATA_ROOT_ROI) / scan_id / f"{scan_id}_img.nii.gz"
    roi_meta_path = Path(DATA_ROOT_ROI) / scan_id / f"{scan_id}_meta.json"
    orig_meta_path = Path(DATA_ROOT_ORIG) / scan_id / f"{scan_id}_meta.json"
    xml_path = Path(XML_ROOT) / f"{patient_id}.xml"

    if not img_path.exists():
        print(f"[!] ROI image not found: {img_path}")
        return

    with open(roi_meta_path, "r") as f:
        roi_meta = json.load(f)
    with open(orig_meta_path, "r") as f:
        orig_meta = json.load(f)

    # 1. Calculate Ground Truth Agatston
    dicom_dir = orig_meta.get("original_path")
    print("[*] Calculating Ground Truth Agatston...")
    xml_agatston = compute_xml_agatston(patient_id, dicom_dir, xml_path)
    print(f"    -> True Agatston: {xml_agatston:.1f}")

    # 2. Model Inference
    transforms = get_transforms()
    batch = transforms({"image": str(img_path)})
    vimg = batch["image"].unsqueeze(0).to(DEVICE)
    
    # Load A1
    model_a1 = build_model()
    model_a1.load_state_dict(torch.load(A1_WEIGHTS, map_location=DEVICE, weights_only=True))
    model_a1.eval()
    
    # Load A3
    model_a3 = build_model()
    model_a3.load_state_dict(torch.load(A3_WEIGHTS, map_location=DEVICE, weights_only=True))
    model_a3.eval()

    print("[*] Running Dual Model Inference...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            # A1 Inference
            out_a1 = sliding_window_inference(vimg, PATCH_SIZE, sw_batch_size=4, predictor=model_a1, overlap=SW_OVERLAP)
            
            # A3 Inference
            out_a3 = sliding_window_inference(vimg, PATCH_SIZE, sw_batch_size=4, predictor=model_a3, overlap=SW_OVERLAP)
            
    batch["pred_a1"] = torch.sigmoid(out_a1[0]).cpu()
    batch["pred_a3"] = torch.sigmoid(out_a3[0]).cpu()

    from monai.transforms import Invertd
    inverter = Invertd(
        keys=["pred_a1", "pred_a3"],
        transform=transforms,
        orig_keys="image",
        meta_keys=["pred_a1_meta_dict", "pred_a3_meta_dict"],
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True
    )
    # Ensure lists are correct for multiple keys
    inverter.orig_keys = ["image", "image"]
    inverter.orig_meta_keys = ["image_meta_dict", "image_meta_dict"]

    inverted = inverter(batch)
    
    pred_a1_inverted = inverted["pred_a1"][0].numpy().transpose(2, 1, 0)
    pred_a1_bin = (pred_a1_inverted > 0.5).astype(np.uint8)
    
    pred_a3_probs = inverted["pred_a3"][0].numpy().transpose(2, 1, 0)

    # Get ROI image array and spacing for scoring
    img_sitk = sitk.ReadImage(str(img_path))
    img_array = sitk.GetArrayFromImage(img_sitk)
    spacing = img_sitk.GetSpacing()

    # Calculate Model Agatston
    print("[*] Calculating Model Agatston Scores...")
    a1_agatston = compute_model_agatston_a1(pred_a1_bin, img_array, spacing)
    a3_agatston = compute_model_agatston_a3(pred_a3_probs, img_array, spacing)
    print(f"    -> A1 Binary Agatston: {a1_agatston:.1f}")
    print(f"    -> A3 Soft Agatston:   {a3_agatston:.1f}")

    # 3. Ground Truth Reconstruction for Visualization
    xml_polys = parse_xml_polygons(xml_path)
    
    orig_sp = orig_meta["original_spacing"]
    tgt_sp = orig_meta["resampled_spacing"]
    sx = orig_sp[0] / tgt_sp[0]
    sy = orig_sp[1] / tgt_sp[1]
    
    crop_offset = roi_meta.get("crop_index_xyz", [0, 0, 0])
    
    # Find slice with maximum true calcium area in the ROI
    max_area = 0
    best_z_roi = -1
    best_scaled_polys = []
    
    Z_roi_len = img_array.shape[0]
    H, W = img_array.shape[1], img_array.shape[2]

    # Map XML to ROI coordinates
    roi_polys = {}
    for z_orig, polys in xml_polys.items():
        z_roi = z_orig - crop_offset[2]
        if 0 <= z_roi < Z_roi_len:
            scaled = []
            area = 0
            for poly in polys:
                sp = poly.copy()
                # Scale from dicom to canonical, then apply crop offset
                sp[:, 0] = (poly[:, 0] * sx) - crop_offset[0]
                sp[:, 1] = (poly[:, 1] * sy) - crop_offset[1]
                scaled.append(sp)
                
                # Shoelace area for finding largest lesion
                x, y = sp[:, 0], sp[:, 1]
                area += 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                
            roi_polys[z_roi] = scaled
            if area > max_area:
                max_area = area
                best_z_roi = z_roi
                best_scaled_polys = scaled

    if best_z_roi == -1:
        print("[!] No calcium found inside the cropped ROI for this patient.")
        return

    print(f"[*] Visualizing Slice z={best_z_roi} (Max Area: {max_area:.1f} px^2)")

    ct_slice = np.clip(img_array[best_z_roi], -100, 400)
    a1_slice = pred_a1_bin[best_z_roi]
    a3_slice = pred_a3_probs[best_z_roi]
    
    true_mask = np.zeros((H, W), dtype=np.uint8)
    for sp in best_scaled_polys:
        rr, cc = sk_polygon(sp[:, 1].clip(0, H - 1), sp[:, 0].clip(0, W - 1))
        true_mask[rr, cc] = 1

    # Calculate Slice-specific Agatston Scores
    a1_agatston_slice = compute_model_agatston_a1(np.array([a1_slice]), np.array([img_array[best_z_roi]]), spacing)
    a3_agatston_slice = compute_model_agatston_a3(np.array([a3_slice]), np.array([img_array[best_z_roi]]), spacing)
    true_agatston_slice = compute_model_agatston_a1(np.array([true_mask]), np.array([img_array[best_z_roi]]), spacing)

    # 4. Generate Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle(f"Patient {patient_id} | Total Agatston - True: {xml_agatston:.1f} | A1: {a1_agatston:.1f} | A3: {a3_agatston:.1f}\n"
                 f"Slice z={best_z_roi} Agatston - True: {true_agatston_slice:.1f} | A1: {a1_agatston_slice:.1f} | A3: {a3_agatston_slice:.1f}", 
                 fontsize=16, fontweight='bold')

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values(): spine.set_edgecolor("#333333")

    # Panel 1: CT
    axes[0].imshow(ct_slice, cmap='gray', origin='lower')
    axes[0].set_title("CT Slice", fontsize=12)

    # Panel 2: XML GT
    axes[1].imshow(ct_slice, cmap='gray', origin='lower')
    xml_rgba = np.zeros((*true_mask.shape, 4))
    xml_rgba[true_mask == 1] = [1.0, 0.0, 0.0, 0.5] # Red
    axes[1].imshow(xml_rgba, origin='lower')
    for sp in best_scaled_polys:
        cl = np.vstack([sp, sp[0]])
        axes[1].plot(cl[:, 0], cl[:, 1], color="lime", lw=1.5, alpha=0.95)
    axes[1].set_title("XML Ground Truth", fontsize=12)

    # Panel 3: A1 Binary
    axes[2].imshow(ct_slice, cmap='gray', origin='lower')
    a1_rgba = np.zeros((*a1_slice.shape, 4))
    a1_rgba[a1_slice == 1] = [0.0, 1.0, 0.0, 0.5] # Green
    axes[2].imshow(a1_rgba, origin='lower')
    axes[2].set_title("Approach 1 (Binary)", fontsize=12)

    # Panel 4: A3 Soft
    axes[3].imshow(ct_slice, cmap='gray', origin='lower')
    # Mask out probabilities near zero so they are completely transparent instead of plotting as black
    import numpy.ma as ma
    a3_masked = ma.masked_where(a3_slice < 0.01, a3_slice)
    axes[3].imshow(a3_masked, cmap='inferno', origin='lower', alpha=0.5, vmin=0, vmax=1)
    axes[3].set_title("Approach 3 (Soft Probabilities)", fontsize=12)

    # Add Zoom / Bounding Box limit
    all_pts = np.vstack(best_scaled_polys)
    pad = 30
    x1, x2 = max(0, int(all_pts[:, 0].min()) - pad), min(W, int(all_pts[:, 0].max()) + pad)
    y1, y2 = max(0, int(all_pts[:, 1].min()) - pad), min(H, int(all_pts[:, 1].max()) + pad)
    
    for ax in axes:
        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)

    plt.tight_layout()
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_file = Path(OUT_DIR) / f"Patient_{patient_id}_A1_vs_A3_visualizer.png"
    plt.savefig(out_file, dpi=200)
    print(f"[*] Saved Visualization to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Inference Visualizer")
    parser.add_argument("--patient_id", type=int, default=None, help="Target Patient ID to visualize")
    parser.add_argument("--all_test", action="store_true", help="Run on all test patients")
    args = parser.parse_args()
    
    # List of all 66 test patients
    patient_list = [294, 140, 427, 412, 46, 306, 24, 431, 260, 256, 320, 172, 307, 205, 355, 
                    214, 145, 183, 170, 202, 299, 117, 31, 259, 82, 148, 258, 396, 89, 34, 
                    354, 65, 49, 252, 53, 69, 371, 52, 382, 75, 5, 179, 246, 196, 18, 386, 
                    387, 173, 17, 224, 122, 29, 271, 343, 42, 91, 51, 344, 403, 147, 4, 60, 
                    80, 119, 152, 48]

    if args.all_test:
        for pid in patient_list:
            run_pipeline(pid)
            plt.close('all') # Prevent matplotlib from eating all your RAM
    else:
        pid = args.patient_id if args.patient_id is not None else 205
        run_pipeline(pid)
        plt.close('all')
