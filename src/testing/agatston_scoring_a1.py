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
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from monai.data import DataLoader, Dataset
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, EnsureTyped
)

# ---------------------------------------------------------------------------
# CONFIGURATION - APPROACH 1 (BINARY / ROI CROPPED)
# ---------------------------------------------------------------------------
TEST_CSV = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\tables\test_split.parquet"
DATA_ROOT_ROI = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\images_roi"
XML_ROOT = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\Gated_release_final\calcium_xml"

MODEL_WEIGHTS = r"C:\SOHAM\runs\approach1_roi_cropped\best_model.pth"
HU_MIN, HU_MAX = 100, 1000
PATCH_SIZE = (96, 96, 32)
SW_OVERLAP = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_agatston_factor(max_hu):
    if max_hu < 130: return 0
    if max_hu < 200: return 1
    if max_hu < 300: return 2
    if max_hu < 400: return 3
    return 4

def shoelace_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def compute_xml_agatston(patient_id, dicom_dir, xml_path):
    """Computes exact XML Agatston score using Shoelace area and original DICOM max HU."""
    if not Path(xml_path).exists():
        return 0.0

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    spacing = image.GetSpacing()
    img_array = sitk.GetArrayFromImage(image)

    total_agatston = 0.0
    with open(xml_path, "rb") as f:
        data = plistlib.load(f)

    for img_entry in data.get("Images", []):
        z = int(img_entry.get("ImageIndex", -1))
        if z < 0 or z >= img_array.shape[0]: continue

        for roi in img_entry.get("ROIs", []):
            points_str = roi.get("Point_px", [])
            if not points_str: continue
            
            poly_points = []
            for p_str in points_str:
                cleaned = p_str.replace("(", "").replace(")", "")
                parts = cleaned.split(",")
                if len(parts) == 2:
                    poly_points.append([float(parts[0]), float(parts[1])])
            
            if poly_points:
                pts = np.array(poly_points)
                area_px = shoelace_area(pts[:, 0], pts[:, 1])
                area_mm2 = area_px * spacing[0] * spacing[1]

                temp_slice = np.zeros(img_array.shape[1:], dtype=np.uint8)
                pts_int = pts.astype(np.int32)
                cv2.fillPoly(temp_slice, [pts_int], 1)

                hu_pixels = img_array[z][temp_slice == 1]
                if len(hu_pixels) > 0:
                    max_hu = np.max(hu_pixels)
                    factor = get_agatston_factor(max_hu)
                    total_agatston += area_mm2 * factor

    return total_agatston

def compute_model_agatston_a1(pred_mask, img_array, spacing):
    """Computes Model Agatston using 2D slice-by-slice binary connected components."""
    total_agatston = 0.0
    for z in range(pred_mask.shape[0]):
        labeled_slice, num_features = scipy.ndimage.label(pred_mask[z])
        for i in range(1, num_features + 1):
            lesion_mask = (labeled_slice == i)
            hu_pixels = img_array[z][lesion_mask]
            if len(hu_pixels) > 0:
                max_hu = np.max(hu_pixels)
                factor = get_agatston_factor(max_hu)
                area_px = np.sum(lesion_mask)
                area_mm2 = area_px * spacing[0] * spacing[1]
                total_agatston += area_mm2 * factor
    return total_agatston

def get_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=PATCH_SIZE),
        EnsureTyped(keys=["image"])
    ])

def build_model():
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, dropout=0.1,
    ).to(DEVICE)

def evaluate_agatston():
    df = pd.read_parquet(TEST_CSV)
    
    results = []
    print("\n--- Evaluating Agatston Scores for Approach 1 (Binary) ---")
    
    # Load Model
    model = build_model()
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE, weights_only=True))
    model.eval()
    transforms = get_transforms()

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sid = str(row['scan_id'])
        pid = str(row['patient_id'])
        
        img_path = Path(DATA_ROOT_ROI) / sid / f"{sid}_img.nii.gz"
        meta_path = Path(DATA_ROOT_ROI) / sid / f"{sid}_meta.json"
        xml_path = Path(XML_ROOT) / f"{pid}.xml"
        
        if not img_path.exists() or not meta_path.exists():
            continue
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
        dicom_dir = meta.get("original_path")
        
        # 1. Compute XML Ground Truth Agatston
        xml_agatston = compute_xml_agatston(pid, dicom_dir, xml_path)
        
        # 2. Compute Model Prediction Agatston
        item = {"image": str(img_path)}
        batch = transforms(item)
        vimg = batch["image"].unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                vout = sliding_window_inference(
                    vimg, PATCH_SIZE, sw_batch_size=4,
                    predictor=model, overlap=SW_OVERLAP
                )
        
        pred = torch.sigmoid(vout[0, 0]).cpu().numpy()
        pred_bin = (pred > 0.5).astype(np.float32)
        
        orig_img_sitk = sitk.ReadImage(str(img_path))
        orig_img_array = sitk.GetArrayFromImage(orig_img_sitk)
        spacing = orig_img_sitk.GetSpacing()
        
        model_agatston = compute_model_agatston_a1(pred_bin, orig_img_array, spacing)
        
        results.append({
            "Scan_ID": sid,
            "Patient_ID": pid,
            "XML_Agatston": xml_agatston,
            "Model_Agatston": model_agatston,
            "Error": model_agatston - xml_agatston,
            "AbsError": abs(model_agatston - xml_agatston)
        })

    # Save and Plot
    res_df = pd.DataFrame(results)
    out_dir = Path(__file__).resolve().parents[2] / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "agatston_comparison_a1.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    sns.set_theme(style="darkgrid")
    
    # Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=res_df, x="XML_Agatston", y="Model_Agatston", s=60, alpha=0.8)
    max_val = max(res_df["XML_Agatston"].max(), res_df["Model_Agatston"].max())
    plt.plot([0, max_val], [0, max_val], 'k--', label='Perfect Prediction (y=x)', alpha=0.5)
    plt.title("Approach 1 (Binary): Predicted vs True Agatston Score")
    plt.xlabel("True Agatston Score (XML Shoelace)")
    plt.ylabel("Predicted Agatston Score (Model)")
    plt.legend()
    plt.savefig(out_dir / "agatston_scatter_a1.png", dpi=150)
    plt.close()
    
    # Summary
    print("\n--- Approach 1 Summary Results ---")
    print(f"Mean XML Agatston:   {res_df['XML_Agatston'].mean():.2f}")
    print(f"Mean Model Agatston: {res_df['Model_Agatston'].mean():.2f}")
    print(f"Mean Absolute Error: {res_df['AbsError'].mean():.2f}")
    print(f"Mean Bias (Error):   {res_df['Error'].mean():.2f}")

if __name__ == "__main__":
    evaluate_agatston()
