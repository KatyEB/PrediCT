import pandas as pd
import torch
import SimpleITK as sitk
import numpy as np
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

# Configuration
TEST_CSV = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\tables\test_split.parquet"
DATA_ROOT_FULL = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\images"
DATA_ROOT_ROI = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\images_roi"

MODELS = {
    "A1_Full_Volume": {
        "weights": r"C:\SOHAM\runs\approach1_binary\best_model.pth",
        "data_root": DATA_ROOT_FULL,
        "is_roi": False
    },
    "A1_ROI_Cropped": {
        "weights": r"C:\SOHAM\runs\approach1_roi_cropped\best_model.pth",
        "data_root": DATA_ROOT_ROI,
        "is_roi": True
    }
}

HU_MIN, HU_MAX = 0, 1200
PATCH_SIZE = (96, 96, 32)
SW_OVERLAP = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
        EnsureTyped(keys=["image", "label"])
    ])

def build_model():
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, dropout=0.1,
    ).to(DEVICE)

def evaluate():
    df = pd.read_parquet(TEST_CSV)
    ids = df['scan_id'].astype(str).tolist()
    
    results = []
    
    for model_name, cfg in MODELS.items():
        print(f"\nEvaluating {model_name}...")
        
        # Build datalist
        items = []
        for sid in ids:
            img = Path(cfg["data_root"]) / sid / f"{sid}_img.nii.gz"
            lbl = Path(cfg["data_root"]) / sid / f"{sid}_seg.nii.gz"
            if img.exists() and lbl.exists():
                items.append({"image": str(img), "label": str(lbl), "scan_id": sid})
                
        ds = Dataset(items, get_transforms())
        loader = DataLoader(ds, batch_size=1, num_workers=0)
        
        # Load Model
        model = build_model()
        model.load_state_dict(torch.load(cfg["weights"], map_location=DEVICE, weights_only=True))
        model.eval()
        
        for batch in tqdm(loader):
            sid = batch["scan_id"][0]
            vimg = batch["image"].to(DEVICE)
            vlbl = batch["label"].to(DEVICE)
            
            with torch.no_grad():
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                    vout = sliding_window_inference(
                        vimg, PATCH_SIZE, sw_batch_size=4,
                        predictor=model, overlap=SW_OVERLAP
                    )
            
            # Post process
            pred = torch.sigmoid(vout[0, 0]).cpu().numpy()
            true = vlbl[0, 0].cpu().numpy()
            
            # Binarize
            pred_bin = (pred > 0.5).astype(np.float32)
            true_bin = (true > 0.5).astype(np.float32)
            
            # Voxel volume (from original image metadata)
            lbl_path = Path(cfg["data_root"]) / sid / f"{sid}_seg.nii.gz"
            orig_lbl = sitk.ReadImage(str(lbl_path))
            spacing = orig_lbl.GetSpacing()
            voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
            
            # Calculate metrics
            true_vol = np.sum(true_bin) * voxel_vol_mm3
            pred_vol = np.sum(pred_bin) * voxel_vol_mm3
            
            # Dice
            intersection = np.sum(pred_bin * true_bin)
            union = np.sum(pred_bin) + np.sum(true_bin)
            dice = (2.0 * intersection) / union if union > 0 else (1.0 if np.sum(true_bin)==0 else 0.0)
            
            results.append({
                "Model": model_name,
                "Scan_ID": sid,
                "True_Vol_mm3": true_vol,
                "Pred_Vol_mm3": pred_vol,
                "Error_Vol_mm3": pred_vol - true_vol,
                "AbsError_Vol_mm3": abs(pred_vol - true_vol),
                "Dice": dice
            })

    # Save Results
    res_df = pd.DataFrame(results)
    out_dir = Path(__file__).resolve().parents[2] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "test_evaluation_results.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Generate Visuals
    sns.set_theme(style="darkgrid")
    
    # 1. Volume Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=res_df, x="True_Vol_mm3", y="Pred_Vol_mm3", hue="Model", style="Model", s=60, alpha=0.8)
    
    # Plot Identity Line (Perfect Prediction)
    max_val = max(res_df["True_Vol_mm3"].max(), res_df["Pred_Vol_mm3"].max())
    plt.plot([0, max_val], [0, max_val], 'k--', label='Perfect Prediction (y=x)', alpha=0.5)
    
    plt.title("Predicted vs True Calcium Volume (Test Set)")
    plt.xlabel("True Volume (mm³)")
    plt.ylabel("Predicted Volume (mm³)")
    plt.legend()
    plt.savefig(out_dir / "volume_scatter_comparison.png", dpi=150)
    plt.close()
    
    # 2. Bland-Altman Plot
    plt.figure(figsize=(12, 6))
    for model_name in MODELS.keys():
        subset = res_df[res_df["Model"] == model_name]
        mean_vol = (subset["True_Vol_mm3"] + subset["Pred_Vol_mm3"]) / 2
        diff_vol = subset["Pred_Vol_mm3"] - subset["True_Vol_mm3"]
        sns.scatterplot(x=mean_vol, y=diff_vol, label=model_name, s=60, alpha=0.7)
        plt.axhline(diff_vol.mean(), linestyle='--', alpha=0.5) # Bias line
        
    plt.axhline(0, color='k', linestyle='-', alpha=0.5)
    plt.title("Bland-Altman Plot: Volume Difference vs Mean Volume")
    plt.xlabel("Mean Volume (mm³)")
    plt.ylabel("Difference (Predicted - True) (mm³)")
    plt.legend()
    plt.savefig(out_dir / "bland_altman_comparison.png", dpi=150)
    plt.close()

    # 3. Bar Plot for MAE
    plt.figure(figsize=(8, 6))
    sns.barplot(data=res_df, x="Model", y="AbsError_Vol_mm3", estimator=np.mean, errorbar=('ci', 95))
    plt.title("Mean Absolute Volume Error (MAE)")
    plt.ylabel("Volume MAE (mm³)")
    plt.savefig(out_dir / "mae_comparison_bar.png", dpi=150)
    plt.close()
    
    print(f"Visuals saved to {out_dir}")
    
    # Summary Table
    summary = res_df.groupby("Model").agg({
        "Dice": "mean",
        "AbsError_Vol_mm3": "mean",
        "Error_Vol_mm3": "mean"
    }).rename(columns={"Error_Vol_mm3": "Bias_Vol_mm3"}).round(3)
    
    print("\n--- Summary Results ---")
    print(summary)

if __name__ == "__main__":
    evaluate()
