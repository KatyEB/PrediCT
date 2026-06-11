"""
PrediCT GSoC 2026 Preprocessing Pipeline

Reads dataset.csv from data_resampled folder and produces splits.json with stratified train/val/test splits, and dataset_stats.json with key statistics about the dataset.

Reads  : DATASET_CSV (dataset_resampled.csv) → must be generated   
Writes : splits.json, dataset_stats.json

Later used by Dataset and DataLoader for training and evaluation.

Run    : python preprocessing.py --max-file 200 (optional max_files for quick debugging)
"""

import sys
import json
import numpy as np
import pandas as pd  
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent))
import config


sys.path.append(str(Path(__file__).parent))
import config

SPLITS_JSON = config.preprocessing_config["SPLITS_JSON"] #path to save the splits json file
STATS_JSON = config.preprocessing_config["STATS_JSON"] #path to save the dataset statistics json file
BATCH_SIZE  = config.dataloader_config["BATCH_SIZE"] #batch size for data loader
NUM_WORKERS = config.dataloader_config["NUM_WORKERS"]
VAL_SIZE    = config.preprocessing_config["VAL_SIZE"]
TEST_SIZE   = config.preprocessing_config["TEST_SIZE"]
SEED        = config.preprocessing_config["RANDOM_SEED"]
TASK = config.preprocessing_config["TASK"]  # "binary" or "multi"
DATASET_CSV   = config.preprocessing_config["DATASET_CSV"] #provdies address for dataset path
HEART_MODEL_PATH = Path(config.dataloader_config["HEART_MODEL_PATH"]) # Path for model



# ==================================================
# Loading Model and Generating ROI Masks
# ==================================================

import torch as torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from LW_UNET_TVERSKY.lw_model import LightweightUNet3D as Heart_Seg_Model
from config import preprocessing_config
import sys

from monai.inferers import sliding_window_inference

from scipy.ndimage import label, sum as nd_sum, binary_closing, binary_fill_holes
import numpy as np
import SimpleITK as sitk

# ══════════════════════════════════════════════════════════════════
#  TRANSFORMS
# ══════════════════════════════════════════════════════════════════

# ── Patch size ────────────────────────────────────────────────────
# (96,128,96)  → safe for 8-12GB VRAM,  batch_size=2
# (112,160,128)→ nnU-Net native,         needs 24GB VRAM, batch_size=2
# Rule of thumb: if OOM → halve one dim, not all three
ROI_SIZE = (96, 128, 96)

# ── HU window (cardiac soft tissue, clinically standard) ──────────
# [-150, 350]: covers myocardium (50-80), blood pool (30-45),
#              pericardial fat (-30 to -100), vessel walls (~200)
# NOT [-500, 1300]: that's a thoracic window — too wide, adds
#              lung/bone noise that confuses cardiac U-Net
HU_MIN = -150.0
HU_MAX =  350.0

SPACING = (1.0, 1.0, 1.0)

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    EnsureTyped,
    ToTensord,
    Invertd,    
)

def get_transforms(mode: str) -> Compose:
    """
    Full transform pipeline for heart segmentation.
    REUSED HERE AGAIN FOR LOADING THE MODEL FOR ROI MASKS

    Pipeline order matters — always:
      Load → Channel → Orient(RAS) → Space(1mm) → HU window
      → CropForeground → [augment if train] → Tensor

    Orientationd MUST come before Spacingd.
    CropForeground MUST come after HU window (needs non-zero content).
    Augmentation MUST come after CropForeground (smaller volume = faster).
    """
    assert mode in ("train", "val", "test")

    # ── Base pipeline (all modes) ─────────────────────────────────
    base = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),

        # Step 1: Orient to RAS — MUST be before Spacingd
        # Without this, left/right flips are anatomically inconsistent
        # because COCA scans may have different orientations
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # Step 2: Resample to 1mm isotropic
        # Already 1mm in COCA but explicit ensures consistency
        # mode="nearest" for label to avoid interpolation artifacts
        Spacingd(
            keys=["image", "label"],
            pixdim=SPACING,
            mode=("bilinear", "nearest"),
        ),

        # Step 3: HU windowing → [0, 1]
        # Cardiac soft tissue window: -150 to 350 HU
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_MIN,
            a_max=HU_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        # Step 4: Crop tight around heart (non-zero image region)
        # margin=15: generous enough to never clip the heart boundary
        # source_key="image": crop based on non-air content
        # This reduces volume ~40% before patch sampling → faster
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            margin=15,
        ),

        #To ensure that we get our desired patch size or img volume in train/val and test respectievly.
        SpatialPadd(
        keys=["image", "label"],
        spatial_size=ROI_SIZE
        ),

        EnsureTyped(keys=["image", "label"]),
    ]

    # ── Train augmentations ───────────────────────────────────────
    if mode == "train":
        aug = [

            # ── Patch sampling ────────────────────────────────────
            # pos=3, neg=1 → 75% of patches contain heart voxels
            # Your data: medium + large hearts → heart occupies
            # a significant fraction of volume, pos=3 is appropriate
            # num_samples=2 → 34 scans × 2 = 68 patches/epoch
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=ROI_SIZE,
                pos=3,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0.0,
            ),

            # ── Geometric — applied jointly to image + label ──────

            # Axis flips: safe for cardiac CT (no handedness constraint)
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

            # 90° rotations: handles scanner orientation differences
            RandRotate90d(
                keys=["image", "label"],
                prob=0.3,
                max_k=3,
                spatial_axes=(0, 2),
            ),

            # Small-angle rotation ±15°: patient positioning variation
            # Reduced from nnU-Net's ±30° — heart is constrained in chest
            # padding_mode="zeros" → safe, fills with background HU
            RandRotated(
                keys=["image", "label"],
                range_x=0.26,
                range_y=0.26,
                range_z=0.26,
                prob=0.4,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),

            # Zoom 0.85-1.15: patient body size + FOV variation
            RandZoomd(
                keys=["image", "label"],
                min_zoom=0.85,
                max_zoom=1.15,
                prob=0.3,
                mode=("trilinear", "nearest"),
            ),

            # ── Intensity — image only ────────────────────────────

            # Gaussian noise: scanner electronic noise
            # std=0.02 ≈ 7 HU equivalent — realistic for CT
            RandGaussianNoised(
                keys=["image"],
                prob=0.3,
                mean=0.0,
                std=0.02,
            ),

            # Gaussian smooth: PSF and reconstruction kernel variation
            # COCA has multiple scanner protocols — this simulates that
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.2,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),

            # Contrast (gamma): tube voltage differences between scans
            RandAdjustContrastd(
                keys=["image"],
                prob=0.3,
                gamma=(0.75, 1.4),
            ),

            # Brightness shift: scanner calibration drift
            RandShiftIntensityd(
                keys=["image"],
                prob=0.3,
                offsets=0.1,
            ),

            ToTensord(keys=["image", "label"]),
        ]
        return Compose(base + aug)

    # Val/Test: no augmentation, full volume (needed for accurate Dice)
    # return Compose(base + [ToTensord(keys=["image", "label"])])
    return Compose(base)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

heart_model = Heart_Seg_Model()  

checkpoint = torch.load(HEART_MODEL_PATH, map_location=device)

heart_model.load_state_dict(
    checkpoint["model"])


heart_model.to(device)
heart_model.eval()

def postprocess_heart_mask(pred_np: np.ndarray) -> np.ndarray:
    """
    Post-process binary heart segmentation mask.
    
    Steps:
        1. Largest Connected Component — removes diaphragm surface / stray voxels
        2. Morphological Closing       — fills thin gaps at valves / myocardial walls
        3. Binary Fill Holes           — fills any remaining internal holes
    
    Args:
        pred_np: binary numpy array (z, y, x), dtype uint8
    
    Returns:
        cleaned binary numpy array, same shape, dtype uint8
    """

    # ── 1. Largest Connected Component ────────────────────────────
    labeled, n = label(pred_np)
    
    if n == 0:
        print("Warning: empty mask, no components found")
        return pred_np
    
    if n > 1:
        sizes = nd_sum(pred_np, labeled, range(1, n + 1))
        largest_label = np.argmax(sizes) + 1
        pred_np = (labeled == largest_label).astype(np.uint8)
        print(f"LCC: kept 1 of {n} components")
    
    # ── 2. Morphological Closing ───────────────────────────────────
    # iterations=2 ~ 2 voxel closing, enough for valve/wall gaps
    pred_np = binary_closing(pred_np, iterations=2).astype(np.uint8)

    # ── 3. Fill Holes ──────────────────────────────────────────────
    pred_np = binary_fill_holes(pred_np).astype(np.uint8)

    return pred_np

def generate_roi_masks(image_paths, model, margin=5):
    roi_paths = []

    for image_path in image_paths:

        image = sitk.ReadImage(image_path)

        test_transforms = get_transforms("test")
        data = test_transforms(
            {"image": image_path, "label": image_path}
        )

        x = data["image"].unsqueeze(0).float().to(next(model.parameters()).device)

        # ── inference ─────────────────────────────────────────────
        with torch.no_grad():
            pred = sliding_window_inference(
                inputs=x,
                roi_size=ROI_SIZE,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
            )
            pred = (torch.sigmoid(pred) > 0.5).float()

        # ── attach transform history from image to pred ────────────
        # data["image"] is MetaTensor with full applied_operations history
        # We clone it and replace values with our prediction
        data["pred"] = data["image"].clone()  # preserves ALL metadata + history
        data["pred"].data = pred[0].cpu()     # swap in prediction values only

        # ── invert ────────────────────────────────────────────────
        post_transform = Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            nearest_interp=True,
            to_tensor=True,
        )

        data = post_transform(data)

        heart_mask = data["pred"][1].numpy().astype(np.uint8)  # channel 1 = heart

        # ── post-process ───────────────────────────────────────────
        heart_mask = postprocess_heart_mask(heart_mask)

        # print(f"Image size: {image.GetSize()}, Heart mask size: {heart_mask.shape}")
        heart_mask_image = sitk.GetImageFromArray(heart_mask.T)
        heart_mask_image.CopyInformation(image)     
        # print(f"Image size: {image.GetSize()}, Heart mask size: {heart_mask_image.GetSize()}")


        # ── bbox ──────────────────────────────────────────────────
        coords = np.argwhere(heart_mask.T > 0)

        if len(coords) == 0:
            print(f"Warning: no heart detected for {image_path}")
            roi_paths.append(None)
            continue

        xmin, ymin, zmin,  = coords.min(0)
        xmax, ymax, zmax   = coords.max(0)

        zmin = max(0, zmin - margin)
        ymin = max(0, ymin - margin)
        xmin = max(0, xmin - margin)
        zmax = min(heart_mask.shape[0] - 1, zmax + margin)
        ymax = min(heart_mask.shape[1] - 1, ymax + margin)
        xmax = min(heart_mask.shape[2] - 1, xmax + margin)

        roi_mask = np.zeros_like(heart_mask, dtype=np.uint8)
        roi_mask[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = 1

        # ── save ──────────────────────────────────────────────────
        roi_img = sitk.GetImageFromArray(roi_mask.T) 
        roi_img.CopyInformation(image)

        image_id = image_path.split("/")[-1].replace("_img.nii.gz", "")

        roi_path = Path(image_path).parent / f"{image_id}_roi_mask.nii.gz"
        sitk.WriteImage(roi_img, str(roi_path))
        roi_paths.append(str(roi_path))

    return roi_paths



# END OF LOADING MODELS ------------------

# ══════════════════════════════════════════════════════════════════
#  LOAD + VALIDATE + PREPROCESS
# ══════════════════════════════════════════════════════════════════

def load_clean_and_pre_process(dataset_resampled_csv, config, max_files=None):

    df = pd.read_csv(dataset_resampled_csv)

    print(
        f"\n📋 Loaded dataset_resampled.csv "
        f"— {len(df)} scans"
    )

    print(
        f"   Columns: {list(df.columns)}"
    )

    task = config.preprocessing_config["TASK"]

    # --------------------------------------------------
    # choose label column based on task
    # --------------------------------------------------

    if task == "binary":

        image_col = "image_path"
        label_col = "binary_mask_path"

    elif task == "multi":

        image_col = "image_path"
        label_col = "multi_mask_path"

    else:

        raise ValueError(
            f"Unknown task '{task}'"
        )

    # --------------------------------------------------
    # normalize paths
    # --------------------------------------------------

    path_cols = [

        "image_path",

        "binary_mask_path",

        "multi_mask_path",

        "resampled_image_path",

        "resampled_binary_seg_path",

        "resampled_multi_label_seg_path",
    ]

    for col in path_cols:

        if col in df.columns:

            df[col] = df[col].astype(str)

            df[col] = df[col].apply(
                lambda p: str(Path(p))
            )

    # --------------------------------------------------
    # validation
    # --------------------------------------------------

    missing_img = ~df[image_col].apply(
        lambda p: Path(p).exists()
    )

    missing_label = ~df[label_col].apply(
        lambda p: Path(p).exists()
    )

    must_drop = (
        missing_img |
        missing_label
    )

    if must_drop.any():

        print(
            f"\n❌ {must_drop.sum()} scans "
            f"missing image or label"
        )

        for _, row in df[must_drop].iterrows():

            reasons = []

            if not Path(row[image_col]).exists():
                reasons.append("image missing")

            if not Path(row[label_col]).exists():
                reasons.append("label missing")

            print(
                f"   {row['scan_id']} : "
                f"{', '.join(reasons)}"
            )

    df = (
        df[~must_drop]
        .reset_index(drop=True)
    )

    # ── Cap AFTER validation so stratify has enough of both classes ──
    if max_files is not None:
        # Check we still have both classes before hard-slicing
        available_pos = (df["agatston_total"] > 0).sum()
        available_neg = (df["agatston_total"] == 0).sum()

        # Proportionally sample to preserve class balance
        pos_ratio = available_pos / len(df)
        n_pos = max(1, int(max_files * pos_ratio))
        n_neg = max(1, max_files - n_pos)

        df_pos = df[df["agatston_total"] > 0].head(n_pos)
        df_neg = df[df["agatston_total"] == 0].head(n_neg)
        df = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)

        print(f"\n⚠️  max_files={max_files} → using {len(df)} scans "
              f"(CAC+: {n_pos}, CAC-: {n_neg})")

    print(f"\n✅ Valid scans : {len(df)}")


    # --------------------------------------------------
    # local editable dataframe
    # --------------------------------------------------

    dataset_df = pd.DataFrame({

        "scan_id":
            df["scan_id"],

        "image":
            df[image_col], #image path

        "label":
            df[label_col], # labels_column

        "agatston_total":
            df["agatston_total"],

        "agatston_available":
            (
                df["agatston_total"] > 0
            ).astype(int),

        # generated later
        "roi_mask":
            generate_roi_masks(image_paths=df[image_col].tolist(), model=heart_model) # returns the path of generated roi masks, which will be used in the monai dataset to load these masks and use them for cropping the heart region during training and inference if the flag is on.
    })

    return dataset_df


# ══════════════════════════════════════════════════════════════════
#  STRATIFIED HOLDOUT SPLIT (70 / 15 / 15)
# ══════════════════════════════════════════════════════════════════

from sklearn.model_selection import train_test_split
import json


def create_splits(dataset_df):
    """
    Creates MONAI-compatible train/val/test splits.

    Stratification:
        agatston_available
            0 -> no calcium
            1 -> calcium present

    Split:
        Train : 70%
        Val   : 15%
        Test  : 15%
    """

    df = dataset_df.copy()

    temp_ratio = VAL_SIZE + TEST_SIZE

    stratify_col = df["agatston_available"]

    # --------------------------------------------------
    # Train / Temp
    # --------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=stratify_col,
        random_state=SEED,
    )

    # --------------------------------------------------
    # Val / Test
    # --------------------------------------------------

    try:

        val_df, test_df = train_test_split(
            temp_df,
            test_size=TEST_SIZE / temp_ratio,
            stratify=temp_df["agatston_available"],
            random_state=SEED,
        )

        print("\nVal/Test split: ✅ stratified")

    except ValueError as e:

        print(f"\nVal/Test split: ⚠️ {e}")
        print("Falling back to random split")

        val_df, test_df = train_test_split(
            temp_df,
            test_size=TEST_SIZE / temp_ratio,
            stratify=None,
            random_state=SEED,
        )

    # --------------------------------------------------
    # Convert DataFrame -> MONAI dictionaries
    # --------------------------------------------------

    def build_monai_dicts(split_df):

        return [
            {
                "id": str(row.scan_id),

                "image": row.image,

                "label": row.label,

                "roi_mask": row.roi_mask,

                "agatston_total": float(
                    row.agatston_total
                ),

                "agatston_available": int(
                    row.agatston_available
                ),
            }
            for row in split_df.itertuples()
        ]

    splits = {
        "train": build_monai_dicts(train_df),
        "val": build_monai_dicts(val_df),
        "test": build_monai_dicts(test_df),
    }

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    with open(SPLITS_JSON, "w") as f:
        json.dump(splits, f, indent=2)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print(f"\n✅ Saved splits → {SPLITS_JSON}")

    print(f"Train : {len(train_df)}")
    print(f"Val   : {len(val_df)}")
    print(f"Test  : {len(test_df)}")

    print("\nAgatston balance:")

    for name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:

        counts = (
            split_df["agatston_available"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(f"{name:5s}: {counts}")

    return splits


# ══════════════════════════════════════════════════════════════════
#  DATASET STATISTICS
# ══════════════════════════════════════════════════════════════════

def compute_stats(df: pd.DataFrame, splits: dict) -> dict:
    """
    Compute and save dataset_stats.json to the STATS_JSON folder.

    Covers:
      1. Dataset-level   : total scans, CAC+/- counts & percentages
      2. Split info      : train/val/test sizes + agatston balance per split
      3. Image-level     : spacing, shape, HU min/mean/max/std  (≤20 sampled scans)
      4. HU window       : parameters + clinical justification
      5. Class imbalance : positive/negative ratio + mitigation note
    """

    print("\n📊 Computing dataset statistics...")

    out_path = Path(STATS_JSON)

    # ── 1. Dataset-level ──────────────────────────────────────────
    total = len(df)
    pos   = int(df["agatston_available"].sum())   # CAC > 0
    neg   = total - pos

    dataset_level = {
        "total_scans"          : total,
        "cac_positive"         : pos,
        "cac_negative"         : neg,
        "cac_positive_pct"     : round(100 * pos / total, 2) if total else 0,
        "cac_negative_pct"     : round(100 * neg / total, 2) if total else 0,
        "agatston_total_mean"  : round(float(df["agatston_total"].mean()), 3),
        "agatston_total_median": round(float(df["agatston_total"].median()), 3),
        "agatston_total_max"   : round(float(df["agatston_total"].max()), 3),
        "agatston_total_std"   : round(float(df["agatston_total"].std()), 3),
    }

    # ── 2. Split info ─────────────────────────────────────────────
    split_info = {}
    for split_name, split_list in splits.items():
        split_df = pd.DataFrame(split_list)
        n        = len(split_df)
        s_pos    = int(split_df["agatston_available"].sum())
        split_info[split_name] = {
            "n"              : n,
            "cac_positive"   : s_pos,
            "cac_negative"   : n - s_pos,
            "pos_pct"        : round(100 * s_pos / n, 2) if n else 0,
        }

    # ── 3. Image-level stats (sampled ≤20 scans) ──────────────────
    sample_paths = df["image"].dropna().tolist()
    sample_paths = sample_paths[:20]   # cap at 20 for speed

    spacings, shapes, hu_mins, hu_means, hu_maxs, hu_stds = [], [], [], [], [], []

    for p in sample_paths:
        try:
            img_sitk  = sitk.ReadImage(str(p))
            spacing   = img_sitk.GetSpacing()          # (x, y, z)
            size      = img_sitk.GetSize()             # (x, y, z)
            arr       = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # (z,y,x)

            spacings.append(list(spacing))
            shapes.append(list(size))
            hu_mins.append(float(arr.min()))
            hu_means.append(float(arr.mean()))
            hu_maxs.append(float(arr.max()))
            hu_stds.append(float(arr.std()))

        except Exception as e:
            print(f"   ⚠️  Skipping {p}: {e}")

    def _agg(lst):
        """Return mean/min/max of a list of scalars."""
        if not lst:
            return {}
        a = np.array(lst, dtype=float)
        return {
            "mean": round(float(a.mean()), 4),
            "min" : round(float(a.min()),  4),
            "max" : round(float(a.max()),  4),
        }

    avg_spacing = _agg([np.mean(s) for s in spacings])
    avg_shape   = _agg([np.mean(s) for s in shapes])

    image_level = {
        "sampled_n"    : len(hu_mins),
        "spacing_mm"   : avg_spacing,
        "volume_voxels": avg_shape,
        "hu_min"       : _agg(hu_mins),
        "hu_mean"      : _agg(hu_means),
        "hu_max"       : _agg(hu_maxs),
        "hu_std"       : _agg(hu_stds),
    }

    # ── 4. HU window ──────────────────────────────────────────────
    hu_window = {
        "hu_min"        : HU_MIN,
        "hu_max"        : HU_MAX,
        "justification" : (
            "Cardiac soft-tissue window [-150, 350] HU. "
            "Covers myocardium (50-80), blood pool (30-45), "
            "pericardial fat (-30 to -100), vessel walls (~200). "
            "Excludes lung/bone noise from a thoracic window [-500, 1300]."
        ),
    }

    # ── 5. Class imbalance ────────────────────────────────────────
    ratio = round(neg / pos, 2) if pos else None
    imbalance = {
        "neg_to_pos_ratio": ratio,
        "mitigation": (
            "Stratified train/val/test splits preserve CAC+/- ratio. "
            "Consider pos_weight in BCE loss or oversampling CAC+ during training."
        ),
    }

    # ── Assemble & save ───────────────────────────────────────────
    stats = {
        "dataset"        : dataset_level,
        "splits"         : split_info,
        "image_level"    : image_level,
        "hu_window"      : hu_window,
        "class_imbalance": imbalance,
    }

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Saved stats → {out_path}")
    return stats


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PrediCT GSoC — Preprocessing Pipeline")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap the number of VALID scans processed (after path validation). "
             "Useful for quick pipeline smoke-tests. e.g. --max-files 100",
    )
    args = parser.parse_args()

    print("═" * 55)
    print(" PrediCT GSoC 26 — Preprocessing Pipeline")
    if args.max_files:
        print(f"  ⚠️  DEBUG MODE — capped at {args.max_files} valid scans")
    print("═" * 55 + "\n")

    # ── Step 1: Load, validate, generate ROI masks ────────────────
    df = load_clean_and_pre_process(
        dataset_resampled_csv=DATASET_CSV,
        config=config,
        max_files=args.max_files,     # <── passed in here
    )

    # ── Step 2: Stratified splits ─────────────────────────────────
    splits = create_splits(df)

    # ── Step 3: Dataset statistics → Insights/ ───────────────────
    stats = compute_stats(df, splits)

    print(f"\n{'═'*55}")
    print(f"  ✅ Done")
    print(f"{'═'*55}")