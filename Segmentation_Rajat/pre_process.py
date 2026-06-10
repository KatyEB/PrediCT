# common_task/preprocessing.py
"""
PrediCT GSoC 2026 Preprocessing Pipeline

Reads dataset.csv from data_resampled folder and produces splits.json with stratified train/val/test splits, and dataset_stats.json with key statistics about the dataset.

Reads  : dataset.csv    
Writes : splits.json, dataset_stats.json

Run    : python preprocessing.py
"""

import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib   
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent))
import config

TRAIN_CSV   = config.preprocessing_config["TRAIN_CSV"]
SPLITS_JSON = config.preprocessing_config["SPLITS_JSON"]
STATS_JSON  = config.preprocessing_config["STATS_JSON"]
VAL_SIZE    = config.preprocessing_config["VAL_SIZE"]
TEST_SIZE   = config.preprocessing_config["TEST_SIZE"]
SEED        = config.preprocessing_config["RANDOM_SEED"]

# ==================================================
# Loading Model and Generating ROI Masks
# ==================================================

import torch
import SimpleITK as sitk
from LW_UNET_TVERSKY.lw_model import Heart_Seg_Model

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

heart_model = Heart_Seg_Model()  
HEART_MODEL_PATH = Path(r"E:\MyProjects\Gsoc_2026_Official\LW_UNET_TVERSKY\best_model.pth")

heart_model.load_state_dict(
    torch.load(
        HEART_MODEL_PATH,
        map_location=device
    )
)

heart_model.to(device)
heart_model.eval()

def generate_roi_masks(
    image_paths,
    model,
    margin=10,
):

    roi_paths = []

    for image_path in image_paths:

        image = sitk.ReadImage(image_path)

        image_np = sitk.GetArrayFromImage(
            image
        ).astype(np.float32)

        # -------------------------
        # preprocessing
        # -------------------------

        image_np = (
            image_np - image_np.mean()
        ) / (
            image_np.std() + 1e-8
        )

        x = torch.tensor(
            image_np,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        x = x.to(next(model.parameters()).device)

        # -------------------------
        # inference
        # -------------------------

        with torch.no_grad():

            pred = model(x)

            pred = torch.sigmoid(pred)

            pred = (
                pred > 0.5
            ).float()

        heart_mask = (
            pred.squeeze()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # -------------------------
        # bbox
        # -------------------------

        coords = np.argwhere(
            heart_mask > 0
        )

        zmin, ymin, xmin = coords.min(0)
        zmax, ymax, xmax = coords.max(0)

        zmin = max(0, zmin - margin)
        ymin = max(0, ymin - margin)
        xmin = max(0, xmin - margin)

        zmax = min(
            heart_mask.shape[0]-1,
            zmax + margin
        )

        ymax = min(
            heart_mask.shape[1]-1,
            ymax + margin
        )

        xmax = min(
            heart_mask.shape[2]-1,
            xmax + margin
        )

        roi_mask = np.zeros_like(
            heart_mask,
            dtype=np.uint8
        )

        roi_mask[
            zmin:zmax+1,
            ymin:ymax+1,
            xmin:xmax+1
        ] = 1

        # -------------------------
        # save
        # -------------------------

        roi_img = sitk.GetImageFromArray(
            roi_mask
        )

        roi_img.CopyInformation(image)

        roi_path = (
            Path(image_path)
            .parent
            / "roi_mask.nii.gz"
        )

        sitk.WriteImage(
            roi_img,
            str(roi_path)
        )

        roi_paths.append(
            str(roi_path)
        )

    return roi_paths

# ══════════════════════════════════════════════════════════════════
#  LOAD + VALIDATE + PREPROCESS
# ══════════════════════════════════════════════════════════════════

def load_clean_and_pre_process(dataset_resampled_csv, config):

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

        image_col = "resampled_image_path"
        label_col = "resampled_binary_seg_path"

    elif task == "multi":

        image_col = "resampled_image_path"
        label_col = "resampled_multi_label_seg_path"

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

    print(
        f"\n✅ Valid scans : "
        f"{len(df)}"
    )

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

        "agatston_positive":
            (
                df["agatston_total"] > 0
            ).astype(int),

        # generated later
        "roi_mask":
            generate_roi_masks(image_paths=df[image_col], model=heart_model) # returns the path of generated roi masks, which will be used in the monai dataset to load these masks and use them for cropping the heart region during training and inference if the flag is on.
    })

    return dataset_df





# OLD -> 

# ══════════════════════════════════════════════════════════════════
#  LOAD + VALIDATE + PRE PROCESS
# ══════════════════════════════════════════════════════════════════

def load_clean_and_pre_process(dataset_csv):    
    df = pd.read_csv(dataset_csv)
    print(f"📋 Loaded dataset.csv — {len(df)} patients")

    print(f"   Columns: {list(df.columns)}")

    # Normalize paths — handles Windows backslashes
    for col in ["image", "CAC_seg", "heart_mask"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda p: str(Path(p)))

    # ── Missing files — these we MUST skip (can't train on nothing)

    # image missing  → no input → can't train
    # heart_mask missing → no label → can't train
    # CAC_seg missing → not needed for training → just warn
    missing_img  = ~df["image"].apply(lambda p: Path(p).exists())
    missing_mask = ~df["heart_mask"].apply(lambda p: Path(p).exists())
    missing_cac  = ~df["CAC_seg"].apply(lambda p: Path(p).exists()) \
                   if "CAC_seg" in df.columns else pd.Series([False]*len(df))
    

    # Only drop if image OR heart_mask is missing
    # These are truly unrecoverable
    must_drop = missing_img | missing_mask

    if must_drop.any():
        print(f"\n❌ {must_drop.sum()} patients missing image or "
              f"heart_mask — cannot train without these:")
        for _, row in df[must_drop].iterrows():
            reasons = []
            if not Path(row["image"]).exists():
                reasons.append("image missing")
            if not Path(row["heart_mask"]).exists():
                reasons.append("heart_mask missing")
            print(f"   {row['id']}: {', '.join(reasons)}")
        print(f"   → Check paths in train.csv")

    valid = df[~must_drop].reset_index(drop=True)

    print(f"\n✅ Usable patients: {len(valid)}/{len(df)}")
    return valid


# ══════════════════════════════════════════════════════════════════
#  STRATIFIED SPLIT  70 / 15 / 15
# ══════════════════════════════════════════════════════════════════

def create_splits(df):
    """
    Stratified 70/15/15 hold-out split by heart size category.

    Why not K-Fold?
      Only 30-50 TotalSeg-labeled scans available.
      K-fold test sets of ~5 scans give unstable Dice estimates
      (one wrong prediction swings Dice by 0.2+).
      5x training cost not justified at this scale.
      Stratified hold-out is standard for small medical imaging datasets.

    Robustness strategy:
      1. Merge rare categories (< 4 samples) into dominant class
         → preserves all patients, no data loss
      2. try/except on val/test split
         → graceful fallback to random if temp set still too small
    """
    df        = df.copy()
    df["cat"] = df["heart_vol_ml"].apply(get_heart_size_category)

    # Step 1: merge rare categories before any splitting
    df   = merge_rare_categories(df, min_count=4)
    cats = df["cat"].tolist()

    # Step 2: build MONAI-compatible dicts
    # "image" + "label" keys are required by MONAI LoadImaged
    data = []
    for _, row in df.iterrows():
        data.append({
            "image":        row["image"],        # CT volume → U-Net input
            "label":        row["heart_mask"],   # TotalSeg GT → training target
            "CAC_seg":      row["CAC_seg"],       # calcium mask → radiomics later
            "id":           str(row["id"]),
            "heart_vol_ml": float(row["heart_vol_ml"]),
            "size_cat":     int(row["cat"]),
        })

    temp_r = VAL_SIZE + TEST_SIZE

    # Split 1 — 70% train, 30% temp
    # Always stratified (full dataset, enough samples per class after merge)
    train_d, temp_d, train_c, temp_c = train_test_split(
        data, cats,
        test_size=temp_r,
        stratify=cats,
        random_state=SEED,
    )

    # Split 2 — temp → 50% val, 50% test
    # temp is small (~15 patients) so stratification may still fail
    # → try stratified first, fall back to random if needed
    try:
        val_d, test_d = train_test_split(
            temp_d, temp_c,
            test_size=TEST_SIZE / temp_r,
            stratify=temp_c,
            random_state=SEED,
        )
        print("\n   Val/Test split: ✅ stratified")

    except ValueError as e:
        print(f"\n   Val/Test split: ⚠️  stratify failed ({e})")
        print(f"   Falling back to random split — valid for evaluation")
        val_d, test_d = train_test_split(
            temp_d,
            test_size=TEST_SIZE / temp_r,
            stratify=None,
            random_state=SEED,
        )

    splits = {"train": train_d, "val": val_d, "test": test_d}

    with open(SPLITS_JSON, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\n✅ splits.json → {SPLITS_JSON}")
    print(f"   Train : {len(train_d)}")
    print(f"   Val   : {len(val_d)}")
    print(f"   Test  : {len(test_d)}")
    print(f"\n   Size category balance (0=Small 1=Med 2=Large):")
    for name, split in splits.items():
        cats_in = [d["size_cat"] for d in split]
        counts  = {c: cats_in.count(c) for c in [0, 1, 2]}
        print(f"   {name:5s}: {counts}")

    return splits


# ══════════════════════════════════════════════════════════════════
#  DATASET STATISTICS  ← deliverable
# ══════════════════════════════════════════════════════════════════

def compute_stats(df, splits):
    """
    Compute dataset_stats.json.

    Includes:
      1. Dataset-level:  patient counts, CAC+/- distribution
      2. Image-level:    spacing, shape, HU statistics (sampled)
      3. Split info:     train/val/test sizes, strategy, rationale
      4. HU window:      parameters + justification
      5. Class imbalance: issue + mitigation strategy
      6. DICOM metadata: scanner info preserved from scan_index.csv
    """
    print(f"\n📊 Computing dataset statistics...")

    # ── Image-level stats (sample 10 to save time) ────────────────
    spacings, shapes, hu_means, hu_stds = [], [], [], []
    hu_min_vals, hu_max_vals            = [], []

    sample = df.sample(min(10, len(df)), random_state=42)
    for _, row in sample.iterrows():
        try:
            img  = nib.load(row["image"])
            arr  = img.get_fdata()
            zooms = list(img.header.get_zooms()[:3])
            spacings.append(zooms)
            shapes.append(list(arr.shape))
            hu_means.append(float(np.mean(arr)))
            hu_stds.append(float(np.std(arr)))
            hu_min_vals.append(float(np.min(arr)))
            hu_max_vals.append(float(np.max(arr)))
        except Exception as e:
            print(f"   ⚠️  Could not load {row['image']}: {e}")

    # ── CAC presence ──────────────────────────────────────────────
    cac_flags = []
    for _, row in df.iterrows():
        try:
            seg = nib.load(row["CAC_seg"]).get_fdata()
            cac_flags.append(int(np.sum(seg > 0) > 0))
        except Exception:
            cac_flags.append(0)

    n_pos = sum(cac_flags)
    n_neg = len(cac_flags) - n_pos
    sp    = np.array(spacings) if spacings else np.zeros((0, 3))
    sh    = np.array(shapes)   if shapes   else np.zeros((0, 3))

    # ── Heart volume distribution ─────────────────────────────────
    vol_cats = df["heart_vol_ml"].apply(get_heart_size_category)

    stats = {

        # ── 1. Dataset level ──────────────────────────────────────
        "dataset": {
            "total_patients":   len(df),
            "cac_positive":     n_pos,
            "cac_negative":     n_neg,
            "cac_positive_pct": round(100 * n_pos / max(len(df), 1), 1),
            "heart_size_dist": {
                "small_lt550ml":  int((vol_cats == 0).sum()),
                "medium_550_800": int((vol_cats == 1).sum()),
                "large_gt800ml":  int((vol_cats == 2).sum()),
            },
        },

        # ── 2. Split info ─────────────────────────────────────────
        "splits": {
            "train":            len(splits["train"]),
            "val":              len(splits["val"]),
            "test":             len(splits["test"]),
            "ratio":            "70/15/15",
            "stratified_by":    "heart_volume_category",
            "rare_cat_strategy":"merge into dominant class",
            "why_not_kfold": (
                "Only 30-50 labeled scans available. K-fold test "
                "sets of ~5 scans produce unstable Dice estimates. "
                "Stratified hold-out is standard for small medical "
                "imaging datasets. K-fold preferred at 200+ scans."
            ),
        },

        # ── 3. Heart volume ───────────────────────────────────────
        "heart_volume_ml": {
            "mean":    round(float(df["heart_vol_ml"].mean()), 2),
            "std":     round(float(df["heart_vol_ml"].std()),  2),
            "min":     round(float(df["heart_vol_ml"].min()),  2),
            "max":     round(float(df["heart_vol_ml"].max()),  2),
            "median":  round(float(df["heart_vol_ml"].median()), 2),
        },

        # ── 4. Image spacing + shape (from sampled scans) ─────────
        "image_spacing_mm": {
            "mean": [round(float(x), 3) for x in sp.mean(0).tolist()],
            "std":  [round(float(x), 3) for x in sp.std(0).tolist()],
            "note": "Averaged over 10 sampled scans",
        } if sp.shape[0] > 0 else {},

        "image_shape_voxels": {
            "mean": [round(float(x), 1) for x in sh.mean(0).tolist()],
            "std":  [round(float(x), 1) for x in sh.std(0).tolist()],
        } if sh.shape[0] > 0 else {},

        # ── 5. HU statistics ──────────────────────────────────────
        "hu_statistics_raw": {
            "mean_of_means": round(float(np.mean(hu_means)), 2),
            "mean_of_stds":  round(float(np.mean(hu_stds)),  2),
            "mean_min_hu":   round(float(np.mean(hu_min_vals)), 2),
            "mean_max_hu":   round(float(np.mean(hu_max_vals)), 2),
            "note": "Computed on raw (unwindowed) HU values",
        } if hu_means else {},

        # ── 6. HU window parameters + justification ───────────────
        "hu_window": {
            "window_level":  400,
            "window_width":  1800,
            "a_min":        -500,
            "a_max":         1300,
            "rationale": (
                "WL=400, WW=1800 chosen for cardiac CAC CT. "
                "Captures calcium deposits (130-1000 HU), myocardium "
                "(40-80 HU), blood pool (30-45 HU), pericardial fat "
                "(-100 HU). Lower bound -500 removes lung air. "
                "Upper 1300 includes dense calcium without "
                "saturating on cortical bone (>1300 HU)."
            ),
        },

        # ── 7. Class imbalance ────────────────────────────────────
        "class_imbalance": {
            "issue": (
                "Heart size distribution is uneven — small hearts "
                "are underrepresented but hardest to segment."
            ),
            "strategy":        "WeightedRandomSampler by heart size category",
            "cac_pos_neg":     f"{n_pos}:{n_neg}",
            "cac_imbalance_note": (
                "CAC_seg not used for training (heart segmentation task). "
                "CAC+/- ratio reported for dataset characterization only."
            ),
        },

    }

    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ dataset_stats.json → {STATS_JSON}")
    print(f"   CAC+   : {n_pos} ({stats['dataset']['cac_positive_pct']}%)")
    print(f"   CAC-   : {n_neg}")
    print(f"   Heart vol: {stats['heart_volume_ml']['mean']} ± "
          f"{stats['heart_volume_ml']['std']} ml")
    if sp.shape[0] > 0:
        print(f"   Spacing: {stats['image_spacing_mm']['mean']} mm")

    return stats


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 55)
    print("  PrediCT GSoC — Preprocessing Pipeline")
    print("  Project 1: Heart Segmentation")
    print("═" * 55 + "\n")

    df     = load_and_clean(TRAIN_CSV)
    splits = create_splits(df)
    stats  = compute_stats(df, splits)

    print(f"\n{'═'*55}")
    print(f"  ✅ Done")
    print(f"  Next: python common_task/dataset.py")
    print(f"{'═'*55}")