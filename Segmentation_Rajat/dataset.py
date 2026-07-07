import sys
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import WeightedRandomSampler

from monai.data import (
    Dataset,
    CacheDataset,
    DataLoader,
)
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandShiftIntensityd,
    RandAffined,
    Rand3DElasticd,
    RandGaussianSmoothd,
    EnsureTyped,
    ToTensord,
)
from monai.transforms import MapTransform
from monai.data import PersistentDataset

sys.path.append(str(Path(__file__).parent))
import config

SPLITS_JSON = config.preprocessing_config["SPLITS_JSON"]
BATCH_SIZE  = config.dataloader_config["BATCH_SIZE"]
ROI_SIZE    = tuple(config.dataloader_config["ROI_SIZE"])
CACHE_RATE  = config.dataloader_config["CACHE_RATE"]
NUM_WORKERS = config.dataloader_config["NUM_WORKERS"]
HU          = config.HU_CONFIG


ADD_COORD_CHANNELS = config.dataloader_config["ADD_COORD_CHANNELS"]
DO_HEART_ROI_MASKING = config.dataloader_config["HEART_MASK_FLAG"]
ADD_HEART_MASK_CHANNEL = config.dataloader_config["ADD_HEART_MASK_CHANNEL"]
COORD_MODE = config.dataloader_config["COORD_MODE"]

# ── Patch size ────────────────────────────────────────────────────
# (96,128,96)  → safe for 8-12GB VRAM,  batch_size=2
# (112,160,128)→ nnU-Net native,         needs 24GB VRAM, batch_size=2

ROI_SIZE = (128, 128, 35)

# For Caching we will go with Persisten Cahcing, where we store the deterministic transforms on disk, and load them into memory during training. This is more efficient than caching in memory for large datasets, and allows us to reuse the cached data across multiple runs.
# So Assuming Float32 images and Binary labels, we can estimate the cache size as follows:
# Each scan is Worst case 300x300x60x4 bytes(float 32) = 21.6MB for image + 300x300x60x1 byte(uint8) = 5.4MB for label → ~27MB per scan.
# 27MB x 789 scans = ~21.3GB total cache size if we cache everything.

# ══════════════════════════════════════════════════════════════════
#  ADDITION OF CHANNELS (HEART MASK, COORD CONV)
# ══════════════════════════════════════════════════════════════════

class AddHeartMaskChanneld(MapTransform):
    def __init__(
        self,
        image_key="image",
        roi_key="roi_mask",
    ):
        super().__init__([image_key, roi_key])
        self.image_key = image_key
        self.roi_key = roi_key

    def __call__(self, data):
        d = dict(data)

        img = d[self.image_key]
        roi = (d[self.roi_key] > 0).astype(img.dtype)

        d[self.image_key] = np.concatenate(
            [img, roi],
            axis=0,
        )

        return d
    
class DualHUWindowingd(MapTransform):
    def __init__(self, image_key="image"):
        super().__init__([image_key])
        self.image_key = image_key

    def __call__(self, data):
        d = dict(data)

        img = d[self.image_key]

        # Tissue window [-100, 400]
        tissue = np.clip(img, -100.0, 400.0)
        tissue = (tissue + 100.0) / 500.0

        # Calcium window [130, 1000]
        calcium = np.clip(img, 130.0, 1000.0)
        calcium = (calcium - 130.0) / (1000.0 - 130.0)

        d[self.image_key] = np.concatenate(
            [tissue, calcium],
            axis=0
        ).astype(np.float32)

        return d

# Right now we are doing PIXEL CO-ORD CONVULTION WITH REALTIVE ENCODING
class AddCoordConvChannelsd(MapTransform):
    def __init__(
        self,
        image_key="image",
        normalized=True,
    ):
        super().__init__([image_key])
        self.image_key = image_key
        self.normalized = normalized

    def __call__(self, data):

        d = dict(data)

        img = d[self.image_key]

        _, D, H, W = img.shape

        if self.normalized:

            z = np.linspace(-1, 1, D, dtype=np.float32)
            y = np.linspace(-1, 1, H, dtype=np.float32)
            x = np.linspace(-1, 1, W, dtype=np.float32)

        else:

            z = np.arange(D, dtype=np.float32)
            y = np.arange(H, dtype=np.float32)
            x = np.arange(W, dtype=np.float32)

        zz = np.broadcast_to(
            z[:, None, None],
            (D, H, W)
        )

        yy = np.broadcast_to(
            y[None, :, None],
            (D, H, W)
        )

        xx = np.broadcast_to(
            x[None, None, :],
            (D, H, W)
        )

        coords = np.stack(
            [zz, yy, xx],
            axis=0
        )

        d[self.image_key] = np.concatenate(
            [img, coords],
            axis=0
        )

        return d


# ══════════════════════════════════════════════════════════════════
#  TRANSFORMS
# ══════════════════════════════════════════════════════════════════


class ApplyHeartROIMaskd(MapTransform):
    def __init__(
        self,
        image_key="image",
        roi_key="roi_mask",
    ):
        super().__init__([image_key, roi_key])
        self.image_key = image_key
        self.roi_key = roi_key

    def __call__(self, data):
        d = dict(data)

        img = d[self.image_key]
        roi = d[self.roi_key]

        roi = (roi > 0).astype(img.dtype)

        d[self.image_key] = img * roi

        return d

def get_transforms(mode: str):

    load_keys = ["image", "label"]

    if DO_HEART_ROI_MASKING or ADD_HEART_MASK_CHANNEL:
        load_keys.append("roi_mask")

    base = []

    # Load
    base.append(
        LoadImaged(
            keys=load_keys,
            image_only=False,
        )
    )

    # Ensure channel first
    base.append(
        EnsureChannelFirstd(keys=load_keys)
    )

    # DUAL HU WINDOWING! 

    base.append(
        DualHUWindowingd(
            image_key="image"
        )
    )
    
    # Add heart mask as extra channel
    if ADD_HEART_MASK_CHANNEL:
        base.append(
            AddHeartMaskChanneld(
                image_key="image",
                roi_key="roi_mask",
            )
        )

    # Add CoordConv channels
    if ADD_COORD_CHANNELS:
        base.append(
            AddCoordConvChannelsd(
                image_key="image",
                normalized=(COORD_MODE == "normalized"),
            )
        )

    # # Resample to isotropic spacing
    # base.append(
    #     Spacingd(
    #         keys=["image", "label"],
    #         pixdim=(1.0, 1.0, 1.0),
    #         mode=("bilinear", "nearest"),
    #     )
    # ) # already resampled before passing 

    # Reorient to RAS
    base.append(
        Orientationd(
            keys=load_keys,
            axcodes="RAS",
        )
    )

    # Apply heart ROI masking if enabled
    if DO_HEART_ROI_MASKING:
        base.append(
            ApplyHeartROIMaskd(
                image_key="image",
                roi_key="roi_mask",
            )
        )

    # Crop around foreground
    base.append(
        CropForegroundd(
            keys=load_keys,
            source_key="image",
            margin=5,
        )
    )

    base.append(
        EnsureTyped(keys=load_keys)
    )

    if mode == "train":

        aug = [

            # Foreground-biased patch sampling
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=ROI_SIZE,
                pos=2,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),

            # Anatomically valid flips
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=0,
            ),

            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=1,
            ),

            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=2,
            ),

            # Small realistic geometric perturbations
            RandAffined(
                keys=["image", "label"],
                prob=0.2,
                rotate_range=(0.1, 0.1, 0.1),  # ~6 degrees
                scale_range=(0.05, 0.05, 0.05),
                mode=("bilinear", "nearest"),
            ),

            # Conservative elastic deformation
            Rand3DElasticd(
                keys=["image", "label"],
                prob=0.15,
                sigma_range=(4, 6),
                magnitude_range=(0.5, 1.5),
                mode=("bilinear", "nearest"),
            ),

            # Simulated scanner blur / reconstruction variability
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.2,
                sigma_x=(0.25, 0.75),
                sigma_y=(0.25, 0.75),
                sigma_z=(0.25, 0.75),
            ),

            # Low-dose CT style noise
            RandGaussianNoised(
                keys=["image"],
                prob=0.25,
                std=0.015,
            ),

            # Gamma / contrast augmentation
            RandAdjustContrastd(
                keys=["image"],
                prob=0.3,
                gamma=(0.7, 1.5),
            ),

            # Small HU calibration shifts
            RandShiftIntensityd(
                keys=["image"],
                prob=0.3,
                offsets=0.05,
            ),

            ToTensord(
                keys=["image", "label"]
            ),
        ]

        return Compose(base + aug)

    # Validation / Test
    return Compose(
        base + [
            ToTensord(keys=["image", "label"])
        ]
    )


# ══════════════════════════════════════════════════════════════════
#  DATASET CLASS
#
#  PersistentDataset is a MONAI dataset that caches transformed data on disk.
#  We will need some where around 30GBs of disk space from my estimate
#
# ══════════════════════════════════════════════════════════════════

from monai.data import PersistentDataset

class CacSegDataset(PersistentDataset):
    """
    Args:
        data  : list of dicts with "image", "label", "id"
        mode  : "train" | "val" | "test"
    """

    def __init__(self, data: list, mode: str):
        n = len(data)

        print(f"   [{mode:5s}] {n} scans | "
              f"cache_method: Persistent | "
              f"roi={ROI_SIZE if mode=='train' else 'full volume'}")
        
        parent_dir = Path(__file__).parent

        super().__init__(
            data=data,
            transform=get_transforms(mode),
            cache_dir=f"{parent_dir}/monai_cache",
        )
        self.mode      = mode
        self.data_list = data

    def __repr__(self):
        return (
            f"CAC_Seg_Dataset("
            f"mode='{self.mode}', "
            f"n={len(self.data_list)})"
        )


# ══════════════════════════════════════════════════════════════════
#  WEIGHTED SAMPLER
# ══════════════════════════════════════════════════════════════════

def make_weighted_sampler(train_data: list) -> WeightedRandomSampler:
    """
    Balance scans based on Agatston availability.

    agatston_available=0 -> no score available
    agatston_available=1 -> score available
    """

    classes = [
        int(d.get("agatston_available", 0))
        for d in train_data
    ]

    class_counts = np.bincount(
        classes,
        minlength=2,
    ).astype(float)

    # avoid divide-by-zero
    class_counts = np.where(
        class_counts == 0,
        1.0,
        class_counts,
    )

    weights = 1.0 / class_counts[classes]

    print(
        "\n   WeightedSampler — "
        f"NoAgatston(0): {int(class_counts[0])}  "
        f"HasAgatston(1): {int(class_counts[1])}"
    )

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


# ══════════════════════════════════════════════════════════════════
#  BUILD DATALOADERS
# ══════════════════════════════════════════════════════════════════

def build_dataloaders(splits_json: str = SPLITS_JSON):
    """
    Build train / val / test DataLoaders from splits.json.

    train_loader:
      CacheDataset + WeightedRandomSampler + augmentation
      batch_size from config (1 for <12GB VRAM)

    val_loader / test_loader:
      CacheDataset, no augmentation, batch_size=1, full volume
      Full volume inference needed for accurate Dice evaluation
    """
    with open(splits_json) as f:
        splits = json.load(f)

    n_train = len(splits["train"])
    n_val   = len(splits["val"])
    n_test  = len(splits["test"])

    print("\n📦 Building CACSegDatasets")
    print(f"   Splits : {splits_json}")
    print(f"   ROI    : {ROI_SIZE}")
    print(f"   Batch  : {BATCH_SIZE}\n")

    # Warn if test set is too small
    if n_test < 3:
        print(f"⚠️  Test set has only {n_test} scan(s).")
        print(f"   Dice score on {n_test} scan is not reliable.")
        print(f"   Run TotalSegmentator on more scans for stable evaluation.\n")

    train_ds = CacSegDataset(splits["train"], mode="train")
    val_ds   = CacSegDataset(splits["val"],   mode="val")
    test_ds  = CacSegDataset(splits["test"],  mode="test")

    sampler = make_weighted_sampler(splits["train"])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"\n✅ DataLoaders ready")
    print(f"   train : {n_train} scans | "
          f"batch={BATCH_SIZE} | weighted sampler")
    print(f"   val   : {n_val} scans | batch=1 | full volume")
    print(f"   test  : {n_test} scans | batch=1 | full volume")

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════
#  STANDALONE VERIFICATION
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print("═" * 55)
    print("  PrediCT GSoC — Dataset + DataLoader Verification")
    print("═" * 55)

    train_loader, val_loader, test_loader = build_dataloaders()

    print("\n🔍 Loading one training batch...")
    print("   (First load triggers caching — may take 30-60s)\n")

    batch = next(iter(train_loader))
    batch = next(iter(train_loader))
    batch = next(iter(train_loader))


    img = batch["image"]
    lbl = batch["label"]

    print(f"\n   image shape      : {list(img.shape)}")
    print(f"   label shape      : {list(lbl.shape)}")
    print(f"   image dtype      : {img.dtype}")
    print(f"   image min        : {img.min():.4f}")
    print(f"   image max        : {img.max():.4f}")
    print(f"   label unique     : {lbl.unique().tolist()}")
    print(f"   label sum        : {int(lbl.sum())} foreground voxels")

    # ---------------------------------------------------------
    # Channel verification
    # ---------------------------------------------------------

    expected_channels = 2  # Dual HU windowing

    if ADD_HEART_MASK_CHANNEL:
        expected_channels += 1

    if ADD_COORD_CHANNELS:
        expected_channels += 3

    print(f"   image channels   : {img.shape[1]}")
    print(f"   expected channels: {expected_channels}")

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    errors = []

    # Shape checks
    if img.shape[1] != expected_channels:
        errors.append(
            f"Expected {expected_channels} channels "
            f"but got {img.shape[1]}"
        )

    if list(img.shape[-3:]) != list(ROI_SIZE):
        errors.append(
            f"patch shape {list(img.shape[-3:])} "
            f"!= ROI {list(ROI_SIZE)}"
        )

    # Image checks
    if torch.isnan(img).any():
        errors.append("NaNs found in image")

    if torch.isinf(img).any():
        errors.append("Infs found in image")

    # Due to Gaussian it can happen 

    # if img.min() < -0.01:
    #     errors.append(
    #         f"image min {img.min():.4f} < 0"
    #     )

    # if img.max() > 1.01:
    #     errors.append(
    #         f"image max {img.max():.4f} > 1"
    #     )

    # Label checks
    label_values = set(
        lbl.unique().cpu().numpy().tolist()
    )

    if not label_values.issubset({0, 1, 0.0, 1.0}):
        errors.append(
            f"label not binary: {sorted(label_values)}"
        )

    # ---------------------------------------------------------
    # Heart mask verification
    # ---------------------------------------------------------

    if ADD_HEART_MASK_CHANNEL:

        heart_idx = 2

        heart_channel = img[:, heart_idx]

        heart_unique = torch.unique(
            heart_channel
        ).cpu().numpy()

        print(
            f"   heart mask vals  : "
            f"{heart_unique[:10]}"
        )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    if errors:

        print("\n❌ Checks FAILED:")

        for e in errors:
            print(f"   • {e}")

    else:

        print("\n✅ All checks passed")

        print(
            f"   channels={expected_channels}"
        )

        print(
            "   Dual HU windowing verified"
        )

        if ADD_HEART_MASK_CHANNEL:
            print(
                "   Heart-mask channel verified"
            )

        if ADD_COORD_CHANNELS:
            print(
                "   CoordConv channels verified"
            )

        print(
            "   Persistent caching active"
        )
