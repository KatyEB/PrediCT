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
    EnsureTyped,
    ToTensord,
)

sys.path.append(str(Path(__file__).parent))
import config

SPLITS_JSON = config.preprocessing_config["SPLITS_JSON"]
BATCH_SIZE  = config.dataloader_config["BATCH_SIZE"]
ROI_SIZE    = tuple(config.dataloader_config["ROI_SIZE"])
CACHE_RATE  = config.dataloader_config["CACHE_RATE"]
NUM_WORKERS = config.dataloader_config["NUM_WORKERS"]
HU          = config.HU_CONFIG


# For Caching we will go with Persisten Cahcing, where we store the deterministic transforms on disk, and load them into memory during training. This is more efficient than caching in memory for large datasets, and allows us to reuse the cached data across multiple runs.
# So Assuming Float32 images and Binary labels, we can estimate the cache size as follows:
# Each scan is Worst case 300x300x60x4 bytes(float 32) = 21.6MB for image + 300x300x60x1 byte(uint8) = 5.4MB for label → ~27MB per scan.
# 27MB x 789 scans = ~21.3GB total cache size if we cache everything.


# ══════════════════════════════════════════════════════════════════
#  TRANSFORMS
# ══════════════════════════════════════════════════════════════════

def get_transforms(mode: str) -> Compose:
    """
    MONAI transform pipeline for Project CAC segmentation.

    ── Shared base (train + val + test) ─────────────────────────────

    Orientationd → RAS:
      Reorients all volumes to standard RAS (Right-Anterior-Superior).
      COCA scans may have different orientations across patients.
      Without this, flipping augmentations are anatomically inconsistent
      e.g. left/right flip would mean different things per patient.
      Always apply before Spacingd.

    ScaleIntensityRanged → HU window -500 to +1300:
      WL=400, WW=1800 — cardiac CAC optimized.
      Captures: calcium (130-1000 HU), myocardium (40-80 HU),
                blood pool (30-45 HU), pericardial fat (-100 HU).
      Clips and normalizes to [0.0, 1.0] for network input.

    CropForegroundd (margin=10):
      Removes large empty air borders before patch extraction.
      source_key="image" → crops based on non-zero image region.
      margin=10 voxels → heart never accidentally cropped out.
      Reduces volume size → faster patch sampling + less VRAM.

    ── Train only ───────────────────────────────────────────────────

    RandCropByPosNegLabeld (pos=2, neg=1, num_samples=2):
      Foreground-biased patch sampling.
      pos=2, neg=1 → 2/3 patches contain heart voxels.
      Prevents degenerate all-background prediction.
      num_samples=2 → 2 patches per scan per epoch.
      With 14 train scans × 2 = 28 patches/epoch — enough.

    Geometric augmentations (image + label jointly):
      RandFlip ×3: safe — heart has no orientation constraint in CT.
      RandRotate90: cardiac CT orientation varies across scanners.
      RandZoom 0.85-1.15: simulates patient size + FOV variation.
      Applied to image AND label simultaneously → spatial alignment.

    Intensity augmentations (image only):
      GaussianNoise: electronic noise variation across scanners.
      AdjustContrast: reconstruction kernel differences in COCA.
      ShiftIntensity: scanner calibration drift between sites.
    """
    assert mode in ("train", "val", "test"), \
        f"mode must be train/val/test, got '{mode}'"

    base = [
        LoadImaged(
            keys=["image", "label"],
            image_only=False,
        ),
        EnsureChannelFirstd(keys=["image", "label"]),

        # Reorient to RAS before any spatial transforms
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # Alreayd we have 1mm isotropic spacing, but ensure consistent resampling across all scans
        # Spacingd must come after Orientationd to ensure consistent resampling
        # Resample to 1.0mm isotropic
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),

        # HU windowing → normalize to [0, 1]
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU["A_MIN"],    # -500
            a_max=HU["A_MAX"],    # +1300
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        # Crop tight around foreground
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            margin=10,
        ),

        EnsureTyped(keys=["image", "label"]),
    ]

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

            # Geometric — image + label together
            RandFlipd(
                keys=["image", "label"],
                prob=0.5, spatial_axis=0),
            RandFlipd(
                keys=["image", "label"],
                prob=0.5, spatial_axis=1),
            RandFlipd(
                keys=["image", "label"],
                prob=0.5, spatial_axis=2),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.3, max_k=3),
            RandZoomd(
                keys=["image", "label"],
                prob=0.3,
                min_zoom=0.85,
                max_zoom=1.15,
                mode=["trilinear", "nearest"],
            ),

            # Intensity — image only
            RandGaussianNoised(
                keys=["image"], prob=0.2, std=0.01),
            RandAdjustContrastd(
                keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            RandShiftIntensityd(
                keys=["image"], prob=0.3, offsets=0.1),

            ToTensord(keys=["image", "label"]),
        ]
        return Compose(base + aug)

    # Val/Test — no augmentation, no patching, full volume
    return Compose(base + [ToTensord(keys=["image", "label"])])


# ══════════════════════════════════════════════════════════════════
#  DATASET CLASS
#
#  Uses CacheDataset — NOT plain Dataset.
#
#  Why CacheDataset with 20 scans?
#    Loading + resampling one NIfTI to 1.5mm isotropic takes ~2-3s.
#    With 14 train scans × 100 epochs = 1400 loads from disk.
#    With caching:  1400 loads → 14 loads (epoch 1) + 1386 from RAM.
#    Time saved: ~1386 × 2.5s = ~58 minutes of IO eliminated.
#    At cache_rate=1.0 with 20 scans, ~1.5GB RAM used — totally safe.

# ══════════════════════════════════════════════════════════════════

class CacSegDataset(CacheDataset):
    """
    Args:
        data  : list of dicts with "image", "label", "id"
        mode  : "train" | "val" | "test"
    """

    def __init__(self, data: list, mode: str):
        n          = len(data)

        print(f"   [{mode:5s}] {n} scans | "
              f"cache_method: Persistent | "
              f"roi={ROI_SIZE if mode=='train' else 'full volume'}")

        super().__init__(
            data=data,
            transform=get_transforms(mode),
            cache_dir="./monai_cache",
            num_workers=NUM_WORKERS,
        )
        self.mode      = mode
        self.data_list = data

    def __repr__(self) -> str:
        return (
            f"CAC_Seg_Dataset("
            f"mode={self.mode}, "
            f"n={len(self.data_list)}, "
        )


# ══════════════════════════════════════════════════════════════════
#  WEIGHTED SAMPLER
# ══════════════════════════════════════════════════════════════════

def make_weighted_sampler(train_data: list) -> WeightedRandomSampler:
    """
    Balance heart size categories during training.

    Your current train set:
      size_cat=0 (Small  <550ml) : 4 scans
      size_cat=1 (Medium 550-800): 10 scans
      size_cat=2 (Large  >800ml) : 0 scans

    Without sampler: model sees medium hearts 71% of the time.
    With sampler:    each category sampled equally → better generalization.
    """
    cats         = [d["size_cat"] for d in train_data]
    class_counts = np.bincount(cats, minlength=3).astype(float)
    class_counts  = np.where(class_counts == 0, 1.0, class_counts)
    weights       = 1.0 / class_counts[cats]

    print(f"\n   WeightedSampler — "
          f"Small(0): {int(class_counts[0])}  "
          f"Med(1): {int(class_counts[1])}  "
          f"Large(2): {int(class_counts[2])}")

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

    print(f"\n📦 Building HeartSegDatasets")
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

    print(f"\n🔍 Loading one training batch...")
    print(f"   (First load triggers caching — may take 30-60s)\n")

    batch = next(iter(train_loader))
    img   = batch["image"]
    lbl   = batch["label"]

    print(f"\n   image shape  : {list(img.shape)}")
    print(f"   label shape  : {list(lbl.shape)}")
    print(f"   image dtype  : {img.dtype}")
    print(f"   image min    : {img.min():.4f}")
    print(f"   image max    : {img.max():.4f}")
    print(f"   label unique : {lbl.unique().tolist()}")
    print(f"   label sum    : {int(lbl.sum())} foreground voxels")

    # ── Sanity checks ─────────────────────────────────────────────
    errors = []
    if img.min() < -0.01:
        errors.append(f"image min {img.min():.4f} < 0 — HU window broken")
    if img.max() > 1.01:
        errors.append(f"image max {img.max():.4f} > 1 — HU window broken")
    if not set(lbl.unique().numpy().flatten().tolist())\
            .issubset({0.0, 1.0}):
        errors.append(f"label not binary: {lbl.unique().tolist()}")
    if lbl.sum() == 0:
        errors.append("label all zeros — check heart_mask paths")
    if list(img.shape[-3:]) != list(ROI_SIZE):
        errors.append(
            f"patch shape {list(img.shape[-3:])} != ROI {list(ROI_SIZE)}")

    if errors:
        print(f"\n❌ Checks FAILED:")
        for e in errors:
            print(f"   • {e}")
    else:
        print(f"\n✅ All checks passed")
        print(f"   Caching active — subsequent epochs will be fast")
        print(f"   Ready → python common_task/train.py")