import pandas as pd
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    RandRotate90d, RandCropByPosNegLabeld,
    RandFlipd, SpatialPadd, ToTensord, RandGaussianNoised,
    ScaleIntensityRanged, CropForegroundd, CastToTyped, AsDiscreted,
    ConcatItemsd, DeleteItemsd,
)
from monai.data import DataLoader, CacheDataset


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
USE_CARDIAC_MASK = True   # Set False if cardiac masks weren't generated
# ---------------------------------------------------------------------------


def _base_load_transforms(use_cardiac_mask: bool) -> list:
    """
    Transforms shared by train and val:
      - Load image + label (+ optional cardiac mask)
      - Ensure channel first
      - Foreground crop on image
      - Intensity scaling
    """
    keys = ["image", "label", "cardiac_mask"] if use_cardiac_mask else ["image", "label"]
    img_keys = ["image", "cardiac_mask"] if use_cardiac_mask else ["image"]

    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),

        # Crop to body, ignoring air/lungs — applied to all keys together
        # so spatial alignment is preserved
        CropForegroundd(
            keys=keys,
            source_key="image",
            select_fn=lambda x: x > -200,
            margin=10,
        ),

        # Intensity normalisation: calcium is 200–700 HU, bone 700–1000 HU.
        # Clipping at 1000 compresses bone without zeroing it out entirely,
        # which helps the model learn to distinguish calcium from bone edges.
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-100, a_max=1000,
            b_min=0.0,  b_max=1.0,
            clip=True,
        ),
    ]

    if use_cardiac_mask:
        # Multiply image by cardiac mask to zero out non-cardiac regions,
        # then drop the mask key — the model only sees the masked image.
        # This is done AFTER intensity scaling so values stay in [0, 1].
        transforms += [
            # cardiac_mask is binary {0,1} — simple elementwise multiply
            # We use a Lambda here since MONAI has no built-in MaskIntensityd
            # that handles the channel dim correctly post-EnsureChannelFirst.
            AsDiscreted(keys=["cardiac_mask"], threshold=0.5),
            # Now apply: image = image * cardiac_mask
            # ConcatItemsd stacks along channel, we then use a custom fn.
            # Simpler: just use a direct torch op via ToTensord + manual mask.
            # We'll apply the mask after ToTensord in the dataset __getitem__
            # by keeping cardiac_mask in the dict — see MaskedCacheDataset below.
        ]

    return transforms


def get_transforms(patch_size: tuple, use_cardiac_mask: bool = USE_CARDIAC_MASK):
    keys = ["image", "label", "cardiac_mask"] if use_cardiac_mask else ["image", "label"]

    shared = _base_load_transforms(use_cardiac_mask)

    train_transforms = Compose(shared + [
        # Sample patches — bias toward calcium-containing regions
        RandCropByPosNegLabeld(
            keys=keys,
            label_key="label",
            spatial_size=(96, 96, 32),
            pos=2,           # increased from 1 — calcium is very sparse
            neg=1,
            num_samples=4,
        ),

        # Pad if any patch is smaller than patch_size (edge cases)
        SpatialPadd(keys=keys, spatial_size=patch_size, mode="constant"),

        # Cast before augmentation to avoid dtype surprises
        CastToTyped(
            keys=["image"] + (["cardiac_mask"] if use_cardiac_mask else []),
            dtype=torch.float32,
        ),
        CastToTyped(keys=["label"], dtype=torch.uint8),
        AsDiscreted(keys=["label"], threshold=0.5),

        # Augmentation — axis-aligned only, since calcium deposits are small
        # and aggressive rotation would destroy the HU-based signal
        RandFlipd(keys=keys, spatial_axis=[0], prob=0.5),
        RandFlipd(keys=keys, spatial_axis=[1], prob=0.5),
        RandRotate90d(keys=keys, prob=0.5, max_k=3),

        # Gaussian noise on image only, AFTER all spatial ops
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),

        ToTensord(keys=keys),
    ])

    val_transforms = Compose(shared + [
        # Val needs the same spatial padding as train for consistent tensor shapes
        SpatialPadd(keys=keys, spatial_size=patch_size, mode="constant"),
        CastToTyped(
            keys=["image"] + (["cardiac_mask"] if use_cardiac_mask else []),
            dtype=torch.float32,
        ),
        CastToTyped(keys=["label"], dtype=torch.uint8),
        AsDiscreted(keys=["label"], threshold=0.5),
        ToTensord(keys=keys),
    ])

    return train_transforms, val_transforms


class MaskedCacheDataset(CacheDataset):
    """
    Thin wrapper around CacheDataset that applies the cardiac mask to the
    image after all cached transforms have run. Doing it here rather than
    inside the transform pipeline avoids caching the masked image — we cache
    the raw normalised image and apply the mask on the fly, which means we
    can ablate USE_CARDIAC_MASK without reprocessing the cache.
    """
    def __init__(self, use_cardiac_mask: bool, **kwargs):
        super().__init__(**kwargs)
        self.use_cardiac_mask = use_cardiac_mask

    def __getitem__(self, index):
        items = super().__getitem__(index)
        if not self.use_cardiac_mask:
            return items

        # items is a list of dicts when num_samples > 1
        if isinstance(items, list):
            for item in items:
                self._apply_mask(item)
        else:
            self._apply_mask(items)
        return items

    @staticmethod
    def _apply_mask(item: dict):
        if "cardiac_mask" in item and "image" in item:
            item["image"] = item["image"] * item["cardiac_mask"]
            del item["cardiac_mask"]


def build_file_dicts(df: pd.DataFrame, use_cardiac_mask: bool) -> list[dict]:
    """Build MONAI-style file dicts from the scan_index DataFrame."""
    rows = []
    for _, r in df.iterrows():
        d = {"image": r["image_path"], "label": r["mask_path"]}
        if use_cardiac_mask and pd.notna(r.get("cardiac_mask_path")):
            d["cardiac_mask"] = r["cardiac_mask_path"]
        elif use_cardiac_mask:
            # Cardiac mask missing for this scan — fall back gracefully
            d["cardiac_mask"] = r["image_path"]   # will be overwritten by ones
        rows.append(d)
    return rows


def get_coca_loaders(
    train_parquet: str,
    val_parquet: str,
    patch_size: tuple = (128, 128, 32),
    batch_size: int = 2,
    use_cardiac_mask: bool = USE_CARDIAC_MASK,
):
    train_df = pd.read_parquet(train_parquet)
    val_df   = pd.read_parquet(val_parquet)

    # Use has_calcium column from processor — no need to open each NIfTI
    pos_df = train_df[train_df["has_calcium"]]
    neg_df = train_df[~train_df["has_calcium"]]

    # All positives + 10% negatives for context
    neg_sample = neg_df.sample(frac=0.1, random_state=42)
    curated_df = pd.concat([pos_df, neg_sample]).reset_index(drop=True)

    print(f"  Train: {len(pos_df)} positive, {len(neg_sample)} negative sampled "
          f"({len(curated_df)} total)")
    print(f"  Val:   {len(val_df)} scans")

    train_files = build_file_dicts(curated_df, use_cardiac_mask)
    val_files   = build_file_dicts(val_df,     use_cardiac_mask)

    train_trans, val_trans = get_transforms(patch_size, use_cardiac_mask)

    train_ds = MaskedCacheDataset(
        use_cardiac_mask=use_cardiac_mask,
        data=train_files,
        transform=train_trans,
        cache_rate=1.0,
    )
    val_ds = MaskedCacheDataset(
        use_cardiac_mask=use_cardiac_mask,
        data=val_files,
        transform=val_trans,
        cache_rate=1.0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0,    # keep 0 for CacheDataset on Windows
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        num_workers=0,    # CacheDataset is not fork-safe, keep 0
        pin_memory=True,
    )

    return train_loader, val_loader
