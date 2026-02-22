import os
import pandas as pd
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    RandRotate90d, RandCropByPosNegLabeld, 
    RandFlipd, SpatialPadd, ToTensord, RandGaussianNoised, CenterSpatialCropd,
    ScaleIntensityRanged, CropForegroundd
)
from monai.data import Dataset, DataLoader, CacheDataset
# Rule: Standardize Heart CT intensities (-100 to 400 HU)
from monai.transforms import CastToTyped, AsDiscreted
import nibabel as nib

def get_transforms(patch_size):
    # Ensure patch_size is explicitly (128, 128, 32)
    print("🚀 LOADER CHECK: THE CODE IS ACTUALLY RUNNING!")
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    # 1. CROP: Find the patient's body (HU > -200 ignores the lungs/air)
        # This centers the "camera" on the soft tissue of the chest.
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            select_fn=lambda x: x > -200, 
            margin=10
        ),

        # 2. INTENSITY: The "Anti-Bone" Filter
        # We clip the max at 600. Why? Because calcium is 200-500.
        # Anything 1000+ (Bone) gets squashed down to 1.0, 
        # making it look less "special" to the model.
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-100, a_max=1000, 
            b_min=0.0, b_max=1.0, 
            clip=True
        ),

        # 3. SAMPLE: Force the model to look at labels
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 32),
            pos=1, # Focus heavily on the calcium labels
            neg=1,
            num_samples=4
        ),
        ToTensord(keys=["image", "label"]),
        # 1. CAST FIRST: Prevents rounding issues on ARM/Windows
        CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.uint8)),
        
        # 2. EXPLICIT PADDING: Added mode to ensure it's not trying to 'reflect'
        SpatialPadd(
            keys=["image", "label"], 
            spatial_size=patch_size, 
            mode="constant"
        ),
        
        AsDiscreted(keys=["label"], threshold=0.5),

        # Adding back the augmentations to help the model learn
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        ToTensord(keys=["image", "label"]),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Padding is still needed here for consistency
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant"),
        AsDiscreted(keys=["label"], threshold=0.5),
        ToTensord(keys=["image", "label"]),
    ])
    return train_transforms, val_transforms
# 1. Define your helper function
def has_calcium(file_dict):
    lbl_img = nib.load(file_dict['label'])
    return lbl_img.get_fdata().max() > 0

def get_coca_loaders(train_parquet, val_parquet, patch_size=(128, 128, 32), batch_size=2):
    train_df = pd.read_parquet(train_parquet)
    val_df = pd.read_parquet(val_parquet)

    train_files = [{"image": r["image_path"], "label": r["mask_path"]} for _, r in train_df.iterrows()]
    val_files = [{"image": r["image_path"], "label": r["mask_path"]} for _, r in val_df.iterrows()]

    train_trans, val_trans = get_transforms(patch_size)

    # 2. Split the list after you've defined your 'train_files' list
    pos_files = [f for f in train_files if has_calcium(f)]
    neg_files = [f for f in train_files if f not in pos_files]

    # 3. Create the 'Curated' list (e.g., all positive + 10% negative for context)
    curated_train_files = pos_files + neg_files[:int(len(pos_files) * 0.1)]

    # 4. Pass the curated list into your MONAI Dataset
    train_ds = CacheDataset(
        data=curated_train_files, 
        transform=train_trans,
        cache_rate=1.0  # Since your curated set is smaller, try to cache it all!
    )
    val_ds = CacheDataset(data=val_files, transform=val_trans, cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)


    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    return train_loader, val_loader