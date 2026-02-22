import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from monai.networks.nets import UNet
from monai.losses import DiceFocalLoss
from monai.visualize import matshow3d
from data.big_data_loader import get_coca_loaders
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


def save_audit_image(inputs, labels, conf_map, batch_idx, epoch):
    # 1. Find the exact slice (z) where the model is most confident
    max_idx = torch.argmax(conf_map[0]).item()
    coords = np.unravel_index(max_idx, conf_map[0].shape)
    z = coords[3] 

    img_slice = inputs[0, 0, :, :, z].cpu().numpy()
    gt_slice = labels[0, 0, :, :, z].cpu().numpy()
    pred_slice = conf_map[0, 0, :, :, z].detach().cpu().numpy()

    # 2. Check if THIS SPECIFIC SLICE has a label
    gt_present_on_slice = "YES" if gt_slice.max() > 0 else "NO"
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Pure CT (The "Judge" panel)
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title(f"CT Slice {z} (Raw)")
    
    # Panel 2: Ground Truth on this slice
    axes[1].imshow(img_slice, cmap='gray')
    axes[1].imshow(gt_slice, cmap='Reds', alpha=0.5)
    axes[1].set_title(f"Label on this slice: {gt_present_on_slice}")
    
    # Panel 3: Model Prediction
    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(pred_slice, cmap='hot', alpha=0.5)
    axes[2].set_title(f"Model Confidence: {pred_slice.max():.2f}")

    plt.suptitle(f"Epoch {epoch} | Batch {batch_idx} | Match: {'✅' if gt_present_on_slice == 'YES' else '❌'}", fontsize=16)
    
    os.makedirs("audit_images", exist_ok=True)
    plt.savefig(f"audit_images/epoch{epoch}_batch{batch_idx}.png")
    plt.close()

def check_ground_truth(batch):
    # This grabs the first 3D volume in the batch
    img = batch["image"][0, 0].cpu().numpy() 
    mask = batch["label"][0, 0].cpu().numpy()
    slice_idx = img.shape[2] // 2 # Grab the middle slice
    
    plt.imshow(img[:, :, slice_idx], cmap='gray')
    plt.imshow(mask[:, :, slice_idx], cmap='Reds', alpha=0.5)
    plt.title(f"Visual Check: Slice {slice_idx}")
    plt.show()

class SliceScroller:
    def __init__(self, image, mask):
        self.image = image.cpu().numpy() if torch.is_tensor(image) else image
        self.mask = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        self.slices = self.image.shape[0]
        self.ind = self.slices // 2
        self.v_min = np.percentile(self.image, 1)
        self.v_max = np.percentile(self.image, 99)

        # AUTO-CONTRAST LOGIC
        # We calculate the 99th percentile to avoid white-out from outliers
        # If the image is still "flat", fall back to raw min/max
        if self.v_min == self.v_max:
            self.v_min, self.v_max = self.image.min(), self.image.max()

        print(f"👁️ Visualizing with dynamic range: {self.v_min:.4f} to {self.v_max:.4f}")

        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.im = self.ax.imshow(self.image[self.ind, :, :], 
                                 cmap="gray", 
                                 vmin=self.v_min, 
                                 vmax=self.v_max)
        
        masked_data = np.ma.masked_where(self.mask[self.ind, :, :] == 0, self.mask[self.ind, :, :])
        self.mask_im = self.ax.imshow(masked_data, cmap="spring", alpha=0.8)
        
        self.ax.set_title(f"Slice {self.ind} | Scroll to move")
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_scroll(self, event):
        self.ind = (self.ind + 1) % self.slices if event.button == 'up' else (self.ind - 1) % self.slices
        self.update()

    def on_key(self, event):
        if event.key == 'up': self.ind = (self.ind + 1) % self.slices
        elif event.key == 'down': self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.image[self.ind, :, :])
        # Force the display to keep the same contrast levels
        self.im.set_clim(vmin=self.v_min, vmax=self.v_max) 
        
        masked_data = np.ma.masked_where(self.mask[self.ind, :, :] == 0, self.mask[self.ind, :, :])
        self.mask_im.set_data(masked_data)
        self.ax.set_title(f"Slice {self.ind}")
        self.fig.canvas.draw()
# 1. ENVIRONMENT TOGGLE
# Set DEBUG = True for your ARM laptop (Weekend)
# Set DEBUG = False for the GPU Cluster (Monday)
DEBUG = True 
def run_validation(model, val_loader, device, patch_size):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_label = AsDiscrete(threshold=0.5)
    post_pred = AsDiscrete(threshold=0.5)

    print("🩺 Running Validation...")
    with torch.no_grad():
        for val_data in val_loader:
            val_images = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            
            # Use sliding window to handle full volumes
            val_outputs = sliding_window_inference(
                val_images, patch_size, 4, model, overlap=0.5
            )
            
            # Post-process to binary (0 or 1)
            val_outputs = [post_pred(i) for i in val_outputs]
            val_labels = [post_label(i) for i in val_labels]
            
            dice_metric(y_pred=val_outputs, y=val_labels)

    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    return metric

def train():
    # --- CONFIGURATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths (Update if paths differ on cluster)
    TRAIN_P = r"C:\coca_project\data_canonical\tables\train_split.parquet"
    VAL_P = r"C:\coca_project\data_canonical\tables\val_split.parquet"
    
    # Scale parameters based on hardware
    if DEBUG:
        print("🛠️ RUNNING IN DEBUG MODE (ARM LAPTOP)")
        PATCH_SIZE = (128, 128, 3)
        BATCH_SIZE = 1  # Safer for 8GB shared memory
        NUM_EPOCHS = 10
        UNET_CHANNELS = (32, 64, 128) # Slim model
    else:
        print("🚀 RUNNING IN CLUSTER MODE (GPU CLUSTER)")
        PATCH_SIZE = (128, 128, 64) # Increased depth
        BATCH_SIZE = 4
        NUM_EPOCHS = 100
        UNET_CHANNELS = (32, 64, 128, 256, 512) # Full model

    # 2. LOADERS
    train_loader, val_loader = get_coca_loaders(TRAIN_P, VAL_P, PATCH_SIZE, BATCH_SIZE)

    if DEBUG:
        print("🔍 Testing loader output for binarization issues...")
    # Grab one sample from the loader
        test_batch = next(iter(val_loader))
        img_data = test_batch["image"][0][0].numpy()
    
        print(f"📊 Image Value Range: {img_data.min()} to {img_data.max()}")
        print(f"📊 Number of Unique Values: {len(np.unique(img_data))}")
    
        if len(np.unique(img_data)) < 10:
            print("❌ ERROR: Image is still binarized! Check ScaleIntensityRanged.")
        else:
            print("✅ Success: Image has grayscale gradients.")
# 3. VISUALIZATION (Only triggers in Debug)
    if DEBUG:
    
        print("🔍 Bypassing loader to find the first POSITIVE scan in the Parquet...")
        df_val = pd.read_parquet(VAL_P)
        pos_cases = df_val[df_val['has_pos'] == 1]
    
        if pos_cases.empty:
            print("❌ No positive cases found in the Validation Parquet!")
        else:
        # Grab the very first positive file path
            sample_row = pos_cases.iloc[6]
            print(f"✅ Found Positive Case: {sample_row['scan_id']}")
        
        # Load directly via SimpleITK to verify the file isn't corrupted/empty
            test_img = sitk.GetArrayFromImage(sitk.ReadImage(sample_row['image_path']))
            test_mask = sitk.GetArrayFromImage(sitk.ReadImage(sample_row['mask_path']))
        
            print(f"📊 Mask Max Value: {test_mask.max()}")
            print(f"📊 Mask Voxel Sum: {test_mask.sum()}")

            if test_mask.sum() == 0:
                print("❗ ALERT: Parquet says 'has_pos', but the NIfTI file is empty. Resampling failed.")
            else:
            # If the files are good, run the scroller
            # Note: SimpleITK reads as [Z, H, W], which SliceScroller expects
                tracker = SliceScroller(torch.from_numpy(test_img), torch.from_numpy(test_mask))
                tracker.ind = np.argmax(np.sum(test_mask, axis=(1, 2)))
                tracker.update()
                plt.show()
         

    # 4. MODEL, LOSS, OPTIMIZER
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=UNET_CHANNELS,
        strides=(2, 2,1),
        num_res_units=2,
    ).to(device)

    # Focal Loss handles the extreme sparsity of calcium voxels
    loss_func = DiceFocalLoss(
    sigmoid=True,
    gamma=2.0,      # Focal "bone-dimmer" power
    lambda_dice=1.0, # High weight to force intersection
    lambda_focal=.5, 
    alpha=.75
    # Optional: squared_pred=True can help "sharpen" the result by punishing 
    # mid-range probabilities (0.5) more than low (0.1) or high (0.9).
)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # 1. Load the existing weights
    model_path = "best_coca_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"🚀 Loaded existing weights from {model_path}. Continuing training...")
    else:
        print("🆕 No existing model found. Starting from scratch.")

# 2. Update your loop range so you don't overwrite previous epoch numbers
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        hallucination_count = 0
        success_count = 0
        step_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(step_bar):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            # DEBUG: Check if the patches actually have labels in them
            pixels_with_label = (labels > 0).sum().item()
            if pixels_with_label == 0:
                print(f"⚠️ Sampler Fail: Batch {i} contains ZERO label pixels! (Defaulted to random crop)")
            else:
                print(f"✅ Sampler Success: Batch {i} has {pixels_with_label} label pixels.")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # --- TRACKING LOGIC (Every Batch) ---
            conf_map = torch.sigmoid(outputs)
            for b in range(inputs.shape[0]): # Loop through images in batch
                max_val = conf_map[b].max().item()
                if max_val > 0.5: # If model thinks it found calcium
                    max_idx = torch.argmax(conf_map[b]).item()
                    coords = np.unravel_index(max_idx, conf_map[b].shape)
                    z_slice = coords[3] # Depth slice
                        
                    # Check if ground truth has calcium at that exact spot
                    if labels[b, 0, :, :, z_slice].max() > 0:
                            success_count += 1
                    else:
                        hallucination_count += 1

                # --- VISUAL CHECK (Every 10 Batches) ---
                if i % 10 == 0:
                    save_audit_image(inputs, labels, conf_map, i, epoch)

            epoch_loss += loss.item()
            step_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            conf_map = torch.sigmoid(outputs)

            # --- EPOCH SUMMARY ---
        print(f"\n--- Epoch {epoch} Report ---")
        print(f"Avg Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Total Confident Detections: {success_count + hallucination_count}")
        print(f"✅ Successes: {success_count} | ❌ Hallucinations: {hallucination_count}")
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Complete. Avg Loss: {avg_loss:.4f}")


# --- VALIDATION STEP ---
        val_dice = run_validation(model, val_loader, device, PATCH_SIZE)
        print(f"Epoch {epoch} Metrics: | Dice: {val_dice:.4f}")
        best_metric=0
        # Save if this is the best version yet
        if val_dice > best_metric:
            best_metric = val_dice
            torch.save(model.state_dict(), "best_coca_model.pth")
            print("🌟 New Best Model Saved!")

if __name__ == "__main__":
    train()