import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from monai.networks.nets import UNet
from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from data.big_data_loader import get_coca_loaders


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DEBUG       = True    # True = ARM laptop / fast iteration. False = cluster.
MODEL_SIZE  = "heavy" # "light" or "heavy"

TRAIN_P = r"C:\coca_project\data_canonical\tables\train_split.parquet"
VAL_P   = r"C:\coca_project\data_canonical\tables\val_split.parquet"
MODEL_PATH  = "best_coca_model.pth"

# Hardware-scaled parameters
if DEBUG:
    PATCH_SIZE  = (96, 96, 16)   # 3 slices is too thin for sliding window
    BATCH_SIZE  = 1
    NUM_EPOCHS  = 5
    AUDIT_EVERY = 2              # save audit image every N batches
else:
    PATCH_SIZE  = (128, 128, 64)
    BATCH_SIZE  = 4
    NUM_EPOCHS  = 100
    AUDIT_EVERY = 10

# Model architecture presets
MODEL_CONFIGS = {
    "light": {
        "channels": (16, 32, 64, 128),
        "strides":  (2, 2, 2),
        "num_res_units": 1,
    },
    "heavy": {
        "channels": (32, 64, 128, 256, 512),
        "strides":  (2, 2, 2, 2),
        "num_res_units": 2,
    },
}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# VISUALISATION TOOLS
# ---------------------------------------------------------------------------

def save_audit_image(inputs, labels, conf_map, batch_idx, epoch, item_idx=0):
    """
    Save a 3-panel audit image for one item in the batch.
    Finds the z-slice where the model is most confident and shows:
      CT | Ground truth overlay | Prediction confidence overlay
    """
    conf   = conf_map[item_idx]                          # (1, H, W, D)
    z      = int(np.unravel_index(conf.cpu().numpy().argmax(), conf.shape)[3])

    img_s  = inputs   [item_idx, 0, :, :, z].cpu().numpy()
    gt_s   = labels   [item_idx, 0, :, :, z].cpu().numpy()
    pred_s = conf_map [item_idx, 0, :, :, z].detach().cpu().numpy()

    gt_present = gt_s.max() > 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.axis("off")

    axes[0].imshow(img_s,  cmap="gray");  axes[0].set_title(f"CT  z={z}")
    axes[1].imshow(img_s,  cmap="gray")
    axes[1].imshow(np.ma.masked_where(gt_s == 0, gt_s), cmap="Reds", alpha=0.6)
    axes[1].set_title(f"GT  {'PRESENT' if gt_present else 'absent'}")
    axes[2].imshow(img_s,  cmap="gray")
    axes[2].imshow(pred_s, cmap="hot",  alpha=0.6)
    axes[2].set_title(f"Pred  max={pred_s.max():.2f}")

    match = "✅" if gt_present else "❌"
    plt.suptitle(f"Epoch {epoch} | Batch {batch_idx} | Item {item_idx} | GT match: {match}", fontsize=14)

    os.makedirs("audit_images", exist_ok=True)
    plt.savefig(f"audit_images/epoch{epoch:03d}_batch{batch_idx:04d}_item{item_idx}.png", dpi=80)
    plt.close(fig)


class SliceScroller:
    """Interactive matplotlib scroller. Scroll wheel moves through z-slices."""

    def __init__(self, image, mask):
        self.image  = image.cpu().numpy() if torch.is_tensor(image) else image
        self.mask   = mask.cpu().numpy()  if torch.is_tensor(mask)  else mask
        self.slices = self.image.shape[0]
        self.ind    = self.slices // 2

        v_min, v_max = np.percentile(self.image, [1, 99])
        if v_min == v_max:
            v_min, v_max = self.image.min(), self.image.max()
        self.v_min, self.v_max = v_min, v_max

        print(f"  Dynamic range: {v_min:.4f} → {v_max:.4f}  |  slices: {self.slices}")

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.im      = self.ax.imshow(self.image[self.ind], cmap="gray",
                                      vmin=v_min, vmax=v_max)
        masked       = np.ma.masked_where(self.mask[self.ind] == 0, self.mask[self.ind])
        self.mask_im = self.ax.imshow(masked, cmap="spring", alpha=0.8)
        self._update_title()

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_scroll(self, event):
        self.ind = (self.ind + (1 if event.button == "up" else -1)) % self.slices
        self._redraw()

    def _on_key(self, event):
        if event.key in ("up", "right"):   self.ind = (self.ind + 1) % self.slices
        elif event.key in ("down", "left"): self.ind = (self.ind - 1) % self.slices
        self._redraw()

    def _redraw(self):
        self.im.set_data(self.image[self.ind])
        self.im.set_clim(self.v_min, self.v_max)
        masked = np.ma.masked_where(self.mask[self.ind] == 0, self.mask[self.ind])
        self.mask_im.set_data(masked)
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        has_label = self.mask[self.ind].max() > 0
        self.ax.set_title(f"Slice {self.ind}/{self.slices-1}  "
                          f"{'🔴 calcium' if has_label else '⬜ no label'}")


def debug_loader_sanity(val_loader):
    """
    Quick checks on one batch pulled through the full transform pipeline.
    This reflects what the model actually sees — NOT the raw HU values on disk.
    Raw CT files will show ~-1024 to 3000 HU range — that is completely normal.
    Expected range after ScaleIntensityRanged: 0.0 to 1.0.
    """
    print("\n── Loader sanity check ──")
    print("  (values below are POST-transform — what the model sees, not raw HU)")
    batch    = next(iter(val_loader))
    img_data = batch["image"][0, 0].numpy()

    i_min, i_max = img_data.min(), img_data.max()
    n_unique     = len(np.unique(img_data))
    print(f"  Image range : {i_min:.4f} → {i_max:.4f}")
    print(f"  Unique vals : {n_unique}")

    if n_unique < 10:
        print("  ❌ Image looks binarised — check ScaleIntensityRanged in dataloader")
    elif i_max > 1.05:
        print("  ❌ Values exceed 1.0 after transforms — ScaleIntensityRanged clip=True may not be active")
    elif i_min < -0.05:
        print("  ⚠️  Values below 0.0 — check a_min in ScaleIntensityRanged")
    else:
        print("  ✅ Intensity looks correct (0→1)")

    lbl_sum = batch["label"][0, 0].sum().item()
    print(f"  Label voxels in first batch: {int(lbl_sum)}")
    if lbl_sum == 0:
        print("  ⚠️  No label voxels — sampler may have returned an all-negative patch")


def debug_visualise_positive(val_parquet: str):
    """Load the 7th positive case directly (bypassing the loader) and launch scroller."""
    print("\n── Direct file visualisation ──")
    df = pd.read_parquet(val_parquet)

    # Graceful fallback if parquet was generated before the new processor added
    # has_calcium — derive it from the voxels column, or scan the mask files.
    if "has_calcium" not in df.columns:
        print("  ⚠️  'has_calcium' column missing — parquet needs regenerating with "
              "the new COCA_processor. Deriving from 'voxels' column as fallback...")
        if "voxels" in df.columns:
            df["has_calcium"] = df["voxels"] > 0
        else:
            print("  ❌ No 'voxels' column either — cannot determine positives. "
                  "Please re-run COCA_processor.py to regenerate the parquet.")
            return

    pos_cases = df[df["has_calcium"]]

    if pos_cases.empty:
        print("  ❌ No positive cases in val parquet")
        return

    row = pos_cases.iloc[min(6, len(pos_cases) - 1)]
    print(f"  Scan: {row['scan_id']}")

    img_arr  = sitk.GetArrayFromImage(sitk.ReadImage(row["image_path"]))
    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(row["mask_path"]))

    print(f"  Mask max: {mask_arr.max()}  |  voxel sum: {int(mask_arr.sum())}")
    if mask_arr.sum() == 0:
        print("  ❗ Parquet says has_calcium but NIfTI is empty — resampling may have failed")
        return

    matplotlib.use("TkAgg")   # force interactive backend for scroll events
    scroller = SliceScroller(img_arr, mask_arr)
    scroller.ind = int(np.argmax(np.sum(mask_arr, axis=(1, 2))))
    scroller._redraw()
    plt.show()


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def run_validation(model, val_loader, device, patch_size):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post        = AsDiscrete(threshold=0.5)

    with torch.no_grad():
        for val_data in tqdm(val_loader, desc="  Validation", leave=False):
            images  = val_data["image"].to(device)
            labels  = val_data["label"].to(device)
            outputs = sliding_window_inference(images, patch_size, 4, model, overlap=0.5)
            dice_metric(
                y_pred=[post(i) for i in outputs],
                y     =[post(i) for i in labels],
            )

    score = dice_metric.aggregate().item()
    dice_metric.reset()
    return score


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Mode   : {'DEBUG' if DEBUG else 'CLUSTER'}")
    print(f"Model  : {MODEL_SIZE}")

    # ── Loaders ──────────────────────────────────────────────────────────────
    train_loader, val_loader = get_coca_loaders(TRAIN_P, VAL_P, PATCH_SIZE, BATCH_SIZE)

    # ── Debug checks (run before model init to fail fast) ────────────────────
    if DEBUG:
        debug_loader_sanity(val_loader)
        debug_visualise_positive(VAL_P)

    # ── Model ─────────────────────────────────────────────────────────────────
    cfg   = MODEL_CONFIGS[MODEL_SIZE]
    model = UNet(
        spatial_dims  = 3,
        in_channels   = 1,
        out_channels  = 1,
        channels      = cfg["channels"],
        strides       = cfg["strides"],
        num_res_units = cfg["num_res_units"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Loss & optimiser ──────────────────────────────────────────────────────
    loss_func = DiceFocalLoss(
        sigmoid      = True,
        gamma        = 2.0,
        lambda_dice  = 1.0,
        lambda_focal = 0.5,
        alpha        = 0.75,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    best_metric    = 0.0   # MUST be outside the epoch loop
    start_epoch    = 0

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        # Support both raw state_dict saves and richer checkpoint dicts
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            best_metric = ckpt.get("best_metric", 0.0)
            start_epoch = ckpt.get("epoch", 0) + 1
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded weights from {MODEL_PATH}  (best Dice so far: {best_metric:.4f})")
    else:
        print("No checkpoint found — training from scratch")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        model.train()
        epoch_loss          = 0.0
        success_count       = 0
        hallucination_count = 0

        step_bar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        for i, batch in enumerate(step_bar):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Warn if sampler failed to include any label voxels
            if DEBUG and (labels > 0).sum().item() == 0:
                print(f"  ⚠️  Batch {i}: zero label pixels (sampler fallback to random crop)")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            conf_map = torch.sigmoid(outputs.detach())
            epoch_loss += loss.item()
            step_bar.set_postfix(loss=f"{loss.item():.4f}")

            # Hallucination tracking — check every item in the batch
            for b in range(inputs.shape[0]):
                max_val = conf_map[b].max().item()
                if max_val > 0.5:
                    z = int(np.unravel_index(conf_map[b].cpu().numpy().argmax(),
                                             conf_map[b].shape)[3])
                    hit = labels[b, 0, :, :, z].max().item() > 0
                    if hit: success_count      += 1
                    else:   hallucination_count += 1

            # Audit images
            if i % AUDIT_EVERY == 0:
                for b in range(inputs.shape[0]):
                    save_audit_image(inputs, labels, conf_map, i, epoch, item_idx=b)

        # ── Epoch summary ─────────────────────────────────────────────────────
        avg_loss    = epoch_loss / len(train_loader)
        total_dets  = success_count + hallucination_count
        halluc_rate = hallucination_count / total_dets if total_dets > 0 else 0.0

        print(f"\nEpoch {epoch:03d}  loss={avg_loss:.4f}  "
              f"detections={total_dets}  "
              f"✅={success_count}  ❌={hallucination_count}  "
              f"halluc_rate={halluc_rate:.1%}")

        # ── Validation & checkpoint ───────────────────────────────────────────
        val_dice = run_validation(model, val_loader, device, PATCH_SIZE)
        print(f"           val_dice={val_dice:.4f}  best={best_metric:.4f}")

        if val_dice > best_metric:
            best_metric = val_dice
            torch.save(
                {
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "best_metric":  best_metric,
                    "patch_size":   PATCH_SIZE,
                    "model_size":   MODEL_SIZE,
                },
                MODEL_PATH,
            )
            print("  🌟 New best — checkpoint saved")


if __name__ == "__main__":
    train()
