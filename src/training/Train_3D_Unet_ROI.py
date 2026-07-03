r"""
train_unet_roi.py — PrediCT CAC Segmentation — 3D U-Net on ROI-cropped data
============================================================================
Identical to the full-volume baseline (Train_3D_Unet.py) except:
  - data_root points at TotalSegmentator-cropped volumes (images_roi/)
  - SpatialPadd added: ROI crops can be smaller than the patch size in one
    dimension (small hearts) -- this guarantees the volume always fits
  - separate cache dirs

Everything else (loss, HU window, architecture, LR, patience) is
UNCHANGED from the full-volume run, so any Dice difference is
attributable to ROI cropping alone, not a confound of multiple changes.

NOTE: this Dice is NOT directly comparable to the full-volume baseline
(mean 0.61 / median 0.69) -- ROI-cropped inference never sees ribs/spine,
so it's measuring an easier task by construction. Report both, labeled.

USAGE (PowerShell):
  & c:\SOHAM\vmenv\Scripts\python.exe c:/SOHAM/Train_3D_Unet_ROI.py `
    --data_root "C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\images_roi" `
    --train_csv "C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\tables\train_split.parquet" `
    --val_csv   "C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\tables\val_split.parquet" `
    --out_dir   "C:\SOHAM\runs\approach1_roi_cropped"
"""

import argparse
import json
import time
import csv
from pathlib import Path

import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.data import DataLoader, PersistentDataset, decollate_batch
from monai.networks.nets import UNet
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, EnsureTyped,
    Activations, AsDiscrete,
)

# ── Config (L4 24GB) — identical to the full-volume baseline ──────
HU_MIN, HU_MAX  = 0, 1200
PATCH_SIZE      = (96, 96, 32)
POS_NEG_RATIO   = (1, 1)
SAMPLES_PER_VOL = 8
BATCH_SIZE      = 4
NUM_WORKERS     = 4
SW_OVERLAP      = 0.5

# Separate cache from the full-volume run -- different data, kept distinct.
CACHE_DIR_TRAIN = r"C:\SOHAM\cache\train_roi"
CACHE_DIR_VAL   = r"C:\SOHAM\cache\val_roi"


def build_data_list(csv_path, data_root, label_suffix):
    csv_path = Path(csv_path)
    df = pd.read_parquet(csv_path) if csv_path.suffix == ".parquet" else pd.read_csv(csv_path)
    id_col = next((c for c in ["scan_id", "patient_id", "id"] if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"No scan_id column in {csv_path}. Columns: {list(df.columns)}")

    data_root = Path(data_root)
    items = []
    for sid in df[id_col].astype(str):
        img = data_root / sid / f"{sid}_img.nii.gz"
        lbl = data_root / sid / f"{sid}{label_suffix}.nii.gz"
        if img.exists() and lbl.exists():
            items.append({"image": str(img), "label": str(lbl)})
        else:
            print(f"  [skip] missing files for {sid}")
    print(f"  Loaded {len(items)} samples from {csv_path.name}")
    return items


def train_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # NEW: guarantees the volume is at least PATCH_SIZE before sampling.
        # ROI crops for small-heart patients can be thinner than 32 slices
        # in z -- without this, patch sampling would fail on those patients.
        # No-op (zero padding) on any patient whose crop is already larger.
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=PATCH_SIZE,
            pos=POS_NEG_RATIO[0], neg=POS_NEG_RATIO[1],
            num_samples=SAMPLES_PER_VOL,
            image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"]),
    ])


def val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Same safety pad -- sliding_window_inference technically handles
        # undersized inputs internally, but this makes it explicit rather
        # than relying on implicit behavior.
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),
        EnsureTyped(keys=["image", "label"]),
    ])


def build_model(device):
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, dropout=0.1,
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--label_suffix", default="_seg")
    ap.add_argument("--epochs",       type=int,   default=200)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--val_interval", type=int,   default=2)
    ap.add_argument("--patience",     type=int,   default=30)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Building datasets...")
    train_items = build_data_list(args.train_csv, args.data_root, args.label_suffix)
    val_items   = build_data_list(args.val_csv,   args.data_root, args.label_suffix)

    train_ds = PersistentDataset(train_items, train_transforms(), cache_dir=CACHE_DIR_TRAIN)
    val_ds   = PersistentDataset(val_items,   val_transforms(),   cache_dir=CACHE_DIR_VAL)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=2, pin_memory=True,
        persistent_workers=True,
    )

    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    loss_fn   = TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    post_pred  = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=True, reduction="none")

    log_path = out / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_dice_mean", "val_dice_median", "lr", "epoch_time_s"])

    best_dice, best_epoch, epochs_no_improve = -1.0, -1, 0
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        model.train()
        epoch_loss, n_steps, n_skipped = 0.0, 0, 0

        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)

            if not torch.isfinite(loss):
                n_skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

        scheduler.step()
        epoch_loss = epoch_loss / n_steps if n_steps else float("nan")
        ep_time = time.time() - ep_start

        val_mean, val_median = float("nan"), float("nan")
        if epoch % args.val_interval == 0:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for vbatch in val_loader:
                    vimg = vbatch["image"].to(device)
                    vlbl = vbatch["label"].to(device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        vout = sliding_window_inference(
                            vimg, PATCH_SIZE, sw_batch_size=4,
                            predictor=model, overlap=SW_OVERLAP)
                    vout = [post_pred(x)  for x in decollate_batch(vout)]
                    vlbl = [post_label(x) for x in decollate_batch(vlbl)]
                    dice_metric(y_pred=vout, y=vlbl)

            scores = dice_metric.aggregate()
            if isinstance(scores, (list, tuple)):
                scores = scores[0]
            scores = scores.flatten()
            valid = scores[torch.isfinite(scores)]
            if valid.numel() > 0:
                val_mean   = valid.mean().item()
                val_median = valid.median().item()

            if val_mean > best_dice:
                best_dice, best_epoch, epochs_no_improve = val_mean, epoch, 0
                torch.save(model.state_dict(), out / "best_model.pth")
                print(f"  * new best mean Dice {val_mean:.4f} @ epoch {epoch}")
            else:
                epochs_no_improve += args.val_interval

        lr_now = optimizer.param_groups[0]["lr"]
        skip_msg = f" | skipped {n_skipped}" if n_skipped else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | loss {epoch_loss:.4f} | "
              f"dice {val_mean:.4f} (med {val_median:.4f}) | "
              f"lr {lr_now:.2e} | {ep_time:.1f}s{skip_msg}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{epoch_loss:.5f}", f"{val_mean:.5f}",
                 f"{val_median:.5f}", f"{lr_now:.2e}", f"{ep_time:.1f}"])

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    total_time = time.time() - train_start
    summary = {
        "best_val_dice_mean": round(best_dice, 4),
        "best_epoch": best_epoch,
        "total_training_time_min": round(total_time / 60, 1),
        "model_params_M": round(n_params / 1e6, 2),
        "epochs_run": epoch,
        "loss": "TverskyLoss(alpha=0.3, beta=0.7)",
        "hu_window": [HU_MIN, HU_MAX],
        "data": "ROI-cropped (TotalSegmentator heart + 8mm margin)",
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print("\n=== TRAINING COMPLETE ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    try:
        df = pd.read_csv(log_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(df["epoch"], df["train_loss"], color="#CC4444", linewidth=1.5)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
        axes[0].set_title("Training loss (ROI-cropped)"); axes[0].grid(alpha=0.3)

        v = df[df["val_dice_mean"].notna()]
        axes[1].plot(v["epoch"], v["val_dice_mean"],   color="#378ADD",
                     marker="o", markersize=3, linewidth=1.5, label="mean")
        axes[1].plot(v["epoch"], v["val_dice_median"], color="#2D7D46",
                     marker="o", markersize=3, linewidth=1.5, label="median")
        axes[1].axhline(best_dice, color="#888888", linestyle="--", linewidth=1,
                        label=f"best mean = {best_dice:.4f} @ ep {best_epoch}")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val Dice")
        axes[1].set_title("Validation Dice (ROI-cropped)"); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(out / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved training curves -> {out / 'training_curves.png'}")
    except Exception as e:
        print(f"  Could not generate plots: {e}")


if __name__ == "__main__":
    main()