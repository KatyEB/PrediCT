# save as: visualize_masks.py
# run with: python visualize_masks.py

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def visualize_patient(scan_id: str, data_root: str):
    base = Path(data_root) / "data_canonical" / "images" / scan_id

    img_path  = base / f"{scan_id}_img.nii.gz"
    mask_path = base / f"{scan_id}_seg.nii.gz"

    # Load
    img_arr  = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))   # (Z, Y, X) HU values
    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))  # (Z, Y, X) binary 0/1

    # Find slices that have calcium
    calcium_slices = np.where(mask_arr.sum(axis=(1,2)) > 0)[0]

    if len(calcium_slices) == 0:
        print(f"[{scan_id}] No calcium found in mask.")
        return

    print(f"[{scan_id}] Calcium on {len(calcium_slices)} slices: {calcium_slices.tolist()}")

    # Show up to 6 slices
    show_slices = calcium_slices[:6]
    n = len(show_slices)

    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]  # make iterable

    fig.suptitle(f"Patient scan: {scan_id}", fontsize=14, fontweight='bold')

    for row, z in enumerate(show_slices):
        ct_slice   = img_arr[z]
        mask_slice = mask_arr[z]

        # Window CT to cardiac range (HU -100 to 400 shows calcium well)
        ct_display = np.clip(ct_slice, -100, 400)

        # Left: raw CT
        axes[row][0].imshow(ct_display, cmap='gray', origin='lower')
        axes[row][0].set_title(f"Slice z={z} — CT only")
        axes[row][0].axis('off')

        # Right: CT + mask overlay
        axes[row][1].imshow(ct_display, cmap='gray', origin='lower')

        # Overlay mask in orange with transparency
        mask_rgba = np.zeros((*mask_slice.shape, 4))
        mask_rgba[mask_slice == 1] = [1.0, 0.5, 0.0, 0.6]  # orange, 60% opacity
        axes[row][1].imshow(mask_rgba, origin='lower')

        axes[row][1].set_title(f"Slice z={z} — CT + calcium mask")
        axes[row][1].axis('off')

        patch = mpatches.Patch(color='orange', alpha=0.6, label='Calcium (GT mask)')
        axes[row][1].legend(handles=[patch], loc='lower right', fontsize=8)

    plt.tight_layout()
    out_path = f"figures/{scan_id}_mask_check.png"
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  → Saved: {out_path}")
    #plt.show()


# ── Run on all 3 test patients ──────────────────────────────────────
DATA_ROOT = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2"

for scan_id in ["2740d96a230c", "fd14b377bebc", "c3be56167c58"]:
    visualize_patient(scan_id, DATA_ROOT)