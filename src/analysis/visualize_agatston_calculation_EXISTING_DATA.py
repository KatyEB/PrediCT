import sys
import json
import plistlib
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
from pathlib import Path

def get_agatston_factor(max_hu):
    if max_hu < 130: return 0
    if max_hu < 200: return 1
    if max_hu < 300: return 2
    if max_hu < 400: return 3
    return 4

def shoelace_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def visualize_real_data(scan_id):
    csv_path = Path(r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\tables\scan_index_clean.csv")
    df = pd.read_csv(csv_path)
    row = df[df['scan_id'] == scan_id].iloc[0]
    patient_id = row['patient_id']
    
    xml_path = Path(rf"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\Gated_release_final\calcium_xml\{patient_id}.xml")
    
    ct_path = Path(rf"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\images\{scan_id}\{scan_id}_img.nii.gz")
    a1_path = Path(rf"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\images\{scan_id}\{scan_id}_seg.nii.gz")
    a3_path = Path(rf"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\images_cov\{scan_id}\{scan_id}_seg.nii.gz")
    
    ct_img = sitk.ReadImage(str(ct_path))
    ct_arr = sitk.GetArrayFromImage(ct_img)
    a1_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(a1_path)))
    a3_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(a3_path)))
    spacing = ct_img.GetSpacing()
    
    with open(xml_path, "rb") as f:
        data = plistlib.load(f)

    # Find slice with biggest polygon
    best_z = -1
    best_poly = None
    max_area = 0
    
    for img_entry in data.get("Images", []):
        z = int(img_entry.get("ImageIndex", -1))
        if z < 0 or z >= ct_arr.shape[0]: continue
        
        for roi in img_entry.get("ROIs", []):
            points_str = roi.get("Point_px", [])
            if not points_str: continue
            
            poly_points = []
            for p_str in points_str:
                cleaned = p_str.replace("(", "").replace(")", "")
                parts = cleaned.split(",")
                if len(parts) == 2:
                    poly_points.append([float(parts[0]), float(parts[1])])
            
            if poly_points:
                pts = np.array(poly_points)
                area = shoelace_area(pts[:, 0], pts[:, 1])
                if area > max_area:
                    max_area = area
                    best_z = z
                    best_poly = pts

    if best_poly is None:
        print("No calcium found.")
        return

    ct_slice = ct_arr[best_z]
    a1_slice = a1_arr[best_z]
    a3_slice = a3_arr[best_z]
    
    pad = 5
    min_x, max_x = int(np.floor(best_poly[:, 0].min())) - pad, int(np.ceil(best_poly[:, 0].max())) + pad
    min_y, max_y = int(np.floor(best_poly[:, 1].min())) - pad, int(np.ceil(best_poly[:, 1].max())) + pad
    
    # Agatston Calculations
    voxel_area = spacing[0] * spacing[1]
    
    xml_area_px = shoelace_area(best_poly[:, 0], best_poly[:, 1])
    xml_area_mm = xml_area_px * voxel_area
    # For XML max HU, we construct a temp mask since we don't have the official one
    temp_mask = np.zeros_like(ct_slice, dtype=np.uint8)
    cv2.fillPoly(temp_mask, [np.round(best_poly).astype(np.int32)], 1)
    xml_max_hu = np.max(ct_slice[temp_mask == 1])
    xml_factor = get_agatston_factor(xml_max_hu)
    xml_agatston = xml_area_mm * xml_factor
    
    a1_area_px = np.sum(a1_slice)
    a1_area_mm = a1_area_px * voxel_area
    a1_max_hu = np.max(ct_slice[a1_slice == 1]) if a1_area_px > 0 else 0
    a1_factor = get_agatston_factor(a1_max_hu)
    a1_agatston = a1_area_mm * a1_factor
    
    a3_area_px = np.sum(a3_slice[a3_slice > 0])
    a3_area_mm = a3_area_px * voxel_area
    a3_max_hu = np.max(ct_slice[a3_slice > 0.1]) if np.sum(a3_slice > 0.1) > 0 else 0
    a3_factor = get_agatston_factor(a3_max_hu)
    a3_agatston = a3_area_mm * a3_factor

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor='#111111')
    
    titles = [
        "1. Raw CT Slice\n",
        f"2. Original XML Polygon\nArea: {xml_area_px:.2f} px | HU: {xml_max_hu} (F: {xml_factor})\nScore: {xml_agatston:.2f}",
        f"3. Approach 1 (Binary Real Data)\nArea: {a1_area_px:.2f} px | HU: {a1_max_hu} (F: {a1_factor})\nScore: {a1_agatston:.2f}",
        f"4. Approach 3 (Soft Real Data)\nArea: {a3_area_px:.2f} px | HU: {a3_max_hu} (F: {a3_factor})\nScore: {a3_agatston:.2f}"
    ]
    
    for ax in axes:
        ax.set_facecolor('#111111')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
            
    # 1. CT
    im0 = axes[0].imshow(ct_slice[min_y:max_y, min_x:max_x], cmap='gray', vmin=-150, vmax=500)
    axes[0].set_title(titles[0], color='white', fontsize=12)
    
    # Overlay pixel boundaries lightly
    for y in range(max_y - min_y):
        axes[0].axhline(y - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    for x in range(max_x - min_x):
        axes[0].axvline(x - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        
    for y in range(max_y - min_y):
        for x in range(max_x - min_x):
            hu = ct_slice[min_y+y, min_x+x]
            if hu > 130:
                axes[0].text(x, y, str(int(hu)), ha='center', va='center', color='cyan', fontsize=8)

    # 2. XML Polygon
    axes[1].imshow(ct_slice[min_y:max_y, min_x:max_x], cmap='gray', vmin=-150, vmax=500)
    shifted_poly = best_poly.copy()
    shifted_poly[:, 0] -= min_x
    shifted_poly[:, 1] -= min_y
    poly_patch = Polygon(shifted_poly, closed=True, fill=False, edgecolor='lime', linewidth=2)
    axes[1].add_patch(poly_patch)
    axes[1].set_title(titles[1], color='white', fontsize=12)

    # 3. A1 Binary
    axes[2].imshow(ct_slice[min_y:max_y, min_x:max_x], cmap='gray', vmin=-150, vmax=500)
    a1_crop = a1_slice[min_y:max_y, min_x:max_x]
    axes[2].imshow(np.ma.masked_where(a1_crop == 0, a1_crop), cmap='autumn', alpha=0.6)
    axes[2].set_title(titles[2], color='white', fontsize=12)
    for y in range(max_y - min_y):
        for x in range(max_x - min_x):
            val = a1_crop[y, x]
            if val > 0:
                axes[2].text(x, y, "1", ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    poly_patch_3 = Polygon(shifted_poly, closed=True, fill=False, edgecolor='lime', linewidth=1, linestyle='--')
    axes[2].add_patch(poly_patch_3)

    # 4. A3 Soft
    axes[3].imshow(ct_slice[min_y:max_y, min_x:max_x], cmap='gray', vmin=-150, vmax=500)
    a3_crop = a3_slice[min_y:max_y, min_x:max_x]
    im3 = axes[3].imshow(np.ma.masked_where(a3_crop == 0, a3_crop), cmap='viridis', alpha=0.8, vmin=0, vmax=1)
    axes[3].set_title(titles[3], color='white', fontsize=12)
    for y in range(max_y - min_y):
        for x in range(max_x - min_x):
            val = a3_crop[y, x]
            if val > 0.01:
                col = 'black' if val > 0.6 else 'white'
                txt = f"{val:.1f}".lstrip('0')
                if txt == '.0': txt = '.1' 
                axes[3].text(x, y, txt, ha='center', va='center', color=col, fontsize=7, fontweight='bold')
    poly_patch_4 = Polygon(shifted_poly, closed=True, fill=False, edgecolor='lime', linewidth=1, linestyle='--')
    axes[3].add_patch(poly_patch_4)

    fig.suptitle(f"Patient {patient_id} (Scan {scan_id}) | Slice Z={best_z} | Spacing: {spacing[0]:.2f}x{spacing[1]:.2f}mm", color='white', fontsize=16)
    
    out_dir = Path(__file__).resolve().parents[2] / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"agatston_visual_calculation_REAL_{scan_id}_z{best_z}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight', facecolor='#111111')
    print(f"Saved numerical visualization to {out_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        scan_id = sys.argv[1]
    else:
        scan_id = "2ba31806bf79"
    visualize_real_data(scan_id)
