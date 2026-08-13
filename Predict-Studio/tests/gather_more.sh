#!/bin/bash
cd /pscratch/sd/s/soham95/predict_software/Predict-Studio

echo "=== 4. A REAL OUTPUT FOLDER (REMAINDER) ==="
if [ -f "data/out/172/a1-roi/slices.json" ]; then
    echo "--- data/out/172/a1-roi/slices.json (first 3 entries) ---"
    head -n 25 "data/out/172/a1-roi/slices.json"
    echo "Total entries in slices.json: $(grep -c '"idx":' "data/out/172/a1-roi/slices.json")"
fi

cat << 'PY_EOF' > geom.py
import sys
import numpy as np
import SimpleITK as sitk
from PIL import Image
import traceback

print("\n=== 5. GEOMETRY ===")
try:
    img = sitk.ReadImage('data/out/172/a1-roi/ct.nii.gz')
    print(f"GetSize(): {img.GetSize()}")
    print(f"GetSpacing(): {img.GetSpacing()}")
    print(f"GetOrigin(): {img.GetOrigin()}")
    print(f"GetDirection(): {img.GetDirection()}")
    print(f"Orientation: {sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())}")
    
    arr = sitk.GetArrayFromImage(img)
    print(f"Array shape: {arr.shape}")
    
    png = Image.open('data/out/172/a1-roi/slices/ct/slice_025.png')
    print(f"PNG size: {png.size}")
    
    nz, ny, nx = arr.shape
    w, h = png.size
    print(f"Explicitly: Array (nz={nz}, ny={ny}, nx={nx}). PNG (w={w}, h={h}).")
    if (w, h) == (nx, ny):
        print("PNG (width, height) equals array (nx, ny).")
    elif (w, h) == (ny, nx):
        print("PNG (width, height) equals array (ny, nx).")
    else:
        print("PNG (width, height) does not match array (nx, ny) or (ny, nx).")
except Exception as e:
    traceback.print_exc()

print("\n=== 4. COVERAGE ALPHA CHECK ===")
try:
    a3_mask_path = 'data/out/172/a3-coverage/slices/mask/slice_020.png'
    # Try any coverage slice if exists
    import glob
    cov_masks = glob.glob('data/out/*/*coverage*/slices/mask/*.png')
    if cov_masks:
        mask_path = cov_masks[0]
        im = Image.open(mask_path)
        print(f"Found mask: {mask_path}")
        print(f"Mode: {im.mode}")
        print(f"Size: {im.size}")
        a = np.array(im)[:, :, 3]
        print(f"np.unique(alpha).size: {np.unique(a).size}")
    else:
        print("No coverage run with slices exists.")
except Exception as e:
    traceback.print_exc()
PY_EOF
python geom.py

echo -e "\n=== 6. THE FLIP ==="
echo "Lines in render.py that flip:"
cat -n src/render.py | grep -C 1 flipud || true
echo "Lines in scoring.py that compute centroid_y / bbox_y*:"
cat -n src/scoring.py | grep -E "centroid_y|bbox_y" || true

echo "Grep for centroid_x / centroid_y / bbox_x0 across BOTH repos:"
grep -rnE "centroid_x|centroid_y|bbox_x0" /pscratch/sd/s/soham95/predict_software/Predict-Studio /pscratch/sd/s/soham95/SOHAM || true

echo -e "\n=== 7. WHAT DISAGREES WITH THE DOCS ==="
echo "Grep for [100, 1000]:"
grep -rnE "\[100,\s*1000\]" /pscratch/sd/s/soham95/predict_software/Predict-Studio /pscratch/sd/s/soham95/SOHAM || true

echo "Grep for approach3_coverage (v1):"
grep -rn "approach3_coverage" /pscratch/sd/s/soham95/predict_software/Predict-Studio /pscratch/sd/s/soham95/SOHAM | grep -v "v2" || true

echo "Finding dataset_splitter.py and its contents:"
ds_path=$(find /pscratch/sd/s/soham95 -name "dataset_splitter.py" 2>/dev/null | head -1)
if [ -n "$ds_path" ]; then
    echo "Found dataset_splitter.py at $ds_path"
    echo "Splits logic / counts / exclusions in dataset_splitter.py:"
    grep -E "train|val|test|split|exclude|exclude_list" "$ds_path" -C 2 || true
    cat "$ds_path" | grep -A 10 "exclude" || true
else
    echo "dataset_splitter.py does not exist."
fi

echo -e "\n=== 8. THE 172 BASELINE ==="
echo "Grep for 1064.1237:"
grep -rn "1064.1237" /pscratch/sd/s/soham95/predict_software/Predict-Studio /pscratch/sd/s/soham95/SOHAM || true
} > report_more.txt
