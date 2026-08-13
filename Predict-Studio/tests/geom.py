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
