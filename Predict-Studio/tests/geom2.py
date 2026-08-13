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
except Exception as e:
    traceback.print_exc()

