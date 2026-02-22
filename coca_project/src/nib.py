import nibabel as nib
import numpy as np

# Pick one of your resampled image paths from the parquet
img_path = r"C:\coca_project\data_resampled\000a85335c17\000a85335c17_seg.nii.gz"
img = nib.load(img_path).get_fdata()

print(f"Raw Min: {img.min()}")
print(f"Raw Max: {img.max()}")
print(f"Raw Mean: {img.mean()}")
print(f"Data Type: {img.dtype}")