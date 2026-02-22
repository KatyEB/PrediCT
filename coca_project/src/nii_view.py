import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the file
img = nib.load(r'C:\coca_project\data_canonical\images\000a85335c17\000a85335c17_img.nii.gz')

# 2. Get the actual image data as a NumPy array
data = img.get_fdata()

# 3. Check the shape (e.g., 256, 256, 150)
print(f"Image shape: {data.shape}")

# 4. Visualize a single slice (the middle slice on the Z-axis)
middle_slice = data.shape[2] // 2
plt.imshow(data[:, :, middle_slice], cmap='gray')
plt.title(f"Slice {middle_slice}")
plt.show()

# 5. Access metadata (Header and Affine)
print(img.header)
print(img.affine)  # Translation/rotation matrix for world coordinates

# Get unique values in the data
unique_values = np.unique(data)

if len(unique_values) < 20: # Arbitrary threshold, segmentations rarely have many labels
    print(f"Likely a SEGMENTATION. Labels found: {unique_values}")
else:
    print(f"Likely an IMAGE. Total unique intensity values: {len(unique_values)}")
    print(f"Value range: {np.min(data)} to {np.max(data)}")