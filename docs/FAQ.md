# Frequently Asked Questions (FAQ)

## Q: Do the NIfTI folders represent just one slice or the whole scan? Why do the generated figures only show one slice?

**A:** That is a very good question! It is a common point of confusion when working with 3D medical imaging data. 

To answer your question directly: Those folders do **NOT** represent just one slice. They represent the **entire 3D scan (all slices)** for that patient.

### Here is exactly how it works:

#### 1. NIfTI Files are 3D
Inside the patient folder (e.g., `0b137435bb47`), there are files named `0b137435bb47_img.nii.gz` and `0b137435bb47_seg.nii.gz`. These are NIfTI files. NIfTI files are 3D volumes. They contain a stack of all the 2D slices for that scan combined into a single 3D array (Width × Height × Depth).

#### 2. Training Data
When the 3D UNet model trains, it loads that entire `.nii.gz` file and trains on the full 3D volume (or 3D crops of it), which includes all slices containing calcium and all slices without calcium.

#### 3. Why the script only shows one slice
The reason you are only seeing one slice in the generated figures is because of how the visualization scripts (`visualize_predictions_comparison.py` or `visualize_agatston_calculation_EXISTING_DATA.py`) are written.

If you look at the visualization script, you will see a block of code that loops through all the slices, calculates the area of every single calcium polygon it finds, and keeps track of only the largest one:

```python
# Find slice with biggest polygon
best_z = -1
best_poly = None
max_area = 0

for img_entry in data.get("Images", []):
    z = int(img_entry.get("ImageIndex", -1))
    # ... logic to calculate area of polygon ...
    if area > max_area:
        max_area = area
        best_z = z
        best_poly = pts
```

After finding the slice (`best_z`) with the biggest piece of calcium, the script extracts just that single slice from the 3D NIfTI array:

```python
ct_slice = ct_arr[best_z]
a1_slice = a1_arr[best_z]
```

...and plots it. It does this purely because it is hard to plot an entire 3D volume in a 2D image file, so it picks the "most interesting" slice (the one with the largest deposit) to visualize the comparison between Approaches.

So don't worry, **no data is being lost!** Your processed dataset inside `images_roi_cov` correctly contains all slices for every patient.
