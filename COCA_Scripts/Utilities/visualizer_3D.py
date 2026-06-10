import pyvista as pv
import numpy as np
import SimpleITK as sitk


import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# -------------------------
# LOAD CT
# -------------------------
ct_path = r"E:\MyProjects\Gsoc_2026_Official\data_canonical\images\75409e2f4265\75409e2f4265_img.nii.gz"

ct = sitk.ReadImage(ct_path)
ct_np = sitk.GetArrayFromImage(ct)

spacing = ct.GetSpacing()
spacing = (spacing[0], spacing[1], spacing[2])


# -------------------------
# CREATE CT VOLUME
# -------------------------
grid = pv.ImageData()

grid.dimensions = np.array(ct_np.shape[::-1]) + 1
grid.spacing = spacing

grid.cell_data["CT"] = ct_np.flatten(order="F")


# -------------------------
# LOAD SEG
# -------------------------
seg_path = r"E:\MyProjects\Gsoc_2026_Official\data_canonical\images\75409e2f4265\75409e2f4265_binary_seg.nii.gz"

seg = sitk.ReadImage(seg_path)
seg_np = sitk.GetArrayFromImage(seg)


# -------------------------
# PLOTTER
# -------------------------
plotter = pv.Plotter()


# CT VOLUME
plotter.add_volume(
    grid,
    scalars="CT",
    opacity="sigmoid",
    cmap="gray",
    shade=True,
    clim=(0, 300),   # soft tissue window
)


# -------------------------
# SEGMENTATION LABELS
# -------------------------
labels = np.unique(seg_np)
labels = labels[labels > 0]

colors = [
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "magenta",
]

for i, label in enumerate(labels):

    binary = (seg_np == label).astype(np.uint8)

    seg_grid = pv.ImageData()
    seg_grid.dimensions = np.array(binary.shape[::-1]) + 1
    seg_grid.spacing = spacing

    seg_grid.cell_data["mask"] = binary.flatten(order="F")

    seg_grid = seg_grid.cell_data_to_point_data()

    surface = seg_grid.contour([0.5])

    plotter.add_mesh(
        surface,
        color=colors[i % len(colors)],
        opacity=0.5,
    )


plotter.show()