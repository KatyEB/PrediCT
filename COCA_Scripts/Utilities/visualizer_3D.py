import pyvista as pv
import numpy as np
import SimpleITK as sitk

mask_path = r"E:\MyProjects\Gsoc_2026_Official\heart_mask.nii.gz"

mask = sitk.ReadImage(mask_path)
mask_np = sitk.GetArrayFromImage(mask)

print("Shape:", mask_np.shape)
print("Unique values:", np.unique(mask_np))

grid = pv.ImageData()
grid.dimensions = np.array(mask_np.shape[::-1]) + 1
grid.spacing = mask.GetSpacing()

grid.cell_data["mask"] = mask_np.flatten(order="F")

surface = (
    grid.cell_data_to_point_data()
    .contour([0.5])
)

plotter = pv.Plotter()
plotter.add_mesh(surface, color="red")
plotter.show()