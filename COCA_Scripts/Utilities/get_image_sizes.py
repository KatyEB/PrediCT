import os
from pathlib import Path
import nibabel as nib
import pandas as pd

root_dir = r"E:\MyProjects\Gsoc_2026_Official\data_resampled"

results = []

for patient_dir in Path(root_dir).iterdir():
    if not patient_dir.is_dir():
        continue

    img_path = patient_dir / f"{patient_dir.name}_img.nii.gz"

    if not img_path.exists():
        print(f"Missing: {img_path}")
        continue

    try:
        img = nib.load(str(img_path))
        shape = img.shape

        results.append({
            "patient_id": patient_dir.name,
            "X": shape[0],
            "Y": shape[1],
            "Z": shape[2]
        })

    except Exception as e:
        print(f"Error reading {img_path}: {e}")

df = pd.DataFrame(results)

print(df)
print("\nSummary:")
print(df[["X", "Y", "Z"]].describe())

df.to_csv("image_sizes.csv", index=False)