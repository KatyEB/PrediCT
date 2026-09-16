import pandas as pd
import os
from pathlib import Path

tables_dir = Path("/pscratch/sd/s/soham95/SOHAM/coca_raw/cocacoronarycalciumandchestcts-2/data_canonical/tables")
patients_dir = Path("/pscratch/sd/s/soham95/predict_software/Predict-Studio/data/Gated_release_final/patient")
dest_dir = Path("/pscratch/sd/s/soham95/predict_software/Predict-Studio/data/raw")
dest_dir.mkdir(parents=True, exist_ok=True)

splits = ["train_split_clean.parquet", "val_split_clean.parquet", "test_split_clean.parquet"]
patient_ids = set()

for s in splits:
    df = pd.read_parquet(tables_dir / s)
    patient_ids.update(df['patient_id'].tolist())

print(f"Found {len(patient_ids)} unique patients across splits. Creating symlinks...")
count = 0
for pid in patient_ids:
    src = patients_dir / str(pid)
    dst = dest_dir / str(pid)
    if src.exists():
        if not dst.exists():
            os.symlink(src, dst)
            count += 1
    else:
        print(f"Warning: Source {src} does not exist.")

print(f"Symlinked {count} patient folders to {dest_dir}")
