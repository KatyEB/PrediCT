import pandas as pd
df = pd.read_parquet(r"C:\coca_project\derived\calcium_rois_raw.parquet")
df.to_csv(r"C:\coca_project\data_canonical\tables\calcium_rois_raw.csv", index=False)
