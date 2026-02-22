import pandas as pd
df = pd.read_parquet(r"C:\coca_project\data_canonical\tables\metadata_summary.parquet")
df.to_csv(r"C:\coca_project\data_canonical\tables\metadata_summary.csv", index=False)
