
import pandas as pd
from pathlib import Path

DATA_ROOT = Path(r'C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2')
XML_ROOT  = DATA_ROOT / 'Gated_release_final' / 'calcium_xml'
TABLES    = DATA_ROOT / 'data_canonical' / 'tables'

df = pd.read_csv(TABLES / 'scan_index.csv')
print(f'Raw            : {len(df)} scans')

# Keep only xml-annotated gated scans
df['has_xml'] = df['patient_id'].astype(str).apply(
    lambda p: (XML_ROOT / f'{p}.xml').exists()
)
df = df[df['has_xml']].drop(columns=['has_xml'])
print(f'After XML filter: {len(df)} scans')

# Remove patient 263 (known bad mask)
df = df[df['patient_id'] != 263].reset_index(drop=True)
print(f'After P263 remove: {len(df)} scans')

# Deduplicate patients 700 and 726
df = (df.sort_values('voxels', ascending=False)
        .drop_duplicates('patient_id', keep='first')
        .reset_index(drop=True))
print(f'After dedup    : {len(df)} scans')
print(f'Positive       : {df["has_calcium"].sum()}')
print(f'Negative       : {(~df["has_calcium"]).sum()}')

df.to_csv(TABLES / 'scan_index_clean.csv', index=False)
print(f'Saved scan_index_clean.csv')
