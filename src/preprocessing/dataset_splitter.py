import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

TABLES = Path(r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2\data_canonical\tables")

df = pd.read_csv(TABLES / "scan_index_clean.csv")

print(f"Total : {len(df)} | Pos: {df['has_calcium'].sum()} | Neg: {(~df['has_calcium']).sum()}")

neg = df[~df['has_calcium']]
pos = df[df['has_calcium']]

pos_train, pos_temp = train_test_split(pos, test_size=0.30, random_state=42)
pos_val, pos_test   = train_test_split(pos_temp, test_size=0.50, random_state=42)

train = pd.concat([pos_train, neg]).reset_index(drop=True)
val   = pos_val.reset_index(drop=True)
test  = pos_test.reset_index(drop=True)

train.to_parquet(TABLES / "train_split.parquet", index=False)
val.to_parquet(TABLES   / "val_split.parquet", index=False)
test.to_parquet(TABLES  / "test_split.parquet", index=False)

print("\n=== SPLIT SUMMARY ===")
for name, s in [("Train", train), ("Val", val), ("Test", test)]:
    pos_n = s["has_calcium"].sum()
    neg_n = (~s["has_calcium"]).sum()
    print(f"{name:5s}: {len(s):3d} | Pos: {pos_n} ({100*pos_n/len(s):.1f}%) | Neg: {neg_n}")

print("\nSaved train / val / test_split.parquet")