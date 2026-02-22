import pandas as pd

def audit_splits(train_path, val_path):
    for name, path in [("TRAIN", train_path), ("VAL", val_path)]:
        df = pd.read_parquet(path)
        total = len(df)
        positives = df['has_pos'].sum()
        negatives = total - positives
        
        print(f"--- {name} SPLIT AUDIT ---")
        print(f"Total Scans: {total}")
        print(f"Positive (with calcium): {positives} ({positives/total:.1%})")
        print(f"Negative (healthy): {negatives} ({negatives/total:.1%})")
        
        if positives == 0:
            print(f"❌ WARNING: {name} split has ZERO calcium. Check your splitting logic!")
        print("\n")

if __name__ == "__main__":
    audit_splits(
        r"C:\coca_project\data_canonical\tables\train_split.parquet",
        r"C:\coca_project\data_canonical\tables\val_split.parquet"
    )