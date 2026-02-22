import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_nnunet_split(parquet_path, output_dir):
    # Load the frozen contract
    df = pd.read_parquet(parquet_path)
    
    # 1. First split: Separate out the TEST set (15%)
    # Stratify by 'has_pos' to keep calcium cases balanced
    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.15, 
        random_state=42, 
        stratify=df['has_pos']
    )
    
    # 2. Second split: Separate TRAIN and VAL from the remaining 85%
    # 0.176 of 85% is roughly 15% of the total
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.176, 
        random_state=42, 
        stratify=train_val_df['has_pos']
    )

    # 3. Save the splits
    output_dir = Path(output_dir)
    train_df.to_parquet(output_dir / "train_split.parquet", index=False)
    val_df.to_parquet(output_dir / "val_split.parquet", index=False)
    test_df.to_parquet(output_dir / "test_split.parquet", index=False)

    print(f"--- Data Split Complete ---")
    print(f"Train: {len(train_df)} cases ({train_df['has_pos'].sum()} pos)")
    print(f"Val:   {len(val_df)} cases ({val_df['has_pos'].sum()} pos)")
    print(f"Test:  {len(test_df)} cases ({test_df['has_pos'].sum()} pos)")

if __name__ == "__main__":
    PARQUET = r"C:\coca_project\data_canonical\tables\metadata_summary.parquet"
    OUT = r"C:\coca_project\data_canonical\tables"
    create_nnunet_split(PARQUET, OUT)