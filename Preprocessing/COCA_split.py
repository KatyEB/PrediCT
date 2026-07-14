"""
COCA_split.py
-------------
Stratified train / val / test split for the COCA dataset.

Strategy
--------
  • Splits are computed on BASE scans only (is_augmented == False).
  • Stratification variable: has_calcium (binary, addresses class imbalance).
  • Augmented copies (LR-flipped) are appended to train only — they never
    appear in val or test, preventing data leakage.
  • Default split: 70 / 15 / 15.
  • Random seed is fixed (42) for full reproducibility.

Output
------
  <resampled_dir>/split_<spacing_tag>.csv
    Adds columns: split ∈ {train, val, test}
    Augmented rows inherit split = "train".
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class COCASplitter:
    def __init__(
        self,
        resampled_csv: str,
        val_frac: float   = 0.15,
        test_frac: float  = 0.15,
        seed: int         = 42,
    ):
        self.csv_path  = Path(resampled_csv)
        self.val_frac  = val_frac
        self.test_frac = test_frac
        self.seed      = seed

    def run(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        if "is_augmented" not in df.columns:
            df["is_augmented"] = False
        if "has_calcium" not in df.columns:
            df["has_calcium"] = (df["voxels"] > 0).astype(int)

        # Work only on base (non-augmented) scans for splitting
        base = df[~df["is_augmented"].astype(bool)].copy()
        aug  = df[ df["is_augmented"].astype(bool)].copy()

        # Stratified split: train+val+test on base scans
        test_size  = self.test_frac
        val_size   = self.val_frac / (1.0 - self.test_frac)  # adjust for remaining fraction

        base_train_val, base_test = train_test_split(
            base,
            test_size=test_size,
            stratify=base["has_calcium"],
            random_state=self.seed,
        )
        base_train, base_val = train_test_split(
            base_train_val,
            test_size=val_size,
            stratify=base_train_val["has_calcium"],
            random_state=self.seed,
        )

        base_train = base_train.copy(); base_train["split"] = "train"
        base_val   = base_val.copy();   base_val["split"]   = "val"
        base_test  = base_test.copy();  base_test["split"]  = "test"

        # Augmented copies always go to train
        aug = aug.copy(); aug["split"] = "train"

        full = pd.concat([base_train, base_val, base_test, aug], ignore_index=True)

        out_csv = self.csv_path.parent / (
            self.csv_path.stem.replace("resampled_index", "split") + ".csv"
        )
        full.to_csv(out_csv, index=False)

        # Summary
        base_counts = full[~full["is_augmented"].astype(bool)]["split"].value_counts()
        aug_counts  = full[ full["is_augmented"].astype(bool)]["split"].value_counts()

        print(f"\n  Split saved → {out_csv.name}")
        print(f"  {'Split':<8} {'Base':>6}  {'Pos':>5}  {'Aug':>5}")
        print(f"  {'-'*30}")
        for s in ["train", "val", "test"]:
            sub = full[(full["split"] == s) & ~full["is_augmented"].astype(bool)]
            n_aug_s = full[(full["split"] == s) & full["is_augmented"].astype(bool)].shape[0]
            print(f"  {s:<8} {len(sub):>6}  {sub['has_calcium'].sum():>5}  {n_aug_s:>5}")

        return full


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/karan/Desktop/PrediCT/cocacoronarycalciumandchestcts-2"
        "/data_resampled/resampled_index_0.375x0.375x3.000.csv"
    )
    COCASplitter(csv).run()