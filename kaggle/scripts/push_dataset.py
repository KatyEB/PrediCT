
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATASET = REPO_ROOT / "nnunet" / "raw" / "Dataset001_EAT"
PREPROCESSED_DATASET = REPO_ROOT / "nnunet" / "preprocessed" / "Dataset001_EAT"
METADATA_PATH = REPO_ROOT / "kaggle" / "dataset_metadata.json"


def check_dataset_exists():
    if not RAW_DATASET.exists():
        print(f"ERROR: Raw dataset not found at {RAW_DATASET}")
        print("Run experiments/exp-007-nnunet-dataset/experiment.py first.")
        sys.exit(1)
    n_train = len(list((RAW_DATASET / "imagesTr").glob("*.nii.gz")))
    n_val = len(list((RAW_DATASET / "imagesVal").glob("*.nii.gz")))
    raw_mb = sum(f.stat().st_size for f in RAW_DATASET.rglob("*") if f.is_file()) / 1e6
    print(f"Raw dataset: {n_train} train + {n_val} val cases ({raw_mb:.0f} MB)")

    if PREPROCESSED_DATASET.exists():
        pre_mb = sum(f.stat().st_size for f in PREPROCESSED_DATASET.rglob("*") if f.is_file()) / 1e6
        print(f"Preprocessed data: {pre_mb:.0f} MB (will be included — skips preprocessing on Kaggle)")
    else:
        print("No preprocessed data found — Kaggle notebook will run preprocessing (~15 min)")

    return n_train, n_val


def build_staging_dir(tmp: Path) -> Path:
    """Create staging dir: raw/ + preprocessed/ + dataset-metadata.json."""
    staging = tmp / "eat-segmentation-nnunet"
    staging.mkdir()

    # raw data
    print("Copying raw dataset...")
    shutil.copytree(RAW_DATASET, staging / "raw" / "Dataset001_EAT")

    # preprocessed data (optional but saves ~15 min on Kaggle)
    if PREPROCESSED_DATASET.exists():
        print("Copying preprocessed data...")
        shutil.copytree(PREPROCESSED_DATASET, staging / "preprocessed" / "Dataset001_EAT")

    # Kaggle requires dataset-metadata.json in the push directory
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    kaggle_meta = {
        "title": meta["title"],
        "id": meta["id"],
        "licenses": meta["licenses"],
        "keywords": meta.get("keywords", []),
    }
    with open(staging / "dataset-metadata.json", "w") as f:
        json.dump(kaggle_meta, f, indent=2)

    total_mb = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file()) / 1e6
    print(f"Staging dir ready: {total_mb:.0f} MB total")
    return staging


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Push nnU-Net EAT dataset to Kaggle")
    parser.add_argument("--create", action="store_true",
                        help="Create new dataset (first time only)")
    parser.add_argument("--message", type=str, default="Updated dataset",
                        help="Version message for updates")
    args = parser.parse_args()

    check_dataset_exists()

    kaggle_cmd = str(REPO_ROOT / ".venv" / "bin" / "kaggle")

    with tempfile.TemporaryDirectory() as tmp:
        staging = build_staging_dir(Path(tmp))

        if args.create:
            print("\nCreating new Kaggle dataset...")
            rc = run([kaggle_cmd, "datasets", "create",
                      "-p", str(staging),
                      "--dir-mode", "zip"])
        else:
            print("\nUpdating existing Kaggle dataset with new version...")
            rc = run([kaggle_cmd, "datasets", "version",
                      "-p", str(staging),
                      "-m", args.message,
                      "--dir-mode", "zip"])

    with open(METADATA_PATH) as f:
        meta = json.load(f)
    dataset_id = meta["id"]

    if rc == 0:
        print("\nDataset pushed successfully.")
        print(f"View at: https://www.kaggle.com/datasets/{dataset_id}")
        print("\nNext: create a Kaggle notebook, attach this dataset, upload kaggle/notebooks/nnunet_train.py")
    else:
        print(f"\nPush failed (exit code {rc}).")
        print("Check: .venv/bin/kaggle datasets list --mine")


if __name__ == "__main__":
    main()
