
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FILES = [
    REPO_ROOT / "kaggle" / "dataset_metadata.json",
    REPO_ROOT / "kaggle" / "kernels" / "kernel-metadata.json",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Your Kaggle username")
    args = parser.parse_args()

    username = args.username.lower().strip()

    for path in FILES:
        if not path.exists():
            print(f"SKIP (not found): {path}")
            continue
        text = path.read_text()
        if "KAGGLE_USERNAME" not in text:
            print(f"OK (already set): {path.name}")
            continue
        path.write_text(text.replace("KAGGLE_USERNAME", username))
        print(f"Updated: {path.relative_to(REPO_ROOT)}")

    print(f"\nUsername set to: {username}")
    print("Next steps:")
    print("  1. python kaggle/scripts/push_dataset.py --create")
    print("  2. kaggle kernels push -p kaggle/kernels/")


if __name__ == "__main__":
    main()
