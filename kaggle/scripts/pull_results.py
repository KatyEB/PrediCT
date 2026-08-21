
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "models"


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description="Pull nnU-Net results from Kaggle")
    parser.add_argument("--kernel", required=True,
                        help="Kaggle kernel slug (e.g. username/eat-nnunet-2d-train)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Local output directory (default: models/nnunet_{2d|3d}/)")
    args = parser.parse_args()

    model_type = "2d" if "2d" in args.kernel else "3d"
    out_dir = args.out_dir or (MODELS_DIR / f"nnunet_{model_type}")
    out_dir.mkdir(parents=True, exist_ok=True)

    kaggle_cmd = str(REPO_ROOT / ".venv" / "bin" / "kaggle")

    print(f"Pulling output from kernel: {args.kernel}")
    rc = run([kaggle_cmd, "kernels", "output", args.kernel, "-p", str(out_dir)])

    if rc == 0:
        # Unzip if needed
        for zf in out_dir.glob("*.zip"):
            print(f"Extracting {zf.name}...")
            with zipfile.ZipFile(zf) as z:
                z.extractall(out_dir)
            zf.unlink()
        print(f"\nResults saved to {out_dir}")
    else:
        print(f"Pull failed (exit code {rc}). Is the kernel done running?")
        print(f"Check status: kaggle kernels status {args.kernel}")


if __name__ == "__main__":
    main()
