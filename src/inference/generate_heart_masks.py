
from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

import torch
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESAMPLED_DIR = REPO_ROOT / "data" / "COCA_dataset" / "data_resampled"
HEART_MASKS_DIR = REPO_ROOT / "data" / "COCA_dataset" / "heart_masks"
LOG_PATH = HEART_MASKS_DIR / "run_log.csv"


def get_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        print("WARNING: MPS not available, falling back to CPU")
        return "cpu"
    return requested


def process_one(scan_id: str, device: str) -> dict:
    img_path = RESAMPLED_DIR / scan_id / f"{scan_id}_img.nii.gz"
    out_dir = HEART_MASKS_DIR / scan_id
    heart_out = out_dir / "heart.nii.gz"

    result = {"scan_id": scan_id, "status": "ok", "time_s": 0.0, "error": ""}

    if heart_out.exists():
        result["status"] = "cached"
        return result

    if not img_path.exists():
        result["status"] = "missing_input"
        result["error"] = str(img_path)
        return result

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        totalsegmentator(
            input=str(img_path),
            output=str(out_dir),
            task="total",
            fast=True,
            roi_subset=["heart"],
            device=device,
            quiet=True,
        )
        result["time_s"] = time.time() - t0
    except Exception as e:
        result["status"] = "error"
        result["error"] = traceback.format_exc(limit=3)
        result["time_s"] = time.time() - t0

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch heart mask generation")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--total-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}, shard: {args.shard}/{args.total_shards}")

    HEART_MASKS_DIR.mkdir(parents=True, exist_ok=True)

    all_scan_ids = sorted([p.name for p in RESAMPLED_DIR.iterdir() if p.is_dir()])
    scan_ids = [s for i, s in enumerate(all_scan_ids) if i % args.total_shards == args.shard]
    print(f"Patients in this shard: {len(scan_ids)} of {len(all_scan_ids)}")

    already_done = sum(1 for sid in scan_ids if (HEART_MASKS_DIR / sid / "heart.nii.gz").exists())
    print(f"Already complete: {already_done} / {len(scan_ids)}")

    log_rows = []
    times = []

    for sid in tqdm(scan_ids, desc="Heart masks"):
        result = process_one(sid, device)
        log_rows.append(result)
        if result["status"] == "ok":
            times.append(result["time_s"])
            avg = sum(times) / len(times)
            remaining = len(scan_ids) - len(times) - already_done
            eta_h = avg * remaining / 3600
            tqdm.write(f"  {sid}: {result['time_s']:.1f}s | avg={avg:.1f}s | ETA {eta_h:.1f}h")
        elif result["status"] == "error":
            tqdm.write(f"  ERROR {sid}: {result['error'][:100]}")

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scan_id", "status", "time_s", "error"])
        writer.writeheader()
        writer.writerows(log_rows)

    ok = sum(1 for r in log_rows if r["status"] in ("ok", "cached"))
    errors = sum(1 for r in log_rows if r["status"] == "error")
    avg_t = sum(times) / len(times) if times else 0
    print(f"\nDone: {ok}/{len(scan_ids)} ok, {errors} errors")
    print(f"Avg time per patient: {avg_t:.1f}s")
    print(f"Log saved: {LOG_PATH}")


if __name__ == "__main__":
    main()
