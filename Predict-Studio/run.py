"""
run.py — the command line. This is the deliverable that cannot slip.

    python run.py data/patient_0                       score with every model
    python run.py data/ --models a3-coverage-v2        one model
    python run.py data/ --out results/ --keep all      keep everything
    python run.py data/ --legacy                       reproduce old numbers

    python run.py --list                               show available models

One patient or two hundred is the same command. A cohort is a loop, and each
patient's row is written the moment it finishes, so an interrupted run resumes
instead of restarting.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import models as model_lib
import pipeline
from load import load_folder
from scoring import ScoringConfig

MODEL_DIRS = ("models", str(Path.home() / ".predict" / "models"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Coronary artery calcium scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("folder", nargs="?", help="folder of DICOM / NIfTI, any depth")
    p.add_argument("--list", action="store_true", help="list models and exit")
    p.add_argument("--models", nargs="+", metavar="ID",
                   help="model ids to run (default: all that verify)")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--cache", default=".cache",
                   help="preprocessing cache; '' disables it")
    p.add_argument("--keep", choices=["all", "results", "scores-only"],
                   default="results")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")

    g = p.add_argument_group("scoring")
    g.add_argument("--legacy", action="store_true",
                   help="reproduce agatston_scoring_a{1,3}.py: no minimum-lesion "
                        "rule, no thickness correction")
    g.add_argument("--lesions", choices=["2d", "3d"], default="2d",
                   help="lesion definition. 3d gives DIFFERENT Agatston numbers")
    g.add_argument("--min-area", type=float, default=None,
                   metavar="MM2", help="minimum lesion area (default 1.0)")
    g.add_argument("--threshold", type=float, default=None,
                   help="binarisation threshold for binary models")

    p.add_argument("--force", action="store_true",
                   help="run even if the contract gate objects; every objection "
                        "is recorded and stamped on the output")
    p.add_argument("--restart", action="store_true",
                   help="ignore existing rows and rescore everything")
    p.add_argument("--debug", action="store_true")
    return p


def resolve_config(args) -> ScoringConfig:
    config = ScoringConfig.legacy() if args.legacy else ScoringConfig()
    changes = {"lesion_definition": args.lesions}
    if args.min_area is not None:
        changes["min_lesion_area_mm2"] = args.min_area
    if args.threshold is not None:
        changes["binary_threshold"] = args.threshold
    return config.with_(**changes)


def load_done(path: Path) -> set[tuple[str, str]]:
    """Which (patient, model) pairs are already in the CSV."""
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {(r["patient_id"], r["model_id"]) for r in csv.DictReader(f)}


def append_row(path: Path, row: dict, wrote_header: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not wrote_header and not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if need_header:
            writer.writeheader()
        writer.writerow(row)
    return True


def main() -> int:
    args = build_parser().parse_args()
    pipeline.set_debug(args.debug)

    available = model_lib.discover(*MODEL_DIRS)
    if args.list or not args.folder:
        print(f"\nModels found in {' and '.join(MODEL_DIRS)}\n")
        for m in available:
            exists = "ok" if m.weights_path.exists() or m.framework == "dummy" \
                     else "MISSING WEIGHTS"
            print(f"  {m.id:<20} {m.output_type:<7} {exists}")
            print(f"  {'':<20} {m.trained_on}")
        print(f"\nmodels_root = {model_lib.models_root()}")
        print("Set PREDICT_MODELS_ROOT to change it.\n")
        return 0 if args.list else 1

    chosen = available
    if args.models:
        by_id = {m.id: m for m in available}
        unknown = [i for i in args.models if i not in by_id]
        if unknown:
            print(f"Unknown model(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(sorted(by_id))}")
            return 1
        chosen = [by_id[i] for i in args.models]

    usable = []
    for m in chosen:
        try:
            model_lib.verify_weights(m) if m.framework != "dummy" else None
            usable.append(m)
        except Exception as e:
            print(f"[skip] {m.id}: {e}\n")
    if not usable:
        print("No usable models. Check PREDICT_MODELS_ROOT and the manifests.")
        return 1

    print(f"\nLoading {args.folder}")
    volumes = load_folder(args.folder)
    if not volumes:
        print("No loadable scans found.")
        return 1
    print(f"Found {len(volumes)} scan(s), running {len(usable)} model(s)\n")

    out_dir = Path(args.out)
    csv_path = out_dir / "results.csv"
    cache_dir = Path(args.cache) if args.cache else None
    config = resolve_config(args)

    if args.restart and csv_path.exists():
        csv_path.unlink()
    done = load_done(csv_path)
    wrote_header = csv_path.exists()

    total = len(volumes) * len(usable)
    n = 0
    counts = {"ok": 0, "refused": 0, "failed": 0, "skipped": 0}

    for volume in volumes:
        for model in usable:
            n += 1
            tag = f"[{n}/{total}] {volume.patient_id} x {model.id}"

            if (volume.patient_id, model.id) in done:
                counts["skipped"] += 1
                print(f"{tag}  already done")
                continue

            result = pipeline.process_study(
                volume, model, scoring_config=config, out_dir=out_dir,
                cache_dir=cache_dir, keep=args.keep, device=args.device,
                force=args.force,
            )
            counts[result.status] += 1
            wrote_header = append_row(csv_path, result.row(), wrote_header)

            if result.status == "ok":
                s = result.score
                print(f"{tag}  Agatston {s.agatston:9.2f}  "
                      f"{s.risk_category:<8}  {result.seconds:5.1f}s")
                for w in result.warnings:
                    print(f"       warning  {w}")
            elif result.status == "refused":
                print(f"{tag}  REFUSED")
                for p in result.problems:
                    print(f"       {p}")
                print("       use --force to run anyway (it will be stamped)")
            else:
                print(f"{tag}  FAILED")
                for p in result.problems:
                    print(f"       {p}")

    print(f"\n{counts['ok']} scored, {counts['refused']} refused, "
          f"{counts['failed']} failed, {counts['skipped']} already done")
    print(f"-> {csv_path}\n")
    return 0 if counts["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
