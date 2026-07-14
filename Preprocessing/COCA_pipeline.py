import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

print(f"Pipeline running from: {SCRIPT_DIR}")

try:
    from COCA_processor import COCAProcessor
    from COCA_resampler import COCAResampler
    from COCA_split     import COCASplitter
    print("COCA modules imported successfully.")
except ImportError as e:
    print(f"\n[IMPORT ERROR]: {e}")
    print(f"Expected modules in: {SCRIPT_DIR}")
    sys.exit(1)


def prompt_root() -> str:
    default = "/Users/karan/Desktop/PrediCT/cocacoronarycalciumandchestcts-2"
    val = input(f"\nProject Root [{default}]: ").strip()
    return val or default


def prompt_spacing(default: str = "0.375") -> list:
    val = input(f"  XY voxel spacing mm [{default}]: ").strip() or default
    s = float(val)
    return [s, s, 3.0]


def main():
    print("=" * 55)
    print("       COCA DATA PREPROCESSING PIPELINE")
    print("=" * 55)
    print("1) Full pipeline (Process → Resample → Split)")
    print("2) Process only")
    print("3) Resample only  (single spacing)")
    print("4) Resample both  (0.375 + 0.700 mm, dual-branch)")
    print("5) Split only")
    choice = input("Selection: ").strip()

    root = prompt_root()

    # ── Stage 1: Processor ─────────────────────────────────────────────
    if choice in ["1", "2"]:
        print("\n─── Stage 1: Processor (HU windowing + metadata) ───")
        proc = COCAProcessor(root)
        proc.process_all()

    # ── Stage 2: Resampler ─────────────────────────────────────────────
    if choice == "3":
        print("\n─── Stage 2: Resampler (single spacing) ───")
        target = prompt_spacing("0.375")
        COCAResampler(root, target_spacing=target).run()

    elif choice == "4":
        print("\n─── Stage 2: Resampler (dual-branch sensitivity) ───")
        print("  Branch A → 0.375 mm")
        COCAResampler(root, target_spacing=[0.375, 0.375, 3.0]).run()
        print("\n  Branch B → 0.700 mm")
        COCAResampler(root, target_spacing=[0.700, 0.700, 3.0]).run()

    elif choice == "1":
        print("\n─── Stage 2: Resampler ───")
        target = prompt_spacing("0.375")
        COCAResampler(root, target_spacing=target).run()

    # ── Stage 3: Split ─────────────────────────────────────────────────
    if choice in ["1", "5"]:
        print("\n─── Stage 3: Stratified Train/Val/Test Split ───")
        from pathlib import Path
        # Auto-detect available resampled index files
        data_dir = Path("/Volumes/SanDisk/PrediCT/data_resampled")
        csv_files = sorted(data_dir.glob("resampled_index_*.csv"))
        if not csv_files:
            print("[ERROR] No resampled index found. Run resampler first.")
        else:
            for csv_path in csv_files:
                print(f"  Splitting: {csv_path.name}")
                splitter = COCASplitter(str(csv_path))
                splitter.run()

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()