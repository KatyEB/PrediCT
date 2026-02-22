import sys
import os
from pathlib import Path

# 1. Get the absolute path of the directory where THIS script is located
# This bypasses all 'current working directory' issues
SCRIPT_DIR = Path(__file__).resolve().parent

# 2. Force this directory into the system path at the very beginning
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 3. Diagnostic check (optional but helpful)
print(f"Pipeline running from: {SCRIPT_DIR}")

try:
    # Now we import using the exact class names
    from COCA_processor import COCAProcessor
    from COCA_resampler import COCAResampler
    print("Successfully imported COCA modules.")
except ImportError as e:
    print(f"\n[STILL FAILING]: {e}")
    print(f"I am looking for files in: {SCRIPT_DIR}")
    print(f"Files found there: {os.listdir(SCRIPT_DIR)}")
    sys.exit(1)

def main():
    print("="*50)
    print("      COCA DATA PREPROCESSING PIPELINE")
    print("="*50)

    # 1. Configuration
    default_root = r"C:\coca_project"
    project_root = input(f"Project Root [{default_root}]: ").strip() or default_root
    
    # 2. Execution Logic
    print("\n1) Full Pipeline\n2) Process Only\n3) Resample Only")
    choice = input("Selection: ").strip()

    if choice in ['1', '2']:
        print("\n--- Running Processor ---")
        proc = COCAProcessor(project_root)
        proc.process_all()

    if choice in ['1', '3']:
        print("\n--- Running Resampler ---")
        space = input("Voxel Spacing (mm) [1.0]: ").strip() or "1.0"
        target = [float(space)] * 3
        resamp = COCAResampler(project_root, target_spacing=target)
        resamp.run()

    print("\nPipeline Finished.")

if __name__ == "__main__":
    main()