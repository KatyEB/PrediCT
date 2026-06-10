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
    default_root = r"E:\MyProjects\GSOC_Predict_26"  # Change this to your actual project root
    save_path = input(f"Path to Save (folder will be created): ").strip() or default_root
    
    # 2. Execution Logic
    print("\n1) Full Pipeline\n2) Process Only\n3) Resample Only")
    choice = input("Selection: ").strip()

    if choice in ['1', '2']:
        print("\n--- Running Processor ---")
        dataset_folder = input("Dataset Folder Path: ").strip() or r"\cocacoronarycalciumandchestcts-2\Gated_release_final\patient"
        save_path = Path(save_path)
        images_path = Path(dataset_folder +  r"\cocacoronarycalciumandchestcts-2\Gated_release_final\patient")
        calcium_xml_path = Path(dataset_folder + r"\cocacoronarycalciumandchestcts-2\Gated_release_final\calcium_xml")


        proc = COCAProcessor(save_path, images_path, calcium_xml_path)
        proc.process_all()

    if choice in ['1', '3']:
        print("\n--- Running Resampler ---")
        
        space = input("Voxel Spacing (x,y,z) Write as 0.5 0.5 3: ").strip()

        if not space:
            target = [1.0, 1.0, 1.0]
        else:
            target = [float(x) for x in space.split(" ")]

        print(f"Using target spacing: {target}")

        resamp = COCAResampler(save_path, target_spacing=target)
        # resamp = COCAResampler(save_path) # i want to use the default spacing for now, 0.7x0.7x3.0, which is what the original paper used. we can experiment with this later if we want to.
        resamp.run()

    print("\nPipeline Finished.")

if __name__ == "__main__":
    main()