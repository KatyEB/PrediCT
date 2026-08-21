# %% [markdown]
# # EAT Segmentation — nnU-Net Training (2D + 3D)
#
# Trains nnU-Net v2 on the EAT pseudo-label dataset uploaded from the eat-segmentation repo.
#
# **Dataset:** eat-segmentation-nnunet (Kaggle dataset)
# **Models:** 2d, 3d_fullres (3d_lowres if fullres OOM)
# **Folds:** 5-fold cross-validation (nnU-Net default)
#
# Runtime estimate: ~6h for 2D (5 folds × 1000 epochs), ~15-20h for 3D.
# Run 2D first. If 3D OOM, switch trainer to use smaller batch or skip.

# %% [markdown]
# ## 1. Setup

# %%
import os
import subprocess
from pathlib import Path

# Kaggle mounts datasets at /kaggle/input/datasets/<user>/<name> or /kaggle/input/<name>
WORKING_DIR = Path("/kaggle/working")
_candidates = [ 
    Path("/kaggle/input/eat-segmentation-nnunet"),
    Path("/kaggle/input/datasets/shreeeeyyyyy/eat-segmentation-nnunet"),
]
KAGGLE_DATASET = next((p for p in _candidates if p.exists()), None)
if KAGGLE_DATASET is None:
    raise FileNotFoundError(f"Dataset not found. Tried: {_candidates}")
print(f"Dataset path: {KAGGLE_DATASET}")

# nnU-Net environment variables
os.environ["nnUNet_raw"] = str(WORKING_DIR / "nnunet_raw")
os.environ["nnUNet_preprocessed"] = str(WORKING_DIR / "nnunet_preprocessed")
os.environ["nnUNet_results"] = str(WORKING_DIR / "nnunet_results")

for d in [os.environ["nnUNet_raw"], os.environ["nnUNet_preprocessed"], os.environ["nnUNet_results"]]:
    Path(d).mkdir(parents=True, exist_ok=True)

print("nnU-Net paths:")
for k in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
    print(f"  {k} = {os.environ[k]}")

# %%
# Install nnU-Net (Kaggle has PyTorch pre-installed with CUDA)
# PyPI package name is nnunetv2; fall back to GitHub if PyPI unavailable
result = subprocess.run(["pip", "install", "nnunetv2", "--quiet"])
if result.returncode != 0:
    print("PyPI install failed, trying GitHub...")
    subprocess.run([
        "pip", "install",
        "git+https://github.com/MIC-DKFZ/nnUNet.git",
        "--quiet"
    ], check=True)
print("nnunetv2 installed")

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Copy dataset from Kaggle input to nnU-Net raw (+ preprocessed if available)

# %%
import shutil

# Raw data (required)
raw_src = KAGGLE_DATASET / "raw" / "Dataset001_EAT"
raw_dst = Path(os.environ["nnUNet_raw"]) / "Dataset001_EAT"

if not raw_dst.exists():
    shutil.copytree(raw_src, raw_dst)
    print(f"Copied raw dataset → {raw_dst}")
else:
    print(f"Raw dataset already at {raw_dst}")

# Kaggle auto-decompresses .nii.gz → .nii; detect actual extension and patch dataset.json
sample = next((raw_dst / "imagesTr").glob("*.nii*"), None)
actual_ext = "".join(sample.suffixes) if sample else ".nii.gz"
import json as _json
ds_json_path = raw_dst / "dataset.json"
with open(ds_json_path) as f:
    ds_meta = _json.load(f)
if ds_meta.get("file_ending") != actual_ext:
    ds_meta["file_ending"] = actual_ext
    with open(ds_json_path, "w") as f:
        _json.dump(ds_meta, f, indent=2)
    print(f"Patched dataset.json file_ending → {actual_ext}")

n_train = len(list((raw_dst / "imagesTr").glob(f"*{actual_ext}")))
n_val = len(list((raw_dst / "imagesVal").glob(f"*{actual_ext}")))
print(f"Train: {n_train} | Val: {n_val} | file_ending: {actual_ext}")

# Preprocessed data (optional — skips the ~15 min preprocessing step)
pre_src = KAGGLE_DATASET / "preprocessed" / "Dataset001_EAT"
pre_dst = Path(os.environ["nnUNet_preprocessed"]) / "Dataset001_EAT"
SKIP_PREPROCESS = False
 
if pre_src.exists():
    if not pre_dst.exists():
        shutil.copytree(pre_src, pre_dst)
        print(f"Copied preprocessed data → {pre_dst}")
    else:
        print(f"Preprocessed data already at {pre_dst}")
    SKIP_PREPROCESS = True
    print("Will skip nnUNetv2_plan_and_preprocess (preprocessed data present)")
else:
    print("No preprocessed data in input — will run preprocessing")

# %% [markdown]
# ## 3. Plan and preprocess (skipped if preprocessed data was bundled)

# %%
if SKIP_PREPROCESS:
    print("Skipping preprocessing — using bundled preprocessed data.")
else:
    # -c 2d 3d_fullres: plan both configurations
    # -np 4: use 4 processes for preprocessing
    result = subprocess.run(
        ["nnUNetv2_plan_and_preprocess", "-d", "1", "-c", "2d", "3d_fullres", "-np", "4", "--verify_dataset_integrity"],
        capture_output=True, text=True
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError("Preprocessing failed")
    print("\nPreprocessing complete.")

# %% [markdown]
# ## 4a. Train 2D (all 5 folds)

# %%
DATASET_ID = 1
# nnUNetTrainer_250epochs fits in ~10h on T4 (250 × ~150s/epoch).
# Full 1000-epoch training needs CERN cluster or multiple resumed sessions.
TRAINER = "nnUNetTrainer_250epochs"
# Set FOLDS_TO_TRAIN = [0] for a single-fold baseline run (~10h).
# Set FOLDS_TO_TRAIN = list(range(5)) for full 5-fold cross-validation.
FOLDS_TO_TRAIN = [0]

for fold in FOLDS_TO_TRAIN:
    print(f"\n{'='*50}")
    print(f"Training 2D fold {fold} with {TRAINER} (2x GPU via torchrun)...")
    print(f"{'='*50}")
    result = subprocess.run(
        ["torchrun", "--nproc_per_node=2",
         "-m", "nnunetv2.run.run_training",
         str(DATASET_ID), "2d", str(fold),
         "-tr", TRAINER,
         "--npz"],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"WARNING: fold {fold} exited with code {result.returncode}")

print(f"\n2D training complete for folds: {FOLDS_TO_TRAIN}")

# %% [markdown]
# ## 4b. Train 3D fullres (all 5 folds)
#
# If this OOM's on the Kaggle GPU (16GB), catch the error and document it.
# The 2D model is already sufficient for a first submission.

# %%
import json

three_d_ok = True
for fold in range(5):
    print(f"\nTraining 3D fullres fold {fold}/4...")
    try:
        result = subprocess.run(
            ["nnUNetv2_train", str(DATASET_ID), "3d_fullres", str(fold), "--npz"],
            capture_output=True, text=True, timeout=4*3600  # 4h timeout per fold
        )
        if "RuntimeError: CUDA out of memory" in result.stderr or result.returncode != 0:
            print(f"3D fold {fold} failed: OOM or error. Stopping 3D training.")
            print(result.stderr[-500:])
            three_d_ok = False
            break
    except subprocess.TimeoutExpired:
        print(f"3D fold {fold} timed out (>4h). Stopping.")
        three_d_ok = False
        break

if three_d_ok:
    print("\n3D training complete for all folds.")
else:
    print("\n3D training incomplete — see output above. 2D model is complete.")
      
# %% [markdown]
# ## 5. Package results for download
 
# %%
import zipfile, time

results_dir = Path(os.environ["nnUNet_results"])
timestamp = time.strftime("%Y%m%d_%H%M")

def zip_model(config_name: str, out_zip: Path, trainer: str = TRAINER):
    model_dir = results_dir / "Dataset001_EAT" / f"{trainer}__{config_name}__nnUNetPlans"
    if not model_dir.exists():
        print(f"No results for {config_name}")
        return False
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in model_dir.rglob("*"):
            if f.is_file() and f.suffix in {".pth", ".json", ".pkl", ".yaml", ".txt"}:
                zf.write(f, f.relative_to(results_dir))
    size_mb = out_zip.stat().st_size / 1e6
    print(f"Zipped {config_name}: {out_zip.name} ({size_mb:.0f} MB)")
    return True

zip_model("2d",         WORKING_DIR / f"nnunet_2d_{timestamp}.zip")
zip_model("3d_fullres", WORKING_DIR / f"nnunet_3d_{timestamp}.zip")
print(f"\nTrainer used: {TRAINER}")

print(f"\nOutputs in /kaggle/working/ — download from Kaggle UI or via:")
print(f"  python kaggle/scripts/pull_results.py --kernel <your-username>/eat-nnunet-train")