"""
pipeline_nnunet.py — nnUNet specific inference pipeline

Handles invoking nnUNet CLI for models marked with arch="nnunet".
This ensures that the main pipeline.py remains pure PyTorch and doesn't mix execution models.
"""
import os
import sys
import tempfile
import subprocess
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from .pipeline import save_nifti

def predict_nnunet(image: sitk.Image, manifest: dict) -> np.ndarray:
    """Run nnUNet inference on the given resampled image.
    
    Args:
        image: sitk.Image in RAS orientation, already resampled to target spacing.
        manifest: The loaded model manifest containing nnunet_results_path.
        
    Returns:
        prob: (Z, Y, X) numpy array containing probabilities (or binary masks, 
              which will be treated as probabilities [0, 1] by scoring).
    """
    nnunet_results = manifest["nnunet_results_path"]
    if not nnunet_results.exists():
        raise FileNotFoundError(f"nnUNet_results not found at {nnunet_results}")
        
    # Create temp dirs for nnUNet I/O
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_in = Path(tmp_dir) / "in"
        tmp_out = Path(tmp_dir) / "out"
        tmp_in.mkdir()
        tmp_out.mkdir()
        
        # nnUNet requires inputs to be named <case_identifier>_0000.nii.gz
        in_file = tmp_in / "case_0000.nii.gz"
        save_nifti(image, in_file)
        
        # Set nnUNet_results env var so it finds the model
        env = os.environ.copy()
        env["nnUNet_results"] = str(nnunet_results)
        
        # We assume Dataset001_CAC based on the typical structure
        nnunet_exec = Path(sys.executable).parent / "nnUNetv2_predict"
        cmd = [
            str(nnunet_exec),
            "-i", str(tmp_in),
            "-o", str(tmp_out),
            "-d", "001",
            "-c", "3d_fullres",
            "-f", "0",
            "-chk", manifest.get("weights", "checkpoint_final.pth"),
            "--disable_tta" # TTA is slow, skip for speed
        ]
        
        result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"nnUNet prediction failed:\n{result.stdout}")
            
        out_file = tmp_out / "case.nii.gz"
        if not out_file.exists():
            raise FileNotFoundError(f"nnUNet output not found at {out_file}")
            
        prob_img = sitk.ReadImage(str(out_file))
        prob = sitk.GetArrayFromImage(prob_img) # (Z, Y, X)
        
        # nnUNet typically outputs uint8/int for segmentation. 
        # Convert to float32 to match our pipeline.py
        return prob.astype(np.float32)
