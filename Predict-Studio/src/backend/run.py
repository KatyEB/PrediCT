"""
run.py — Inference orchestrator for PrediCT.

Ties together paths, pipeline, scoring, and rendering into a complete execution flow.
Handles caching of prep stages and writes the provenance `run.json` to the output folder.

Does NOT: handle HTTP requests, instantiate PyTorch models, or implement math.
Called by: CLI entrypoint, server.py.

Usage:
    # Python API
    run(study_id="some_hash", model_id="a1-roi", custom_input=Path("/some/path"))

    # CLI
    python -m src.run --study patient_1 --model a1-roi
    python -m src.run --input /path/to/dicom --model a3-coverage
"""
import json
import argparse
from datetime import datetime
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import csv

import sys
# Make sure we can import 'src' even if the script is executed directly from another directory
_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from src.backend.paths import upload_dir, work_dir, out_dir, study_id_from_series
from src.backend.registry import load_manifest
from src.backend.pipeline import load, resample, crop_heart, normalize, predict, save_nifti
from src.backend.scoring import score, totals
from src.backend.render import save_slices

def write_csv(rows: list[dict], path: Path):
    if not rows:
        with open(path, "w") as f:
            f.write("No lesions found.\n")
        return
    with open(path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def run(study_id: str, model_id: str, crop: bool = None, progress=None, custom_input: Path = None):
    """Execute the inference pipeline for a given study and model."""
    m = load_manifest(model_id)
    crop = m["crop"] if crop is None else crop
    
    def p(stage, pct):
        if progress: 
            progress(stage, pct)

    w = work_dir(study_id)
    w.mkdir(parents=True, exist_ok=True)
    ct_path = w / "ct.nii.gz"
    
    image = None
    if ct_path.exists():
        p("cached", 0.3)
        image = sitk.ReadImage(str(ct_path))
        # Note: cached image already has RAS orientation applied from prior prep.
    else:
        p("load", 0.05)
        load_path = custom_input if custom_input else upload_dir(study_id)
        image = load(load_path)
        
        p("resample", 0.15)
        image = resample(image, tuple(m["spacing"]))
        
        if crop:
            p("crop", 0.3)
            image = crop_heart(image, m["margin_mm"], fast=m.get("ts_fast", False))
        else:
            print("WARNING: cropping OFF — models trained on cropped volumes.")
            
        # Orient to RAS AFTER cropping to match training pipeline order!
        print("Reorienting to RAS...")
        image = sitk.DICOMOrient(image, m["orientation"])
        
        # Ensure spacing didn't permute in a way that alters the volume area calculations
        spc = image.GetSpacing()
        req_spc = tuple(m["spacing"])
        assert np.allclose(spc, req_spc, atol=1e-4), f"RAS permuted axes: {spc}"
        
        # render.py's flips assume RAS (diag(-1,-1,1)). DICOMOrient guarantees it, but
        # a wrong direction matrix would silently mirror the display, not error.
        assert np.allclose(image.GetDirection(), (-1,0,0, 0,-1,0, 0,0,1), atol=1e-6), \
            f"expected RAS direction cosines, got {image.GetDirection()}"
        
        sitk.WriteImage(image, str(ct_path))

    array = sitk.GetArrayFromImage(image)                 # (Z, Y, X)
    x = normalize(np.transpose(array, (2, 1, 0)), tuple(m["hu_window"])) # (X, Y, Z)
    
    p("predict", 0.5)
    prob_xyz = predict(
        x, 
        m["weights_path"], 
        m["arch"],
        tuple(m["patch"]), 
        m["overlap"], 
        m["activation"]
    )
    prob = np.transpose(prob_xyz, (2, 1, 0)) # back to (Z, Y, X)
    
    o = out_dir(study_id, model_id)
    o.mkdir(parents=True, exist_ok=True)
    
    prob_img = sitk.GetImageFromArray(prob)
    prob_img.CopyInformation(image)
    
    # Save NIfTIs in the RAS orientation
    save_nifti(prob_img, o / "pred.nii.gz")
    save_nifti(image, o / "ct.nii.gz")

    p("score", 0.8)
    rows = score(prob, array, image.GetSpacing(), m["output"], m["threshold"])
    write_csv(rows, o / "lesions.csv")
    summary = totals(rows)
    
    run_provenance = {
        **summary,
        "study_id": study_id,
        "model_id": model_id,
        "cropped": crop,
        "shape": list(array.shape),
        "output": m["output"],
        "hu_window": m["hu_window"],
        "spacing": list(image.GetSpacing()),
        "sha256": m.get("sha256"),
        "locator_version": m.get("locator_version"),
        "date": datetime.now().isoformat()
    }
    
    with open(o / "run.json", "w") as f:
        json.dump(run_provenance, f, indent=2)
        
    p("render", 0.9)
    save_slices(array, prob, o / "slices", m, rows)
    
    p("done", 1.0)
    return o

if __name__ == "__main__":
    import sys
    
    # MANUAL RUN BLOCK (Triggered if no CLI arguments are provided)
    if len(sys.argv) == 1:
        print("No CLI arguments provided. Running in MANUAL mode...")
        
        # EDIT THESE VALUES FOR MANUAL RUNS:
        MANUAL_INPUT_PATH = Path("/pscratch/sd/s/soham95/SOHAM/coca_raw/cocacoronarycalciumandchestcts-2/Gated_release_final/patient/173/Pro_Gated_Calcium_Score_(CS)_3.0_Qr36_2_BestSyst_255_ms")
        MANUAL_MODEL_ID = "a1-roi"
        
        study_id = MANUAL_INPUT_PATH.parent.name # Usually the patient ID folder
        
        def log_progress(stage, pct):
            print(f"[{pct*100:3.0f}%] {stage}")
            
        print(f"Starting manual run for study '{study_id}' with model '{MANUAL_MODEL_ID}'...")
        out_path = run(study_id, MANUAL_MODEL_ID, progress=log_progress, custom_input=MANUAL_INPUT_PATH)
        print(f"Done. Outputs saved to {out_path}")
        sys.exit(0)
        
    # EXISTING CLI LOGIC
    parser = argparse.ArgumentParser(description="Run PrediCT CAC inference pipeline.")
    parser.add_argument("--study", help="Study ID (if using data/uploads/<study_id>)")
    parser.add_argument("--input", help="Absolute path to a DICOM directory (bypasses data/uploads/)")
    parser.add_argument("--model", required=True, help="Model ID (e.g., a1-roi)")
    parser.add_argument("--crop", action="store_true", default=None, help="Force crop=True")
    parser.add_argument("--no-crop", dest="crop", action="store_false", help="Force crop=False")
    
    args = parser.parse_args()
    
    if args.input:
        study_id = Path(args.input).name
        input_path = Path(args.input)
    elif args.study:
        study_id = args.study
        input_path = upload_dir(args.study)
    else:
        parser.error("Must provide either --study or --input")
        
    print(f"Starting run for study '{study_id}' with model '{args.model}'...")
    def log_progress(stage, pct):
        print(f"[{pct*100:3.0f}%] {stage}")
        
    out_path = run(study_id, args.model, crop=args.crop, progress=log_progress, custom_input=input_path)
    print(f"Done. Outputs saved to {out_path}")
