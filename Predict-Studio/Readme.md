# PrediCT CAC Pipeline & Studio

## What is this?
PrediCT is an automated machine learning pipeline and web application for computing the Coronary Artery Calcium (CAC) Agatston score from raw cardiac CT volumes. 
It rigorously scores lesions using established PyTorch models trained on annotated calcium data.

## The Pipeline in Ten Lines
1. A single DICOM study is mapped to a robust SHA1 `study_id`.
2. The DICOM series is loaded into a native 3D SimpleITK image.
3. The volume is resampled to precisely `0.37 x 0.37 x 3.0` mm spacing.
4. `TotalSegmentator` detects the heart and crops the volume with an 8mm margin.
5. The cropped volume is explicitly reoriented to the `RAS` axis.
6. Hounsfield Units are normalized based on model training bounds (e.g., `0 to 1200`).
7. Inference runs using a 3D UNet with sliding windows.
8. Component analysis extracts lesion areas multiplied by classical density weights.
9. Results are packaged into a lesion-by-lesion CSV alongside a detailed `run.json` provenance.
10. Slices with prediction overlays are rendered for diagnostic visualization.

## Folder Layout
```text
predict_software/Predict-Studio/
├── src/
│   └── backend/
│       ├── paths.py      # Centralized DICOM hashing and path registry
│       ├── pipeline.py   # Pure imaging routines (load, resample, crop, inference)
│       ├── scoring.py    # Agatston math & connected components (no torch, no file I/O)
│       ├── render.py     # PIL PNG slice overlays
│       ├── registry.py   # YAML manifest validation & SHA256 locking
│       ├── run.py        # Pipeline orchestrator & CLI entrypoint
│       └── server.py     # FastAPI server providing the web UI and REST API
├── ui/                   # Frontend assets for PrediCT Studio (HTML, JS, CSS)
├── models/
│   ├── a1-roi/           # Binary masking Approach 1
│   └── a3-coverage-v2/   # Continuous probability Coverage Approach 3 (v2)
└── data/                 # Root output for inference jobs (uploads, work, out)
```

## How to Run PrediCT Studio (Web UI)
The easiest way to interact with PrediCT is via the included web UI.
From the `predict_software/Predict-Studio/` directory, start the FastAPI server:
```bash
python -m src.backend.server
```
Then navigate to `http://127.0.0.1:8000` in your browser. You can upload DICOM files directly through the web interface, run models, and view the results.

## How to Run via CLI
Place a patient's DICOM folder into `data/uploads/<patient_id>`, then run the orchestrator:
```bash
python -m src.backend.run --study <patient_id> --model a1-roi
```
Alternatively, bypass the `data/uploads` directory by providing an absolute path:
```bash
python -m src.backend.run --input /path/to/dicom --model a3-coverage-v2
```
Results, including `pred.nii.gz`, `ct.nii.gz`, `lesions.csv`, `run.json`, and PNG `slices/` will be generated in `data/out/<patient_id>/<model_id>`.

## Models & Manifests
The PyTorch checkpoints live in the `models/` directory, grouped by approach.
Every approach requires a `manifest.yaml` (a manifest).
Because raw model weights (`best_model.pth`) cannot store semantic configurations (like HU windows, target spacing, or whether the output represents binary vs. coverage probabilities), this information is explicitly defined in `manifest.yaml`. 
The `registry.py` locks this file to the `.pth` by computing a SHA256 hash on load, preventing any silent training/inference mismatches.
