"""
registry.py — Model manifest loading and validation.

Reads manifest.yaml for a given model ID. Validates that all required fields
are present and that the checkpoint SHA256 matches the manifest precisely,
preventing silent model drift.

Does NOT: load PyTorch weights, run models, or instantiate architectures.
Called by: run.py, server.py.

Usage:
    manifest = load_manifest("approach1_roi_cropped")
    models = list_models()
"""
import yaml
import hashlib
from .paths import MODELS

REQUIRED_COMMON = ["id", "name", "output", "threshold", "hu_window",
                   "spacing", "crop", "arch"]
REQUIRED_PYTORCH = ["weights", "activation", "patch", "overlap"]

def load_manifest(model_id: str) -> dict:
    """Load and validate a model manifest."""
    d = MODELS / model_id
    manifest_path = d / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found for model {model_id} at {manifest_path}")
        
    m = yaml.safe_load(manifest_path.read_text()) or {}
    
    req = REQUIRED_COMMON.copy()
    if m.get("arch") != "nnunet":
        req.extend(REQUIRED_PYTORCH)
        
    missing = [k for k in req if k not in m or m[k] == "REQUIRED"]
    if missing:
        raise ValueError(f"{model_id}: manifest missing {missing}")
        
    if m.get("arch") != "nnunet":
        m["weights_path"] = d / m["weights"]
        if m.get("sha256"):
            got = hashlib.sha256(m["weights_path"].read_bytes()).hexdigest()
            if got != m["sha256"]:
                raise ValueError(f"{model_id}: checkpoint sha256 mismatch (expected {m['sha256']}, got {got})")
    else:
        m["nnunet_results_path"] = d / m.get("nnunet_results", "nnUNet_results")
            
    return m

def list_models() -> list[dict]:
    """List all valid models in the models directory."""
    if not MODELS.exists():
        return []
    return [load_manifest(d.name) for d in MODELS.iterdir() 
            if d.is_dir() and (d / "manifest.yaml").exists()]
