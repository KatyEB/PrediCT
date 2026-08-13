"""
paths.py — Centralized path definitions and study ID generation.

Provides absolute paths for all data and models. Ensures no other file builds
paths by hand. Generates deterministic study IDs from DICOM metadata so that
re-uploading the same series produces the same ID.

Does NOT: read images (only metadata), create directories, or manage models.
Called by: run.py, registry.py, server.py.

Usage:
    from .paths import MODELS, work_dir, out_dir
"""
from pathlib import Path
import hashlib
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

def upload_dir(study_id: str) -> Path: 
    return DATA / "uploads" / study_id

def work_dir(study_id: str) -> Path:   
    return DATA / "work" / study_id

def out_dir(study_id: str, model_id: str) -> Path: 
    return DATA / "out" / study_id / model_id

def study_id_from_series(dicom_dir: str | Path) -> str:
    """Generate a consistent 12-char ID from the DICOM SeriesInstanceUID."""
    r = sitk.ImageFileReader()
    r.SetFileName(str(next(Path(dicom_dir).rglob("*.dcm"))))
    r.ReadImageInformation()
    uid = r.GetMetaData("0020|000e")  # SeriesInstanceUID
    return hashlib.sha1(uid.encode()).hexdigest()[:12]
