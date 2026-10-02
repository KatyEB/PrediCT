"""
paths.py — Centralized path definitions and study ID generation.

Provides absolute paths for all data and models. Ensures no other file builds
paths by hand. Generates deterministic study IDs from DICOM metadata so that
re-uploading the same series produces the same ID.

Does NOT: read images (only metadata), create directories, or manage models.
Called by: run.py, registry.py, server.py.

Usage:
    from .paths import MODELS, work_dir, out_dir, safe_name, study_dirs
"""
from pathlib import Path
import hashlib
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

# Characters that would let a name leave its folder (separators, drive colon)
# or that Windows forbids in file names.
_FORBIDDEN = set('/\\:*?"<>|')

def safe_name(name: str) -> str:
    """Return `name` trimmed if it is usable as ONE folder name, else raise ValueError.

    Every study / model / temp id that arrives over HTTP becomes a folder under
    DATA, so it must not contain separators or '..' tricks. Ordinary names
    ("Patient 11", "Pro_Gated_CS_3.0_I30f_3_70%") pass unchanged.
    """
    name = (name or "").strip()
    if (not name or name.startswith(".")
            or any(c in _FORBIDDEN or ord(c) < 32 for c in name)):
        raise ValueError(f'invalid name {name!r}: must be a single folder name, '
                         'not starting with "." and without / \\ : * ? " < > |')
    return name

def upload_dir(study_id: str) -> Path:
    return DATA / "uploads" / study_id

def raw_dir(study_id: str) -> Path:
    """Where an uploaded scan is kept (server.py upload endpoints)."""
    return DATA / "raw" / study_id

def work_dir(study_id: str) -> Path:
    return DATA / "work" / study_id

def out_dir(study_id: str, model_id: str) -> Path:
    return DATA / "out" / study_id / model_id

def study_dirs(study_id: str) -> list[Path]:
    """Every folder that belongs to one study: uploaded scan, prep cache, results.

    This is the one place that decides what a study owns. When studies become
    per-user, the user is added here (and in raw_dir / work_dir / out_dir) and
    every caller — listing, running, deleting — follows.
    """
    return [raw_dir(study_id), work_dir(study_id), DATA / "out" / study_id]

def study_id_from_series(input_dir: str | Path) -> str:
    """Generate a consistent 12-char ID from the DICOM SeriesInstanceUID or NIfTI filename."""
    input_dir = Path(input_dir)
    try:
        first_dcm = next(input_dir.rglob("*.dcm"))
        r = sitk.ImageFileReader()
        r.SetFileName(str(first_dcm))
        r.ReadImageInformation()
        uid = r.GetMetaData("0020|000e")  # SeriesInstanceUID
        return hashlib.sha1(uid.encode()).hexdigest()[:12]
    except StopIteration:
        nifti_files = list(input_dir.rglob("*.nii")) + list(input_dir.rglob("*.nii.gz"))
        if not nifti_files:
            raise StopIteration("No DICOM or NIfTI files found.")
        filename = nifti_files[0].name
        return hashlib.sha1(filename.encode()).hexdigest()[:12]
