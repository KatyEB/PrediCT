"""
paths.py — Centralized path definitions and study ID generation.

Provides absolute paths for all data and models. Ensures no other file builds
paths by hand. Generates deterministic study IDs from DICOM metadata so that
re-uploading the same series produces the same ID.

Does NOT: read images (only metadata), create directories, or manage models.
Called by: run.py, registry.py, server.py.

Usage:
    from .paths import MODELS, user_root, work_dir, out_dir, safe_name, study_dirs
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

# ── per-account folders ───────────────────────────────────────────────────
# Everything an account owns lives under data/users/<id>/. Ownership IS the
# location: there is no table mapping studies to users, so nothing can drift.
# Every function below takes the account id first; no caller builds these
# paths by hand.
#
#   data/users/<id>/raw/<scan>/            uploaded scan (DICOM or NIfTI)
#   data/users/<id>/work/<study>/crop|full prep cache (resampled CT, heart mask)
#   data/users/<id>/out/<study>/<model>/   results the UI renders
#   data/users/<id>/tmp/<temp_id>/         an upload being checked

def user_root(user_id: int) -> Path:
    """The account's own folder. int() guarantees a plain number, never a path."""
    return DATA / "users" / str(int(user_id))

def raw_dir(user_id: int, study_id: str) -> Path:
    """Where an uploaded scan is kept (server.py upload endpoints)."""
    return user_root(user_id) / "raw" / study_id

def work_dir(user_id: int, study_id: str) -> Path:
    return user_root(user_id) / "work" / study_id

def out_dir(user_id: int, study_id: str, model_id: str) -> Path:
    return user_root(user_id) / "out" / study_id / model_id

def tmp_dir(user_id: int, temp_id: str) -> Path:
    return user_root(user_id) / "tmp" / temp_id

def study_dirs(user_id: int, study_id: str) -> list[Path]:
    """Every folder that belongs to one study of one account: uploaded scan,
    prep cache, results. "Delete everything" removes exactly these."""
    return [raw_dir(user_id, study_id), work_dir(user_id, study_id),
            user_root(user_id) / "out" / study_id]

def scan_input_dir(user_id: int, scan: str) -> Path:
    """The folder a run reads for an uploaded scan: the series sub-folder if the
    upload kept one, else the scan folder itself."""
    d = raw_dir(user_id, scan)
    subs = sorted(p for p in d.iterdir() if p.is_dir())
    return subs[0] if subs else d

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
