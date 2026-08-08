"""
load.py — point this at any folder, get volumes out.

    python load.py C:/SOHAM/coca_raw/.../patient/0
    python load.py C:/some/random/folder

This is step 1 of the pipeline. It does one job: turn a folder a user dropped
into a list of Volumes, each with correct pixel data and correct spacing.

Three things it does differently from COCA_processor_main.py, and why:

  1. Finds DICOM by MAGIC BYTES, not by "*.dcm".
     Real hospital exports often have no file extension at all. rglob("*.dcm")
     finds zero files in those folders.

  2. Groups by SeriesInstanceUID, not by directory.
     Two series in one folder currently get stacked into one broken volume.
     This is how patient 159 ended up appearing twice.

  3. Reads PatientID from the DICOM tag, not from folder depth.
     A stranger's folder tree has no fixed depth to count from.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

# DICOM files carry the four bytes "DICM" at offset 128. This is the only
# reliable way to identify one; the extension is a convention, not a rule.
DICOM_MAGIC_OFFSET = 128
DICOM_MAGIC = b"DICM"

# Extensions SimpleITK can read directly as a whole volume.
VOLUME_SUFFIXES = {".nii", ".gz", ".nrrd", ".mha", ".mhd", ".img", ".hdr"}


@dataclass
class Volume:
    """One loaded 3-D image, with everything needed to interpret it."""

    array: np.ndarray                # (Z, Y, X) in raw Hounsfield Units
    spacing: tuple[float, float, float]   # (x, y, z) in mm
    patient_id: str
    series_id: str
    source: Path
    n_files: int = 1
    warnings: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    """What preprocessing did to this volume. Block 10 checks it."""

    def describe(self) -> str:
        z, y, x = self.array.shape
        sx, sy, sz = self.spacing
        return (
            f"patient {self.patient_id}  series {self.series_id[:20]}\n"
            f"    shape       {z} slices of {y} x {x}\n"
            f"    spacing     {sx:.4f} x {sy:.4f} x {sz:.4f} mm\n"
            f"    HU range    {self.array.min():.0f} to {self.array.max():.0f}\n"
            f"    files       {self.n_files}"
        )


# ---------------------------------------------------------------------------
# Finding files
# ---------------------------------------------------------------------------

def is_dicom(path: Path) -> bool:
    """True if the file has the DICM magic bytes, whatever it is named."""
    try:
        with open(path, "rb") as f:
            f.seek(DICOM_MAGIC_OFFSET)
            return f.read(4) == DICOM_MAGIC
    except (OSError, ValueError):
        return False


def find_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Walk a folder. Return (dicom_files, volume_files).

    Everything else is ignored — PNGs, spreadsheets, notes. A CT scan cannot
    be recovered from a PNG because the Hounsfield values are already gone.
    """
    dicoms: list[Path] = []
    volumes: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in VOLUME_SUFFIXES:
            volumes.append(path)
        elif is_dicom(path):
            dicoms.append(path)

    return dicoms, volumes


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------

def group_by_series(files: list[Path]) -> dict[str, list[Path]]:
    """Group DICOM files by SeriesInstanceUID.

    Two different scans of the same patient sitting in one folder is common,
    and stacking them together produces a volume with duplicated or jumbled
    slice positions. Grouping by the UID is what the tag is for.
    """
    groups: dict[str, list[Path]] = {}

    for path in files:
        try:
            # stop_before_pixels reads only the header — fast enough to do
            # this for thousands of files.
            header = pydicom.dcmread(path, stop_before_pixels=True)
            uid = str(getattr(header, "SeriesInstanceUID", "unknown"))
        except Exception:
            continue
        groups.setdefault(uid, []).append(path)

    return groups


def load_dicom_series(files: list[Path]) -> Volume:
    """Load one series into a Volume, sorted by physical position.

    SimpleITK's series reader sorts by ImagePositionPatient projected onto the
    slice normal — the slice's actual location in space. InstanceNumber is a
    counter that can restart or run backwards, so it is not used.

    The reader also applies RescaleSlope and RescaleIntercept, which is what
    converts stored pixel values into Hounsfield Units.
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in files])
    image = reader.Execute()

    header = pydicom.dcmread(files[0], stop_before_pixels=True)
    patient_id = str(getattr(header, "PatientID", "")).strip()
    series_id = str(getattr(header, "SeriesInstanceUID", "unknown"))

    warnings: list[str] = []
    if not patient_id:
        patient_id = files[0].parent.name
        warnings.append(f"no PatientID tag; using folder name '{patient_id}'")

    volume = Volume(
        array=sitk.GetArrayFromImage(image),   # SimpleITK gives (Z, Y, X)
        spacing=tuple(float(s) for s in image.GetSpacing()),  # (x, y, z)
        patient_id=patient_id,
        series_id=series_id,
        source=files[0].parent,
        n_files=len(files),
        warnings=warnings,
    )
    volume.warnings.extend(check(volume, header))
    return volume


# ---------------------------------------------------------------------------
# NIfTI / NRRD / MHA
# ---------------------------------------------------------------------------

def load_volume_file(path: Path) -> Volume:
    """Load a single-file volume. SimpleITK handles all these formats natively."""
    image = sitk.ReadImage(str(path))

    # Strip .nii.gz properly — .stem alone leaves "scan.nii".
    name = path.name
    for suffix in (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    volume = Volume(
        array=sitk.GetArrayFromImage(image),
        spacing=tuple(float(s) for s in image.GetSpacing()),
        patient_id=name,
        series_id=name,
        source=path,
        warnings=["identity taken from filename; this format has no patient tag"],
    )
    volume.warnings.extend(check(volume, None))
    return volume


# ---------------------------------------------------------------------------
# Sanity checks — warnings, never refusals
# ---------------------------------------------------------------------------

def check(volume: Volume, header) -> list[str]:
    """Flag anything that will make a downstream number untrustworthy.

    These are warnings, not errors. Refusing to load a scan because it is
    unusual is worse than loading it and saying so.
    """
    out: list[str] = []
    lo, hi = float(volume.array.min()), float(volume.array.max())

    # Real CT spans roughly -1000 (air) to +3000 (dense bone / metal).
    if lo >= -100:
        out.append(
            f"minimum value is {lo:.0f}; real CT reaches about -1000 in air. "
            "This may not be in Hounsfield Units."
        )
    if hi <= 100:
        out.append(f"maximum value is {hi:.0f}; this looks normalised, not raw HU.")

    if volume.array.shape[0] < 32:
        out.append(
            f"only {volume.array.shape[0]} slices; 3-D models need at least 32."
        )

    if header is not None:
        modality = str(getattr(header, "Modality", "")).strip()
        if modality and modality != "CT":
            out.append(f"modality is {modality}, not CT.")

        kvp = getattr(header, "KVP", None)
        if kvp and abs(float(kvp) - 120) > 1:
            out.append(
                f"acquired at {float(kvp):.0f} kVp. Agatston thresholds are "
                "defined at 120 kVp; scores will be biased."
            )

        # Iodine contrast in the coronary lumen sits at 300-400 HU, which is
        # indistinguishable from calcium by threshold. CAC scoring is defined
        # on non-contrast scans only.
        if str(getattr(header, "ContrastBolusAgent", "")).strip():
            out.append(
                "contrast agent recorded. Agatston scoring is only valid on "
                "non-contrast scans; the score will be large and meaningless."
            )

    return out


# ---------------------------------------------------------------------------
# The one function the rest of the pipeline calls
# ---------------------------------------------------------------------------

def load_folder(root: str | Path) -> list[Volume]:
    """Load every scan found under a folder. One patient or two hundred."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    if root.is_file():
        return [load_volume_file(root)] if root.suffix.lower() in VOLUME_SUFFIXES else []

    dicoms, volume_files = find_files(root)
    volumes: list[Volume] = []

    for uid, files in group_by_series(dicoms).items():
        if len(files) < 2:
            continue          # a single slice is not a volume
        try:
            volumes.append(load_dicom_series(files))
        except Exception as e:
            print(f"  [skip] series {uid[:20]}: {e}", file=sys.stderr)

    for path in volume_files:
        try:
            volumes.append(load_volume_file(path))
        except Exception as e:
            print(f"  [skip] {path.name}: {e}", file=sys.stderr)

    return volumes


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    root = Path(sys.argv[1])
    print(f"\nScanning {root}\n")

    volumes = load_folder(root)
    if not volumes:
        print("No loadable scans found.")
        print("Looked for: DICOM (by magic bytes) and .nii/.nii.gz/.nrrd/.mha")
        sys.exit(1)

    print(f"Found {len(volumes)} scan(s)\n")
    for i, volume in enumerate(volumes, 1):
        print(f"[{i}] {volume.describe()}")
        for warning in volume.warnings:
            print(f"    warning     {warning}")
        print()


if __name__ == "__main__":
    main()
