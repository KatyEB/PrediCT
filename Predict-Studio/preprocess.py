"""
preprocess.py — blocks 5 to 8. Turn a raw Volume into what a model needs.

    orient -> RAS        block 5
    resample             block 6
    locate heart         block 7   TotalSegmentator, on RAW HU
    crop + margin        block 8

ORDER MATTERS AND THE ORIGINAL DIAGRAM HAD IT WRONG
    TotalSegmentator does its own normalisation and expects real Hounsfield
    Units. Windowing to [0, 1200] before locating the heart would break it.
    So windowing is NOT here. It happens per-model, in memory, in models.py.

WHY THIS RUNS ONCE PER PATIENT
    Every model that declares the same spacing and ROI shares this output.
    66 patients x 3 models: without caching that is 198 heart locations at
    ~30 s each. With caching it is 66.

Everything here returns a new Volume and records what it did in .history, so
models.py can check that what was actually done matches what a model requires.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from load import Volume


# ---------------------------------------------------------------------------
# Block 5 — orientation
# ---------------------------------------------------------------------------

def to_ras(volume: Volume) -> Volume:
    """Reorient to RAS (x right, y anterior, z superior).

    Cheap, and silently catastrophic if skipped: a volume stored LPS looks
    completely normal but has left and right swapped, so a lesion in the LAD
    is reported in the circumflex.
    """
    image = _to_sitk(volume)
    current = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )
    if current == "RAS":
        return _record(volume, "orient", {"from": current, "to": "RAS", "noop": True})

    oriented = sitk.DICOMOrient(image, "RAS")
    out = _from_sitk(oriented, volume)
    return _record(out, "orient", {"from": current, "to": "RAS"})


# ---------------------------------------------------------------------------
# Block 6 — resampling
# ---------------------------------------------------------------------------

def resample(volume: Volume, target_spacing: tuple[float, float, float],
             interpolator=sitk.sitkLinear) -> Volume:
    """Resample to a target voxel spacing in mm, given as (x, y, z).

    This is not cosmetic. A convolution's receptive field is fixed in VOXELS,
    so a 3 mm lesion is about 8 voxels wide at 0.37 mm spacing and about 4.4
    at 0.68 mm. To the model the lesion has physically changed size.

    Warns on large upsampling, because interpolating 5 mm slices up to 3 mm
    invents detail that was never acquired.
    """
    current = volume.spacing
    if all(abs(c - t) < 1e-6 for c, t in zip(current, target_spacing)):
        return _record(volume, "resample",
                       {"spacing": list(target_spacing), "noop": True})

    image = _to_sitk(volume)
    size = image.GetSize()

    new_size = [
        int(round(size[i] * current[i] / target_spacing[i])) for i in range(3)
    ]

    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing([float(s) for s in target_spacing])
    rs.SetSize(new_size)
    rs.SetOutputDirection(image.GetDirection())
    rs.SetOutputOrigin(image.GetOrigin())
    rs.SetInterpolator(interpolator)
    # Air, so padding never looks like tissue.
    rs.SetDefaultPixelValue(float(volume.array.min()))

    out = _from_sitk(rs.Execute(image), volume)

    warnings = []
    for axis, (c, t) in enumerate(zip(current, target_spacing)):
        if c / t > 1.5:
            warnings.append(
                f"axis {'xyz'[axis]} upsampled {c / t:.1f}x "
                f"({c:.2f} -> {t:.2f} mm); detail is interpolated, not measured"
            )
    out.warnings.extend(warnings)

    return _record(out, "resample", {
        "from": list(current), "to": list(target_spacing),
        "size_from": list(size), "size_to": new_size,
    })


# ---------------------------------------------------------------------------
# Block 7 — locate the heart
# ---------------------------------------------------------------------------

def locate_heart(volume: Volume, quality: str = "fast") -> np.ndarray:
    """Binary mask of the heart, from TotalSegmentator, on RAW HU.

    Imported lazily so the rest of the pipeline stays runnable without it.
    Raises a clear error rather than silently falling back, because a silent
    fallback to "the whole chest is the heart" would produce a plausible and
    completely wrong score.
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as e:
        raise RuntimeError(
            "TotalSegmentator is not installed, so the heart cannot be located.\n"
            "  pip install TotalSegmentator\n"
            "Or run with roi: none, which feeds the whole volume to the model — "
            "valid only for a model trained on uncropped data."
        ) from e

    image = _to_sitk(volume)
    result = totalsegmentator(
        image, task="total", roi_subset=["heart"], fast=(quality == "fast"),
        quiet=True,
    )
    mask = sitk.GetArrayFromImage(result) > 0

    if not mask.any():
        raise RuntimeError(
            f"TotalSegmentator found no heart in {volume.patient_id}. "
            "The scan may not cover the chest, or may not be a CT."
        )
    return mask


def bounding_box(mask: np.ndarray, spacing: tuple[float, float, float],
                 margin_mm: float = 8.0) -> tuple[slice, slice, slice]:
    """Bounding box of a mask, expanded by a physical margin.

    The margin is in millimetres, not voxels, so it means the same thing at
    every spacing. Converted per axis because voxels are not cubes: 8 mm is
    about 22 voxels in-plane at 0.37 mm but under 3 slices at 3.0 mm.
    """
    zs, ys, xs = np.nonzero(mask)
    if zs.size == 0:
        raise ValueError("mask is empty; nothing to crop to")

    sx, sy, sz = spacing
    pad_z = int(np.ceil(margin_mm / sz))
    pad_y = int(np.ceil(margin_mm / sy))
    pad_x = int(np.ceil(margin_mm / sx))

    def span(idx, pad, limit):
        return slice(max(0, int(idx.min()) - pad),
                     min(limit, int(idx.max()) + pad + 1))

    return (span(zs, pad_z, mask.shape[0]),
            span(ys, pad_y, mask.shape[1]),
            span(xs, pad_x, mask.shape[2]))


def crop(volume: Volume, box: tuple[slice, slice, slice]) -> Volume:
    """Crop to a bounding box, recording it so masks can be pasted back later."""
    z, y, x = box
    out = replace(volume, array=volume.array[z, y, x].copy())
    out.warnings = list(volume.warnings)
    out.history = list(getattr(volume, "history", []))
    return _record(out, "crop", {
        "box": [[z.start, z.stop], [y.start, y.stop], [x.start, x.stop]],
        "shape_from": list(volume.array.shape),
        "shape_to": list(out.array.shape),
    })


def paste_back(cropped_mask: np.ndarray, full_shape: tuple[int, int, int],
               box: tuple[slice, slice, slice]) -> np.ndarray:
    """Put a mask computed on a crop back into full-volume coordinates.

    Needed for display and for export, so a saved mask lines up with the
    original scan rather than with an intermediate the user never saw.
    """
    full = np.zeros(full_shape, dtype=cropped_mask.dtype)
    full[box] = cropped_mask
    return full


# ---------------------------------------------------------------------------
# The chain, with caching
# ---------------------------------------------------------------------------

def preprocess(volume: Volume, requires: dict, cache_dir: Path | None = None,
               locator_quality: str = "fast") -> Volume:
    """Run blocks 5-8 according to a model's `requires` block.

    `requires` comes straight from a model YAML:

        spacing_mm: [0.37, 0.37, 3.0]
        orientation: RAS
        roi: heart
        roi_margin_mm: 8

    Caching is keyed on the patient, the series, and the requirements — so
    changing the spacing produces a different key rather than a stale hit.
    Two models with identical requirements share one cached result.
    """
    key = _cache_key(volume, requires)

    if cache_dir is not None:
        cached = Path(cache_dir) / f"{key}.npz"
        if cached.exists():
            return _load_cached(cached)

    out = volume

    if requires.get("orientation", "RAS") == "RAS":
        out = to_ras(out)

    if requires.get("spacing_mm"):
        out = resample(out, tuple(requires["spacing_mm"]))

    roi = requires.get("roi", "none")
    if roi == "heart":
        heart = locate_heart(out, quality=locator_quality)
        box = bounding_box(heart, out.spacing, requires.get("roi_margin_mm", 8.0))
        out = crop(out, box)
    elif roi not in ("none", None):
        raise ValueError(f"unknown roi {roi!r}; expected 'heart' or 'none'")

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _save_cached(Path(cache_dir) / f"{key}.npz", out)

    return out


def _cache_key(volume: Volume, requires: dict) -> str:
    payload = json.dumps({
        "patient": volume.patient_id,
        "series": volume.series_id,
        "shape": list(volume.array.shape),
        "spacing": list(volume.spacing),
        "requires": {k: requires[k] for k in sorted(requires)},
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _save_cached(path: Path, volume: Volume) -> None:
    np.savez_compressed(
        path,
        array=volume.array,
        meta=json.dumps({
            "spacing": list(volume.spacing),
            "patient_id": volume.patient_id,
            "series_id": volume.series_id,
            "source": str(volume.source),
            "warnings": volume.warnings,
            "history": getattr(volume, "history", []),
        }),
    )


def _load_cached(path: Path) -> Volume:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    volume = Volume(
        array=data["array"],
        spacing=tuple(meta["spacing"]),
        patient_id=meta["patient_id"],
        series_id=meta["series_id"],
        source=Path(meta["source"]),
        warnings=list(meta["warnings"]),
    )
    volume.history = list(meta["history"])
    return volume


# ---------------------------------------------------------------------------
# SimpleITK bridging and history
# ---------------------------------------------------------------------------

def _to_sitk(volume: Volume) -> sitk.Image:
    image = sitk.GetImageFromArray(volume.array)
    image.SetSpacing([float(s) for s in volume.spacing])
    return image


def _from_sitk(image: sitk.Image, template: Volume) -> Volume:
    out = replace(
        template,
        array=sitk.GetArrayFromImage(image),
        spacing=tuple(float(s) for s in image.GetSpacing()),
    )
    out.warnings = list(template.warnings)
    out.history = list(getattr(template, "history", []))
    return out


def _record(volume: Volume, step: str, detail: dict) -> Volume:
    """Append to the volume's history.

    This is what block 10 checks against a model's `requires`. Without it the
    gate would have to trust that preprocessing did what it was asked, which
    is exactly the assumption that let the HU-window mismatch survive.
    """
    if not hasattr(volume, "history"):
        volume.history = []
    volume.history.append({"step": step, **detail})
    return volume


def provenance(volume: Volume) -> dict:
    """Flatten history into the form the contract gate compares against."""
    out = {
        "spacing_mm": [round(s, 6) for s in volume.spacing],
        "orientation": "unknown",
        "roi": "none",
        "n_slices": int(volume.array.shape[0]),
        "hu_range": [float(volume.array.min()), float(volume.array.max())],
    }
    for entry in getattr(volume, "history", []):
        if entry["step"] == "orient":
            out["orientation"] = entry["to"]
        elif entry["step"] == "crop":
            out["roi"] = "heart"
            out["roi_box"] = entry["box"]
    return out
