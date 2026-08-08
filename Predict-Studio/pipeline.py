"""
pipeline.py — the spine. One patient in, one score out.

    load -> preprocess -> gate -> infer -> score -> write

This is the file to read if you want to understand the system. Everything else
is a block it calls. It is deliberately one function you can follow top to
bottom, because a pipeline you cannot trace is a pipeline you cannot debug.

ONE PATIENT AND SIXTY-SIX ARE THE SAME CODE
    A cohort is a loop over process_study(). There is no separate cohort path,
    which is how end_to_end_inference_visualizer.py and agatston_scoring_a3.py
    drifted to different HU windows for the same operation.

RESUMABILITY
    Each study writes its own row on completion. Kill the run at patient 60 and
    the first 59 are on disk. The current scripts accumulate in a list and only
    write at the end, so a crash loses everything.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import models as model_lib
from load import Volume
from preprocess import paste_back, preprocess, provenance
from scoring import Score, ScoringConfig, Spacing, score_volume


@dataclass
class StudyResult:
    """Everything one patient-model pair produced."""

    patient_id: str
    series_id: str
    model_id: str
    status: str                      # ok | refused | failed
    score: Score | None = None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mask_path: Path | None = None
    seconds: float = 0.0

    def row(self) -> dict:
        """One flat line for the results CSV."""
        out = {
            "patient_id": self.patient_id,
            "series_id": self.series_id,
            "model_id": self.model_id,
            "status": self.status,
            "seconds": round(self.seconds, 2),
            "problems": " | ".join(self.problems),
            "warnings": " | ".join(self.warnings),
            "mask_path": str(self.mask_path) if self.mask_path else "",
        }
        if self.score is not None:
            out.update(self.score.summary())
        return out


def process_study(
    volume: Volume,
    model: model_lib.Model,
    scoring_config: ScoringConfig | None = None,
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
    keep: str = "results",
    device: str = "auto",
    force: bool = False,
) -> StudyResult:
    """Score one volume with one model.

    keep : "all"          write mask, ledger and per-study JSON
           "results"      write mask and ledger  (default)
           "scores-only"  write nothing; the CSV row is the only output

    force : run even if the contract gate objects. Every objection is then
            recorded in the result and stamped on the output, so an override is
            never invisible.
    """
    started = datetime.now(timezone.utc)
    scoring_config = scoring_config or ScoringConfig()

    result = StudyResult(
        patient_id=volume.patient_id,
        series_id=volume.series_id,
        model_id=model.id,
        status="failed",
        warnings=list(volume.warnings),
    )

    try:
        # ---- blocks 5-8 -------------------------------------------------
        prepared = preprocess(volume, model.requires, cache_dir=cache_dir)
        result.warnings.extend(
            w for w in prepared.warnings if w not in result.warnings
        )

        # ---- block 10: the gate -----------------------------------------
        prov = provenance(prepared)
        problems = model_lib.check_contract(prov, model)
        result.problems = problems

        if problems and not force:
            result.status = "refused"
            result.seconds = _elapsed(started)
            return result

        # Domain shift satisfies every contract field and still ruins a score.
        for w in model_lib.check_protocol(volume.warnings):
            if w not in result.warnings:
                result.warnings.append(w)

        # ---- blocks 11-12 -----------------------------------------------
        mask = model_lib.infer(prepared.array, model, device=device)

        if mask.shape != prepared.array.shape:
            raise RuntimeError(
                f"model returned shape {mask.shape} for input "
                f"{prepared.array.shape}"
            )

        # ---- blocks 13-14 -----------------------------------------------
        spacing = Spacing(*prepared.spacing)
        mask_kind = "soft" if model.output_type == "soft" else "binary"
        score = score_volume(mask, prepared.array, spacing,
                             scoring_config, mask_kind=mask_kind)

        result.score = score
        result.status = "ok"

        # ---- block 15: write --------------------------------------------
        if out_dir is not None and keep != "scores-only":
            result.mask_path = _write_outputs(
                out_dir, volume, prepared, model, mask, score, prov, keep
            )

    except Exception as e:
        result.status = "failed"
        result.problems.append(f"{type(e).__name__}: {e}")
        if _DEBUG:
            traceback.print_exc()

    result.seconds = _elapsed(started)
    return result


def _write_outputs(out_dir: Path, volume: Volume, prepared: Volume,
                   model: model_lib.Model, mask: np.ndarray, score: Score,
                   prov: dict, keep: str) -> Path:
    """Write the mask, the lesion ledger, and a record of how it was made.

    The mask is written as .nii.gz float32 in the ORIGINAL volume's geometry,
    not the cropped one, so it opens aligned with the source scan in any
    viewer. PNG is never used: it cannot carry spacing, and it quantises a
    coverage fraction to 8 bits.
    """
    study_dir = Path(out_dir) / volume.patient_id / model.id
    study_dir.mkdir(parents=True, exist_ok=True)

    full = mask
    box = prov.get("roi_box")
    if box is not None:
        slices = tuple(slice(a, b) for a, b in box)
        original_shape = _shape_before_crop(prepared)
        if original_shape is not None:
            full = paste_back(mask, original_shape, slices)

    image = sitk.GetImageFromArray(full.astype(np.float32))
    image.SetSpacing([float(s) for s in prepared.spacing])
    mask_path = study_dir / "mask.nii.gz"
    sitk.WriteImage(image, str(mask_path))

    import csv
    with open(study_dir / "lesions.csv", "w", newline="") as f:
        rows = score.ledger()
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys() if rows else ["lesion_id"]
        )
        writer.writeheader()
        writer.writerows(rows)

    if keep == "all":
        (study_dir / "record.json").write_text(json.dumps({
            "patient_id": volume.patient_id,
            "series_id": volume.series_id,
            "source": str(volume.source),
            "model": {
                "id": model.id,
                "manifest": str(model.manifest_path),
                "weights": str(model.weights_path),
                "sha256": model.sha256,
                "output_type": model.output_type,
            },
            "preprocessing": prepared.history,
            "provenance": prov,
            "score": score.summary(),
            "warnings": volume.warnings,
        }, indent=2, default=str))

    return mask_path


def _shape_before_crop(volume: Volume) -> tuple[int, int, int] | None:
    for entry in reversed(getattr(volume, "history", [])):
        if entry["step"] == "crop":
            return tuple(entry["shape_from"])
    return None


def _elapsed(started: datetime) -> float:
    return (datetime.now(timezone.utc) - started).total_seconds()


_DEBUG = False


def set_debug(on: bool) -> None:
    """Print full tracebacks instead of one-line failure messages."""
    global _DEBUG
    _DEBUG = on
