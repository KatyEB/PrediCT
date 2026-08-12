"""
scoring.py - Agatston scoring. Needs no ground truth; this is the user path.

    score = SUM over lesions of  area_mm2 * density_factor(peak_HU)

Two rules the current scripts omit are restored here, both configurable and
both no-ops under ScoringConfig.legacy():

  * lesions below min_lesion_area_mm2 are EXCLUDED (Agatston et al., 1990)
  * a slice-thickness correction, since the definition assumes 3 mm slices

Excluded lesions are recorded with a reason, never dropped. An exclusion rule
that operates invisibly is indistinguishable from a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from scipy import ndimage

from geometry import polygon_area_xy, touched_mask



# ========================================================================
# types
# ========================================================================

MaskKind = Literal["binary", "soft"]
LesionDefinition = Literal["2d", "3d"]

# Agatston is defined on 3 mm contiguous slices (Agatston et al., 1990).
# Scanners routinely acquire other thicknesses, so a correction factor is
# applied. At 3.0 mm the factor is exactly 1.0 and this is a no-op.
AGATSTON_REFERENCE_THICKNESS_MM = 3.0


@dataclass(frozen=True)
class Spacing:
    """Physical voxel size in millimetres, in (x, y, z) order.

    x/y are in-plane; z is slice thickness. This is SimpleITK's GetSpacing()
    order, deliberately, so no transposition is needed at the I/O boundary.
    """

    x: float
    y: float
    z: float

    @property
    def pixel_area_mm2(self) -> float:
        return self.x * self.y

    @property
    def voxel_volume_mm3(self) -> float:
        return self.x * self.y * self.z

    @property
    def slice_thickness_mm(self) -> float:
        return self.z

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_sitk(cls, spacing) -> "Spacing":
        x, y, z = spacing
        return cls(float(x), float(y), float(z))


class RiskCategory:
    """Agatston risk strata, as plain strings with a defined order."""

    ZERO = "0"
    MINIMAL = "1-99"
    MODERATE = "100-399"
    SEVERE = "400+"

    ORDER = (ZERO, MINIMAL, MODERATE, SEVERE)

    @staticmethod
    def rank(category: str) -> int:
        return RiskCategory.ORDER.index(category)


@dataclass(frozen=True)
class ScoringConfig:
    """Every knob that changes an Agatston number.

    Defaults are the CLINICALLY CORRECT ones, not the ones that reproduce the
    project's current published figures. Use ScoringConfig.legacy() for that.
    The two produce different numbers and are stamped differently on output so
    they cannot be mixed in one table by accident.
    """

    # --- lesion identification -------------------------------------------
    lesion_definition: LesionDefinition = "2d"

    connectivity: int = 1
    """1 = face adjacency (4-connected in 2D, 6 in 3D), matching
    scipy.ndimage.label's default structure. 2 = full adjacency (8 / 26)."""

    binary_threshold: float = 0.5
    """Probability above which a voxel is calcium, for binary-output models."""

    soft_threshold: float = 0.1
    """Coverage fraction above which a voxel joins a component, for
    coverage-output models. Area still uses the full fractional sum."""

    # --- clinical rules ---------------------------------------------------
    min_lesion_area_mm2: float = 1.0
    """Agatston's minimum lesion area. Lesions below it are excluded from the
    total but RECORDED, never silently dropped."""

    hu_threshold: float = 130.0
    """The Agatston calcium threshold; also the density-factor floor."""

    apply_hu_threshold: bool = False
    """Whether to intersect the predicted mask with HU >= hu_threshold before
    scoring. False trusts the model's mask (the project's current behaviour);
    True enforces the strict published definition."""

    # --- geometry ---------------------------------------------------------
    slice_thickness_factor: float | None = None
    """None means derive it as slice_thickness / 3.0. Set to 1.0 to disable."""

    def resolved_thickness_factor(self, spacing: Spacing) -> float:
        if self.slice_thickness_factor is not None:
            return self.slice_thickness_factor
        return spacing.slice_thickness_mm / AGATSTON_REFERENCE_THICKNESS_MM

    @classmethod
    def legacy(cls) -> "ScoringConfig":
        """Reproduces src/testing/agatston_scoring_a{1,3}.py exactly.

        No minimum-lesion rule, no thickness correction. Exists so the
        regression test can prove the new module matches the published numbers
        BEFORE those numbers are intentionally changed by the fixes.
        """
        return cls(min_lesion_area_mm2=0.0, slice_thickness_factor=1.0)

    def stamp(self, spacing: Spacing) -> dict:
        """Human-readable record of every choice that shaped a number.

        Attached to every Score and carried into CSV headers, cohort tables and
        the PDF. Two Scores with different stamps must not be compared.
        """
        return {
            "lesion_definition": self.lesion_definition,
            "connectivity": self.connectivity,
            "binary_threshold": self.binary_threshold,
            "soft_threshold": self.soft_threshold,
            "min_lesion_area_mm2": self.min_lesion_area_mm2,
            "hu_threshold": self.hu_threshold,
            "apply_hu_threshold": self.apply_hu_threshold,
            "slice_thickness_factor": round(self.resolved_thickness_factor(spacing), 6),
            "spacing_mm": [spacing.x, spacing.y, spacing.z],
        }

    def with_(self, **kw) -> "ScoringConfig":
        return replace(self, **kw)


@dataclass(frozen=True)
class Lesion:
    """One scored lesion. Excluded lesions appear here too, with a reason."""

    lesion_id: int
    n_voxels: int
    area_mm2: float
    peak_hu: float
    density_factor: int
    score: float
    included: bool
    slice_index: int | None = None
    """Slice index for 2D lesions; None for 3D components."""

    slice_span: tuple[int, int] | None = None
    """(first, last) slice for 3D components; None for 2D."""

    exclusion_reason: str | None = None

    def as_row(self) -> dict:
        return {
            "lesion_id": self.lesion_id,
            "slice_index": self.slice_index,
            "slice_span": self.slice_span,
            "n_voxels": self.n_voxels,
            "area_mm2": round(self.area_mm2, 6),
            "peak_hu": round(float(self.peak_hu), 2),
            "density_factor": self.density_factor,
            "score": round(self.score, 6),
            "included": self.included,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True)
class Score:
    """The result of scoring one volume. Carries its own provenance."""

    agatston: float
    calcium_volume_mm3: float
    risk_category: str

    lesions: tuple[Lesion, ...] = field(default_factory=tuple)
    mask_kind: MaskKind = "binary"
    stamp: dict = field(default_factory=dict)

    @property
    def included_lesions(self) -> tuple[Lesion, ...]:
        return tuple(les for les in self.lesions if les.included)

    @property
    def excluded_lesions(self) -> tuple[Lesion, ...]:
        return tuple(les for les in self.lesions if not les.included)

    @property
    def excluded_area_mm2(self) -> float:
        return sum(les.area_mm2 for les in self.excluded_lesions)

    @property
    def excluded_score_if_included(self) -> float:
        """How much Agatston the minimum-lesion rule removed.

        Surfaced in the UI so an exclusion rule is never invisible.
        """
        return sum(les.score for les in self.excluded_lesions)

    def ledger(self) -> list[dict]:
        return [les.as_row() for les in self.lesions]

    def summary(self) -> dict:
        return {
            "agatston": round(self.agatston, 4),
            "calcium_volume_mm3": round(self.calcium_volume_mm3, 4),
            "risk_category": self.risk_category,
            "n_lesions_included": len(self.included_lesions),
            "n_lesions_excluded": len(self.excluded_lesions),
            "excluded_area_mm2": round(self.excluded_area_mm2, 4),
            "excluded_score_if_included": round(self.excluded_score_if_included, 4),
            "mask_kind": self.mask_kind,
            **{f"cfg.{k}": v for k, v in self.stamp.items()},
        }


# ========================================================================
# lesions
# ========================================================================

@dataclass(frozen=True)
class Component:
    """One connected region, as explicit voxel indices.

    Indices rather than a boolean mask because calcium is sparse: a component
    is tens of voxels inside a volume of tens of millions, and indices let
    every downstream computation touch only those voxels.
    """

    label: int
    zs: np.ndarray
    ys: np.ndarray
    xs: np.ndarray
    slice_index: int | None
    slice_span: tuple[int, int] | None

    @property
    def n_voxels(self) -> int:
        return int(self.zs.size)

    def values_from(self, volume: np.ndarray) -> np.ndarray:
        """Sample a (Z, Y, X) volume at this component's voxels."""
        return volume[self.zs, self.ys, self.xs]

    def voxels_per_slice(self) -> dict[int, int]:
        """Voxel count keyed by slice. Used for the 3D minimum-area test."""
        uniq, counts = np.unique(self.zs, return_counts=True)
        return {int(z): int(c) for z, c in zip(uniq, counts)}


def _structure(rank: int, connectivity: int) -> np.ndarray:
    if connectivity not in (1, 2):
        raise ValueError(
            f"connectivity must be 1 (face) or 2 (full), got {connectivity}"
        )
    return ndimage.generate_binary_structure(rank, connectivity)


def binarise(mask: np.ndarray, mask_kind: str, binary_threshold: float,
             soft_threshold: float) -> np.ndarray:
    """Reduce a prediction to the boolean membership used for connectivity.

    For soft (coverage) predictions this only decides which voxels JOIN a
    component. Area is still computed from the full fractional values, which is
    the whole point of Approach 3 — thresholding for grouping does not throw
    away the sub-voxel information.
    """
    if mask_kind == "binary":
        return mask > binary_threshold
    if mask_kind == "soft":
        return mask > soft_threshold
    raise ValueError(f"mask_kind must be 'binary' or 'soft', got {mask_kind!r}")


def find_components(
    membership: np.ndarray,
    definition: LesionDefinition = "2d",
    connectivity: int = 1,
) -> list[Component]:
    """Connected components of a boolean (Z, Y, X) volume.

    Returns them in a stable order — slice then label — so that lesion_id is
    reproducible across runs and diffable between models.
    """
    if membership.ndim != 3:
        raise ValueError(
            f"expected a 3-D (Z, Y, X) volume, got {membership.ndim}-D"
        )

    membership = np.asarray(membership, dtype=bool)
    components: list[Component] = []

    if definition == "2d":
        struct = _structure(2, connectivity)
        next_id = 1
        for z in range(membership.shape[0]):
            plane = membership[z]
            if not plane.any():
                continue
            labelled, n = ndimage.label(plane, structure=struct)
            for lab in range(1, n + 1):
                ys, xs = np.nonzero(labelled == lab)
                components.append(
                    Component(
                        label=next_id,
                        zs=np.full(ys.size, z, dtype=np.int64),
                        ys=ys,
                        xs=xs,
                        slice_index=z,
                        slice_span=None,
                    )
                )
                next_id += 1
        return components

    if definition == "3d":
        struct = _structure(3, connectivity)
        labelled, n = ndimage.label(membership, structure=struct)
        for lab in range(1, n + 1):
            zs, ys, xs = np.nonzero(labelled == lab)
            components.append(
                Component(
                    label=lab,
                    zs=zs,
                    ys=ys,
                    xs=xs,
                    slice_index=None,
                    slice_span=(int(zs.min()), int(zs.max())),
                )
            )
        return components

    raise ValueError(f"definition must be '2d' or '3d', got {definition!r}")


# ========================================================================
# volume
# ========================================================================

def calcium_volume_mm3(mask: np.ndarray, spacing: Spacing,
                       config: ScoringConfig | None = None,
                       mask_kind: MaskKind = "binary") -> float:
    """Total calcium volume.

    binary — count of voxels above the threshold, times voxel volume.
    soft   — SUM of coverage fractions times voxel volume. A voxel 30% covered
             contributes 30% of a voxel's volume, so partial-volume calcium at
             lesion edges is measured rather than rounded to 0 or 1.
    """
    config = config or ScoringConfig()
    mask = np.asarray(mask)

    if mask_kind == "binary":
        n = float(np.count_nonzero(mask > config.binary_threshold))
        return n * spacing.voxel_volume_mm3

    if mask_kind == "soft":
        # Coverage below soft_threshold is treated as noise and excluded, to
        # stay consistent with which voxels formed components for scoring.
        vals = np.asarray(mask, dtype=np.float64)
        vals = np.where(vals > config.soft_threshold, vals, 0.0)
        return float(vals.sum()) * spacing.voxel_volume_mm3

    raise ValueError(f"mask_kind must be 'binary' or 'soft', got {mask_kind!r}")


# ========================================================================
# risk
# ========================================================================

# (upper bound exclusive, label), applied only to strictly positive scores.
# ZERO is handled separately below and is NOT a band: a score of 0.4 means
# detectable calcium and belongs in the minimal band, not with true zeros.
_BANDS: tuple[tuple[float, str], ...] = (
    (100.0, RiskCategory.MINIMAL),   # 0 < score < 100
    (400.0, RiskCategory.MODERATE),  # 100 - 399
)


def categorise(agatston: float) -> str:
    """Map an Agatston score to its risk band.

    A score of exactly 0 is its own category, not the bottom of the first band:
    the clinical meaning of zero calcium is qualitatively different from a
    small amount, which is why 'agatston == 0' is tested before the ranges.
    """
    if agatston <= 0.0:
        return RiskCategory.ZERO
    for upper, label in _BANDS:
        if agatston < upper:
            return label
    return RiskCategory.SEVERE


def agrees(a: float, b: float) -> bool:
    """Whether two scores fall in the same risk band.

    This is the project's headline metric — A3 improves risk categorisation
    even where mean volumetric error does not favour it — so it is defined
    once, here, rather than recomputed per analysis script.
    """
    return categorise(a) == categorise(b)


def distance(a: float, b: float) -> int:
    """How many bands apart two scores are. 0 means agreement.

    Distinguishes a near-miss at a boundary from a two-band error, which a
    plain accuracy percentage hides.
    """
    return abs(RiskCategory.rank(categorise(a)) - RiskCategory.rank(categorise(b)))


# ========================================================================
# agatston
# ========================================================================

def density_factor(peak_hu: float, hu_threshold: float = 130.0) -> int:
    """Agatston's density weighting.

        < threshold -> 0    (not calcium; contributes nothing)
        130 - 199   -> 1
        200 - 299   -> 2
        300 - 399   -> 3
        >= 400      -> 4

    The upper bins are fixed by the definition and are not rescaled when
    hu_threshold moves; only the floor moves.
    """
    if peak_hu < hu_threshold:
        return 0
    if peak_hu < 200.0:
        return 1
    if peak_hu < 300.0:
        return 2
    if peak_hu < 400.0:
        return 3
    return 4


def _component_area_mm2(component: Component, mask: np.ndarray,
                        mask_kind: MaskKind, pixel_area_mm2: float) -> float:
    """Area of one component.

    binary — voxel count times pixel area.
    soft   — SUM OF COVERAGE FRACTIONS times pixel area. A voxel 30% covered
             contributes 0.3 of a pixel, which is the sub-voxel precision that
             Approach 3 exists to preserve.
    """
    if mask_kind == "binary":
        return component.n_voxels * pixel_area_mm2
    return float(component.values_from(mask).sum()) * pixel_area_mm2


def _max_slice_area_mm2(component: Component, mask: np.ndarray,
                        mask_kind: MaskKind, pixel_area_mm2: float) -> float:
    """Largest single-slice area of a component.

    The minimum-lesion rule is a per-slice rule in the original definition, so
    for a 3D component the test is applied to its largest slice: an object that
    clears 1 mm2 on any slice is a lesion.
    """
    if component.slice_index is not None:
        return _component_area_mm2(component, mask, mask_kind, pixel_area_mm2)

    best = 0.0
    for z in np.unique(component.zs):
        sel = component.zs == z
        if mask_kind == "binary":
            area = int(sel.sum()) * pixel_area_mm2
        else:
            vals = mask[component.zs[sel], component.ys[sel], component.xs[sel]]
            area = float(vals.sum()) * pixel_area_mm2
        best = max(best, area)
    return best


def score_volume(
    mask: np.ndarray,
    hu: np.ndarray,
    spacing: Spacing,
    config: ScoringConfig | None = None,
    mask_kind: MaskKind = "binary",
) -> Score:
    """Score one volume. No ground truth required — this is the user path.

    Parameters
    ----------
    mask : (Z, Y, X) prediction. 0/1 or probabilities for mask_kind='binary';
        coverage fractions in [0, 1] for mask_kind='soft'.
    hu : (Z, Y, X) volume in RAW HOUNSFIELD UNITS.

        Not windowed, not normalised to [0, 1]. The density factor compares
        against literal HU values, so a normalised volume yields peak values
        below 1.0, a density factor of 0 everywhere, and a total score of
        exactly zero. That failure is silent, which is why it is stated here
        and asserted below.

    spacing : physical voxel size; supplies pixel area and slice thickness.
    """
    config = config or ScoringConfig()

    mask = np.asarray(mask)
    hu = np.asarray(hu)

    if mask.shape != hu.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match HU volume shape {hu.shape}"
        )
    if mask.ndim != 3:
        raise ValueError(f"expected 3-D volumes, got {mask.ndim}-D")

    membership = binarise(
        mask, mask_kind, config.binary_threshold, config.soft_threshold
    )

    # Optional strict-definition step: require the voxel itself to be calcium.
    if config.apply_hu_threshold:
        membership &= hu >= config.hu_threshold

    components = find_components(
        membership, config.lesion_definition, config.connectivity
    )

    pixel_area = spacing.pixel_area_mm2
    thickness_factor = config.resolved_thickness_factor(spacing)

    lesions: list[Lesion] = []
    total = 0.0

    for component in components:
        area = _component_area_mm2(component, mask, mask_kind, pixel_area)
        peak = float(component.values_from(hu).max())
        factor = density_factor(peak, config.hu_threshold)
        raw_score = area * factor * thickness_factor

        test_area = _max_slice_area_mm2(component, mask, mask_kind, pixel_area)

        included = True
        reason: str | None = None
        if test_area < config.min_lesion_area_mm2:
            included, reason = False, "below_min_lesion_area"
        elif factor == 0:
            # Contributes nothing either way, but flagging it makes predicted
            # regions that are not radiographically calcium visible as such.
            included, reason = False, "peak_hu_below_threshold"

        if included:
            total += raw_score

        lesions.append(
            Lesion(
                lesion_id=component.label,
                n_voxels=component.n_voxels,
                area_mm2=area,
                peak_hu=peak,
                density_factor=factor,
                score=raw_score,
                included=included,
                slice_index=component.slice_index,
                slice_span=component.slice_span,
                exclusion_reason=reason,
            )
        )

    return Score(
        agatston=total,
        calcium_volume_mm3=calcium_volume_mm3(mask, spacing, config, mask_kind),
        risk_category=categorise(total),
        lesions=tuple(lesions),
        mask_kind=mask_kind,
        stamp=config.stamp(spacing),
    )


def score_from_polygons(
    polygons_by_slice: dict[int, list[np.ndarray]],
    hu: np.ndarray,
    spacing: Spacing,
    config: ScoringConfig | None = None,
) -> Score:
    """Score annotator polygons directly — the ground-truth path.

    Area comes from the exact polygon, never from a rasterisation. Peak HU is
    sampled over every pixel the polygon touches at all, computed by analytic
    clipping.

    This replaces `pts.astype(np.int32)` + cv2.fillPoly. Under truncation, a
    polygon narrower than one pixel collapses to a degenerate shape, fillPoly
    draws nothing, the HU sample comes back empty and the lesion is dropped
    from the ground-truth total entirely. Here every polygon with non-zero area
    touches at least one pixel, so it cannot vanish.
    """

    config = config or ScoringConfig()
    hu = np.asarray(hu)

    pixel_area = spacing.pixel_area_mm2
    thickness_factor = config.resolved_thickness_factor(spacing)
    plane_shape = (hu.shape[1], hu.shape[2])

    lesions: list[Lesion] = []
    total = 0.0
    next_id = 1
    covered_voxels = 0

    for z in sorted(polygons_by_slice):
        if not 0 <= z < hu.shape[0]:
            continue

        for poly in polygons_by_slice[z]:
            area_px = polygon_area_xy(poly)
            if area_px <= 0.0:
                continue

            area_mm2 = area_px * pixel_area
            touched = touched_mask(poly, plane_shape)
            n_vox = int(touched.sum())
            if n_vox == 0:
                continue

            peak = float(hu[z][touched].max())
            factor = density_factor(peak, config.hu_threshold)
            raw_score = area_mm2 * factor * thickness_factor

            included = True
            reason: str | None = None
            if area_mm2 < config.min_lesion_area_mm2:
                included, reason = False, "below_min_lesion_area"
            elif factor == 0:
                included, reason = False, "peak_hu_below_threshold"

            if included:
                total += raw_score
            covered_voxels += n_vox

            lesions.append(
                Lesion(
                    lesion_id=next_id,
                    n_voxels=n_vox,
                    area_mm2=area_mm2,
                    peak_hu=peak,
                    density_factor=factor,
                    score=raw_score,
                    included=included,
                    slice_index=z,
                    exclusion_reason=reason,
                )
            )
            next_id += 1

    return Score(
        agatston=total,
        calcium_volume_mm3=covered_voxels * spacing.voxel_volume_mm3,
        risk_category=categorise(total),
        lesions=tuple(lesions),
        mask_kind="binary",
        stamp={**config.stamp(spacing), "source": "polygons"},
    )
