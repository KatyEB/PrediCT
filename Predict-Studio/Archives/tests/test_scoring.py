"""Scoring tests. Every expected number is derived by hand in the comment
above the assertion.

Spacing throughout is 0.5 x 0.5 x 3.0 mm, chosen so pixel area is exactly
0.25 mm2 and four pixels make exactly 1.0 mm2 — the minimum-lesion boundary.
Slice thickness is exactly 3.0 mm so the thickness factor is exactly 1.0 and
never obscures an arithmetic error.
"""

import numpy as np
import pytest

from scoring import (
    calcium_volume_mm3,
    categorise,
    density_factor,
    distance,
    find_components,
    score_from_polygons,
    score_volume,
)
from scoring import RiskCategory, ScoringConfig, Spacing

SP = Spacing(0.5, 0.5, 3.0)          # pixel area 0.25 mm2, voxel 0.75 mm3
SHAPE = (4, 16, 16)


def empty_volumes():
    mask = np.zeros(SHAPE, dtype=np.float64)
    hu = np.full(SHAPE, -100.0)      # air/soft tissue background
    return mask, hu


def put_block(mask, hu, z, r0, c0, rows, cols, value=1.0, peak=250.0):
    mask[z, r0:r0 + rows, c0:c0 + cols] = value
    hu[z, r0:r0 + rows, c0:c0 + cols] = peak
    return mask, hu


# ---------------------------------------------------------------------------
# Density factor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hu,expected", [
    (129.9, 0), (130.0, 1), (199.9, 1),
    (200.0, 2), (299.9, 2),
    (300.0, 3), (399.9, 3),
    (400.0, 4), (3000.0, 4),
])
def test_density_factor_bin_edges(hu, expected):
    assert density_factor(hu) == expected


# ---------------------------------------------------------------------------
# Risk categories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (0.0, RiskCategory.ZERO),
    (0.4, RiskCategory.MINIMAL),
    (99.9, RiskCategory.MINIMAL),
    (100.0, RiskCategory.MODERATE),
    (399.9, RiskCategory.MODERATE),
    (400.0, RiskCategory.SEVERE),
])
def test_risk_bands(score, expected):
    assert categorise(score) == expected


def test_zero_is_its_own_band_not_the_bottom_of_the_first():
    assert categorise(0.0) != categorise(0.01)


def test_band_distance_separates_near_miss_from_gross_error():
    assert distance(95.0, 105.0) == 1     # adjacent, straddling 100
    assert distance(10.0, 900.0) == 2     # minimal vs severe


# ---------------------------------------------------------------------------
# The minimum-lesion rule
# ---------------------------------------------------------------------------

def test_lesion_of_exactly_one_mm2_is_included():
    # 4 pixels x 0.25 mm2 = 1.00 mm2, exactly at the boundary. Peak 250 -> f=2.
    # score = 1.00 * 2 * 1.0 = 2.0
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, peak=250.0)

    score = score_volume(mask, hu, SP)
    assert score.agatston == pytest.approx(2.0)
    assert len(score.included_lesions) == 1
    assert len(score.excluded_lesions) == 0


def test_lesion_below_one_mm2_is_excluded_but_recorded():
    # 3 pixels x 0.25 = 0.75 mm2 < 1.0 -> excluded.
    mask, hu = empty_volumes()
    mask[1, 5, 5:8] = 1.0
    hu[1, 5, 5:8] = 250.0

    score = score_volume(mask, hu, SP)
    assert score.agatston == pytest.approx(0.0)
    assert len(score.excluded_lesions) == 1

    excluded = score.excluded_lesions[0]
    assert excluded.exclusion_reason == "below_min_lesion_area"
    assert excluded.area_mm2 == pytest.approx(0.75)
    # The rule's effect is visible, not silent: 0.75 * 2 = 1.5 was removed.
    assert score.excluded_score_if_included == pytest.approx(1.5)


def test_legacy_config_keeps_sub_threshold_lesions():
    # Reproduces the current scripts, which apply no minimum-lesion rule.
    mask, hu = empty_volumes()
    mask[1, 5, 5:8] = 1.0
    hu[1, 5, 5:8] = 250.0

    score = score_volume(mask, hu, SP, ScoringConfig.legacy())
    assert score.agatston == pytest.approx(1.5)
    assert len(score.excluded_lesions) == 0


def test_predicted_region_that_is_not_calcium_is_flagged():
    # 4 pixels at 90 HU: above the mask threshold, below 130 HU.
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, peak=90.0)

    score = score_volume(mask, hu, SP)
    assert score.agatston == pytest.approx(0.0)
    assert score.excluded_lesions[0].exclusion_reason == "peak_hu_below_threshold"


# ---------------------------------------------------------------------------
# Multiple lesions and slices
# ---------------------------------------------------------------------------

def test_scores_sum_across_lesions_and_slices():
    mask, hu = empty_volumes()
    # A: 2x2 = 1.00 mm2, peak 150 -> f=1 -> 1.0
    put_block(mask, hu, z=0, r0=2, c0=2, rows=2, cols=2, peak=150.0)
    # B: 2x4 = 2.00 mm2, peak 450 -> f=4 -> 8.0
    put_block(mask, hu, z=2, r0=8, c0=8, rows=2, cols=4, peak=450.0)

    score = score_volume(mask, hu, SP)
    assert score.agatston == pytest.approx(9.0)
    assert len(score.included_lesions) == 2


def test_two_lesions_on_one_slice_stay_separate_under_face_connectivity():
    # Separated by one background column, so 4-connectivity keeps them apart.
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=4, c0=4, rows=2, cols=2, peak=250.0)
    put_block(mask, hu, z=1, r0=4, c0=8, rows=2, cols=2, peak=250.0)

    score = score_volume(mask, hu, SP)
    assert len(score.lesions) == 2
    assert score.agatston == pytest.approx(4.0)      # 2.0 + 2.0


def test_diagonal_touching_lesions_merge_only_under_full_connectivity():
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=4, c0=4, rows=2, cols=2, peak=250.0)
    put_block(mask, hu, z=1, r0=6, c0=6, rows=2, cols=2, peak=250.0)

    face = score_volume(mask, hu, SP, ScoringConfig(connectivity=1))
    full = score_volume(mask, hu, SP, ScoringConfig(connectivity=2))
    assert len(face.lesions) == 2
    assert len(full.lesions) == 1


# ---------------------------------------------------------------------------
# 2D vs 3D — the definitions disagree, on purpose
# ---------------------------------------------------------------------------

def test_2d_and_3d_give_different_numbers_on_the_same_mask():
    # One object spanning two slices. Slice 1 peaks at 150 (f=1), slice 2 at
    # 450 (f=4).
    #   2D: two lesions -> 1.0*1 + 1.0*4 = 5.0
    #   3D: one component, global peak 450 -> (1.0 + 1.0) * 4 = 8.0
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, peak=150.0)
    put_block(mask, hu, z=2, r0=5, c0=5, rows=2, cols=2, peak=450.0)

    two_d = score_volume(mask, hu, SP, ScoringConfig(lesion_definition="2d"))
    three_d = score_volume(mask, hu, SP, ScoringConfig(lesion_definition="3d"))

    assert two_d.agatston == pytest.approx(5.0)
    assert three_d.agatston == pytest.approx(8.0)
    assert two_d.agatston != three_d.agatston

    # The stamps differ, so the two can never be merged into one table by
    # accident downstream.
    assert two_d.stamp["lesion_definition"] == "2d"
    assert three_d.stamp["lesion_definition"] == "3d"


def test_3d_component_records_its_slice_span():
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, peak=250.0)
    put_block(mask, hu, z=2, r0=5, c0=5, rows=2, cols=2, peak=250.0)

    score = score_volume(mask, hu, SP, ScoringConfig(lesion_definition="3d"))
    assert len(score.lesions) == 1
    assert score.lesions[0].slice_span == (1, 2)
    assert score.lesions[0].slice_index is None


def test_volume_is_identical_under_both_lesion_definitions():
    # Volume is a voxel count; connectivity cannot change it. This is why
    # volumetric analysis does not need the 3D switch.
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, peak=250.0)
    put_block(mask, hu, z=2, r0=5, c0=5, rows=2, cols=2, peak=450.0)

    a = score_volume(mask, hu, SP, ScoringConfig(lesion_definition="2d"))
    b = score_volume(mask, hu, SP, ScoringConfig(lesion_definition="3d"))
    assert a.calcium_volume_mm3 == pytest.approx(b.calcium_volume_mm3)
    assert a.calcium_volume_mm3 == pytest.approx(8 * 0.75)   # 8 voxels


# ---------------------------------------------------------------------------
# Soft / coverage masks — the Approach 3 path
# ---------------------------------------------------------------------------

def test_soft_area_uses_fractions_not_voxel_counts():
    # Four voxels at coverage 0.5 -> 2.0 voxels -> 0.5 mm2.
    # Below 1 mm2, so excluded; the recorded area proves fractions were summed.
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, value=0.5, peak=250.0)

    score = score_volume(mask, hu, SP, mask_kind="soft")
    assert score.excluded_lesions[0].area_mm2 == pytest.approx(0.5)


def test_soft_and_binary_differ_on_the_same_partial_mask():
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=4, cols=4, value=0.5, peak=250.0)

    soft = score_volume(mask, hu, SP, mask_kind="soft")
    # 16 voxels x 0.5 = 8.0 voxels x 0.25 mm2 = 2.0 mm2, f=2 -> 4.0
    assert soft.agatston == pytest.approx(4.0)

    # Binary at threshold 0.5 excludes voxels equal to 0.5 (strictly greater),
    # so the same array scores zero. Same data, different declared semantics.
    binary = score_volume(mask, hu, SP, mask_kind="binary")
    assert binary.agatston == pytest.approx(0.0)


def test_soft_threshold_controls_grouping_not_area():
    # Coverage 0.05 is below the default soft_threshold of 0.1, so those voxels
    # do not join any component.
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, value=1.0, peak=250.0)
    mask[1, 5:7, 7:9] = 0.05
    hu[1, 5:7, 7:9] = 250.0

    score = score_volume(mask, hu, SP)
    assert score.agatston == pytest.approx(2.0)   # only the full block counts


# ---------------------------------------------------------------------------
# Slice-thickness correction
# ---------------------------------------------------------------------------

def test_thickness_factor_is_exactly_one_at_three_mm():
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, peak=250.0)
    score = score_volume(mask, hu, SP)
    assert score.stamp["slice_thickness_factor"] == pytest.approx(1.0)
    assert score.agatston == pytest.approx(2.0)


def test_thinner_slices_scale_the_score_down():
    # At 1.5 mm the factor is 0.5, so the same lesion scores half.
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, peak=250.0)
    thin = Spacing(0.5, 0.5, 1.5)
    score = score_volume(mask, hu, thin, ScoringConfig())
    assert score.stamp["slice_thickness_factor"] == pytest.approx(0.5)
    assert score.agatston == pytest.approx(1.0)


def test_legacy_config_disables_the_thickness_correction():
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, peak=250.0)
    thin = Spacing(0.5, 0.5, 1.5)
    score = score_volume(mask, hu, thin, ScoringConfig.legacy())
    assert score.agatston == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Ground-truth path from polygons
# ---------------------------------------------------------------------------

def test_polygon_area_is_exact_not_rasterised():
    # A 2x2 square in continuous coordinates has area 4 px = 1.00 mm2.
    hu = np.full(SHAPE, -100.0)
    hu[1, 4:8, 4:8] = 250.0
    square = np.array([[5.0, 5.0], [7.0, 5.0], [7.0, 7.0], [5.0, 7.0]])

    score = score_from_polygons({1: [square]}, hu, SP)
    assert score.lesions[0].area_mm2 == pytest.approx(4 * 0.25)
    assert score.agatston == pytest.approx(1.0 * 2)


def test_sub_pixel_polygon_survives_instead_of_vanishing():
    # THE TRUNCATION BUG. Vertices at x in [10.1, 10.5] all become 10 under
    # .astype(np.int32); cv2.fillPoly then draws nothing, the HU sample is
    # empty and this lesion disappears from the ground-truth total.
    hu = np.full(SHAPE, -100.0)
    hu[1, 10, 10] = 450.0
    tiny = np.array([[10.1, 10.1], [10.5, 10.1], [10.5, 10.5], [10.1, 10.5]])

    score = score_from_polygons({1: [tiny]}, hu, SP, ScoringConfig.legacy())
    assert len(score.lesions) == 1
    assert score.lesions[0].peak_hu == pytest.approx(450.0)
    # area 0.16 px x 0.25 mm2 x factor 4 = 0.16
    assert score.agatston == pytest.approx(0.16)


def test_polygon_outside_the_volume_is_skipped_not_crashed():
    hu = np.full(SHAPE, -100.0)
    outside = np.array([[100.0, 100.0], [102.0, 100.0], [102.0, 102.0]])
    score = score_from_polygons({1: [outside]}, hu, SP)
    assert score.agatston == pytest.approx(0.0)
    assert len(score.lesions) == 0


def test_polygon_on_a_slice_index_beyond_the_volume_is_skipped():
    hu = np.full(SHAPE, -100.0)
    square = np.array([[5.0, 5.0], [7.0, 5.0], [7.0, 7.0], [5.0, 7.0]])
    score = score_from_polygons({99: [square]}, hu, SP)
    assert score.agatston == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def test_binary_volume_is_voxel_count_times_voxel_volume():
    mask, _ = empty_volumes()
    mask[1, 5:7, 5:7] = 1.0
    assert calcium_volume_mm3(mask, SP) == pytest.approx(4 * 0.75)


def test_soft_volume_sums_fractions():
    mask, _ = empty_volumes()
    mask[1, 5:7, 5:7] = 0.5
    assert calcium_volume_mm3(mask, SP, mask_kind="soft") == pytest.approx(2 * 0.75)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def test_normalised_volume_scores_zero_and_this_is_why_hu_must_be_raw():
    # Feeding a [0,1]-windowed volume makes every peak < 130, so every density
    # factor is 0 and the total is silently zero. Documented as a trap.
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, peak=250.0)
    normalised = np.clip(hu / 1200.0, 0, 1)

    score = score_volume(mask, normalised, SP)
    assert score.agatston == pytest.approx(0.0)
    assert score.excluded_lesions[0].exclusion_reason == "peak_hu_below_threshold"


def test_shape_mismatch_raises_rather_than_broadcasting():
    mask = np.zeros((4, 16, 16))
    hu = np.zeros((4, 16, 8))
    with pytest.raises(ValueError, match="does not match"):
        score_volume(mask, hu, SP)


def test_empty_mask_scores_zero_with_no_lesions():
    mask, hu = empty_volumes()
    score = score_volume(mask, hu, SP)
    assert score.agatston == 0.0
    assert score.risk_category == RiskCategory.ZERO
    assert score.lesions == ()


def test_every_score_carries_a_reproducible_stamp():
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, peak=250.0)
    stamp = score_volume(mask, hu, SP).stamp

    for key in ("lesion_definition", "connectivity", "min_lesion_area_mm2",
                "hu_threshold", "slice_thickness_factor", "spacing_mm"):
        assert key in stamp


def test_ledger_rows_are_serialisable():
    mask, hu = empty_volumes()
    put_block(mask, hu, z=1, r0=5, c0=5, rows=2, cols=2, peak=250.0)
    rows = score_volume(mask, hu, SP).ledger()
    assert rows[0]["included"] is True
    assert rows[0]["density_factor"] == 2


def test_lesion_ids_are_stable_across_repeated_runs():
    mask, hu = empty_volumes()
    put_block(mask, hu, z=0, r0=2, c0=2, rows=2, cols=2, peak=250.0)
    put_block(mask, hu, z=2, r0=8, c0=8, rows=2, cols=2, peak=450.0)

    first = [les.lesion_id for les in score_volume(mask, hu, SP).lesions]
    second = [les.lesion_id for les in score_volume(mask, hu, SP).lesions]
    assert first == second == [1, 2]


def test_find_components_rejects_a_2d_input():
    with pytest.raises(ValueError, match="3-D"):
        find_components(np.zeros((16, 16), dtype=bool))
