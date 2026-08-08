"""Geometry tests. Every expected value here is computed by hand, not by
running the code and recording what it printed.

Pixel convention under test: centres at integers, so pixel (i, j) covers
x in [j-0.5, j+0.5] and y in [i-0.5, i+0.5].
"""

import numpy as np
import pytest

from geometry import (
    clip_polygon_to_rect,
    clipped_area,
    coverage_from_polygons,
    is_degenerate,
    pixel_bounds,
    polygon_area_xy,
    polygon_coverage,
    signed_area,
    touched_mask,
)

TOL = 1e-10


# ---------------------------------------------------------------------------
# Shoelace
# ---------------------------------------------------------------------------

def test_unit_square_area_is_one():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert polygon_area_xy(sq) == pytest.approx(1.0, abs=TOL)


def test_winding_order_flips_sign_not_magnitude():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    ccw = signed_area(sq[:, 0], sq[:, 1])
    cw = signed_area(sq[::-1, 0], sq[::-1, 1])
    assert ccw == pytest.approx(-cw, abs=TOL)
    assert abs(ccw) == pytest.approx(1.0, abs=TOL)


def test_right_triangle_area_is_half_base_times_height():
    tri = np.array([[0, 0], [4, 0], [0, 3]], dtype=float)
    assert polygon_area_xy(tri) == pytest.approx(6.0, abs=TOL)


def test_regular_hexagon_matches_closed_form():
    # Area of a regular hexagon with circumradius r is 3*sqrt(3)/2 * r^2.
    r = 2.0
    ang = np.linspace(0, 2 * np.pi, 7)[:-1]
    hexagon = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    expected = 1.5 * np.sqrt(3) * r**2
    assert polygon_area_xy(hexagon) == pytest.approx(expected, rel=1e-12)


def test_concave_L_shape():
    # An L: 3x3 square with a 2x2 bite out of the top-right. 9 - 4 = 5.
    ell = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], dtype=float)
    assert polygon_area_xy(ell) == pytest.approx(5.0, abs=TOL)


def test_sub_pixel_polygon_keeps_its_area():
    # This is the case integer truncation destroys: a lesion smaller than one
    # pixel, entirely inside pixel (10, 10).
    tiny = np.array([[10.1, 10.1], [10.5, 10.1], [10.5, 10.5], [10.1, 10.5]])
    assert polygon_area_xy(tiny) == pytest.approx(0.16, abs=TOL)
    assert not is_degenerate(tiny)


def test_degenerate_shapes_are_detected():
    assert is_degenerate(np.array([[0, 0], [1, 1]], dtype=float))          # 2 points
    assert is_degenerate(np.array([[0, 0], [1, 1], [2, 2]], dtype=float))  # collinear
    assert is_degenerate(np.array([[0, 0], [0, 0], [0, 0]], dtype=float))  # repeated


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

def test_polygon_fully_inside_rect_is_unchanged_in_area():
    tri = np.array([[1, 1], [3, 1], [1, 4]], dtype=float)
    assert clipped_area(tri, 0, 0, 10, 10) == pytest.approx(3.0, abs=TOL)


def test_polygon_fully_outside_rect_clips_to_nothing():
    tri = np.array([[20, 20], [22, 20], [20, 22]], dtype=float)
    assert clipped_area(tri, 0, 0, 10, 10) == pytest.approx(0.0, abs=TOL)
    assert clip_polygon_to_rect(tri, 0, 0, 10, 10).shape[0] == 0


def test_half_overlap_gives_exactly_half_the_area():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)   # area 4
    assert clipped_area(sq, 1, 0, 3, 2) == pytest.approx(2.0, abs=TOL)


def test_quarter_overlap_at_a_corner():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    assert clipped_area(sq, 1, 1, 3, 3) == pytest.approx(1.0, abs=TOL)


def test_triangle_clipped_by_a_diagonal_cut():
    # Triangle (0,0),(2,0),(0,2) has area 2. Clipping to x <= 1 removes the
    # sub-triangle (1,0),(2,0),(1,1), whose area is 0.5. Expect 1.5.
    tri = np.array([[0, 0], [2, 0], [0, 2]], dtype=float)
    assert clipped_area(tri, -10, -10, 1, 10) == pytest.approx(1.5, abs=TOL)


def test_pixel_bounds_follow_centres_at_integers():
    assert pixel_bounds(0, 0) == (-0.5, -0.5, 0.5, 0.5)
    assert pixel_bounds(3, 7) == (6.5, 2.5, 7.5, 3.5)


# ---------------------------------------------------------------------------
# Coverage — the identity that makes A3 defensible
# ---------------------------------------------------------------------------

def _assert_coverage_sums_to_area(poly, shape):
    cov = polygon_coverage(poly, shape)
    assert cov.sum() == pytest.approx(polygon_area_xy(poly), abs=1e-9)
    assert cov.min() >= 0.0
    assert cov.max() <= 1.0 + 1e-12
    return cov


def test_coverage_sums_to_exact_area_for_a_square():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    cov = _assert_coverage_sums_to_area(sq, (8, 8))
    # Under centres-at-integers, the corner pixel (0,0) spans [-0.5,0.5]^2 and
    # overlaps the square on [0,0.5]^2 -> 0.25. Pixel (1,1) is fully inside.
    assert cov[0, 0] == pytest.approx(0.25, abs=TOL)
    assert cov[1, 1] == pytest.approx(1.00, abs=TOL)
    assert cov[2, 2] == pytest.approx(0.25, abs=TOL)


def test_coverage_sums_to_exact_area_for_a_triangle():
    tri = np.array([[1.0, 1.0], [6.5, 1.0], [1.0, 5.25]], dtype=float)
    _assert_coverage_sums_to_area(tri, (12, 12))


def test_coverage_sums_to_exact_area_for_a_rotated_polygon():
    ang = np.linspace(0, 2 * np.pi, 8)[:-1] + 0.37
    poly = np.stack([5 + 3.1 * np.cos(ang), 5 + 2.4 * np.sin(ang)], axis=1)
    _assert_coverage_sums_to_area(poly, (14, 14))


def test_coverage_sums_to_exact_area_for_a_concave_polygon():
    ell = np.array([[1, 1], [6, 1], [6, 2.5], [2.5, 2.5], [2.5, 6], [1, 6]], dtype=float)
    _assert_coverage_sums_to_area(ell, (12, 12))


def test_sub_pixel_polygon_produces_one_partially_covered_pixel():
    # The truncation bug in the current scorer makes this lesion disappear.
    tiny = np.array([[10.1, 10.1], [10.5, 10.1], [10.5, 10.5], [10.1, 10.5]])
    cov = polygon_coverage(tiny, (20, 20))
    assert cov.sum() == pytest.approx(0.16, abs=1e-12)
    assert np.count_nonzero(cov) == 1
    assert touched_mask(tiny, (20, 20)).sum() == 1


def test_polygon_straddling_the_volume_edge_is_clipped_not_wrapped():
    poly = np.array([[-1.0, 1.0], [1.0, 1.0], [1.0, 3.0], [-1.0, 3.0]])
    cov = polygon_coverage(poly, (8, 8))
    # Half the polygon lies at negative x and is outside the array.
    assert cov.sum() < polygon_area_xy(poly)
    assert cov.sum() == pytest.approx(3.0, abs=1e-9)


def test_max_mode_does_not_double_count_overlapping_polygons():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    both = coverage_from_polygons([sq, sq], (8, 8), accumulate=False)
    assert both.max() <= 1.0 + 1e-12
    assert both.sum() == pytest.approx(polygon_area_xy(sq), abs=1e-9)


def test_accumulate_mode_reveals_overlapping_annotations():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    both = coverage_from_polygons([sq, sq], (8, 8), accumulate=True)
    # Deliberately exceeds 1.0 so overlapping ROIs are visible, not hidden.
    assert both.max() == pytest.approx(2.0, abs=TOL)


def test_degenerate_polygon_contributes_nothing():
    line = np.array([[1, 1], [4, 4], [2, 2]], dtype=float)
    assert polygon_coverage(line, (8, 8)).sum() == pytest.approx(0.0, abs=TOL)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_mismatched_coordinate_arrays_raise():
    with pytest.raises(ValueError, match="same shape"):
        signed_area(np.zeros(4), np.zeros(3))


def test_wrong_point_array_shape_raises():
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        polygon_area_xy(np.zeros((4, 3)))
