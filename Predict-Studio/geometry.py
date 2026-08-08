"""
geometry.py - exact polygon maths. No I/O, no torch, no model.

Ground-truth area comes from here, never from a rasterised mask. A polygon of
area 0.4 px still has area 0.4 px; rasterising it gives 0 or 1.

PIXEL CONVENTION
    Pixel centres sit at integer coordinates (ITK/DICOM).
    Pixel (row i, col j) covers x in [j-0.5, j+0.5], y in [i-0.5, i+0.5].
    Asserted by tests. Changing it changes every area in the system.
"""

from __future__ import annotations

import numpy as np



# ========================================================================
# shoelace
# ========================================================================

def signed_area(xs: np.ndarray, ys: np.ndarray) -> float:
    """Signed area of a simple polygon. Positive counter-clockwise.

    The sign is retained because it tells you the vertex winding order, which
    is what distinguishes an outer boundary from a hole. Callers that only want
    magnitude should use polygon_area.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if xs.shape != ys.shape:
        raise ValueError(f"xs and ys must have the same shape, got {xs.shape} and {ys.shape}")
    if xs.ndim != 1:
        raise ValueError(f"expected 1-D coordinate arrays, got {xs.ndim}-D")
    if xs.size < 3:
        return 0.0

    # Roll by -1 to pair each vertex with its successor, closing the ring.
    x_next = np.roll(xs, -1)
    y_next = np.roll(ys, -1)
    return 0.5 * float(np.sum(xs * y_next - x_next * ys))


def polygon_area(xs: np.ndarray, ys: np.ndarray) -> float:
    """Unsigned area of a simple polygon, in squared coordinate units."""
    return abs(signed_area(xs, ys))


def polygon_area_xy(points: np.ndarray) -> float:
    """Same, for an (N, 2) array of [x, y] vertices."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) array of points, got shape {pts.shape}")
    if pts.shape[0] < 3:
        return 0.0
    return polygon_area(pts[:, 0], pts[:, 1])


def is_degenerate(points: np.ndarray, tol: float = 1e-12) -> bool:
    """True if the polygon encloses no area (fewer than 3 vertices, or collinear).

    Callers should record degenerate polygons rather than dropping them
    silently: in annotation data a degenerate ROI usually means a labelling
    slip worth surfacing, not an empty region.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return True
    return polygon_area_xy(pts) <= tol


def bounding_box(points: np.ndarray) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) in continuous coordinates."""
    pts = np.asarray(points, dtype=np.float64)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


# ========================================================================
# clip
# ========================================================================

# Edge codes for the four half-planes of the clip rectangle.
_LEFT, _RIGHT, _BOTTOM, _TOP = 0, 1, 2, 3


def _inside(point: np.ndarray, edge: int, xmin: float, ymin: float,
            xmax: float, ymax: float) -> bool:
    x, y = point
    if edge == _LEFT:
        return x >= xmin
    if edge == _RIGHT:
        return x <= xmax
    if edge == _BOTTOM:
        return y >= ymin
    return y <= ymax


def _intersect(a: np.ndarray, b: np.ndarray, edge: int, xmin: float, ymin: float,
               xmax: float, ymax: float) -> np.ndarray:
    """Point where segment a->b crosses the given clip edge.

    Division is safe because this is only reached when a and b lie strictly on
    opposite sides of the edge, which makes the denominator non-zero.
    """
    ax, ay = a
    bx, by = b

    if edge in (_LEFT, _RIGHT):
        xe = xmin if edge == _LEFT else xmax
        t = (xe - ax) / (bx - ax)
        return np.array([xe, ay + t * (by - ay)], dtype=np.float64)

    ye = ymin if edge == _BOTTOM else ymax
    t = (ye - ay) / (by - ay)
    return np.array([ax + t * (bx - ax), ye], dtype=np.float64)


def clip_polygon_to_rect(points: np.ndarray, xmin: float, ymin: float,
                         xmax: float, ymax: float) -> np.ndarray:
    """Clip a simple polygon to a rectangle. Returns an (M, 2) array, possibly empty.

    Sutherland-Hodgman clips against each of the four half-planes in turn. It
    is exact for convex clip regions, which a rectangle always is. The subject
    polygon may be concave; the result may then contain degenerate connecting
    edges, but its enclosed area remains correct, which is all this is used for.
    """
    poly = np.asarray(points, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) array of points, got shape {poly.shape}")
    if poly.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float64)

    for edge in (_LEFT, _RIGHT, _BOTTOM, _TOP):
        if poly.shape[0] == 0:
            break

        output: list[np.ndarray] = []
        prev = poly[-1]
        prev_in = _inside(prev, edge, xmin, ymin, xmax, ymax)

        for curr in poly:
            curr_in = _inside(curr, edge, xmin, ymin, xmax, ymax)

            if curr_in:
                if not prev_in:
                    output.append(_intersect(prev, curr, edge, xmin, ymin, xmax, ymax))
                output.append(curr)
            elif prev_in:
                output.append(_intersect(prev, curr, edge, xmin, ymin, xmax, ymax))

            prev, prev_in = curr, curr_in

        poly = np.asarray(output, dtype=np.float64) if output else np.empty((0, 2))

    return poly


def clipped_area(points: np.ndarray, xmin: float, ymin: float,
                 xmax: float, ymax: float) -> float:
    """Exact area of the polygon's intersection with the rectangle."""
    clipped = clip_polygon_to_rect(points, xmin, ymin, xmax, ymax)
    if clipped.shape[0] < 3:
        return 0.0
    return polygon_area_xy(clipped)


def pixel_bounds(row: int, col: int) -> tuple[float, float, float, float]:
    """Continuous bounds of one pixel, under the centres-at-integers convention.

    Returns (xmin, ymin, xmax, ymax).
    """
    return (col - 0.5, row - 0.5, col + 0.5, row + 0.5)


# ========================================================================
# coverage
# ========================================================================

def polygon_coverage(points: np.ndarray, shape: tuple[int, int],
                     out: np.ndarray | None = None,
                     accumulate: bool = False) -> np.ndarray:
    """Coverage fraction in [0, 1] for every pixel a polygon touches.

    Parameters
    ----------
    points : (N, 2) array of [x, y] in continuous pixel-index coordinates.
    shape  : (rows, cols) of the slice, i.e. numpy (y, x) order.
    out    : optional array to write into, for accumulating many polygons.
    accumulate : add to `out` rather than overwrite. Overlapping polygons can
        then exceed 1.0, which the caller must decide how to handle; this
        function will not silently clip, because a coverage above 1 means the
        annotation has overlapping ROIs and that is worth seeing.

    Only pixels within the polygon's bounding box are visited, so cost scales
    with lesion size and not with slice size.
    """
    pts = np.asarray(points, dtype=np.float64)
    rows, cols = shape

    if out is None:
        out = np.zeros(shape, dtype=np.float64)
    elif out.shape != shape:
        raise ValueError(f"out has shape {out.shape}, expected {shape}")

    if is_degenerate(pts):
        return out

    xmin, ymin, xmax, ymax = bounding_box(pts)

    # Pixel centres at integers means pixel j spans [j-0.5, j+0.5], so the
    # first pixel the polygon can touch is the one whose upper bound exceeds
    # xmin: that is ceil(xmin - 0.5). round() would be wrong on exact halves.
    col0 = max(0, int(np.ceil(xmin - 0.5)))
    col1 = min(cols - 1, int(np.floor(xmax + 0.5)))
    row0 = max(0, int(np.ceil(ymin - 0.5)))
    row1 = min(rows - 1, int(np.floor(ymax + 0.5)))

    if col0 > col1 or row0 > row1:
        return out

    for row in range(row0, row1 + 1):
        for col in range(col0, col1 + 1):
            px_xmin, px_ymin, px_xmax, px_ymax = pixel_bounds(row, col)
            area = clipped_area(pts, px_xmin, px_ymin, px_xmax, px_ymax)
            if area <= 0.0:
                continue
            if accumulate:
                out[row, col] += area
            else:
                out[row, col] = max(out[row, col], area)

    return out


def coverage_from_polygons(polygons: list[np.ndarray], shape: tuple[int, int],
                           accumulate: bool = False) -> np.ndarray:
    """Coverage map for several polygons on one slice.

    accumulate=False takes the maximum where polygons overlap, which is the
    right choice for labels (a voxel cannot be more than fully calcium).
    accumulate=True sums, which is the right choice when you need to detect
    that the annotation contains overlapping ROIs at all.
    """
    out = np.zeros(shape, dtype=np.float64)
    for poly in polygons:
        polygon_coverage(poly, shape, out=out, accumulate=accumulate)
    return out


def touched_mask(points: np.ndarray, shape: tuple[int, int],
                 min_coverage: float = 0.0) -> np.ndarray:
    """Boolean mask of pixels the polygon overlaps at all.

    This replaces the `pts.astype(np.int32)` + cv2.fillPoly pattern used to
    sample peak HU. Integer truncation collapses a sub-pixel polygon to a
    degenerate one, fillPoly then draws nothing, the HU sample comes back empty
    and the lesion vanishes from the total. Any polygon with non-zero area
    touches at least one pixel here, so that failure cannot occur.
    """
    cov = polygon_coverage(points, shape)
    return cov > min_coverage
