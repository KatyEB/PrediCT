"""
grouping.py — Link per-slice lesion components into 3D lesions.

scoring.py finds connected components independently on each slice, because that
is how Agatston is defined: area x density weight, computed in-plane, summed
over slices. That is correct and must not change. But it means one anatomical
lesion spanning slices 24, 25 and 26 appears as three unrelated rows.

This module adds the missing identity. It never recomputes area, never touches
a density weight, and never changes a score. It only answers: which per-slice
components are the same lesion?

    The rule: two components on nearby slices are the same lesion if they
    share at least one (y, x) pixel. Links chain transitively, so a lesion
    that drifts sideways up the stack stays one lesion even when its first
    and last slices do not overlap each other at all.

Why overlap and not scipy 3D labelling: voxels here are 0.37 x 0.37 x 3.0 mm.
A 3x3x3 structuring element treats a 3 mm z-neighbour as equivalent to a
0.37 mm in-plane neighbour, so two specks 3 mm apart in z would merge on a
diagonal touch. Overlap makes the adjacency rule explicit and defensible.

max_gap_slices exists because 3 mm slices are coarse: a real vessel
calcification can fade below threshold on one slice and reappear on the next.
Default 0 (strictly adjacent slices) — turn it up only with a measurement.

Does NOT: read files, import torch, compute areas, or decide inclusion.
Called by: scoring.py, at the end of score().

Usage:
    link_lesions(planes, rows, max_gap_slices=0)   # annotates rows in place
    groups = lesion_3d_table(rows)                 # one row per 3D lesion
"""
import numpy as np


def _find(parent: dict, a):
    """Union-find root with path compression. Keys are (slice_idx, label)."""
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _union(parent: dict, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def overlapping_pairs(labelled_a: np.ndarray, labelled_b: np.ndarray) -> set:
    """Labels that share at least one pixel between two labelled slices.

    Args:
        labelled_a: (Y, X) int, output of ndimage.label on one slice.
        labelled_b: (Y, X) int, same shape, a different slice.

    Returns:
        set of (label_in_a, label_in_b) pairs that co-occur at some pixel.
    """
    assert labelled_a.shape == labelled_b.shape, \
        f"slice shapes differ: {labelled_a.shape} vs {labelled_b.shape}"
    both = (labelled_a > 0) & (labelled_b > 0)
    if not both.any():
        return set()
    return set(zip(labelled_a[both].tolist(), labelled_b[both].tolist()))


def link_lesions(planes: dict, rows: list[dict], max_gap_slices: int = 0) -> list[dict]:
    """Assign a 3D lesion id to every per-slice component row.

    Args:
        planes: {slice_idx: (Y, X) int array} from ndimage.label, one entry per
                slice that had any component. Slices absent from this dict are
                empty and simply break any chain running through them.
        rows:   the list of per-slice lesion dicts from scoring.score(). Each
                must carry slice_idx and label_2d. Mutated in place.
        max_gap_slices: how many slices a link may jump over. 0 links z to z+1
                only. 1 also links z to z+2, so one intervening slice may be
                empty or non-overlapping. Unconditional — it does not check
                whether the skipped slice was empty.

    Returns:
        the same rows list, each row gaining lesion_3d_id and lesion_3d_key.

    Grouping is computed over ALL components including sub-minimum ones, because
    adjacency is a geometric fact and the 1 mm2 rule is a scoring convention.
    A withheld speck can therefore bridge two components; lesion_3d_table()
    reports n_components_included so that case stays visible.
    """
    keys = [(r["slice_idx"], r["label_2d"]) for r in rows]
    parent = {k: k for k in keys}

    zs = sorted(planes.keys())
    for z in zs:
        for gap in range(1, max_gap_slices + 2):
            z2 = z + gap
            if z2 not in planes:
                continue
            for la, lb in overlapping_pairs(planes[z], planes[z2]):
                _union(parent, (z, la), (z2, lb))

    # Number groups by first appearance so ids are stable across re-runs:
    # lowest slice first, then lowest 2D label on that slice.
    order = {}
    for r in sorted(rows, key=lambda r: (r["slice_idx"], r["label_2d"])):
        root = _find(parent, (r["slice_idx"], r["label_2d"]))
        if root not in order:
            order[root] = len(order) + 1

    for r in rows:
        gid = order[_find(parent, (r["slice_idx"], r["label_2d"]))]
        r["lesion_3d_id"] = gid
        r["lesion_3d_key"] = f"L{gid:03d}"
    return rows


def lesion_3d_table(rows: list[dict], spacing_z_mm: float) -> list[dict]:
    """Roll per-slice components up into one row per 3D lesion.

    Totals here cover INCLUDED components only, so sum(total_agatston) over this
    table equals the patient total from scoring.totals(). n_components counts
    every component in the group, included or not.

    Centroids are area-weighted over the components that contributed to the
    total, in unflipped array coordinates — the same frame lesions.csv uses.
    """
    by_gid = {}
    for r in rows:
        by_gid.setdefault(r["lesion_3d_id"], []).append(r)

    out = []
    for gid in sorted(by_gid):
        g = sorted(by_gid[gid], key=lambda r: r["slice_idx"])
        kept = [r for r in g if r["included"]]
        w = kept if kept else g          # centroid still defined for withheld-only groups

        wsum = sum(r["area_mm2"] for r in w) or 1.0
        peak = max(kept, key=lambda r: r["agatston"]) if kept else g[0]
        zi = sorted({r["slice_idx"] for r in g})

        out.append(dict(
            lesion_3d_key=f"L{gid:03d}",
            lesion_3d_id=gid,
            included=bool(kept),
            n_slices=len(zi),
            n_components=len(g),
            n_components_included=len(kept),
            slice_min=zi[0],
            slice_max=zi[-1],
            # span is inclusive of both end slices: a single-slice lesion is 3 mm thick
            span_mm=round((zi[-1] - zi[0] + 1) * spacing_z_mm, 2),
            slices=" ".join(str(i) for i in zi),
            total_area_mm2=sum(r["area_mm2"] for r in kept),
            total_agatston=sum(r["agatston"] for r in kept),
            max_peak_hu=max(r["peak_hu"] for r in g),
            max_density_weight=max(r["density_weight"] for r in g),
            peak_slice_idx=peak["slice_idx"],
            centroid_x=sum(r["centroid_x"] * r["area_mm2"] for r in w) / wsum,
            centroid_y=sum(r["centroid_y"] * r["area_mm2"] for r in w) / wsum,
            bbox_x0=min(r["bbox_x0"] for r in g),
            bbox_y0=min(r["bbox_y0"] for r in g),
            bbox_x1=max(r["bbox_x1"] for r in g),
            bbox_y1=max(r["bbox_y1"] for r in g),
        ))
    return out
