"""
test_grouping.py — 3D lesion linking, on hand-checkable volumes.

Every case here is small enough to verify by eye. The last two are the ones
that matter: grouping must never move a score, and it must not merge lesions
that are only diagonally adjacent across a 3 mm gap.

Run:  python -m pytest tests/test_grouping.py -v
      python tests/test_grouping.py          (no pytest needed)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.backend.scoring import score, totals
from src.backend.grouping import lesion_3d_table

SPACING = (0.37, 0.37, 3.0)
PIX = SPACING[0] * SPACING[1]


def build(shape, blobs, hu=350.0):
    """blobs: {z: [(y0, y1, x0, x1), ...]} inclusive slices. Returns prob, hu arrays."""
    prob = np.zeros(shape, dtype=np.float32)
    for z, boxes in blobs.items():
        for y0, y1, x0, x1 in boxes:
            prob[z, y0:y1 + 1, x0:x1 + 1] = 1.0
    hu_vol = np.where(prob > 0, hu, -50.0).astype(np.float32)
    return prob, hu_vol


def keys(rows):
    return {(r["slice_idx"], r["label_2d"]): r["lesion_3d_key"] for r in rows}


def test_column_is_one_lesion():
    """Same blob on three consecutive slices -> one 3D lesion, three components."""
    prob, hu = build((6, 40, 40), {2: [(10, 15, 10, 15)],
                                   3: [(10, 15, 10, 15)],
                                   4: [(10, 15, 10, 15)]})
    rows = score(prob, hu, SPACING, "binary", 0.1)
    assert len(rows) == 3
    assert len({r["lesion_3d_key"] for r in rows}) == 1
    g = lesion_3d_table(rows, SPACING[2])
    assert len(g) == 1 and g[0]["n_slices"] == 3 and g[0]["span_mm"] == 9.0


def test_drifting_lesion_chains():
    """The worked example: 24/25/26 drift sideways. z=24 and z=26 do NOT overlap
    each other; they are one lesion only because z=25 bridges them. A second,
    separate blob appears on z=26."""
    prob, hu = build((30, 40, 40), {
        24: [(10, 11, 2, 3)],
        25: [(10, 11, 3, 4)],
        26: [(10, 11, 4, 5), (20, 21, 30, 31)],
    })
    rows = score(prob, hu, SPACING, "binary", 0.1)
    k = keys(rows)
    assert len(rows) == 4
    # the drifting chain
    chain = {k[(24, 1)], k[(25, 1)], k[(26, 1)]}
    assert len(chain) == 1, f"chain broke: {chain}"
    # z=24 and z=26 really do not overlap — proving transitivity did the work
    assert not ((prob[24] > 0.1) & (prob[26] > 0.1)).any(), "example is wrong: 24 and 26 overlap"
    # the far blob is its own lesion
    assert k[(26, 2)] != k[(26, 1)]
    assert len({v for v in k.values()}) == 2


def test_near_miss_does_not_merge():
    """One pixel apart in-plane, adjacent slices, no shared pixel. 26-connectivity
    would merge these; the overlap rule must not, because they are 3 mm apart."""
    prob, hu = build((6, 40, 40), {2: [(10, 10, 10, 11)],
                                   3: [(10, 10, 12, 13)]})
    rows = score(prob, hu, SPACING, "binary", 0.1)
    assert len({r["lesion_3d_key"] for r in rows}) == 2


def test_gap_bridging_is_opt_in():
    """Blob on 2 and 4, nothing on 3. Strict -> two lesions. max_gap_slices=1 -> one."""
    prob, hu = build((8, 40, 40), {2: [(10, 12, 10, 12)],
                                   4: [(10, 12, 10, 12)]})
    strict = score(prob, hu, SPACING, "binary", 0.1, max_gap_slices=0)
    bridged = score(prob, hu, SPACING, "binary", 0.1, max_gap_slices=1)
    assert len({r["lesion_3d_key"] for r in strict}) == 2
    assert len({r["lesion_3d_key"] for r in bridged}) == 1
    # and it changed nothing about the score
    assert totals(strict)["agatston_total"] == totals(bridged)["agatston_total"]


def test_ids_are_stable_and_ordered():
    """Group 1 must be the group whose first component sits lowest in z."""
    prob, hu = build((10, 40, 40), {1: [(30, 32, 30, 32)],
                                    5: [(5, 7, 5, 7)]})
    rows = score(prob, hu, SPACING, "binary", 0.1)
    k = keys(rows)
    assert k[(1, 1)] == "L001" and k[(5, 1)] == "L002"


def test_total_is_unchanged_by_grouping():
    """The invariant. Sum over per-slice rows == sum over the 3D rollup, and
    neither depends on max_gap_slices."""
    rng = np.random.default_rng(0)
    prob = (rng.random((12, 60, 60)) > 0.985).astype(np.float32)
    prob = np.repeat(np.repeat(prob, 3, axis=1), 3, axis=2)[:, :60, :60]
    hu = np.where(prob > 0, 420.0, -80.0).astype(np.float32)

    for gap in (0, 1, 2):
        rows = score(prob, hu, SPACING, "binary", 0.1, max_gap_slices=gap)
        t = totals(rows)
        g = lesion_3d_table(rows, SPACING[2])
        assert abs(sum(r["total_agatston"] for r in g) - t["agatston_total"]) < 1e-9
        assert abs(sum(r["total_area_mm2"] for r in g)
                   - sum(r["area_mm2"] for r in rows if r["included"])) < 1e-9
        assert sum(r["n_components"] for r in g) == len(rows)
        assert sum(r["n_components_included"] for r in g) == t["n_lesions"]
    print("   invariant holds for gap = 0, 1, 2")


def test_withheld_component_still_grouped():
    """A sub-1 mm2 speck is grouped like anything else, but does not enter totals."""
    prob, hu = build((6, 40, 40), {2: [(10, 15, 10, 15)],
                                   3: [(12, 12, 12, 12)]})   # 1 voxel = 0.137 mm2
    rows = score(prob, hu, SPACING, "binary", 0.1)
    g = lesion_3d_table(rows, SPACING[2])
    assert len(g) == 1
    assert g[0]["n_components"] == 2 and g[0]["n_components_included"] == 1
    assert abs(g[0]["total_agatston"] - totals(rows)["agatston_total"]) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\n{len(fns)} passed")
