"""
test_mesh.py — Geometry checks for mesh.py, on shapes with known answers.

The failure this file exists to catch is a mesh that looks plausible on screen
and is wrong: inverted normals, a transposed axis, a millimetre/voxel mix-up.
None of those raise. All of them are visible in a number here.

Run:  python -m pytest src/backend/test_mesh.py -v
      python src/backend/test_mesh.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from src.backend.mesh import isosurface, heart_shell, write_ply, build_meshes

SPACING_SITK = (0.37, 0.37, 3.0)      # (sx, sy, sz), as GetSpacing() returns
SPACING_ZYX = (3.0, 0.37, 0.37)       # array axis order


def signed_volume(verts, faces):
    """Divergence-theorem volume. Positive iff triangles wind counter-clockwise
    seen from outside — i.e. iff the normals point outward."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def ball(shape, centre_vox, radius_vox, spacing_zyx=None):
    """Binary ball. If spacing is given the radius is in mm and the ball is a
    true sphere in physical space, not in voxel space."""
    zz, yy, xx = np.indices(shape).astype(np.float32)
    if spacing_zyx is None:
        d = ((zz - centre_vox[0]) ** 2 + (yy - centre_vox[1]) ** 2 + (xx - centre_vox[2]) ** 2)
        return (d <= radius_vox ** 2).astype(np.float32)
    sz, sy, sx = spacing_zyx
    d = (((zz - centre_vox[0]) * sz) ** 2 + ((yy - centre_vox[1]) * sy) ** 2
         + ((xx - centre_vox[2]) * sx) ** 2)
    return (d <= radius_vox ** 2).astype(np.float32)


def test_normals_point_outward():
    """The whole point of the winding flip in _reorder_zyx_to_xyz. If this ever
    goes negative, every surface renders inside-out and nobody notices."""
    v, f = isosurface(ball((24, 24, 24), (12, 12, 12), 7), 0.5, (1.0, 1.0, 1.0))
    assert len(f) > 0
    assert signed_volume(v, f) > 0, "normals inverted — winding flip is wrong"


def test_volume_matches_analytic_sphere():
    """Isotropic sphere of radius 7 voxels: mesh volume within 5 % of 4/3 pi r^3.
    Catches a spacing applied to the wrong axis, which a shape check would not."""
    v, f = isosurface(ball((32, 32, 32), (16, 16, 16), 7), 0.5, (1.0, 1.0, 1.0))
    got = signed_volume(v, f)
    want = 4 / 3 * np.pi * 7 ** 3
    assert abs(got - want) / want < 0.05, f"volume {got:.1f} vs analytic {want:.1f}"


def test_anisotropic_spacing_lands_on_the_right_axis():
    """A slab one voxel thick in z must be 3.0 mm thick in the mesh, not 0.37.
    This is the mix-up that would make every lesion look like a wafer."""
    vol = np.zeros((9, 40, 40), np.float32)
    vol[4, 10:30, 10:30] = 1.0
    v, f = isosurface(vol, 0.5, SPACING_ZYX)
    ext = v.max(axis=0) - v.min(axis=0)          # (x, y, z) mm
    assert abs(ext[2] - 3.0) < 0.4, f"z extent {ext[2]:.2f} mm, expected ~3.0"
    assert 6.0 < ext[0] < 8.5, f"x extent {ext[0]:.2f} mm, expected ~7.4"
    assert abs(ext[0] - ext[1]) < 0.1, "x and y should match for a square slab"


def test_axis_identity_is_not_transposed():
    """A blob offset only in array x must be offset only in mesh x. A swapped
    pair of axes gives a mesh that is geometrically fine and anatomically wrong."""
    vol = np.zeros((10, 40, 60), np.float32)
    vol[4:6, 18:22, 40:46] = 1.0                  # far along x, middle of y
    v, _ = isosurface(vol, 0.5, SPACING_ZYX)
    cx, cy, cz = v.mean(axis=0)
    assert abs(cx - 42.5 * 0.37) < 0.6, f"x centroid {cx:.2f}"
    assert abs(cy - 19.5 * 0.37) < 0.6, f"y centroid {cy:.2f}"
    assert abs(cz - 4.5 * 3.0) < 0.6, f"z centroid {cz:.2f}"


def test_empty_volume_returns_empty_not_error():
    v, f = isosurface(np.zeros((8, 20, 20), np.float32), 0.5, SPACING_ZYX)
    assert len(v) == 0 and len(f) == 0


def test_level_below_max_only():
    """A volume whose maximum is 0.4 has no 0.5 surface and must not invent one."""
    vol = np.zeros((8, 20, 20), np.float32)
    vol[3:5, 8:12, 8:12] = 0.4
    assert len(isosurface(vol, 0.50, SPACING_ZYX)[0]) == 0
    assert len(isosurface(vol, 0.25, SPACING_ZYX)[0]) > 0


def test_nested_levels_are_nested():
    """A soft-edged blob: the 0.75 surface must enclose less volume than 0.25.
    This is the coverage argument the 3D view exists to make."""
    zz, yy, xx = np.indices((16, 48, 48)).astype(np.float32)
    d = np.sqrt(((zz - 8) * 3.0) ** 2 + ((yy - 24) * 0.37) ** 2 + ((xx - 24) * 0.37) ** 2)
    vol = np.clip(1.0 - d / 8.0, 0, 1).astype(np.float32)   # 1 at centre, fading out
    vols = []
    for lv in (0.25, 0.50, 0.75):
        v, f = isosurface(vol, lv, SPACING_ZYX)
        vols.append(signed_volume(v, f))
    assert vols[0] > vols[1] > vols[2] > 0, f"levels not nested: {vols}"


def test_ply_roundtrip_header_and_size():
    import tempfile
    v, f = isosurface(ball((20, 20, 20), (10, 10, 10), 6), 0.5, (1.0, 1.0, 1.0))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.ply"
        size = write_ply(v, f, p)
        raw = p.read_bytes()
        head = raw[:raw.index(b"end_header") + 11].decode("ascii")
        assert f"element vertex {len(v)}" in head
        assert f"element face {len(f)}" in head
        body = len(raw) - (raw.index(b"end_header") + 11)
        assert body == len(v) * 12 + len(f) * 13, "binary payload size wrong"
        assert size == len(raw)


def test_build_meshes_manifest():
    import tempfile
    zz, yy, xx = np.indices((20, 60, 60)).astype(np.float32)
    d = np.sqrt(((zz - 10) * 3.0) ** 2 + ((yy - 30) * 0.37) ** 2 + ((xx - 30) * 0.37) ** 2)
    prob = np.clip(1.0 - d / 7.0, 0, 1).astype(np.float32)
    heart = (d < 14).astype(np.uint8)

    with tempfile.TemporaryDirectory() as t:
        out = Path(t)
        cov = build_meshes(prob, heart, SPACING_SITK, "coverage", out)
        assert len(cov["surfaces"]) == 3
        assert cov["axes"] == dict(x="R", y="A", z="S")
        assert cov["heart"]["smoothed"] is True
        assert all(s["smoothed"] is False for s in cov["surfaces"])
        for s in cov["surfaces"]:
            assert (out / s["file"]).exists()
        assert (out / cov["heart"]["file"]).exists()

        bina = build_meshes(prob, None, SPACING_SITK, "binary", out)
        assert len(bina["surfaces"]) == 1 and bina["heart"] is None

        kb = sum(s["bytes"] for s in cov["surfaces"]) / 1024
        print(f"   3 lesion surfaces = {kb:.0f} KB, "
              f"heart = {cov['heart']['bytes'] / 1024:.0f} KB "
              f"({cov['heart']['n_faces']} faces)")


def test_heart_shell_is_decimated():
    """Downsampling must actually reduce the triangle count, or the shell will
    dominate the payload."""
    heart = ball((30, 200, 200), (15, 100, 100), 20.0, SPACING_ZYX)
    full_v, full_f = isosurface(heart, 0.5, SPACING_ZYX)
    shell_v, shell_f = heart_shell(heart, SPACING_ZYX)
    assert len(shell_f) < len(full_f) / 4, \
        f"decimation weak: {len(full_f)} -> {len(shell_f)}"
    assert signed_volume(shell_v, shell_f) > 0
    print(f"   heart faces {len(full_f)} -> {len(shell_f)} "
          f"({len(shell_f) / len(full_f) * 100:.0f} %)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} passed")
