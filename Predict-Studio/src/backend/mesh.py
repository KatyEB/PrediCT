"""
mesh.py — Isosurfaces from the prediction volume, for the 3D view.

Turns the probability volume into triangle meshes the browser can display, plus
a heart shell from the TotalSegmentator mask so the lesions have somewhere to
sit. Everything is computed here, in Python, in millimetres; the viewer loads
files and draws them and does no geometry of its own.

Two things about this module are load-bearing and easy to get wrong:

1. AN ISOSURFACE NEEDS A THRESHOLD, WHICH A COVERAGE MODEL DOES NOT HAVE.
   Drawing A3 at a single level would render a binary picture of a model whose
   entire contribution is that it does not threshold. So a coverage model gets
   THREE nested surfaces (0.25 / 0.50 / 0.75) and a binary model gets one, and
   the difference between the two is visible rather than argued.

   No isosurface level is used in scoring. scoring.py sums coverage; these
   levels exist only so a surface can be drawn at all.

2. NO SMOOTHING ON LESIONS. Slices are 3.0 mm and pixels are 0.37 mm, so
   marching cubes returns visible stacked plates. That is what the data is.
   Smoothing would invent geometry between slices that was never measured, in
   a project whose central finding is about label geometry. The heart shell IS
   smoothed, because it is context rather than measurement, and says so in the
   manifest.

Coordinate frame — declared, not inferred (Principle 7). Vertices are in
millimetres on the volume's own grid, origin at the centre of voxel (0,0,0):

    mesh x = array index i (columns) * sx   ->  patient RIGHT
    mesh y = array index j (rows)    * sy   ->  patient ANTERIOR
    mesh z = array index k (slices)  * sz   ->  patient SUPERIOR

which follows from direction cosines diag(-1,-1,1) in ITK's LPS convention.
No display flip is baked in — render.py's flipud/fliplr are for 2D PNGs only.
The viewer orients its camera from the "axes" field in mesh_index.json.

Does NOT: read or write NIfTI, load models, compute scores, or know about HTTP.
Called by: run.py, after scoring.

Usage:
    manifest = build_meshes(prob, heart_mask, spacing, "coverage", out_dir)
"""
import struct
import numpy as np
from pathlib import Path
from skimage import measure
import scipy.ndimage as ndimage

# Nested levels for a coverage model. Chosen to bracket the 0.5 a binary model
# would use, so the middle surface is directly comparable to A1 and the outer
# and inner ones show what A1 rounds away. Not swept — see open item 11.
LEVELS_COVERAGE = (0.25, 0.50, 0.75)
LEVELS_BINARY = (0.50,)

# The heart shell is context, not measurement, so it is downsampled in-plane
# before extraction. Averaging into larger voxels smooths the surface for free
# and decimates it in the same step. Measured on a 44 x 358 x 403 heart crop:
#   none  411,064 faces  903.1 cm3
#   x4     40,644 faces  895.6 cm3   754 KB
#   x8     14,876 faces  866.9 cm3   276 KB   <- chosen
#   x12     8,864 faces  866.0 cm3   164 KB
# x8 costs ~4 % of enclosed volume to smoothing erosion. Acceptable for a
# context shell and unacceptable for a lesion, which is why lesions are never
# downsampled.
HEART_DOWNSAMPLE_XY = 8


def _reorder_zyx_to_xyz(verts_zyx: np.ndarray, faces: np.ndarray):
    """Convert marching-cubes output to the mesh frame documented above.

    marching_cubes returns vertices as (z, y, x), wound so that normals point
    toward LOWER values — inward, for a solid — in that frame. Swapping axis 0
    and 2 is a reflection, which inverts handedness and therefore flips the
    effective winding to outward. So the reorder alone is correct and the faces
    must be left ALONE.

    Adding a winding flip here "to compensate for the reflection" is the
    intuitive move and it is wrong: it cancels the reflection and every surface
    renders inside-out, lit from within, with no error raised. That mistake was
    made and caught by test_normals_point_outward, which asserts the signed
    volume of a sphere is positive. Do not remove that test.
    """
    return verts_zyx[:, ::-1].copy(), faces


def isosurface(volume_zyx: np.ndarray, level: float, spacing_zyx: tuple):
    """Extract one isosurface.

    Args:
        volume_zyx:  (Z, Y, X) float array.
        level:       iso value.
        spacing_zyx: (sz, sy, sx) in mm — array axis order, NOT SimpleITK order.

    Returns:
        (verts, faces) with verts (N, 3) float32 in mm as (x, y, z), faces
        (M, 3) int32. Both empty if nothing in the volume reaches `level`.
    """
    if float(volume_zyx.max()) <= level:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    # Pad by one voxel of background so lesions touching the volume edge close
    # into a solid instead of producing an open surface. The pad is subtracted
    # back off the vertex coordinates so the frame is unchanged.
    padded = np.pad(volume_zyx, 1, mode="constant", constant_values=0.0)
    verts, faces, _, _ = measure.marching_cubes(padded, level=level, spacing=spacing_zyx)
    verts -= np.array(spacing_zyx, dtype=verts.dtype)

    verts, faces = _reorder_zyx_to_xyz(verts, faces)
    return verts.astype(np.float32), faces.astype(np.int32)


def heart_shell(mask_zyx: np.ndarray, spacing_zyx: tuple, downsample_xy: int = HEART_DOWNSAMPLE_XY):
    """Smoothed outer surface of the TotalSegmentator heart mask.

    The mask is binary. Downsampling it with linear interpolation turns it into
    a fractional occupancy field, and extracting at 0.5 from that field gives a
    smooth shell with far fewer triangles — smoothing and decimation in one
    step, with no mesh-adjacency code. Display context only; no number uses it.
    """
    if not mask_zyx.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    f = 1.0 / downsample_xy
    small = ndimage.zoom(mask_zyx.astype(np.float32), (1.0, f, f), order=1)
    sz, sy, sx = spacing_zyx
    return isosurface(small, 0.5, (sz, sy / f, sx / f))


def write_ply(verts: np.ndarray, faces: np.ndarray, path: Path, comment: str = ""):
    """Write a binary little-endian PLY. Chosen over glTF because it needs no
    dependency, is readable by three.js PLYLoader, and can be inspected with a
    text editor down to the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # PLY headers are ASCII by specification. A stray en-dash in a comment would
    # otherwise raise here, or worse, write a file some loaders reject.
    comment = (comment or "PrediCT").encode("ascii", "replace").decode("ascii")
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"comment {comment}",
        f"element vertex {len(verts)}",
        "property float x", "property float y", "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header", "",
    ]
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        f.write(verts.astype("<f4").tobytes())
        for tri in faces:
            f.write(struct.pack("<B3i", 3, int(tri[0]), int(tri[1]), int(tri[2])))
    return path.stat().st_size


def build_meshes(prob: np.ndarray, heart_mask: np.ndarray | None, spacing: tuple,
                 output_mode: str, out_dir: Path) -> dict:
    """Write every mesh for one run and return the manifest to embed in run.json.

    Args:
        prob:        (Z, Y, X) float32 in [0, 1], the model output.
        heart_mask:  (Z, Y, X) binary on the SAME grid, or None if the study was
                     not cropped. None is recorded, never silently substituted.
        spacing:     (sx, sy, sz) in mm — SimpleITK order, as GetSpacing()
                     returns it. Reversed internally for the array axes.
        output_mode: "binary" | "coverage" — decides how many surfaces.
        out_dir:     run output folder; meshes go in out_dir / "mesh".

    Returns:
        dict describing every file written, its level, and the coordinate frame.
    """
    if output_mode == "coverage":
        levels = LEVELS_COVERAGE
    elif output_mode == "binary":
        levels = LEVELS_BINARY
    else:
        raise ValueError(f"unknown output mode: {output_mode}")

    sx, sy, sz = spacing
    spacing_zyx = (sz, sy, sx)
    mesh_dir = out_dir / "mesh"

    extent_mm = [round(float(n - 1) * s, 3) for n, s in
                 zip(prob.shape[::-1], (sx, sy, sz))]   # (X, Y, Z) in mm

    surfaces = []
    for lv in levels:
        verts, faces = isosurface(prob, lv, spacing_zyx)
        name = f"lesions_p{int(round(lv * 100)):03d}.ply"
        size = write_ply(verts, faces, mesh_dir / name,
                         f"PrediCT lesion isosurface p={lv} mm x=R y=A z=S")
        if len(verts):
            # A vertex outside the padded volume means the axis reorder is
            # wrong, which would look plausible on screen rather than error.
            spacing_xyz = np.array([sx, sy, sz])
            assert (verts.min(axis=0) >= -spacing_xyz - 1e-3).all(), f"{name}: vertex below origin pad"
            assert (verts.max(axis=0) <= np.array(extent_mm) + spacing_xyz + 1e-3).all(), \
                f"{name}: vertex outside volume extent pad"
        surfaces.append(dict(level=lv, file=f"mesh/{name}",
                             n_vertices=len(verts), n_faces=len(faces),
                             bytes=size, smoothed=False))

    heart = None
    if heart_mask is not None:
        hv, hf = heart_shell(heart_mask, spacing_zyx)
        size = write_ply(hv, hf, mesh_dir / "heart.ply",
                         "PrediCT heart shell - display context, smoothed")
        heart = dict(file="mesh/heart.ply", n_vertices=len(hv), n_faces=len(hf),
                     bytes=size, smoothed=True,
                     downsample_xy=HEART_DOWNSAMPLE_XY)

    return dict(
        # Declared so the viewer never infers orientation from the geometry.
        axes=dict(x="R", y="A", z="S"),
        units="mm",
        origin="centre of voxel (0,0,0); no display flip applied",
        extent_mm=extent_mm,
        spacing_mm=[sx, sy, sz],
        output_mode=output_mode,
        levels=list(levels),
        surfaces=surfaces,
        heart=heart,
        note=("Lesion surfaces are unsmoothed: 3.0 mm slices against 0.37 mm "
              "pixels produce visible stepping, which is the true resolution "
              "of the data. Isosurface levels are for display only and are "
              "not used by any score."),
    )
