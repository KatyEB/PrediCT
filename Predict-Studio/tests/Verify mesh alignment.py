"""
verify_mesh_alignment.py — Check a real run's meshes against its own volume.

test_mesh.py checks the geometry code on synthetic shapes. This checks the
OUTPUT of a real run against the data it came from, which is where a mismatch
between stages would show up: a heart mask on a different grid, an isosurface
built from the wrong array, a display flip leaking into the geometry.

It also verifies the premise the 3D slice-plane UVs rest on — that render.py's
PNGs are the array flipped up-down and left-right — because that derivation is
in a comment in view3d.js and nothing else tests it.

Every check prints a number and PASS/FAIL. Nothing is fixed automatically.

Run:
    python tests/verify_mesh_alignment.py data/out/172/a3-coverage-v2
"""
import sys
import json
import csv
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage


# ── minimal binary PLY reader (mesh.py writes; this reads back) ──────────
def read_ply(path: Path):
    raw = path.read_bytes()
    end = raw.index(b"end_header") + len(b"end_header") + 1
    head = raw[:end].decode("ascii")
    nv = nf = 0
    for line in head.splitlines():
        if line.startswith("element vertex"):
            nv = int(line.split()[-1])
        elif line.startswith("element face"):
            nf = int(line.split()[-1])
    assert "binary_little_endian" in head, f"{path.name}: not binary LE"
    body = raw[end:]
    verts = np.frombuffer(body[:nv * 12], "<f4").reshape(nv, 3)
    faces = np.zeros((nf, 3), np.int32)
    off = nv * 12
    for i in range(nf):
        n = body[off]
        assert n == 3, f"{path.name}: face {i} has {n} vertices, expected 3"
        faces[i] = np.frombuffer(body[off + 1:off + 13], "<i4")
        off += 13
    assert off == len(body), f"{path.name}: {len(body) - off} trailing bytes"
    return verts, faces


def signed_volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


results = []


def check(name, ok, detail):
    results.append((ok, name))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")


def main(out_dir: Path):
    import SimpleITK as sitk

    run = json.loads((out_dir / "run.json").read_text())
    man = run.get("mesh")
    if man is None:
        print("run.json has no 'mesh' block — this study predates meshing. "
              "Delete data/work/<study>/ and re-run.")
        return 1

    prob = sitk.GetArrayFromImage(sitk.ReadImage(str(out_dir / "pred.nii.gz")))
    sx, sy, sz = man["spacing_mm"]
    print(f"\nvolume {prob.shape} (Z,Y,X)  spacing {sx} x {sy} x {sz} mm"
          f"  extent {man['extent_mm']} mm")

    # ---------------------------------------------------------------- 1
    print("\n1. DECLARED FRAME")
    check("axes declared as R/A/S",
          man["axes"] == {"x": "R", "y": "A", "z": "S"},
          f"axes = {man['axes']}")
    want = [round((n - 1) * s, 3) for n, s in zip(prob.shape[::-1], (sx, sy, sz))]
    check("extent_mm matches the volume",
          np.allclose(man["extent_mm"], want, atol=1e-3),
          f"manifest {man['extent_mm']} vs computed {want}")

    # ---------------------------------------------------------------- 2
    print("\n2. LESION SURFACES vs THE VOLUME THEY CAME FROM")
    for s in man["surfaces"]:
        lv = s["level"]
        v, f = read_ply(out_dir / s["file"])
        if not len(v):
            check(f"p{lv} empty surface", float(prob.max()) <= lv,
                  f"volume max {prob.max():.3f} <= level {lv}")
            continue

        # normals outward — the failure that renders every surface inside-out
        vol = signed_volume(v, f)
        check(f"p{lv} normals point outward", vol > 0,
              f"signed volume {vol:,.1f} mm3 ({vol/1000:.2f} cm3)")

        # mesh bbox must match the voxel bbox of prob > level, to ~1 voxel.
        # An axis transpose or a mirrored flip fails here loudly.
        k, j, i = np.nonzero(prob > lv)
        vox_lo = np.array([i.min() * sx, j.min() * sy, k.min() * sz])
        vox_hi = np.array([i.max() * sx, j.max() * sy, k.max() * sz])
        m_lo, m_hi = v.min(axis=0), v.max(axis=0)
        tol = np.array([sx, sy, sz]) * 1.5
        ok = (np.abs(m_lo - vox_lo) <= tol).all() and (np.abs(m_hi - vox_hi) <= tol).all()
        check(f"p{lv} bbox matches voxels above {lv}", ok,
              f"mesh  lo {m_lo.round(2)} hi {m_hi.round(2)}\n"
              f"         voxel lo {vox_lo.round(2)} hi {vox_hi.round(2)}  tol {tol.round(2)}")

        # Enclosed volume, against a HARD geometric bound rather than a guess.
        #
        # An earlier version of this check compared mesh volume to the count of
        # supra-threshold voxels and failed anything above 1.25x. That was wrong,
        # and a real run failed it at 1.27. Marching cubes places the surface by
        # interpolation BETWEEN voxel centres, so it legitimately extends up to
        # one voxel beyond the last supra-threshold centre — which at 3 mm slice
        # spacing, on a lesion one or two slices thick, is a large fraction of
        # the object. Inflation above 1x is ordinary here, not a defect.
        #
        # The bound that is actually true: the surface cannot reach further than
        # one voxel from any supra-threshold voxel. So dilating that voxel set by
        # one in every direction gives a volume the mesh must not exceed. A unit
        # or spacing error blows past it; honest interpolation never does.
        mask = prob > lv
        vox_vol = float(mask.sum()) * sx * sy * sz
        bound = float(ndimage.binary_dilation(mask, iterations=1).sum()) * sx * sy * sz
        check(f"p{lv} enclosed volume within the interpolation bound",
              0.02 * vox_vol < vol <= bound,
              f"mesh {vol:,.0f} mm3 · voxel centres {vox_vol:,.0f} mm3 "
              f"(ratio {vol/vox_vol if vox_vol else 0:.2f}) · hard bound {bound:,.0f} mm3")

    # ---------------------------------------------------------------- 3
    print("\n3. NESTING — the coverage argument")
    vols = []
    for s in man["surfaces"]:
        v, f = read_ply(out_dir / s["file"])
        vols.append((s["level"], signed_volume(v, f) if len(v) else 0.0))
    if len(vols) > 1:
        ok = all(vols[a][1] > vols[a + 1][1] for a in range(len(vols) - 1))
        check("higher levels enclose less", ok,
              " > ".join(f"p{lv}: {vv/1000:.2f} cm3" for lv, vv in vols))
    else:
        print(f"         single surface (binary model): p{vols[0][0]} "
              f"{vols[0][1]/1000:.2f} cm3 — nothing to nest")

    # ---------------------------------------------------------------- 4
    print("\n4. EVERY COUNTED LESION HAS GEOMETRY")
    rows = list(csv.DictReader((out_dir / "lesions_3d.csv").open()))
    counted = [r for r in rows if r["included"].lower() == "true"]
    lowest = min(man["surfaces"], key=lambda s: s["level"])
    v, _ = read_ply(out_dir / lowest["file"])
    missing = []
    for r in counted:
        lo = np.array([(float(r["bbox_x0"]) - 2) * sx, (float(r["bbox_y0"]) - 2) * sy,
                       (float(r["slice_min"]) - 1) * sz])
        hi = np.array([(float(r["bbox_x1"]) + 2) * sx, (float(r["bbox_y1"]) + 2) * sy,
                       (float(r["slice_max"]) + 1) * sz])
        if not ((v >= lo) & (v <= hi)).all(axis=1).any():
            missing.append(r["lesion_3d_key"])
    check(f"all {len(counted)} counted lesions have vertices in their bbox",
          not missing,
          "none missing" if not missing else
          f"NO GEOMETRY for {missing} — click-to-select will not find them")

    # ---------------------------------------------------------------- 5
    print("\n5. HEART SHELL")
    if man.get("heart") is None:
        print("         no heart shell in this run (uncropped, or work/ cached "
              "before meshing existed). Not a failure; report it.")
    else:
        hv, hf = read_ply(out_dir / man["heart"]["file"])
        check("heart normals point outward", signed_volume(hv, hf) > 0,
              f"signed volume {signed_volume(hv, hf)/1000:.1f} cm3 "
              f"(an adult heart is roughly 500-900 cm3 including the +8 mm margin)")
        lo, hi = hv.min(axis=0), hv.max(axis=0)
        # extent_mm is measured between voxel CENTRES. When the mask reaches the
        # edge of the crop — which the heart does, because the +8 mm margin gets
        # clipped by the original volume — the isosurface sits half a voxel
        # further out, by construction. An earlier version of this check used a
        # flat 1 mm slack and failed on z, where half a voxel is 1.5 mm.
        pad = np.array([sx, sy, sz]) * 0.5 + 0.01
        ext = np.array(man["extent_mm"])
        touches = [ax for ax, h, e, pd in zip("xyz", hi, ext, pad) if h > e - pd]
        check("heart shell within half a voxel of the volume bounds",
              (lo >= -pad).all() and (hi <= ext + pad).all(),
              f"lo {lo.round(1)} hi {hi.round(1)} vs extent {ext} (slack {pad.round(2)})"
              + (f"\n         mask reaches the crop edge on: {', '.join(touches)} "
                 f"— expected when the 8 mm margin is clipped" if touches else ""))

        # lesions should mostly sit inside the heart box; if they sit outside,
        # image and mask were cropped or reoriented differently.
        lv_mesh, _ = read_ply(out_dir / lowest["file"])
        inside = ((lv_mesh >= lo) & (lv_mesh <= hi)).all(axis=1).mean() if len(lv_mesh) else 0
        check("lesions fall inside the heart shell bounds", inside > 0.85,
              f"{inside*100:.1f} % of lesion vertices are within the heart bbox "
              f"(low means the mask and CT are misaligned)")

    # ---------------------------------------------------------------- 6
    print("\n6. PNG <-> ARRAY FLIP  (the premise view3d.js's plane UVs rest on)")
    try:
        from PIL import Image
    except ImportError:
        print("         Pillow not installed — skipped. Install it and re-run.")
    else:
        z = int(np.argmax((prob > 0.1).sum(axis=(1, 2))))
        png = out_dir / "slices" / "mask" / f"slice_{z:03d}.png"
        if not png.exists():
            print(f"         {png} missing — skipped")
        else:
            im = np.array(Image.open(png).convert("RGB")).astype(np.float32)
            # the mask overlay is ember (201,84,31): redness rises where prob does
            redness = im[..., 0] - im[..., 2]
            # render.py: PNG(r,c) = array(H-1-r, W-1-c). Undo it and correlate.
            unflipped = np.flipud(np.fliplr(redness))
            p = prob[z]
            corr = float(np.corrcoef(unflipped.ravel(), p.ravel())[0, 1])
            mirrored = float(np.corrcoef(redness.ravel(), p.ravel())[0, 1])
            check("un-flipping the PNG reproduces the array",
                  corr > 0.5 and corr > mirrored + 0.1,
                  f"slice {z}: corr(unflipped, prob) = {corr:.3f}, "
                  f"corr(as-stored, prob) = {mirrored:.3f}\n"
                  f"         if these are close, the slice is near-symmetric — "
                  f"try another study before trusting it")

    # ---------------------------------------------------------------- 7
    print("\n7. PAYLOAD")
    tot = sum(s["bytes"] for s in man["surfaces"])
    if man.get("heart"):
        tot += man["heart"]["bytes"]
    for s in man["surfaces"]:
        print(f"         p{s['level']:<5} {s['n_faces']:>8,} faces  {s['bytes']/1024:>8.0f} KB")
    if man.get("heart"):
        print(f"         heart  {man['heart']['n_faces']:>8,} faces  "
              f"{man['heart']['bytes']/1024:>8.0f} KB")
    check("total mesh payload under 3 MB", tot < 3 * 1024 * 1024,
          f"{tot/1024:.0f} KB total")

    # ---------------------------------------------------------------- 8
    # Not pass/fail — a measurement. A3's whole argument is that its output is
    # graded rather than binary, so how much of the predicted volume actually
    # sits in the soft band is the number that argument rests on. It has never
    # been reported.
    print("\n8. HOW SOFT IS THE OUTPUT  (measurement, not a check)")
    edges = [0.10, 0.25, 0.50, 0.75, 0.90, 1.01]
    inside = prob[prob > 0.10]
    if inside.size == 0:
        print("         nothing above 0.10")
    else:
        vv = sx * sy * sz
        for a, b in zip(edges[:-1], edges[1:]):
            n = int(((prob >= a) & (prob < b)).sum())
            print(f"         p {a:.2f}-{b if b <= 1 else 1.0:.2f}  {n:>7,} voxels  "
                  f"{n*vv:>9,.1f} mm3  {n/inside.size*100:>5.1f} % of predicted volume")
        soft = int(((prob >= 0.25) & (prob < 0.75)).sum())
        print(f"         --> {soft/inside.size*100:.1f} % of predicted volume lies "
              f"between p 0.25 and p 0.75")
        print(f"         mean p above 0.10 = {float(inside.mean()):.3f}, "
              f"median = {float(np.median(inside)):.3f}")
        if man["output_mode"] == "coverage" and soft / inside.size < 0.10:
            print("         NOTE: this coverage model is behaving close to binary. "
                  "Report this — it bears directly on the A1-vs-A3 claim.")

    n_fail = sum(1 for ok, _ in results if not ok)
    print(f"\n{'='*66}\n{len(results)-n_fail} passed, {n_fail} failed\n{'='*66}")
    if n_fail:
        print("FAILED:", ", ".join(n for ok, n in results if not ok))
    return 1 if n_fail else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))