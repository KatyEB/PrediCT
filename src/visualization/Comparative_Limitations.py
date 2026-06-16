# xml_vs_mask_comparison_v3.py
# Run: python xml_vs_mask_comparison_v3.py

import json
import plistlib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import binary_erosion
from skimage.draw import polygon as sk_polygon
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT = r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2"
XML_ROOT  = rf"{DATA_ROOT}\Gated_release_final\calcium_xml"
OUT_DIR   = Path("figures")

RUNS = [
    ("0",  "2740d96a230c"),
    ("1",  "fd14b377bebc"),
    ("10", "c3be56167c58"),
]

CT_LO, CT_HI = -100, 600


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_xml_polygons(xml_path: Path) -> dict:
    result = {}
    if not xml_path.exists():
        print(f"  [!] XML not found: {xml_path}")
        return result
    with open(xml_path, "rb") as f:
        data = plistlib.load(f)
    for entry in data.get("Images", []):
        z = int(entry.get("ImageIndex", -1))
        if z < 0:
            continue
        polys = []
        for roi in entry.get("ROIs", []):
            pts = []
            for p in roi.get("Point_px", []):
                c = p.replace("(","").replace(")","").split(",")
                if len(c) == 2:
                    pts.append([float(c[0]), float(c[1])])
            if len(pts) >= 3:
                polys.append(np.array(pts, dtype=np.float32))
        if polys:
            result[z] = polys
    return result


def shoelace(pts):
    n = len(pts)
    a = sum(pts[i,0]*pts[(i+1)%n,1] - pts[(i+1)%n,0]*pts[i,1]
            for i in range(n))
    return abs(a) / 2.0


def setup_ax(ax):
    ax.set_facecolor("black")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def show_ct(ax, ct_crop, extent=None):
    kw = dict(cmap="gray", origin="upper", vmin=CT_LO, vmax=CT_HI)
    if extent is not None:
        kw["extent"] = extent
    ax.imshow(ct_crop, **kw)
    setup_ax(ax)


# ── Main ──────────────────────────────────────────────────────────────────────
def make_comparison(patient_id, scan_id, max_slices=2):
    base     = Path(DATA_ROOT) / "data_canonical" / "images" / scan_id
    xml_path = Path(XML_ROOT) / f"{patient_id}.xml"
    meta     = json.loads((base / f"{scan_id}_meta.json").read_text())

    img_arr  = sitk.GetArrayFromImage(
        sitk.ReadImage(str(base / f"{scan_id}_img.nii.gz")))   # Z,Y,X
    mask_arr = sitk.GetArrayFromImage(
        sitk.ReadImage(str(base / f"{scan_id}_seg.nii.gz")))

    orig_sp = meta["original_spacing"]
    tgt_sp  = meta["resampled_spacing"]
    sx = orig_sp[0] / tgt_sp[0]
    sy = orig_sp[1] / tgt_sp[1]
    print(f"\nPatient {patient_id} | {scan_id} | scale x={sx:.4f} y={sy:.4f}")

    xml_polys = parse_xml_polygons(xml_path)
    if not xml_polys:
        return

    H, W = img_arr.shape[1], img_arr.shape[2]
    OUT_DIR.mkdir(exist_ok=True)

    for z in sorted(xml_polys.keys())[:max_slices]:
        raw_polys = xml_polys[z]

        # Scale XML coords to resampled pixel space — NO y-flip (origin='upper')
        scaled = []
        for poly in raw_polys:
            sp = poly.copy()
            sp[:, 0] = poly[:, 0] * sx
            sp[:, 1] = poly[:, 1] * sy
            scaled.append(sp)

        mask_z = min(z, mask_arr.shape[0] - 1)
        ct     = np.clip(img_arr[min(z, img_arr.shape[0]-1)],
                         CT_LO, CT_HI).astype(float)
        mask   = mask_arr[mask_z]

        # Bounding box for zoom
        all_pts = np.vstack(scaled)
        pad = 40
        x1 = max(0, int(all_pts[:,0].min()) - pad)
        x2 = min(W, int(all_pts[:,0].max()) + pad)
        y1 = max(0, int(all_pts[:,1].min()) - pad)
        y2 = min(H, int(all_pts[:,1].max()) + pad)
        ext = [x1, x2, y2, y1]   # origin='upper' extent

        # ── Compute true mask via skimage (float-precision polygon) ────────
        true_mask = np.zeros((H, W), dtype=np.uint8)
        for sp in scaled:
            rr, cc = sk_polygon(
                sp[:, 1].clip(0, H-1),
                sp[:, 0].clip(0, W-1),
                shape=(H, W)
            )
            true_mask[rr, cc] = 1

        # ── Error categories ───────────────────────────────────────────────
        fill  = mask.astype(bool)
        true  = true_mask.astype(bool)

        correct    = fill &  true    # both agree
        overshoot  = fill & ~true    # fillPoly extra (FP)
        undershoot = ~fill &  true   # fillPoly missed (FN)

        n_correct    = int(correct.sum())
        n_over       = int(overshoot.sum())
        n_under      = int(undershoot.sum())

        # Boundary pixels for scatter (z-order fix)
        boundary   = fill & ~binary_erosion(fill)
        bd_y, bd_x = np.where(boundary)

        # ── RGBA layers ────────────────────────────────────────────────────
        orange = np.zeros((H, W, 4))
        orange[fill] = [1.0, 0.55, 0.0, 0.55]

        error_map = np.zeros((H, W, 4))
        error_map[correct]   = [1.00, 0.92, 0.00, 0.95]   # yellow
        error_map[overshoot] = [1.00, 0.30, 0.00, 0.95]   # orange-red
        error_map[undershoot]= [0.00, 0.85, 1.00, 0.95]   # cyan

        # ── Stats ──────────────────────────────────────────────────────────
        fill_area = int(mask.sum())
        xml_area  = sum(shoelace(p / np.array([sx, sy])) for p in scaled)
        err_pct   = abs(fill_area - xml_area) / xml_area * 100 if xml_area else 0

        # ── Figure: 2 rows × 5 cols ────────────────────────────────────────
        fig = plt.figure(figsize=(30, 12), facecolor="#0d0d1a")
        gs  = gridspec.GridSpec(
            2, 5, figure=fig,
            hspace=0.06, wspace=0.04,
            top=0.86, bottom=0.07,
            left=0.01, right=0.99
        )
        axes = [[fig.add_subplot(gs[r, c])
                 for c in range(5)] for r in range(2)]

        COL_TITLES = [
            "① CT Slice (Raw)",
            "② fillPoly Mask\n(integer pixel grid)",
            "③ XML Subpixel Outline\n(radiologist annotation)",
            "④ Overlay\nGreen=XML  Red=boundary (on top)",
            "⑤ ERROR MAP\nYellow=correct  Orange=over-seg  Cyan=under-seg",
        ]
        ZOOM_TITLES = [
            "Zoom — CT Only",
            "Zoom — fillPoly",
            "Zoom — XML Outline",
            "Zoom — Overlay",
            "Zoom — PURE ERROR\n(pixel grid, black bg)",
        ]

        # ── ROW 0: full slice ──────────────────────────────────────────────
        for c in range(5):
            setup_ax(axes[0][c])
            axes[0][c].set_title(COL_TITLES[c], color="white",
                                  fontsize=8.5, pad=4, fontweight="bold")
            if c != 4:
                show_ct(axes[0][c], ct)
            # cyan zoom box on cols 1-4
            if c > 0:
                axes[0][c].add_patch(
                    plt.Rectangle((x1,y1), x2-x1, y2-y1,
                                   lw=1.5, ec="cyan", fc="none", ls="--", zorder=5))

        # Col 1: orange mask
        axes[0][1].imshow(orange, origin="upper")

        # Col 2: XML green outline
        for sp in scaled:
            cl = np.vstack([sp, sp[0]])
            axes[0][2].plot(cl[:,0], cl[:,1], color="lime", lw=1.5, zorder=2)
            axes[0][2].scatter(sp[:,0], sp[:,1],
                               c="yellow", s=5, zorder=3)

        # Col 3: overlay — SWAP zorder so green is ON TOP
        axes[0][3].imshow(orange, origin="upper")
        axes[0][3].scatter(bd_x, bd_y, c="red", s=5, zorder=1, alpha=0.9)   # red BEHIND
        for sp in scaled:
            cl = np.vstack([sp, sp[0]])
            axes[0][3].plot(cl[:,0], cl[:,1],
                            color="lime", lw=1.5, zorder=10, alpha=0.95)     # green ON TOP

        # Col 4: error map (full slice, dark bg)
        axes[0][4].imshow(error_map, origin="upper")

        # ── ROW 1: zoom ───────────────────────────────────────────────────
        for c in range(5):
            setup_ax(axes[1][c])
            axes[1][c].set_title(ZOOM_TITLES[c], color="white",
                                  fontsize=8.5, pad=4, fontweight="bold")
            axes[1][c].set_xlim(x1, x2)
            axes[1][c].set_ylim(y2, y1)

        # Zoom col 0: CT
        show_ct(axes[1][0], ct[y1:y2, x1:x2], extent=ext)

        # Zoom col 1: fillPoly
        show_ct(axes[1][1], ct[y1:y2, x1:x2], extent=ext)
        axes[1][1].imshow(orange[y1:y2, x1:x2], origin="upper", extent=ext)

        # Zoom col 2: XML outline
        show_ct(axes[1][2], ct[y1:y2, x1:x2], extent=ext)
        for sp in scaled:
            cl = np.vstack([sp, sp[0]])
            axes[1][2].plot(cl[:,0], cl[:,1], color="lime", lw=2.5, zorder=2)
            axes[1][2].scatter(sp[:,0], sp[:,1],
                               c="yellow", s=35, zorder=3)
        # Zoom col 3: overlay — green ON TOP
        show_ct(axes[1][3], ct[y1:y2, x1:x2], extent=ext)
        axes[1][3].imshow(orange[y1:y2, x1:x2], origin="upper", extent=ext)
        axes[1][3].scatter(bd_x, bd_y, c="red", s=20, zorder=1, alpha=0.95) # red BEHIND
        for sp in scaled:
            cl = np.vstack([sp, sp[0]])
            axes[1][3].plot(cl[:,0], cl[:,1],
                            color="lime", lw=2.5, zorder=10, alpha=0.95)     # green ON TOP
        axes[1][3].set_xlim(x1, x2); axes[1][3].set_ylim(y2, y1)

        # Zoom col 4: PURE ERROR MAP — black bg, pixel grid, hard edges
        axes[1][4].imshow(
            error_map[y1:y2, x1:x2],
            origin="upper", extent=ext,
            interpolation="nearest"      # hard pixel squares — no blur
        )
        # Pixel grid lines so individual voxels are visible
        for xg in np.arange(x1, x2+1):
            axes[1][4].axvline(xg - 0.5, color="#2a2a2a", lw=0.4, zorder=0)
        for yg in np.arange(y1, y2+1):
            axes[1][4].axhline(yg - 0.5, color="#2a2a2a", lw=0.4, zorder=0)

        # ── Stat box ──────────────────────────────────────────────────────
        stat = (
            f"Patient {patient_id}  |  Slice z={z}  |  "
            f"fillPoly: {fill_area}px  |  XML area: {xml_area:.1f}px²  |  "
            f"Area error: {err_pct:.1f}%\n"
            f"✓ Correct: {n_correct}px    "
            f"▲ Over-seg (orange): {n_over}px    "
            f"▼ Under-seg (cyan): {n_under}px    "
            f"Total error pixels: {n_over + n_under}"
        )
        fig.text(0.5, 0.895, stat, ha="center", color="#ffcc44",
                 fontsize=9.5, fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4",
                           fc="#1a1a2e", ec="#ffcc44", alpha=0.9))

        fig.suptitle(
            f"fillPoly vs XML Subpixel Annotation  —  "
            f"Patient {patient_id}  |  Slice z={z}",
            color="white", fontsize=13, fontweight="bold", y=0.975
        )

        legend_els = [
            plt.Line2D([0],[0], color="lime", lw=2,
                       label="Green = XML subpixel outline"),
            plt.scatter([],[],c="red",    s=40, label="Red = boundary uncertainty"),
            plt.scatter([],[],c="yellow", s=40, label="Yellow = correct pixels"),
            plt.scatter([],[],c="orange", s=40, label="Orange = over-seg (FP)"),
            plt.scatter([],[],c="cyan",   s=40, label="Cyan = under-seg (FN)"),
        ]
        fig.legend(handles=legend_els, loc="lower center", ncol=5,
                   fontsize=9, facecolor="#1a1a2e",
                   labelcolor="white", edgecolor="#555",
                   bbox_to_anchor=(0.5, 0.0))

        out = OUT_DIR / f"v3_P{patient_id}_z{z:03d}.png"
        plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0d0d1a")
        print(f"  → {out}")
        print(f"     correct={n_correct}  over={n_over}  under={n_under}  "
              f"err={err_pct:.1f}%")
        #plt.show()
        plt.close()


# ── Run ───────────────────────────────────────────────────────────────────────
for pid, sid in RUNS:
    make_comparison(pid, sid, max_slices=2)