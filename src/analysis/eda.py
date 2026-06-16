# eda.py  — Run after COCA_processor_main.py finishes
# python eda.py

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"C:\SOHAM\coca_raw\cocacoronarycalciumandchestcts-2")
CSV_PATH  = DATA_ROOT / "data_canonical" / "tables" / "scan_index.csv"
IMG_ROOT  = DATA_ROOT / "data_canonical" / "images"
OUT_DIR   = Path(r"C:\SOHAM\figures")
OUT_DIR.mkdir(exist_ok=True)


# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} scans from CSV")

pos = df[df["has_calcium"] == True]
neg = df[df["has_calcium"] == False]

print(f"\n=== DATASET SUMMARY ===")
print(f"Total scans      : {len(df)}")
print(f"Unique patients  : {df['patient_id'].nunique()}")
print(f"Positive (CAC>0) : {len(pos)}  ({100*len(pos)/len(df):.1f}%)")
print(f"Negative (CAC=0) : {len(neg)}  ({100*len(neg)/len(df):.1f}%)")
print(f"\n=== CALCIUM BURDEN (positive patients only) ===")
print(pos["voxels"].describe().round(1))
print(f"\n=== SLICES WITH CALCIUM ===")
print(pos["num_slices"].describe().round(1))


# ── Load spacing from meta.json files ────────────────────────────────────────
spacings = []
for scan_id in df["scan_id"]:
    meta_path = IMG_ROOT / scan_id / f"{scan_id}_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        spacings.append(meta["original_spacing"][:2])   # x, y only

spacings = np.array(spacings)
print(f"\n=== ORIGINAL X/Y SPACING ===")
print(f"Mean  : {spacings.mean(axis=0).round(4)}")
print(f"Min   : {spacings.min(axis=0).round(4)}")
print(f"Max   : {spacings.max(axis=0).round(4)}")


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14), facecolor="#0d0d1a")
gs  = gridspec.GridSpec(2, 3, figure=fig,
                         hspace=0.38, wspace=0.32,
                         top=0.91, bottom=0.07,
                         left=0.07, right=0.96)

DARK  = "#0d0d1a"
GRID  = "#2a2a3a"
TEXT  = "white"
POS_C = "#f5a623"    # orange — positive patients
NEG_C = "#4a90d9"    # blue   — negative patients
ACC_C = "#43d692"    # green  — accent

def style_ax(ax, title):
    ax.set_facecolor("#12122a")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)


# ── Panel 1: Positive vs Negative bar ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(["Positive\n(CAC > 0)", "Negative\n(CAC = 0)"],
               [len(pos), len(neg)],
               color=[POS_C, NEG_C], width=0.5, edgecolor="none")
for bar, val in zip(bars, [len(pos), len(neg)]):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 3,
             f"{val}\n({100*val/len(df):.1f}%)",
             ha="center", va="bottom", color=TEXT, fontsize=10)
style_ax(ax1, "① Class Distribution")
ax1.set_ylabel("Number of Scans")
ax1.set_ylim(0, max(len(pos), len(neg)) * 1.18)


# ── Panel 2: Voxel distribution histogram (log x) ────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
voxels = pos["voxels"].values
ax2.hist(voxels, bins=40, color=POS_C, edgecolor="none", alpha=0.85)
ax2.axvline(np.median(voxels), color=ACC_C, lw=2,
            linestyle="--", label=f"Median: {np.median(voxels):.0f}")
ax2.axvline(np.mean(voxels),   color="white", lw=1.5,
            linestyle=":",  label=f"Mean:   {np.mean(voxels):.0f}")
ax2.set_xscale("log")
ax2.set_xlabel("Calcium Voxels (log scale)")
ax2.set_ylabel("Number of Patients")
ax2.legend(fontsize=8, facecolor=DARK, labelcolor=TEXT, edgecolor=GRID)
style_ax(ax2, "② Calcium Burden Distribution")


# ── Panel 3: Slices with calcium histogram ───────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
slices = pos["num_slices"].values
ax3.hist(slices, bins=range(1, slices.max()+2), color=POS_C,
         edgecolor=DARK, alpha=0.85, align="left")
ax3.axvline(np.median(slices), color=ACC_C, lw=2,
            linestyle="--", label=f"Median: {np.median(slices):.0f}")
ax3.set_xlabel("Number of Slices With Calcium")
ax3.set_ylabel("Number of Patients")
ax3.legend(fontsize=8, facecolor=DARK, labelcolor=TEXT, edgecolor=GRID)
style_ax(ax3, "③ Calcium Slice Spread")


# ── Panel 4: Voxels vs Slices scatter ────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(pos["num_slices"], pos["voxels"],
            color=POS_C, alpha=0.5, s=18, edgecolors="none")
ax4.set_xlabel("Slices With Calcium")
ax4.set_ylabel("Total Calcium Voxels")
ax4.set_yscale("log")
style_ax(ax4, "④ Voxels vs Slices (positive patients)")


# ── Panel 5: Original spacing distribution ───────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
if len(spacings) > 0:
    ax5.hist(spacings[:, 0], bins=20, color=ACC_C,
             edgecolor="none", alpha=0.85, label="x spacing")
    ax5.hist(spacings[:, 1], bins=20, color=NEG_C,
             edgecolor="none", alpha=0.5,  label="y spacing")
    ax5.axvline(0.37, color="white", lw=2, linestyle="--",
                label="Target: 0.37mm")
    ax5.set_xlabel("Original Pixel Spacing (mm)")
    ax5.set_ylabel("Number of Scans")
    ax5.legend(fontsize=8, facecolor=DARK, labelcolor=TEXT, edgecolor=GRID)
style_ax(ax5, "⑤ Original Spacing Distribution")


# ── Panel 6: Voxel percentile table ──────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
style_ax(ax6, "⑥ Key Statistics")

pcts  = [10, 25, 50, 75, 90, 95, 99]
vals  = np.percentile(pos["voxels"], pcts).astype(int)
rows  = [[f"p{p}", f"{v:,} vox"] for p, v in zip(pcts, vals)]
rows += [["", ""],
         ["Positive", f"{len(pos)} scans"],
         ["Negative", f"{len(neg)} scans"],
         ["Ratio",    f"{len(pos)/len(neg):.2f}:1"]]

table = ax6.table(cellText=rows,
                   colLabels=["Metric", "Value"],
                   cellLoc="center", loc="center",
                   bbox=[0.05, 0.05, 0.90, 0.85])
table.auto_set_font_size(False)
table.set_fontsize(10)
for (r, c), cell in table.get_celld().items():
    cell.set_facecolor("#1a1a3a" if r > 0 else "#2a2a5a")
    cell.set_text_props(color=TEXT)
    cell.set_edgecolor(GRID)


# ── Title + save ─────────────────────────────────────────────────────────────
fig.suptitle(
    f"COCA Dataset EDA  —  {len(df)} Total Scans  |  "
    f"{len(pos)} Positive  |  {len(neg)} Negative",
    color=TEXT, fontsize=14, fontweight="bold", y=0.96
)

out = OUT_DIR / "eda_full_dataset.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=DARK)
print(f"\n✓ EDA figure saved → {out}")
plt.show()