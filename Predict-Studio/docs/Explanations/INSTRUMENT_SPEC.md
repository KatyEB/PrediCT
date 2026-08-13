# PrediCT Studio — Instrument screen specification

Extracted from the "02 Instrument" direction of the design prototypes. This file
is the authoritative UI spec. **Do not read the `.dc.html` prototypes or
`support.js`** — they use a custom templating format that is not part of this
project and must not be reproduced.

One study, one model, one screen. The page never scrolls; only the right rail
body scrolls.

---

## 1. Design tokens

```
--bg            #23272B    page background
--viewport      #000000    image pane
--image-bg      #0B0D0F    behind the CT image itself
--ink           #E4DED0    primary text
--ink-dim       #B7AF9E    provenance text
--muted         #8B939B    labels, captions, legends
--muted-2       #6E7C86    withheld / inactive
--rule          #3A4148    major borders (between regions)
--rule-2        #31373D    minor borders (table rows, bars)
--accent        #C98B2E    UI chrome for predictions: bars, tier, selection
--truth         #4FA8C5    ground truth (unused on this screen; reserved)
```

Fonts (Google Fonts):

- `Space Grotesk` 400/500/700 — headings and the few large numbers
- `IBM Plex Mono` 400/500/600 — **all numbers, labels, captions, tables.**
  Anything a reader might compare against another number is mono.

Type scale actually used: 9.5, 10, 10.5, 11, 11.5, 13, 19, 26 px. Do not invent
sizes outside this set.

Letter-spacing: `0.12em`–`0.16em` on uppercase mono labels; `-0.02em` on the
large numbers (19px SLICE, 26px TOTAL).

### Colour note — resolve before writing code

The mask overlay PNG is painted by `render.py` as RGB (201, 84, 31) = `#C9541F`,
while `--accent` here is `#C98B2E`. The PNG colour cannot be changed in the
browser. Either accept two related warm tones (chrome amber, prediction ember)
or change `render.py`. **Ask before assuming.**

---

## 2. Layout

Four horizontal bands, top to bottom, in a `height:100vh` flex column:

```
┌──────────────────────────────────────────────────────────────────┐
│ A  PROVENANCE STRIP                              (fixed height)  │
├──────────────────────────────────────────────────────────────────┤
│ B  VOLUME TRACK                                  (fixed height)  │
├───────────────────────────────────┬──────────────────────────────┤
│                                   │  D  RIGHT RAIL      340px    │
│  C  VIEWPORT           1fr        │   ├ header   (fixed)         │
│                                   │   ├ body     (scrolls)       │
│                                   │   └ footer   (fixed)         │
└───────────────────────────────────┴──────────────────────────────┘
```

Band C+D is a CSS grid `grid-template-columns: 1fr 340px` with
`flex: 1; min-height: 0`. The `min-height: 0` is required — without it the grid
row refuses to shrink and the page scrolls.

### A — Provenance strip

`padding: 7px 18px`, `border-bottom: 1px solid --rule`, mono 10.5px,
`color: --ink-dim`, `display:flex; gap:18px; flex-wrap:wrap`.

Three spans, third pushed right with `margin-left:auto`. Content comes verbatim
from `run.json` — never hardcoded:

1. `{model_id} · {output} · component threshold {threshold} (delineation only — area = Σ coverage)`
2. `model HU window {hu_window} · display window −100…400 (W 500 / L 150) · {spacing} · RAS`
3. `crop heart +8 mm, TotalSegmentator {locator_version} · ckpt {sha256[:12]} · min lesion 1.0 mm² · {date}`

For a binary model, line 1 reads `binary · threshold 0.50 · area = voxel count`.
The strip must tell the truth about whichever model ran.

### B — Volume track

`padding: 10px 18px 8px`, `border-bottom: 1px solid --rule`, flex column,
`gap: 6px`. Three rows:

1. Label row — mono 10px `--muted`, `letter-spacing: 0.12em`, space-between:
   left `VOLUME · SLICE {n}`, right `{n} of {N} slices carry scored calcium · {pct} % · z {z0}–{z1} mm`
2. The track — `display:flex; gap:1px; align-items:flex-end; height:46px`.
   One `<button>` per slice, `flex:1; min-width:0; border:0; padding:0`.
   - height: proportional to that slice's `slice_score`, min 4px so empty
     slices are still clickable
   - background: `--accent` if `has_calcium`, else `--rule-2`
   - `border-bottom: 3px solid` — `--accent` if scored, `--muted-2` if the
     slice has lesions but all withheld, transparent otherwise
   - the cursor slice gets a visible marker (1px `--ink` top line)
   - `title` attribute: `slice {idx} · z {z_mm} mm · {slice_score}`
   - click sets cursor slice
3. Legend row — mono 10px, space-between:
   `█ carries scored calcium` (accent) · `▌ sub-minimum, recorded not counted`
   (muted-2) · `▲ cursor slice {n}` · `all {N} slices reachable`

### C — Viewport

`background: #000`, flex centred, `position: relative`, `min-height: 0`,
`overflow: hidden`. Wheel anywhere in this pane steps the slice.

Inner image box: square, `background: --image-bg`, sized to fit the pane
(`max-width: 100%; max-height: 100%`), `position: relative`. Inside it, stacked
with `position:absolute; inset:0`:

1. `<img>` CT slice — `image-rendering: pixelated`
2. `<img>` mask slice — same box, `image-rendering: pixelated`. Hidden in
   view 1. **Never pre-composite and never threshold** — the alpha channel is
   the coverage fraction and the browser blends it for free.
3. `<canvas>` — selection ring only. Nothing else is drawn here.

Bottom-left caption, `left:16px; bottom:12px`, mono 10px `--muted`,
`line-height:1.7`, three lines:

```
{nx} × {ny} · 0.37 mm in-plane · ├── 10 mm ──┤
display W 500 / L 150 (−100…400 HU) · model saw {hu_window} HU
{overlay note for the current view}
```

Dimensions come from `run.json.shape`, **not hardcoded 512** — volumes are
heart-cropped and every study differs. The 10 mm scale bar is
`10 / spacing_x` image-pixels wide, scaled by the current display zoom.

Top-right: three view chips, mono 10px, `padding: 4px 8px`, 1px border.
Selected chip: `background --accent; color #23272B`.

| chip | shows | overlay note |
|---|---|---|
| `1 original` | CT only | `original stack · prediction not drawn here` |
| `2 prediction` | CT + mask | `coverage overlay · alpha = fraction · never thresholded` |
| `3 calcium only` | CT + mask, stack restricted to slices where `has_calcium` | `restricted stack · {n} scored slices` |

View 3 changes what the track and the wheel index over — that is the point of
it. When it is active, the legend line becomes
`{N-n} of {N} slices unreachable in this stack`.

### D — Right rail

`border-left: 1px solid --rule`, flex column, `min-height: 0`.

**Header** (fixed) — `padding: 12px 16px`, `border-bottom: 1px solid --rule`,
gap 10px:

- Row: `SLICE {n}` (Space Grotesk 700, 19px) left, `z {z} mm` (mono 11px,
  `--ink-dim`) right, `align-items: baseline`
- Lesion chips — one per lesion on this slice, mono 10.5px, `padding: 4px 8px`,
  1px border, click selects. Withheld lesions use `--muted-2` text and a dashed
  border. Empty state: `no lesion on this slice` in `--muted-2`.

**Body** (scrolls) — `flex:1; min-height:0; overflow-y:auto`,
`padding: 12px 16px`, `gap: 14px`. Four sections, each headed by a mono 10px
uppercase `--muted` label with `letter-spacing: 0.14em`:

1. `SELECTED LESION` — key/value rows, mono 11.5px, space-between,
   `border-bottom: 1px solid --rule-2`, `padding-bottom: 4px`:
   lesion (with "n of m here"), area mm², peak HU, density weight,
   mean coverage (rendered in `--accent` when < 0.5), score (or `withheld`).
2. `COVERAGE ON THIS SLICE` — four horizontal bars from the per-slice histogram
   in `slices.json`. Each row: an 84×8px `--rule-2` track with an `--accent`
   fill, then a mono 10.5px label `{lo}–{hi} · {n} voxels`. Below the bars, the
   standing note:
   > Coverage is not binary. A voxel at 0.35 contributes 0.35 of its area and is
   > drawn as grain, never as an edge. The 0.10 threshold only decides where one
   > component stops and the next begins — it does not gate the score.

   Hide this whole section for a binary model; it is meaningless there.
3. `LESIONS · CLICK TO SELECT` — table, mono 10.5px, `border-collapse: collapse`,
   rows `text-align:right` with the tag column left. Columns:
   tag · `sl {n}` · area · `p {mean_coverage}` · score. Row borders `--rule-2`.
   Selected row inverts (background `--accent`, text `#23272B`).
   Withheld rows are `--muted-2` and show `—` for score.
4. `EXCLUDED · {n}` — one summary line, mono 10.5px `--muted-2`:
   `{n} withheld · {area} mm² · below 1.0 mm² · would add {score} if admitted`.
   These lesions exist in `lesions.csv` with `included=false`; they are shown
   because hiding them would make the total unauditable.

**Footer** (fixed) — `border-top: 1px solid --rule`, `padding: 10px 16px`,
gap 5px:

- Row, `align-items: baseline`, gap 10px: `TOTAL` (mono 10px `--muted`,
  `letter-spacing: 0.12em`) · the number (Space Grotesk 700, 26px) · the tier
  (700, 13px, `letter-spacing: 0.06em`) coloured `--accent` for severe,
  `--muted` otherwise
- Tier bar — 7px tall, `--rule-2`, with an `--accent` fill at
  `total / SCALE_MAX`. `SCALE_MAX` is a declared constant (use 1400; state it in
  a comment) — tick marks at the 100 and 400 positions, 1px `--muted`,
  `top:-3px; bottom:-3px`
- Note line, mono 9.5px `--muted`: `bounds 0 / 1–100 / 101–400 / >400` — or when
  over 400, `tier >400 · {x} above the bound` · `E export · ? keys`

---

## 3. Required states

The prototype was built with four states and all four must work:

| state | condition | behaviour |
|---|---|---|
| **result** | normal | as specified above |
| **running** | job in flight | viewport shows the CT (originals are already scrollable) with a sweeping highlight; rail shows the stage line `step {stage} · elapsed {s} s`; no total, no tier — **the screen must never show a score that does not exist yet** |
| **zero** | no lesion cleared 1.0 mm² | track empty but all slices still steppable; tier `ZERO`; body explains: `No component reached 1.0 mm². {N} slices were predicted and scored; the largest connected coverage component measured {x} mm², below the minimum, and is recorded as withheld.` |
| **dense** | very high burden | track and tier bar must not overflow; the lesion table must scroll, not stretch the page |

---

## 4. Interaction

- Wheel over the viewport **or** the track: step one slice. `preventDefault`.
- Arrow up/down: step one slice. Left/right: previous/next scored slice.
- `1` `2` `3`: view chips. `E`: export. `?`: key list.
- Click a track bar, a lesion chip, or a table row: select. Selecting a lesion
  on a different slice moves the cursor to that slice.
- `Escape`: clear selection.
- `body { overflow: hidden }`. The page never scrolls.
