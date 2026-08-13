# PrediCT Studio — build brief (v2)

Supersedes the earlier version of this file. Two instructions in v1 were wrong
and have been reversed — see "Corrections" at the end so you don't reapply
them from memory or from an older copy.

You are extending a working pipeline. **Read `Principles.md` first and follow
it.** The two rules that matter most here:

- **No abstraction until the third time.** Write the boring, flat version.
- **A wrong number that looks reasonable is the worst possible failure.**
  Never invent a default; never swallow an exception.

The backend pipeline is finished and correct. Do not redesign it, do not
reorder the pipeline stages, do not "improve" `scoring.py` or `pipeline.py`
beyond the changes specified here.

Repo root: `/pscratch/sd/s/soham95/predict_software/Predict-Studio`
(a folder inside the `predict_software` repo, branch `predict_software`).

Work in four steps, in order. **Stop at the end of each step and report.**
Do not begin a step until the previous step's verification has passed.

---

## Step 0 — Fix model naming, and produce coverage output

Two blockers found in the audit. Neither is optional; both must clear before
any UI work.

### 0a. Rename the A3 model so the checkpoint is identifiable

`models/a3-coverage/manifest.yaml` declares `val_dice: 0.72265`, which is the
**v2** checkpoint. But the directory and the `id` field both say
`a3-coverage`, so every `run.json` this model produces records
`"model_id": "a3-coverage"` — which cannot later be resolved to a specific
checkpoint.

This is the exact failure `Principles.md` documents: an earlier scorer loaded
the v1 checkpoint (val Dice 0.6156) while the evaluator loaded v2 (0.7227), and
nothing errored. Do not let it back in through naming.

- `git mv models/a3-coverage models/a3-coverage-v2`
- in the manifest, set `id: a3-coverage-v2`
- add to **both** manifests a line recording which training run produced the
  checkpoint, e.g. `source_run: runs/approach3_coverage_v2`
- delete the stale `data/out/172/a3-coverage/` folder — it was produced under
  the old name and by older code

Do this before generating any output, so no folder is ever written under the
ambiguous name.

### 0b. There is no coverage-model PNG output anywhere in the repo

`data/out/172/a3-coverage/` contains `run.json` and `lesions.csv` but no
`slices/`. **Every rendered mask currently in the repo came from a binary
model**, so every alpha channel is `{0, 255}`.

Soft coverage overlay is the entire point of this screen. Building the viewport
against binary output would mean the case that matters is never tested.

First, establish *why* rendering was skipped: compare the date in
`data/out/172/a3-coverage/run.json` against commit `cdfe28b` ("migrate
render.py to PIL"). If the run predates that commit, it is benign. If it does
not, there is a silent failure in `run.py` — find it and report it before
proceeding.

Then re-run: `python -m src.run --study 172 --model a3-coverage-v2`

### Step 0 verification

Report all four:

1. `data/out/172/a3-coverage-v2/slices/mask/` exists and has one PNG per slice.
2. **Alpha range** — must be many distinct values, not two:
   ```python
   from PIL import Image; import numpy as np
   a = np.array(Image.open(".../slices/mask/slice_022.png"))[:, :, 3]
   print(np.unique(a).size, a.min(), a.max())
   ```
3. **File size, as an independent check.** The binary a1-roi masks are ~1.4 KB
   each (44 files, 60 KB total) because a near-all-zero alpha channel
   compresses hard. Coverage masks should be **substantially larger** — varying
   alpha does not compress well. If they come out at ~1.4 KB each, something
   thresholded them. Report the total size of the folder either way.
4. `lesions.csv` for the coverage run has non-integer `area_mm2` values that are
   *not* multiples of the pixel area (0.1369 mm²) — confirming area is a sum of
   probabilities, not a voxel count.

---

## Step 1 — Close the output contract

The UI reads the output folder. Five small gaps. Patches to existing files
only; no new files.

### 1a. `scoring.py` — add mean coverage

One field in the row dict:

```python
mean_coverage=float(prob[z][ys, xs].mean()),
```

The UI shows this per lesion and cannot derive it from a PNG or the CSV.
Nothing else in `scoring.py` changes.

### 1b. `render.py` — KEEP the vertical flip, and document why

**Do not remove `np.flipud`.** An earlier draft of this brief said to remove
it. That was wrong, and the audit settled it:

`GetDirection()` is `diag(-1, -1, 1)`, which in ITK's LPS convention means
+x→Right, +y→Anterior, +z→Superior. Increasing array row is therefore
increasing *anterior*. Rendering row 0 at the top would show every slice
posterior-up — upside down. `flipud` produces the radiological view.

Keep both `flipud` calls and replace the existing comment with:

```python
# Direction cosines are diag(-1,-1,1): increasing array row is increasing
# anterior. Rendering row 0 at the top would show the slice posterior-up, so
# flipud gives the radiological view. This is DISPLAY ONLY — scoring.py reads
# the unflipped array and pred.nii.gz is saved unflipped, so no number depends
# on it. lesions.csv stays in array coordinates; the UI applies the same flip
# when it draws overlays.
```

### 1c. `render.py` — per-slice coverage histogram

Add to each entry in `slices.json`:

```python
# Bins for the COVERAGE ON THIS SLICE panel. Lower edge is the component
# threshold: voxels below it are not part of any lesion.
COVERAGE_BINS = [0.1, 0.25, 0.5, 0.75, 1.01]
...
"coverage_hist": [int(n) for n in
                  np.histogram(p_slice[p_slice > threshold], bins=COVERAGE_BINS)[0]],
```

### 1d. `render.py` — no default for `threshold`

Change `manifest.get("threshold", 0.5)` to `manifest["threshold"]`.
`threshold` is in `registry.REQUIRED`, so it is always present. The default is
dead code that violates Principle 6 and would silently mis-bin the histogram if
a manifest ever lost the field.

### 1e. `run.py` — add shape and output type to `run.json`

```python
"shape": list(array.shape),   # (nz, ny, nx). Volumes are heart-cropped, so
                              # every study differs — the UI must not assume
                              # 512x512. Note the UI caption prints nx x ny.
"output": m["output"],        # binary | coverage — the UI renders and labels
                              # itself differently for each
```

### 1f. Stale files in `slices/` — check and clear

`data/out/172/a1-roi/slices/` reports ~16 MB, but `ct/` (1.7 MB) plus `mask/`
(60 KB) is only ~1.8 MB. Around 14 MB is unaccounted for — most likely loose
matplotlib PNGs left directly in `slices/` from before commit `cdfe28b`.

`ls data/out/172/a1-roi/slices/`. Report what is there. If there are loose
files alongside `ct/` and `mask/`, delete them — the UI could otherwise pick up
stale images.

### 1g. `Readme.md` — one stale line

It says `render.py  # Matplotlib PNG slice overlays`. `render.py` uses PIL and
imports no matplotlib. Fix the line. Stale docs are what produced the
`[100, 1000]` scorer bug; do not leave a known-wrong one in place.

### Step 1 verification — mandatory

Re-run patient 172 with `a1-roi`. Nothing in this step touches scoring math,
and `scoring.py` reads the unflipped array regardless of what `render.py` does,
so the total must be **bit-identical**:

```
agatston_total  == 1064.1237274277692
n_lesions       == 20
n_lesions_all   == 25
```

Baseline for comparison: `data/out/172/a1-roi/run.json` (copy it aside before
you re-run). If any of the three moves at all, **stop and report** — do not
assume the new run is right. Diff the per-lesion rows and find the first row
that disagrees.

---

## Step 2 — The Instrument screen

Read `INSTRUMENT_SPEC.md`. It is the authoritative UI spec.

**Do not read the `.dc.html` files or `support.js` in
`docs/Explanations/Six design directions explored/`.** They use a custom
templating format that is not part of this project. Everything you need is in
the spec.

Also read `Archives/ui/index.html` and `Archives/backend/server.py` before
starting — an earlier attempt exists and its endpoint shapes are probably
already worked out. **Read them; do not copy them** (Principle 9: never copy a
file to modify it).

### Files — exactly three

```
ui/index.html      structure only, no inline styles beyond the tokens block
ui/app.css         all styling
ui/app.js          one state object, one render()
```

Not one file (900 lines of mixed HTML/CSS/JS fails the reader test), not five.

### Architecture

Plain HTML/CSS/JS. **No React, no Next.js, no build step, no npm.**

```js
const state = { studyId, modelId, run, slices, lesions,
                slice, view, selected, status };

function render() { /* writes the whole screen from state */ }
```

Every event handler mutates `state` then calls `render()`. There is no other
update path. `render()` must be idempotent.

### Data source

Fetch these URLs, and only these:

```
/data/out/<study>/<model>/run.json
/data/out/<study>/<model>/slices.json
/data/out/<study>/<model>/lesions.csv
/data/out/<study>/<model>/slices/ct/slice_NNN.png
/data/out/<study>/<model>/slices/mask/slice_NNN.png
```

Use these exact paths with **no `API_BASE` constant and no configuration.**
During development `python -m http.server` from the repo root serves them from
disk; in Step 3 FastAPI mounts `data/` at the same path. Identical URLs in both
modes, nothing to switch.

Read the study and model from the query string —
`?study=172&model=a3-coverage-v2` — with a documented fallback, rather than a
hardcoded constant you then have to hunt down in Step 3.

Parse `lesions.csv` with a ~15-line splitter. Do not add PapaParse.

### The vertical flip — exactly one place in the UI

`lesions.csv` is in array coordinates; the PNGs are flipped. Convert in one
named function and route every lesion y through it:

```js
// lesions.csv is in array coordinates (y=0 posterior); render.py flips
// vertically so anterior lands at the top. This is the only place that
// conversion happens.
const toDisplayY = (yArray) => state.run.shape[1] - 1 - yArray;
```

x needs no conversion.

### Colour — two tokens, deliberately

```css
--accent:     #C98B2E;   /* chrome that refers to predictions: track bars,
                            tier bar, selection, chip highlights */
--prediction: #C9541F;   /* the mask itself. MUST match render.py RGB
                            (201, 84, 31) — it is baked into the PNG and
                            cannot be changed in the browser. */
```

Any legend swatch standing for the mask uses `--prediction`, so the legend
never lies about what is on the image. Do not change `render.py`'s RGB values.

### Build order — six passes

Each pass must run and be demoable before the next. Commit after each.

1. Viewport: black pane, CT `<img>`, wheel steps slice, page never scrolls
2. Mask `<img>` layered on top + the three view chips
3. Volume track strip
4. Right rail: lesion table, click to select, selected-lesion panel, selection
   ring on the canvas
5. Provenance strip + footer total/tier bar
6. Coverage bands, excluded summary, keyboard shortcuts, export

**Pass 5 is never cut.** A screen showing a score with no provenance is exactly
the failure this project exists to avoid. Pass 6 is the first thing to cut if
time runs short.

### Image handling

Two stacked `<img>` elements plus a small transparent `<canvas>` for the
selection ring — **not** a canvas re-implementation of image display. The mask
PNG is already RGBA with alpha equal to the coverage fraction; the browser
composites it for free. Never threshold it client-side, never pre-composite.

Preload the adjacent 3 slices in each direction so wheel scrolling does not
flicker. Nothing more elaborate.

### Verification — use the coverage study, not the binary one

1. Wheel through every slice: no flicker, no page scroll, page never scrolls
2. On `a3-coverage-v2`, the overlay must read as **grain** — soft, varying
   opacity — not a hard-edged blob. This is the visible form of the alpha range
   from Step 0 and is the scientific point of the screen.
3. Select the lesion at `slice_idx=5, lesion_id=2` in the 172 a1-roi run
   (`centroid_x=206.5, centroid_y=188.7`, image 403×358). The ring must land on
   the calcium. If it lands mirrored top-to-bottom, `toDisplayY` is not being
   applied.
4. Zero state: still steppable through all slices. Dense state: rail scrolls,
   page does not.

---

## Step 3 — `server.py`

Only after Step 2 works against static files.

One file, FastAPI, **no science in it.** It uploads, launches jobs, reads
folders. If you find yourself importing numpy into `server.py`, stop.

```
POST /studies          upload DICOM -> study_id_from_series() -> data/uploads/<id>
GET  /studies          list data/uploads
GET  /models           registry.list_models()
POST /jobs             {study_id, model_id} -> background thread calling
                       run(..., progress=cb) -> job_id
GET  /jobs/{id}        {stage, pct, status, error}
```

Plus `StaticFiles` mounts: `data/` at `/data`, `ui/` at `/`.

Job state is a plain in-memory dict. **No Celery, no Redis, no database.** One
user, one machine.

`run()` already accepts `progress=None`, called as `progress(stage, pct)`. Wire
the callback into the job dict; do not change `run()`'s signature.

If `run()` raises, store the traceback in the job dict and return it from
`GET /jobs/{id}`. Do not swallow it — a job that silently reports "done" with
no output is the same class of failure as a wrong number.

Note: `study_id_from_series()` in `paths.py` uses
`next(Path(dicom_dir).rglob("*.dcm"))`, which raises `StopIteration` on an
upload with no `.dcm` extension. Catch that at the endpoint and return a clear
400 — many DICOM exports have no file extension at all. Report this rather than
changing `paths.py` unilaterally.

---

## Open question for the human — do not decide this yourself

**Left/right convention.** With +x→Right and PNG column 0 at image left,
patient *left* currently renders on the viewer's left (neurological
convention). CT is conventionally read radiologically — patient right on the
viewer's left. No score depends on it, but a clinician notices immediately.

If it is changed, `fliplr` goes next to `flipud` in `render.py` and the x
transform joins `toDisplayY` in the same UI function — one place, both axes.
**Flag it and wait for an answer. Do not change it on your own judgement.**

---

## What NOT to build

Not in scope. Do not add speculatively:

- A/B compare view (A1 vs A3 side by side) — second screen, later
- Session or history view; cohort view; MPR panes
- User accounts, auth, multi-tenancy
- A database
- Any plugin or adapter layer

---

## Corrections from v1 of this brief

If you have seen an earlier copy of this file, two instructions changed:

| v1 said | v2 says | why |
|---|---|---|
| Remove `np.flipud` from `render.py` | **Keep it**; fix the coordinate mismatch in the UI via `toDisplayY` | Direction cosines `diag(-1,-1,1)` make the flip necessary for a radiological view. v1 was reasoning without the geometry. |
| Change `render.py`'s mask RGB to match `--accent` | **Do not change it**; add a separate `--prediction` token | Ember is the prediction colour by design; amber is chrome. Two deliberate tones, not a drift. |