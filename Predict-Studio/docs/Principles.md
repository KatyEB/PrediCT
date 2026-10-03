# PrediCT — Code Design & Principles

**Read this before adding a file.** How the system fits together is in
[`SYSTEM.md`](SYSTEM.md); this document is how code in it is written.

This project computes coronary artery calcium (CAC) Agatston scores from cardiac
CT. The numbers it produces are compared against radiologist ground truth and
reported in a GSoC final submission. That has one consequence that drives every
rule below:

> **A wrong number that looks reasonable is the worst possible failure.**

A crash is cheap — you see it and fix it. A score of 340 that should have been
190 is invisible, gets written into a report, and survives for weeks. This
project has already paid for that twice:

- Both Agatston scorers normalised at HU `[100, 1000]` while every model was
  trained at `[0, 1200]`. Nothing errored. The scores were simply wrong.
- `agatston_scoring_a3.py` loaded the v1 checkpoint (val Dice 0.6156) while the
  evaluator loaded v2 (0.7227). Nothing errored. A3 was represented by its
  weaker model in reported results.

Neither was a hard bug. Both were **silent assumptions**. Every principle here
exists to make that class of failure loud.

---

## 1. The reader test

> Could someone unfamiliar with this repo open any file, read the top 30 lines,
> and understand what it does and why?

If not, the header is wrong. Fix the header before adding code.

The relevant reader is a mentor or reviewer, not the author. Code that only the
author can follow cannot be defended in a review, and cannot be checked by
anyone else.

---

## 2. File headers

Every module starts with a docstring covering four things, in this order:

1. **What it does** — one sentence.
2. **Why it exists / what's subtle about it** — the domain knowledge.
3. **What it does NOT do** — the boundary.
4. **How to call it** — a usage line.

```python
"""
scoring.py — Agatston scoring from a probability volume.

Takes a model's output probability map plus the raw HU volume and produces one
row per detected lesion. Patient-level totals are a sum over those rows, never
a separate calculation.

The A1 vs A3 difference lives here and nowhere else:
    binary   (A1): lesion area = voxel count      x pixel_area
    coverage (A3): lesion area = sum(probabilities) x pixel_area  (never thresholded)
That single line is the scientific claim of this project.

Does NOT: load models, read files, or know about HU windows.
Called by: run.py, after predict().

Usage:
    rows    = score(prob, hu, spacing, mode="coverage", threshold=0.1)
    summary = totals(rows)
"""
```

The **"Does NOT"** line is the most valuable one — it is what stops a file
growing into something else six months later.

---

## 3. Comment the *why*, never the *what*

Delete any comment a reader could derive from the code itself.

```python
# BAD — adds nothing
pixel_area = sx * sy            # multiply sx by sy

# GOOD — states a fact that is not visible in the syntax
# Agatston is defined per 3 mm slice with no thickness term. Because we resample
# to exactly 3.0 mm that factor is 1.0 and is omitted. If spacing ever changes,
# every score silently scales by sz/3 — hence the assert.
assert abs(sz - 3.0) < 1e-6, f"Agatston assumes 3.0 mm slices, got {sz}"
```

**Rule of thumb:** if the comment would still be true in a different project,
it probably isn't worth writing.

### Every magic number cites its origin

```python
threshold = 0.1   # component delineation only; does NOT gate area. Unswept —
                  # see progress report open item 11.
margin_mm = 8     # matches the training crop (progress report §10.2). Changing
                  # this without retraining alters the input distribution.
```

"The progress report" here and below is
[`docs/progress_report.md` on the `soham_segmentation` branch](https://github.com/ML4Sci/PrediCT/blob/soham_segmentation/docs/progress_report.md),
where the models were trained and evaluated.

A number with no provenance is a number nobody can defend or safely change.

---

## 4. Docstrings state units and axis order

Most bugs in this project were shape or unit confusion, not logic errors. State
both, every time, even when it feels redundant.

```python
def score(prob, hu, spacing, mode, threshold, min_area_mm2=1.0):
    """Score every lesion in a probability volume.

    Args:
        prob:    (Z, Y, X) float32 in [0, 1]. Model output, already activated.
        hu:      (Z, Y, X) raw Hounsfield Units — NOT the normalised array.
                 Density weights are defined on true HU values.
        spacing: (sx, sy, sz) in mm — SimpleITK order, the REVERSE of the
                 array axis order above.
        mode:    "binary" | "coverage"

    Returns:
        list[dict], one row per lesion. Sub-threshold lesions are included and
        flagged included=False, so conformant and non-conformant totals both
        come from one inference run.
    """
```

### The axis-order rule

- **SimpleITK images** are `(x, y, z)` — `GetSpacing()`, `GetSize()`, indexing.
- **NumPy arrays** from `GetArrayFromImage()` are `(z, y, x)`.
- **MONAI models** here expect `(x, y, z)` — patch `(96, 96, 32)` requires the
  32 to land on the slice axis.

Transposes must be adjacent to the call that needs them, with a comment saying
why. A wrong transpose produces a plausible-looking volume, not an error.

---

## 5. Names carry units

```
area_mm2    not  area
z_mm        not  z_pos
margin_mm   not  margin
hu_window   not  window
n_voxels    not  n
spacing_mm  not  spacing        (where ambiguity is possible)
```

No exceptions. This is the cheapest bug prevention available.

---

## 6. Fail loudly, never silently

**The highest-value rule in this repo.**

Every silent fallback is a future wrong number. Three forms to avoid:

```python
# BAD — a missing package silently changes the science
try:
    from totalsegmentator...
except ImportError:
    return image                       # now running uncropped, invisibly

# BAD — a default that is right today and wrong for the next model
hu_window = manifest.get("hu_window", (0, 1200))

# BAD — swallowing the case you didn't think about
if mode == "binary":
    area = ...
else:
    area = ...                         # "coverage"? typo? who knows
```

```python
# GOOD
except ImportError:
    raise RuntimeError(
        "TotalSegmentator not installed but crop=True. Models were trained on "
        "heart-cropped volumes; uncropped inference produces invalid scores. "
        "Install it, or pass crop=False for plumbing tests only."
    )

hu_window = manifest["hu_window"]      # KeyError is the correct behaviour

if   mode == "binary":   area = ...
elif mode == "coverage": area = ...
else: raise ValueError(f"unknown mode: {mode}")
```

An exception costs you five minutes. A silent wrong number costs a reported
result.

---

## 7. Declared vs derived

A `.pth` file contains tensors. From them you can derive architecture — channel
counts, strides, parameter count. You **cannot** derive:

| Fact | Recoverable from the checkpoint? |
|---|---|
| HU window used in training | No |
| Voxel spacing | No |
| Whether input was heart-cropped | No |
| Output semantics — binary vs coverage | **No** |
| Activation, patch size, overlap, threshold | No |

A1 and A3 have byte-identical architecture. Only their **training labels**
differ. Auto-generating a manifest by inspecting weights would give both the
same config, score A3 as binary, and delete the entire contribution of
Approach 3 — with no error.

Therefore: **derive what is derivable, declare what is not.** Declared fields go
in `models/<id>/manifest.yaml`, written once by whoever trained the model, who
already knows them.

Every result folder carries a `run.json` recording what was actually used:
`model_id`, `sha256`, `hu_window`, `spacing`, `threshold`, `date`, and the crop
decision — `cropped`, the model's own `crop_default` (they differ when a run
overrode it) and `crop_margin_mm`.

> **"Which checkpoint and which window produced this number?" must be
> answerable from the output folder alone** — never from memory, never by
> reading code.

---

## 8. Size limits

| Limit | Meaning |
|---|---|
| File > 300 lines | Doing two things. Split it. |
| Function > 50 lines | Same. |
| Nesting > 3 levels | Extract the inner part. |
| Can't summarise a file in one sentence | Wrong contents. |

Roughly the amount a reader holds in their head at once. Not arbitrary.

**Known exceptions (debt, not precedent):** `server.py` (all HTTP routes in one
module — split into route modules when it next grows), `ui/app.js` and
`ui/app.css` (one script and one stylesheet for the whole UI — split per view
when a build step or a new view arrives). Do not add a new file over the limit
because these exist.

---

## 9. No abstraction until the third time

**The rule most likely to be violated, and the most damaging when it is.**

Do not add a base class, plugin system, config framework, DAG, or registry
object until you have hit the same problem *three separate times*. Twice is a
coincidence. Three times is a pattern.

An earlier version of the scoring layer wrapped a loop computing `area x weight`
in a `ScoringConfig` object, a `Spacing.from_sitk` constructor, a `mask_kind`
string parameter, and separate `included_lesions` / excluded collections — five
concepts for one multiplication. It was harder to read, harder to verify, and
the author could not trace a score through it. That is a defect in a project
whose output must be defended in review.

**Write the boring version.** A flat function with a clear name and an honest
docstring beats a well-designed hierarchy in a project with one developer and a
deadline. The boring version is easy to refactor later; the clever version is
not easy to un-abstract.

Corollary: **never copy a file to modify it.** This repo once held four copies
of `COCA_processor_main.py`. They drifted, and the drift is where the HU-window
and checkpoint defects came from. One module, imported by everything that needs
it, parameterised where it must differ.

---

## 10. Layering

Dependencies point one direction only:

```
server.py     HTTP only. No science. Sign-in, per-user access, uploads,
    |         launches jobs (as a subprocess), serves a user's own files.
    +-- accounts.py   users, passwords, sessions, access tokens (no HTTP)
    +-- ratelimit.py  token buckets                             (no HTTP, no users)
    +-- ingest.py     fix extensionless DICOM files
    |
run.py        Orchestration. Owns the order of stages and the output folder.
    |
    +-- pipeline.py / pipeline_nnunet.py   load / resample / crop / normalize / predict
    +-- scoring.py    lesions and totals        (no file I/O, no torch)
    +-- grouping.py   lesion 3D table grouping  (no file I/O, no torch)
    +-- render.py     PNG output                (display only)
    +-- mesh.py       3D surfaces               (display only, measures nothing)
    +-- registry.py   read manifests            (no torch)
    |
paths.py      Every path in the project. Nothing else builds paths by hand.
```

Rules:

- `scoring.py` imports no torch and touches no files. It is pure: arrays in,
  rows out. That makes it testable by hand and reviewable on its own.
- `registry.py` imports no torch, so `GET /models` is instant.
- Nothing below `run.py` knows about HTTP, jobs, users or the UI. `run.py` only
  learns *which account* a run belongs to, to choose folders — never to change
  what is computed.
- Paths come from `paths.py`. A hardcoded path anywhere else is a bug.
- The UI displays; it never computes a score. Every number it shows is read
  from a result file.

---

## 11. Data isolation (multi-user)

Each account's data lives in `data/users/<id>/` and nowhere else. The rule that
keeps users apart is structural, so it must hold in every new route:

- **The user comes from the session, never from the request.** Every API route
  takes `user = Depends(current_user)`. A user id, owner or folder in a URL,
  query string or body is never trusted.
- **Paths are built from that user's id** with the `paths.py` helpers
  (`raw_dir`, `work_dir`, `out_dir`, `tmp_dir`, `study_dirs`), so a link copied
  from someone else resolves inside your own folder and finds nothing.
- **Every name from a request passes `safe_name`** (one folder name — no
  separators, no `..`). A file path must still be inside its base folder after
  `resolve()`.
- **`data/` is never served statically.** Files go through `/files/...`, which
  applies the rules above.
- **Someone else's object does not exist for you:** answer `404`, not `403`, and
  never put another user's study names in a message.
- **Limits are explicit:** anything a stranger can repeat (sign-up, sign-in,
  upload) goes through a token bucket in `ratelimit.py`; anything that can fill
  a disk has a size cap checked before the body is read.

The reasoning and the full threat table are in [`SYSTEM.md` §5 and §13](SYSTEM.md).

---

## 12. Verify a refactor changed nothing

Before and after any restructuring, run the same patient through both versions
and compare the total Agatston score.

If they differ, **do not assume the new one is right.** Diff the per-lesion
rows, find the first row that disagrees, and explain it. One of the two versions
has a bug and you have just found it — that is a result, not a setback.

Refactors that "should be equivalent" are exactly where silent numeric changes
enter.

---

## 13. Settled decisions

These are fixed. Changing one requires stating the new evidence explicitly, not
a preference.

| Decision | Value | Basis |
|---|---|---|
| HU window | `[0, 1200]` | Clipped windows plateaued near Dice 0.25; this reached ~0.61 mean / 0.69 median. Progress report §3.3. |
| Target spacing | `0.37 x 0.37 x 3.0 mm` | 3.0 mm makes the Agatston thickness factor exactly 1.0. |
| Heart locator | TotalSegmentator v2.15.0, `--roi_subset heart`, 8 mm margin | Alternative failed on 2/5 test patients. Progress report §10.1. |
| Same locator at train and test | Required | Different crops between train and inference is a distribution-shift bug. |
| Crop per run | Default = the model's `manifest.yaml`; a run may override it | The override is recorded (`cropped` vs `crop_default` in `run.json`) and flagged in the UI. Cropped and full CTs are cached separately (`work/<study>/crop` vs `/full`), never mixed. |
| Mask storage format | `.nii.gz` float32 | PNG has no voxel spacing, so a PNG mask cannot be scored. |
| PNG role | Display only, regenerated from NIfTI | Never an input to anything. |
| Risk tiers | 4-tier in `run.json` (scoring.py); 6-tier is a UI view only | Tiers are labels on the score; changing the view never recomputes a number. |
| User data | `data/users/<id>/` only; ownership = folder location | One source of truth; see §11. |

> **Known documentation defect:** the progress report records the HU window as
> `[100, 1000]`. That is wrong and is the likely origin of the scorer bug.
> `[0, 1200]` is correct.

---

## 14. Checklist before committing

- [ ] File header states what it does, what it does NOT do, and how to call it
- [ ] Every magic number has a comment saying where it came from
- [ ] Every array argument documents axis order; every physical quantity, units
- [ ] No `dict.get()` with a default for anything a model or run depends on
- [ ] No silent `except` that changes behaviour instead of raising
- [ ] No copied file — parameterise the original instead
- [ ] Output folder contains `run.json` with the config actually used
- [ ] Tests must live strictly in the `tests/` directory, cleanly separated from the `src/backend/` logic
- [ ] A new API route takes `Depends(current_user)` and builds paths with the user's id
- [ ] Every name taken from a request goes through `safe_name`
- [ ] Changed `ui/` files? Bump the `?v=` cache version in `ui/index.html`
- [ ] If a number changed, you can explain why