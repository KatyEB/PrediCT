PrediCT Studio — Application Plan v2

Project: PrediCT · GSoC 2026 @ ML4Sci · Soham Jadhav · mentor Katy Butler Repo: github.com/KatyEB/PrediCT, branch soham_segmentation Final: 24 September 2026 · This doc: August 2026 Status: approved direction, not yet implemented

Reading this in a fresh session? This document is self-contained. It supersedes v1. The source of truth for code is the soham_segmentation branch — clone and read it, don't trust report versions. PrediCT_Continuation_v2.md does not exist; ignore any instruction pointing at it.

0. Locked decisions
Question	Decision
Final deadline	24 September 2026. ~7 weeks from 6 Aug.
Does Katy want this?	Yes — discussed and agreed.
Web or desktop	Web first. Convert to desktop (Tauri shell) after the product is complete.
Models	Multiple registered simultaneously, user chooses in the interface. 3D UNet and nnU-Net both first-class. Not one-at-a-time.
Current numbers	Not a blocker. Checkpoints improve over time; swapping them is what the registry is for.
Perlmutter / Slurm	Not in v1. Normal local compute — CUDA if present, else MPS, else CPU.
Input formats	DICOM folder, DICOM file, NIfTI, and single slice — all supported. See §3.
0.1 One correction I'm holding, and three I'm dropping

You said not to care about current results. You're right, and I'm dropping the hard gate from v1. Specifically:

Model quality (A3 v1 vs v2, Dice 0.61 vs 0.72): agreed, irrelevant to the app. Swap the checkpoint, the number improves. That's the design working.
Defect 2 — hardcoded checkpoint path in agatston_scoring_a3.py: dissolves completely into the manifest. Once weights are declared with a sha256, loading the wrong file becomes impossible. No separate fix needed.

But three of them are not results. They're source lines that get copied into predict-core if nobody stops them:

#	Defect	Why it isn't a "result"
1	Scorers window at [100, 1000]; all models trained at [0, 1200]	This is a preprocessing literal. In the new design the window comes from manifest.requires.hu_window, so there is no literal to get wrong. Fixing it costs nothing if we write the module correctly the first time.
3	No ≥1 mm² minimum-lesion rule	Agatston-conformance. A reviewer who knows the standard will catch it, and it's ~5 lines.
4	compute_xml_agatston() truncates with .astype(np.int32) when sampling peak HU	This is in the ground-truth path. Ground truth is the one thing that is not swappable. If the reference is wrong, every model comparison the app ever makes is wrong — forever, regardless of which checkpoint is loaded.

So: no separate 3-day gate. These become hard requirements on predict-core/scoring/ when we write it in week 1, which we're doing anyway. Zero extra calendar cost. The corrected Agatston numbers then fall out of the cohort runner for free, as a product output rather than a manual task.

Also worth clearing up while we're in the docs: docs/progress_report.md records the HU window as "settled on [100, 1000]" — it's [0, 1200]. And docs/Final_Testing_Report.md has its entire body duplicated.

1. What this product is

PrediCT Studio — a research workstation for coronary artery calcium. Load a cardiac CT, run one or more segmentation models, and get an Agatston score whose arithmetic you can audit lesion by lesion.

Users
User	Needs
Imaging researcher (primary)	Run models on their own scans without reading Python. Masks, scores, reproducible manifest.
Next contributor	Drop in nnU-Net or Rajat's attention-hybrid and get a fair comparison against the baselines without touching core code.
Mentor / reviewer	See results, inspect a case, verify the arithmetic, re-run cohort tables.
You	Regenerate every table and figure in the final report from one command.
Not
Not a clinical device. Research use only, marked persistently in the UI and stamped on every export.
Not a training platform. Inference and scoring only; training stays in your existing scripts.
Not a PACS. Reads folders, not a hospital network.
Not a general medical viewer. One anatomy, one biomarker, deliberately.

A tool that does CAC scoring completely is a real contribution. A half-built general viewer is not.

2. Architecture
┌──────────────────────────────────────────────────────────┐
│  predict-ui/        React + Vite. HTTP/SSE only.         │
└──────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────┐
│  predict-api/       FastAPI. Jobs, SSE progress,         │
│                     registry endpoints, exports.         │
└──────────────────────────────────────────────────────────┘
                            │
┌──────────────────────────────────────────────────────────┐
│  predict-core/      Pure Python. No UI. No top-level     │
│                     torch import. The whole science.     │
│    ingest/      scanner, series grouping, sniffing       │
│    io/          DICOM, NIfTI, XML plist parser           │
│    geometry/    Sutherland–Hodgman, shoelace, coverage   │
│    preprocess/  resample, HU window, ROI crop            │
│    scoring/     volume, Agatston (binary + soft), risk   │
│    registry/    plugin discovery + contract enforcement  │
│    pipeline/    DAG, content-addressed cache, executors  │
│    provenance/  run manifests, hashing, versioning       │
└──────────────────────────────────────────────────────────┘
         │                              │
┌────────────────────┐      ┌──────────────────────────────┐
│ predict-cli/       │      │ predict-plugins/             │
│ headless, same     │      │ one package per model,       │
│ core, for batch    │      │ brings its own framework     │
└────────────────────┘      └──────────────────────────────┘

Load-bearing rule: predict-core imports nothing from predict-api or predict-ui, and does not import torch at module level. Torch enters only inside a plugin. This is what lets someone write a scoring-only plugin without a GPU, and what keeps the CLI lean.

Why the split and not a monolith: you currently have four models, three scoring paths and two evaluation scripts expressed as ~6 top-level scripts with copy-pasted overlap. That's exactly where the HU-window and checkpoint-path defects came from. Extracting predict-core is a refactor of code you already trust, not new invention — the cheapest high-value work on the list.

3. Data ingest and the Study model

This section answers the loading question directly.

3.1 The single unifying idea

A Study is always a volume. A single slice is a volume with depth 1.

One data model, one code path, no if single_slice: branches scattered through the codebase. Everything downstream — viewer, scorer, provenance — operates on a Volume. What differs between a 120-slice scan and a 1-slice scan is which plugins are allowed to run on it, and that is already handled by the contract system in §4. No new machinery.

This matters because the alternative — a separate "slice mode" — would duplicate the viewer, the scorer, and the export path, and those duplicates would drift. You've already paid for that pattern once with the four copies of COCA_processor_main.py.

3.2 What can be dropped on the app
Input	Handling	Fully scoreable?
Folder of DICOM files (one series)	Group, sort, stack → volume	✅ Yes
Folder of folders (COCA layout: patient/<id>/<series>/)	Recursive scan → N studies, review before import	✅ Yes
Single multi-frame DICOM (enhanced CT, whole volume in one file)	Read frames → volume	✅ Yes
Single classic DICOM (one slice)	Volume with depth 1	⚠️ Degraded — see §3.5
NIfTI .nii / .nii.gz	Direct read; look for your _meta.json sidecar	✅ Yes
NIfTI, single slice	Volume with depth 1	⚠️ Degraded
XML plist annotation	Imported as ground-truth overlay alongside a series	—
PNG / JPG of a CT slice	Rejected, with an explanation	❌ Never

Why PNG/JPG is rejected rather than half-supported: Agatston is area(mm²) × density-weight(peak HU). An 8-bit image has neither Hounsfield Units nor pixel spacing. You cannot compute area in mm², and you cannot compute a density weight. A PNG is unscoreable in principle, not just inconvenient. Accepting it would mean either fabricating a spacing or silently producing a meaningless number — both worse than a clear refusal:

Can't score an image file. Agatston scoring needs Hounsfield Units and pixel spacing in millimetres. PNG and JPEG carry neither. → Import the original DICOM or NIfTI instead

3.3 The ingest pipeline
drop path
   │
   ├─▶ walk  ──▶ sniff each file
   │              DICM magic at byte 128  → DICOM
   │              NIfTI magic (348 / n+1) → NIfTI
   │              else                    → skip, count as unknown
   │
   ├─▶ group DICOM by SeriesInstanceUID
   │
   ├─▶ sort slices by ImagePositionPatient projected onto the slice normal
   │        ⚠ NOT by InstanceNumber — unreliable in real exports
   │
   ├─▶ validate: duplicate z-positions · non-uniform spacing ·
   │             missing slices · mixed orientations · mixed series in one folder
   │
   └─▶ REVIEW SCREEN — user confirms what to import
3.4 The review screen (and why it earns its place)

Ingest never silently imports. It shows a table first:

Series	Slices	Thickness	Spacing	Type	Warnings
1.2.840…4471	120	3.0 mm	0.37 mm	gated cardiac	—
1.2.840…8813	118	3.0 mm	0.37 mm	gated cardiac	⚠ 4 duplicate z-positions
1.2.840…2205	289	1.25 mm	0.68 mm	chest (non-gated)	⚠ not a calcium-scoring protocol

Two of those warnings are directly from your own project history:

Duplicate z-positions is the multi-series problem that produced patient 159 appearing twice under two scan IDs. That went undetected for weeks. Here it's visible at import.
Non-gated protocol is the 339 chest CTs in the raw COCA download that shouldn't be scored at all. A warning, not a hard block — the user may have a reason.

Detection heuristics: slice thickness (≈3 mm gated vs 1–1.25 mm chest), cardiac-gating DICOM tags where present, and SeriesDescription keywords. Stated as a guess in the UI, never as a fact.

Sidecar detection: if an XML plist sits next to a series, offer to import it as ground truth. If a _meta.json sits next to a NIfTI, read it to learn whether the volume is already resampled and windowed — otherwise treat external NIfTI as raw and show the detected spacing for confirmation.

3.5 Single-slice mode — what actually works and what doesn't

This is worth understanding precisely, because it's the kind of thing Katy may ask about.

What still works on one slice:

Viewing, window/level, overlays — fine.
Agatston arithmetic — fine. Agatston is defined per slice: each lesion contributes area × density-weight, summed across slices. One slice gives you that slice's contribution correctly. It just isn't a patient score, and the UI must say so: "Slice contribution: 25. Not a patient total."
Ground-truth scoring from XML for that slice — fine.
Any registered 2D model — fine.

What does not work, and why:

Blocked	Reason
3D models (your MONAI UNet, nnU-Net 3d_fullres)	Patch size is 96×96×32. Depth 1 can't fill it. Padding to 32 by replication feeds the network a physically impossible 96 mm uniform slab; its z-context features become noise, and the output is not a real prediction.
Heart-ROI crop	TotalSegmentator needs a volume to localise the heart. A bounding box from one slice is meaningless.
Anything with requires.roi: heart	Follows from the above — including your A1-ROI and A3-Coverage models.

How this is enforced: not with special-case code, but with one manifest field:

yaml
requires:
  min_slices: 32        # 3D UNet
  # min_slices: 1       # a 2D plugin

The registry filters the model list against the loaded study. A 1-slice study shows 3D models greyed out with the reason on hover:

a3-coverage-v2 needs at least 32 slices. This study has 1.

Recommendation: support single-slice for viewing, ground-truth inspection, and 2D plugins, and be explicit in the UI that no 3D model can run. Don't build a padding hack to make 3D models appear to work — a plausible-looking wrong answer is worse than a clear refusal, and it's the exact failure mode that produced the Approach 2 int32 collapse (accepted on a "nonzero" check that never validated against truth).

3.6 The Volume object
python
@dataclass(frozen=True)
class Volume:
    array: np.ndarray              # (z, y, x)
    spacing_mm: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]   # 9-element cosine matrix
    provenance: VolumeProvenance   # ← the anti-bug mechanism

@dataclass(frozen=True)
class VolumeProvenance:
    source: SourceRef              # path + sha256 + format
    resampled_to: tuple | None
    hu_window: tuple | None
    roi: ROIInfo | None            # locator id, margin, bbox
    n_slices: int
    steps: list[StepHash]          # full DAG chain

A Volume always knows what was done to it. That is what makes contract enforcement possible instead of aspirational.

4. Plugin system
4.1 Three kinds, one mechanism
Kind	Does	Now	Later
Locator	finds an anatomical ROI	TotalSegmentator (--roi_subset heart)	Rajat's LW-UNet
Segmenter	CT volume → calcium prediction	A1-full, A1-ROI, A3-coverage-v2	nnU-Net, attention hybrid, 2D models
Scorer	prediction → clinical numbers	volume, Agatston-binary, Agatston-soft	connectivity variants

Separating scorer from segmenter is what lets you ask "what does A1's mask score under A3's scorer?" — a question you currently can't answer cleanly, and one a mentor is likely to ask.

4.2 The manifest
yaml
id: a3-coverage-v2
name: Approach 3 — Coverage Fraction (v2)
version: 2.0.0
kind: segmenter
framework: monai
authors: ["Soham Jadhav"]

output:
  type: soft                  # soft | binary
  range: [0.0, 1.0]
  semantics: coverage_fraction

# THE CONTRACT. The runtime enforces every line.
requires:
  dimensionality: 3d
  min_slices: 32
  spacing_mm: [0.37, 0.37, 3.0]
  hu_window: [0, 1200]        # ← the only place this number exists
  roi: heart
  locator: totalsegmentator@2.15.0
  roi_margin_mm: 8

inference:
  patch_size: [96, 96, 32]
  sw_overlap: 0.5
  precision: bfloat16

weights:
  uri: file://runs/approach3_coverage_v2/best_model.pth
  sha256: a1b2c3…            # verified on every load
  size_bytes: 19284736

declared_metrics:             # shown as CLAIMED, never as measured
  val_dice_mean: 0.7227
  val_dice_median: 0.7865
  best_epoch: 140
  trained_on_split: splits_v8_310_65_66

scoring_defaults:
  scorer: agatston-soft
  min_lesion_area_mm2: 1.0
4.3 Enforcement

Before dispatch, the runtime compares volume.provenance against plugin.requires. On mismatch it refuses and says exactly what's wrong:

Can't run this model. a3-coverage-v2 needs HU window [0, 1200]. This volume was prepared at [100, 1000]. → Re-prepare at [0, 1200] · Pick a different model

That message is the fix for the HU-window defect expressed as a product feature. Unlike a literal buried in a script, it cannot be silently ignored.

sha256 verification on every weight load means a manifest pointing at v2 while v1 sits on disk fails loudly instead of quietly scoring you the worse model.

4.4 The interface
python
class Segmenter(Protocol):
    manifest: PluginManifest
    def load(self, device: Device) -> None: ...
    def predict(self, volume: Volume) -> Prediction: ...
    def unload(self) -> None: ...

Three methods. Resist growing this — every method added is a method every future contributor must implement.

Prediction carries the array, its dtype/range semantics (binary vs soft, which the scorer needs), and back-references to the volume provenance and the plugin id. A prediction always knows what made it.

4.5 Multiple models at once

Per your requirement, models aren't one-at-a-time.

All registered models are live simultaneously. The registry loads manifests at startup; weights load lazily on first use and unload under memory pressure (LRU, configurable ceiling).
A model bar in the reading view shows registered models as chips, filtered to those whose contract this study satisfies. Incompatible ones are greyed with the reason on hover.
Multi-select runs them all in one job. Because the DAG cache is content-addressed, three models sharing requires.roi: heart run TotalSegmentator once, not three times. On a 66-patient cohort that's over an hour saved — this is the concrete payoff of the caching design.
Results are layers. Each model's prediction is a toggleable overlay with its own colour and its own row in the lesion ledger. Colour discipline in §8.
Cap simultaneous overlays at 3 for legibility; the rest stay available as toggles.

Deliberately not in v1: ensembling / consensus masks. It's a natural v2 feature and the layer model already sets it up, but it's a research question (how do you combine a binary A1 with a soft A3?) rather than a UI feature, and it would eat week 5.

4.6 Discovery
Entry points — [project.entry-points."predict.plugins"] in pyproject.toml. pip install predict-nnunet and it appears. This is how a stranger contributes.
Drop-in folder — ~/.predict/plugins/<name>/manifest.yaml + plugin.py. For when you have a new checkpoint at 11pm and want it in the dropdown in two minutes.
4.7 Honesty rules
Plugins are arbitrary Python; there is no sandbox. Say so in the docs. For a research tool that's the right tradeoff — pretending otherwise would be worse.
declared_metrics display with a "claimed by plugin" marker. The registry never presents self-reported Dice as measured truth. Where the app has measured a model on a cohort, it shows claimed and measured side by side. This is the difference between a tool that helps you evaluate honestly and one that launders claims.
5. Pipeline and compute
5.1 DAG with content-addressed caching

Every step is (input_hash, param_hash) → output.

ingest ─▶ resample ─▶ window ─▶ locate(heart) ─▶ crop ─▶ infer ─▶ score
                                     ↑                     ↑        ↑
                               slow, cache hard       plugin    plugin
Comparison is nearly free — models sharing preprocessing recompute only infer and score.
Resumability by construction — a killed cohort run restarts at the first uncached step. (Your standing rule: anything over a few minutes must be resumable. Cache hits give this for free.)
Provenance is the hash chain — a result's manifest is its reproduction recipe.
5.2 Compute — local only, no Slurm
python
class Executor(Protocol):
    def submit(self, dag: JobDAG) -> JobHandle: ...

Only LocalExecutor is built. The protocol stays because it costs one file today and retrofitting it later is expensive — but no Slurm/Perlmutter implementation in v1, per your call.

Device selection: auto-detect cuda → mps → cpu, overridable in settings, always displayed in the UI. Never silently fall back — a user waiting 8 minutes for a CPU run should know why.

Honest performance expectations (to set in the docs, and to verify on your machine in week 3 rather than assuming):

Step	GPU	CPU
Resample + window, 512×512×120	~1 s	~2 s
TotalSegmentator heart, --fast	~20–40 s	~1–3 min
TotalSegmentator heart, full	~1 min	~5–10 min
3D UNet sliding window, ROI crop	~5–15 s	~1–3 min
Agatston scoring	<1 s	<1 s

Use --fast for interactive single-study runs. It's a 3 mm-resolution model, and you are drawing a bounding box with an 8 mm margin — 1.5 mm precision buys nothing. Reserve full resolution for cohort runs where wall-clock matters less. This is a defensible optimisation, not a shortcut.

Memory: a 512×512×120 float32 volume is ~126 MB. The content-addressed store needs a size ceiling with LRU eviction (default 20 GB, configurable) or it will quietly fill a disk during a cohort run.

5.3 Job lifecycle

queued → running(step k/n, pct) → done | failed | cancelled

Progress streams over SSE. Failed jobs keep their partial cache and name the failing step; "retry from failed step" is one click.

6. Storage
Thing	Where	Why
Studies, jobs, results, registry	SQLite	Single-user local tool. Zero setup.
Volumes, masks, cached outputs	Content-addressed store ~/.predict/store/<sha[:2]>/<sha>	Dedup is automatic.
Cohort results	Parquet	You already use it; keeps continuity with your splits.
Run manifests	JSON, one per job	Human-readable, diffable, greppable.

Deliberately not: Postgres, Redis, Celery. A local imaging workstation should not need three services to start. The SQLAlchemy layer swaps if hosting ever matters.

7. API surface
GET    /api/studies                     list, filter, sort
POST   /api/ingest/scan                 walk a path → detected series + warnings
POST   /api/ingest/import               confirm which series to import
GET    /api/studies/{id}
GET    /api/studies/{id}/volume/{step}  streamed NIfTI for the viewer

GET    /api/plugins                     registry + contract + verify status
GET    /api/plugins?compatible_with={study_id}   ← drives the model bar
POST   /api/plugins/verify/{id}         re-check weight hash

POST   /api/jobs                        {study_ids, segmenters[], scorer, params}
GET    /api/jobs/{id}
GET    /api/jobs/{id}/events            SSE progress
POST   /api/jobs/{id}/cancel

GET    /api/results/{job_id}            scores, lesion ledger, per-slice burden
GET    /api/results/{job_id}/manifest   full provenance
POST   /api/compare                     {job_ids[]} → agreement map + deltas

POST   /api/cohorts                     run over a split parquet
GET    /api/cohorts/{id}/tables         Dice dist, Bland–Altman, Agatston scatter,
                                        risk confusion matrix, McNemar
GET    /api/export/{job_id}?fmt=…       csv | nifti | pdf | manifest

segmenters[] being a list, not a string, is the multi-model requirement expressed in the API.

8. UI and UX
8.1 Direction

The reference world is the reading room, not the dashboard. Radiology workstations (Slicer, OsiriX, RadiAnt) get the discipline right — dark, dense, image-sovereign, keyboard-driven — and the craft wrong: grey Qt chrome, tiny widgets, no typographic care. The opening is to keep the discipline and bring real craft to it.

Every choice below has a functional reason. That's the point — decoration is what makes software look generated.

Surfaces — two temperatures, and the split means something

Token	Hex	Use
surface-image	
#08090A	Panels containing a scan. Near-true-black so the CT's own blacks are the darkest thing on screen.
surface-data	
#141517	Panels containing numbers, tables, controls. One step up, faintly warm.
surface-raised	
#1C1E21	Menus, popovers, command palette.
hairline	rgba(255,255,255,0.07)	Every division.
text-primary	
#E8E6E3	Warm off-white, never pure white.
text-muted	
#8A8781	Labels, units, secondary.

Warm-neutral, not blue-black: blue-tinted UI shifts perceived grayscale in a dim room, which matters when the user is judging HU by eye. The image/data surface split is a structural device encoding something true — where pixels are sovereign vs where numbers are — and it does the work drop-shadowed cards would otherwise do.

Colour is semantic and rationed. There is no brand accent.

Role	Ramp	Means
Primary prediction	ember 
#7A3B14 → 
#FF8A3D → 
#FFC46B	The active model. A ramp, not a swatch.
Reference	cyan 
#4EC9D9	XML ground truth. Outline only, never filled.
Second model	violet 
#A78BE8	Comparison layer.
Third model	sage 
#8FBC8F	Third layer only. Cap at three.

Making the primary colour a colormap instead of an accent is the deliberate departure from the "near-black plus one bright accent" look that every AI-generated dark UI converges on. It's also your thesis made visible: A1 renders flat because it is binary; A3 renders as a gradient because it is fractional. Someone comparing the two understands your contribution before reading a word.

Verify ember/cyan/violet/sage under deuteranopia and protanopia with a simulator before shipping.

Typography — one superfamily, three cuts, three jobs

The default reach is Inter + JetBrains Mono. Instead: IBM Plex, which has genuine lineage in scientific instrumentation and gives a coherent family story.

Role	Face	Setting
Interface	IBM Plex Sans	13px / 500, -0.005em
All numbers	IBM Plex Mono, font-variant-numeric: tabular-nums	13px / 450
Instrument labels	IBM Plex Sans Condensed	10px / 600, uppercase, 0.12em

Every HU, mm³, mm² and Agatston value is tabular-aligned. Misaligned digits in a column of measurements is the clearest tell of software written by someone who doesn't work with measurements.

Layout — tiled panes, hairline dividers, no cards

Panes tile edge to edge and resize; hairlines divide them. No floating cards, no drop shadows, border-radius: 2px maximum. Cards-with-shadows is the generated look; tiled panes is the workstation look, and it gives maximum pixels to the image.

One true detail: a divider between two synced panes glows faintly (rgba(255,138,61,0.15)). Sync state is otherwise invisible and easy to get wrong; the hairline carries the information.

Motion — nearly none, two exceptions

Medical software that bounces reads as a toy. 120 ms crossfades on pane swap, nothing else. Two earned exceptions:

Slice scrubbing is 1:1 with input, zero easing — pre-decode the stack to a texture atlas so it never drops a frame. This is what separates a real viewer from a demo.
The compare wipe is a draggable divider — direct manipulation, not animation.

One orchestrated moment: on job completion the lesion ledger populates row-by-row in a ~20 ms stagger while the calcium spine fills top to bottom. Fast, once, and it encodes real information. prefers-reduced-motion skips it.

8.2 The signature: the calcium spine

A persistent vertical strip on the left edge of the reading view. One row per slice, top to bottom, full scan depth. Each row's ember intensity encodes calcium burden on that slice. The current slice rides it as a bright cursor.

Why this rather than a scrollbar: your own EDA says the median positive scan has calcium on 6 slices out of ~120. Calcium is radically sparse in z. A scrollbar tells you where you are; the spine tells you where the disease is and lets you jump straight there. It's the scan's actual shape turned into the navigation device.

With multiple models loaded the spine splits into parallel columns — one per model, in that model's layer colour — so every slice where they disagree is visible as a mismatched pair. A per-scan disagreement summary in 40 pixels of width.

┌──┬─────────────────────────────┬────────────────────────┐
│▓▓│                             │  LESION LEDGER         │
│▓░│                             │  ────────────────────  │
│░▓│                             │  z    mm²   HU   w  →  │
│  │       AXIAL (primary)       │  34   4.21  387  3  25 │
│  │                             │  35   2.80  241  2  11 │
│██│  ◀ current slice            │  41   9.04  512  4  72 │
│▓▓│                             │  …                     │
│░░│                             │  ────────────────────  │
│  │                             │  AGATSTON        108   │
│  │                             │  RISK        MODERATE  │
├──┼──────────────┬──────────────┼────────────────────────┤
│  │   CORONAL    │   SAGITTAL   │  W/L ─────●── 400/40   │
└──┴──────────────┴──────────────┴────────────────────────┘
 ▲▲
 spine — one row per slice, one column per model
8.3 Screens

1 · Studies — a dense table, not a card grid. 34 px rows, ~40 visible. Columns: ID, patient, slices, inline calcium-per-slice sparkline, Agatston, risk as a four-segment burden bar, last run, status. The sparkline is what makes 40 rows scannable — you see the shape of each study's disease without opening it.

Empty state: "No studies yet. Drop a DICOM folder anywhere on this window."

2 · Import review (§3.4) — detected series, slice counts, thickness, protocol guess, warnings. Checkboxes for what to import. This screen is where duplicate z-positions and non-gated protocols surface instead of hiding.

3 · Reading — the core. Axial primary, coronal + sagittal secondary, spine, model bar, and the lesion ledger as the fourth pane.

The ledger is the workhorse of the product. Every lesion is a row: slice, area mm², peak HU, density weight, contribution to total. Click a row → axial jumps there and rings the lesion. The Agatston score stops being a number and becomes an audit trail. Most viewers hide this arithmetic; showing it is your project's honesty made into UI, and it's the single thing most likely to make a mentor trust the tool.

With multiple models the ledger gets a layer selector, and a diff mode showing per-lesion disagreement.

Keyboard: ↑↓/scroll = slice · W/L drag = window/level · 1–4 = layouts · space = toggle overlay · tab = next lesion · ⌘K = command palette.

4 · Compare — synced viewports with a draggable wipe for two models; layer toggles plus an agreement map for three. Agreement in ember, model-only regions in each model's colour. Split spine. Per-lesion delta table.

5 · Cohort — run selected models over a split parquet. Produces live: Dice distribution (violin), volume Bland–Altman, Agatston scatter with risk-threshold lines at 1/100/400, risk-category confusion matrix, McNemar. These are exactly your report's tables — one command regenerates every figure in the final deliverable. Highest-value screen for you personally.

6 · Models — the registry. Per plugin: contract, weight-hash status (green = matches, red = drifted), claimed vs measured metrics, load/unload, compatibility against the current study with the failing requirement named.

7 · Run inspector — the job DAG. Per step: cache hit/miss, duration, input/output hashes. A "Copy reproduction command" button emitting the exact CLI invocation. This is the screen that makes the tool credible to a reviewer.

8.4 Copy voice

Plain, active, instrument-like. Buttons name what happens: "Run model," not "Submit." An action keeps its name through the flow — "Run model" produces "Model run complete." Errors state what happened and what to do, and never apologise:

Heart locator found no cardiac region. TotalSegmentator returned an empty mask for this series. This usually means it isn't a gated cardiac CT. → Run without ROI crop · Inspect the series

8.5 Research-use marking

Persistent and non-dismissible in the reading view, stamped on every PDF export: "Research use only — not for clinical decision-making." Not a modal you click once. A reviewer will look for this, and its absence is a bad look on a project touching cardiac risk.

9. Stack
Layer	Choice	Why / alternative
Backend	FastAPI + Pydantic	Pydantic expresses the plugin contract in the same language as the API schema.
Frontend	React + Vite + TypeScript	Boring and correct. Vite because HMR on a viewer matters.
Viewer	NiiVue	WebGL2, NIfTI-native, MPR + overlays out of the box, small API. Alt: Cornerstone3D — DICOM-native and far heavier; only needed if you want raw-DICOM reading in the browser, which you don't (you preprocess server-side).
Charts	Observable Plot or visx	Not Recharts — its defaults are the visual signature of a generated dashboard and hard to escape.
Styling	CSS custom properties + CSS Modules	Tailwind pulls toward its own defaults, which is what you're avoiding. A ~40-token system gives tighter control.
DB	SQLite + SQLAlchemy	Zero setup, swappable.
DICOM	pydicom + SimpleITK	You already use SimpleITK; pydicom for tag inspection during ingest.
Packaging	docker-compose v1	Reviewer runs docker compose up. Tauri shell later — the API layer makes that a wrapper, not a rewrite.

Deliberately not: Postgres, Redis, Celery, Electron, Tailwind, Recharts, Slurm.

10. Sequence to 24 September

7 weeks. The app doesn't own all of them — the final report and any remaining research work need room.

When	Work	Done means
Aug 7 → Aug 20	Extract predict-core: ingest, io, geometry, preprocess, scoring written correctly (window from manifest, ≥1 mm² rule, no int32 truncation in ground truth). Plugin registry + contracts. predict-cli. Cohort runner.	predict run --models a1-roi,a3-coverage-v2 --cohort test_split.parquet reproduces every table in your report, with corrected Agatston. This alone is a defensible GSoC deliverable.
Aug 21 → Sep 3	FastAPI + job DAG + SSE. Ingest + review screen. Reading view: spine, axial, lesion ledger, model bar. Models screen.	Drop a DICOM folder, run two models, audit the score in a browser.
Sep 4 → Sep 12	Compare view. Cohort view. MPR panes. Full design pass against §8.	Multi-model side by side; report figures generated from the UI.
Sep 13 → Sep 19	docker-compose packaging, plugin-authoring guide, demo video, final report.	A stranger clones, runs docker compose up, scores a scan, and writes a plugin from the guide.
Sep 20 → Sep 24	Buffer + submission.	—

Cut order if you slip — decide now, not in week 6:

Compare view (the CLI can compare)
Cohort view (the CLI can produce the tables)
MPR coronal/sagittal (axial alone is usable)
Single-slice mode (nice, not essential)
Never cut: core + CLI + plugin registry + reading view.

The rule: the UI must never be on the critical path for your GSoC final. Core + CLI is the deliverable; everything visual is upside.

11. Risks
Risk	Mitigation
App becomes the project; the report slips	Week 4 is packaging and report. Cut order decided in advance.
Plugin abstraction over-engineered for 3 models	Justified by the bugs it prevents, not by model count. Three methods on the interface; resist growth.
TotalSegmentator slow and heavy to ship	--fast for interactive use; cache hard; document as an optional heavy dependency, not a hard one.
Viewer performance on 512×512×120	Texture atlas; downsample MPR panes; profile with your largest scan (patient 321) in week 2, not week 5.
Split-count integrity unresolved (310/65/66 vs 308/66/66)	The manifest records which split file produced each result. Doesn't fix it — stops it propagating silently. Still worth resolving before the final report.
Single-slice mode produces a plausible wrong answer	Hard refusal for 3D models, no padding hack. §3.5.
Mistaken for a clinical tool	Persistent research-use marking, stamped on exports.
12. Still open
NiiVue or Cornerstone3D — decide before Aug 21; close to irreversible. Recommendation stands: NiiVue.
Does Rajat's attention-hybrid go in as a plugin? If yes, get his preprocessing requirements before the manifest schema freezes (~Aug 20).
nnU-Net packaging — nnU-Net expects its own dataset folder layout and environment variables. Wrapping it as a plugin means an adapter that translates a Volume into what nnU-Net wants. Budget half a day and prototype it early; it's the strongest test that the plugin interface is actually general and not just shaped around your own models.
13. Immediate next step

Pull the real soham_segmentation tree and produce a file-by-file extraction map: every existing script → the predict-core module it becomes, with duplicated logic named explicitly and the three scoring corrections marked at their exact call sites.

That's the document that turns this plan into work, and it needs the actual repo — not a recollection of it.