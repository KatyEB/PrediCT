/* PrediCT Studio — frontend.
 *
 * Served by src/backend/server.py (FastAPI): ui/ at /ui, data/ at /data.
 *
 *     python -m src.backend.server
 *     http://127.0.0.1:8001/ui/?study=172&model=a1-roi
 *
 * Reads an already-produced output folder (data/out/<study>/<model>) and
 * renders it. With no study in the URL a start screen is shown instead.
 * Nothing here computes a score; tiers are a display classification only.
 *
 * One state object, one render(). Every handler mutates state then calls
 * render(). There is no other update path.
 */

// ── configuration ────────────────────────────────────────────────────────
const qs = new URLSearchParams(location.search);
const STUDY = qs.get('study');   // null on the start screen
const MODEL = qs.get('model');
const HAS_STUDY = Boolean(STUDY && MODEL);
const BASE = `/data/out/${encodeURIComponent(STUDY)}/${encodeURIComponent(MODEL)}`;
const studyUrl = (study, model) =>
  `?study=${encodeURIComponent(study)}&model=${encodeURIComponent(model)}`;

// render.py flips vertically (flipud) because the direction cosines are
// diag(-1,-1,1): increasing array row is increasing anterior, so row 0 at the
// top would show the slice posterior-up. lesions.csv stays in unflipped array
// coordinates, so every lesion coordinate drawn here must be flipped to match.
// If fliplr is later added to render.py for radiological left/right, flip
// FLIP_X to true here in the same commit — the two must always agree.
const FLIP_Y = true;
const FLIP_X = true;

const TIER_SCALE_MAX = 1400;   // full width of the tier bar; ticks at the tier bounds

// Risk tier schemes. Display only: the score is never recomputed, only the
// label it is shown under. Each entry is [inclusive upper bound, name].
// 4-tier is what scoring.py writes to run.json as risk_category; the 6-tier
// bounds are the ones used in the soham_segmentation evaluation.
const TIER_SCHEMES = {
  4: [[0, 'ZERO'], [100, 'MILD'], [400, 'MODERATE'], [Infinity, 'SEVERE']],
  6: [[0, 'ZERO'], [100, 'MILD'], [300, 'MODERATE'], [400, 'MOD-HIGH'],
      [1000, 'SEVERE'], [Infinity, 'EXTENSIVE']],
};
const TIER_COLORS = {
  ZERO: 'var(--md-sys-color-outline)',
  MILD: 'var(--md-sys-color-success)',
  MODERATE: 'var(--md-sys-color-warning)',
  'MOD-HIGH': 'var(--tier-mod-high)',
  SEVERE: 'var(--md-sys-color-error)',
  EXTENSIVE: 'var(--tier-extensive)',
};

const ZOOM_MAX = 8;            // Instrument viewer, ×
const RAIL_DEFAULT = 380, RAIL_MIN = 280, RAIL_MAX = 600;   // right panel, px

const SOFT_NOTE =
  'Coverage is not binary. A voxel at 0.35 contributes 0.35 of its area and ' +
  'is drawn as grain, never as an edge. The 0.10 threshold only decides where ' +
  'one component stops and the next begins — it does not gate the score.';

let ACCENT_COLOR = '#C98B2E';

// ── state ────────────────────────────────────────────────────────────────
const state = {
  dir: 1,          // 1 argument, 2 instrument, 3 contact sheet, 4 anatomy
  view: 2,         // 1 original, 2 prediction, 3 calcium only
  tiers: loadPref('tiers') === '6' ? 6 : 4,   // risk tier scheme shown
  zoom: { s: 1, x: 0, y: 0 },   // Instrument viewer: scale, and top-left offset
                                // as a fraction of the pane (survives resizes)
  slice: 0,
  sel: null,       // "sliceIdx:lesionId"
  sel3d: null,     // "L004" — the selected 3D lesion, or null
  run: null,
  slices: [],
  lesions: [],
  lesions3d: [],
  imgW: 0,
  imgH: 0,
  covCache: {},    // slice idx -> [n,n,n,n] recovered from the mask alpha channel
};

// ── load ─────────────────────────────────────────────────────────────────
async function boot() {
  // The sidebar works with or without a study open.
  loadSidebar();
  initRunForm();
  initUpload();
  initSections();

  if (!HAS_STUDY) { showWelcome(); return; }

  try {
    ACCENT_COLOR = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#C98B2E';

    const [run, slices, csv, csv3d] = await Promise.all([
      getJson(`${BASE}/run.json`),
      getJson(`${BASE}/slices.json`),
      getText(`${BASE}/lesions.csv`),
      getText(`${BASE}/lesions_3d.csv`),
    ]);
    state.run = run;
    state.slices = slices;
    state.lesions = parseCsv(csv);
    state.lesions3d = parseCsv(csv3d);

    // Image dimensions are read from the first CT PNG. run.json may or may not
    // carry "shape" depending on which patches have landed, and volumes are
    // heart-cropped so no size can be assumed.
    const probe = await loadImage(ctUrl(0));
    state.imgW = probe.naturalWidth;
    state.imgH = probe.naturalHeight;

    const first = slices.find(s => s.has_calcium);
    state.slice = first ? first.idx : 0;

    wire();
    render();
  } catch (e) {
    showWelcome(
      `Could not load study "${STUDY}" with model "${MODEL}".\n\n${e.message}\n\n` +
      `Check that data/out/${STUDY}/${MODEL}/ contains run.json, slices.json, ` +
      `lesions.csv and lesions_3d.csv, or run the pipeline again.`);
  }
}

/* Start screen: shown when no study is in the URL, or when one failed to load.
   The tabs stay visible but inert (body.no-study), and render() never runs. */
function showWelcome(errMsg) {
  document.body.classList.add('no-study');
  document.getElementById('welcome').hidden = false;
  document.getElementById('welcome-err').hidden = !errMsg;
  document.getElementById('welcome-errmsg').textContent = errMsg || '';
}

// Per-browser UI preferences (tier scheme, panel width, folded sections).
// Storage can be unavailable (private mode, blocked site data); the UI then
// simply starts from its defaults.
function loadPref(key) {
  try { return localStorage.getItem('predict.' + key); } catch { return null; }
}
function savePref(key, value) {
  try { localStorage.setItem('predict.' + key, value); } catch { /* defaults next time */ }
}

async function getJson(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u} → HTTP ${r.status}`); return r.json(); }
async function getText(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u} → HTTP ${r.status}`); return r.text(); }
function loadImage(u) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = () => rej(new Error(`Failed to load ${u}`));
    img.src = u;
  });
}

function initUpload() {
  const btn = document.getElementById('upload-study-btn');
  const input = document.getElementById('upload-input');
  const overlay = document.getElementById('drop-overlay');
  
  if (!btn || !input || !overlay) return;

  btn.onclick = () => input.click();
  document.getElementById('welcome-upload').onclick = () => input.click();

  input.onchange = async (e) => {
    if (e.target.files.length > 0) {
      await handleUpload(e.target.files);
    }
    input.value = '';
  };

  document.body.addEventListener('dragover', (e) => {
    e.preventDefault();
    overlay.style.display = 'flex';
  });

  document.body.addEventListener('dragleave', (e) => {
    e.preventDefault();
    if (e.clientX === 0 || e.clientY === 0) {
      overlay.style.display = 'none';
    }
  });

  document.body.addEventListener('drop', async (e) => {
    e.preventDefault();
    overlay.style.display = 'none';
    
    const files = [];
    if (e.dataTransfer.items) {
      for (let i = 0; i < e.dataTransfer.items.length; i++) {
        const item = e.dataTransfer.items[i].webkitGetAsEntry();
        if (item) await scanFiles(item, files);
      }
    } else {
      for (let i = 0; i < e.dataTransfer.files.length; i++) {
        files.push(e.dataTransfer.files[i]);
      }
    }
    
    if (files.length > 0) await handleUpload(files);
  });
}

async function scanFiles(item, files) {
  if (item.isFile) {
    const file = await new Promise((resolve) => item.file(resolve));
    files.push(file);
  } else if (item.isDirectory) {
    const dirReader = item.createReader();
    const entries = await new Promise((resolve) => {
      dirReader.readEntries(resolve);
    });
    for (let i = 0; i < entries.length; i++) {
      await scanFiles(entries[i], files);
    }
  }
}

/* Upload a folder to data/raw/<id> (server.py). On success the new patient is
   selected in the Run Pipeline form; nothing reloads and nothing runs until
   the user presses Run. */
async function handleUpload(files) {
  const name = await askDialog({
    title: 'Upload study',
    message: `${files.length} file(s) selected. Name this study, or leave it blank ` +
      'to generate a name from the series.',
    input: '',
    okLabel: 'Upload',
  });
  if (name === null) return;   // cancelled: nothing was sent

  const btn = document.getElementById('upload-study-btn');
  btn.disabled = true;
  btn.textContent = 'hourglass_empty';

  try {
    const fd = new FormData();
    if (name) fd.append('custom_name', name);
    for (const f of files) fd.append('files', f);

    const res = await fetch('/studies', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await errorText(res, 'Upload failed'));
    let data = await res.json();

    if (data.requires_cleaning) {
      const clean = await askDialog({
        title: 'Some files are not DICOM or NIfTI',
        message: `${data.invalid_files.length} file(s) cannot be used ` +
          `(e.g. ${data.invalid_files[0]}). Remove them and continue with the rest?`,
        okLabel: 'Remove and continue',
      });
      if (!clean) {
        await fetch(`/studies/clean/${data.temp_id}`, { method: 'DELETE' });
        toast('Upload cancelled. Nothing was saved.');
        return;
      }
      btn.textContent = 'cleaning_services';
      const q = name ? `?custom_name=${encodeURIComponent(name)}` : '';
      const cleanRes = await fetch(`/studies/clean/${data.temp_id}${q}`, { method: 'POST' });
      if (!cleanRes.ok) throw new Error(await errorText(cleanRes, 'Cleaning failed'));
      data = await cleanRes.json();
    }

    await refreshPatients(data.study_id);
    toast(`Uploaded "${data.study_id}". Choose a model and press Run.`, 'ok');
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'upload';
  }
}

// FastAPI errors are {"detail": "..."}; anything else falls back to the status.
async function errorText(res, fallback) {
  try {
    const j = await res.json();
    return j.detail ? `${fallback}: ${j.detail}` : fallback;
  } catch {
    return `${fallback} (HTTP ${res.status})`;
  }
}

/* The one modal the UI uses. Resolves to the typed text when `input` is given,
   true when confirmed without input, and null when cancelled (button or Esc). */
function askDialog({ title, message, input = null, okLabel = 'OK' }) {
  const dlg = document.getElementById('dlg');
  const field = document.getElementById('dlg-input');
  document.getElementById('dlg-title').textContent = title;
  document.getElementById('dlg-msg').textContent = message;
  document.getElementById('dlg-ok').textContent = okLabel;
  document.getElementById('dlg-cancel').onclick = () => dlg.close();
  field.hidden = input === null;
  field.value = input || '';
  dlg.returnValue = '';
  dlg.showModal();
  if (input !== null) field.focus();
  return new Promise(resolve => {
    dlg.onclose = () => {
      if (dlg.returnValue !== 'ok') resolve(null);
      else resolve(input === null ? true : field.value.trim());
    };
  });
}

// Short non-blocking message at the bottom of the screen. Click to dismiss.
let toastTimer = null;
function toast(msg, kind = 'info') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast ${kind}`;
  el.hidden = false;
  el.onclick = () => { el.hidden = true; };
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, kind === 'error' ? 8000 : 4000);
}

// lesions.csv has no quoted fields, so a split is enough.
function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const head = lines[0].split(',');
  return lines.slice(1).filter(Boolean).map(line => {
    const cells = line.split(',');
    const o = {};
    head.forEach((h, i) => {
      const v = (cells[i] ?? '').trim();
      o[h] = v === 'True' ? true : v === 'False' ? false
        : (v !== '' && !isNaN(v)) ? Number(v) : v;
    });
    return o;
  });
}

const pad = i => String(i).padStart(3, '0');
const ctUrl = i => `${BASE}/slices/ct/slice_${pad(i)}.png`;
const maskUrl = i => `${BASE}/slices/mask/slice_${pad(i)}.png`;

// ── derived values ───────────────────────────────────────────────────────
// The model's output semantics cannot be read off the weights, which is why
// manifests exist. run.json carries "output" once that patch lands; until then
// fall back to the model id, and say so rather than guessing silently.
function outputType() {
  if (state.run.output) return state.run.output;
  return MODEL.includes('coverage') ? 'coverage' : 'binary';
}
function threshold() {
  if (state.run.threshold != null) return state.run.threshold;
  return outputType() === 'coverage' ? 0.10 : 0.50;
}
function pixelArea() {
  const sp = state.run.spacing || [0.37, 0.37, 3.0];
  return sp[0] * sp[1];
}

// mean coverage per lesion. If scoring.py writes it, use it. Otherwise derive
// it exactly: for a coverage model area_mm2 = Σp × pixel_area and n_voxels is
// the component's voxel count, so Σp / n_voxels is the mean. For a binary
// model this returns 1.00, which is correct and not a placeholder.
function meanCoverage(l) {
  if (l.mean_coverage != null && l.mean_coverage !== '') return l.mean_coverage;
  if (!l.n_voxels) return null;
  return l.area_mm2 / (l.n_voxels * pixelArea());
}

const counted = () => state.lesions.filter(l => l.included);
const excluded = () => state.lesions.filter(l => !l.included);

// Every per-slice component belonging to a 3D lesion, in slice order.
const membersOf = key => state.lesions
  .filter(l => l.lesion_3d_key === key)
  .sort((a, b) => a.slice_idx - b.slice_idx);

const group3d = key => state.lesions3d.find(g => g.lesion_3d_key === key) || null;

// Slices a 3D lesion appears on — used to mark the volume track.
const slicesOf = key => membersOf(key).map(l => l.slice_idx);

const counted3d = () => state.lesions3d.filter(g => g.included);
const onSlice = i => state.lesions.filter(l => l.slice_idx === i);
const calcSlices = () => state.slices.filter(s => s.has_calcium).map(s => s.idx);
const keyOf = l => `${l.slice_idx}:${l.lesion_id}`;
const selLesion = () => state.lesions.find(l => keyOf(l) === state.sel) || null;
const sliceMeta = i => state.slices.find(s => s.idx === i) || { idx: i, z_mm: 0, slice_score: 0 };

const tierIn = (t, n) => TIER_SCHEMES[n].find(([max]) => t <= max)[1];
const tierOf = t => tierIn(t, state.tiers);
const tierColor = tier => TIER_COLORS[tier];

// ── interaction ──────────────────────────────────────────────────────────
function stack() { return state.view === 3 ? calcSlices() : state.slices.map(s => s.idx); }

function step(d) {
  const st = stack();
  if (!st.length) return;
  let i = st.indexOf(state.slice);
  if (i === -1) { state.slice = st[0]; render(); return; }
  i = Math.max(0, Math.min(st.length - 1, i + d));
  state.slice = st[i];
  state.sel = null;
  render();
}

function goTo(sliceIdx, key) {
  state.slice = sliceIdx;
  state.sel = key || null;
  // Selecting a component always implies its 3D lesion. There is no state in
  // which a component is selected and its lesion is not.
  const l = key ? state.lesions.find(x => keyOf(x) === key) : null;
  state.sel3d = l ? l.lesion_3d_key : null;
  if (state.view === 3 && !calcSlices().includes(sliceIdx)) state.view = 2;
  render();
}

function goToLesion(key3d) {
  const g = group3d(key3d);
  if (!g) return;
  const m = membersOf(key3d).find(l => l.slice_idx === g.peak_slice_idx)
    || membersOf(key3d)[0];
  goTo(m.slice_idx, keyOf(m));
}

function wire() {
  document.querySelectorAll('#tabs button[data-dir]').forEach(b =>
    b.onclick = () => { state.dir = Number(b.dataset.dir); render(); });
  document.querySelectorAll('#tabs button[data-view]').forEach(b =>
    b.onclick = () => {
      state.view = Number(b.dataset.view);
      if (state.view === 3) {
        const c = calcSlices();
        if (c.length && !c.includes(state.slice)) state.slice = c[0];
      }
      render();
    });
  document.querySelectorAll('#tabs button[data-tiers]').forEach(b =>
    b.onclick = () => {
      state.tiers = Number(b.dataset.tiers);
      savePref('tiers', state.tiers);
      render();
    });

  const onWheel = e => { e.preventDefault(); step(e.deltaY > 0 ? 1 : -1); };
  document.getElementById('i-track').addEventListener('wheel', onWheel, { passive: false });
  document.getElementById('a-pane').addEventListener('wheel', onWheel, { passive: false });
  wireZoom();   // the Instrument viewport's wheel: slices, or zoom with ctrl
  initRailResize();
  initThumbSize();

  document.addEventListener('keydown', e => {
    // Shortcuts belong to the viewer: not to a form field, an open dialog, or
    // browser shortcuts such as Ctrl+0 / Ctrl+1.
    if (e.target.closest('input, select, textarea, [contenteditable]')) return;
    if (document.querySelector('dialog[open]')) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    const cols = state.dir === 3 ? currentColumnCount() : 1;
    if (e.key === 'ArrowDown') { step(cols); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { step(-cols); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
    else if (e.key === 'Escape') { state.sel = null; state.sel3d = null; render(); }
    else if (e.key === 'Home') {
      const c = calcSlices(); if (c.length) goTo(c[0]); e.preventDefault();
    }
    else if (e.key === 'End') {
      const c = calcSlices(); if (c.length) goTo(c[c.length - 1]); e.preventDefault();
    }
    else if (e.key === '1' || e.key === '2' || e.key === '3') {
      const btn = document.querySelector(`#tabs button[data-view="${e.key}"]`);
      if (btn) btn.click();
    }
    else if (state.dir === 2 && (e.key === '+' || e.key === '=')) zoomBy(1.25);
    else if (state.dir === 2 && e.key === '-') zoomBy(1 / 1.25);
    else if (state.dir === 2 && e.key === '0') resetZoom();
  });

  window.addEventListener('resize', scheduleRender);
}

// Coalesce bursts (window resize, panel drag, pan) into one render per frame.
let renderQueued = false;
function scheduleRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; render(); });
}

// ── zoom (02 Instrument) ─────────────────────────────────────────────────
// Display only: the images are scaled with a CSS transform, so voxels stay
// voxels (image-rendering: pixelated) and nothing is resampled or measured.

// Zoom to scale s, keeping the point (cx, cy) fixed. cx/cy are fractions of
// the pane, e.g. the pointer position or 0.5/0.5 for the centre.
function zoomAt(s, cx, cy) {
  const z = state.zoom;
  s = Math.min(ZOOM_MAX, Math.max(1, s));
  // the image must always cover the pane: offset stays within [1 - s, 0]
  const clamp = v => Math.min(0, Math.max(1 - s, v));
  z.x = clamp(cx - (cx - z.x) * s / z.s);
  z.y = clamp(cy - (cy - z.y) * s / z.s);
  z.s = s;
  scheduleRender();
}
const zoomBy = f => zoomAt(state.zoom.s * f, 0.5, 0.5);
function resetZoom() { state.zoom = { s: 1, x: 0, y: 0 }; render(); }

function wireZoom() {
  const vp = document.getElementById('i-viewport');
  const pane = document.getElementById('i-pane');

  // Plain wheel steps slices, as before. Ctrl/⌘ + wheel zooms about the
  // pointer; a trackpad pinch arrives as ctrl + wheel too. preventDefault is
  // what stops the browser zooming the whole page while over the image.
  vp.addEventListener('wheel', e => {
    e.preventDefault();
    if (!e.ctrlKey && !e.metaKey) { step(e.deltaY > 0 ? 1 : -1); return; }
    const r = pane.getBoundingClientRect();
    const fx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const fy = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    zoomAt(state.zoom.s * Math.exp(-e.deltaY * 0.002), fx, fy);
  }, { passive: false });

  // Drag to pan while zoomed in.
  pane.addEventListener('pointerdown', e => {
    if (state.zoom.s === 1 || e.button !== 0) return;
    pane.setPointerCapture(e.pointerId);
    pane.classList.add('panning');
    let lastX = e.clientX, lastY = e.clientY;
    pane.onpointermove = ev => {
      const z = state.zoom, r = pane.getBoundingClientRect();
      z.x = Math.min(0, Math.max(1 - z.s, z.x + (ev.clientX - lastX) / r.width));
      z.y = Math.min(0, Math.max(1 - z.s, z.y + (ev.clientY - lastY) / r.height));
      lastX = ev.clientX; lastY = ev.clientY;
      scheduleRender();
    };
    pane.onpointerup = pane.onpointercancel = () => {
      pane.onpointermove = null;
      pane.classList.remove('panning');
    };
  });
  pane.addEventListener('dblclick', resetZoom);

  document.querySelectorAll('#i-zoom button').forEach(b => b.onclick = () => {
    if (b.dataset.zoom === 'in') zoomBy(1.25);
    else if (b.dataset.zoom === 'out') zoomBy(1 / 1.25);
    else resetZoom();
  });
}

// ── layout controls ──────────────────────────────────────────────────────
// Right panel width: one value for every tab, dragged from .rail-handle.
function initRailResize() {
  const root = document.getElementById('main-content');
  const setWidth = px => root.style.setProperty('--rail-w', px + 'px');
  let width = Number(loadPref('rail-w')) || RAIL_DEFAULT;
  setWidth(width);

  document.querySelectorAll('.rail-handle').forEach(h => {
    h.onpointerdown = e => {
      h.setPointerCapture(e.pointerId);
      h.classList.add('dragging');
      const body = h.parentElement.getBoundingClientRect();
      const off = parseFloat(getComputedStyle(h.parentElement).getPropertyValue('--rail-off')) || 0;
      // never squeeze the image area below 400 px
      const max = Math.max(RAIL_MIN, Math.min(RAIL_MAX, body.width - off - 400));
      h.onpointermove = ev => {
        width = Math.round(Math.min(max, Math.max(RAIL_MIN, body.right - off - ev.clientX)));
        setWidth(width);
        scheduleRender();
      };
      h.onpointerup = h.onpointercancel = () => {
        h.onpointermove = null;
        h.classList.remove('dragging');
        savePref('rail-w', width);
      };
    };
    h.ondblclick = () => {
      width = RAIL_DEFAULT;
      setWidth(width);
      savePref('rail-w', width);
      scheduleRender();
    };
  });
}

// Contact Sheet tile size.
function initThumbSize() {
  const input = document.getElementById('cs-size');
  const grid = document.getElementById('cs-grid');
  const apply = () => grid.style.setProperty('--cs-thumb', input.value + 'px');
  input.value = loadPref('cs-thumb') || 68;
  apply();
  input.oninput = () => { apply(); savePref('cs-thumb', input.value); };
}

// Side sections are <details data-sec>; remember which ones were folded.
function initSections() {
  const closed = new Set((loadPref('closed') || '').split(',').filter(Boolean));
  document.querySelectorAll('details[data-sec]').forEach(d => {
    d.open = !closed.has(d.dataset.sec);
    d.addEventListener('toggle', () => {
      if (d.open) closed.delete(d.dataset.sec); else closed.add(d.dataset.sec);
      savePref('closed', [...closed].join(','));
    });
  });
}

// ── render ───────────────────────────────────────────────────────────────
function render() {
  document.getElementById('d1').classList.toggle('on', state.dir === 1);
  document.getElementById('d2').classList.toggle('on', state.dir === 2);
  document.getElementById('d3').classList.toggle('on', state.dir === 3);
  document.getElementById('d4').classList.toggle('on', state.dir === 4);
  document.querySelectorAll('#tabs button[data-dir]').forEach(b =>
    b.classList.toggle('on', Number(b.dataset.dir) === state.dir));
  document.querySelectorAll('#tabs button[data-view]').forEach(b =>
    b.classList.toggle('on', Number(b.dataset.view) === state.view));
  document.querySelectorAll('#tabs button[data-tiers]').forEach(b =>
    b.classList.toggle('on', Number(b.dataset.tiers) === state.tiers));
  document.getElementById('tabs-study').textContent =
    `study ${STUDY} · ${MODEL} · ${state.slices.length} slices`;

  if (state.dir === 1) renderArgument();
  else if (state.dir === 2) renderInstrument();
  else if (state.dir === 3) renderContactSheet();
  else renderAnatomy();
  preload(state.slice);
}

/* 04 Anatomy. view3d.js is an ES module and therefore deferred, so it may not
   have registered window.View3D by the time the tab is first clicked. Mount
   lazily on first entry and no earlier — WebGL context and ~1 MB of meshes are
   not worth creating for a session that never opens this tab. */
function renderAnatomy() {
  const man = state.run.mesh;
  const err = document.getElementById('v-err');

  paintScore('v', ' · scored in 2D, per slice — this view measures nothing');

  if (!man) {
    err.hidden = false;
    err.textContent =
      'run.json has no "mesh" block, so this study was produced before meshing ' +
      'existed. Re-run it to generate surfaces. Nothing is drawn rather than ' +
      'something approximate.';
    return;
  }
  if (!window.View3D) { setTimeout(render, 50); return; }   // module still loading
  window.View3D.mount(BASE, man).then(() => window.View3D.update());
}

/* The explicit contract view3d.js reads. It is a separate module with its own
   scope, so what it may touch is listed here rather than left to whatever
   happens to be in the global lexical environment. */
window.PrediCT = { state, group3d, counted3d, goToLesion, ctUrl, maskUrl, render };

function preload(i) {
  for (let d = -3; d <= 3; d++) {
    const j = i + d;
    if (j >= 0 && j < state.slices.length) { new Image().src = ctUrl(j); new Image().src = maskUrl(j); }
  }
}

/* Fill an image pane: CT underneath, mask on top, selection ring on the canvas.
   The pane keeps the volume's aspect ratio inside whatever box CSS gives it.
   A pane with a .pane-zoom wrapper (Instrument) also gets state.zoom. */
function paintPane(paneId, boxW, boxH) {
  const pane = document.getElementById(paneId);
  const ct = pane.querySelector('.pane-ct');
  const mask = pane.querySelector('.pane-mask');
  const cv = pane.querySelector('.pane-ring');
  const zoomEl = pane.querySelector('.pane-zoom');
  const z = zoomEl ? state.zoom : { s: 1, x: 0, y: 0 };

  const ar = state.imgW / state.imgH;
  let w = boxW, h = boxW / ar;
  if (h > boxH) { h = boxH; w = boxH * ar; }
  pane.style.width = Math.round(w) + 'px';
  pane.style.height = Math.round(h) + 'px';

  if (zoomEl) {
    zoomEl.style.transform = `translate(${z.x * 100}%, ${z.y * 100}%) scale(${z.s})`;
    pane.classList.toggle('zoomed', z.s > 1);
  }

  if (ct.getAttribute('src') !== ctUrl(state.slice)) ct.src = ctUrl(state.slice);
  if (mask.getAttribute('src') !== maskUrl(state.slice)) {
    mask.onerror = () => pane.classList.add('missing');
    mask.onload = () => pane.classList.remove('missing');
    mask.src = maskUrl(state.slice);
  }
  pane.classList.toggle('view-1', state.view === 1);

  // orientation labels. Anterior is at the top because of flipud. Left/right
  // follows FLIP_X: with no horizontal flip, increasing array x is patient
  // right and column 0 is at the viewer's left, so the viewer's left is the
  // patient's LEFT (neurological). Setting FLIP_X flips both the image and
  // this label together.
  pane.querySelector('.pane-labels').innerHTML =
    `<span class="oA">A</span><span class="oP">P</span>` +
    `<span class="oL">${FLIP_X ? 'R' : 'L'}</span><span class="oR">${FLIP_X ? 'L' : 'R'}</span>`;

  // selection ring
  cv.width = Math.round(w); cv.height = Math.round(h);
  const g = cv.getContext('2d');
  g.clearRect(0, 0, cv.width, cv.height);

  // Two rings, one rule: solid = the component you selected, dashed = the same
  // 3D lesion on the slice you are looking at now. Both are drawn in array
  // coordinates and flipped here, the only place that conversion happens.
  // The canvas is not zoomed; the zoom is applied to the coordinates instead,
  // so the ring stays a crisp 1.5 px line at any scale.
  const drawRing = (l, dashed) => {
    const sx = cv.width / state.imgW * z.s, sy = cv.height / state.imgH * z.s;
    const ox = z.x * cv.width, oy = z.y * cv.height;
    let x0 = l.bbox_x0, x1 = l.bbox_x1, y0 = l.bbox_y0, y1 = l.bbox_y1;
    if (FLIP_X) { const a = state.imgW - 1 - x1, b = state.imgW - 1 - x0; x0 = a; x1 = b; }
    if (FLIP_Y) { const a = state.imgH - 1 - y1, b = state.imgH - 1 - y0; y0 = a; y1 = b; }
    const m = 3;
    g.setLineDash(dashed ? [3, 3] : []);
    g.strokeStyle = dashed ? ACCENT_COLOR : '#4FA8C5';
    g.lineWidth = 1.5;
    g.strokeRect(ox + x0 * sx - m, oy + y0 * sy - m, (x1 - x0 + 1) * sx + 2 * m, (y1 - y0 + 1) * sy + 2 * m);
    g.setLineDash([]);
  };

  const sel = selLesion();
  if (state.view !== 1) {
    if (state.sel3d) {
      membersOf(state.sel3d)
        .filter(l => l.slice_idx === state.slice && (!sel || keyOf(l) !== state.sel))
        .forEach(l => drawRing(l, true));
    }
    if (sel && sel.slice_idx === state.slice) drawRing(sel, false);
  }
  return { w, h };
}

/* Coverage bands. slices.json carries coverage_hist once render.py writes it;
   otherwise recover the same histogram from the mask PNG's alpha channel,
   which is exactly round(p*255). Same numbers either way. */
function coverageBands(idx, done) {
  if (state.covCache[idx]) return done(state.covCache[idx]);
  const meta = sliceMeta(idx);
  if (meta.coverage_hist) { state.covCache[idx] = meta.coverage_hist; return done(meta.coverage_hist); }

  const im = new Image();
  im.onload = () => {
    const c = document.createElement('canvas');
    c.width = im.naturalWidth; c.height = im.naturalHeight;
    const g = c.getContext('2d');
    g.drawImage(im, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    const edges = [0.25, 0.5, 0.75].map(v => v * 255);
    const h = [0, 0, 0, 0];
    const lo = threshold() * 255;
    for (let i = 3; i < d.length; i += 4) {
      const a = d[i];
      if (a <= lo) continue;
      h[a < edges[0] ? 0 : a < edges[1] ? 1 : a < edges[2] ? 2 : 3]++;
    }
    state.covCache[idx] = h;
    done(h);
  };
  im.onerror = () => done(null);
  im.src = maskUrl(idx);
}

// ══ 01 ARGUMENT ══════════════════════════════════════════════════════════
function renderArgument() {
  const cs = calcSlices(), cnt = counted(), ex = excluded();

  paintScore('a');

  const zs = cs.map(i => sliceMeta(i).z_mm);
  const wts = cnt.map(l => l.density_weight);
  const zmin = Math.min(...zs).toFixed(1);
  const zmax = Math.max(...zs).toFixed(1);
  const wmin = Math.min(...wts);
  const wmax = Math.max(...wts);

  let becauseStr = '';
  if (cnt.length === 0) {
    becauseStr = `No component reached 1.0 mm². ${state.slices.length} slices were predicted and scored; every connected component measured below the minimum and is recorded as withheld.`;
  } else {
    becauseStr = `${cnt.length} of ${state.lesions.length} components cleared 1.0 mm². `;
    if (counted3d().length === cnt.length) {
      becauseStr += `They sit on ${cs.length} of ${state.slices.length} slices, z ${zmin}–${zmax} mm, weights ${wmin}–${wmax}. No lesion in this study spans more than one slice. `;
    } else {
      becauseStr += `They group into ${counted3d().length} lesions across ${cs.length} of ${state.slices.length} slices, z ${zmin}–${zmax} mm, weights ${wmin}–${wmax}. `;
    }
    becauseStr += (outputType() === 'coverage'
      ? 'Area is the sum of per-voxel coverage, so partial voxels enter at their own fraction.'
      : 'Area is a count of voxels above threshold, so each boundary voxel is either fully counted or fully discarded.');
  }
  document.getElementById('a-because').textContent = becauseStr;

  document.getElementById('a-viewname').textContent = viewName();
  document.getElementById('a-overlay').textContent = overlayNote();

  paintPane('a-pane', 230, 230);
  document.getElementById('a-panecap').textContent =
    `selected exhibit · slice ${state.slice} · z ${sliceMeta(state.slice).z_mm.toFixed(1)} mm`;

  // exhibit strip — every calcium-bearing slice. Built once per study; a
  // render only moves the highlight, so thumbnails are never re-created (and
  // never flash) while stepping, scrolling or resizing.
  const strip = document.getElementById('a-strip');
  if (!strip.children.length) {
    cs.forEach(i => {
      const sc = sliceMeta(i).slice_score || 0;
      const b = document.createElement('button');
      b.dataset.idx = i;
      b.innerHTML =
        `<span class="pane">` +
        `<img class="pane-ct" loading="lazy" src="${ctUrl(i)}" alt="">` +
        `<img class="pane-mask" loading="lazy" src="${maskUrl(i)}" alt="">` +
        `</span><span>${i} · ${sc.toFixed(1)}</span>`;
      b.onclick = () => goTo(i);
      strip.appendChild(b);
    });
  }
  for (const b of strip.children) {
    const on = Number(b.dataset.idx) === state.slice;
    b.classList.toggle('on', on);
    b.firstChild.classList.toggle('view-1', state.view === 1);
    // bring the selected thumbnail into view once per slice change, scrolling
    // only the strip so the page does not jump
    if (on && strip.dataset.shown !== String(state.slice)) {
      strip.dataset.shown = state.slice;
      keepVisible(strip, b);
    }
  }

  // lesion 3d index
  const lb = document.getElementById('a-l3drows');
  lb.innerHTML = '';
  counted3d().forEach(g => {
    const tr = document.createElement('tr');
    tr.className = g.lesion_3d_key === state.sel3d ? 'on' : '';
    tr.innerHTML =
      `<td class="l">${g.lesion_3d_key}</td>` +
      `<td>${g.n_slices} sl</td>` +
      `<td>${g.span_mm.toFixed(0)} mm</td>` +
      `<td>${g.total_agatston.toFixed(1)}</td>`;
    tr.onclick = () => goToLesion(g.lesion_3d_key);
    lb.appendChild(tr);
  });
  const spans = counted3d().filter(g => g.n_slices > 1).length;
  document.getElementById('a-l3dsummary').textContent =
    counted3d().length === 0 ? 'no scored lesion'
      : `${counted3d().length} lesions · ${spans} span more than one slice · ` +
      `largest ${Math.max(...counted3d().map(g => g.n_slices))} slices`;

  // excluded
  const eb = document.getElementById('a-exrows');
  eb.innerHTML = '';
  ex.forEach(l => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td class="l">sl ${l.slice_idx}</td><td>${l.area_mm2.toFixed(2)} mm²</td>` +
      `<td>${l.peak_hu} HU</td><td class="w">withheld</td>`;
    tr.onclick = () => goTo(l.slice_idx, keyOf(l));
    eb.appendChild(tr);
  });
  document.getElementById('a-exsummary').textContent = exSummary();
  document.getElementById('a-soft').textContent =
    outputType() === 'coverage' ? SOFT_NOTE
      : 'This model outputs a binary mask. Area is a voxel count above threshold ' +
      `${threshold().toFixed(2)}, so every boundary voxel is either fully counted or fully ` +
      'discarded. That rounding is what the coverage model exists to remove.';
  document.getElementById('a-provblock').textContent = provBlock();
}

// Scroll box (only) so that el is visible inside it.
function keepVisible(box, el) {
  const b = box.getBoundingClientRect(), r = el.getBoundingClientRect();
  if (r.top < b.top) box.scrollTop -= b.top - r.top;
  else if (r.bottom > b.bottom) box.scrollTop += r.bottom - b.bottom;
}

// ══ 02 INSTRUMENT ════════════════════════════════════════════════════════
function renderInstrument() {
  const r = state.run, cs = calcSlices(), ex = excluded();
  const here = onSlice(state.slice);
  const meta = sliceMeta(state.slice);



  // ── track
  document.getElementById('i-tracklabel').textContent =
    `VOLUME · SLICE ${state.slice}` + (state.view === 3 ? ` · CALCIUM-ONLY STACK, ${cs.length} SLICES` : '');
  const zs = cs.map(i => sliceMeta(i).z_mm);
  document.getElementById('i-tracksummary').textContent =
    `${cs.length} of ${state.slices.length} slices carry scored calcium · ` +
    `${Math.round(cs.length / state.slices.length * 100)} %` +
    (cs.length ? ` · z ${Math.min(...zs).toFixed(1)}–${Math.max(...zs).toFixed(1)} mm` : '');

  const maxScore = Math.max(1, ...state.slices.map(s => s.slice_score || 0));
  const track = document.getElementById('i-track');
  track.innerHTML = '';
  state.slices.forEach(s => {
    const sc = s.slice_score || 0;
    const any = onSlice(s.idx).length > 0;
    const reach = state.view !== 3 || s.has_calcium;
    const b = document.createElement('button');
    b.style.height = (sc > 0 ? Math.max(16, 16 + sc / maxScore * 26) : any ? 16 : 5) + 'px';
    if (sc > 0) {
      b.style.background = 'var(--accent)';
      b.style.border = '0';
    } else if (any) {
      b.style.background = 'transparent';
      b.style.border = '2px solid var(--accent)';
      b.style.boxSizing = 'border-box';
    } else {
      b.style.background = 'var(--rule-2)';
      b.style.border = '0';
    }
    b.style.opacity = reach ? 1 : .3;
    if (s.idx === state.slice) b.classList.add('cur');
    if (state.sel3d && slicesOf(state.sel3d).includes(s.idx)) b.classList.add('in3d');
    b.title = `slice ${s.idx} · z ${s.z_mm.toFixed(1)} mm · ${sc ? sc.toFixed(1) : 'empty'}` +
      (reach ? '' : ' · unreachable in calcium-only');
    if (reach) b.onclick = () => goTo(s.idx);
    track.appendChild(b);
  });
  document.getElementById('i-legend3d').hidden = !state.sel3d;
  document.getElementById('i-cursor').textContent = `▲ cursor slice ${state.slice}`;
  document.getElementById('i-reach').textContent = state.view === 3
    ? `${state.slices.length - cs.length} of ${state.slices.length} slices unreachable in this stack`
    : `all ${state.slices.length} slices reachable`;

  // ── viewport
  const vp = document.getElementById('i-viewport');
  const { w } = paintPane('i-pane', vp.clientWidth - 28, vp.clientHeight - 28);
  // 10 mm scale bar at the current zoom: mm -> source pixels -> screen pixels
  const sp = r.spacing || [0.37, 0.37, 3.0];
  const barPx = Math.round((10 / sp[0]) * (w / state.imgW) * state.zoom.s);
  document.getElementById('i-caption').innerHTML =
    `<i class="i-scalebar" style="width:${barPx}px"></i><span>10 mm</span>`;
  document.getElementById('i-zoomlvl').textContent = state.zoom.s.toFixed(1) + '×';

  // ── rail head
  document.getElementById('i-slice').textContent = `SLICE ${state.slice}`;
  document.getElementById('i-z').textContent = `z ${meta.z_mm.toFixed(1)} mm`;
  const chips = document.getElementById('i-chips');
  chips.innerHTML = '';
  if (!here.length) {
    chips.innerHTML = '<span class="none">no lesion on this slice</span>';
  } else {
    here.forEach((l, n) => {
      const p = meanCoverage(l);
      const b = document.createElement('button');
      b.className = (keyOf(l) === state.sel ? 'on ' : '') + (l.included ? '' : 'ex');
      b.textContent = `${String.fromCharCode(97 + n)} · ${l.lesion_3d_key} · ${l.area_mm2.toFixed(2)} mm²` +
        (p == null ? '' : ` · p${p.toFixed(2)}`) + (l.included ? '' : ' · withheld');
      b.onclick = () => goTo(l.slice_idx, keyOf(l));
      chips.appendChild(b);
    });
  }

  // ── selected 3D lesion
  const kv = (k, v, cls) => `<div class="i-kv"><span>${k}</span><span class="${cls || ''}">${v}</span></div>`;
  const l3dSec = document.getElementById('i-l3d-sec');
  const g3 = state.sel3d ? group3d(state.sel3d) : null;
  l3dSec.hidden = !g3;
  if (g3) {
    const mem = membersOf(state.sel3d);
    document.getElementById('i-l3d').innerHTML =
      kv('lesion', g3.lesion_3d_key) +
      kv('slices', `${g3.n_slices} (${g3.slice_min}–${g3.slice_max})`) +
      kv('z extent', `${g3.span_mm.toFixed(1)} mm`) +
      kv('components', g3.n_components_included === g3.n_components
        ? g3.n_components
        : `${g3.n_components_included} of ${g3.n_components} counted`) +
      kv('total area', `${g3.total_area_mm2.toFixed(2)} mm²`) +
      kv('peak HU', `${g3.max_peak_hu} (slice ${g3.peak_slice_idx})`) +
      kv('lesion score', g3.total_agatston.toFixed(2), 'score');

    const box = document.getElementById('i-l3d-slices');
    box.innerHTML = '';
    const peak = Math.max(...mem.map(l => l.agatston), 1);
    mem.forEach(l => {
      const b = document.createElement('button');
      b.className = (l.slice_idx === state.slice ? 'on ' : '') + (l.included ? '' : 'ex');
      b.innerHTML =
        `<span class="sl">sl ${l.slice_idx}</span>` +
        `<i><b style="width:${Math.round(l.agatston / peak * 100)}%"></b></i>` +
        `<span class="sc">${l.included ? l.agatston.toFixed(1) : 'withheld'}</span>`;
      b.onclick = () => goTo(l.slice_idx, keyOf(l));
      box.appendChild(b);
    });
  }

  // ── selected lesion
  const sel = selLesion();
  const selBox = document.getElementById('i-sel');
  if (!sel) {
    selBox.innerHTML = kv('lesion', here.length ? 'none selected' : 'none on slice', 'withheld');
  } else {
    const p = meanCoverage(sel);
    const n = here.indexOf(sel) + 1;
    selBox.innerHTML =
      kv('lesion', `L${sel.lesion_id} (${n} of ${here.length} here)`) +
      kv('area', `${sel.area_mm2.toFixed(3)} mm²`) +
      kv('voxels', sel.n_voxels) +
      kv('peak HU', sel.peak_hu) +
      kv('density weight', sel.density_weight) +
      kv('mean coverage', p == null ? '—' : p.toFixed(3), p != null && p < 0.5 ? 'soft' : '') +
      kv('score', sel.included ? sel.agatston.toFixed(2) : 'withheld', sel.included ? 'score' : 'withheld');
  }

  // ── coverage bands (coverage models only)
  const bandsSec = document.getElementById('i-bands-sec');
  bandsSec.hidden = outputType() !== 'coverage';
  if (!bandsSec.hidden) {
    document.getElementById('i-softnote').textContent = SOFT_NOTE;
    const box = document.getElementById('i-bands');
    const at = state.slice;
    coverageBands(at, h => {
      if (state.slice !== at) return;                 // slice moved while decoding
      if (!h) { box.innerHTML = '<span class="i-empty">no mask PNG for this slice</span>'; return; }
      const tot = h.reduce((a, b) => a + b, 0) || 1;
      const labels = ['0.10–0.25 · soft rim', '0.25–0.50 · partial',
        '0.50–0.75 · partial', '0.75–1.00 · dense core'];
      box.innerHTML = h.map((n, i) =>
        `<div class="i-band"><i><b style="width:${Math.round(n / tot * 100)}%"></b></i>` +
        `<span>${labels[i]} · ${n} vox</span></div>`).join('');
    });
  }

  // ── lesion table
  const tb = document.getElementById('i-rows');
  tb.innerHTML = '';
  const ordered = [...state.lesions].sort((a, b) =>
    a.lesion_3d_id - b.lesion_3d_id || a.slice_idx - b.slice_idx);

  let prev = null;
  ordered.forEach(l => {
    const first = l.lesion_3d_key !== prev; prev = l.lesion_3d_key;
    const p = meanCoverage(l);
    const tr = document.createElement('tr');
    tr.className = [keyOf(l) === state.sel ? 'on' : l.slice_idx === state.slice ? 'cur' : '',
    l.included ? '' : 'ex'].filter(Boolean).join(' ');
    tr.classList.toggle('g3-first', first);
    tr.classList.toggle('g3-on', l.lesion_3d_key === state.sel3d);
    tr.innerHTML =
      `<td class="g3">${first ? l.lesion_3d_key : ''}</td>` +
      `<td class="l">sl ${l.slice_idx}</td>` +
      `<td>${l.area_mm2.toFixed(2)}</td>` +
      `<td class="${p != null && p < 0.5 ? 'soft' : ''}">${p == null ? '—' : 'p ' + p.toFixed(2)}</td>` +
      `<td>${l.included ? l.agatston.toFixed(1) : '—'}</td>`;
    tr.onclick = () => goTo(l.slice_idx, keyOf(l));
    tb.appendChild(tr);
  });
  const empty = document.getElementById('i-empty');
  empty.hidden = state.lesions.length > 0;
  empty.textContent = `Track empty. ${state.slices.length} slices predicted, no voxel above ` +
    `component threshold ${threshold().toFixed(2)}. The instrument still steps through all of them.`;

  document.getElementById('i-nex').textContent = ex.length;
  document.getElementById('i-exsummary').textContent = exSummary();

  // ── footer
  paintScore('i', ' · ↑↓ step · ctrl+wheel zoom · 1/2/3 view · esc clear');
}

// ══ 03 CONTACT SHEET ═════════════════════════════════════════════════════

function currentColumnCount() {
  const grid = document.getElementById('cs-grid');
  if (!grid || !grid.children.length) return 1;
  const cols = getComputedStyle(grid).gridTemplateColumns.split(' ').length;
  return cols || 1;
}

function renderContactSheet() {
  const ex = excluded();
  const here = onSlice(state.slice);
  const meta = sliceMeta(state.slice);



  // grid
  const grid = document.getElementById('cs-grid');
  const sel3dSlices = state.sel3d ? slicesOf(state.sel3d) : [];

  if (grid.children.length !== state.slices.length) {
    grid.innerHTML = '';
    state.slices.forEach(s => {
      const b = document.createElement('button');
      b.className = 'cs-frame';
      b.id = 'cs-frame-' + s.idx;
      b.innerHTML =
        `<span class="pane"><img class="pane-ct" loading="lazy" decoding="async" src="${ctUrl(s.idx)}" alt="">` +
        `<img class="pane-mask" loading="lazy" decoding="async" src="${maskUrl(s.idx)}" alt=""></span>` +
        `<span class="cs-cap">${s.idx}</span>`;
      b.onclick = () => goTo(s.idx);

      const maskImg = b.querySelector('.pane-mask');
      maskImg.onerror = () => b.classList.add('cs-missing');

      grid.appendChild(b);
    });
  }

  let nCounted = 0, nSubmin = 0, nEmpty = 0;

  state.slices.forEach(s => {
    const b = document.getElementById('cs-frame-' + s.idx);
    b.className = 'cs-frame';
    if (s.idx === state.slice) b.classList.add('cs-cursor');
    if (state.sel3d && sel3dSlices.includes(s.idx)) b.classList.add('cs-in3d');

    // View 3 dims unreachable frames
    b.style.opacity = (state.view !== 3 || s.has_calcium) ? 1 : 0.3;
    const pane = b.querySelector('.pane');
    pane.className = 'pane' + (state.view === 1 ? ' view-1' : '');

    const hasCalc = s.has_calcium;
    const any = onSlice(s.idx).length > 0;
    if (hasCalc) { b.classList.add('cs-counted'); nCounted++; }
    else if (any) { b.classList.add('cs-submin'); nSubmin++; }
    else { nEmpty++; }
  });

  document.getElementById('cs-n-counted').textContent = nCounted;
  document.getElementById('cs-n-submin').textContent = nSubmin;
  document.getElementById('cs-n-empty').textContent = nEmpty;
  document.getElementById('cs-legend-note').textContent = `all ${state.slices.length} slices reachable · every frame windowed -100–400 HU`;

  // Enlargement pane
  paintPane('cs-pane', 186, 186);
  document.getElementById('cs-slice').textContent = `FRAME ${state.slice} — `;
  document.getElementById('cs-z').textContent = `z ${meta.z_mm.toFixed(1)} mm`;

  // Chips
  const chips = document.getElementById('cs-chips');
  chips.innerHTML = '';
  if (!here.length) {
    chips.innerHTML = '<span class="none">no lesion on this slice</span>';
  } else {
    here.forEach((l, n) => {
      const p = meanCoverage(l);
      const b = document.createElement('button');
      b.className = (keyOf(l) === state.sel ? 'on ' : '') + (l.included ? '' : 'ex');
      b.textContent = `${String.fromCharCode(97 + n)} · ${l.lesion_3d_key} · ${l.area_mm2.toFixed(2)} mm²` +
        (p == null ? '' : ` · p${p.toFixed(2)}`) + (l.included ? '' : ' · withheld');
      b.onclick = () => goTo(l.slice_idx, keyOf(l));
      chips.appendChild(b);
    });
  }

  // 3D Lesion Info
  const l3dInfo = document.getElementById('cs-l3d-info');
  const g3 = state.sel3d ? group3d(state.sel3d) : null;
  if (g3) {
    l3dInfo.textContent = `${g3.lesion_3d_key} · ${g3.n_slices} frames · slices ${g3.slice_min}–${g3.slice_max} · ${g3.span_mm.toFixed(1)} mm · ${g3.total_agatston.toFixed(1)} Agatston`;
  } else {
    l3dInfo.textContent = '';
  }

  // ── lesion table
  const cnt = counted();
  const tb = document.getElementById('cs-rows');
  tb.innerHTML = '';
  const ordered = [...cnt].sort((a, b) =>
    a.lesion_3d_id - b.lesion_3d_id || a.slice_idx - b.slice_idx);

  let prev = null;
  ordered.forEach(l => {
    const first = l.lesion_3d_key !== prev; prev = l.lesion_3d_key;
    const p = meanCoverage(l);
    const tr = document.createElement('tr');
    tr.className = [keyOf(l) === state.sel ? 'on' : l.slice_idx === state.slice ? 'cur' : '',
    l.included ? '' : 'ex'].filter(Boolean).join(' ');
    tr.classList.toggle('g3-first', first);
    tr.classList.toggle('g3-on', l.lesion_3d_key === state.sel3d);
    tr.innerHTML =
      `<td class="g3">${first ? l.lesion_3d_key : ''}</td>` +
      `<td class="l">sl ${l.slice_idx}</td>` +
      `<td>${l.area_mm2.toFixed(2)}</td>` +
      `<td class="${p != null && p < 0.5 ? 'soft' : ''}">${p == null ? '—' : 'p ' + p.toFixed(2)}</td>` +
      `<td>${l.included ? l.agatston.toFixed(1) : '—'}</td>`;
    tr.onclick = () => goTo(l.slice_idx, keyOf(l));
    tb.appendChild(tr);
  });

  document.getElementById('cs-ncounted-text').textContent = cnt.length;
  const empty = document.getElementById('cs-empty');
  empty.hidden = cnt.length > 0;
  empty.textContent = `No components counted.`;

  document.getElementById('cs-nex').textContent = ex.length;
  document.getElementById('cs-exsummary').textContent = exSummary();

  // ── footer
  paintScore('cs', ' · ↑↓ step · 1/2/3 view · esc clear');
}

// ── shared text ──────────────────────────────────────────────────────────
function viewName() { return state.view === 1 ? 'ORIGINAL' : state.view === 2 ? 'PREDICTION' : 'CALCIUM ONLY'; }

function overlayNote() {
  if (state.view === 1) return 'original stack · prediction not drawn here';
  if (state.view === 3) return `restricted stack · ${calcSlices().length} scored slices`;
  return outputType() === 'coverage'
    ? 'coverage overlay · alpha = fraction · never thresholded'
    : `binary overlay · thresholded at ${threshold().toFixed(2)}`;
}

function prov(n) {
  const r = state.run;
  if (n === 1) return outputType() === 'coverage'
    ? `${r.model_id} · coverage · component threshold ${threshold().toFixed(2)} (delineation only — area = Σ coverage)`
    : `${r.model_id} · binary · threshold ${threshold().toFixed(2)} · area = voxel count`;
  if (n === 2) {
    const sp = r.spacing.map(v => v.toFixed(2)).join(' × ');
    return `model HU window ${r.hu_window[0]}–${r.hu_window[1]} · ${sp} mm · RAS`;
  }
  return `crop heart +8 mm, TotalSegmentator ${r.locator_version} · ckpt ${String(r.sha256).slice(0, 12)} · ` +
    `min lesion 1.0 mm² · ${r.date}` +
    ` · 3D link: in-plane overlap, max gap ${state.run.max_gap_slices} slice(s)`;
}

function provBlock() { return [prov(1), prov(2), prov(3)].join('\n'); }

function tierNote(t) {
  const scheme = TIER_SCHEMES[state.tiers];
  const top = scheme[scheme.length - 2][0];   // last finite bound
  const n = `${state.tiers}-tier`;
  if (t === 0) return `${n} · Zero (0) · nothing to bound`;
  if (t > top) return `${n} · >${top} · ${(t - top).toFixed(1)} above the bound`;
  const bounds = scheme.map(([max], i) =>
    i === 0 ? '0' : max === Infinity ? `>${top}` : `${scheme[i - 1][0] + 1}–${max}`);
  return `${n} · bounds ${bounds.join(' / ')}`;
}

// scoring.py writes the 4-tier category into run.json. If the UI's own 4-tier
// reading of the same total disagrees, say so rather than silently pick one.
function tierMismatch() {
  const rc = state.run.risk_category;
  if (!rc) return '';
  const ui = tierIn(state.run.agatston_total, 4);
  return ui === String(rc).toUpperCase() ? '' : ` · ⚠ run.json says ${rc}`;
}

// Total, tier label, tier bar and note: the same block on every tab.
// prefix is the id prefix: 'a', 'i', 'cs' or 'v' (Anatomy has no bar).
function paintScore(prefix, hint = '') {
  const total = state.run.agatston_total, tier = tierOf(total);
  document.getElementById(`${prefix}-total`).textContent = total.toFixed(1);
  const tierEl = document.getElementById(`${prefix}-tier`);
  tierEl.textContent = tier;
  tierEl.style.color = tierColor(tier);
  document.getElementById(`${prefix}-tiernote`).textContent = tierNote(total) + tierMismatch() + hint;

  const bar = document.getElementById(`${prefix}-tierbar`);
  if (!bar) return;
  const pct = v => Math.min(100, v / TIER_SCALE_MAX * 100);
  bar.innerHTML =
    `<i style="width:${pct(total)}%;background:${tierColor(tier)}"></i>` +
    TIER_SCHEMES[state.tiers].slice(1, -1).map(([max]) => `<u style="left:${pct(max)}%"></u>`).join('');
}

function exSummary() {
  const ex = excluded();
  if (!ex.length) return 'no withheld components';
  const area = ex.reduce((a, l) => a + l.area_mm2, 0);
  const would = ex.reduce((a, l) => a + l.area_mm2 * l.density_weight, 0);
  return `${ex.length} withheld · ${area.toFixed(2)} mm² · below 1.0 mm² · ` +
    `would add ${would.toFixed(1)} if admitted`;
}

async function loadSidebar() {
  const toggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');
  if (toggle && sidebar) {
    toggle.onclick = () => sidebar.classList.toggle('collapsed');
  }

  try {
    const res = await fetch('/studies');
    if (!res.ok) return;
    const studies = await res.json();
    
    const list = document.getElementById('sidebar-list');
    if (!list) return;
    list.innerHTML = '';
    
    // Study names are typed by users at upload, so they are set as text,
    // never as HTML, and encoded in the link.
    for (const s of studies) {
      const el = document.createElement('div');
      el.className = 'study-item' + (s.id === STUDY && s.model === MODEL ? ' active' : '');
      const title = document.createElement('span');
      title.className = 'study-item-title';
      title.textContent = s.id;
      const sub = document.createElement('span');
      sub.className = 'study-item-subtitle';
      sub.title = s.model;
      sub.textContent = s.model.length > 20 ? s.model.substring(0, 18) + '...' : s.model;
      el.append(title, sub);
      el.onclick = () => { window.location.href = studyUrl(s.id, s.model); };
      list.appendChild(el);
    }
  } catch (e) {
    console.error("Could not load sidebar", e);
  }
}

// ── run pipeline form (sidebar) ──────────────────────────────────────────
let rawPatients = [];   // [{id, path}] from /raw_patients, i.e. data/raw/*

async function initRunForm() {
  const modelSelect = document.getElementById('run-model');
  const search = document.getElementById('run-patient-search');
  const select = document.getElementById('run-patient');

  search.oninput = () => renderPatients(search.value.toLowerCase());
  select.onchange = () => {
    const opt = select.options[select.selectedIndex];
    if (!opt.value) return;
    document.getElementById('run-path').value = opt.value;
    document.getElementById('run-name').value = opt.dataset.id;   // auto-populate the name too
  };
  document.getElementById('run-btn').onclick = startRun;

  try {
    const res = await fetch('/models');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const models = await res.json();
    modelSelect.innerHTML = '';
    for (const m of models) modelSelect.add(new Option(m.id, m.id));
  } catch (e) {
    console.error('Could not fetch models for run form', e);
    modelSelect.innerHTML = '<option value="">Error loading models</option>';
  }

  await refreshPatients();
}

// Reload the patient list; optionally select one (e.g. a study just uploaded).
async function refreshPatients(selectId) {
  try {
    const res = await fetch('/raw_patients');
    if (res.ok) rawPatients = await res.json();
  } catch (e) {
    console.error('Could not fetch raw patients', e);
  }
  const search = document.getElementById('run-patient-search');
  if (selectId) search.value = '';
  renderPatients(search.value.toLowerCase());

  if (selectId) {
    const select = document.getElementById('run-patient');
    const opt = [...select.options].find(o => o.dataset.id === selectId);
    if (opt) { select.value = opt.value; select.onchange(); }
  }
}

function renderPatients(term) {
  const select = document.getElementById('run-patient');
  select.innerHTML = '<option value="">Select a raw patient...</option>';
  rawPatients
    .filter(p => !term || p.id.toLowerCase().includes(term))
    .forEach(p => {
      const opt = new Option(p.id, p.path);
      opt.dataset.id = p.id;
      select.add(opt);
    });
}

async function startRun() {
  const path = document.getElementById('run-path').value.trim();
  const model = document.getElementById('run-model').value;
  const runName = document.getElementById('run-name').value.trim();
  if (!path) { runMessage('Choose a patient, or enter the path to a DICOM folder.', 'error'); return; }
  if (!model) { runMessage('Choose a model.', 'error'); return; }

  runMessage('');
  setRunFormBusy(true);
  showProgress(0, 'starting');
  try {
    const payload = { input_path: path, model_id: model };
    if (runName) payload.study_id = runName;
    const res = await fetch('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await errorText(res, 'Could not start the job'));
    const { job_id } = await res.json();
    pollJob(job_id, path, model, runName);
  } catch (e) {
    document.getElementById('run-progress-container').hidden = true;
    runMessage(e.message, 'error');
    setRunFormBusy(false);
  }
}

function pollJob(jobId, path, model, runName) {
  let failures = 0;
  const timer = setInterval(async () => {
    try {
      const res = await fetch(`/jobs/${jobId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const job = await res.json();
      failures = 0;
      showProgress(job.pct, job.stage);

      if (job.status === 'done') {
        clearInterval(timer);
        // Same rule as server.py start_job: with no name, the study is named
        // after the input's parent folder.
        const parts = path.split(/[\/]/).filter(Boolean);
        const studyId = runName || (parts.length > 1 ? parts[parts.length - 2] : parts[0]);
        setTimeout(() => { window.location.href = studyUrl(studyId, model); }, 500);
      } else if (job.status === 'failed') {
        clearInterval(timer);
        jobFailed(job.error);
      }
    } catch (e) {
      // A restarted server forgets its jobs (404), a stopped one refuses the
      // connection: stop polling instead of spinning forever.
      console.error('Polling error', e);
      if (++failures >= 10) { clearInterval(timer); jobFailed('Lost connection to the server.'); }
    }
  }, 500);
}

function showProgress(pct, stage) {
  document.getElementById('run-progress-container').hidden = false;
  const fill = document.getElementById('run-progress-fill');
  const p = Math.round(pct * 100);
  fill.style.backgroundColor = '';
  fill.style.width = `${p}%`;
  document.getElementById('run-progress-text').textContent = `${p}% - ${stage}`;
}

function jobFailed(error) {
  document.getElementById('run-progress-fill').style.backgroundColor = 'var(--md-sys-color-error)';
  document.getElementById('run-progress-text').textContent = 'Failed';
  // The server sends a traceback or "Process exited with code N"; the last
  // line names the problem, the full text is kept in the tooltip.
  const full = String(error || 'Unknown error').trim();
  runMessage(full.split('\n').pop(), 'error');
  document.getElementById('run-msg').title = full;
  setRunFormBusy(false);
}

function runMessage(text, kind = 'info') {
  const el = document.getElementById('run-msg');
  el.hidden = !text;
  el.textContent = text;
  el.title = '';
  el.className = `run-msg ${kind}`;
}

function setRunFormBusy(busy) {
  document.querySelectorAll('.sidebar-run input, .sidebar-run select, .sidebar-run button')
    .forEach(el => { el.disabled = busy; });
}

boot();
