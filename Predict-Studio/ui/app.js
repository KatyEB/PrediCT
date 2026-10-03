/* PrediCT Studio — frontend.
 *
 * Served by src/backend/server.py (FastAPI): ui/ at /ui. Nothing works until
 * the user signs in; every request then carries the session cookie and the
 * server answers only with that user's own data.
 *
 *     python -m src.backend.server
 *     http://127.0.0.1:8001/ui/?study=172&model=a1-roi
 *
 * Reads one of the user's result folders through /files/<study>/<model>/...
 * and renders it. With no study in the URL a start screen is shown instead.
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
// Result files come through the server's /files route, which resolves them
// inside the signed-in user's own folder (never a public path).
const BASE = `/files/${encodeURIComponent(STUDY)}/${encodeURIComponent(MODEL)}`;
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
  initAuthForm();
  if (!await whoAmI()) return;       // signed out: the sign-in screen is showing

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

// ── session ──────────────────────────────────────────────────────────────
let ME = null;   // { username, is_admin } of the signed-in user

// Every server call goes through api(). A 401 means the session ended (signed
// out in another tab, expired, or access revoked): show the sign-in screen.
async function api(url, opts) {
  const res = await fetch(url, opts);
  if (res.status === 401) {
    showAuth('Your session has ended. Please sign in again.');
    throw new Error('Not signed in.');
  }
  return res;
}

async function whoAmI() {
  try {
    const res = await fetch('/auth/me');
    if (res.status === 401) { showAuth(); return null; }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    ME = await res.json();
  } catch (e) {
    showAuth(`The server could not be reached (${e.message}).`);
    return null;
  }
  document.body.classList.remove('checking');
  document.body.classList.toggle('is-admin', ME.is_admin);
  document.getElementById('me-name').textContent = ME.username;
  document.getElementById('logout-btn').onclick = async () => {
    try { await fetch('/auth/logout', { method: 'POST' }); } catch { /* signed out anyway */ }
    window.location.href = '?';
  };
  return ME;
}

function showAuth(message = '') {
  clearTimeout(jobsTimer);           // nothing to watch while signed out
  document.body.classList.remove('checking');
  document.body.classList.add('signed-out');
  document.getElementById('auth').hidden = false;
  const err = document.getElementById('auth-error');
  err.hidden = !message;
  err.textContent = message;
  document.getElementById('auth-username').focus();
}

/* Sign in, or create an account. Both need the access token from the
   administrator (the admin account itself does not). On success the page
   reloads and starts as the signed-in user, on the same URL. */
function initAuthForm() {
  const form = document.getElementById('auth-form');
  const submit = document.getElementById('auth-submit');
  const err = document.getElementById('auth-error');
  let mode = 'login';
  const setMode = m => {
    mode = m;
    document.querySelectorAll('#auth [data-auth-mode]').forEach(b =>
      b.classList.toggle('on', b.dataset.authMode === m));
    submit.textContent = m === 'login' ? 'Sign in' : 'Create account';
    document.getElementById('auth-password').autocomplete = m === 'login' ? 'current-password' : 'new-password';
    document.getElementById('auth-hint').textContent = m === 'login'
      ? 'Given to you by the administrator.'
      : 'Given to you by the administrator. Passwords need at least 8 characters.';
    err.hidden = true;
  };
  document.querySelectorAll('#auth [data-auth-mode]').forEach(b => { b.onclick = () => setMode(b.dataset.authMode); });

  form.onsubmit = async e => {
    e.preventDefault();
    submit.disabled = true;
    err.hidden = true;
    try {
      const res = await fetch(`/auth/${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: document.getElementById('auth-username').value,
          password: document.getElementById('auth-password').value,
          access_token: document.getElementById('auth-token').value,
        }),
      });
      if (!res.ok) throw new Error(await errorText(res, mode === 'login' ? 'Sign-in failed' : 'Could not create the account'));
      window.location.reload();
    } catch (ex) {
      err.textContent = ex.message;
      err.hidden = false;
      submit.disabled = false;
    }
  };
}

async function getJson(u) { const r = await api(u); if (!r.ok) throw new Error(`${u} → HTTP ${r.status}`); return r.json(); }
async function getText(u) { const r = await api(u); if (!r.ok) throw new Error(`${u} → HTTP ${r.status}`); return r.text(); }
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

/* Upload a folder to data/raw/<id> (server.py). Progress is shown in the
   sidebar card; on success the new scan is picked in Run Pipeline. Nothing
   reloads and nothing runs until the user presses Run. */
let uploading = false;
async function handleUpload(files) {
  if (uploading) { toast('An upload is already in progress.'); return; }
  const name = await askDialog({
    title: 'Upload study',
    message: `${files.length} file(s) selected. Name this study, or leave it blank ` +
      'to generate a name from the series.',
    input: '',
    okLabel: 'Upload',
  });
  if (name === null) return;   // cancelled: nothing was sent

  setUploading(true, `Uploading ${name ? `"${name}"` : 'scan'} · ${files.length} files`);
  try {
    const fd = new FormData();
    if (name) fd.append('custom_name', name);
    for (const f of files) fd.append('files', f);

    const res = await postWithProgress('/studies', fd, uploadProgress);
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
        await api(`/studies/clean/${data.temp_id}`, { method: 'DELETE' });
        toast('Upload cancelled. Nothing was saved.');
        return;
      }
      uploadProgress(null, 'Removing unusable files…');
      const q = name ? `?custom_name=${encodeURIComponent(name)}` : '';
      const cleanRes = await api(`/studies/clean/${data.temp_id}${q}`, { method: 'POST' });
      if (!cleanRes.ok) throw new Error(await errorText(cleanRes, 'Cleaning failed'));
      data = await cleanRes.json();
    }

    await refreshPatients(data.study_id);
    toast(`Uploaded "${data.study_id}". It is selected in Run Pipeline: choose a model and press Run.`, 'ok');
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    setUploading(false);
  }
}

// fetch() cannot report upload progress; XMLHttpRequest can. Resolves to a
// small fetch-like response so errorText() works on it unchanged.
function postWithProgress(url, body, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);
    xhr.upload.onprogress = e => { if (e.lengthComputable) onProgress(e.loaded, e.total); };
    xhr.upload.onload = () => onProgress(null, 'Checking files…');   // all bytes sent
    xhr.onload = () => {
      if (xhr.status === 401) showAuth('Your session has ended. Please sign in again.');
      resolve({
        ok: xhr.status >= 200 && xhr.status < 300,
        status: xhr.status,
        json: async () => JSON.parse(xhr.responseText),
      });
    };
    xhr.onerror = () => reject(new Error('Upload failed: the server could not be reached.'));
    xhr.send(body);
  });
}

// Upload card: a filling bar while bytes are sent (loaded, total), an
// animated one while the server works (loaded === null, label).
function uploadProgress(loaded, totalOrLabel) {
  const card = document.getElementById('upload-card');
  const fill = document.getElementById('upload-fill');
  const text = document.getElementById('upload-text');
  card.classList.toggle('busy', loaded === null);
  if (loaded === null) { fill.style.width = ''; text.textContent = totalOrLabel; return; }
  const total = totalOrLabel;
  const mb = n => (n / 1048576).toFixed(n < 10485760 ? 1 : 0);
  const pct = total ? Math.round(loaded / total * 100) : 0;
  fill.style.width = pct + '%';
  text.textContent = `${pct}% · ${mb(loaded)} of ${mb(total)} MB`;
}

function setUploading(on, title = '') {
  uploading = on;
  const btn = document.getElementById('upload-study-btn');
  btn.disabled = on;
  btn.classList.toggle('spin', on);
  btn.textContent = on ? 'progress_activity' : 'upload';
  document.getElementById('welcome-upload').disabled = on;
  document.getElementById('upload-card').hidden = !on;
  document.getElementById('upload-title').textContent = title;
  if (on) uploadProgress(0, 0);
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

/* The one modal the UI uses. Resolves to null when cancelled (button or Esc);
   otherwise to the typed text when `input` is given, to an array of booleans
   when `checks` is given, else to true. A check with `all: true` stands for
   every other check: ticking it ticks and locks them. */
function askDialog({ title, message, input = null, checks = null, okLabel = 'OK', danger = false }) {
  const dlg = document.getElementById('dlg');
  const field = document.getElementById('dlg-input');
  const box = document.getElementById('dlg-checks');
  const ok = document.getElementById('dlg-ok');
  document.getElementById('dlg-title').textContent = title;
  document.getElementById('dlg-msg').textContent = message;
  document.getElementById('dlg-cancel').onclick = () => dlg.close();
  ok.textContent = okLabel;
  ok.classList.toggle('danger', danger);
  ok.disabled = false;
  field.hidden = input === null;
  field.value = input || '';

  box.hidden = !checks;
  box.replaceChildren();
  const boxes = (checks || []).map(c => {
    const label = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = Boolean(c.checked);
    label.append(cb, ' ' + c.label);
    box.append(label);
    return cb;
  });
  const allBox = boxes[(checks || []).findIndex(c => c.all)];
  const sync = () => {
    for (const b of boxes) {
      if (!allBox || b === allBox) continue;
      b.disabled = allBox.checked;
      if (allBox.checked) b.checked = true;
    }
    ok.disabled = Boolean(checks) && !boxes.some(b => b.checked);
  };
  boxes.forEach(b => { b.onchange = sync; });
  if (checks) sync();

  dlg.returnValue = '';
  dlg.showModal();
  if (input !== null) field.focus();
  return new Promise(resolve => {
    dlg.onclose = () => {
      if (dlg.returnValue !== 'ok') resolve(null);
      else if (input !== null) resolve(field.value.trim());
      else resolve(checks ? boxes.map(b => b.checked) : true);
    };
  });
}

// Short non-blocking message at the bottom of the screen. Click to dismiss.
// action: optional { label, href } shown as a link, e.g. "Open".
let toastTimer = null;
function toast(msg, kind = 'info', action = null) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  if (action) {
    const a = document.createElement('a');
    a.className = 'toast-action';
    a.href = action.href;
    a.textContent = action.label;
    el.append(a);
  }
  el.className = `toast ${kind}`;
  el.hidden = false;
  el.onclick = () => { el.hidden = true; };
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, action ? 12000 : kind === 'error' ? 8000 : 4000);
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
  paintParams();
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

// Pipeline parameters, grouped. Every value is read from run.json, so the
// panel describes the run that produced these numbers, not today's settings.
// A field an older run.json does not have is shown as '—', never guessed.
function paramGroups() {
  const r = state.run, cov = outputType() === 'coverage';
  const known = v => (v == null ? '—' : v);
  const overridden = r.crop_default != null && r.cropped !== r.crop_default;
  const crop = !r.cropped ? 'none · full field of view'
    : `heart${r.crop_margin_mm != null ? ` +${r.crop_margin_mm} mm` : ''} · ` +
      `TotalSegmentator ${known(r.locator_version)}`;
  return [
    ['Model', [
      ['model', r.model_id],
      ['output', cov ? 'coverage (fractional)' : 'binary mask'],
      ['threshold', threshold().toFixed(2) + (cov ? ' · delineation only' : '')],
      ['checkpoint', r.sha256 ? String(r.sha256).slice(0, 12) : '—', r.sha256],
    ]],
    ['Pre-processing', [
      ['crop', crop + (overridden ? ' · not the model default' : ''), null, overridden],
      ['HU window', `${r.hu_window[0]} – ${r.hu_window[1]}`],
      ['voxel spacing', r.spacing.map(v => v.toFixed(2)).join(' × ') + ' mm'],
      ['orientation', 'RAS'],
    ]],
    ['Scoring', [
      ['min lesion', r.min_area_mm2 != null ? `${r.min_area_mm2.toFixed(1)} mm²` : '—'],
      ['area', cov ? 'Σ coverage × pixel area' : 'voxel count × pixel area'],
      ['3D linking', `in-plane overlap · max gap ${known(r.max_gap_slices)} slice(s)`],
    ]],
    ['Run', [
      ['date', r.date ? new Date(r.date).toLocaleString(undefined,
        { dateStyle: 'medium', timeStyle: 'short' }) : '—'],
      ['volume', r.shape ? r.shape.join(' × ') + ' (z × y × x)' : '—'],
    ]],
  ];
}

function paintParams() {
  const box = document.getElementById('a-params');
  box.replaceChildren();
  for (const [group, rows] of paramGroups()) {
    box.append(el('div', 'a-params-group', group));
    for (const [key, value, full, warn] of rows) {
      const v = el('span', 'v' + (warn ? ' warn' : ''), value);
      if (full) v.title = full;
      box.append(el('span', 'k', key), v);
    }
  }
}

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

// ══ SIDEBAR: studies, uploads, runs ══════════════════════════════════════
// An element with its text set safely: study names are typed by users at
// upload and must never be parsed as HTML.
function el(tag, cls = '', text = '') {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text) e.textContent = text;
  return e;
}

let studies = [];       // [{id, models: [...]}] from /studies, grouped by study
let rawPatients = [];   // [{id, path, uploaded}] from /raw_patients, newest first
let models = [];        // [{id, name, crop}] from /models

async function loadSidebar() {
  const sidebar = document.getElementById('sidebar');
  document.getElementById('sidebar-toggle').onclick = () => sidebar.classList.toggle('collapsed');
  document.getElementById('study-filter').oninput = renderStudies;
  try {
    const res = await api('/studies');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const byId = new Map();
    for (const s of await res.json()) {
      if (!byId.has(s.id)) byId.set(s.id, []);
      byId.get(s.id).push(s.model);
    }
    studies = [...byId].map(([id, ms]) => ({ id, models: ms }));
  } catch (e) {
    console.error('Could not load studies', e);
  }
  renderStudies();
}

// One row per study with its results as chips; the filter matches either.
function renderStudies() {
  const list = document.getElementById('sidebar-list');
  const term = document.getElementById('study-filter').value.trim().toLowerCase();
  const shown = studies.filter(s => !term || s.id.toLowerCase().includes(term) ||
    s.models.some(m => m.toLowerCase().includes(term)));
  list.replaceChildren();
  for (const s of shown) {
    const row = el('div', 'study-item' + (s.id === STUDY ? ' active' : ''));
    const head = el('div', 'study-item-head');
    const title = el('a', 'study-item-title', s.id);
    title.href = studyUrl(s.id, s.id === STUDY && s.models.includes(MODEL) ? MODEL : s.models[0]);
    const del = el('button', 'study-del material-symbols-outlined', 'delete');
    del.title = `Delete results of ${s.id}, or the whole study`;
    del.onclick = () => deleteStudy(s.id);
    head.append(title, del);
    const chips = el('div', 'study-item-models');
    for (const m of s.models) {
      const chip = el('a', 'model-chip' + (s.id === STUDY && m === MODEL ? ' on' : ''), m);
      chip.href = studyUrl(s.id, m);
      chips.append(chip);
    }
    row.append(head, chips);
    list.append(row);
  }
  if (!shown.length) list.append(el('div', 'sidebar-empty', term ? 'No study matches.' : 'No results yet.'));
}

/* Delete some results of a study, or everything it owns (DELETE /studies).
   The same dialog serves a study in the list and an uploaded scan that was
   never run (then "everything" is the only choice). */
async function deleteStudy(studyId) {
  const results = (studies.find(s => s.id === studyId) || { models: [] }).models;
  const checks = results.map(m => ({ label: `Result ${m}`, checked: studyId === STUDY && m === MODEL }));
  checks.push({ label: 'Everything: the uploaded scan, its prep cache and all results',
                checked: !results.length, all: true });
  const picks = await askDialog({
    title: `Delete from "${studyId}"`,
    message: 'Deleted files cannot be recovered from the app.',
    checks, okLabel: 'Delete', danger: true,
  });
  if (!picks) return;

  const everything = picks[picks.length - 1];
  const chosen = results.filter((m, i) => picks[i]);
  const q = new URLSearchParams(everything ? { all: 'true' } : chosen.map(m => ['model', m]));
  const res = await api(`/studies/${encodeURIComponent(studyId)}?${q}`, { method: 'DELETE' });
  if (!res.ok) { toast(await errorText(res, 'Delete failed'), 'error'); return; }

  // The result on screen no longer exists: back to the start screen.
  if (studyId === STUDY && (everything || chosen.includes(MODEL))) { window.location.href = '?'; return; }
  toast(everything ? `Deleted everything for "${studyId}".`
    : `Deleted ${chosen.join(', ')} from "${studyId}".`, 'ok');
  await Promise.all([loadSidebar(), refreshPatients()]);
}

// ── run pipeline form ────────────────────────────────────────────────────
async function initRunForm() {
  const modelSelect = document.getElementById('run-model');
  const crop = document.getElementById('run-crop');
  modelSelect.onchange = () => {
    const m = models.find(x => x.id === modelSelect.value);
    if (m) crop.checked = m.crop;   // each model starts at its own default
    cropHint();
  };
  crop.onchange = cropHint;
  document.getElementById('run-btn').onclick = startRun;
  initPatientPicker();

  try {
    const res = await api('/models');
    if (!res.ok) throw new Error(await errorText(res, 'Could not load models'));
    models = await res.json();
    modelSelect.replaceChildren(...models.map(m => {
      const o = new Option(m.id, m.id);
      o.title = m.name;
      return o;
    }));
    modelSelect.onchange();
  } catch (e) {
    console.error(e);
    modelSelect.innerHTML = '<option value="">Error loading models</option>';
    runMessage(e.message, 'error');
  }

  await Promise.all([refreshPatients(), refreshJobs()]);
}

// What the crop box means for the chosen model; an override is flagged.
function cropHint() {
  const m = models.find(x => x.id === document.getElementById('run-model').value);
  const hint = document.getElementById('run-crop-hint');
  if (!m) { hint.textContent = ''; return; }
  const differs = document.getElementById('run-crop').checked !== m.crop;
  hint.textContent = `model default: ${m.crop ? 'on' : 'off'}` +
    (differs ? ' · differs from how this model was trained' : '');
  hint.classList.toggle('warn', differs);
}

// ── patient picker: one field to search and pick an uploaded scan ───────
let picked = null;       // id of the chosen scan, or null
let pickerItems = [];    // the scans currently listed
let pickerIndex = -1;    // the highlighted one

// Reload the list of uploaded scans; keep (or make) a pick if it still exists.
async function refreshPatients(selectId) {
  try {
    const res = await api('/raw_patients');
    if (res.ok) rawPatients = await res.json();
  } catch (e) {
    console.error('Could not fetch uploaded scans', e);
  }
  const p = rawPatients.find(x => x.id === (selectId || picked));
  if (p) pickPatient(p);
  else if (picked) clearPatient();
}

function pickPatient(p) {
  picked = p.id;
  document.getElementById('run-patient').value = p.id;
  document.getElementById('run-path').value = p.path || '';   // shown to the admin only
  document.getElementById('run-name').value = p.id;   // auto-populate the name too
  document.getElementById('run-patient-clear').hidden = false;
  closePicker();
}

function clearPatient() {
  picked = null;
  for (const id of ['run-patient', 'run-path', 'run-name']) document.getElementById(id).value = '';
  document.getElementById('run-patient-clear').hidden = true;
}

function initPatientPicker() {
  const input = document.getElementById('run-patient');
  const list = document.getElementById('run-patient-list');
  input.onfocus = input.onclick = openPicker;
  input.onblur = closePicker;
  input.oninput = () => {
    // Editing the field drops the previous pick and the path it had filled.
    if (picked) {
      picked = null;
      document.getElementById('run-path').value = '';
      document.getElementById('run-name').value = '';
    }
    document.getElementById('run-patient-clear').hidden = !input.value;
    openPicker();
  };
  input.onkeydown = e => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (list.hidden) openPicker();
      const last = pickerItems.length - 1;
      pickerIndex = Math.max(0, Math.min(last, pickerIndex + (e.key === 'ArrowDown' ? 1 : -1)));
      highlightPick();
    } else if (e.key === 'Enter' && !list.hidden && pickerItems[pickerIndex]) {
      e.preventDefault();
      pickPatient(pickerItems[pickerIndex]);
    } else if (e.key === 'Escape') {
      closePicker();
    }
  };
  // Pressing on the list must not blur the field before the click lands.
  list.onmousedown = e => e.preventDefault();
  document.getElementById('run-patient-clear').onclick = () => { clearPatient(); input.focus(); };
}

function openPicker() {
  const input = document.getElementById('run-patient');
  const list = document.getElementById('run-patient-list');
  // While a scan is picked its name fills the field: list everything then.
  const term = (picked ? '' : input.value).trim().toLowerCase();
  pickerItems = rawPatients.filter(p => !term || p.id.toLowerCase().includes(term));
  pickerIndex = pickerItems.length ? Math.max(0, pickerItems.findIndex(p => p.id === picked)) : -1;

  list.replaceChildren();
  if (!rawPatients.length) list.append(el('li', 'combo-empty', 'No uploaded scans yet. Use the upload button above.'));
  else if (!pickerItems.length) list.append(el('li', 'combo-empty', 'No scan matches.'));
  pickerItems.forEach((p, i) => {
    const li = el('li', 'combo-option');
    li.id = `run-patient-opt-${i}`;
    li.setAttribute('role', 'option');
    const name = el('span', 'combo-name');
    const at = term ? p.id.toLowerCase().indexOf(term) : -1;
    if (at >= 0) {   // mark the typed part
      name.append(p.id.slice(0, at), el('mark', '', p.id.slice(at, at + term.length)),
                  p.id.slice(at + term.length));
    } else {
      name.textContent = p.id;
    }
    const when = el('span', 'combo-when', ago(p.uploaded));
    when.title = `uploaded ${new Date(p.uploaded).toLocaleString()}`;
    const del = el('button', 'combo-del material-symbols-outlined', 'delete');
    del.type = 'button';
    del.title = `Delete ${p.id}`;
    del.onclick = e => { e.stopPropagation(); closePicker(); deleteStudy(p.id); };
    li.append(name, when, del);
    li.onclick = () => pickPatient(p);
    list.append(li);
  });
  list.hidden = false;
  input.setAttribute('aria-expanded', 'true');
  highlightPick();
}

function closePicker() {
  document.getElementById('run-patient-list').hidden = true;
  document.getElementById('run-patient').setAttribute('aria-expanded', 'false');
  pickerIndex = -1;
}

function highlightPick() {
  const list = document.getElementById('run-patient-list');
  const options = list.querySelectorAll('[role="option"]');
  options.forEach((li, i) => {
    li.classList.toggle('active', i === pickerIndex);
    li.setAttribute('aria-selected', String(i === pickerIndex));
  });
  const cur = options[pickerIndex];
  document.getElementById('run-patient').setAttribute('aria-activedescendant', cur ? cur.id : '');
  if (cur) keepVisible(list, cur);
}

function ago(iso) {
  const s = (Date.now() - new Date(iso)) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

// ── runs: they live on the server; this page only watches them ─────────
let jobStatus = null;   // job_id -> status at the previous look; null before the first
let jobsTimer = null;

async function startRun() {
  // Everyone runs one of their own uploads (`scan`); only the admin may give
  // a folder on the server instead. The server enforces the same rule.
  const path = ME.is_admin ? document.getElementById('run-path').value.trim() : '';
  const model = document.getElementById('run-model').value;
  const runName = document.getElementById('run-name').value.trim();
  if (!picked && !path) {
    runMessage(ME.is_admin ? 'Choose an uploaded scan, or enter a folder on the server.'
                           : 'Choose one of your uploaded scans.', 'error');
    return;
  }
  if (!model) { runMessage('Choose a model.', 'error'); return; }

  runMessage('');
  try {
    const payload = { model_id: model, crop: document.getElementById('run-crop').checked };
    if (picked) payload.scan = picked; else payload.input_path = path;
    if (runName) payload.study_id = runName;
    const res = await api('/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await errorText(res, 'Could not start the run'));
    await refreshJobs();
  } catch (e) {
    runMessage(e.message, 'error');
  }
}

/* Read GET /jobs and show the runs. Polls once a second only while a run is
   active. A finished run is announced with an Open link; the page never
   navigates by itself, so whatever the user is doing is not interrupted. */
async function refreshJobs() {
  clearTimeout(jobsTimer);
  let jobs;
  try {
    const res = await api('/jobs');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    jobs = await res.json();
  } catch (e) {
    const wasRunning = jobStatus && Object.values(jobStatus).includes('running');
    const signedOut = document.body.classList.contains('signed-out');
    if (!signedOut) console.error('Could not read runs', e);
    if (wasRunning && !signedOut) jobsTimer = setTimeout(refreshJobs, 5000);   // server busy or restarting
    return;
  }

  if (jobStatus) {
    for (const j of jobs) {
      if (jobStatus[j.job_id] !== 'running') continue;
      const name = `${j.study_id} · ${j.model_id}`;
      if (j.status === 'done') {
        toast(`Finished ${name}.`, 'ok', { label: 'Open', href: studyUrl(j.study_id, j.model_id) });
        loadSidebar();
      } else if (j.status === 'failed') {
        toast(`Run failed: ${name}.`, 'error');
      }
    }
  }
  jobStatus = Object.fromEntries(jobs.map(j => [j.job_id, j.status]));
  renderJobs(jobs);

  const running = jobs.some(j => j.status === 'running');
  const btn = document.getElementById('run-btn');
  btn.disabled = running;
  btn.textContent = running ? 'Run in progress…' : 'Run';
  if (running) jobsTimer = setTimeout(refreshJobs, 1000);
}

// The two most recent runs of this server session.
function renderJobs(jobs) {
  const box = document.getElementById('run-jobs');
  box.replaceChildren();
  for (const j of jobs.slice(0, 2)) {
    const row = el('div', `job ${j.status}`);
    const head = el('div', 'job-head');
    const pct = Math.round(j.pct * 100);
    head.append(el('span', 'job-name', `${j.study_id} · ${j.model_id}`),
                el('span', 'job-state', j.status === 'running' ? `${pct}% · ${j.stage}` : j.status));
    row.append(head);
    if (j.status === 'running') {
      const bar = el('div', 'run-progress-bar');
      const fill = el('div', 'run-progress-fill');
      fill.style.width = `${pct}%`;
      bar.append(fill);
      row.append(bar);
    } else if (j.status === 'done') {
      const open = el('a', 'job-open', 'Open result');
      open.href = studyUrl(j.study_id, j.model_id);
      row.append(open);
    } else {
      const err = String(j.error || 'Unknown error').trim();
      const line = el('span', 'job-err', err.split('\n').pop());
      line.title = err;
      row.append(line);
    }
    box.append(row);
  }
}

function runMessage(text, kind = 'info') {
  const msg = document.getElementById('run-msg');
  msg.hidden = !text;
  msg.textContent = text;
  msg.className = `run-msg ${kind}`;
}

boot();
