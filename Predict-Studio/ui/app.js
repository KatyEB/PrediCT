/* PrediCT Studio — phase 1.
 *
 * Reads an already-produced output folder and renders it. No backend, no
 * inference, no job launching. Serve the repo root and open /ui/.
 *
 *     python -m http.server 8000
 *     http://localhost:8000/ui/?study=172&model=a1-roi
 *
 * Later, FastAPI mounts data/ at /data and ui/ at /, so these URLs do not
 * change.
 *
 * One state object, one render(). Every handler mutates state then calls
 * render(). There is no other update path.
 */

// ── configuration ────────────────────────────────────────────────────────
const qs = new URLSearchParams(location.search);
const STUDY = qs.get('study') || '172';
const MODEL = qs.get('model') || 'a1-roi';
const BASE = `/data/out/${STUDY}/${MODEL}`;

// render.py flips vertically (flipud) because the direction cosines are
// diag(-1,-1,1): increasing array row is increasing anterior, so row 0 at the
// top would show the slice posterior-up. lesions.csv stays in unflipped array
// coordinates, so every lesion coordinate drawn here must be flipped to match.
// If fliplr is later added to render.py for radiological left/right, flip
// FLIP_X to true here in the same commit — the two must always agree.
const FLIP_Y = true;
const FLIP_X = true;

const TIER_SCALE_MAX = 1400;   // full width of the tier bar; ticks at 100 / 400

const SOFT_NOTE =
  'Coverage is not binary. A voxel at 0.35 contributes 0.35 of its area and ' +
  'is drawn as grain, never as an edge. The 0.10 threshold only decides where ' +
  'one component stops and the next begins — it does not gate the score.';

let ACCENT_COLOR = '#C98B2E';

// ── state ────────────────────────────────────────────────────────────────
const state = {
  dir: 2,          // 1 argument, 2 instrument
  view: 2,         // 1 original, 2 prediction, 3 calcium only
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
    const el = document.getElementById('err');
    el.hidden = false;
    el.textContent =
      `Could not load ${BASE}\n\n${e.message}\n\n` +
      `Serve the repo root (python -m http.server) and open /ui/.\n` +
      `Check that the folder exists and contains run.json, slices.json, lesions.csv.`;
  }
}

async function getJson(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u} → HTTP ${r.status}`); return r.json(); }
async function getText(u) { const r = await fetch(u); if (!r.ok) throw new Error(`${u} → HTTP ${r.status}`); return r.text(); }
function loadImage(u) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error(`${u} → not found`));
    im.src = u;
  });
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

function tierOf(t) { return t === 0 ? 'ZERO' : t <= 100 ? 'MILD' : t <= 400 ? 'MODERATE' : 'SEVERE'; }

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

  const onWheel = e => { e.preventDefault(); step(e.deltaY > 0 ? 1 : -1); };
  document.getElementById('i-viewport').addEventListener('wheel', onWheel, { passive: false });
  document.getElementById('i-track').addEventListener('wheel', onWheel, { passive: false });
  document.getElementById('a-pane').addEventListener('wheel', onWheel, { passive: false });

  document.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { step(1); e.preventDefault(); }
    if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
    if (e.key === 'Escape') { state.sel = null; state.sel3d = null; render(); }
    if (e.key === '1' || e.key === '2' || e.key === '3') {
      document.querySelector(`#tabs button[data-view="${e.key}"]`).click();
    }
  });

  window.addEventListener('resize', () => render());
}

// ── render ───────────────────────────────────────────────────────────────
function render() {
  document.getElementById('d1').classList.toggle('on', state.dir === 1);
  document.getElementById('d2').classList.toggle('on', state.dir === 2);
  document.querySelectorAll('#tabs button[data-dir]').forEach(b =>
    b.classList.toggle('on', Number(b.dataset.dir) === state.dir));
  document.querySelectorAll('#tabs button[data-view]').forEach(b =>
    b.classList.toggle('on', Number(b.dataset.view) === state.view));
  document.getElementById('tabs-study').textContent =
    `study ${STUDY} · ${MODEL} · ${state.slices.length} slices`;

  if (state.dir === 1) renderArgument(); else renderInstrument();
  preload(state.slice);
}

function preload(i) {
  for (let d = -3; d <= 3; d++) {
    const j = i + d;
    if (j >= 0 && j < state.slices.length) { new Image().src = ctUrl(j); new Image().src = maskUrl(j); }
  }
}

/* Fill an image pane: CT underneath, mask on top, selection ring on the canvas.
   The pane keeps the volume's aspect ratio inside whatever box CSS gives it. */
function paintPane(paneId, boxW, boxH) {
  const pane = document.getElementById(paneId);
  const ct = pane.querySelector('.pane-ct');
  const mask = pane.querySelector('.pane-mask');
  const cv = pane.querySelector('.pane-ring');

  const ar = state.imgW / state.imgH;
  let w = boxW, h = boxW / ar;
  if (h > boxH) { h = boxH; w = boxH * ar; }
  pane.style.width = Math.round(w) + 'px';
  pane.style.height = Math.round(h) + 'px';

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
  const drawRing = (l, dashed) => {
    const sx = cv.width / state.imgW, sy = cv.height / state.imgH;
    let x0 = l.bbox_x0, x1 = l.bbox_x1, y0 = l.bbox_y0, y1 = l.bbox_y1;
    if (FLIP_X) { const a = state.imgW - 1 - x1, b = state.imgW - 1 - x0; x0 = a; x1 = b; }
    if (FLIP_Y) { const a = state.imgH - 1 - y1, b = state.imgH - 1 - y0; y0 = a; y1 = b; }
    const m = 3;
    g.setLineDash(dashed ? [3, 3] : []);
    g.strokeStyle = dashed ? ACCENT_COLOR : '#4FA8C5';
    g.lineWidth = 1.5;
    g.strokeRect(x0 * sx - m, y0 * sy - m, (x1 - x0 + 1) * sx + 2 * m, (y1 - y0 + 1) * sy + 2 * m);
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
  const r = state.run, cs = calcSlices(), cnt = counted(), ex = excluded();
  const total = r.agatston_total, tier = tierOf(total);

  document.getElementById('a-prov1').textContent = prov(1);
  document.getElementById('a-prov2').textContent = prov(2);
  document.getElementById('a-prov3').textContent = prov(3);

  document.getElementById('a-total').textContent = total.toFixed(1);
  const tierEl = document.getElementById('a-tier');
  tierEl.textContent = tier;
  tierEl.style.color = tier === 'SEVERE' ? 'var(--red)' : tier === 'ZERO' ? 'var(--paper-muted)' : '#8A6A1F';
  document.getElementById('a-tiernote').textContent = tierNote(total);

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

  // exhibit strip — every calcium-bearing slice
  const strip = document.getElementById('a-strip');
  strip.innerHTML = '';
  cs.forEach(i => {
    const sc = sliceMeta(i).slice_score || 0;
    const b = document.createElement('button');
    b.className = i === state.slice ? 'on' : '';
    b.innerHTML =
      `<span class="pane${state.view === 1 ? ' view-1' : ''}">` +
      `<img class="pane-ct" src="${ctUrl(i)}" alt=""><img class="pane-mask" src="${maskUrl(i)}" alt="">` +
      `</span><span>${i} · ${sc.toFixed(1)}</span>`;
    b.onclick = () => goTo(i);
    strip.appendChild(b);
  });
  document.getElementById('a-stripnote').textContent = cs.length
    ? `${cs.length} exhibits · every calcium-bearing slice is shown · wheel or arrows step`
    : 'No exhibits: no slice carries a scored component. The claim is an absence, and the evidence is that all slices were examined.';

  // counted table
  const tb = document.getElementById('a-rows');
  tb.innerHTML = '';
  const orderedCnt = [...cnt].sort((a, b) =>
    a.lesion_3d_id - b.lesion_3d_id || a.slice_idx - b.slice_idx);
  let prevCnt = null;
  orderedCnt.forEach(l => {
    const first = l.lesion_3d_key !== prevCnt; prevCnt = l.lesion_3d_key;
    const p = meanCoverage(l);
    const tr = document.createElement('tr');
    tr.className = keyOf(l) === state.sel ? 'on' : l.slice_idx === state.slice ? 'cur' : '';
    tr.classList.toggle('g3-first', first);
    tr.classList.toggle('g3-on', l.lesion_3d_key === state.sel3d);
    tr.innerHTML =
      `<td class="g3">${first ? l.lesion_3d_key : ''}</td>` +
      `<td class="l">sl ${l.slice_idx}</td><td>${l.z_mm.toFixed(1)}</td>` +
      `<td>${l.area_mm2.toFixed(2)}</td><td>${l.peak_hu}</td><td>${l.density_weight}</td>` +
      `<td class="${p != null && p < 0.5 ? 'soft' : ''}">${p == null ? '—' : p.toFixed(2)}</td>` +
      `<td>${l.agatston.toFixed(1)}</td>`;
    tr.onclick = () => goTo(l.slice_idx, keyOf(l));
    tb.appendChild(tr);
  });
  document.getElementById('a-ncounted').textContent = cnt.length;

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

// ══ 02 INSTRUMENT ════════════════════════════════════════════════════════
function renderInstrument() {
  const r = state.run, cs = calcSlices(), ex = excluded();
  const total = r.agatston_total, tier = tierOf(total);
  const here = onSlice(state.slice);
  const meta = sliceMeta(state.slice);

  document.getElementById('i-prov1').textContent = prov(1);
  document.getElementById('i-prov2').textContent = prov(2);
  document.getElementById('i-prov3').textContent = prov(3);

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
    b.style.height = (sc > 0 ? Math.max(16, 16 + sc / maxScore * 26) : any ? 11 : 5) + 'px';
    b.style.background = sc > 0 ? 'var(--accent)' : any ? 'var(--muted-2)' : 'var(--rule-2)';
    b.style.borderBottomColor = sc > 0 ? 'var(--accent)' : any ? 'var(--muted-2)' : 'transparent';
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
  const sp = r.spacing || [0.37, 0.37, 3.0];
  const barPx = Math.round((10 / sp[0]) * (w / state.imgW));
  document.getElementById('i-caption').textContent = '';

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
  document.getElementById('i-total').textContent = total.toFixed(1);
  const tEl = document.getElementById('i-tier');
  tEl.textContent = tier;
  tEl.classList.toggle('severe', tier === 'SEVERE');
  document.getElementById('i-tierfill').style.width =
    Math.min(100, total / TIER_SCALE_MAX * 100) + '%';
  document.getElementById('i-tiernote').textContent = tierNote(total) + ' · ↑↓ step · 1/2/3 view · esc clear';
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
  if (t === 0) return 'tier Zero (0) · nothing to bound';
  if (t > 400) return `tier >400 · ${(t - 400).toFixed(1)} above the bound`;
  return 'bounds 0 / 1–100 / 101–400 / >400';
}

function exSummary() {
  const ex = excluded();
  if (!ex.length) return 'no withheld components';
  const area = ex.reduce((a, l) => a + l.area_mm2, 0);
  const would = ex.reduce((a, l) => a + l.area_mm2 * l.density_weight, 0);
  return `${ex.length} withheld · ${area.toFixed(2)} mm² · below 1.0 mm² · ` +
         `would add ${would.toFixed(1)} if admitted`;
}

boot();
