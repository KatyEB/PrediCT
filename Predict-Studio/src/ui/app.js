// State
const state = {
  run: null,
  slices: [],
  lesions: [],
  currentSlice: 0,
  selectedLesionId: null,
  viewMode: 2, // 1: original, 2: prediction, 3: calcium only
  showKeys: false
};

// Parse query params
const params = new URLSearchParams(window.location.search);
const studyId = params.get('study') || '172';
const modelId = params.get('model') || 'a3-coverage-v2';

const BASE_URL = `/data/out/${studyId}/${modelId}/`;

async function init() {
  try {
    const [runRes, slicesRes, lesionsText] = await Promise.all([
      fetch(`${BASE_URL}run.json`).then(r => r.json()),
      fetch(`${BASE_URL}slices.json`).then(r => r.json()),
      fetch(`${BASE_URL}lesions.csv`).then(r => r.text())
    ]);
    
    state.run = runRes;
    state.slices = slicesRes;
    state.lesions = parseCSV(lesionsText);
    
    // Find first slice with calcium for initial view
    const firstCalc = state.slices.find(s => s.has_calcium);
    state.currentSlice = firstCalc ? firstCalc.idx : Math.floor(state.run.shape[0] / 2);
    
    setupEvents();
    renderAll();
  } catch (e) {
    console.error("Failed to load data:", e);
  }
}

function parseCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',');
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    const obj = {};
    headers.forEach((h, i) => {
      let val = vals[i];
      if (val === 'True') val = true;
      else if (val === 'False') val = false;
      else if (!isNaN(Number(val))) val = Number(val);
      obj[h] = val;
    });
    return obj;
  });
}

// Coordinate transforms
const toDisplayY = (y) => state.run.shape[1] - 1 - y;
const toDisplayX = (x) => state.run.shape[2] - 1 - x;

// Drawing & Rendering
function renderAll() {
  renderProvenance();
  renderTrack();
  renderViewport();
  renderRail();
}

function renderProvenance() {
  const r = state.run;
  const isBinary = r.output === "binary";
  const thresh = r.threshold || 0.5;
  const eq = isBinary ? "area = voxel count" : "area = Σ coverage";
  document.getElementById('prov-1').innerText = `${r.model_id} · ${r.output} · component threshold ${thresh.toFixed(2)} (delineation only — ${eq})`;
  
  const spc = r.spacing.map(n => n.toFixed(2)).join(' × ');
  document.getElementById('prov-2').innerText = `model HU window ${r.hu_window.join('…')} · display window −100…400 (W 500 / L 150) · ${spc} mm · RAS`;
  
  const sha = r.sha256 ? r.sha256.substring(0,12) : "unknown";
  document.getElementById('prov-3').innerText = `crop heart +8 mm, TotalSegmentator ${r.locator_version || 'N/A'} · ckpt ${sha} · min lesion 1.0 mm² · ${new Date(r.date).toISOString().split('T')[0]}`;
}

function renderTrack() {
  const container = document.getElementById('volume-track');
  container.innerHTML = '';
  
  const numSlices = state.run.shape[0];
  const slicesWithCalc = state.slices.filter(s => s.has_calcium).length;
  const pct = Math.round((slicesWithCalc / numSlices) * 100);
  
  const sliceMeta = state.slices[state.currentSlice];
  const z0 = state.slices[0].z_mm.toFixed(1);
  const z1 = state.slices[state.slices.length - 1].z_mm.toFixed(1);
  
  document.getElementById('track-left-label').innerText = `VOLUME · SLICE ${state.currentSlice}`;
  document.getElementById('track-right-label').innerText = `${slicesWithCalc} of ${numSlices} slices carry scored calcium · ${pct} % · z ${z0}–${z1} mm`;
  
  const maxScore = Math.max(...state.slices.map(s => s.slice_score || 0));
  
  state.slices.forEach((s, idx) => {
    const btn = document.createElement('button');
    btn.className = 'track-bar';
    btn.onclick = () => setSlice(idx);
    
    // Height
    let h = 4;
    if (s.slice_score > 0 && maxScore > 0) {
      h = Math.max(4, (s.slice_score / maxScore) * 46);
    }
    btn.style.height = `${h}px`;
    
    // Background and border
    if (s.has_calcium) {
      btn.style.background = 'var(--accent)';
      btn.style.borderBottom = '3px solid var(--accent)';
    } else if (s.n_lesions_all > 0) {
      btn.style.background = 'var(--rule-2)';
      btn.style.borderBottom = '3px solid var(--muted-2)';
    } else {
      btn.style.background = 'var(--rule-2)';
      btn.style.borderBottom = '3px solid transparent';
    }
    
    // Cursor marker
    if (idx === state.currentSlice) {
      btn.style.borderTop = '1px solid var(--ink)';
    }
    
    btn.title = `slice ${idx} · z ${s.z_mm.toFixed(2)} mm · ${s.slice_score.toFixed(1)}`;
    container.appendChild(btn);
  });
  
  document.getElementById('track-legend-cursor').innerText = `▲ cursor slice ${state.currentSlice}`;
  
  if (state.viewMode === 3) {
    document.getElementById('track-legend-reach').innerText = `${numSlices - slicesWithCalc} of ${numSlices} slices unreachable in this stack`;
  } else {
    document.getElementById('track-legend-reach').innerText = `all ${numSlices} slices reachable`;
  }
}

function renderViewport() {
  const z = state.currentSlice;
  const r = state.run;
  
  // Images
  const ctImg = document.getElementById('img-ct');
  const maskImg = document.getElementById('img-mask');
  
  const padZ = String(z).padStart(3, '0');
  ctImg.src = `${BASE_URL}slices/ct/slice_${padZ}.png`;
  
  if (state.viewMode >= 2) {
    maskImg.style.display = 'block';
    maskImg.src = `${BASE_URL}slices/mask/slice_${padZ}.png`;
  } else {
    maskImg.style.display = 'none';
  }
  
  // Caption
  const nx = r.shape[2];
  const ny = r.shape[1];
  const dx = r.spacing[0];
  document.getElementById('vp-dim').innerText = `${nx} × ${ny} · ${dx.toFixed(2)} mm in-plane · ├── 10 mm ──┤`;
  document.getElementById('vp-hu').innerText = `display W 500 / L 150 (−100…400 HU) · model saw ${r.hu_window.join('…')} HU`;
  
  let note = '';
  if (state.viewMode === 1) note = 'original stack · prediction not drawn here';
  else if (state.viewMode === 2) note = 'coverage overlay · alpha = fraction · never thresholded';
  else note = `restricted stack · ${state.slices.filter(s=>s.has_calcium).length} scored slices`;
  document.getElementById('vp-overlay').innerText = note;
  
  // Chips
  [1,2,3].forEach(i => {
    const btn = document.getElementById(`btn-view-${i}`);
    if (i === state.viewMode) btn.classList.add('active');
    else btn.classList.remove('active');
  });
  
  // Canvas Ring
  drawCanvasRing();
}

function drawCanvasRing() {
  const canvas = document.getElementById('canvas-ring');
  const img = document.getElementById('img-ct');
  
  const nx = state.run.shape[2];
  const ny = state.run.shape[1];
  canvas.width = nx;
  canvas.height = ny;
  
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (!state.selectedLesionId) return;
  const lesion = state.lesions.find(l => l.lesion_id === state.selectedLesionId);
  if (!lesion || lesion.slice_idx !== state.currentSlice) return;
  
  const cx = toDisplayX(lesion.centroid_x);
  const cy = toDisplayY(lesion.centroid_y);
  
  ctx.beginPath();
  ctx.arc(cx, cy, 12, 0, 2 * Math.PI);
  ctx.strokeStyle = 'var(--accent)';
  ctx.lineWidth = 1;
  ctx.stroke();
}

function renderRail() {
  const z = state.currentSlice;
  const sMeta = state.slices[z];
  const zmm = sMeta.z_mm.toFixed(2);
  
  document.getElementById('rail-slice-num').innerText = `SLICE ${z}`;
  document.getElementById('rail-z-mm').innerText = `z ${zmm} mm`;
  
  const lesionsHere = state.lesions.filter(l => l.slice_idx === z);
  
  // Chips
  const chipsCont = document.getElementById('rail-lesion-chips');
  chipsCont.innerHTML = '';
  if (lesionsHere.length === 0) {
    chipsCont.innerHTML = `<span class="mono-11" style="color:var(--muted-2);">no lesion on this slice</span>`;
  } else {
    lesionsHere.forEach(l => {
      const btn = document.createElement('button');
      btn.className = 'lesion-chip';
      if (!l.included) btn.classList.add('withheld');
      if (l.lesion_id === state.selectedLesionId) btn.classList.add('active');
      btn.innerText = `lesion ${l.lesion_id}`;
      btn.onclick = () => {
        state.selectedLesionId = l.lesion_id;
        renderAll();
      };
      chipsCont.appendChild(btn);
    });
  }
  
  // Selected Lesion
  const selRowCont = document.getElementById('selected-lesion-rows');
  if (state.selectedLesionId) {
    const l = state.lesions.find(ll => ll.lesion_id === state.selectedLesionId);
    if (l) {
      let pColor = l.mean_coverage < 0.5 ? 'var(--prediction)' : 'var(--ink)';
      selRowCont.innerHTML = `
        <div class="kv-row"><span>lesion</span><span>${l.lesion_id}</span></div>
        <div class="kv-row"><span>area mm²</span><span>${l.area_mm2.toFixed(2)}</span></div>
        <div class="kv-row"><span>peak HU</span><span>${l.peak_hu.toFixed(1)}</span></div>
        <div class="kv-row"><span>density weight</span><span>${l.density_weight}</span></div>
        <div class="kv-row"><span>mean coverage</span><span style="color:${pColor}">${l.mean_coverage.toFixed(2)}</span></div>
        <div class="kv-row"><span>score</span><span>${l.included ? l.agatston.toFixed(2) : '<span style="color:var(--muted-2);">withheld</span>'}</span></div>
      `;
    }
  } else {
    selRowCont.innerHTML = `<span class="mono-11" style="color:var(--muted-2);">none selected</span>`;
  }
  
  // Coverage section
  const covSec = document.getElementById('coverage-section');
  if (state.run.output === 'binary') {
    covSec.style.display = 'none';
  } else {
    covSec.style.display = 'flex';
    const barsCont = document.getElementById('coverage-bars');
    barsCont.innerHTML = '';
    const bins = sMeta.coverage_hist || [0,0,0,0];
    const labels = ["0.10–0.25", "0.25–0.50", "0.50–0.75", "0.75–1.00"];
    const maxBin = Math.max(...bins, 1);
    bins.forEach((val, i) => {
      const pct = (val / maxBin) * 100;
      barsCont.innerHTML += `
        <div class="cov-bar-row">
          <div class="cov-bar-bg"><div class="cov-bar-fill" style="width:${pct}%"></div></div>
          <span>${labels[i]} · ${val} voxels</span>
        </div>
      `;
    });
  }
  
  // Table
  const tbody = document.getElementById('lesions-table-body');
  tbody.innerHTML = '';
  
  const includedLesions = state.lesions.filter(l => l.included);
  if (includedLesions.length === 0) {
    document.getElementById('empty-state-msg').style.display = 'block';
  } else {
    document.getElementById('empty-state-msg').style.display = 'none';
    includedLesions.forEach(l => {
      const tr = document.createElement('tr');
      if (l.lesion_id === state.selectedLesionId) tr.className = 'active';
      tr.onclick = () => {
        state.selectedLesionId = l.lesion_id;
        state.currentSlice = l.slice_idx;
        renderAll();
      };
      
      let pColor = '';
      if (l.lesion_id !== state.selectedLesionId && l.mean_coverage < 0.5) {
        pColor = 'color:var(--prediction)';
      }
      
      tr.innerHTML = `
        <td>lesion ${l.lesion_id}</td>
        <td>sl ${l.slice_idx}</td>
        <td>${l.area_mm2.toFixed(1)}</td>
        <td style="${pColor}">p ${l.mean_coverage.toFixed(2)}</td>
        <td>${l.agatston.toFixed(2)}</td>
      `;
      tbody.appendChild(tr);
    });
  }
  
  // Excluded
  const excluded = state.lesions.filter(l => !l.included);
  document.getElementById('excluded-title').innerText = `EXCLUDED · ${excluded.length}`;
  const exArea = excluded.reduce((sum, l) => sum + l.area_mm2, 0);
  const exScore = excluded.reduce((sum, l) => sum + (l.area_mm2 * l.density_weight), 0);
  document.getElementById('excluded-summary').innerText = `${excluded.length} withheld · ${exArea.toFixed(1)} mm² · below 1.0 mm² · would add ${exScore.toFixed(1)} if admitted`;
  
  // Footer
  const score = state.run.agatston_total;
  document.getElementById('total-score').innerText = score.toFixed(1);
  const tierEl = document.getElementById('total-tier');
  tierEl.innerText = state.run.risk_category.toUpperCase();
  if (score > 400) tierEl.className = 'tier-label severe';
  else tierEl.className = 'tier-label mild';
  
  const tierFill = document.getElementById('tier-fill');
  const maxS = 1400;
  tierFill.style.width = `${Math.min(100, (score / maxS) * 100)}%`;
  
  const noteEl = document.getElementById('tier-note');
  if (score > 400) {
    noteEl.innerText = `tier >400 · ${(score - 400).toFixed(1)} above the bound · E export · ? keys`;
  } else {
    noteEl.innerText = `bounds 0 / 1–100 / 101–400 / >400 · E export · ? keys`;
  }
}

// Events
function setSlice(idx) {
  state.currentSlice = Math.max(0, Math.min(state.run.shape[0] - 1, idx));
  renderAll();
}

function setupEvents() {
  window.addEventListener('wheel', (e) => {
    // Only step if over track or viewport
    const tgt = e.target.closest('.pane-c') || e.target.closest('.band-b');
    if (tgt) {
      e.preventDefault();
      let step = e.deltaY > 0 ? 1 : -1;
      stepSlice(step);
    }
  }, { passive: false });
  
  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      stepSlice(-1);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      stepSlice(1);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      jumpToCalc(-1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      jumpToCalc(1);
    } else if (e.key === '1') {
      state.viewMode = 1; renderAll();
    } else if (e.key === '2') {
      state.viewMode = 2; renderAll();
    } else if (e.key === '3') {
      state.viewMode = 3; 
      if (!state.slices[state.currentSlice].has_calcium) jumpToCalc(1);
      renderAll();
    } else if (e.key === '?') {
      state.showKeys = !state.showKeys;
      document.getElementById('overlay-keys').style.display = state.showKeys ? 'block' : 'none';
    } else if (e.key === 'Escape') {
      state.selectedLesionId = null;
      renderAll();
    }
  });
  
  [1,2,3].forEach(i => {
    document.getElementById(`btn-view-${i}`).onclick = () => {
      state.viewMode = i;
      if (i === 3 && !state.slices[state.currentSlice].has_calcium) jumpToCalc(1);
      renderAll();
    }
  });
}

function stepSlice(dir) {
  if (state.viewMode === 3) {
    jumpToCalc(dir);
  } else {
    setSlice(state.currentSlice + dir);
  }
}

function jumpToCalc(dir) {
  let z = state.currentSlice + dir;
  const numSlices = state.run.shape[0];
  while (z >= 0 && z < numSlices) {
    if (state.slices[z].has_calcium) {
      setSlice(z);
      return;
    }
    z += dir;
  }
}

// Ensure resize repaints the canvas properly
window.addEventListener('resize', drawCanvasRing);

init();
