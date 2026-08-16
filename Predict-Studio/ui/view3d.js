/* PrediCT Studio — 04 Anatomy. The 3D view.
 *
 * WHAT THIS IS FOR. Not diagnosis. CAC scoring is a per-slice task and stays
 * one; this view exists to do three things the axial stack cannot:
 *   1. navigate — see where calcium is, click it, land on the right slice
 *   2. show extent — one long streak in one territory, or scatter across three
 *   3. explain — to a referring physician, or to the patient
 * Every number still comes from the 2D pipeline. Nothing here is measured.
 *
 * WHAT IT DRAWS. Meshes built by src/backend/mesh.py and listed in
 * run.json.mesh. A coverage model gets three nested surfaces (p 0.25 / 0.50 /
 * 0.75); a binary model gets one at 0.50, and the difference between the two
 * is the point. The lesion surfaces are UNSMOOTHED — the visible stepping is
 * 3 mm slice spacing against 0.37 mm pixels, which is the true resolution of
 * the data. Do not add smoothing here to make it look better; that would
 * invent geometry between slices that was never measured.
 *
 * WHAT IT MUST NOT CLAIM. There is no coronary centreline extraction and no
 * atlas registration in this pipeline, so calcium cannot be attributed to LAD,
 * LCX or RCA. Commercial tools do this and readers will expect it. The banner
 * saying we do not is permanent and load-bearing. Never colour by vessel,
 * never label by vessel, never infer one from position.
 *
 * COORDINATES. Mesh vertices are millimetres on the volume's own grid:
 * x = column * sx -> patient RIGHT, y = row * sy -> ANTERIOR,
 * z = slice * sz -> SUPERIOR. Declared in run.json.mesh.axes, never inferred.
 * No 2D display flip is baked into the geometry; the slice-plane texture is
 * un-flipped here instead, in ONE place (see planeUVs).
 *
 * Does NOT: fetch CSVs, compute scores, or own selection state. It reads
 * window.PrediCT and calls back into it.
 * Called by: app.js, when state.dir === 4.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

// ── palette ──────────────────────────────────────────────────────────────
// The mask colour is baked into render.py's PNGs (201, 84, 31); the p 0.50
// surface uses exactly it so the 3D lesion and the 2D overlay are the same
// object to the eye. The outer and inner surfaces are tints of it, not new
// hues — three hues would read as three categories rather than one gradient.
const C = {
  p25:   0xE0A184,   // soft rim
  p50:   0xC9541F,   // matches render.py, and what a binary model would draw
  p75:   0x7E2F0E,   // dense core
  heart: 0x6E7C86,
  sel:   0x4FA8C5,   // same selection blue as the 2D ring
  plane: 0xFFFFFF,
};

const OPACITY = { p25: 0.22, p50: 0.45, p75: 0.95, heart: 0.07 };

let scene, camera, renderer, controls, raycaster, root;
let meshDir = null;                 // "/data/out/<study>/<model>"
let surfaces = [];                  // [{level, mesh, on}]
let heartMesh = null, planeMesh = null, planeTex = null;
let ready = false, loading = false, frameQueued = false;
let lastSel = null, lastSlice = -1, lastView = -1;

const el = id => document.getElementById(id);
const P = () => window.PrediCT;

// ── mount ────────────────────────────────────────────────────────────────
export async function mount(base, meshManifest) {
  if (loading || ready) return;
  loading = true;
  meshDir = base;

  const host = el('v-canvas');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14181B);

  camera = new THREE.PerspectiveCamera(35, 1, 1, 4000);
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  host.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.addEventListener('change', () => draw());

  raycaster = new THREE.Raycaster();

  // Flat, shadowless lighting. Cinematic shadowing is known to obscure small
  // structures in cardiac CT rendering, and the structures here are small.
  scene.add(new THREE.AmbientLight(0xffffff, 1.5));
  const key = new THREE.DirectionalLight(0xffffff, 1.4); key.position.set(1, 1, 1);
  const fill = new THREE.DirectionalLight(0xffffff, 0.7); fill.position.set(-1, -0.6, -0.8);
  scene.add(key, fill);

  // The volume is centred on the origin so orbiting spins about the anatomy
  // rather than about a corner. Every child keeps its own mm coordinates.
  root = new THREE.Group();
  const [ex, ey, ez] = meshManifest.extent_mm;
  root.position.set(-ex / 2, -ey / 2, -ez / 2);
  scene.add(root);

  // z is SUPERIOR, so z is up. three.js defaults to y-up; saying this once
  // here is what makes every camera preset below mean what it says.
  camera.up.set(0, 0, 1);

  await loadMeshes(meshManifest);
  buildPlane(meshManifest);
  wire(meshManifest);
  resetCamera(meshManifest, 'A');

  ready = true; loading = false;
  update();
}

async function loadMeshes(man) {
  const loader = new PLYLoader();
  const get = url => new Promise((res, rej) => loader.load(url, res, undefined, rej));

  for (const s of man.surfaces) {
    let geo;
    try { geo = await get(`${meshDir}/${s.file}`); }
    catch (e) { fail(`${s.file} did not load: ${e.message}`); continue; }
    geo.computeVertexNormals();
    const c = s.level <= 0.3 ? C.p25 : s.level >= 0.7 ? C.p75 : C.p50;
    const o = s.level <= 0.3 ? OPACITY.p25 : s.level >= 0.7 ? OPACITY.p75 : OPACITY.p50;
    const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({
      color: c, transparent: o < 1, opacity: o,
      side: THREE.DoubleSide, depthWrite: o > 0.8,
    }));
    mesh.userData.level = s.level;
    root.add(mesh);
    surfaces.push({ level: s.level, mesh, on: true, faces: s.n_faces });
  }

  if (man.heart) {
    try {
      const geo = await get(`${meshDir}/${man.heart.file}`);
      geo.computeVertexNormals();
      heartMesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({
        color: C.heart, transparent: true, opacity: OPACITY.heart,
        side: THREE.BackSide, depthWrite: false,
      }));
      root.add(heartMesh);
    } catch (e) { fail(`heart shell did not load: ${e.message}`); }
  }
}

/* The slice plane. Its texture is the 2D PNG, which render.py wrote flipped
   (flipud then fliplr) for radiological display. The mesh is unflipped array
   space, so the flip is undone here — in the UVs, once, and nowhere else.

   Derivation, corner by corner. render.py maps PNG (row r, col c) to array
   (y = H-1-r, x = W-1-c). Texture flipY is left at three.js's default true,
   so v = 0 is the BOTTOM of the image, i.e. PNG row H-1.

     mesh (0,  0)  = array (i 0,   j 0)   = PNG (col W-1, row H-1) -> uv (1, 0)
     mesh (ex, 0)  = array (i W-1, j 0)   = PNG (col 0,   row H-1) -> uv (0, 0)
     mesh (0,  ey) = array (i 0,   j H-1) = PNG (col W-1, row 0)   -> uv (1, 1)
     mesh (ex, ey) = array (i W-1, j H-1) = PNG (col 0,   row 0)   -> uv (0, 1)

   If render.py's flips ever change, these four pairs change with them in the
   same commit — the same rule that binds FLIP_X in app.js. */
function buildPlane(man) {
  const [ex, ey] = man.extent_mm;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(
    [0, 0, 0, ex, 0, 0, 0, ey, 0, ex, ey, 0], 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(
    [1, 0, 0, 0, 1, 1, 0, 1], 2));
  g.setIndex([0, 1, 2, 2, 1, 3]);
  g.computeVertexNormals();

  planeMesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
    color: C.plane, transparent: true, opacity: 0.85,
    side: THREE.DoubleSide, depthWrite: false,
  }));
  root.add(planeMesh);
}

// ── update from state ────────────────────────────────────────────────────
export function update() {
  if (!ready) return;
  const st = P().state, man = st.run.mesh;

  if (st.slice !== lastSlice || st.view !== lastView) {
    lastSlice = st.slice; lastView = st.view;
    planeMesh.position.z = st.slice * man.spacing_mm[2];
    const url = st.view === 1 ? P().ctUrl(st.slice) : P().maskUrl(st.slice);
    new THREE.TextureLoader().load(url, t => {
      t.flipY = true;                       // see planeUVs derivation
      t.colorSpace = THREE.SRGBColorSpace;
      if (planeTex) planeTex.dispose();
      planeTex = t;
      planeMesh.material.map = t;
      planeMesh.material.needsUpdate = true;
      draw();
    });
  }

  if (st.sel3d !== lastSel) { lastSel = st.sel3d; paintSelection(); }
  writeRail();
  draw();
}

/* Selection highlight. There is no per-lesion mesh — one surface per level
   holds every lesion — so the selected lesion is marked with a wireframe box
   built from its bounding box in lesions_3d.csv, converted voxels -> mm. That
   is honest: the box is derived from the same numbers the table shows, and it
   does not pretend to be a shape. */
let selBox = null;
function paintSelection() {
  if (selBox) { root.remove(selBox); selBox.geometry.dispose(); selBox = null; }
  const st = P().state, g = st.sel3d ? P().group3d(st.sel3d) : null;
  if (!g) return;
  const [sx, sy, sz] = st.run.mesh.spacing_mm;
  const b = new THREE.Box3(
    new THREE.Vector3((g.bbox_x0 - 1) * sx, (g.bbox_y0 - 1) * sy, (g.slice_min - 0.5) * sz),
    new THREE.Vector3((g.bbox_x1 + 1) * sx, (g.bbox_y1 + 1) * sy, (g.slice_max + 0.5) * sz));
  selBox = new THREE.Box3Helper(b, new THREE.Color(C.sel));
  selBox.material.depthTest = false;
  root.add(selBox);
}

/* Click to select. The ray hits a surface; the hit point in mm converts back
   to voxel indices, and the lesion is whichever bounding box in lesions_3d.csv
   contains it. Smallest box wins, because a large lesion's box can enclose a
   small neighbour's and the specific answer is the useful one. */
function pick(ev) {
  const st = P().state, r = renderer.domElement.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(
    ((ev.clientX - r.left) / r.width) * 2 - 1,
    -((ev.clientY - r.top) / r.height) * 2 + 1), camera);

  const targets = surfaces.filter(s => s.on).map(s => s.mesh);
  const hit = raycaster.intersectObjects(targets, false)[0];
  if (!hit) return;

  const p = root.worldToLocal(hit.point.clone());
  const [sx, sy, sz] = st.run.mesh.spacing_mm;
  const i = p.x / sx, j = p.y / sy, k = p.z / sz;
  const PAD = 1.5;   // the isosurface sits up to a voxel outside the mask bbox

  const inside = st.lesions3d.filter(g =>
    i >= g.bbox_x0 - PAD && i <= g.bbox_x1 + PAD &&
    j >= g.bbox_y0 - PAD && j <= g.bbox_y1 + PAD &&
    k >= g.slice_min - PAD && k <= g.slice_max + PAD);
  if (!inside.length) return;

  inside.sort((a, b) =>
    (a.bbox_x1 - a.bbox_x0) * (a.bbox_y1 - a.bbox_y0) * (a.slice_max - a.slice_min + 1) -
    (b.bbox_x1 - b.bbox_x0) * (b.bbox_y1 - b.bbox_y0) * (b.slice_max - b.slice_min + 1));
  P().goToLesion(inside[0].lesion_3d_key);   // moves the 2D viewer too
}

// ── camera ───────────────────────────────────────────────────────────────
// Named for the direction the VIEWER LOOKS FROM, radiological convention:
// "A" is the anterior view, i.e. the camera stands in front of the patient.
const VIEWS = { A: [0, -1, 0], P: [0, 1, 0], R: [1, 0, 0], L: [-1, 0, 0], S: [0, 0, 1] };

function resetCamera(man, which) {
  const [ex, ey, ez] = man.extent_mm;
  const d = Math.max(ex, ey, ez) * 2.1;
  const v = VIEWS[which] || VIEWS.A;
  camera.position.set(v[0] * d, v[1] * d, v[2] * d);
  // z is superior, so z-up is right for every lateral view. Looking straight
  // down z makes "up" undefined, so the superior view uses anterior-up instead
  // — which is also how an axial slice is conventionally shown.
  camera.up.set(...(which === 'S' ? [0, 1, 0] : [0, 0, 1]));
  controls.target.set(0, 0, 0);
  controls.update();
  draw();
}

function draw() {
  if (frameQueued) return;
  frameQueued = true;
  requestAnimationFrame(() => {
    frameQueued = false;
    const host = el('v-canvas');
    const w = host.clientWidth, h = host.clientHeight;
    if (w && h) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    controls.update();
    renderer.render(scene, camera);
  });
}

// ── chrome ───────────────────────────────────────────────────────────────
function wire(man) {
  renderer.domElement.addEventListener('click', pick);
  window.addEventListener('resize', () => draw());

  const box = el('v-layers');
  box.innerHTML = '';
  surfaces.forEach(s => {
    const b = document.createElement('button');
    b.className = 'on';
    b.innerHTML = `<i style="background:${
      '#' + (s.level <= 0.3 ? C.p25 : s.level >= 0.7 ? C.p75 : C.p50).toString(16).padStart(6, '0')
    }"></i>p ${s.level.toFixed(2)}<em>${s.faces.toLocaleString()} faces</em>`;
    b.onclick = () => {
      s.on = !s.on; s.mesh.visible = s.on;
      b.classList.toggle('on', s.on); draw();
    };
    box.appendChild(b);
  });

  const toggle = (id, obj) => {
    const b = el(id); if (!b) return;
    b.classList.add('on');
    b.onclick = () => { obj().visible = !obj().visible; b.classList.toggle('on', obj().visible); draw(); };
  };
  if (heartMesh) toggle('v-heart', () => heartMesh); else el('v-heart').disabled = true;
  toggle('v-plane', () => planeMesh);

  document.querySelectorAll('#v-cams button').forEach(b =>
    b.onclick = () => resetCamera(man, b.dataset.cam));

  el('v-canvas').addEventListener('wheel', e => e.stopPropagation(), { passive: true });
}

function writeRail() {
  const st = P().state, man = st.run.mesh;
  const g = st.sel3d ? P().group3d(st.sel3d) : null;

  el('v-mode').textContent = man.output_mode === 'coverage'
    ? `coverage · ${man.levels.length} nested surfaces at p ${man.levels.map(v => v.toFixed(2)).join(' / ')}`
    : `binary · one surface at p ${man.levels[0].toFixed(2)}`;

  el('v-sel').innerHTML = !g
    ? '<div class="i-kv"><span>lesion</span><span class="withheld">none selected — click a surface</span></div>'
    : [['lesion', g.lesion_3d_key],
       ['slices', `${g.n_slices} (${g.slice_min}–${g.slice_max})`],
       ['z extent', `${g.span_mm.toFixed(1)} mm`],
       ['peak HU', `${g.max_peak_hu} (slice ${g.peak_slice_idx})`],
       ['lesion score', g.total_agatston.toFixed(2)]]
      .map(([k, v]) => `<div class="i-kv"><span>${k}</span><span>${v}</span></div>`).join('');

  const n = P().counted3d().length;
  const spans = P().counted3d().filter(x => x.n_slices > 1).length;
  el('v-counts').textContent = n === 0
    ? 'no scored lesion in this study'
    : `${n} lesions · ${spans} span more than one slice · surfaces drawn from pred.nii.gz`;
}

function fail(msg) {
  const e = el('v-err');
  e.hidden = false;
  e.textContent += (e.textContent ? '\n' : '') + msg;
}

window.View3D = { mount, update };
