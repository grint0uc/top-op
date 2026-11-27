// top-op lite UI: STL import, voxelization, face selection, solver worker, live view, export.
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const MAX_ACTIVE = 40000;
  const FLAT_COS = Math.cos((10 * Math.PI) / 180);
  const NODE_SNAP = 0.75; // x h
  // ms per iteration ~ A * (n / 1e4)^(4/3); A is re-measured after every run.
  let estA = 320;

  const S = {
    mesh: null,
    res: 40,
    grid: null,
    bcs: [],
    // Keep regions: { id, mode: 'solid'|'void', min, max (box, or null), cells: Set<element id> | null,
    // cellsGrid: the grid the cell ids belong to }.
    keeps: [],
    keepSel: null,
    paint: { on: false, mode: 'add', radius: 1 },
    nextId: 1,
    selection: null,
    status: 'idle',
    history: [],
    rho8: null,
    rhoFinal: null,
    surface: null,
    view: 'design',
    selftest: null,
    worker: null,
    downloads: undefined,
    lastError: null,
    runInfo: null,
  };
  window.topopLite = S;

  const fmt = {
    int: (n) => n.toLocaleString('en-US'),
    num: (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '\u2013'),
    sci: (x) => (Number.isFinite(x) ? (Math.abs(x) >= 1e5 || Math.abs(x) < 1e-2 ? x.toExponential(3) : x.toPrecision(5)) : '\u2013'),
    len: (x) => (Math.abs(x) >= 100 ? x.toFixed(1) : Math.abs(x) >= 1 ? x.toFixed(2) : x.toPrecision(3)),
    ms: (x) => (x >= 1000 ? (x / 1000).toFixed(2) + ' s' : Math.round(x) + ' ms'),
  };

  // ---------------------------------------------------------------- example parts
  // Prism: polygon in the (x, z) plane (counter-clockwise), extruded along y.
  function extrudeXZ(poly, tris2d, y0, y1) {
    const out = [];
    const P = (i, y) => [poly[i][0], y, poly[i][1]];
    const push = (a, b, c) => out.push(...a, ...b, ...c);
    for (const [a, b, c] of tris2d) {
      push(P(a, y0), P(b, y0), P(c, y0)); // normal -y
      push(P(a, y1), P(c, y1), P(b, y1)); // normal +y
    }
    for (let i = 0; i < poly.length; i++) {
      const j = (i + 1) % poly.length;
      const A = P(i, y0), B = P(j, y0), C = P(j, y1), D = P(i, y1);
      push(A, C, B);
      push(A, D, C);
    }
    return new Float32Array(out);
  }

  const EXAMPLES = {
    // Plate 80 x 60 x 10 plus wall 10 x 60 x 60 along x = 0 (examples/bracket.stl without holes), mm.
    bracket: {
      name: 'L-bracket (80 \u00d7 60 \u00d7 60 mm)',
      res: 40,
      make: () =>
        extrudeXZ([[0, 0], [80, 0], [80, 10], [10, 10], [10, 60], [0, 60]],
          [[3, 4, 5], [3, 5, 0], [3, 0, 1], [3, 1, 2]], 0, 60),
      // Same case as examples/bracket.json: plate bottom fixed, wall pushed sideways at the top.
      bcs: [
        { kind: 'support', normal: [0, 0, -1], at: [40, 30, 0] },
        { kind: 'load', normal: [0, 0, 1], at: [5, 30, 60], force: [100, 0, 0] },
      ],
    },
    cantilever: {
      name: 'Cantilever (60 \u00d7 20 \u00d7 20 mm)',
      res: 60,
      make: () => extrudeXZ([[0, 0], [60, 0], [60, 20], [0, 20]], [[0, 1, 2], [0, 2, 3]], 0, 20),
      bcs: [
        { kind: 'support', normal: [-1, 0, 0], at: [0, 10, 10] },
        { kind: 'load', normal: [1, 0, 0], at: [60, 10, 10], force: [0, 0, -1000] },
      ],
    },
  };

  // ---------------------------------------------------------------- mesh preparation
  function prepareMesh(pos, name) {
    const ntri = pos.length / 9;
    const normals = new Float32Array(ntri * 3);
    const areas = new Float32Array(ntri);
    const bmin = [Infinity, Infinity, Infinity], bmax = [-Infinity, -Infinity, -Infinity];
    for (let v = 0; v < ntri * 3; v++) {
      for (let d = 0; d < 3; d++) {
        const x = pos[3 * v + d];
        if (x < bmin[d]) bmin[d] = x;
        if (x > bmax[d]) bmax[d] = x;
      }
    }
    for (let t = 0; t < ntri; t++) {
      const o = 9 * t;
      const ux = pos[o + 3] - pos[o], uy = pos[o + 4] - pos[o + 1], uz = pos[o + 5] - pos[o + 2];
      const vx = pos[o + 6] - pos[o], vy = pos[o + 7] - pos[o + 1], vz = pos[o + 8] - pos[o + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const l = Math.hypot(nx, ny, nz);
      areas[t] = l / 2;
      if (l > 0) {
        normals[3 * t] = nx / l;
        normals[3 * t + 1] = ny / l;
        normals[3 * t + 2] = nz / l;
      }
    }
    const diag = Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]) || 1;
    // Weld coincident vertices, then build triangle adjacency across shared edges.
    const q = diag * 1e-6;
    const vmap = new Map();
    const vid = new Int32Array(ntri * 3);
    for (let v = 0; v < ntri * 3; v++) {
      const key = Math.round(pos[3 * v] / q) + ',' + Math.round(pos[3 * v + 1] / q) + ',' + Math.round(pos[3 * v + 2] / q);
      let id = vmap.get(key);
      if (id === undefined) {
        id = vmap.size;
        vmap.set(key, id);
      }
      vid[v] = id;
    }
    const nv = vmap.size;
    const edges = new Map();
    const adj = Array.from({ length: ntri }, () => []);
    for (let t = 0; t < ntri; t++) {
      for (let e = 0; e < 3; e++) {
        const a = vid[3 * t + e], b = vid[3 * t + ((e + 1) % 3)];
        if (a === b) continue;
        const key = a < b ? a * nv + b : b * nv + a;
        const list = edges.get(key);
        if (list) {
          for (const o of list) {
            adj[t].push(o);
            adj[o].push(t);
          }
          list.push(t);
        } else edges.set(key, [t]);
      }
    }
    // A closed, manifold surface has every edge shared by exactly two triangles.
    let openEdges = 0, badEdges = 0;
    for (const list of edges.values()) {
      if (list.length === 1) openEdges++;
      else if (list.length > 2) badEdges++;
    }
    return { name, pos, ntri, normals, areas, bmin, bmax, diag, adj, openEdges, badEdges };
  }

  // ---------------------------------------------------------------- voxelization
  // Scan-line parity along z through element centres (exact for watertight meshes).
  function voxelize(mesh, res) {
    const { pos, ntri, bmin, bmax } = mesh;
    const ext = [0, 1, 2].map((d) => bmax[d] - bmin[d]);
    const L = Math.max(...ext);
    const h = L / res;
    const [nx, ny, nz] = ext.map((e) => Math.max(1, Math.ceil(e / h - 1e-6)));
    const origin = bmin.slice();
    const cols = nx * ny;
    const start = new Int32Array(cols + 1);
    const range = (t) => {
      const o = 9 * t;
      const x0 = Math.min(pos[o], pos[o + 3], pos[o + 6]), x1 = Math.max(pos[o], pos[o + 3], pos[o + 6]);
      const y0 = Math.min(pos[o + 1], pos[o + 4], pos[o + 7]), y1 = Math.max(pos[o + 1], pos[o + 4], pos[o + 7]);
      return [
        Math.max(0, Math.ceil((x0 - origin[0]) / h - 0.5)), Math.min(nx - 1, Math.floor((x1 - origin[0]) / h - 0.5)),
        Math.max(0, Math.ceil((y0 - origin[1]) / h - 0.5)), Math.min(ny - 1, Math.floor((y1 - origin[1]) / h - 0.5)),
      ];
    };
    const flatTol = 1e-12 * h * h;
    const usable = new Uint8Array(ntri);
    for (let t = 0; t < ntri; t++) usable[t] = Math.abs(mesh.normals[3 * t + 2]) * mesh.areas[t] > flatTol ? 1 : 0;
    for (let t = 0; t < ntri; t++) {
      if (!usable[t]) continue;
      const [i0, i1, j0, j1] = range(t);
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) start[i * ny + j + 1]++;
    }
    for (let c = 0; c < cols; c++) start[c + 1] += start[c];
    const bins = new Int32Array(start[cols]);
    const fillp = start.slice(0, cols);
    for (let t = 0; t < ntri; t++) {
      if (!usable[t]) continue;
      const [i0, i1, j0, j1] = range(t);
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) bins[fillp[i * ny + j]++] = t;
    }
    const active = new Uint8Array(nx * ny * nz);
    const ex = h * 1.37e-7, ey = h * 2.71e-7; // nudge the ray off shared edges
    const zs = [];
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const px = origin[0] + (i + 0.5) * h + ex, py = origin[1] + (j + 0.5) * h + ey;
        zs.length = 0;
        const c = i * ny + j;
        for (let b = start[c]; b < start[c + 1]; b++) {
          const o = 9 * bins[b];
          const x0 = pos[o], y0 = pos[o + 1], x1 = pos[o + 3], y1 = pos[o + 4], x2 = pos[o + 6], y2 = pos[o + 7];
          const d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
          if (d === 0) continue;
          const l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / d;
          const l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / d;
          const l2 = 1 - l0 - l1;
          if (l0 < 0 || l1 < 0 || l2 < 0) continue;
          zs.push(l0 * pos[o + 2] + l1 * pos[o + 5] + l2 * pos[o + 8]);
        }
        if (zs.length < 2) continue;
        zs.sort((a, b) => a - b);
        for (let q = 0; q + 1 < zs.length; q += 2) {
          const k0 = Math.max(0, Math.ceil((zs[q] - origin[2]) / h - 0.5));
          const k1 = Math.min(nz - 1, Math.floor((zs[q + 1] - origin[2]) / h - 0.5));
          for (let k = k0; k <= k1; k++) active[(i * ny + j) * nz + k] = 1;
        }
      }
    }
    let count = 0;
    const NZ1 = nz + 1, NYZ1 = (ny + 1) * NZ1;
    const nodeActive = new Uint8Array((nx + 1) * NYZ1);
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
      if (!active[(i * ny + j) * nz + k]) continue;
      count++;
      const n0 = i * NYZ1 + j * NZ1 + k;
      for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let cc = 0; cc < 2; cc++) nodeActive[n0 + a * NYZ1 + b * NZ1 + cc] = 1;
    }
    return { nx, ny, nz, h, origin, active, count, nodeActive, res };
  }

  function buildGrid(mesh, res) {
    let g = voxelize(mesh, res);
    let capped = false;
    while (g.count > MAX_ACTIVE && res > 6) {
      res = Math.max(6, Math.min(res - 1, Math.floor(res * Math.cbrt(MAX_ACTIVE / g.count))));
      g = voxelize(mesh, res);
      capped = true;
    }
    g.capped = capped;
    return g;
  }

  // ---------------------------------------------------------------- selection -> nodes
  function growFace(mesh, seed) {
    const n = mesh.normals;
    const sx = n[3 * seed], sy = n[3 * seed + 1], sz = n[3 * seed + 2];
    const seen = new Uint8Array(mesh.ntri);
    const out = [seed];
    seen[seed] = 1;
    for (let q = 0; q < out.length; q++) {
      for (const o of mesh.adj[out[q]]) {
        if (seen[o]) continue;
        seen[o] = 1;
        if (n[3 * o] * sx + n[3 * o + 1] * sy + n[3 * o + 2] * sz >= FLAT_COS) out.push(o);
      }
    }
    return Int32Array.from(out);
  }

  // Squared distance from p to triangle abc (Ericson, Real-Time Collision Detection 5.1.5).
  function distSqPointTri(px, py, pz, P, o) {
    const ax = P[o], ay = P[o + 1], az = P[o + 2];
    const abx = P[o + 3] - ax, aby = P[o + 4] - ay, abz = P[o + 5] - az;
    const acx = P[o + 6] - ax, acy = P[o + 7] - ay, acz = P[o + 8] - az;
    const apx = px - ax, apy = py - ay, apz = pz - az;
    const d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
    let cx, cy, cz;
    if (d1 <= 0 && d2 <= 0) { cx = ax; cy = ay; cz = az; }
    else {
      const bpx = px - P[o + 3], bpy = py - P[o + 4], bpz = pz - P[o + 5];
      const d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
      const cpx = px - P[o + 6], cpy = py - P[o + 7], cpz = pz - P[o + 8];
      const d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
      const vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
      if (d3 >= 0 && d4 <= d3) { cx = P[o + 3]; cy = P[o + 4]; cz = P[o + 5]; }
      else if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); cx = ax + v * abx; cy = ay + v * aby; cz = az + v * abz; }
      else if (d6 >= 0 && d5 <= d6) { cx = P[o + 6]; cy = P[o + 7]; cz = P[o + 8]; }
      else if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); cx = ax + w * acx; cy = ay + w * acy; cz = az + w * acz; }
      else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
        cx = P[o + 3] + w * (P[o + 6] - P[o + 3]); cy = P[o + 4] + w * (P[o + 7] - P[o + 4]); cz = P[o + 5] + w * (P[o + 8] - P[o + 5]);
      } else {
        const den = 1 / (va + vb + vc), v = vb * den, w = vc * den;
        cx = ax + abx * v + acx * w; cy = ay + aby * v + acy * w; cz = az + abz * v + acz * w;
      }
    }
    return (px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2;
  }

  // Full-grid node ids of active nodes within 0.75 h of any of the triangles.
  function resolveNodes(mesh, grid, tris) {
    const { nx, ny, nz, h, origin, nodeActive } = grid;
    const NZ1 = nz + 1, NYZ1 = (ny + 1) * NZ1;
    const r = NODE_SNAP * h, r2 = r * r;
    const mark = new Uint8Array(nodeActive.length);
    const out = [];
    const P = mesh.pos;
    for (const t of tris) {
      const o = 9 * t;
      const lo = [0, 1, 2].map((d) => Math.min(P[o + d], P[o + 3 + d], P[o + 6 + d]) - r);
      const hi = [0, 1, 2].map((d) => Math.max(P[o + d], P[o + 3 + d], P[o + 6 + d]) + r);
      const n = [nx, ny, nz];
      const i0 = lo.map((v, d) => Math.max(0, Math.ceil((v - origin[d]) / h)));
      const i1 = hi.map((v, d) => Math.min(n[d], Math.floor((v - origin[d]) / h)));
      for (let i = i0[0]; i <= i1[0]; i++) for (let j = i0[1]; j <= i1[1]; j++) for (let k = i0[2]; k <= i1[2]; k++) {
        const id = i * NYZ1 + j * NZ1 + k;
        if (!nodeActive[id] || mark[id]) continue;
        if (distSqPointTri(origin[0] + i * h, origin[1] + j * h, origin[2] + k * h, P, o) <= r2) {
          mark[id] = 1;
          out.push(id);
        }
      }
    }
    return Int32Array.from(out);
  }

  function nodeXYZ(grid, id) {
    const { ny, nz, h, origin } = grid;
    const k = id % (nz + 1), t = (id - k) / (nz + 1), j = t % (ny + 1), i = (t - j) / (ny + 1);
    return [origin[0] + i * h, origin[1] + j * h, origin[2] + k * h];
  }

  // Active elements whose centre lies in the region's box (inclusive), then its painted cells.
  function forEachBoxCell(grid, kp, fn) {
    if (!kp.min) return;
    const { nx, ny, nz, h, origin, active } = grid;
    for (let i = 0; i < nx; i++) {
      const x = origin[0] + (i + 0.5) * h;
      if (x < kp.min[0] || x > kp.max[0]) continue;
      for (let j = 0; j < ny; j++) {
        const y = origin[1] + (j + 0.5) * h;
        if (y < kp.min[1] || y > kp.max[1]) continue;
        for (let k = 0; k < nz; k++) {
          const z = origin[2] + (k + 0.5) * h;
          if (z < kp.min[2] || z > kp.max[2]) continue;
          const e = (i * ny + j) * nz + k;
          if (active[e]) fn(e);
        }
      }
    }
  }

  function forEachRegionCell(grid, kp, fn) {
    forEachBoxCell(grid, kp, fn);
    if (kp.cells && kp.cellsGrid === grid) for (const e of kp.cells) if (grid.active[e]) fn(e);
  }

  function regionCells(grid, kp) {
    const out = new Set();
    forEachRegionCell(grid, kp, (e) => out.add(e));
    return out;
  }

  function buildPassive(grid, keeps) {
    if (!keeps.length) return null;
    const pas = new Int8Array(grid.nx * grid.ny * grid.nz);
    for (const kp of keeps) {
      const v = kp.mode === 'solid' ? 1 : -1;
      forEachRegionCell(grid, kp, (e) => { pas[e] = v; });
    }
    return pas;
  }

  function cellAt(g, x, y, z) {
    const i = Math.floor((x - g.origin[0]) / g.h), j = Math.floor((y - g.origin[1]) / g.h), k = Math.floor((z - g.origin[2]) / g.h);
    if (i < 0 || j < 0 || k < 0 || i >= g.nx || j >= g.ny || k >= g.nz) return -1;
    return (i * g.ny + j) * g.nz + k;
  }

  function cellIJK(g, e) {
    const k = e % g.nz, t = (e - k) / g.nz, j = t % g.ny;
    return [(t - j) / g.ny, j, k];
  }

  // Painted cells follow the part when the grid changes: each old cell centre marks the new cell that
  // contains it, and every new cell whose centre lies in an old painted cell is marked too (refinement).
  function remapCells(g0, g1, cells) {
    const out = new Set();
    if (!cells.size) return out;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const e of cells) {
      const c = cellIJK(g0, e);
      const p = c.map((v, d) => g0.origin[d] + (v + 0.5) * g0.h);
      const n = cellAt(g1, p[0], p[1], p[2]);
      if (n >= 0 && g1.active[n]) out.add(n);
      for (let d = 0; d < 3; d++) { lo[d] = Math.min(lo[d], c[d]); hi[d] = Math.max(hi[d], c[d]); }
    }
    const dims = [g1.nx, g1.ny, g1.nz];
    const r0 = [0, 1, 2].map((d) => Math.max(0, Math.floor((g0.origin[d] + lo[d] * g0.h - g1.origin[d]) / g1.h)));
    const r1 = [0, 1, 2].map((d) => Math.min(dims[d] - 1, Math.ceil((g0.origin[d] + (hi[d] + 1) * g0.h - g1.origin[d]) / g1.h)));
    for (let i = r0[0]; i <= r1[0]; i++) for (let j = r0[1]; j <= r1[1]; j++) for (let k = r0[2]; k <= r1[2]; k++) {
      const n = (i * g1.ny + j) * g1.nz + k;
      if (!g1.active[n] || out.has(n)) continue;
      const o = cellAt(g0, g1.origin[0] + (i + 0.5) * g1.h, g1.origin[1] + (j + 0.5) * g1.h, g1.origin[2] + (k + 0.5) * g1.h);
      if (o >= 0 && cells.has(o)) out.add(n);
    }
    return out;
  }

  // ---------------------------------------------------------------- three.js view
  const V = {};
  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888888';
  }

  function initView() {
    const canvas = $('gl');
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 10000);
    camera.up.set(0, 0, 1);
    const controls = new THREE.OrbitControls(camera, canvas);
    controls.enableDamping = false;
    controls.screenSpacePanning = true;
    controls.addEventListener('change', requestRender);
    const hemi = new THREE.HemisphereLight(0xffffff, 0x8090a0, 0.75);
    hemi.up = new THREE.Vector3(0, 0, 1);
    hemi.position.set(0, 0, 1);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 0.65);
    key.position.set(1, -0.6, 1.4);
    camera.add(key);
    scene.add(camera);
    const root = new THREE.Group();
    scene.add(root);
    Object.assign(V, { renderer, scene, camera, controls, root, key, raycaster: new THREE.Raycaster() });
    const ro = new ResizeObserver(resize);
    ro.observe($('viewport'));
    resize();
  }

  function resize() {
    if (!V.renderer) return;
    const el = $('viewport');
    const w = Math.max(1, el.clientWidth), h = Math.max(1, el.clientHeight);
    V.renderer.setSize(w, h, false);
    V.camera.aspect = w / h;
    V.camera.updateProjectionMatrix();
    requestRender();
  }

  let renderPending = false;
  function requestRender() {
    if (renderPending || !V.renderer) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      updateHandleScale();
      V.renderer.render(V.scene, V.camera);
    });
  }

  function disposeObj(o) {
    if (!o) return;
    o.traverse?.((c) => {
      c.geometry?.dispose?.();
      if (c.material) (Array.isArray(c.material) ? c.material : [c.material]).forEach((m) => m.dispose());
    });
    o.parent?.remove(o);
  }

  function fitView() {
    const m = S.mesh;
    if (!m) return;
    const c = new THREE.Vector3((m.bmin[0] + m.bmax[0]) / 2, (m.bmin[1] + m.bmax[1]) / 2, (m.bmin[2] + m.bmax[2]) / 2);
    const dir = new THREE.Vector3(1.0, -1.55, 0.95).normalize();
    const fov = (V.camera.fov * Math.PI) / 180;
    const dist = (m.diag * 0.55) / Math.sin(fov / 2) / Math.min(1, V.camera.aspect);
    V.camera.position.copy(c).addScaledVector(dir, dist);
    V.camera.near = m.diag / 200;
    V.camera.far = m.diag * 50;
    V.camera.updateProjectionMatrix();
    V.controls.target.copy(c);
    V.controls.update();
    requestRender();
  }

  function buildDesignMesh() {
    disposeObj(V.design);
    disposeObj(V.ground);
    const m = S.mesh;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(m.pos, 3));
    const nrm = new Float32Array(m.ntri * 9);
    for (let t = 0; t < m.ntri; t++) for (let v = 0; v < 3; v++) nrm.set(m.normals.subarray(3 * t, 3 * t + 3), 9 * t + 3 * v);
    geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
    const mat = new THREE.MeshStandardMaterial({ color: token('--model'), roughness: 0.75, metalness: 0.05, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
    V.design = new THREE.Mesh(geo, mat);
    V.design.renderOrder = 0;
    V.root.add(V.design);
    const size = Math.ceil((m.diag * 1.6) / 10) * 10;
    const grid = new THREE.GridHelper(size, 20, token('--grid-major'), token('--grid-minor'));
    grid.rotation.x = Math.PI / 2;
    grid.position.set((m.bmin[0] + m.bmax[0]) / 2, (m.bmin[1] + m.bmax[1]) / 2, m.bmin[2] - m.diag * 0.002);
    grid.material.transparent = true;
    grid.material.opacity = 0.8;
    V.ground = grid;
    V.root.add(grid);
  }

  function trisGeometry(tris) {
    const P = S.mesh.pos;
    const arr = new Float32Array(tris.length * 9);
    tris.forEach((t, i) => arr.set(P.subarray(9 * t, 9 * t + 9), 9 * i));
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    return g;
  }

  function overlayMaterial(color, opacity) {
    return new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
  }

  function buildSelectionMesh() {
    disposeObj(V.sel);
    V.sel = null;
    if (!S.selection) return requestRender();
    V.sel = new THREE.Mesh(trisGeometry(S.selection.tris), overlayMaterial(token('--select'), 0.75));
    V.sel.renderOrder = 3;
    V.root.add(V.sel);
    requestRender();
  }

  function buildBCMarkers() {
    disposeObj(V.bc);
    V.bc = new THREE.Group();
    const g = S.grid;
    for (const bc of S.bcs) {
      const color = token(bc.kind === 'support' ? '--support' : '--load');
      const face = new THREE.Mesh(trisGeometry(bc.tris), overlayMaterial(color, 0.35));
      face.renderOrder = 2;
      V.bc.add(face);
      if (!g || !bc.nodes.length) continue;
      const pts = new Float32Array(bc.nodes.length * 3);
      const c = [0, 0, 0];
      bc.nodes.forEach((id, i) => {
        const p = nodeXYZ(g, id);
        pts.set(p, 3 * i);
        c[0] += p[0]; c[1] += p[1]; c[2] += p[2];
      });
      const pg = new THREE.BufferGeometry();
      pg.setAttribute('position', new THREE.BufferAttribute(pts, 3));
      const pm = new THREE.PointsMaterial({ color, size: bc.kind === 'support' ? 4 : 3.5, sizeAttenuation: false, depthTest: true });
      const points = new THREE.Points(pg, pm);
      points.renderOrder = 4;
      V.bc.add(points);
      if (bc.kind === 'load') {
        const f = new THREE.Vector3(...bc.force);
        if (f.lengthSq() > 0) {
          const dir = f.clone().normalize();
          const len = S.mesh.diag * 0.22;
          const cen = new THREE.Vector3(c[0], c[1], c[2]).divideScalar(bc.nodes.length);
          const arrow = new THREE.ArrowHelper(dir, cen.clone().addScaledVector(dir, -len), len, color, len * 0.28, len * 0.14);
          arrow.renderOrder = 5;
          arrow.traverse((o) => { if (o.material) { o.material.depthTest = false; o.material.transparent = true; } });
          V.bc.add(arrow);
        }
      }
    }
    V.root.add(V.bc);
    buildKeepView();
  }

  // ---------------------------------------------------------------- keep regions: view
  const AXIS_TOKENS = ['--axis-x', '--axis-y', '--axis-z'];
  const HANDLE_PX = 26; // arrow length on screen
  const keepToken = (kp) => (kp.mode === 'solid' ? '--keep-solid' : '--keep-void');
  const selectedKeep = () => S.keeps.find((k) => k.id === S.keepSel) || null;

  function buildKeepView() {
    disposeObj(V.keep);
    V.keep = new THREE.Group();
    V.handles = null;
    V.keepVox = [];
    const g = S.grid;
    for (const kp of S.keeps) {
      const color = new THREE.Color(token(keepToken(kp)));
      const sel = kp.id === S.keepSel;
      if (kp.min) {
        const helper = new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(...kp.min), new THREE.Vector3(...kp.max)), color);
        if (sel) {
          helper.material.depthTest = false;
          helper.material.transparent = true;
          helper.renderOrder = 6;
        }
        V.keep.add(helper);
      }
      if (g && kp.cells && kp.cells.size && kp.cellsGrid === g) {
        const vox = keepVoxelMesh(g, kp, color, sel);
        V.keep.add(vox);
        V.keepVox.push(vox);
      }
    }
    const kp = selectedKeep();
    if (kp && kp.min && S.status !== 'running') {
      V.handles = makeHandles(kp);
      V.keep.add(V.handles);
    }
    V.root.add(V.keep);
    requestRender();
  }

  let keepViewPending = false;
  function scheduleKeepView() {
    if (keepViewPending) return;
    keepViewPending = true;
    requestAnimationFrame(() => {
      keepViewPending = false;
      buildKeepView();
      renderKeepList();
    });
  }

  // Translucent cubes for the painted cells; only cells on the outside of the set are drawn. Drawn
  // without depth test so cells inside the part show through it.
  function keepVoxelMesh(g, kp, color, sel) {
    const { nx, ny, nz, h, origin } = g;
    const cells = kp.cells;
    const has = (i, j, k) => i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz && cells.has((i * ny + j) * nz + k);
    const list = [];
    for (const e of cells) {
      if (!g.active[e]) continue;
      const [i, j, k] = cellIJK(g, e);
      if (has(i - 1, j, k) && has(i + 1, j, k) && has(i, j - 1, k) && has(i, j + 1, k) && has(i, j, k - 1) && has(i, j, k + 1)) continue;
      list.push(i, j, k);
    }
    const n = list.length / 3;
    const geo = new THREE.BoxGeometry(h * 0.92, h * 0.92, h * 0.92);
    const mat = new THREE.MeshLambertMaterial({ color, transparent: true, opacity: sel ? 0.5 : 0.32, depthTest: false, depthWrite: false });
    const mesh = new THREE.InstancedMesh(geo, mat, Math.max(1, n));
    const m4 = new THREE.Matrix4();
    for (let q = 0; q < n; q++) {
      m4.makeTranslation(origin[0] + (list[3 * q] + 0.5) * h, origin[1] + (list[3 * q + 1] + 0.5) * h, origin[2] + (list[3 * q + 2] + 0.5) * h);
      mesh.setMatrixAt(q, m4);
    }
    mesh.count = n;
    mesh.frustumCulled = false;
    mesh.renderOrder = 6;
    mesh.userData.keepId = kp.id;
    return mesh;
  }

  // One arrow per face of the selected box, at the face centre, pointing outward. Sized in pixels.
  function makeHandles(kp) {
    const grp = new THREE.Group();
    const cone = new THREE.ConeGeometry(0.2, 0.42, 16).translate(0, 0.79, 0);
    const stem = new THREE.CylinderGeometry(0.045, 0.045, 0.6, 8).translate(0, 0.3, 0);
    const plate = new THREE.PlaneGeometry(0.34, 0.34).rotateX(-Math.PI / 2);
    const up = new THREE.Vector3(0, 1, 0);
    const c = [0, 1, 2].map((d) => (kp.min[d] + kp.max[d]) / 2);
    for (let a = 0; a < 3; a++) for (let side = 0; side < 2; side++) {
      const mat = new THREE.MeshBasicMaterial({ color: token(AXIS_TOKENS[a]), depthTest: false, transparent: true, side: THREE.DoubleSide });
      const hd = new THREE.Group();
      for (const geo of [cone, stem, plate]) {
        const m = new THREE.Mesh(geo, mat);
        m.renderOrder = 8;
        hd.add(m);
      }
      const p = c.slice();
      p[a] = side ? kp.max[a] : kp.min[a];
      const n = new THREE.Vector3();
      n.setComponent(a, side ? 1 : -1);
      hd.position.set(...p);
      hd.quaternion.setFromUnitVectors(up, n);
      hd.userData = { axis: a, side, normal: n };
      grp.add(hd);
    }
    return grp;
  }

  function worldPerPixel(p) {
    const dist = V.camera.position.distanceTo(p);
    return (2 * dist * Math.tan((V.camera.fov * Math.PI) / 360)) / Math.max(1, $('gl').clientHeight);
  }

  function updateHandleScale() {
    if (!V.handles) return;
    for (const hd of V.handles.children) hd.scale.setScalar(worldPerPixel(hd.position) * HANDLE_PX);
  }

  function toClient(p) {
    const v = p.clone().project(V.camera);
    const r = $('gl').getBoundingClientRect();
    return { x: r.left + ((v.x + 1) / 2) * r.width, y: r.top + ((1 - v.y) / 2) * r.height, ok: v.z > -1 && v.z < 1 };
  }

  function distToSeg(px, py, a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const l2 = dx * dx + dy * dy;
    const t = l2 ? Math.max(0, Math.min(1, ((px - a.x) * dx + (py - a.y) * dy) / l2)) : 0;
    return Math.hypot(px - a.x - t * dx, py - a.y - t * dy);
  }

  function handleScreen(hd) {
    const len = worldPerPixel(hd.position) * HANDLE_PX;
    return { a: toClient(hd.position), b: toClient(hd.position.clone().addScaledVector(hd.userData.normal, len)) };
  }

  function handleAt(cx, cy, tol) {
    if (!V.handles) return null;
    V.camera.updateMatrixWorld();
    let best = null, bd = tol;
    for (const hd of V.handles.children) {
      const { a, b } = handleScreen(hd);
      if (!a.ok || !b.ok) continue;
      const d = distToSeg(cx, cy, a, b);
      if (d < bd) { bd = d; best = hd; }
    }
    return best;
  }

  // Region whose box outline passes within tol pixels of the click.
  function keepOutlineAt(cx, cy, tol) {
    V.camera.updateMatrixWorld();
    let best = null, bd = tol;
    for (const kp of S.keeps) {
      if (!kp.min) continue;
      const P = [];
      for (let q = 0; q < 8; q++) P.push(toClient(new THREE.Vector3(q & 1 ? kp.max[0] : kp.min[0], q & 2 ? kp.max[1] : kp.min[1], q & 4 ? kp.max[2] : kp.min[2])));
      for (let q = 0; q < 8; q++) for (const bit of [1, 2, 4]) {
        if (q & bit) continue;
        const a = P[q], b = P[q | bit];
        if (!a.ok || !b.ok) continue;
        const d = distToSeg(cx, cy, a, b);
        if (d < bd) { bd = d; best = kp; }
      }
    }
    return best;
  }

  function ensureVoxelMesh() {
    const g = S.grid;
    if (V.vox && V.voxCap === g.count && V.voxH === g.h) return;
    disposeObj(V.vox);
    const geo = new THREE.BoxGeometry(g.h * 0.97, g.h * 0.97, g.h * 0.97);
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.7, metalness: 0.0 });
    const mesh = new THREE.InstancedMesh(geo, mat, Math.max(1, g.count));
    mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, g.count) * 3), 3);
    mesh.count = 0;
    mesh.frustumCulled = false;
    V.vox = mesh;
    V.voxCap = g.count;
    V.voxH = g.h;
    V.root.add(mesh);
  }

  function updateVoxels() {
    const g = S.grid, rho = S.rho8;
    if (!g || !rho) return;
    ensureVoxelMesh();
    const thr = Math.round(+$('thr').value * 255);
    const { nx, ny, nz, h, origin } = g;
    const M = V.vox.instanceMatrix.array, C = V.vox.instanceColor.array;
    const lo = new THREE.Color(token('--rho-lo')), hi = new THREE.Color(token('--rho-hi')), col = new THREE.Color();
    const cap = V.voxCap;
    let n = 0;
    const at = (i, j, k) => (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz ? 0 : rho[(i * ny + j) * nz + k]);
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) for (let k = 0; k < nz; k++) {
      const v = rho[(i * ny + j) * nz + k];
      if (v < thr || v === 0) continue;
      if (at(i - 1, j, k) >= thr && at(i + 1, j, k) >= thr && at(i, j - 1, k) >= thr && at(i, j + 1, k) >= thr && at(i, j, k - 1) >= thr && at(i, j, k + 1) >= thr) continue;
      if (n >= cap) break;
      const o = 16 * n;
      M[o] = 1; M[o + 1] = 0; M[o + 2] = 0; M[o + 3] = 0;
      M[o + 4] = 0; M[o + 5] = 1; M[o + 6] = 0; M[o + 7] = 0;
      M[o + 8] = 0; M[o + 9] = 0; M[o + 10] = 1; M[o + 11] = 0;
      M[o + 12] = origin[0] + (i + 0.5) * h; M[o + 13] = origin[1] + (j + 0.5) * h; M[o + 14] = origin[2] + (k + 0.5) * h; M[o + 15] = 1;
      col.lerpColors(lo, hi, Math.min(1, Math.max(0, (v / 255 - thr / 255) / Math.max(1e-6, 1 - thr / 255))));
      C[3 * n] = col.r; C[3 * n + 1] = col.g; C[3 * n + 2] = col.b;
      n++;
    }
    V.vox.count = n;
    V.vox.instanceMatrix.needsUpdate = true;
    V.vox.instanceColor.needsUpdate = true;
    requestRender();
  }

  let voxPending = false;
  function scheduleVoxels() {
    if (voxPending) return;
    voxPending = true;
    requestAnimationFrame(() => {
      voxPending = false;
      if (S.view === 'voxels') updateVoxels();
    });
  }

  function buildSurface() {
    const g = S.grid;
    const field = S.rhoFinal || (S.rho8 && Float32Array.from(S.rho8, (v) => v / 255));
    if (!g || !field) return null;
    const t0 = performance.now();
    const iso = +$('thr').value;
    const mc = TopopMesh.marchingCubes(field, g.nx, g.ny, g.nz, g.h, g.origin, iso);
    const passes = +$('smooth').value;
    const positions = TopopMesh.smooth(mc.positions, mc.indices, passes);
    const vol = TopopMesh.signedVolume(positions, mc.indices);
    S.surface = { positions, indices: mc.indices, ntri: mc.indices.length / 3, nvert: positions.length / 3, volume: vol, ms: performance.now() - t0 };
    disposeObj(V.surf);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(mc.indices, 1));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ color: token('--surface-model'), roughness: 0.55, metalness: 0.08 });
    V.surf = new THREE.Mesh(geo, mat);
    V.root.add(V.surf);
    updateSurfaceInfo();
    applyView();
    return S.surface;
  }

  let surfTimer = 0;
  function scheduleSurface() {
    clearTimeout(surfTimer);
    surfTimer = setTimeout(() => { if (S.view === 'surface') buildSurface(); }, 120);
  }

  function applyView() {
    const v = S.view;
    if (V.design) {
      const ghost = v !== 'design';
      const m = V.design.material;
      m.transparent = ghost;
      m.opacity = ghost ? 0.13 : 1;
      m.depthWrite = !ghost;
      m.needsUpdate = true;
    }
    if (V.vox) V.vox.visible = v === 'voxels';
    if (V.surf) V.surf.visible = v === 'surface';
    if (V.bc) V.bc.visible = true;
    $('rho-bar').hidden = v !== 'voxels';
    $('rho-bar-lo').textContent = (+$('thr').value).toFixed(2);
    document.querySelectorAll('.seg [data-view]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === v)));
    requestRender();
  }

  function setView(v) {
    S.view = v;
    if (v === 'voxels') updateVoxels();
    if (v === 'surface' && (!S.surface || S.surfaceStale)) {
      S.surfaceStale = false;
      buildSurface();
    }
    applyView();
  }

  function refreshThemeColors() {
    if (!V.renderer) return;
    if (V.design) V.design.material.color.set(token('--model'));
    if (V.surf) V.surf.material.color.set(token('--surface-model'));
    if (S.mesh) {
      const cam = V.camera.position.clone(), tgt = V.controls.target.clone();
      buildDesignMesh();
      V.camera.position.copy(cam);
      V.controls.target.copy(tgt);
      V.controls.update();
    }
    buildSelectionMesh();
    buildBCMarkers();
    if (S.view === 'voxels') updateVoxels();
    applyView();
    drawSpark();
  }

  // ---------------------------------------------------------------- picking, handle drags, voxel paint
  // pointerdown is caught on the viewport in the capture phase, before OrbitControls sees it on the
  // canvas, so a handle drag or a paint stroke can switch orbiting off for that pointer.
  function initPicking() {
    const canvas = $('gl');
    let down = null;
    let touches = 0;
    let drag = null;
    const endDrag = () => {
      if (!drag) return;
      const d = drag;
      drag = null;
      V.controls.enabled = true;
      d.end();
    };
    $('viewport').addEventListener('pointerdown', (e) => {
      if (e.target !== canvas) return;
      touches++;
      if (drag) { endDrag(); down = null; return; }
      if (touches === 1 && e.button === 0 && S.status !== 'running' && V.design) {
        const tol = e.pointerType === 'touch' ? 26 : 14;
        const hd = handleAt(e.clientX, e.clientY, tol);
        drag = hd ? startHandleDrag(hd, e) : S.paint.on ? startStroke(e) : null;
        if (drag) {
          drag.id = e.pointerId;
          V.controls.enabled = false;
          canvas.setPointerCapture?.(e.pointerId);
          down = null;
          return;
        }
      }
      down = touches === 1 ? { x: e.clientX, y: e.clientY, t: performance.now() } : null;
    }, true);
    canvas.addEventListener('pointermove', (e) => {
      if (drag && e.pointerId === drag.id) drag.move(e);
    });
    const end = (e) => {
      touches = Math.max(0, touches - 1);
      if (drag && e.pointerId === drag.id) { endDrag(); return; }
      if (!down || e.type === 'pointercancel') { down = null; return; }
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      const quick = performance.now() - down.t < 600;
      down = null;
      if (moved < 6 && quick && e.button === 0 && !S.paint.on) pick(e.clientX, e.clientY, e.pointerType === 'touch');
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
  }

  function rayAt(cx, cy) {
    const rect = $('gl').getBoundingClientRect();
    const p = new THREE.Vector2(((cx - rect.left) / rect.width) * 2 - 1, -((cy - rect.top) / rect.height) * 2 + 1);
    V.camera.updateMatrixWorld();
    V.raycaster.setFromCamera(p, V.camera);
    return V.raycaster.ray;
  }

  // Drag one face of the selected box along its axis, in whole voxels from where the drag started.
  function startHandleDrag(hd, e) {
    const kp = selectedKeep();
    if (!kp || !kp.min) return null;
    const { axis: a, side } = hd.userData;
    const anchor = hd.position.clone();
    const dir = new THREE.Vector3();
    dir.setComponent(a, 1);
    const param = (ev) => {
      const ray = rayAt(ev.clientX, ev.clientY);
      const w0 = ray.origin.clone().sub(anchor);
      const b = ray.direction.dot(dir), d = ray.direction.dot(w0), ee = dir.dot(w0);
      const den = 1 - b * b;
      return den < 1e-6 ? null : (ee - b * d) / den;
    };
    const t0 = param(e);
    if (t0 === null) return null;
    const v0 = side ? kp.max[a] : kp.min[a];
    const m = S.mesh;
    let changed = false;
    return {
      move(ev) {
        const t = param(ev);
        if (t === null) return;
        const g = S.grid, step = g ? g.h : 0;
        const gap = step || m.diag * 1e-3;
        let dv = Math.max(-2 * m.diag, Math.min(2 * m.diag, t - t0));
        if (step) dv = Math.round(dv / step) * step;
        let v = v0 + dv;
        if (side && v < kp.min[a] + gap) v = step ? v0 + Math.ceil((kp.min[a] + gap - v0) / step - 1e-9) * step : kp.min[a] + gap;
        if (!side && v > kp.max[a] - gap) v = step ? v0 + Math.floor((kp.max[a] - gap - v0) / step + 1e-9) * step : kp.max[a] - gap;
        const arr = side ? kp.max : kp.min;
        if (arr[a] === v) return;
        arr[a] = v;
        if (!changed) { changed = true; invalidateResult(); }
        syncKeepFields();
        scheduleKeepView();
      },
      end() {
        renderKeepList();
        buildKeepView();
      },
    };
  }

  // Nearest hit on the part or on the shown keep voxels; erasing prefers the voxels so it can dig in.
  function paintHit(cx, cy) {
    const ray = rayAt(cx, cy);
    const dh = V.raycaster.intersectObject(V.design, false)[0];
    const sel = S.keepSel;
    const vox = (V.keepVox || []).filter((m) => sel === null || m.userData.keepId === sel);
    const vh = vox.length ? V.raycaster.intersectObjects(vox, false)[0] : null;
    const hit = S.paint.mode === 'erase' && vh ? vh : !dh ? vh : !vh ? dh : dh.distance <= vh.distance ? dh : vh;
    return hit ? { point: hit.point, dir: ray.direction.clone() } : null;
  }

  function startStroke(e) {
    const g = S.grid;
    if (!g || !g.count || !paintHit(e.clientX, e.clientY)) return null;
    let kp = selectedKeep();
    if (!kp && S.paint.mode === 'add') {
      kp = { id: S.nextId++, mode: $('keep-mode').value, min: null, max: null, cells: new Set(), cellsGrid: g };
      S.keeps.push(kp);
      S.keepSel = kp.id;
      syncKeepFields();
    }
    if (kp && (!kp.cells || kp.cellsGrid !== g)) { kp.cells = new Set(); kp.cellsGrid = g; }
    const targets = kp ? [kp] : S.keeps.filter((k) => k.cells && k.cellsGrid === g);
    let changed = false;
    let last = { x: e.clientX, y: e.clientY };
    const dab = (cx, cy) => {
      const hit = paintHit(cx, cy);
      if (!hit) return;
      const p = hit.point.clone().addScaledVector(hit.dir, g.h / 2);
      const c = cellAt(g, p.x, p.y, p.z);
      if (c < 0) return;
      if (brushCells(g, cellIJK(g, c), targets) && !changed) { changed = true; invalidateResult(); }
    };
    dab(e.clientX, e.clientY);
    buildKeepView();
    renderKeepList();
    return {
      move(ev) {
        const n = Math.max(1, Math.ceil(Math.hypot(ev.clientX - last.x, ev.clientY - last.y) / 4));
        for (let q = 1; q <= n; q++) dab(last.x + ((ev.clientX - last.x) * q) / n, last.y + ((ev.clientY - last.y) * q) / n);
        last = { x: ev.clientX, y: ev.clientY };
        scheduleKeepView();
      },
      end() {
        buildKeepView();
        renderKeepList();
      },
    };
  }

  function brushCells(g, [i0, j0, k0], targets) {
    const r = S.paint.radius, add = S.paint.mode === 'add';
    let n = 0;
    for (let di = -r; di <= r; di++) for (let dj = -r; dj <= r; dj++) for (let dk = -r; dk <= r; dk++) {
      if (di * di + dj * dj + dk * dk > r * r + 1e-9) continue;
      const i = i0 + di, j = j0 + dj, k = k0 + dk;
      if (i < 0 || j < 0 || k < 0 || i >= g.nx || j >= g.ny || k >= g.nz) continue;
      const e = (i * g.ny + j) * g.nz + k;
      for (const kp of targets) {
        if (add) { if (g.active[e] && !kp.cells.has(e)) { kp.cells.add(e); n++; } }
        else if (kp.cells.delete(e)) n++;
      }
    }
    return n;
  }

  function pick(cx, cy, touch = false) {
    if (!V.design || S.status === 'running') return;
    const kp = keepOutlineAt(cx, cy, touch ? 16 : 7);
    if (kp) {
      selectKeep(kp.id, true);
      return;
    }
    rayAt(cx, cy);
    const hit = V.raycaster.intersectObject(V.design, false)[0];
    if (!hit) {
      selectTris(null);
      selectKeep(null);
      return;
    }
    selectTris(growFace(S.mesh, hit.faceIndex));
  }

  function selectTris(tris) {
    if (!tris || !tris.length) {
      S.selection = null;
    } else {
      const m = S.mesh;
      let area = 0;
      for (const t of tris) area += m.areas[t];
      const seed = tris[0];
      const normal = [m.normals[3 * seed], m.normals[3 * seed + 1], m.normals[3 * seed + 2]];
      const nodes = S.grid ? resolveNodes(m, S.grid, tris) : new Int32Array(0);
      S.selection = { tris, area, normal, nodes };
    }
    buildSelectionMesh();
    renderSelCard();
  }

  function renderSelCard() {
    const sel = S.selection;
    $('selcard').hidden = !sel;
    if (!sel) return;
    const n = sel.normal.map((x) => (x >= 0 ? '+' : '') + x.toFixed(2)).join(', ');
    $('sel-info').textContent = `${fmt.int(sel.tris.length)} tri \u00b7 ${fmt.len(sel.area)} u\u00b2 \u00b7 n (${n}) \u00b7 ${fmt.int(sel.nodes.length)} nodes`;
    const busy = S.status === 'running';
    $('add-support').disabled = busy || !sel.nodes.length;
    $('add-load').disabled = busy || !sel.nodes.length;
  }

  // ---------------------------------------------------------------- boundary conditions
  function addBC(kind, tris, force) {
    const bc = { id: S.nextId++, kind, tris, force: force || null, nodes: new Int32Array(0) };
    bc.nodes = S.grid ? resolveNodes(S.mesh, S.grid, tris) : bc.nodes;
    S.bcs.push(bc);
    invalidateResult();
    renderBCList();
    buildBCMarkers();
    return bc;
  }

  function removeBC(id) {
    S.bcs = S.bcs.filter((b) => b.id !== id);
    invalidateResult();
    renderBCList();
    buildBCMarkers();
  }

  function renderBCList() {
    const ul = $('bc-list');
    ul.replaceChildren();
    let ns = 0, nl = 0;
    for (const bc of S.bcs) {
      const li = document.createElement('li');
      const sw = document.createElement('i');
      sw.className = 'sw';
      sw.style.background = `var(${bc.kind === 'support' ? '--support' : '--load'})`;
      const t = document.createElement('div');
      t.className = 't';
      const idx = bc.kind === 'support' ? ++ns : ++nl;
      t.textContent = bc.kind === 'support' ? `Support ${idx} \u00b7 fixed x y z` : `Load ${idx} \u00b7 F = (${bc.force.map((x) => +x.toPrecision(4)).join(', ')}) N`;
      const small = document.createElement('small');
      small.textContent = `${fmt.int(bc.tris.length)} triangles \u2192 ${fmt.int(bc.nodes.length)} nodes`;
      if (!bc.nodes.length) small.textContent += ' (no grid nodes: refine the grid)';
      t.append(small);
      const x = document.createElement('button');
      x.className = 'x';
      x.type = 'button';
      x.textContent = '\u00d7';
      x.setAttribute('aria-label', `Remove ${bc.kind} ${idx}`);
      x.disabled = S.status === 'running';
      x.addEventListener('click', () => removeBC(bc.id));
      li.append(sw, t, x);
      ul.append(li);
    }
    if (!S.bcs.length) {
      const li = document.createElement('li');
      li.className = 'empty';
      li.textContent = 'No supports or loads yet.';
      ul.append(li);
    }
    $('bc-tag').textContent = `${ns} fixed \u00b7 ${nl} load${nl === 1 ? '' : 's'}`;
  }

  function keepLabel(kp) {
    let n = 0;
    for (const k of S.keeps) { if (k.mode === kp.mode) n++; if (k === kp) break; }
    return `${kp.mode === 'solid' ? 'Keep-in' : 'Keep-out'} ${n}`;
  }

  function keepKind(kp) {
    const nv = kp.cells && kp.cellsGrid === S.grid ? kp.cells.size : 0;
    if (kp.min && nv) return `box + voxels (${fmt.int(nv)})`;
    if (kp.min) return 'box';
    return `voxels (${fmt.int(nv)})`;
  }

  function renderKeepList() {
    const ul = $('keep-list');
    ul.replaceChildren();
    const busy = S.status === 'running';
    for (const kp of S.keeps) {
      const li = document.createElement('li');
      li.dataset.id = kp.id;
      li.setAttribute('aria-current', String(kp.id === S.keepSel));
      const sw = document.createElement('i');
      sw.className = 'sw';
      sw.style.background = `var(${keepToken(kp)})`;
      const t = document.createElement('button');
      t.type = 'button';
      t.className = 't';
      t.textContent = `${keepLabel(kp)} \u00b7 ${keepKind(kp)}`;
      t.title = kp.id === S.keepSel ? 'Selected: click to deselect' : 'Select to edit';
      t.addEventListener('click', () => selectKeep(kp.id === S.keepSel ? null : kp.id));
      const small = document.createElement('small');
      const g = S.grid;
      small.textContent = kp.min
        ? `(${kp.min.map(fmt.len).join(', ')}) \u2192 (${kp.max.map(fmt.len).join(', ')})`
        : g ? `painted on the ${g.nx} \u00d7 ${g.ny} \u00d7 ${g.nz} grid` : 'painted cells';
      t.append(small);
      const x = document.createElement('button');
      x.className = 'x';
      x.type = 'button';
      x.textContent = '\u00d7';
      x.setAttribute('aria-label', `Remove ${keepLabel(kp)}`);
      x.disabled = busy;
      x.addEventListener('click', () => {
        S.keeps = S.keeps.filter((k) => k !== kp);
        if (S.keepSel === kp.id) S.keepSel = null;
        invalidateResult();
        syncKeepFields();
        renderKeepList();
        buildKeepView();
      });
      li.append(sw, t, x);
      ul.append(li);
    }
    $('keep-tag').textContent = S.keeps.length ? String(S.keeps.length) : '';
    updateKeepButtons();
  }

  const KEEP_FIELDS = ['keep-x0', 'keep-y0', 'keep-z0', 'keep-x1', 'keep-y1', 'keep-z1'];

  // The min/max fields show the selected region's box; with no box selected they describe a new box.
  function syncKeepFields() {
    const kp = selectedKeep();
    const note = $('keep-edit');
    if (kp && kp.min) {
      KEEP_FIELDS.forEach((id, q) => {
        const v = q < 3 ? kp.min[q] : kp.max[q - 3];
        if (document.activeElement !== $(id) || +$(id).value !== v) $(id).value = +v.toPrecision(7);
      });
      $('keep-mode').value = kp.mode;
      note.textContent = `Editing ${keepLabel(kp)}: type values or drag the arrows in the view (steps of one voxel).`;
    } else if (kp) {
      $('keep-mode').value = kp.mode;
      note.textContent = `${keepLabel(kp)} has no box: paint it, or Add box to start a new box.`;
    } else {
      note.textContent = 'New box: enter min and max, then Add box.';
    }
    updateKeepButtons();
  }

  function updateKeepButtons() {
    const kp = selectedKeep();
    const busy = S.status === 'running';
    $('keep-convert').disabled = busy || !kp || !kp.min || !S.grid;
    $('keep-paint').disabled = busy || !S.grid;
  }

  function selectKeep(id, reveal = false) {
    S.keepSel = id;
    if (reveal) $('sec-keep').open = true;
    syncKeepFields();
    renderKeepList();
    buildKeepView();
  }

  function onKeepFieldInput(final) {
    const kp = selectedKeep();
    if (!kp || !kp.min || S.status === 'running') return;
    const v = KEEP_FIELDS.map((id) => parseFloat($(id).value));
    const a = v.slice(0, 3), b = v.slice(3);
    if (v.every(Number.isFinite) && a.every((x, d) => x < b[d])) {
      if (a.some((x, d) => x !== kp.min[d]) || b.some((x, d) => x !== kp.max[d])) {
        kp.min = a;
        kp.max = b;
        invalidateResult();
        renderKeepList();
        buildKeepView();
      }
    } else if (final) syncKeepFields();
  }

  function convertKeepToVoxels() {
    const kp = selectedKeep(), g = S.grid;
    if (!kp || !kp.min || !g || S.status === 'running') return;
    const cells = kp.cells && kp.cellsGrid === g ? kp.cells : new Set();
    forEachBoxCell(g, kp, (e) => cells.add(e));
    kp.cells = cells;
    kp.cellsGrid = g;
    kp.min = kp.max = null;
    syncKeepFields();
    renderKeepList();
    buildKeepView();
  }

  function setPaint(on) {
    on = !!on && !!S.grid && S.status !== 'running';
    S.paint.on = on;
    $('keep-paint').setAttribute('aria-pressed', String(on));
    $('paint-opts').hidden = !on;
    $('vp-paint').hidden = !on;
    if (on) {
      $('sec-keep').open = true;
      if (S.selection) selectTris(null);
    }
  }

  function setBrushMode(mode) {
    S.paint.mode = mode;
    document.querySelectorAll('[data-brush]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.brush === mode)));
    $('vp-paint').textContent = `Paint voxels (${mode === 'add' ? 'add' : 'erase'}): drag on the part \u00b7 Esc to leave`;
  }

  function defaultKeepFields() {
    const m = S.mesh;
    const c = [0, 1, 2].map((d) => (m.bmin[d] + m.bmax[d]) / 2);
    const s = [0, 1, 2].map((d) => (m.bmax[d] - m.bmin[d]) * 0.15);
    ['x', 'y', 'z'].forEach((a, d) => {
      $(`keep-${a}0`).value = +(c[d] - s[d]).toPrecision(4);
      $(`keep-${a}1`).value = +(c[d] + s[d]).toPrecision(4);
    });
  }

  // ---------------------------------------------------------------- part + grid lifecycle
  function setMesh(mesh, opts = {}) {
    stopWorker();
    S.mesh = mesh;
    S.bcs = [];
    S.keeps = [];
    S.keepSel = null;
    setPaint(false);
    $('keep-note').hidden = true;
    S.selection = null;
    S.res = opts.res || S.res;
    $('res').value = S.res;
    invalidateResult();
    $('part-name').textContent = mesh.name;
    $('part-tris').textContent = fmt.int(mesh.ntri);
    $('part-size').textContent = [0, 1, 2].map((d) => fmt.len(mesh.bmax[d] - mesh.bmin[d])).join(' \u00d7 ');
    $('part-tag').textContent = '';
    const pn = $('part-note');
    pn.hidden = !mesh.openEdges && !mesh.badEdges;
    pn.textContent = mesh.openEdges
      ? `The mesh is not closed (${fmt.int(mesh.openEdges)} open edge${mesh.openEdges === 1 ? '' : 's'}), so the inside of the part is guessed and the voxels may be wrong. Repair it in a CAD or mesh tool for reliable results.`
      : `The mesh has ${fmt.int(mesh.badEdges)} edge${mesh.badEdges === 1 ? '' : 's'} shared by more than two triangles; the voxels may be wrong.`;
    buildDesignMesh();
    fitView();
    regrid();
    for (const bc of opts.bcs || []) {
      const seed = findTri(mesh, bc.normal, bc.at);
      if (seed >= 0) addBC(bc.kind, growFace(mesh, seed), bc.force);
    }
    renderBCList();
    renderKeepList();
    defaultKeepFields();
    syncKeepFields();
    buildSelectionMesh();
    renderSelCard();
    buildBCMarkers();
    setView('design');
    setStatus('idle', 'Ready. Click a face to add supports or loads, then Run.');
  }

  function findTri(mesh, normal, at) {
    let best = -1, bestD = Infinity;
    for (let t = 0; t < mesh.ntri; t++) {
      const n = mesh.normals;
      if (n[3 * t] * normal[0] + n[3 * t + 1] * normal[1] + n[3 * t + 2] * normal[2] < 0.999) continue;
      const d = distSqPointTri(at[0], at[1], at[2], mesh.pos, 9 * t);
      if (d < bestD) { bestD = d; best = t; }
    }
    return best;
  }

  function regrid() {
    if (!S.mesh) return;
    const g = buildGrid(S.mesh, S.res);
    S.grid = g;
    if (g.capped) {
      S.res = g.res;
      $('res').value = g.res;
    }
    $('res-out').textContent = g.res;
    $('grid-dims').textContent = `${g.nx} \u00d7 ${g.ny} \u00d7 ${g.nz}`;
    $('grid-h').textContent = fmt.len(g.h);
    $('grid-active').textContent = fmt.int(g.count);
    $('grid-tag').textContent = fmt.int(g.count);
    updateEstimate();
    const note = $('grid-note');
    note.hidden = !g.capped && g.count > 0;
    note.className = 'note warn';
    note.textContent = g.count === 0 ? 'No voxels inside the part: the mesh may not be closed, or the grid is too coarse.' : `Capped at ${g.res} to stay under ${fmt.int(MAX_ACTIVE)} elements.`;
    for (const bc of S.bcs) bc.nodes = resolveNodes(S.mesh, g, bc.tris);
    if (S.selection) S.selection.nodes = resolveNodes(S.mesh, g, S.selection.tris);
    let before = 0, after = 0, nr = 0;
    for (const kp of S.keeps) {
      if (!kp.cells || !kp.cellsGrid || kp.cellsGrid === g) continue;
      before += kp.cells.size;
      kp.cells = remapCells(kp.cellsGrid, g, kp.cells);
      kp.cellsGrid = g;
      after += kp.cells.size;
      nr++;
    }
    if (nr) {
      const kn = $('keep-note');
      kn.hidden = false;
      kn.textContent = `Painted voxels are tied to the grid, so they were remapped to the new ${g.nx} \u00d7 ${g.ny} \u00d7 ${g.nz} grid by cell centre: ${fmt.int(before)} \u2192 ${fmt.int(after)} cells.`;
    }
    disposeObj(V.vox);
    V.vox = null;
    invalidateResult();
    renderBCList();
    renderKeepList();
    renderSelCard();
    buildBCMarkers();
  }

  function updateEstimate() {
    const g = S.grid;
    if (!g) return;
    const ms = estA * Math.pow(g.count / 1e4, 4 / 3);
    const it = +$('maxit').value;
    const el = $('grid-est');
    el.textContent = `\u2248 ${fmt.ms(ms)}`;
    el.title = `${it} iterations \u2248 ${fmt.ms(ms * it)} on this device`;
  }

  function invalidateResult() {
    S.rho8 = null;
    S.rhoFinal = null;
    S.surface = null;
    S.history = [];
    disposeObj(V.surf);
    V.surf = null;
    if (V.vox) V.vox.count = 0;
    setView('design');
    setViewButtons();
    updateSurfaceInfo();
    updateReadout(null);
    drawSpark();
  }

  function setViewButtons() {
    const has = !!S.rho8;
    document.querySelector('.seg [data-view="voxels"]').disabled = !has;
    document.querySelector('.seg [data-view="surface"]').disabled = !has;
  }

  // ---------------------------------------------------------------- solver worker
  function makeWorker() {
    const src = '(' + topopWorkerMain.toString() + ')();';
    const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
    const w = new Worker(url);
    URL.revokeObjectURL(url);
    return w;
  }

  function runSelfTest() {
    try {
      const w = makeWorker();
      w.onmessage = (ev) => {
        if (ev.data.type !== 'selftest') return;
        S.selftest = ev.data;
        const r = ev.data;
        const line = `[top-op lite] KE self-test: symmetric=${r.symmetric} (max |K-K\u1d40| ${r.asym.toExponential(1)}), zero modes=${r.zeroModes}/6, \u03bb+min=${r.lmin.toFixed(4)}, \u03bbmax=${r.lmax.toFixed(4)}, kernel err=${r.kernelError.toExponential(1)}`;
        if (r.ok) console.info(line + ' \u2014 ok');
        else console.warn(line + ' \u2014 FAILED');
        w.terminate();
      };
      w.postMessage({ type: 'selftest' });
    } catch (err) {
      console.warn('[top-op lite] could not start the solver worker:', err);
    }
  }

  function stopWorker() {
    if (S.worker) {
      S.worker.terminate();
      S.worker = null;
    }
    if (S.status === 'running') S.status = 'idle';
  }

  function rejectRun(msg) {
    S.status = 'error';
    S.lastError = msg;
    setStatus('error', msg);
  }

  function startRun() {
    const g = S.grid;
    if (!g || S.status === 'running') return;
    const sup = S.bcs.filter((b) => b.kind === 'support' && b.nodes.length);
    const lds = S.bcs.filter((b) => b.kind === 'load' && b.nodes.length && b.force.some((x) => x !== 0));
    if (!g.count) return rejectRun('No voxels to optimize. Increase the resolution or check the mesh.');
    if (!sup.length) return rejectRun('Add at least one support: click a face, then Fix as support.');
    if (!lds.length) return rejectRun('Add at least one load with a non-zero force.');
    const fixedNodes = new Int32Array(sup.reduce((s, b) => s + b.nodes.length, 0));
    let o = 0;
    for (const b of sup) { fixedNodes.set(b.nodes, o); o += b.nodes.length; }
    // A force on fixed nodes goes straight into the support: reject it instead of solving for nothing.
    const isFixed = new Set(fixedNodes);
    let nl = 0;
    for (const b of S.bcs) {
      if (b.kind !== 'load') continue;
      nl++;
      if (lds.includes(b) && b.nodes.every((id) => isFixed.has(id))) {
        return rejectRun(`Load ${nl} is on a face that is fully fixed by a support, so it cannot deform the part. Move the load to another face or remove that support.`);
      }
    }
    const loads = [];
    for (const b of lds) {
      const n = b.nodes.length;
      for (const id of b.nodes) loads.push(id, b.force[0] / n, b.force[1] / n, b.force[2] / n);
    }
    invalidateResult();
    stopWorker();
    const w = makeWorker();
    S.worker = w;
    const maxIter = +$('maxit').value;
    const volfrac = +$('volfrac').value;
    S.runInfo = { maxIter, volfrac, ne: g.count, t0: performance.now() };
    S.status = 'running';
    S.lastError = null;
    setRunning(true);
    setStatus('running', `Optimizing ${fmt.int(g.count)} elements\u2026`);
    w.onmessage = (ev) => onWorkerMessage(w, ev.data);
    w.onerror = (ev) => {
      ev.preventDefault?.();
      finishRun('error', ev.message || 'Solver crashed.');
    };
    w.postMessage({
      type: 'run', nx: g.nx, ny: g.ny, nz: g.nz, h: g.h, active: g.active, passive: buildPassive(g, S.keeps),
      fixedNodes, loads: Float64Array.from(loads), volfrac, maxIter, rmin: +$('rmin').value, penal: 3, move: 0.2,
    });
  }

  function onWorkerMessage(w, m) {
    if (w !== S.worker) return;
    if (m.type === 'iter') {
      S.history.push({ it: m.it, c: m.c, vol: m.vol, change: m.change, ms: m.ms, cg: m.cgIters, tol: m.tol, live: m.live });
      S.rho8 = m.rho;
      S.surfaceStale = true;
      updateReadout(m);
      drawSpark();
      setViewButtons();
      if (S.history.length === 1 && S.view === 'design') setView('voxels');
      else scheduleVoxels();
      setStatus('running', `Iteration ${m.it} / ${S.runInfo.maxIter} \u00b7 ${fmt.ms(m.ms)} per iteration`);
    } else if (m.type === 'done') {
      S.rhoFinal = m.rho;
      const reason = { converged: 'Converged', 'max-iterations': 'Reached max iterations', stopped: 'Stopped' }[m.reason] || m.reason;
      finishRun('done', `${reason} after ${m.it} iterations.`);
    } else if (m.type === 'error') {
      finishRun('error', m.message);
    }
  }

  function finishRun(state, msg) {
    const w = S.worker;
    S.worker = null;
    w?.terminate();
    S.status = state === 'error' ? 'error' : 'done';
    if (state === 'error') S.lastError = msg;
    setRunning(false);
    const h = S.history;
    if (h.length >= 3 && S.runInfo) {
      const avg = h.slice(1).reduce((s, x) => s + x.ms, 0) / (h.length - 1);
      S.runInfo.avgMs = avg;
      estA = avg / Math.pow(S.runInfo.ne / 1e4, 4 / 3);
      updateEstimate();
    }
    if (S.rho8 || S.rhoFinal) {
      S.surfaceStale = false;
      buildSurface();
      setView('surface');
    }
    setStatus(state, msg);
  }

  function setRunning(on) {
    $('run').disabled = on;
    $('stop').disabled = !on;
    for (const id of ['res', 'volfrac', 'maxit', 'rmin', 'file', 'example', 'keep-add', 'keep-mode', ...KEEP_FIELDS]) $(id).disabled = on;
    if (on) setPaint(false);
    renderBCList();
    renderKeepList();
    buildKeepView();
    renderSelCard();
    updateExportUi();
  }

  function setStatus(state, text) {
    $('vp-status').dataset.state = state;
    $('vp-status-text').textContent = text;
    const note = $('run-note');
    if (state === 'error') {
      note.className = 'note err';
      note.textContent = text;
    } else {
      note.className = 'note';
      note.textContent = 'SIMP, penalty 3, density filter, OC update. Compliance is f\u00b7u with E = 1.';
    }
  }

  function updateReadout(m) {
    const max = S.runInfo?.maxIter || +$('maxit').value;
    $('ro-it').textContent = m ? `${m.it} / ${max}` : '\u2013';
    $('ro-c').textContent = m ? fmt.sci(m.c) : '\u2013';
    $('ro-v').textContent = m ? fmt.num(m.vol, 3) : '\u2013';
    $('ro-ch').textContent = m ? fmt.num(m.change, 3) : '\u2013';
    $('ro-ms').textContent = m ? fmt.int(Math.round(m.ms)) : '\u2013';
    $('ro-cg').textContent = m ? fmt.int(m.cgIters) : '\u2013';
    $('progress').style.width = m ? `${Math.min(100, (100 * m.it) / max)}%` : '0';
    $('opt-tag').textContent = m ? `it ${m.it}` : '';
  }

  function drawSpark() {
    const cv = $('spark');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const data = S.history.map((x) => x.c);
    const pad = { l: 6, r: 6, t: 16, b: 6 };
    ctx.strokeStyle = token('--line');
    ctx.lineWidth = 1;
    for (let i = 0; i <= 2; i++) {
      const y = Math.round(pad.t + ((h - pad.t - pad.b) * i) / 2) + 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    }
    ctx.font = `10px ${token('--font-num')}`;
    ctx.fillStyle = token('--muted');
    ctx.textBaseline = 'top';
    if (data.length < 2) {
      ctx.fillText('compliance history', pad.l, 3);
      return;
    }
    const lo = Math.min(...data), hi = Math.max(...data);
    const span = hi - lo || 1;
    const n = Math.max(data.length, 2);
    const X = (i) => pad.l + ((w - pad.l - pad.r) * i) / (n - 1);
    const Y = (v) => pad.t + (h - pad.t - pad.b) * (1 - (v - lo) / span);
    ctx.beginPath();
    data.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    const accent = token('--accent');
    ctx.save();
    ctx.lineTo(X(data.length - 1), h - pad.b);
    ctx.lineTo(X(0), h - pad.b);
    ctx.closePath();
    ctx.globalAlpha = 0.14;
    ctx.fillStyle = accent;
    ctx.fill();
    ctx.restore();
    ctx.beginPath();
    data.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    const lx = X(data.length - 1), ly = Y(data[data.length - 1]);
    ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2); ctx.fillStyle = accent; ctx.fill();
    ctx.fillStyle = token('--muted');
    ctx.fillText(`c ${fmt.sci(data[0])} \u2192 ${fmt.sci(data[data.length - 1])}`, pad.l, 3);
  }

  // ---------------------------------------------------------------- export
  function updateSurfaceInfo() {
    const s = S.surface;
    $('surf-info').textContent = s ? `${fmt.int(s.ntri)} triangles \u00b7 V ${fmt.len(s.volume)} u\u00b3` : 'run the optimizer first';
    $('res-tag').textContent = s ? `${fmt.int(s.ntri)} tri` : '';
    updateExportUi();
  }

  function asciiText(s) {
    if (s && s.ascii === undefined) s.ascii = s.ntri * 120 < 2e6 ? TopopMesh.stlAscii(s.positions, s.indices) : null;
    return s ? s.ascii : null;
  }
  const COPY_LIMIT = 2e6;

  function updateExportUi() {
    const s = S.surface;
    const busy = S.status === 'running';
    const exp = $('export'), copy = $('copy'), note = $('export-note');
    const text = asciiText(s);
    const canCopy = !!text && text.length < COPY_LIMIT;
    copy.disabled = busy || !canCopy;
    if (S.downloads === undefined) {
      exp.disabled = true;
      note.textContent = 'Checking whether this view can save files\u2026';
    } else if (S.downloads === null) {
      exp.hidden = true;
      note.textContent = canCopy || !s
        ? 'Saving files is not available in this view. Copy the ASCII STL and paste it into a .stl file.'
        : 'Saving files is not available in this view, and this mesh is too large to copy. Lower the resolution or raise the threshold.';
    } else {
      exp.hidden = false;
      exp.disabled = busy || !s;
      note.textContent = s ? (canCopy ? 'Binary STL in a .zip, same units as the input part.' : 'Binary STL in a .zip. Too large to copy as text (over 2 MB).') : 'Run the optimizer to get a surface.';
    }
  }

  async function saveStl() {
    const s = S.surface;
    const dl = S.downloads;
    if (!s || !dl) return;
    // .stl is not on the downloads allowlist; .zip is, so the binary STL is saved inside a zip.
    const zip = TopopMesh.zipOne('topop-result.stl', TopopMesh.stlBinary(s.positions, s.indices));
    const note = $('export-note');
    try {
      await dl.save({ filename: 'topop-result.zip', data: zip });
      note.textContent = 'Saved topop-result.zip (contains topop-result.stl).';
    } catch (err) {
      const code = err && err.code;
      if (['unavailable', 'not_granted', 'capability_disabled', 'capability_removed', 'host_unavailable'].includes(code)) {
        S.downloads = null;
        updateExportUi();
      }
      note.textContent = saveErrorText(err);
    }
  }

  function saveErrorText(err) {
    const code = err && err.code;
    if (code === 'declined') return 'Save cancelled.';
    if (code === 'extension_not_enabled') return 'Zip files cannot be saved in this view. Copy the ASCII STL instead.';
    if (code === 'rate_limited') return 'Another save is in progress. Try again in a moment.';
    if (code === 'host_unavailable') return 'This app cannot save files. Open the page in a browser to save it, or copy the ASCII STL.';
    return 'The file could not be saved here. Copy the ASCII STL instead.';
  }

  async function copyAscii() {
    const s = S.surface;
    const text = asciiText(s);
    if (!text || text.length >= COPY_LIMIT) return;
    const note = $('export-note');
    try {
      await navigator.clipboard.writeText(text);
      note.textContent = `Copied ${fmt.int(s.ntri)} triangles as ASCII STL (${(text.length / 1e6).toFixed(2)} MB).`;
    } catch {
      const ta = $('copy-fallback');
      ta.classList.remove('visually-hidden');
      ta.removeAttribute('aria-hidden');
      ta.value = text;
      ta.style.cssText = 'width:100%;height:90px;font:11px var(--font-num);';
      ta.focus();
      ta.select();
      note.textContent = 'Clipboard access was refused. The STL text is selected above: press Ctrl+C / \u2318C.';
    }
  }

  // ---------------------------------------------------------------- import
  async function importFile(file) {
    if (!file || S.status === 'running') return;
    try {
      setStatus('idle', `Reading ${file.name}\u2026`);
      const buf = await file.arrayBuffer();
      if (!buf.byteLength) {
        setStatus('error', `${file.name} is empty (0 bytes). Choose an STL file that contains a mesh.`);
        return;
      }
      let pos = null;
      try {
        const geo = new THREE.STLLoader().parse(buf);
        pos = Float32Array.from(geo.getAttribute('position').array);
        geo.dispose();
      } catch {
        pos = null;
      }
      if (!pos || pos.length < 9) {
        setStatus('error', `${file.name} contains no triangles. It is not an STL file, or it is damaged.`);
        return;
      }
      let finite = true;
      for (let i = 0; i < pos.length; i++) if (!Number.isFinite(pos[i])) { finite = false; break; }
      if (!finite) {
        setStatus('error', `${file.name} has invalid vertex coordinates (NaN or infinity), so it cannot be used.`);
        return;
      }
      const mesh = prepareMesh(pos, file.name);
      const sel = $('example');
      sel.querySelector('[value="custom"]').hidden = false;
      sel.value = 'custom';
      setMesh(mesh, { res: S.res });
    } catch (err) {
      setStatus('error', `Could not read ${file.name} as STL (${err.message || err}).`);
    }
  }

  function loadExample(key) {
    const ex = EXAMPLES[key];
    setMesh(prepareMesh(ex.make(), ex.name), { res: ex.res, bcs: ex.bcs });
  }

  // ---------------------------------------------------------------- wiring
  function bindUI() {
    const bindSlider = (id, fmtFn, onChange, delay = 0) => {
      let timer = 0;
      $(id).addEventListener('input', () => {
        $(id + '-out').textContent = fmtFn(+$(id).value);
        clearTimeout(timer);
        timer = setTimeout(onChange, delay);
      });
      $(id + '-out').textContent = fmtFn(+$(id).value);
    };
    bindSlider('res', (v) => v, () => { S.res = +$('res').value; regrid(); }, 160);
    bindSlider('volfrac', (v) => v.toFixed(2), () => {});
    bindSlider('maxit', (v) => v, updateEstimate);
    bindSlider('rmin', (v) => v.toFixed(1), () => {});
    bindSlider('thr', (v) => v.toFixed(2), () => {
      S.surfaceStale = true;
      $('rho-bar-lo').textContent = (+$('thr').value).toFixed(2);
      if (S.view === 'voxels') updateVoxels();
      if (S.view === 'surface') scheduleSurface();
    });
    $('smooth').addEventListener('change', () => { S.surfaceStale = true; if (S.view === 'surface') buildSurface(); });
    $('run').addEventListener('click', startRun);
    $('stop').addEventListener('click', () => {
      if (S.worker) {
        S.worker.postMessage({ type: 'stop' });
        setStatus('running', 'Stopping after this iteration\u2026');
      }
    });
    document.querySelectorAll('.seg [data-view]').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));
    $('fit').addEventListener('click', fitView);
    $('example').addEventListener('change', (e) => { if (EXAMPLES[e.target.value]) loadExample(e.target.value); });
    $('file').addEventListener('change', (e) => {
      importFile(e.target.files[0]);
      e.target.value = '';
    });
    $('sel-clear').addEventListener('click', () => selectTris(null));
    $('add-support').addEventListener('click', () => {
      if (!S.selection) return;
      addBC('support', S.selection.tris);
      selectTris(null);
    });
    $('add-load').addEventListener('click', () => {
      if (!S.selection) return;
      const f = ['fx', 'fy', 'fz'].map((id) => +$(id).value || 0);
      if (!f.some((x) => x !== 0)) {
        setStatus('error', 'Enter a non-zero force (Fx, Fy, Fz in N) before adding the load.');
        return;
      }
      addBC('load', S.selection.tris, f);
      selectTris(null);
    });
    $('keep-add').addEventListener('click', () => {
      const v = (id) => +$(id).value;
      const a = [v('keep-x0'), v('keep-y0'), v('keep-z0')], b = [v('keep-x1'), v('keep-y1'), v('keep-z1')];
      if (a.concat(b).some((x) => !Number.isFinite(x))) return;
      const lo = a.map((x, i) => Math.min(x, b[i])), hi = a.map((x, i) => Math.max(x, b[i]));
      if (lo.some((x, i) => x === hi[i])) return;
      const kp = { id: S.nextId++, mode: $('keep-mode').value, min: lo, max: hi, cells: null, cellsGrid: null };
      S.keeps.push(kp);
      invalidateResult();
      selectKeep(kp.id);
    });
    for (const id of KEEP_FIELDS) {
      $(id).addEventListener('input', () => onKeepFieldInput(false));
      $(id).addEventListener('change', () => onKeepFieldInput(true));
    }
    $('keep-mode').addEventListener('change', () => {
      const kp = selectedKeep();
      if (!kp || S.status === 'running') return;
      kp.mode = $('keep-mode').value;
      invalidateResult();
      syncKeepFields();
      renderKeepList();
      buildKeepView();
    });
    $('keep-convert').addEventListener('click', convertKeepToVoxels);
    $('keep-paint').addEventListener('click', () => setPaint(!S.paint.on));
    document.querySelectorAll('[data-brush]').forEach((b) => b.addEventListener('click', () => setBrushMode(b.dataset.brush)));
    bindSlider('brush', (v) => v, () => { S.paint.radius = +$('brush').value; });
    setBrushMode('add');
    $('export').addEventListener('click', saveStl);
    $('copy').addEventListener('click', copyAscii);
    const about = $('about');
    $('about-open').addEventListener('click', () => { about.hidden = false; $('about-close').focus(); });
    $('about-close').addEventListener('click', () => { about.hidden = true; });
    about.addEventListener('click', (e) => { if (e.target === about) about.hidden = true; });
    document.addEventListener('keydown', (e) => {
      if (e.key !== 'Escape') return;
      if (!about.hidden) about.hidden = true;
      else if (S.paint.on) setPaint(false);
      else {
        selectTris(null);
        if (S.keepSel !== null) selectKeep(null);
      }
    });
    // Drag and drop anywhere on the page.
    let depth = 0;
    const drop = $('drop');
    window.addEventListener('dragenter', (e) => { if ([...(e.dataTransfer?.types || [])].includes('Files')) { depth++; drop.hidden = false; e.preventDefault(); } });
    window.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) drop.hidden = true; });
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => {
      e.preventDefault();
      depth = 0;
      drop.hidden = true;
      const f = e.dataTransfer?.files?.[0];
      if (f) importFile(f);
    });
    // Theme changes: the host sets data-theme, or the OS scheme flips.
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener?.('change', refreshThemeColors);
    new MutationObserver(refreshThemeColors).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    new ResizeObserver(drawSpark).observe($('spark'));
  }

  async function initDownloads() {
    let dl = null;
    try {
      dl = (await window.claude?.use?.('downloads')) ?? null;
    } catch {
      dl = null;
    }
    S.downloads = dl;
    updateExportUi();
  }

  function boot() {
    if (!window.THREE || !THREE.OrbitControls || !THREE.STLLoader || !window.TopopMesh || typeof topopWorkerMain !== 'function') {
      const d = document.createElement('div');
      d.className = 'fatal';
      d.textContent = 'The 3D library could not be loaded, so the viewer cannot start. Check the connection and reload.';
      $('viewport').append(d);
      $('vp-status').hidden = true;
      return;
    }
    initView();
    initPicking();
    bindUI();
    runSelfTest();
    initDownloads();
    loadExample('bracket');
  }

  // test hooks
  S.api = {
    loadExample, startRun, setView, buildSurface, regrid, selectTris, growFace, addBC, removeBC, prepareMesh, voxelize,
    fit: fitView,
    selectKeep, setPaint, setBrushMode, regionCells, remapCells, buildPassive,
    // Screen positions of the selected box's face handles: base and tip of each arrow.
    keepHandles: () => {
      if (!V.handles) return [];
      V.camera.updateMatrixWorld();
      updateHandleScale();
      return V.handles.children.map((hd) => {
        const { a, b } = handleScreen(hd);
        return { axis: hd.userData.axis, side: hd.userData.side, base: [a.x, a.y], tip: [b.x, b.y] };
      });
    },
    controls: () => V.controls,
    camera: () => V.camera.position.toArray(),
    // World point -> client (page) coordinates on the canvas.
    project: (p) => {
      const v = new THREE.Vector3(...p).project(V.camera);
      const r = $('gl').getBoundingClientRect();
      return [r.left + ((v.x + 1) / 2) * r.width, r.top + ((1 - v.y) / 2) * r.height];
    },
  };
  boot();
})();
