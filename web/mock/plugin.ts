// Mock backend as a Vite plugin (enabled with VITE_MOCK=1). Implements every endpoint of
// docs/PLAN.md section 3 in memory so the frontend is testable without the Python server.
// Deterministic by design: ids are counters/content hashes, runs advance on a fixed 150 ms tick.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { resolve } from 'node:path';
import type { Plugin } from 'vite';
import { type WebSocket, WebSocketServer } from 'ws';
import type { components } from '../src/api/types.gen.ts';
import {
  adjacencyPairs,
  boxSTL,
  bboxOf,
  computeFacets,
  isWatertight,
  type MeshGeom,
  meshBuffer,
  parseMultipart,
  parseSTL,
  StlError,
  volumeOf,
} from './stl.ts';

type S = components['schemas'];
type ProjectIn = S['ProjectIn'];
type Project = S['Project'];
type RunInfo = S['RunInfo'];
type VoxelStats = S['VoxelStats'];
type Selection = S['Selection'];

const TICK_MS = 150;
const MAX_MOCK_CELLS = 1_500_000;
const PNG_1X1 = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==',
  'base64',
);

interface StoredMesh {
  info: S['MeshInfo'];
  geom: MeshGeom;
  adjacency: Uint32Array;
}

interface Ctx {
  req: IncomingMessage;
  res: ServerResponse;
  url: URL;
  params: string[];
  body: () => Promise<Buffer>;
}

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

function json(res: ServerResponse, data: unknown, status = 200): void {
  const body = Buffer.from(JSON.stringify(data));
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': body.length });
  res.end(body);
}

function bin(res: ServerResponse, buf: Buffer, type: string, extra: Record<string, string> = {}): void {
  res.writeHead(200, { 'Content-Type': type, 'Content-Length': buf.length, ...extra });
  res.end(buf);
}

function apply(m: number[], p: number[]): number[] {
  const [x, y, z] = p as [number, number, number];
  return [
    m[0]! * x + m[4]! * y + m[8]! * z + m[12]!,
    m[1]! * x + m[5]! * y + m[9]! * z + m[13]!,
    m[2]! * x + m[6]! * y + m[10]! * z + m[14]!,
  ];
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Exports exist once a run is done (or cancelled with a partial result); before that the server answers 409. */
function finished(run: { info: RunInfo }): void {
  const { id, status } = run.info;
  if (status !== 'done' && status !== 'cancelled') throw new HttpError(409, `run ${id} is ${status}; no result to export`);
}

function worldBBox(mesh: StoredMesh, transform: number[] | undefined): [number[], number[]] {
  const [lo, hi] = mesh.info.bbox as [number[], number[]];
  const t = transform && transform.length === 16 ? transform : IDENTITY;
  const wlo = [Infinity, Infinity, Infinity];
  const whi = [-Infinity, -Infinity, -Infinity];
  for (let c = 0; c < 8; c++) {
    const q = apply(t, [c & 1 ? hi[0]! : lo[0]!, c & 2 ? hi[1]! : lo[1]!, c & 4 ? hi[2]! : lo[2]!]);
    for (let k = 0; k < 3; k++) {
      wlo[k] = Math.min(wlo[k]!, q[k]!);
      whi[k] = Math.max(whi[k]!, q[k]!);
    }
  }
  return [wlo, whi];
}

function voxelStats(project: ProjectIn, mesh: StoredMesh): VoxelStats {
  const grid = project.grid ?? { elements_along_longest: 60, padding: 1 };
  const [lo, hi] = worldBBox(mesh, project.design_mesh?.transform);
  const size = hi.map((v, k) => v - lo[k]!);
  const h = Math.max(...size) / grid.elements_along_longest;
  const inner = size.map((s) => Math.max(1, Math.ceil(s / h - 1e-9)));
  const pad = grid.padding;
  const [nx, ny, nz] = inner.map((n) => n + 2 * pad) as [number, number, number];
  const bboxVol = size[0]! * size[1]! * size[2]!;
  const fill = mesh.info.volume && bboxVol > 0 ? Math.min(1, Math.max(0.05, mesh.info.volume / bboxVol)) : 0.4;
  const nActive = Math.round(inner[0]! * inner[1]! * inner[2]! * fill);
  const nNodes = (nx + 1) * (ny + 1) * (nz + 1);
  const warnings: string[] = [];
  if (nActive > 300_000) warnings.push(`${nActive} active elements: expect very slow iterations and > 4 GB`);
  else if (nActive > 150_000) warnings.push(`${nActive} active elements: iterations will be slow`);
  return {
    nx,
    ny,
    nz,
    h,
    origin: lo.map((v) => v - pad * h),
    n_active: nActive,
    n_free: nActive,
    n_passive_solid: 0,
    n_passive_void: 0,
    n_nodes: nNodes,
    n_dof: 3 * nNodes,
    est_bytes: nActive * 12_000,
    est_sec_per_iter: Math.round(nActive * 1.5e-5 * 1000) / 1000,
    warnings,
  };
}

function round4(v: number): number {
  return Math.round(v * 1e4) / 1e4;
}

/** A few hundred fake points on the bbox face the selection points towards. */
function resolveFake(sel: Selection, mesh: StoredMesh, project: ProjectIn): S['ResolvedNodes'] {
  const [lo, hi] = worldBBox(mesh, project.design_mesh?.transform);
  const centre = lo.map((v, k) => (v + hi[k]!) / 2);
  const half = lo.map((v, k) => (hi[k]! - v) / 2);
  let normal = [0, 0, 1];
  let extLo = lo.slice();
  let extHi = hi.slice();
  const g = mesh.geom;

  const fromFaces = (faces: number[]) => {
    const n = [0, 0, 0];
    const l = [Infinity, Infinity, Infinity];
    const h = [-Infinity, -Infinity, -Infinity];
    for (const f of faces) {
      if (f < 0 || f >= g.areas.length) continue;
      for (let k = 0; k < 3; k++) n[k]! += g.normals[f * 3 + k]! * g.areas[f]!;
      for (let v = 0; v < 3; v++)
        for (let k = 0; k < 3; k++) {
          const x = g.positions[g.tris[f * 3 + v]! * 3 + k]!;
          l[k] = Math.min(l[k]!, x);
          h[k] = Math.max(h[k]!, x);
        }
    }
    if (Number.isFinite(l[0]!)) {
      normal = n;
      extLo = l;
      extHi = h;
    }
  };

  switch (sel.kind) {
    case 'faces':
      fromFaces(sel.face_ids);
      break;
    case 'facets': {
      const facets = computeFacets(g, mesh.adjacency, sel.angle_deg);
      const picked = facets.filter((f) => sel.facet_ids.includes(f.id));
      if (picked.length) {
        normal = [0, 0, 0].map((_, k) => picked.reduce((s, f) => s + f.normal[k]! * f.area, 0));
        extLo = [0, 1, 2].map((k) => Math.min(...picked.map((f) => f.bbox[0]![k]!)));
        extHi = [0, 1, 2].map((k) => Math.max(...picked.map((f) => f.bbox[1]![k]!)));
      }
      break;
    }
    case 'normal':
      normal = sel.direction;
      break;
    case 'plane':
      normal = sel.normal;
      break;
    default: {
      const c = apply(sel.transform ?? IDENTITY, [0, 0, 0]);
      const size = sel.size ?? [1, 1, 1];
      normal = c.map((v, k) => (v - centre[k]!) / (half[k]! || 1));
      extLo = c.map((v, k) => v - size[sel.kind === 'box' ? k : 0]! / 2);
      extHi = c.map((v, k) => v + size[sel.kind === 'box' ? k : 0]! / 2);
    }
  }

  let axis = 0;
  for (let k = 1; k < 3; k++) if (Math.abs(normal[k]!) > Math.abs(normal[axis]!)) axis = k;
  const sign = normal[axis]! >= 0 ? 1 : -1;
  const [u, v] = [0, 1, 2].filter((k) => k !== axis) as [number, number];
  const clip = (k: number) => [Math.max(lo[k]!, extLo[k]!), Math.min(hi[k]!, extHi[k]!)] as const;
  const [u0, u1] = clip(u);
  const [v0, v1] = clip(v);
  const nu = 20;
  const nv = 15;
  const xyz: number[][] = [];
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      const p = [0, 0, 0];
      p[axis] = sign > 0 ? hi[axis]! : lo[axis]!;
      p[u] = u0 + ((u1 - u0) * i) / (nu - 1);
      p[v] = v0 + ((v1 - v0) * j) / (nv - 1);
      xyz.push(p.map(round4));
    }
  }
  return { count: xyz.length, xyz, truncated: false };
}

function densityFrame(
  it: number,
  maxIter: number,
  stats: VoxelStats,
  pad: number,
): Buffer {
  const { nx, ny, nz } = stats;
  const buf = Buffer.alloc(16 + nx * ny * nz);
  buf.writeUInt32LE(it, 0);
  buf.writeUInt32LE(nx, 4);
  buf.writeUInt32LE(ny, 8);
  buf.writeUInt32LE(nz, 12);
  const r = 0.95 - 0.45 * (it / maxIter); // blob radius (fraction of the half-extent) shrinks
  const cx = nx / 2;
  const cy = ny / 2;
  const cz = nz / 2;
  const ax = Math.max(1, nx / 2 - pad);
  const ay = Math.max(1, ny / 2 - pad);
  const az = Math.max(1, nz / 2 - pad);
  let o = 16;
  for (let ix = 0; ix < nx; ix++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let iz = 0; iz < nz; iz++, o++) {
        const inPad = ix < pad || iy < pad || iz < pad || ix >= nx - pad || iy >= ny - pad || iz >= nz - pad;
        if (inPad) continue;
        const d = Math.hypot((ix + 0.5 - cx) / ax, (iy + 0.5 - cy) / ay, (iz + 0.5 - cz) / az);
        const rho = Math.min(1, Math.max(0, 1 - (d - r) / 0.15));
        buf[o] = Math.round(rho * 255);
      }
    }
  }
  return buf;
}

class MockRun {
  /** One run at a time, like the server's semaphore; later runs wait in `waiting` and report `queued`. */
  static active: MockRun | null = null;
  static waiting: MockRun[] = [];

  info: RunInfo;
  clients = new Set<WebSocket>();
  private timer: NodeJS.Timeout | null = null;
  private stats: VoxelStats;
  private pad: number;

  constructor(
    id: string,
    readonly projectId: string,
    readonly project: ProjectIn,
    mesh: StoredMesh,
  ) {
    let stats = voxelStats(project, mesh);
    let k = 1;
    while ((Math.ceil(stats.nx / k) * Math.ceil(stats.ny / k) * Math.ceil(stats.nz / k)) > MAX_MOCK_CELLS) k++;
    if (k > 1) stats = { ...stats, nx: Math.ceil(stats.nx / k), ny: Math.ceil(stats.ny / k), nz: Math.ceil(stats.nz / k), h: stats.h * k };
    this.stats = stats;
    this.pad = Math.ceil((project.grid?.padding ?? 1) / k);
    this.info = {
      id,
      project_id: projectId,
      status: 'queued',
      created_at: new Date().toISOString(),
      finished_at: null,
      history: [],
      stats,
      error: null,
      outcome: null,
      message: null,
    };
  }

  private send(msg: unknown): void {
    const text = JSON.stringify(msg);
    for (const ws of this.clients) if (ws.readyState === ws.OPEN) ws.send(text);
  }

  private sendBinary(buf: Buffer): void {
    for (const ws of this.clients) if (ws.readyState === ws.OPEN) ws.send(buf, { binary: true });
  }

  attach(ws: WebSocket): void {
    this.clients.add(ws);
    ws.on('close', () => this.clients.delete(ws));
    const s = this.info.status;
    if (s === 'queued') {
      if (!MockRun.active) this.start();
      else {
        // like the server: `started` carries the current status, so a waiting run announces "queued";
        // another `started` (status running) follows when its turn comes
        ws.send(JSON.stringify({ type: 'started', message: null, run: this.info }));
        if (!MockRun.waiting.includes(this)) MockRun.waiting.push(this);
      }
    } else if (s === 'running') ws.send(JSON.stringify({ type: 'started', message: null, run: this.info }));
    else {
      ws.send(JSON.stringify({ type: s, message: null, run: this.info }));
      ws.close(1000);
    }
  }

  private start(): void {
    MockRun.active = this;
    MockRun.waiting = MockRun.waiting.filter((r) => r !== this);
    this.info.status = 'running';
    this.send({ type: 'started', message: null, run: this.info });
    const maxIter = this.project.params?.max_iter ?? 100;
    const every = Math.max(1, this.project.params?.density_every ?? 1);
    const volfrac = this.project.params?.volfrac ?? 0.3;
    let it = 0;
    this.timer = setInterval(() => {
      it++;
      const rec = {
        it,
        compliance: round4(100 * (0.35 + 0.65 * Math.exp(-it / 12))),
        volume: round4(volfrac + (1 - volfrac) * Math.exp(-it / 6)),
        change: round4(0.2 * Math.exp(-it / 10)),
        t_iter: 0.15,
        stress_max: null,
        constraint: null,
      };
      this.info.history!.push(rec);
      this.send({ type: 'progress', ...rec });
      // the server sends a frame every `every` iterations plus a final one right before `done`
      if (it % every === 0 || it === maxIter) {
        this.sendBinary(densityFrame(it, maxIter, this.stats, this.pad));
      }
      if (it >= maxIter) this.finish('done');
    }, TICK_MS);
  }

  private finish(status: 'done' | 'cancelled'): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.info.status = status;
    this.info.finished_at = new Date().toISOString();
    this.info.outcome = status === 'done' ? 'max_iter' : 'cancelled';
    this.info.message = status === 'done' ? `stopped at max_iter=${this.info.history?.length ?? 0}` : 'cancelled';
    this.send({ type: status, message: status === 'done' ? `stopped at max_iter=${this.info.history?.length ?? 0}` : 'cancelled', run: this.info });
    for (const ws of this.clients) ws.close(1000);
    this.clients.clear();
    MockRun.waiting = MockRun.waiting.filter((r) => r !== this);
    if (MockRun.active === this) {
      MockRun.active = null;
      MockRun.waiting.shift()?.start();
    }
  }

  cancel(): void {
    if (this.info.status === 'queued' || this.info.status === 'running') this.finish('cancelled');
  }
}

export function mockApi(): Plugin {
  const meshes = new Map<string, StoredMesh>();
  const projects = new Map<string, Project>();
  const runs = new Map<string, MockRun>();
  let nProject = 0;
  let nRun = 0;
  let repoRoot = resolve(process.cwd(), '..');

  const meshOf = (id: string): StoredMesh => {
    const m = meshes.get(id);
    if (!m) throw new HttpError(404, `mesh ${id} not found`);
    return m;
  };
  const projectOf = (id: string): Project => {
    const p = projects.get(id);
    if (!p) throw new HttpError(404, `project ${id} not found`);
    return p;
  };
  const runOf = (id: string): MockRun => {
    const r = runs.get(id);
    if (!r) throw new HttpError(404, `run ${id} not found`);
    return r;
  };
  const designMesh = (p: ProjectIn): StoredMesh => {
    const id = p.design_mesh?.mesh_id;
    if (!id) throw new HttpError(409, 'project has no design mesh');
    return meshOf(id);
  };
  const readJson = async <T>(c: Ctx): Promise<T> => {
    try {
      return JSON.parse((await c.body()).toString('utf8')) as T;
    } catch {
      throw new HttpError(422, 'invalid JSON body');
    }
  };

  type Route = [string, RegExp, (c: Ctx) => void | Promise<void>];
  const routes: Route[] = [
    ['GET', /^\/api\/health$/, (c) => json(c.res, { status: 'ok', version: 'mock' })],

    [
      'POST',
      /^\/api\/meshes$/,
      async (c) => {
        const ct = c.req.headers['content-type'] ?? '';
        if (!ct.startsWith('multipart/form-data')) throw new HttpError(422, 'multipart/form-data required');
        const file = parseMultipart(await c.body(), ct).find((p) => p.name === 'file');
        if (!file) throw new HttpError(422, 'form field "file" missing');
        let geom: MeshGeom;
        try {
          geom = parseSTL(file.data);
        } catch (e) {
          throw new HttpError(400, e instanceof StlError ? e.message : 'could not parse mesh');
        }
        // same as the server: id = first 16 hex chars of the sha256 of the bytes, so a re-upload returns the same id
        const id = createHash('sha256').update(file.data).digest('hex').slice(0, 16);
        const watertight = isWatertight(geom);
        const info: S['MeshInfo'] = {
          id,
          name: file.filename ?? 'mesh.stl',
          n_faces: geom.tris.length / 3,
          n_vertices: geom.positions.length / 3,
          bbox: bboxOf(geom),
          is_watertight: watertight,
          source: 'mesh',
          volume: watertight ? volumeOf(geom) : null,
        };
        meshes.set(id, { info, geom, adjacency: adjacencyPairs(geom) });
        json(c.res, info);
      },
    ],
    ['GET', /^\/api\/meshes\/([^/]+)$/, (c) => json(c.res, meshOf(c.params[0]!).info)],
    [
      'GET',
      /^\/api\/meshes\/([^/]+)\/buffer$/,
      (c) => bin(c.res, meshBuffer(meshOf(c.params[0]!).geom), 'application/octet-stream'),
    ],
    [
      'GET',
      /^\/api\/meshes\/([^/]+)\/adjacency$/,
      (c) => {
        const a = meshOf(c.params[0]!).adjacency;
        bin(c.res, Buffer.from(a.buffer, a.byteOffset, a.byteLength), 'application/octet-stream');
      },
    ],
    [
      'GET',
      /^\/api\/meshes\/([^/]+)\/facets$/,
      (c) => {
        const m = meshOf(c.params[0]!);
        const angle = Number(c.url.searchParams.get('angle_deg') ?? 5);
        const all = computeFacets(m.geom, m.adjacency, angle);
        json(c.res, { mesh_id: m.info.id, angle_deg: angle, facets: all.slice(0, 300), n_facets_total: all.length });
      },
    ],
    [
      'GET',
      /^\/api\/meshes\/([^/]+)\/preview\.png$/,
      (c) => {
        meshOf(c.params[0]!);
        bin(c.res, PNG_1X1, 'image/png');
      },
    ],

    ['GET', /^\/api\/projects$/, (c) => json(c.res, [...projects.values()])],
    [
      'POST',
      /^\/api\/projects$/,
      async (c) => {
        const body = await readJson<ProjectIn>(c);
        const now = new Date().toISOString();
        const p: Project = { ...body, id: `proj-${++nProject}`, created_at: now, updated_at: now };
        projects.set(p.id, p);
        json(c.res, p);
      },
    ],
    ['GET', /^\/api\/projects\/([^/]+)$/, (c) => json(c.res, projectOf(c.params[0]!))],
    [
      'PUT',
      /^\/api\/projects\/([^/]+)$/,
      async (c) => {
        const old = projectOf(c.params[0]!);
        const body = await readJson<Project>(c);
        const p: Project = { ...body, id: old.id, created_at: old.created_at, updated_at: new Date().toISOString() };
        projects.set(p.id, p);
        json(c.res, p);
      },
    ],
    [
      'POST',
      /^\/api\/projects\/([^/]+)\/voxelize$/,
      (c) => {
        const p = projectOf(c.params[0]!);
        json(c.res, voxelStats(p, designMesh(p)));
      },
    ],
    [
      'POST',
      /^\/api\/projects\/([^/]+)\/resolve-selection$/,
      async (c) => {
        const p = projectOf(c.params[0]!);
        const sel = await readJson<Selection>(c);
        const mesh = 'mesh_id' in sel && meshes.has(sel.mesh_id) ? meshOf(sel.mesh_id) : designMesh(p);
        json(c.res, resolveFake(sel, mesh, p));
      },
    ],

    [
      'POST',
      /^\/api\/runs$/,
      async (c) => {
        const { project_id } = await readJson<S['RunCreate']>(c);
        const p = projectOf(project_id);
        designMesh(p);
        // same wording as Problem.validate() on the server
        const issues = [...(p.loads?.length ? [] : ['no loads defined']), ...(p.supports?.length ? [] : ['no supports defined'])];
        if (issues.length) throw new HttpError(422, `project is not runnable: ${issues.join('; ')}`);
        const run = new MockRun(`run-${++nRun}`, p.id, structuredClone(p), designMesh(p));
        runs.set(run.info.id, run);
        json(c.res, run.info);
      },
    ],
    ['GET', /^\/api\/runs$/, (c) => json(c.res, [...runs.values()].map((r) => r.info))],
    ['GET', /^\/api\/runs\/([^/]+)$/, (c) => json(c.res, runOf(c.params[0]!).info)],
    [
      'POST',
      /^\/api\/runs\/([^/]+)\/cancel$/,
      (c) => {
        const r = runOf(c.params[0]!);
        r.cancel();
        json(c.res, r.info);
      },
    ],
    [
      'GET',
      /^\/api\/runs\/([^/]+)\/result\.stl$/,
      (c) => {
        const r = runOf(c.params[0]!);
        finished(r);
        const [lo, hi] = worldBBox(designMesh(r.project), r.project.design_mesh?.transform);
        const shrink = lo.map((v, k) => (hi[k]! - v) * 0.2);
        bin(
          c.res,
          boxSTL(lo.map((v, k) => v + shrink[k]!), hi.map((v, k) => v - shrink[k]!)),
          'model/stl',
          { 'Content-Disposition': `attachment; filename="${r.info.id}.stl"` },
        );
      },
    ],
    [
      'GET',
      /^\/api\/runs\/([^/]+)\/result\.vti$/,
      (c) => {
        const r = runOf(c.params[0]!);
        finished(r);
        const s = r.info.stats!;
        const xml =
          `<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">\n` +
          `<ImageData WholeExtent="0 ${s.nx} 0 ${s.ny} 0 ${s.nz}" Origin="${s.origin.join(' ')}" Spacing="${s.h} ${s.h} ${s.h}">\n` +
          `<Piece Extent="0 ${s.nx} 0 ${s.ny} 0 ${s.nz}"><CellData/></Piece>\n</ImageData>\n</VTKFile>\n`;
        bin(c.res, Buffer.from(xml), 'application/xml', {
          'Content-Disposition': `attachment; filename="${r.info.id}.vti"`,
        });
      },
    ],
    [
      'GET',
      /^\/api\/runs\/([^/]+)\/result\.npz$/,
      (c) => {
        const r = runOf(c.params[0]!);
        finished(r);
        const emptyZip = Buffer.concat([Buffer.from([0x50, 0x4b, 0x05, 0x06]), Buffer.alloc(18)]);
        bin(c.res, emptyZip, 'application/zip', { 'Content-Disposition': `attachment; filename="${r.info.id}.npz"` });
      },
    ],
    [
      'GET',
      /^\/api\/runs\/([^/]+)\/project\.json$/,
      (c) => {
        const r = runOf(c.params[0]!);
        json(c.res, { project: r.project as Project, run: r.info }); // the project as it was when the run started
      },
    ],
    [
      'GET',
      /^\/api\/runs\/([^/]+)\/preview\.png$/,
      (c) => {
        runOf(c.params[0]!);
        bin(c.res, PNG_1X1, 'image/png');
      },
    ],

    // Test convenience: lets in-browser code fetch the example meshes from the repo root.
    [
      'GET',
      /^\/mock\/examples\/([\w.-]+\.stl)$/,
      (c) => {
        const file = resolve(repoRoot, 'examples', c.params[0]!);
        if (!existsSync(file)) throw new HttpError(404, 'example not found');
        bin(c.res, readFileSync(file), 'model/stl');
      },
    ],
  ];

  return {
    name: 'topop-mock-api',
    apply: 'serve',
    configureServer(server) {
      repoRoot = resolve(server.config.root, '..');

      server.middlewares.use((req, res, next) => {
        const url = new URL(req.url ?? '/', 'http://mock');
        if (!url.pathname.startsWith('/api/') && !url.pathname.startsWith('/mock/')) return next();
        const route = routes.find(([method, re]) => method === req.method && re.test(url.pathname));
        if (!route) return json(res, { detail: `no mock route for ${req.method} ${url.pathname}` }, 404);
        const params = route[1].exec(url.pathname)!.slice(1);
        const body = () =>
          new Promise<Buffer>((ok, fail) => {
            const chunks: Buffer[] = [];
            req.on('data', (d: Buffer) => chunks.push(d));
            req.on('end', () => ok(Buffer.concat(chunks)));
            req.on('error', fail);
          });
        Promise.resolve()
          .then(() => route[2]({ req, res, url, params, body }))
          .catch((e: unknown) => {
            if (res.headersSent) return res.end();
            if (e instanceof HttpError) json(res, { detail: e.message }, e.status);
            else json(res, { detail: e instanceof Error ? e.message : String(e) }, 500);
          });
      });

      const wss = new WebSocketServer({ noServer: true });
      server.httpServer?.on('upgrade', (req, socket, head) => {
        const m = /^\/api\/runs\/([^/?]+)\/stream(?:\?|$)/.exec(req.url ?? '');
        if (!m) return; // leave Vite's HMR socket alone
        const run = runs.get(m[1]!);
        if (!run) {
          socket.write('HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n');
          socket.destroy();
          return;
        }
        wss.handleUpgrade(req, socket, head, (ws) => run.attach(ws));
      });
      server.httpServer?.on('close', () => wss.close());
    },
  };
}
