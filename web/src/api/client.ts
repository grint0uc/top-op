// All server calls live here (REST + run WebSocket) so swapping the mock for the real API is mechanical.
import type { components } from './types.gen';

type Schemas = components['schemas'];
export type MeshInfo = Schemas['MeshInfo'];
export type MeshFacets = Schemas['MeshFacets'];
export type ProjectIn = Schemas['ProjectIn'];
export type Project = Schemas['Project'];
export type MeshRef = Schemas['MeshRef'];
export type RefModel = Schemas['RefModel'];
export type GridSpec = Schemas['GridSpec'];
export type MaterialSpec = Schemas['MaterialSpec'];
export type ParamsSpec = Schemas['ParamsSpec'];
export type LoadSpec = Schemas['LoadSpec'];
export type SupportSpec = Schemas['SupportSpec'];
export type Selection = Schemas['Selection'];
export type FaceSelection = Schemas['FaceSelection'];
export type FacetSelection = Schemas['FacetSelection'];
export type NormalSelection = Schemas['NormalSelection'];
export type PlaneSelection = Schemas['PlaneSelection'];
export type PrimitiveSelection = Schemas['PrimitiveSelection'];
/** The selection kinds an agent writes from the facet table / bbox alone (no viewport editor of their own). */
export type QuerySelection = FacetSelection | NormalSelection | PlaneSelection;
export type FacetInfo = Schemas['FacetInfo'];
export type RunExport = Schemas['RunExport'];
export type VoxelStats = Schemas['VoxelStats'];
export type ResolvedNodes = Schemas['ResolvedNodes'];
export type RunInfo = Schemas['RunInfo'];
export type RunStatus = RunInfo['status'];
export type IterationRecord = Schemas['IterationRecord'];
export type ProgressMsg = Schemas['ProgressMsg'];
export type StatusMsg = Schemas['StatusMsg'];

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** FastAPI errors: `{detail: "text"}` (our HTTPException) or `{detail: [{loc, msg}, ...]}` (request validation). */
export function errorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((d: unknown) => {
      const e = d as { loc?: unknown[]; msg?: string };
      const where = Array.isArray(e.loc) ? e.loc.filter((x) => x !== 'body').join('.') : '';
      return e.msg ? (where ? `${where}: ${e.msg}` : e.msg) : JSON.stringify(d);
    });
    if (parts.length > 0) return parts.join('; ');
  }
  return detail === undefined ? fallback : JSON.stringify(detail);
}

async function fail(res: Response): Promise<never> {
  let detail = res.statusText || `HTTP ${res.status}`;
  try {
    detail = errorDetail(((await res.json()) as { detail?: unknown }).detail, detail);
  } catch {
    // non-JSON error body: keep statusText
  }
  throw new ApiError(res.status, detail);
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) return fail(res);
  return (await res.json()) as T;
}

const JSON_HEADERS = { 'Content-Type': 'application/json' };

export async function apiGet<T>(path: string): Promise<T> {
  return parse<T>(await fetch(path));
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return parse<T>(
    await fetch(path, {
      method: 'POST',
      headers: JSON_HEADERS,
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  );
}

export async function apiPut<T>(path: string, body: unknown): Promise<T> {
  return parse<T>(await fetch(path, { method: 'PUT', headers: JSON_HEADERS, body: JSON.stringify(body) }));
}

/** multipart/form-data upload, field name `file` (POST /api/meshes). */
export async function apiUpload<T>(path: string, file: File): Promise<T> {
  const form = new FormData();
  form.append('file', file, file.name);
  return parse<T>(await fetch(path, { method: 'POST', body: form }));
}

export async function apiGetBuffer(path: string): Promise<ArrayBuffer> {
  const res = await fetch(path);
  if (!res.ok) return fail(res);
  return res.arrayBuffer();
}

export interface ResultOptions {
  threshold: number;
  smooth: number;
  /** intersect the result with the original CAD (server: `trim=true|false`; omitted -> server default) */
  trim?: boolean;
}

/** GET /api/meshes/{id}/facets/{facet_id}/faces */
export interface FacetFaces {
  face_ids: number[];
}

/** The result mesh bytes plus the server's `X-Topop-Warnings` header (e.g. why a CAD trim fell back). */
export interface ResultMesh {
  buffer: ArrayBuffer;
  warnings: string | null;
}

/** An export fetched into memory: the bytes plus the server's `X-Topop-Warnings` header (null when absent). */
export interface Download {
  blob: Blob;
  warnings: string | null;
}

/** GET /api/runs/{id}/stress: von Mises per element, [ix][iy][iz] (iz fastest), 0 on inactive cells. */
export interface StressField {
  shape: [number, number, number];
  data: Float32Array;
  max: number;
}

const enc = encodeURIComponent;
const resultQuery = (o: ResultOptions) =>
  `threshold=${o.threshold}&smooth=${o.smooth}${o.trim === undefined ? '' : `&trim=${o.trim}`}`;

/** Binary layout: u32 nx, u32 ny, u32 nz, f32[nx*ny*nz], little-endian. */
export function parseStress(buf: ArrayBuffer): StressField {
  if (buf.byteLength < 12) throw new Error('truncated stress field');
  const dv = new DataView(buf);
  const nx = dv.getUint32(0, true);
  const ny = dv.getUint32(4, true);
  const nz = dv.getUint32(8, true);
  const n = nx * ny * nz;
  if (buf.byteLength < 12 + 4 * n) throw new Error('truncated stress field');
  const data = new Float32Array(buf, 12, n); // offset 12 is 4-byte aligned
  let max = 0;
  for (let i = 0; i < n; i++) if (data[i]! > max) max = data[i]!;
  return { shape: [nx, ny, nz], data, max };
}

/** Fetches an export so the status and the warnings header are visible (a plain `<a download>` shows neither). */
async function fetchDownload(url: string): Promise<Download> {
  const res = await fetch(url);
  if (!res.ok) return fail(res);
  return { blob: await res.blob(), warnings: res.headers.get('X-Topop-Warnings') };
}

export const api = {
  // meshes
  /** Mesh ids are content hashes: uploading the same bytes again returns the same id. */
  uploadMesh: (file: File) => apiUpload<MeshInfo>('/api/meshes', file),
  /** 404 (ApiError.status) when the server does not know the id; the store persists across restarts. */
  getMesh: (id: string) => apiGet<MeshInfo>(`/api/meshes/${enc(id)}`),
  meshBuffer: (id: string) => apiGetBuffer(`/api/meshes/${enc(id)}/buffer`),
  /** Flat u32 array of face pairs [a0, b0, a1, b1, ...]. */
  meshAdjacency: async (id: string) => new Uint32Array(await apiGetBuffer(`/api/meshes/${enc(id)}/adjacency`)),
  meshFacets: (id: string, angleDeg = 5) => apiGet<MeshFacets>(`/api/meshes/${enc(id)}/facets?angle_deg=${angleDeg}`),
  /** Exact triangle ids of one facet (also for cylinders and STEP B-rep faces). */
  facetFaces: (id: string, facetId: number, angleDeg = 5) =>
    apiGet<FacetFaces>(`/api/meshes/${enc(id)}/facets/${facetId}/faces?angle_deg=${angleDeg}`),
  meshPreviewUrl: (id: string, view = 'iso') => `/api/meshes/${enc(id)}/preview.png?view=${enc(view)}`,

  // projects
  createProject: (p: ProjectIn) => apiPost<Project>('/api/projects', p),
  getProject: (id: string) => apiGet<Project>(`/api/projects/${enc(id)}`),
  listProjects: () => apiGet<Project[]>('/api/projects'),
  updateProject: (id: string, p: Project) => apiPut<Project>(`/api/projects/${enc(id)}`, p),
  voxelize: (projectId: string) => apiPost<VoxelStats>(`/api/projects/${enc(projectId)}/voxelize`),
  resolveSelection: (projectId: string, sel: Selection) =>
    apiPost<ResolvedNodes>(`/api/projects/${enc(projectId)}/resolve-selection`, sel),

  // runs
  createRun: (projectId: string) => apiPost<RunInfo>('/api/runs', { project_id: projectId }),
  getRun: (id: string) => apiGet<RunInfo>(`/api/runs/${enc(id)}`),
  listRuns: () => apiGet<RunInfo[]>('/api/runs'),
  cancelRun: (id: string) => apiPost<RunInfo>(`/api/runs/${enc(id)}/cancel`),
  resultStl: async (id: string, o: ResultOptions): Promise<ResultMesh> => {
    const res = await fetch(api.resultStlUrl(id, o));
    if (!res.ok) return fail(res);
    return { buffer: await res.arrayBuffer(), warnings: res.headers.get('X-Topop-Warnings') };
  },
  /** 409 (ApiError.status) while the run has no stress field yet. */
  runStress: async (id: string) => parseStress(await apiGetBuffer(`/api/runs/${enc(id)}/stress`)),
  /** Any of the export URLs below (`resultStlUrl`, `resultVtiUrl`, ...): ApiError on 409 (no result yet) and the like. */
  download: fetchDownload,
  resultStlUrl: (id: string, o: ResultOptions) => `/api/runs/${enc(id)}/result.stl?${resultQuery(o)}`,
  resultVtiUrl: (id: string) => `/api/runs/${enc(id)}/result.vti`,
  resultNpzUrl: (id: string) => `/api/runs/${enc(id)}/result.npz`,
  projectJsonUrl: (id: string) => `/api/runs/${enc(id)}/project.json`,
  runExport: (id: string) => apiGet<RunExport>(`/api/runs/${enc(id)}/project.json`),
  runPreviewUrl: (id: string, o: { threshold: number; view?: string; trim?: boolean }) =>
    `/api/runs/${enc(id)}/preview.png?threshold=${o.threshold}&view=${enc(o.view ?? 'iso')}${o.trim === undefined ? '' : `&trim=${o.trim}`}`,
};

/** Binary WS frame: [u32 it][u32 nx][u32 ny][u32 nz][u8 rho*255 ...], C-order [ix][iy][iz], little-endian. */
export interface DensityFrame {
  it: number;
  shape: [number, number, number];
  rho: Uint8Array;
}

export interface RunStreamHandlers {
  onProgress?: (msg: ProgressMsg) => void;
  onStatus?: (msg: StatusMsg) => void;
  onDensity?: (frame: DensityFrame) => void;
  onClose?: (ev: CloseEvent) => void;
}

export function parseDensityFrame(buf: ArrayBuffer): DensityFrame {
  const dv = new DataView(buf);
  const nx = dv.getUint32(4, true);
  const ny = dv.getUint32(8, true);
  const nz = dv.getUint32(12, true);
  if (buf.byteLength < 16 + nx * ny * nz) throw new Error('truncated density frame');
  return {
    it: dv.getUint32(0, true),
    shape: [nx, ny, nz],
    rho: new Uint8Array(buf, 16, nx * ny * nz),
  };
}

/** Opens /api/runs/{id}/stream; returns a function that closes it. */
export function openRunStream(runId: string, handlers: RunStreamHandlers): () => void {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${scheme}://${location.host}/api/runs/${enc(runId)}/stream`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (ev: MessageEvent<string | ArrayBuffer>) => {
    if (typeof ev.data !== 'string') {
      handlers.onDensity?.(parseDensityFrame(ev.data));
      return;
    }
    const msg = JSON.parse(ev.data) as Schemas['WsMessage'];
    if (msg.type === 'progress') handlers.onProgress?.(msg);
    else handlers.onStatus?.(msg);
  };
  ws.onclose = (ev) => handlers.onClose?.(ev);
  return () => ws.close();
}
