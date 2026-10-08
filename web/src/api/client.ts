import type { components } from './types.gen';

type Schemas = components['schemas'];
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

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === 'string') detail = body.detail;
      else if (body.detail !== undefined) detail = JSON.stringify(body.detail);
    } catch {
      // non-JSON error body: keep statusText
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export async function apiGet<T>(path: string): Promise<T> {
  return parse<T>(await fetch(path));
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return parse<T>(
    await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  );
}

/** multipart/form-data upload, field name `file` (POST /api/meshes). */
export async function apiUpload<T>(path: string, file: File): Promise<T> {
  const form = new FormData();
  form.append('file', file, file.name);
  return parse<T>(await fetch(path, { method: 'POST', body: form }));
}

/** Binary WS frame: [u32 it][u32 nx][u32 ny][u32 nz][u8 rho*255 ...], C-order, little-endian. */
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
  return {
    it: dv.getUint32(0, true),
    shape: [nx, ny, nz],
    rho: new Uint8Array(buf, 16, nx * ny * nz),
  };
}

/** Opens /api/runs/{id}/stream; returns a function that closes it. */
export function openRunStream(runId: string, handlers: RunStreamHandlers): () => void {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${scheme}://${location.host}/api/runs/${runId}/stream`);
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
