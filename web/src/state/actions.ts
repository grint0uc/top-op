// Orchestration: store + api/client. Panels and hotkeys call these; the viewport follows the store (viewport/bind.ts).
import {
  api,
  type LoadSpec,
  type ProjectIn,
  type RefModel,
  type Selection,
  type SupportSpec,
  openRunStream,
} from '../api/client';
import { buildAdjacency, parseMeshBuffer } from '../viewport/meshData';
import { bboxDiagonal, designEntry, isPrimitive, toPrim } from './derived';
import { IDENTITY } from './defaults';
import { type Prim, type PrimitiveKind, currentSelectionSpec, genId, useStore } from './store';

const get = useStore.getState;

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Runs an async user action with a busy label and error notice. */
async function guard<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
  useStore.getState().setBusy(label);
  try {
    return await fn();
  } catch (e) {
    useStore.getState().setNotice({ kind: 'error', text: `${label} failed: ${message(e)}` });
    return undefined;
  } finally {
    useStore.getState().setBusy(null);
  }
}

// ------------------------------------------------------------------ meshes

async function uploadAndLoad(file: File, role: 'design' | 'ref') {
  const info = await api.uploadMesh(file);
  const buffer = await api.meshBuffer(info.id);
  const data = parseMeshBuffer(buffer);
  get().addMesh({ info, role, buffer, data });
  return info;
}

export async function importDesignMesh(file: File): Promise<void> {
  await guard('Upload', async () => {
    const before = get();
    const old = before.project.design_mesh?.mesh_id ?? null;
    const info = await uploadAndLoad(file, 'design');
    const s = get();
    if (old && old !== info.id) {
      const wasLoaded = !!before.meshes[old];
      const memo = before.meshMemo[old];
      if (!wasLoaded && memo && memo.n_faces === info.n_faces) {
        s.remapMesh(old, info.id); // restored project + same mesh re-uploaded: keep loads/supports
      } else {
        // a different mesh: face ids no longer mean anything
        const stale = (x: { selection: Selection }) => x.selection.kind === 'faces' && x.selection.mesh_id === old;
        const dropped = s.project.loads.filter(stale).length + s.project.supports.filter(stale).length;
        s.project.loads.filter(stale).forEach((l) => s.removeLoad(l.id));
        s.project.supports.filter(stale).forEach((x) => s.removeSupport(x.id));
        if (dropped > 0) {
          s.setNotice({ kind: 'warn', text: `Removed ${dropped} load/support(s) that were defined on the previous mesh.` });
        }
      }
    }
    get().setDesignMesh(info.id);
    get().setPreview(null);
    const entry = designEntry(get());
    if (entry) get().setBrushRadius(Math.round((bboxDiagonal(entry.data) / 25) * 10) / 10 || 1);
    if (get().restored) useStore.setState({ restored: false, notice: null });
  });
}

export async function importRefMesh(file: File): Promise<void> {
  await guard('Upload', async () => {
    const info = await uploadAndLoad(file, 'ref');
    const ref: RefModel = {
      id: genId('ref'),
      name: info.name,
      mesh_id: info.id,
      transform: [...IDENTITY],
      mode: 'keep_out',
      visible: true,
    };
    get().addRef(ref);
    get().setActiveItem({ kind: 'ref', id: ref.id });
    get().setTool('gizmo');
  });
}

/** After a reload the project knows its ref models but not their geometry: attach a fresh upload. */
export async function reuploadRefMesh(refId: string, file: File): Promise<void> {
  await guard('Upload', async () => {
    const info = await uploadAndLoad(file, 'ref');
    get().updateRef(refId, { mesh_id: info.id, name: info.name });
  });
}

export async function ensureAdjacency() {
  const entry = designEntry(get());
  if (!entry) return null;
  if (!entry.data.adjacency) {
    const pairs = await api.meshAdjacency(entry.info.id);
    entry.data.adjacency = buildAdjacency(pairs, entry.data.nTri);
  }
  return entry.data.adjacency;
}

// ------------------------------------------------------------------ selection -> loads / supports / primitives

export function addPrimitive(kind: PrimitiveKind): void {
  const entry = designEntry(get());
  const { min, max } = entry?.data.bbox ?? { min: [-10, -10, -10], max: [10, 10, 10] };
  const centre = [0, 1, 2].map((k) => (min[k]! + max[k]!) / 2);
  const span = Math.max(max[0]! - min[0]!, max[1]! - min[1]!, max[2]! - min[2]!) || 20;
  const edge = span * 0.25;
  const size = kind === 'box' ? [edge, edge, edge] : kind === 'sphere' ? [edge / 2, edge / 2, edge / 2] : [edge / 2, edge, edge / 2];
  const prim: Prim = {
    kind,
    transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, centre[0]!, centre[1]!, centre[2]!, 1],
    size,
    surface_only: true,
  };
  get().setPrimitive(prim);
  get().setActiveItem(null);
  get().setTool('gizmo');
}

function nextName(prefix: string, existing: readonly { name: string }[]): string {
  return `${prefix} ${existing.length + 1}`;
}

export function addLoadFromSelection(): LoadSpec | null {
  const s = get();
  const sel = currentSelectionSpec(s);
  if (!sel) {
    s.setNotice({ kind: 'warn', text: 'Select faces (pick/paint) or add a primitive first.' });
    return null;
  }
  const load: LoadSpec = { id: genId('load'), name: nextName('Load', s.project.loads), selection: sel, force: [0, 0, -1], case: 0 };
  s.addLoad(load);
  s.setActiveItem({ kind: 'load', id: load.id });
  if (!isPrimitive(sel)) s.clearSelection();
  else s.setPrimitive(null);
  return load;
}

export function addSupportFromSelection(): SupportSpec | null {
  const s = get();
  const sel = currentSelectionSpec(s);
  if (!sel) {
    s.setNotice({ kind: 'warn', text: 'Select faces (pick/paint) or add a primitive first.' });
    return null;
  }
  const support: SupportSpec = {
    id: genId('sup'),
    name: nextName('Support', s.project.supports),
    selection: sel,
    fix: [true, true, true],
  };
  s.addSupport(support);
  s.setActiveItem({ kind: 'support', id: support.id });
  if (!isPrimitive(sel)) s.clearSelection();
  else s.setPrimitive(null);
  return support;
}

/** Make an existing load/support's selection the current selection again (to inspect or re-use it). */
export function reselect(selection: Selection): void {
  const s = get();
  if (selection.kind === 'faces') s.setFaceSelection([...selection.face_ids]);
  else if (isPrimitive(selection)) {
    s.setPrimitive(toPrim(selection));
    s.setTool('gizmo');
  } else s.setNotice({ kind: 'info', text: `A ${selection.kind} selection has no viewport editor.` });
}

export function deleteActive(): void {
  const s = get();
  const a = s.activeItem;
  if (!a) return;
  if (a.kind === 'load') s.removeLoad(a.id);
  else if (a.kind === 'support') s.removeSupport(a.id);
  else s.removeRef(a.id);
}

// ------------------------------------------------------------------ server project sync

let syncChain: Promise<unknown> = Promise.resolve();
let lastSynced: { id: string; body: string } | null = null;

/** Pushes the document to the server (create or PUT) and returns its id. Serialised, skips unchanged documents. */
export function syncProject(): Promise<string> {
  const job = async (): Promise<string> => {
    const s = get();
    if (!s.project.design_mesh?.mesh_id) throw new Error('import a design mesh first');
    const payload: ProjectIn = {
      ...s.project,
      // ref models whose geometry was never re-uploaded would 404 on the server
      ref_models: s.project.ref_models.filter((r) => r.mesh_id && s.meshes[r.mesh_id]),
    };
    const body = JSON.stringify(payload);
    const meta = s.projectMeta;
    if (meta && lastSynced?.id === meta.id && lastSynced.body === body) return meta.id;
    const saved = meta ? await api.updateProject(meta.id, { ...payload, ...meta }) : await api.createProject(payload);
    get().setProjectMeta({ id: saved.id, created_at: saved.created_at, updated_at: saved.updated_at });
    lastSynced = { id: saved.id, body };
    return saved.id;
  };
  const next = syncChain.then(job, job);
  syncChain = next.catch(() => undefined);
  return next;
}

let voxelSeq = 0;
export async function runVoxelize(): Promise<void> {
  const seq = ++voxelSeq;
  const st = get();
  if (!designEntry(st)) return;
  st.setVoxel({ loading: true, error: null });
  try {
    const id = await syncProject();
    const stats = await api.voxelize(id);
    if (seq === voxelSeq) get().setVoxel({ stats, loading: false });
  } catch (e) {
    if (seq === voxelSeq) get().setVoxel({ loading: false, error: message(e) });
  }
}

export async function resolvePreview(target?: Selection): Promise<void> {
  const s = get();
  let sel = target ?? currentSelectionSpec(s);
  let source = 'selection';
  if (!sel && s.activeItem) {
    const it =
      s.activeItem.kind === 'load'
        ? s.project.loads.find((l) => l.id === s.activeItem?.id)
        : s.project.supports.find((x) => x.id === s.activeItem?.id);
    if (it) {
      sel = it.selection;
      source = it.name || it.id;
    }
  }
  if (!sel) {
    s.setNotice({ kind: 'warn', text: 'Nothing to preview: select faces or pick a load/support first.' });
    return;
  }
  await guard('Resolve preview', async () => {
    const id = await syncProject();
    const res = await api.resolveSelection(id, sel);
    const xyz = new Float32Array(res.xyz.length * 3);
    res.xyz.forEach((p, i) => xyz.set([p[0] ?? 0, p[1] ?? 0, p[2] ?? 0], i * 3));
    get().setPreview({ count: res.count, truncated: res.truncated, xyz, source });
  });
}

// ------------------------------------------------------------------ runs

let closeStream: (() => void) | null = null;

export async function startRun(): Promise<void> {
  const st = get();
  if (st.run.status === 'queued' || st.run.status === 'running') return;
  const entry = designEntry(st);
  if (!entry) {
    st.setNotice({ kind: 'warn', text: 'Import a design mesh before running.' });
    return;
  }
  closeStream?.();
  st.resetRun();
  st.patchRun({ status: 'queued', maxIter: st.project.params.max_iter });
  await guard('Start run', async () => {
    try {
      const projectId = await syncProject();
      const info = await api.createRun(projectId);
      get().patchRun({ id: info.id, stats: info.stats ?? null, status: info.status });
      closeStream = openRunStream(info.id, {
        onStatus: (msg) => {
          const run = get().run;
          const stats = msg.run?.stats ?? run.stats;
          if (msg.type === 'started') get().patchRun({ status: 'running', stats });
          else if (msg.type === 'done') {
            const hist = msg.run?.history ?? [];
            get().patchRun({ status: 'done', stats, history: hist.length > run.history.length ? hist : run.history });
          } else if (msg.type === 'error') {
            get().patchRun({ status: 'error', error: msg.message ?? msg.run?.error ?? 'run failed' });
          } else get().patchRun({ status: 'cancelled' });
          if (msg.type !== 'started') closeStream?.();
        },
        onProgress: (rec) => get().pushProgress(rec),
        onDensity: (f) => get().pushDensity(f),
        // a dropped socket is not a failed run: ask the server what happened before giving up
        onClose: () => {
          const r = get().run;
          if ((r.status !== 'queued' && r.status !== 'running') || !r.id) return;
          api
            .getRun(r.id)
            .then((info) => {
              if (info.status === 'queued' || info.status === 'running') throw new Error('connection to the run stream was lost');
              const hist = info.history ?? [];
              get().patchRun({
                status: info.status,
                error: info.error,
                history: hist.length > get().run.history.length ? hist : get().run.history,
              });
            })
            .catch((e: unknown) => get().patchRun({ status: 'error', error: message(e) }));
        },
      });
    } catch (e) {
      get().patchRun({ status: 'error', error: message(e) });
      throw e;
    }
  });
}

export async function stopRun(): Promise<void> {
  const id = get().run.id;
  if (!id) return;
  await guard('Stop run', async () => {
    const info = await api.cancelRun(id);
    if (info.status === 'cancelled') get().patchRun({ status: 'cancelled' });
  });
}

export async function loadResultMesh(): Promise<void> {
  const s = get();
  if (!s.run.id) return;
  const id = s.run.id;
  await guard('Load result', async () => {
    get().setResultStl(await api.resultStl(id, { threshold: s.threshold, smooth: s.smoothIterations }));
  });
}

export function hideResultMesh(): void {
  get().setResultStl(null);
}
