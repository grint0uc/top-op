// Orchestration: store + api/client. Panels and hotkeys call these; the viewport follows the store (viewport/bind.ts).
import {
  ApiError,
  api,
  type FacetInfo,
  type LoadSpec,
  type MeshFacets,
  type MeshInfo,
  type ProjectIn,
  type RefModel,
  type RunInfo,
  type Selection,
  type SupportSpec,
  openRunStream,
} from '../api/client';
import { buildAdjacency, parseMeshBuffer } from '../viewport/meshData';
import { designBox, designEntry, facetKey, isPrimitive, requiredMeshes, toPrim } from './derived';
import { IDENTITY } from './defaults';
import { parseProjectFile } from './projectFile';
import { primitiveStl, type RefPrimitiveKind } from './primitiveMesh';
import { type QueryKind, facetFaces, facetFacesKey, formFromSelection, paddedBox } from './query';
import { type Prim, type PrimitiveKind, currentSelectionSpec, genId, useStore } from './store';
import { composeTRS } from './transform';

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

/** Fetches the render buffer of a mesh the server already knows and registers it in the store. */
async function loadMeshInto(info: MeshInfo): Promise<void> {
  const buffer = await api.meshBuffer(info.id);
  get().addMesh({ info, buffer, data: parseMeshBuffer(buffer) });
}

/** Mesh ids are content hashes (sha256 prefix): the same bytes always come back with the same id. */
async function uploadAndLoad(file: File): Promise<MeshInfo> {
  const info = await api.uploadMesh(file);
  await loadMeshInto(info);
  return info;
}

export const isStepFile = (name: string): boolean => /\.(step|stp)$/i.test(name);

/**
 * uploadAndLoad, except that a STEP file the server cannot read (400, e.g. the OpenCascade extra is not installed) puts
 * the server's message in the notice verbatim: it carries the install hint. null = nothing was imported.
 */
async function uploadOrExplain(file: File): Promise<MeshInfo | null> {
  try {
    return await uploadAndLoad(file);
  } catch (e) {
    if (e instanceof ApiError && e.status === 400 && isStepFile(file.name)) {
      get().setNotice({ kind: 'error', text: e.message });
      return null;
    }
    throw e;
  }
}

/** After a design mesh is in the store: brush size, Query form defaults, and the "restored" notice. */
function designReady(): void {
  const s = get();
  const entry = designEntry(s);
  if (!entry) return;
  const box = designBox(s) ?? entry.data.bbox;
  s.setBrushRadius(Math.round((Math.hypot(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2]) / 25) * 10) / 10 || 1);
  if (!s.selection.query) {
    const { min, max } = box;
    s.setQueryForm({ ...s.queryForm, plane: { ...s.queryForm.plane, point: [0, 1, 2].map((k) => (min[k]! + max[k]!) / 2) as [number, number, number] } });
  }
  if (s.restored && requiredMeshes(get()).every((m) => m.loaded)) useStore.setState({ restored: false, notice: null });
  void prefetchFacets();
}

export async function importDesignMesh(file: File): Promise<void> {
  await guard('Upload', async () => {
    const before = get();
    const old = before.project.design_mesh?.mesh_id ?? null;
    const info = await uploadOrExplain(file);
    if (!info) return;
    let warn: string | null = null;
    if (old === info.id) {
      // same file as the project expects (restored project, or simply re-uploaded): nothing to remap
      get().setPreview(null);
    } else {
      const dropped = old ? get().adoptDesignMesh(old, info.id) : 0;
      const expected = old && !before.meshes[old]; // the project was restored and wanted a specific file
      get().setDesignMesh(info.id);
      if (expected || dropped > 0) {
        warn =
          `${expected ? 'This is not the file the project was made with. ' : ''}` +
          (dropped > 0 ? `Removed ${dropped} load/support(s) that were defined by faces/facets of the previous mesh. ` : '') +
          'Normal, plane and primitive selections were kept.';
      }
    }
    designReady();
    if (warn) get().setNotice({ kind: 'warn', text: warn });
  });
}

export async function importRefMesh(file: File): Promise<void> {
  await guard('Upload', async () => {
    const info = await uploadOrExplain(file);
    if (info) addRefFor(info, info.name);
  });
}

function addRefFor(info: MeshInfo, name: string, transform: number[] = [...IDENTITY], mode: RefModel['mode'] = 'keep_out'): void {
  const ref: RefModel = { id: genId('ref'), name, mesh_id: info.id, transform, mode, visible: true };
  get().addRef(ref);
  get().setActiveItem({ kind: 'ref', id: ref.id });
  get().setTool('gizmo');
}

/**
 * A keep-out / keep-in volume without a file: a generated unit primitive uploaded as a reference model. Its transform
 * is the full matrix (translation * rotation * scale) applied to the unit mesh, the same convention the gizmo writes.
 */
export async function addRefPrimitive(kind: RefPrimitiveKind, mode: RefModel['mode'] = 'keep_out'): Promise<void> {
  await guard('Upload', async () => {
    const label = `${mode === 'keep_out' ? 'Keep-out' : 'Keep-in'} ${kind}`;
    const file = new File([primitiveStl(kind)], `${label.toLowerCase().replace(/\s+/g, '-')}.stl`, { type: 'model/stl' });
    const info = await uploadAndLoad(file);
    const bbox = designBox(get()) ?? { min: [-10, -10, -10], max: [10, 10, 10] };
    const centre = [0, 1, 2].map((k) => (bbox.min[k]! + bbox.max[k]!) / 2);
    const span = Math.max(bbox.max[0]! - bbox.min[0]!, bbox.max[1]! - bbox.min[1]!, bbox.max[2]! - bbox.min[2]!) || 20;
    const edge = span * 0.25;
    const scale = kind === 'box' ? [edge, edge, edge] : kind === 'sphere' ? [edge / 2, edge / 2, edge / 2] : [edge / 2, edge, edge / 2];
    const n = get().project.ref_models.length + 1;
    addRefFor(info, `${label} ${n}`, composeTRS(centre, [0, 0, 0], scale), mode);
  });
}

/** After a reload the project knows its ref models but not their geometry: attach a fresh upload. */
export async function reuploadRefMesh(refId: string, file: File): Promise<void> {
  await guard('Upload', async () => {
    const expected = get().project.ref_models.find((r) => r.id === refId)?.mesh_id;
    const info = await uploadOrExplain(file);
    if (!info) return;
    get().updateRef(refId, { mesh_id: info.id, name: info.name });
    if (expected && expected !== info.id) {
      get().setNotice({ kind: 'warn', text: `"${info.name}" is not the file this reference model was made with (id ${info.id}, expected ${expected}).` });
    }
  });
}

/** Asks the server for every mesh the project needs but the browser does not hold (the server keeps uploads on disk). */
export async function fetchMissingMeshes(): Promise<void> {
  await guard('Fetch meshes', async () => {
    const missing = requiredMeshes(get()).filter((m) => !m.loaded);
    const gone: string[] = [];
    for (const m of [...new Map(missing.map((x) => [x.id, x])).values()]) {
      try {
        await loadMeshInto(await api.getMesh(m.id));
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) gone.push(m.label);
        else throw e;
      }
    }
    designReady();
    if (gone.length > 0) {
      get().setNotice({ kind: 'warn', text: `The server no longer has ${gone.join(', ')}: re-upload the file${gone.length > 1 ? 's' : ''}.` });
    } else if (missing.length > 0) {
      get().setNotice(null);
    }
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

// ------------------------------------------------------------------ query selections (facets / normal / plane)

/** Facet table of the design mesh at `angleDeg` (cached by mesh and angle). */
export async function fetchFacets(angleDeg?: number): Promise<MeshFacets | null> {
  const s = get();
  const meshId = s.project.design_mesh?.mesh_id;
  if (!meshId) return null;
  const angle = angleDeg ?? s.queryForm.facets.angle;
  const key = facetKey(meshId, angle);
  const hit = get().facetCache[key];
  if (hit) return hit;
  s.setQueryUi({ loading: true, error: null });
  try {
    const table = await api.meshFacets(meshId, angle);
    get().putFacets(key, table);
    return table;
  } catch (e) {
    get().setQueryUi({ error: message(e) });
    return null;
  } finally {
    get().setQueryUi({ loading: false });
  }
}

/** Facet tables and exact faces that loads/supports with `facets` selections need (markers, colours), fetched in the background. */
async function prefetchFacets(): Promise<void> {
  const p = get().project;
  const angles = new Set<number>();
  for (const it of [...p.loads, ...p.supports]) {
    if (it.selection.kind !== 'facets') continue;
    angles.add(it.selection.angle_deg);
    prefetchFacetFaces(it.selection);
  }
  for (const a of angles) await fetchFacets(a);
}

const facePending = new Map<string, Promise<number[] | null>>();

/**
 * Exact triangle ids of one facet (GET /api/meshes/{id}/facets/{facet_id}/faces), cached per facet in the store.
 * Resolves null if the request failed; callers keep their approximation then.
 */
export function ensureFacetFaces(meshId: string, angleDeg: number, facetId: number): Promise<number[] | null> {
  const key = facetFacesKey(meshId, angleDeg, facetId);
  const hit = get().facetFaceCache[key];
  if (hit) return Promise.resolve(hit);
  const inflight = facePending.get(key);
  if (inflight) return inflight;
  const p = api
    .facetFaces(meshId, facetId, angleDeg)
    .then((r) => {
      get().putFacetFaces(key, r.face_ids);
      return r.face_ids;
    })
    .catch(() => null)
    .finally(() => facePending.delete(key));
  facePending.set(key, p);
  return p;
}

function prefetchFacetFaces(sel: { mesh_id: string; angle_deg: number; facet_ids: number[] }): void {
  for (const id of sel.facet_ids) void ensureFacetFaces(sel.mesh_id, sel.angle_deg, id);
}

export function setQueryKind(kind: QueryKind): void {
  get().editQueryForm((f) => ({ ...f, kind }));
  if (kind === 'facets') void fetchFacets();
}

/** Toggle the optional `within` box; switching it on fills it from the design bbox padded by one voxel h. */
export function setNormalWithin(on: boolean): void {
  const s = get();
  const b = designBox(s); // world space: the box clips grid nodes, which sit in the transformed domain
  let box = { min: [0, 0, 0] as [number, number, number], max: [0, 0, 0] as [number, number, number] };
  if (b) {
    const span = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    const h = s.voxel.stats?.h ?? span / s.project.grid.elements_along_longest;
    box = paddedBox(b, h);
  }
  s.editQueryForm((f) => ({ ...f, normal: { ...f.normal, within: on ? box : null } }));
}

export function toggleFacet(id: number): void {
  get().editQueryForm((f) => {
    const ids = f.facets.ids.includes(id) ? f.facets.ids.filter((x) => x !== id) : [...f.facets.ids, id];
    return { ...f, facets: { ...f.facets, ids } };
  });
  const sel = get().selection.query;
  if (sel?.kind === 'facets') prefetchFacetFaces(sel);
}

let hoverSeq = 0;

/**
 * Highlight (or un-highlight with null) the faces of a facet row in the viewport: the exact ids from the faces
 * endpoint (cached per facet). While that request is in flight the old normal + bbox reconstruction stands in.
 */
export function hoverFacet(facet: FacetInfo | null): void {
  const s = get();
  const entry = designEntry(s);
  const seq = ++hoverSeq;
  if (!facet || !entry) {
    s.setHoverFaces([]);
    return;
  }
  const angle = s.queryForm.facets.angle;
  const cached = s.facetFaceCache[facetFacesKey(entry.info.id, angle, facet.id)];
  if (cached) {
    s.setHoverFaces(cached);
    return;
  }
  s.setHoverFaces(facetFaces(entry.data, facet, angle));
  void ensureFacetFaces(entry.info.id, angle, facet.id).then((ids) => {
    if (ids && seq === hoverSeq) get().setHoverFaces(ids); // a newer hover (or leaving the row) wins
  });
}

// ------------------------------------------------------------------ selection -> loads / supports / primitives

export function addPrimitive(kind: PrimitiveKind): void {
  const { min, max } = designBox(get()) ?? { min: [-10, -10, -10], max: [10, 10, 10] };
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

/** The current selection has been turned into a load/support: faces/query selections are consumed like primitives. */
function consumeSelection(sel: Selection): void {
  const s = get();
  if (isPrimitive(sel)) s.setPrimitive(null);
  else s.clearSelection();
}

export function addLoadFromSelection(): LoadSpec | null {
  const s = get();
  const sel = currentSelectionSpec(s);
  if (!sel) {
    s.setNotice({ kind: 'warn', text: 'Select faces (pick/paint), a query or a primitive first.' });
    return null;
  }
  const load: LoadSpec = { id: genId('load'), name: nextName('Load', s.project.loads), selection: sel, force: [0, 0, -1], case: 0 };
  s.addLoad(load);
  s.setActiveItem({ kind: 'load', id: load.id });
  consumeSelection(sel);
  return load;
}

export function addSupportFromSelection(): SupportSpec | null {
  const s = get();
  const sel = currentSelectionSpec(s);
  if (!sel) {
    s.setNotice({ kind: 'warn', text: 'Select faces (pick/paint), a query or a primitive first.' });
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
  consumeSelection(sel);
  return support;
}

/** Make an existing load/support's selection the current selection again (to inspect or re-use it). */
export function reselect(selection: Selection): void {
  const s = get();
  if (selection.kind === 'faces') s.setFaceSelection([...selection.face_ids]);
  else if (isPrimitive(selection)) {
    s.setPrimitive(toPrim(selection));
    s.setTool('gizmo');
  } else {
    s.setQueryForm(formFromSelection(selection, s.queryForm));
    s.setQuerySelection(structuredClone(selection));
    s.setTool('query');
    if (selection.kind === 'facets') {
      void fetchFacets(selection.angle_deg);
      prefetchFacetFaces(selection);
    }
  }
}

export function deleteActive(): void {
  const s = get();
  const a = s.activeItem;
  if (!a) return;
  if (a.kind === 'load') s.removeLoad(a.id);
  else if (a.kind === 'support') s.removeSupport(a.id);
  else if (a.kind === 'ref') s.removeRef(a.id);
}

// ------------------------------------------------------------------ project.json import / export

/**
 * Loads a project.json (RunExport or Project). The document replaces the current one; meshes the browser lacks are
 * requested from the server by id (it keeps uploads on disk), the rest are listed in the Import panel for re-upload.
 * Mesh ids are content hashes, so a re-uploaded file restores every selection that refers to it.
 */
export async function loadProjectFile(file: File): Promise<void> {
  const st = get();
  if (st.run.status === 'queued' || st.run.status === 'running') {
    st.setNotice({ kind: 'warn', text: 'Stop the running job before loading another project.' });
    return;
  }
  await guard('Load project', async () => {
    let json: unknown;
    try {
      json = JSON.parse(await file.text());
    } catch {
      throw new Error(`${file.name} is not valid JSON`);
    }
    const { doc, run, warnings } = parseProjectFile(json);
    closeStream?.();
    get().setProjectDoc(doc);
    get().setLoadedRun(run ? { run, fileName: file.name, onServer: null } : null);

    const missing = requiredMeshes(get()).filter((m) => !m.loaded);
    const gone: string[] = [];
    for (const m of [...new Map(missing.map((x) => [x.id, x])).values()]) {
      try {
        await loadMeshInto(await api.getMesh(m.id));
      } catch (e) {
        if (!(e instanceof ApiError && e.status === 404)) throw e;
        gone.push(m.id);
      }
    }
    designReady();

    if (run) await attachServerRun(run);
    const notes = [...warnings];
    if (gone.length > 0) notes.push(`${gone.length} mesh file(s) must be re-uploaded (the server does not have them). Loads and supports are kept.`);
    get().setNotice({
      kind: notes.length > 0 ? 'warn' : 'info',
      text: `Loaded "${doc.name}": ${doc.loads.length} load(s), ${doc.supports.length} support(s), ${doc.ref_models.length} reference model(s).${notes.length ? ' ' + notes.join(' ') : ''}`,
    });
  });
}

/** If the server still has the run of a loaded project.json, bind the Run/Results panels to it (exports, result mesh). */
async function attachServerRun(run: RunInfo): Promise<void> {
  let info: RunInfo | null = null;
  try {
    info = await api.getRun(run.id);
  } catch (e) {
    if (!(e instanceof ApiError && e.status === 404)) throw e;
  }
  const loaded = get().loadedRun;
  if (loaded) get().setLoadedRun({ ...loaded, onServer: info !== null });
  if (info && info.status !== 'queued' && info.status !== 'running') {
    get().patchRun({
      id: info.id,
      status: info.status,
      history: info.history ?? [],
      stats: info.stats ?? null,
      maxIter: get().project.params.max_iter,
      error: info.error,
    });
  }
}

/** The current document as a Project JSON file (no run). */
export function exportProjectJson(): void {
  const { project } = get();
  const payload: ProjectIn = { ...project };
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `${project.name.replace(/[^\w.-]+/g, '_') || 'project'}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
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

export async function resolvePreview(target?: Selection, label?: string): Promise<void> {
  const s = get();
  let sel = target ?? currentSelectionSpec(s);
  let source = label ?? 'selection';
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
    s.setNotice({ kind: 'warn', text: 'Nothing to preview: select faces, a query or a primitive, or pick a load/support first.' });
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
          const stats = msg.run?.stats ?? run.stats; // origin / h / nx.. position the density cells
          const hist = msg.run?.history ?? [];
          const history = hist.length > run.history.length ? hist : run.history;
          if (msg.type === 'started') {
            // sent on connect with the current status, so a run waiting behind another one reads "queued";
            // a second `started` (running) follows when it gets its turn
            get().patchRun({ status: msg.run?.status === 'queued' ? 'queued' : 'running', stats, history });
          } else if (msg.type === 'done') {
            get().patchRun({ status: 'done', stats, history, message: msg.message ?? null });
          } else if (msg.type === 'error') {
            get().patchRun({ status: 'error', error: msg.message ?? msg.run?.error ?? 'run failed', stats, history });
          } else {
            get().patchRun({ status: 'cancelled', stats, history, message: msg.message ?? null });
          }
          if (msg.type !== 'started') closeStream?.();
        },
        onProgress: (rec) => {
          const last = get().run.history[get().run.history.length - 1];
          if (!last || rec.it > last.it) get().pushProgress(rec);
        },
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
      // 422 "project is not runnable: ..." (no loads/supports, empty selection) and friends: nothing ran, show why
      get().patchRun({ status: e instanceof ApiError ? 'idle' : 'error', error: message(e) });
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

/** Options of every result request (download link, mesh load): the sliders plus the Trim to CAD switch. */
export function resultOptions(s: Pick<ReturnType<typeof get>, 'threshold' | 'smoothIterations' | 'trimToCad'>) {
  return { threshold: s.threshold, smooth: s.smoothIterations, trim: s.trimToCad };
}

export async function loadResultMesh(): Promise<void> {
  const s = get();
  if (!s.run.id) return;
  const id = s.run.id;
  await guard('Load result', async () => {
    const res = await api.resultStl(id, resultOptions(s));
    get().setResultStl(res.buffer, res.warnings);
  });
}

export function hideResultMesh(): void {
  get().setResultStl(null);
}

/** "Trim to CAD": affects the STL link and, when a result mesh is on screen, reloads it trimmed (or untrimmed). */
export function setTrimToCad(on: boolean): void {
  get().setTrimToCad(on);
  if (get().resultStl) void loadResultMesh();
}

/** "Color by stress": fetches GET /runs/{id}/stress once per run; the density cells are recoloured by it. */
export async function setColorByStress(on: boolean): Promise<void> {
  const s = get();
  if (!on) {
    s.setColorByStress(false);
    return;
  }
  const id = s.run.id;
  if (!id) return;
  s.setColorByStress(true);
  if (s.stress?.runId === id) return;
  s.setStressUi({ loading: true, error: null });
  try {
    const field = await api.runStress(id);
    if (get().run.id === id) get().setStress({ runId: id, ...field });
    get().setStressUi({ loading: false });
  } catch (e) {
    get().setColorByStress(false);
    get().setStressUi({ loading: false, error: message(e) });
  }
}
