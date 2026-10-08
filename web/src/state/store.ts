import { create } from 'zustand';
import type {
  DensityFrame,
  GridSpec,
  IterationRecord,
  LoadSpec,
  MaterialSpec,
  MeshFacets,
  MeshInfo,
  MeshRef,
  ParamsSpec,
  PrimitiveSelection,
  QuerySelection,
  RefModel,
  RunInfo,
  RunStatus,
  Selection,
  StressField,
  SupportSpec,
  VoxelStats,
} from '../api/client';
import type { MeshData } from '../viewport/meshData';
import { IDENTITY, defaultProject } from './defaults';
import { loadPersisted } from './persist';
import { type QueryForm, defaultQueryForm, formToSelection } from './query';

export type ToolMode = 'orbit' | 'pick' | 'paint' | 'gizmo' | 'query';
export type PrimitiveKind = PrimitiveSelection['kind'];
export type GizmoMode = 'translate' | 'rotate' | 'scale';

/** PrimitiveSelection with the optional wire fields filled in. */
export interface Prim {
  kind: PrimitiveKind;
  transform: number[]; // column-major Matrix4.toArray(), rotation + translation only
  size: number[];
  surface_only: boolean;
}

/**
 * ProjectIn with every optional section present.
 *
 * Transforms: `ref_models[].transform` and `design_mesh.transform` are the full column-major matrix (translation,
 * rotation AND scale) applied to the mesh's own coordinates. The viewport draws the design mesh through its matrix, so
 * picking, painting and markers work in world space; face ids still index the raw mesh.
 */
export interface ProjectDoc {
  name: string;
  design_mesh: MeshRef | null;
  ref_models: RefModel[];
  grid: GridSpec;
  material: MaterialSpec;
  params: ParamsSpec;
  loads: LoadSpec[];
  supports: SupportSpec[];
}

export interface ProjectMeta {
  id: string;
  created_at: string;
  updated_at: string;
}

export interface MeshEntry {
  info: MeshInfo;
  buffer: ArrayBuffer;
  data: MeshData;
}

/** A project.json that was loaded: its run record (if any) and whether the server still knows that run. */
export interface LoadedRun {
  run: RunInfo;
  fileName: string;
  onServer: boolean | null; // null while checking
}

export type ActiveItem = { kind: 'load' | 'support' | 'ref' | 'design'; id: string } | null;
export type RunPhase = 'idle' | RunStatus;

export interface RunState {
  id: string | null;
  status: RunPhase;
  history: IterationRecord[];
  stats: VoxelStats | null;
  maxIter: number;
  error: string | null;
  message: string | null; // server's final status text, e.g. "stopped at max_iter=8"
  densityFrame: DensityFrame | null;
  densityFrames: number; // binary frames received so far (also asserted by e2e)
}

export interface VoxelState {
  stats: VoxelStats | null;
  loading: boolean;
  error: string | null;
}

/** A fetched /stress field and the run it belongs to. */
export interface StressState extends StressField {
  runId: string;
}

export interface PreviewState {
  count: number;
  truncated: boolean;
  xyz: Float32Array;
  source: string;
}

export interface Notice {
  kind: 'info' | 'warn' | 'error';
  text: string;
}

const idleRun = (): RunState => ({
  id: null,
  status: 'idle',
  history: [],
  stats: null,
  maxIter: 0,
  error: null,
  message: null,
  densityFrame: null,
  densityFrames: 0,
});

export { genId } from './defaults';

/** Selection as the wire type, from whatever is currently selected in the viewport. */
export function currentSelectionSpec(s: Pick<State, 'selection' | 'project'>): Selection | null {
  const meshId = s.project.design_mesh?.mesh_id;
  if (s.selection.primitive) {
    const { kind, transform, size, surface_only } = s.selection.primitive;
    return { kind, transform: [...transform], size: [...size], surface_only };
  }
  if (s.selection.query) return structuredClone(s.selection.query);
  if (meshId && s.selection.faceIds.length > 0) {
    return { kind: 'faces', mesh_id: meshId, face_ids: [...s.selection.faceIds] };
  }
  return null;
}

export interface State {
  // ---- project document (what is saved to localStorage and sent to the server)
  project: ProjectDoc;
  projectMeta: ProjectMeta | null; // server-side id once created
  meshes: Record<string, MeshEntry>;
  /** mesh_id -> {name, n_faces} remembered across reloads so a restored project can tell the user what to re-upload. */
  meshMemo: Record<string, { name: string; n_faces: number }>;
  restored: boolean;

  // ---- interaction
  tool: ToolMode;
  gizmoMode: GizmoMode;
  growAngleDeg: number;
  brushRadius: number;
  /** At most one of the three is set: clicked/painted faces, a primitive, or a query (facets/normal/plane). */
  selection: { faceIds: number[]; primitive: Prim | null; query: QuerySelection | null };
  activeItem: ActiveItem;
  preview: PreviewState | null;
  /** Query tab: the form is the source of truth; every edit re-derives `selection.query` from it. */
  queryForm: QueryForm;
  queryUi: { loading: boolean; error: string | null };
  /** GET /facets responses keyed `${mesh_id}@${angle_deg}`. */
  facetCache: Record<string, MeshFacets>;
  /** GET .../facets/{id}/faces responses keyed `${mesh_id}@${angle_deg}#${facet_id}` (exact triangle ids). */
  facetFaceCache: Record<string, number[]>;
  /** Faces drawn in the hover colour (facet row under the pointer). */
  hoverFaces: number[];
  loadedRun: LoadedRun | null;

  // ---- analysis
  voxel: VoxelState;
  run: RunState;
  threshold: number;
  smoothIterations: number;
  ghostDesign: boolean;
  densityInfo: { mode: 'none' | 'instanced' | 'points'; count: number; it: number };
  resultStl: ArrayBuffer | null;
  /** `X-Topop-Warnings` of the last result mesh load */
  resultWarnings: string | null;
  /** apply `trim=true` (intersect with the original CAD) to the STL download and the result mesh */
  trimToCad: boolean;
  colorByStress: boolean;
  stress: StressState | null;
  stressUi: { loading: boolean; error: string | null };
  busy: string | null;
  notice: Notice | null;

  // ---- actions
  setNotice: (n: Notice | null) => void;
  setBusy: (b: string | null) => void;
  setTool: (t: ToolMode) => void;
  setGizmoMode: (m: GizmoMode) => void;
  setGrowAngle: (deg: number) => void;
  setBrushRadius: (r: number) => void;
  setFaceSelection: (ids: number[]) => void;
  setPrimitive: (p: Prim | null) => void;
  setQuerySelection: (q: QuerySelection | null) => void;
  clearSelection: () => void;
  /** Edit the Query form; the resulting selection (null while the form is invalid) replaces the current one. */
  editQueryForm: (fn: (f: QueryForm) => QueryForm) => void;
  /** Put the form's selection in the store without editing it (e.g. after a load was added). */
  commitQueryForm: () => void;
  /** Set the form without committing (new design mesh: sensible defaults). */
  setQueryForm: (f: QueryForm) => void;
  setQueryUi: (patch: Partial<State['queryUi']>) => void;
  putFacets: (key: string, facets: MeshFacets) => void;
  putFacetFaces: (key: string, ids: number[]) => void;
  setHoverFaces: (ids: number[]) => void;
  setLoadedRun: (r: LoadedRun | null) => void;
  setActiveItem: (a: ActiveItem) => void;
  setPreview: (p: PreviewState | null) => void;

  resetProject: () => void;
  /** Replace the whole document (project.json import). Loaded meshes are kept: they are content-addressed. */
  setProjectDoc: (doc: ProjectDoc) => void;
  setProjectName: (name: string) => void;
  setGrid: (patch: Partial<GridSpec>) => void;
  setMaterial: (patch: Partial<MaterialSpec>) => void;
  setParams: (patch: Partial<ParamsSpec>) => void;
  setProjectMeta: (m: ProjectMeta | null) => void;
  addMesh: (entry: MeshEntry) => void;
  setDesignMesh: (meshId: string) => void;
  /** Full column-major matrix of the design mesh (MeshRef.transform). */
  setDesignTransform: (matrix: number[]) => void;
  /**
   * The design mesh was replaced by a different file (ids are content hashes). Face and facet selections index the
   * old mesh and are dropped; normal/plane/primitive selections are geometry-agnostic and follow the new id.
   * Returns how many loads/supports were dropped.
   */
  adoptDesignMesh: (oldId: string, newId: string) => number;

  addLoad: (l: LoadSpec) => void;
  updateLoad: (id: string, patch: Partial<LoadSpec>) => void;
  removeLoad: (id: string) => void;
  addSupport: (s: SupportSpec) => void;
  updateSupport: (id: string, patch: Partial<SupportSpec>) => void;
  removeSupport: (id: string) => void;
  addRef: (r: RefModel) => void;
  updateRef: (id: string, patch: Partial<RefModel>) => void;
  removeRef: (id: string) => void;

  setVoxel: (patch: Partial<VoxelState>) => void;
  resetRun: () => void;
  patchRun: (patch: Partial<RunState>) => void;
  pushProgress: (rec: IterationRecord) => void;
  pushDensity: (f: DensityFrame) => void;
  setThreshold: (t: number) => void;
  setSmoothIterations: (n: number) => void;
  setGhostDesign: (b: boolean) => void;
  setDensityInfo: (i: State['densityInfo']) => void;
  setResultStl: (b: ArrayBuffer | null, warnings?: string | null) => void;
  setTrimToCad: (b: boolean) => void;
  setStress: (s: StressState | null) => void;
  setColorByStress: (b: boolean) => void;
  setStressUi: (patch: Partial<State['stressUi']>) => void;
}

const persisted = loadPersisted();

export const useStore = create<State>()((set, get) => ({
  project: persisted?.project ?? defaultProject(),
  projectMeta: null,
  meshes: {},
  meshMemo: persisted?.meshMemo ?? {},
  restored: !!persisted?.project.design_mesh,

  tool: 'orbit',
  gizmoMode: 'translate',
  growAngleDeg: 15,
  brushRadius: 5,
  selection: { faceIds: [], primitive: null, query: null },
  activeItem: null,
  preview: null,
  queryForm: defaultQueryForm(),
  queryUi: { loading: false, error: null },
  facetCache: {},
  facetFaceCache: {},
  hoverFaces: [],
  loadedRun: null,

  voxel: { stats: null, loading: false, error: null },
  run: idleRun(),
  threshold: 0.5,
  smoothIterations: 3,
  ghostDesign: true,
  densityInfo: { mode: 'none', count: 0, it: 0 },
  resultStl: null,
  resultWarnings: null,
  trimToCad: false,
  colorByStress: false,
  stress: null,
  stressUi: { loading: false, error: null },
  busy: null,
  notice: persisted?.project.design_mesh
    ? {
        kind: 'info',
        text: 'Project restored from this browser. Mesh files are not stored here: re-upload them (same file, same id) or fetch them from the server.',
      }
    : null,

  setNotice: (notice) => set({ notice }),
  setBusy: (busy) => set({ busy }),
  setTool: (tool) => set({ tool }),
  setGizmoMode: (gizmoMode) => set({ gizmoMode }),
  setGrowAngle: (growAngleDeg) => set({ growAngleDeg }),
  setBrushRadius: (brushRadius) => set({ brushRadius }),
  setFaceSelection: (faceIds) => set({ selection: { faceIds, primitive: null, query: null } }),
  // keep the existing (empty) faceIds array so face-layer watchers do not refire while a primitive is dragged
  setPrimitive: (primitive) =>
    set((s) => ({
      selection: {
        faceIds: s.selection.faceIds.length ? [] : s.selection.faceIds,
        primitive,
        query: primitive ? null : s.selection.query,
      },
    })),
  setQuerySelection: (query) =>
    set((s) => ({ selection: { faceIds: s.selection.faceIds.length ? [] : s.selection.faceIds, primitive: null, query } })),
  clearSelection: () =>
    set((s) =>
      s.selection.faceIds.length || s.selection.primitive || s.selection.query
        ? { selection: { faceIds: [], primitive: null, query: null } }
        : s,
    ),
  editQueryForm: (fn) => {
    const queryForm = fn(get().queryForm);
    set({ queryForm });
    get().setQuerySelection(formToSelection(queryForm, get().project.design_mesh?.mesh_id ?? null));
  },
  commitQueryForm: () => get().setQuerySelection(formToSelection(get().queryForm, get().project.design_mesh?.mesh_id ?? null)),
  setQueryForm: (queryForm) => set({ queryForm }),
  setQueryUi: (patch) => set((s) => ({ queryUi: { ...s.queryUi, ...patch } })),
  putFacets: (key, facets) => set((s) => ({ facetCache: { ...s.facetCache, [key]: facets } })),
  putFacetFaces: (key, ids) => set((s) => ({ facetFaceCache: { ...s.facetFaceCache, [key]: ids } })),
  setHoverFaces: (hoverFaces) => set((s) => (hoverFaces.length === 0 && s.hoverFaces.length === 0 ? s : { hoverFaces })),
  setLoadedRun: (loadedRun) => set({ loadedRun }),
  setActiveItem: (activeItem) => set({ activeItem }),
  setPreview: (preview) => set({ preview }),

  resetProject: () =>
    set({
      project: defaultProject(),
      projectMeta: null,
      meshes: {},
      meshMemo: {},
      restored: false,
      selection: { faceIds: [], primitive: null, query: null },
      queryForm: defaultQueryForm(),
      queryUi: { loading: false, error: null },
      facetCache: {},
      facetFaceCache: {},
      hoverFaces: [],
      loadedRun: null,
      activeItem: null,
      preview: null,
      voxel: { stats: null, loading: false, error: null },
      run: idleRun(),
      resultStl: null,
      resultWarnings: null,
      colorByStress: false,
      stress: null,
      stressUi: { loading: false, error: null },
      notice: null,
    }),
  setProjectDoc: (project) =>
    set({
      project,
      projectMeta: null,
      restored: false,
      selection: { faceIds: [], primitive: null, query: null },
      queryForm: defaultQueryForm(),
      hoverFaces: [],
      activeItem: null,
      preview: null,
      voxel: { stats: null, loading: false, error: null },
      run: idleRun(),
      resultStl: null,
      resultWarnings: null,
      colorByStress: false,
      stress: null,
      stressUi: { loading: false, error: null },
      densityInfo: { mode: 'none', count: 0, it: 0 },
    }),
  setProjectName: (name) => set((s) => ({ project: { ...s.project, name } })),
  setGrid: (patch) => set((s) => ({ project: { ...s.project, grid: { ...s.project.grid, ...patch } } })),
  setMaterial: (patch) => set((s) => ({ project: { ...s.project, material: { ...s.project.material, ...patch } } })),
  setParams: (patch) => set((s) => ({ project: { ...s.project, params: { ...s.project.params, ...patch } } })),
  setProjectMeta: (projectMeta) => set({ projectMeta }),
  addMesh: (entry) =>
    set((s) => ({
      meshes: { ...s.meshes, [entry.info.id]: entry },
      meshMemo: { ...s.meshMemo, [entry.info.id]: { name: entry.info.name, n_faces: entry.info.n_faces } },
    })),
  setDesignMesh: (meshId) =>
    set((s) => ({
      project: { ...s.project, design_mesh: { mesh_id: meshId, transform: [...IDENTITY] } },
      selection: { faceIds: [], primitive: null, query: null },
      hoverFaces: [],
      preview: null,
    })),
  setDesignTransform: (transform) =>
    set((s) =>
      s.project.design_mesh
        ? {
            project: { ...s.project, design_mesh: { ...s.project.design_mesh, transform: [...transform] } },
            preview: null, // resolved points were computed for the old pose
          }
        : s,
    ),
  adoptDesignMesh: (oldId, newId) => {
    let dropped = 0;
    const follow = <T extends { selection: Selection }>(items: T[]): T[] =>
      items.flatMap((item) => {
        const sel = item.selection;
        if (!('mesh_id' in sel) || sel.mesh_id !== oldId) return [item];
        if (sel.kind === 'faces' || sel.kind === 'facets') {
          dropped++;
          return [];
        }
        return [{ ...item, selection: { ...sel, mesh_id: newId } }];
      });
    set((s) => ({
      project: { ...s.project, loads: follow(s.project.loads), supports: follow(s.project.supports) },
      activeItem: null,
    }));
    return dropped;
  },

  addLoad: (l) => set((s) => ({ project: { ...s.project, loads: [...s.project.loads, l] } })),
  updateLoad: (id, patch) =>
    set((s) => ({ project: { ...s.project, loads: s.project.loads.map((l) => (l.id === id ? { ...l, ...patch } : l)) } })),
  removeLoad: (id) =>
    set((s) => ({
      project: { ...s.project, loads: s.project.loads.filter((l) => l.id !== id) },
      activeItem: s.activeItem?.id === id ? null : s.activeItem,
    })),
  addSupport: (sp) => set((s) => ({ project: { ...s.project, supports: [...s.project.supports, sp] } })),
  updateSupport: (id, patch) =>
    set((s) => ({
      project: { ...s.project, supports: s.project.supports.map((x) => (x.id === id ? { ...x, ...patch } : x)) },
    })),
  removeSupport: (id) =>
    set((s) => ({
      project: { ...s.project, supports: s.project.supports.filter((x) => x.id !== id) },
      activeItem: s.activeItem?.id === id ? null : s.activeItem,
    })),
  addRef: (r) => set((s) => ({ project: { ...s.project, ref_models: [...s.project.ref_models, r] } })),
  updateRef: (id, patch) =>
    set((s) => ({
      project: { ...s.project, ref_models: s.project.ref_models.map((r) => (r.id === id ? { ...r, ...patch } : r)) },
    })),
  removeRef: (id) =>
    set((s) => ({
      project: { ...s.project, ref_models: s.project.ref_models.filter((r) => r.id !== id) },
      activeItem: s.activeItem?.id === id ? null : s.activeItem,
    })),

  setVoxel: (patch) => set((s) => ({ voxel: { ...s.voxel, ...patch } })),
  resetRun: () =>
    set({
      run: idleRun(),
      resultStl: null,
      resultWarnings: null,
      colorByStress: false,
      stress: null,
      stressUi: { loading: false, error: null },
      densityInfo: { mode: 'none', count: 0, it: 0 },
    }),
  patchRun: (patch) => set((s) => ({ run: { ...s.run, ...patch } })),
  pushProgress: (rec) => set((s) => ({ run: { ...s.run, history: [...s.run.history, rec] } })),
  pushDensity: (f) => set((s) => ({ run: { ...s.run, densityFrame: f, densityFrames: s.run.densityFrames + 1 } })),
  setThreshold: (threshold) => set({ threshold }),
  setSmoothIterations: (smoothIterations) => set({ smoothIterations }),
  setGhostDesign: (ghostDesign) => set({ ghostDesign }),
  setDensityInfo: (densityInfo) => set({ densityInfo }),
  setResultStl: (resultStl, warnings = null) => set({ resultStl, resultWarnings: resultStl ? warnings : null }),
  setTrimToCad: (trimToCad) => set({ trimToCad }),
  setStress: (stress) => set({ stress }),
  setColorByStress: (colorByStress) => set({ colorByStress }),
  setStressUi: (patch) => set((s) => ({ stressUi: { ...s.stressUi, ...patch } })),
}));

// Exposed for Playwright (e2e/*.spec.ts reads window.__topop.getState()); never in production builds.
if (import.meta.env.DEV || import.meta.env.VITE_MOCK === '1') window.__topop = useStore;
