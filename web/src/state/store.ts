import { create } from 'zustand';
import type {
  DensityFrame,
  GridSpec,
  IterationRecord,
  LoadSpec,
  MaterialSpec,
  MeshInfo,
  MeshRef,
  ParamsSpec,
  PrimitiveSelection,
  RefModel,
  RunStatus,
  Selection,
  SupportSpec,
  VoxelStats,
} from '../api/client';
import type { MeshData } from '../viewport/meshData';
import { IDENTITY, defaultProject } from './defaults';
import { loadPersisted } from './persist';

export type ToolMode = 'orbit' | 'pick' | 'paint' | 'gizmo';
export type PrimitiveKind = PrimitiveSelection['kind'];
export type GizmoMode = 'translate' | 'rotate' | 'scale';

/** PrimitiveSelection with the optional wire fields filled in. */
export interface Prim {
  kind: PrimitiveKind;
  transform: number[]; // column-major Matrix4.toArray(), rotation + translation only
  size: number[];
  surface_only: boolean;
}

/** ProjectIn with every optional section present. */
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
  role: 'design' | 'ref';
  buffer: ArrayBuffer;
  data: MeshData;
}

export type ActiveItem = { kind: 'load' | 'support' | 'ref'; id: string } | null;
export type RunPhase = 'idle' | RunStatus;

export interface RunState {
  id: string | null;
  status: RunPhase;
  history: IterationRecord[];
  stats: VoxelStats | null;
  maxIter: number;
  error: string | null;
  densityFrame: DensityFrame | null;
  densityFrames: number; // binary frames received so far (also asserted by e2e)
}

export interface VoxelState {
  stats: VoxelStats | null;
  loading: boolean;
  error: string | null;
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
  densityFrame: null,
  densityFrames: 0,
});

export function genId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 8)}`;
}

/** Selection as the wire type, from whatever is currently selected in the viewport. */
export function currentSelectionSpec(s: Pick<State, 'selection' | 'project'>): Selection | null {
  const meshId = s.project.design_mesh?.mesh_id;
  if (s.selection.primitive) {
    const { kind, transform, size, surface_only } = s.selection.primitive;
    return { kind, transform: [...transform], size: [...size], surface_only };
  }
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
  selection: { faceIds: number[]; primitive: Prim | null };
  activeItem: ActiveItem;
  preview: PreviewState | null;

  // ---- analysis
  voxel: VoxelState;
  run: RunState;
  threshold: number;
  smoothIterations: number;
  ghostDesign: boolean;
  densityInfo: { mode: 'none' | 'instanced' | 'points'; count: number; it: number };
  resultStl: ArrayBuffer | null;
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
  clearSelection: () => void;
  setActiveItem: (a: ActiveItem) => void;
  setPreview: (p: PreviewState | null) => void;

  resetProject: () => void;
  setProjectName: (name: string) => void;
  setGrid: (patch: Partial<GridSpec>) => void;
  setMaterial: (patch: Partial<MaterialSpec>) => void;
  setParams: (patch: Partial<ParamsSpec>) => void;
  setProjectMeta: (m: ProjectMeta | null) => void;
  addMesh: (entry: MeshEntry) => void;
  setDesignMesh: (meshId: string) => void;
  /** After re-uploading a restored project's mesh: point every reference at the new id. */
  remapMesh: (oldId: string, newId: string) => void;

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
  setResultStl: (b: ArrayBuffer | null) => void;
}

const persisted = loadPersisted();

export const useStore = create<State>()((set) => ({
  project: persisted?.project ?? defaultProject(),
  projectMeta: null,
  meshes: {},
  meshMemo: persisted?.meshMemo ?? {},
  restored: !!persisted?.project.design_mesh,

  tool: 'orbit',
  gizmoMode: 'translate',
  growAngleDeg: 15,
  brushRadius: 5,
  selection: { faceIds: [], primitive: null },
  activeItem: null,
  preview: null,

  voxel: { stats: null, loading: false, error: null },
  run: idleRun(),
  threshold: 0.5,
  smoothIterations: 3,
  ghostDesign: true,
  densityInfo: { mode: 'none', count: 0, it: 0 },
  resultStl: null,
  busy: null,
  notice: persisted?.project.design_mesh
    ? { kind: 'info', text: 'Project restored from this browser. Mesh files are not stored: re-upload them to continue.' }
    : null,

  setNotice: (notice) => set({ notice }),
  setBusy: (busy) => set({ busy }),
  setTool: (tool) => set({ tool }),
  setGizmoMode: (gizmoMode) => set({ gizmoMode }),
  setGrowAngle: (growAngleDeg) => set({ growAngleDeg }),
  setBrushRadius: (brushRadius) => set({ brushRadius }),
  setFaceSelection: (faceIds) => set({ selection: { faceIds, primitive: null } }),
  // keep the existing (empty) faceIds array so face-layer watchers do not refire while a primitive is dragged
  setPrimitive: (primitive) =>
    set((s) => ({ selection: { faceIds: s.selection.faceIds.length ? [] : s.selection.faceIds, primitive } })),
  clearSelection: () =>
    set((s) => (s.selection.faceIds.length || s.selection.primitive ? { selection: { faceIds: [], primitive: null } } : s)),
  setActiveItem: (activeItem) => set({ activeItem }),
  setPreview: (preview) => set({ preview }),

  resetProject: () =>
    set({
      project: defaultProject(),
      projectMeta: null,
      meshes: {},
      meshMemo: {},
      restored: false,
      selection: { faceIds: [], primitive: null },
      activeItem: null,
      preview: null,
      voxel: { stats: null, loading: false, error: null },
      run: idleRun(),
      resultStl: null,
      notice: null,
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
      selection: { faceIds: [], primitive: null },
      preview: null,
    })),
  remapMesh: (oldId, newId) =>
    set((s) => {
      const remapSel = <T extends { selection: Selection }>(item: T): T =>
        'mesh_id' in item.selection && item.selection.mesh_id === oldId
          ? { ...item, selection: { ...item.selection, mesh_id: newId } }
          : item;
      const p = s.project;
      return {
        project: {
          ...p,
          design_mesh: p.design_mesh?.mesh_id === oldId ? { ...p.design_mesh, mesh_id: newId } : p.design_mesh,
          loads: p.loads.map(remapSel),
          supports: p.supports.map(remapSel),
        },
      };
    }),

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
  resetRun: () => set({ run: idleRun(), resultStl: null, densityInfo: { mode: 'none', count: 0, it: 0 } }),
  patchRun: (patch) => set((s) => ({ run: { ...s.run, ...patch } })),
  pushProgress: (rec) => set((s) => ({ run: { ...s.run, history: [...s.run.history, rec] } })),
  pushDensity: (f) => set((s) => ({ run: { ...s.run, densityFrame: f, densityFrames: s.run.densityFrames + 1 } })),
  setThreshold: (threshold) => set({ threshold }),
  setSmoothIterations: (smoothIterations) => set({ smoothIterations }),
  setGhostDesign: (ghostDesign) => set({ ghostDesign }),
  setDensityInfo: (densityInfo) => set({ densityInfo }),
  setResultStl: (resultStl) => set({ resultStl }),
}));

// Exposed for Playwright (e2e/*.spec.ts reads window.__topop.getState()); never in production builds.
if (import.meta.env.DEV || import.meta.env.VITE_MOCK === '1') window.__topop = useStore;
