// Parsing of a project.json: a RunExport ({project, run}, GET /api/runs/{id}/project.json) or a bare Project/ProjectIn.
// Pure (no store, no network) so every defaulting rule is in one place. Throws Error with a user-readable message.
import type { LoadSpec, RefModel, RunInfo, Selection, SupportSpec } from '../api/client';
import { DEFAULT_GRID, DEFAULT_MATERIAL, DEFAULT_PARAMS, IDENTITY, genId } from './defaults';
import type { ProjectDoc } from './store';

export interface ParsedProject {
  doc: ProjectDoc;
  run: RunInfo | null;
  warnings: string[];
}

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v);
const num = (v: unknown, d: number): number => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const nums = (v: unknown, n: number, what: string): number[] => {
  if (!Array.isArray(v) || v.length !== n || !v.every((x) => typeof x === 'number' && Number.isFinite(x))) {
    throw new Error(`${what} must be ${n} numbers`);
  }
  return [...(v as number[])];
};

function selection(raw: unknown, where: string): Selection {
  if (!isObj(raw)) throw new Error(`${where}: selection missing`);
  const meshId = (): string => {
    if (typeof raw.mesh_id !== 'string' || !raw.mesh_id) throw new Error(`${where}: ${String(raw.kind)} selection needs mesh_id`);
    return raw.mesh_id;
  };
  switch (raw.kind) {
    case 'faces':
      if (!Array.isArray(raw.face_ids)) throw new Error(`${where}: faces selection needs face_ids`);
      return { kind: 'faces', mesh_id: meshId(), face_ids: raw.face_ids.map(Number) };
    case 'facets':
      if (!Array.isArray(raw.facet_ids)) throw new Error(`${where}: facets selection needs facet_ids`);
      return { kind: 'facets', mesh_id: meshId(), facet_ids: raw.facet_ids.map(Number), angle_deg: num(raw.angle_deg, 5) };
    case 'normal': {
      let within: number[][] | null = null;
      if (raw.within != null) {
        if (!Array.isArray(raw.within) || raw.within.length !== 2) throw new Error(`${where}: within must be [[xmin,ymin,zmin],[xmax,ymax,zmax]]`);
        within = [nums(raw.within[0], 3, `${where}: within[0]`), nums(raw.within[1], 3, `${where}: within[1]`)];
      }
      return {
        kind: 'normal',
        mesh_id: meshId(),
        direction: nums(raw.direction, 3, `${where}: direction`),
        angle_deg: num(raw.angle_deg, 10),
        within,
      };
    }
    case 'plane':
      return {
        kind: 'plane',
        point: nums(raw.point, 3, `${where}: point`),
        normal: nums(raw.normal, 3, `${where}: normal`),
        tol: num(raw.tol, 0),
      };
    case 'box':
    case 'sphere':
    case 'cylinder':
      return {
        kind: raw.kind,
        transform: raw.transform === undefined ? [...IDENTITY] : nums(raw.transform, 16, `${where}: transform`),
        size: raw.size === undefined ? [1, 1, 1] : nums(raw.size, 3, `${where}: size`),
        surface_only: typeof raw.surface_only === 'boolean' ? raw.surface_only : true,
      };
    default:
      throw new Error(`${where}: unknown selection kind ${JSON.stringify(raw.kind)}`);
  }
}

export function parseProjectFile(json: unknown): ParsedProject {
  if (!isObj(json)) throw new Error('the file is not a JSON object');
  const isExport = isObj(json.project) && isObj(json.run);
  const p = isExport ? (json.project as Obj) : json;
  if (!isExport && !('design_mesh' in p) && !('loads' in p) && !('supports' in p)) {
    throw new Error('not a top-op project: expected a RunExport {project, run} or a Project');
  }
  const warnings: string[] = [];

  let design: ProjectDoc['design_mesh'] = null;
  if (isObj(p.design_mesh)) {
    const mesh_id = p.design_mesh.mesh_id;
    if (typeof mesh_id !== 'string' || !mesh_id) {
      throw new Error(
        'the design mesh is given as a disk path (a CLI case file). The GUI works on uploaded meshes: upload the file, then use a project with mesh ids.',
      );
    }
    const transform = p.design_mesh.transform === undefined ? [...IDENTITY] : nums(p.design_mesh.transform, 16, 'design_mesh.transform');
    design = { mesh_id, transform };
  }

  const refs: RefModel[] = (Array.isArray(p.ref_models) ? p.ref_models : []).map((r: unknown, i) => {
    if (!isObj(r)) throw new Error(`ref_models[${i}] is not an object`);
    if (typeof r.mesh_id !== 'string' || !r.mesh_id) throw new Error(`ref_models[${i}] has no mesh_id (disk paths are CLI-only)`);
    return {
      id: typeof r.id === 'string' && r.id ? r.id : genId('ref'),
      name: typeof r.name === 'string' ? r.name : '',
      mesh_id: r.mesh_id,
      transform: r.transform === undefined ? [...IDENTITY] : nums(r.transform, 16, `ref_models[${i}].transform`),
      mode: r.mode === 'keep_in' ? 'keep_in' : 'keep_out',
      visible: typeof r.visible === 'boolean' ? r.visible : true,
    };
  });

  const loads: LoadSpec[] = (Array.isArray(p.loads) ? p.loads : []).map((l: unknown, i) => {
    if (!isObj(l)) throw new Error(`loads[${i}] is not an object`);
    const where = `loads[${i}]`;
    return {
      id: typeof l.id === 'string' && l.id ? l.id : genId('load'),
      name: typeof l.name === 'string' ? l.name : '',
      selection: selection(l.selection, where),
      force: nums(l.force, 3, `${where}.force`),
      case: Math.max(0, Math.round(num(l.case, 0))),
    };
  });
  const supports: SupportSpec[] = (Array.isArray(p.supports) ? p.supports : []).map((s: unknown, i) => {
    if (!isObj(s)) throw new Error(`supports[${i}] is not an object`);
    const where = `supports[${i}]`;
    const fix = Array.isArray(s.fix) && s.fix.length === 3 ? s.fix.map(Boolean) : [true, true, true];
    return {
      id: typeof s.id === 'string' && s.id ? s.id : genId('sup'),
      name: typeof s.name === 'string' ? s.name : '',
      selection: selection(s.selection, where),
      fix,
    };
  });

  const doc: ProjectDoc = {
    name: typeof p.name === 'string' ? p.name : 'untitled',
    design_mesh: design,
    ref_models: refs,
    grid: { ...DEFAULT_GRID, ...(isObj(p.grid) ? p.grid : {}) },
    material: { ...DEFAULT_MATERIAL, ...(isObj(p.material) ? p.material : {}) },
    params: { ...DEFAULT_PARAMS, ...(isObj(p.params) ? p.params : {}) },
    loads,
    supports,
  };

  let run: RunInfo | null = null;
  if (isExport) {
    const r = json.run as Obj;
    if (typeof r.id === 'string' && typeof r.status === 'string') run = { history: [], stats: null, error: null, finished_at: null, ...r } as unknown as RunInfo;
  }
  return { doc, run, warnings };
}
