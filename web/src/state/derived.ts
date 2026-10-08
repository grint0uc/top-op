// Pure helpers over store data (no three.js objects, no network).
import type { MeshFacets, PrimitiveSelection, Selection } from '../api/client';
import { type Adjacency, type MeshData, selectionCentroid as facesCentroid } from '../viewport/meshData';
import { queryLabel, selectionFaces } from './query';
import type { MeshEntry, Prim, State } from './store';
import { applyPoint, worldBBox, worldView } from './worldMesh';

export const CASE_COLORS = [0xff5a5f, 0xffb020, 0xc77dff, 0x3ddc97, 0x56ccf2, 0xf78fb3];
export const SUPPORT_COLOR = 0x4f9cf9;

export const caseColor = (c: number): number => CASE_COLORS[Math.abs(c) % CASE_COLORS.length]!;
export const cssHex = (c: number): string => `#${c.toString(16).padStart(6, '0')}`;

export function toPrim(sel: PrimitiveSelection): Prim {
  return {
    kind: sel.kind,
    transform: sel.transform ? [...sel.transform] : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    size: sel.size ? [...sel.size] : [1, 1, 1],
    surface_only: sel.surface_only,
  };
}

export function isPrimitive(sel: Selection): sel is PrimitiveSelection {
  return sel.kind === 'box' || sel.kind === 'sphere' || sel.kind === 'cylinder';
}

export const facetKey = (meshId: string, angleDeg: number): string => `${meshId}@${angleDeg}`;

/** The design mesh and the matrix that puts it in world space (anchors, bbox defaults and brush work in world space). */
export interface DesignPose {
  meshId: string | null;
  transform: readonly number[] | undefined;
}

export const designPose = (s: Pick<State, 'project'>): DesignPose => ({
  meshId: s.project.design_mesh?.mesh_id ?? null,
  transform: s.project.design_mesh?.transform,
});

/** The design mesh in world coordinates (the raw MeshData when the transform is the identity). */
export function designWorld(s: Pick<State, 'project' | 'meshes'>): MeshData | null {
  const e = designEntry(s);
  return e ? worldView(e.data, s.project.design_mesh?.transform) : null;
}

/**
 * World-space anchor for arrows/glyphs: face centroid mean, primitive translation, plane point, the mean centroid of the
 * faces a normal query matches, or the area-weighted centroid of the chosen facets (needs the cached /facets table).
 * Selections on the design mesh follow its transform.
 */
export function selectionAnchor(
  sel: Selection,
  meshes: Record<string, MeshEntry>,
  facetCache: Record<string, MeshFacets> = {},
  design: DesignPose = { meshId: null, transform: undefined },
): [number, number, number] | null {
  const xf = 'mesh_id' in sel && sel.mesh_id === design.meshId ? design.transform : undefined;
  if (sel.kind === 'faces') {
    const m = meshes[sel.mesh_id];
    const c = m ? facesCentroid(m.data, sel.face_ids) : null;
    return c && xf && xf.length === 16 ? applyPoint(xf, ...c) : c; // a mean of points commutes with the affine pose
  }
  if (sel.kind === 'normal') {
    const m = meshes[sel.mesh_id];
    if (!m) return null;
    const w = worldView(m.data, xf);
    return facesCentroid(w, selectionFaces(sel, m.data, null, undefined, w));
  }
  if (sel.kind === 'facets') {
    const table = facetCache[facetKey(sel.mesh_id, sel.angle_deg)];
    const want = new Set(sel.facet_ids);
    let a = 0;
    const c = [0, 0, 0];
    for (const f of table?.facets ?? []) {
      if (!want.has(f.id)) continue;
      a += f.area;
      for (let k = 0; k < 3; k++) c[k]! += f.centroid[k]! * f.area;
    }
    if (a <= 0) return null;
    const local: [number, number, number] = [c[0]! / a, c[1]! / a, c[2]! / a];
    return xf && xf.length === 16 ? applyPoint(xf, ...local) : local;
  }
  if (isPrimitive(sel)) {
    const t = sel.transform;
    return t ? [t[12] ?? 0, t[13] ?? 0, t[14] ?? 0] : [0, 0, 0];
  }
  if (sel.kind === 'plane') return [sel.point[0] ?? 0, sel.point[1] ?? 0, sel.point[2] ?? 0];
  return null;
}

export function designEntry(s: Pick<State, 'project' | 'meshes'>): MeshEntry | null {
  const id = s.project.design_mesh?.mesh_id;
  return id ? (s.meshes[id] ?? null) : null;
}

export interface Box3 {
  min: [number, number, number];
  max: [number, number, number];
}

/**
 * The simulation domain as a box: the voxel grid when the server has reported one, otherwise the design bbox through
 * its transform. null without a design mesh.
 */
export function domainBox(s: Pick<State, 'project' | 'meshes' | 'voxel'>): Box3 | null {
  const st = s.voxel.stats;
  if (st) {
    const o = st.origin;
    return { min: [o[0]!, o[1]!, o[2]!], max: [o[0]! + st.h * st.nx, o[1]! + st.h * st.ny, o[2]! + st.h * st.nz] };
  }
  return designBox(s);
}

/** World-space bounding box of the design mesh (null without one). */
export function designBox(s: Pick<State, 'project' | 'meshes'>): Box3 | null {
  const e = designEntry(s);
  return e ? worldBBox(e.data, s.project.design_mesh?.transform) : null;
}

export function bboxDiagonal(data: MeshData): number {
  const { min, max } = data.bbox;
  return Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
}

export function selectionSummary(s: Pick<State, 'selection'>): string {
  if (s.selection.primitive) return `${s.selection.primitive.kind} primitive`;
  if (s.selection.query) return queryLabel(s.selection.query);
  const n = s.selection.faceIds.length;
  return n === 0 ? 'nothing selected' : `${n} face${n === 1 ? '' : 's'}`;
}

export function hasSelection(s: Pick<State, 'selection'>): boolean {
  return s.selection.faceIds.length > 0 || s.selection.primitive !== null || s.selection.query !== null;
}

export interface RequiredMesh {
  id: string;
  /** what the project uses it for */
  role: 'design' | 'reference';
  refId?: string;
  label: string;
  loaded: boolean;
}

/** Every mesh id the project document points at, with whether its geometry is in the browser right now. */
export function requiredMeshes(s: Pick<State, 'project' | 'meshes' | 'meshMemo'>): RequiredMesh[] {
  const out: RequiredMesh[] = [];
  const name = (id: string) => s.meshes[id]?.info.name ?? s.meshMemo[id]?.name;
  const d = s.project.design_mesh?.mesh_id;
  if (d) out.push({ id: d, role: 'design', label: name(d) ?? 'design mesh', loaded: !!s.meshes[d] });
  for (const r of s.project.ref_models) {
    if (!r.mesh_id) continue;
    out.push({ id: r.mesh_id, role: 'reference', refId: r.id, label: r.name || name(r.mesh_id) || r.id, loaded: !!s.meshes[r.mesh_id] });
  }
  return out;
}

export type { Adjacency };
