// Pure helpers over store data (no three.js objects, no network).
import type { MeshFacets, PrimitiveSelection, Selection } from '../api/client';
import { type Adjacency, type MeshData, selectionCentroid as facesCentroid } from '../viewport/meshData';
import { queryLabel, selectionFaces } from './query';
import type { MeshEntry, Prim, State } from './store';

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

/**
 * World-space anchor for arrows/glyphs: face centroid mean, primitive translation, plane point, the mean centroid of the
 * faces a normal query matches, or the area-weighted centroid of the chosen facets (needs the cached /facets table).
 */
export function selectionAnchor(
  sel: Selection,
  meshes: Record<string, MeshEntry>,
  facetCache: Record<string, MeshFacets> = {},
): [number, number, number] | null {
  if (sel.kind === 'faces') {
    const m = meshes[sel.mesh_id];
    return m ? facesCentroid(m.data, sel.face_ids) : null;
  }
  if (sel.kind === 'normal') {
    const m = meshes[sel.mesh_id];
    return m ? facesCentroid(m.data, selectionFaces(sel, m.data, null)) : null;
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
    return a > 0 ? [c[0]! / a, c[1]! / a, c[2]! / a] : null;
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
