// Pure helpers over store data (no three.js objects, no network).
import type { PrimitiveSelection, Selection } from '../api/client';
import { type Adjacency, type MeshData, selectionCentroid as facesCentroid } from '../viewport/meshData';
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

/** World-space anchor for arrows/glyphs: face centroid mean, primitive translation, or plane point. */
export function selectionAnchor(
  sel: Selection,
  meshes: Record<string, MeshEntry>,
): [number, number, number] | null {
  if (sel.kind === 'faces') {
    const m = meshes[sel.mesh_id];
    return m ? facesCentroid(m.data, sel.face_ids) : null;
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
  const n = s.selection.faceIds.length;
  return n === 0 ? 'nothing selected' : `${n} face${n === 1 ? '' : 's'}`;
}

export type { Adjacency };
