// Query selections (facets / normal / plane): form model, wire conversion and client-side previews.
// The server resolves these to grid nodes; the viewport only needs an approximate set of mesh faces to
// colour, so the face tests here follow the documented semantics but are not the resolver itself.
import type { FacetInfo, MeshFacets, QuerySelection, Selection } from '../api/client';
import type { MeshData } from '../viewport/meshData';

export type Vec3 = [number, number, number];
export type QueryKind = QuerySelection['kind'];

export const AXIS_PRESETS: { label: string; v: Vec3 }[] = [
  { label: '+X', v: [1, 0, 0] },
  { label: '-X', v: [-1, 0, 0] },
  { label: '+Y', v: [0, 1, 0] },
  { label: '-Y', v: [0, -1, 0] },
  { label: '+Z', v: [0, 0, 1] },
  { label: '-Z', v: [0, 0, -1] },
];

const fix = (v: number, digits = 3): string => String(+v.toFixed(digits));

/** "+Z" for an axis-aligned vector, otherwise "[0, 0.71, 0.71]". */
export function dirName(v: readonly number[]): string {
  const len = Math.hypot(v[0] ?? 0, v[1] ?? 0, v[2] ?? 0);
  if (len > 0) {
    const u = v.map((x) => x / len);
    const hit = AXIS_PRESETS.find((p) => p.v.every((c, k) => Math.abs(c - (u[k] ?? 0)) < 1e-3));
    if (hit) return hit.label;
  }
  return `[${v.map((x) => fix(x, 2)).join(', ')}]`;
}

// ---------------------------------------------------------------- form model (what the Query tab edits)

export interface QueryForm {
  kind: QueryKind;
  normal: { dir: Vec3; angle: number; within: { min: Vec3; max: Vec3 } | null };
  plane: { point: Vec3; normal: Vec3; tol: number };
  facets: { angle: number; ids: number[] };
}

export function defaultQueryForm(centre: Vec3 = [0, 0, 0]): QueryForm {
  return {
    kind: 'normal',
    normal: { dir: [0, 0, 1], angle: 10, within: null },
    plane: { point: [...centre], normal: [0, 0, 1], tol: 0 },
    facets: { angle: 5, ids: [] },
  };
}

const valid3 = (v: readonly number[]): boolean => v.length === 3 && v.every(Number.isFinite);
const nonZero = (v: readonly number[]): boolean => Math.hypot(v[0]!, v[1]!, v[2]!) > 1e-9;

/** Wire selection for the current form, or null while the form is incomplete / invalid. */
export function formToSelection(form: QueryForm, meshId: string | null): QuerySelection | null {
  if (form.kind === 'plane') {
    const { point, normal, tol } = form.plane;
    if (!valid3(point) || !valid3(normal) || !nonZero(normal) || !(tol >= 0)) return null;
    return { kind: 'plane', point: [...point], normal: [...normal], tol };
  }
  if (!meshId) return null;
  if (form.kind === 'normal') {
    const { dir, angle, within } = form.normal;
    if (!valid3(dir) || !nonZero(dir) || !(angle > 0 && angle <= 180)) return null;
    if (within && !(valid3(within.min) && valid3(within.max))) return null;
    return {
      kind: 'normal',
      mesh_id: meshId,
      direction: [...dir],
      angle_deg: angle,
      within: within ? [[...within.min], [...within.max]] : null,
    };
  }
  const { angle, ids } = form.facets;
  if (ids.length === 0 || !(angle >= 0 && angle <= 90)) return null;
  return { kind: 'facets', mesh_id: meshId, facet_ids: [...ids], angle_deg: angle };
}

export function formFromSelection(sel: QuerySelection, prev: QueryForm): QueryForm {
  if (sel.kind === 'plane') {
    return { ...prev, kind: 'plane', plane: { point: sel.point as Vec3, normal: sel.normal as Vec3, tol: sel.tol } };
  }
  if (sel.kind === 'normal') {
    const w = sel.within;
    return {
      ...prev,
      kind: 'normal',
      normal: {
        dir: sel.direction as Vec3,
        angle: sel.angle_deg,
        within: w && w.length === 2 ? { min: w[0] as Vec3, max: w[1] as Vec3 } : null,
      },
    };
  }
  return { ...prev, kind: 'facets', facets: { angle: sel.angle_deg, ids: [...sel.facet_ids] } };
}

/** Bbox padded by one voxel: grid nodes sit up to h/2 outside the surface (see NormalSelection.within). */
export function paddedBox(bbox: { min: readonly number[]; max: readonly number[] }, pad: number): { min: Vec3; max: Vec3 } {
  const r = (v: number) => Math.round(v * 1e4) / 1e4;
  return {
    min: [r(bbox.min[0]! - pad), r(bbox.min[1]! - pad), r(bbox.min[2]! - pad)],
    max: [r(bbox.max[0]! + pad), r(bbox.max[1]! + pad), r(bbox.max[2]! + pad)],
  };
}

// ---------------------------------------------------------------- readable labels

export function queryLabel(sel: QuerySelection): string {
  if (sel.kind === 'normal') {
    return `normal ${dirName(sel.direction)} ±${fix(sel.angle_deg, 1)}°${sel.within ? ' in box' : ''}`;
  }
  if (sel.kind === 'plane') {
    const n = sel.normal;
    const len = Math.hypot(n[0]!, n[1]!, n[2]!) || 1;
    const ax = n.findIndex((c) => Math.abs(Math.abs(c) / len - 1) < 1e-3);
    const where =
      ax >= 0
        ? `${'xyz'[ax]}=${fix(sel.point[ax] ?? 0)}`
        : `n=${dirName(n)} through [${sel.point.map((x) => fix(x, 2)).join(', ')}]`;
    return `plane ${where}${sel.tol > 0 ? ` ±${fix(sel.tol)}` : ''}`;
  }
  const ids = sel.facet_ids;
  const shown = ids.slice(0, 4).map((i) => `#${i}`).join(', ');
  return `facet${ids.length === 1 ? '' : 's'} ${shown}${ids.length > 4 ? ` +${ids.length - 4}` : ''} (${fix(sel.angle_deg, 1)}°)`;
}

// ---------------------------------------------------------------- client-side face previews

function inBox(c: Float32Array, f: number, lo: readonly number[], hi: readonly number[]): boolean {
  for (let k = 0; k < 3; k++) {
    const v = c[f * 3 + k]!;
    if (v < lo[k]! || v > hi[k]!) return false;
  }
  return true;
}

export function normalFaces(data: MeshData, dir: readonly number[], angleDeg: number, within?: readonly (readonly number[])[] | null): number[] {
  const len = Math.hypot(dir[0]!, dir[1]!, dir[2]!);
  if (len === 0) return [];
  const [dx, dy, dz] = [dir[0]! / len, dir[1]! / len, dir[2]! / len];
  const cosLim = Math.cos((angleDeg * Math.PI) / 180) - 1e-9;
  const n = data.normals;
  const out: number[] = [];
  for (let f = 0; f < data.nTri; f++) {
    if (n[f * 3]! * dx + n[f * 3 + 1]! * dy + n[f * 3 + 2]! * dz < cosLim) continue;
    if (within && within.length === 2 && !inBox(data.centroids, f, within[0]!, within[1]!)) continue;
    out.push(f);
  }
  return out;
}

/** Key of the exact-faces cache: `${mesh_id}@${angle_deg}#${facet_id}`. */
export const facetFacesKey = (meshId: string, angleDeg: number, facetId: number): string => `${meshId}@${angleDeg}#${facetId}`;

/**
 * Approximate faces of one facet, reconstructed from what /facets returns (normal, bbox, no face list): faces whose
 * normal is within `angleDeg` of the facet normal and whose centroid lies in the facet bbox. Closed curved groups
 * report a zero normal; for those the bbox alone decides. Only a stand-in while GET .../facets/{id}/faces is in
 * flight (or failed): the endpoint returns the exact ids.
 */
export function facetFaces(data: MeshData, facet: FacetInfo, angleDeg: number): number[] {
  const { min, max } = data.bbox;
  const diag = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  const eps = diag * 1e-4 + 1e-6;
  const lo = facet.bbox[0]!.map((v) => v - eps);
  const hi = facet.bbox[1]!.map((v) => v + eps);
  const nl = Math.hypot(facet.normal[0]!, facet.normal[1]!, facet.normal[2]!);
  const cosLim = Math.cos((angleDeg * Math.PI) / 180) - 1e-9;
  const [nx, ny, nz] = nl > 1e-9 ? [facet.normal[0]! / nl, facet.normal[1]! / nl, facet.normal[2]! / nl] : [0, 0, 0];
  const n = data.normals;
  const out: number[] = [];
  for (let f = 0; f < data.nTri; f++) {
    if (nl > 1e-9 && n[f * 3]! * nx + n[f * 3 + 1]! * ny + n[f * 3 + 2]! * nz < cosLim) continue;
    if (inBox(data.centroids, f, lo, hi)) out.push(f);
  }
  return out;
}

const memo = new WeakMap<object, { data: MeshData; world: MeshData; facets: MeshFacets | null; exact: object; ids: number[] }>();
const NO_EXACT: Record<string, number[]> = {};

/**
 * Faces to colour for a query selection (plane: none). Memoised per selection object, mesh and tables.
 * `data` is the raw mesh (facet tables and face ids refer to it); `world` the same mesh through the design transform
 * (a normal query is a world-space direction). Facets use the exact ids from `exact` (the faces endpoint) and fall back
 * to the bbox reconstruction for facets whose request has not come back.
 */
export function selectionFaces(
  sel: Selection,
  data: MeshData,
  table: MeshFacets | null,
  exact: Record<string, number[]> = NO_EXACT,
  world: MeshData = data,
): number[] {
  if (sel.kind !== 'normal' && sel.kind !== 'facets') return [];
  const facets = sel.kind === 'facets' ? table : null; // normal queries do not depend on the table
  const ex = sel.kind === 'facets' ? exact : NO_EXACT;
  const hit = memo.get(sel);
  if (hit && hit.data === data && hit.world === world && hit.facets === facets && hit.exact === ex) return hit.ids;
  let ids: number[];
  if (sel.kind === 'normal') ids = normalFaces(world, sel.direction, sel.angle_deg, sel.within);
  else {
    const byId = new Map((facets?.facets ?? []).map((fc) => [fc.id, fc] as const));
    const set = new Set<number>();
    for (const id of sel.facet_ids) {
      const known = ex[facetFacesKey(sel.mesh_id, sel.angle_deg, id)];
      const fc = byId.get(id);
      if (known) for (const f of known) set.add(f);
      else if (fc) for (const f of facetFaces(data, fc, sel.angle_deg)) set.add(f);
    }
    ids = [...set];
  }
  memo.set(sel, { data, world, facets, exact: ex, ids });
  return ids;
}
