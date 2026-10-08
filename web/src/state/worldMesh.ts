// A MeshData seen through a column-major 4x4 (the design mesh transform), for code that reasons in world space
// (marker anchors, normal queries, brush, bbox defaults). Face ids and topology are unchanged.
import { Matrix3, Matrix4 } from 'three';
import type { MeshData } from '../viewport/meshData';
import { IDENTITY } from './defaults';

export function isIdentity(m: readonly number[] | null | undefined): boolean {
  return !m || m.length !== 16 || m.every((v, i) => Math.abs(v - IDENTITY[i]!) < 1e-12);
}

export function applyPoint(m: readonly number[], x: number, y: number, z: number): [number, number, number] {
  return [
    m[0]! * x + m[4]! * y + m[8]! * z + m[12]!,
    m[1]! * x + m[5]! * y + m[9]! * z + m[13]!,
    m[2]! * x + m[6]! * y + m[10]! * z + m[14]!,
  ];
}

const boxMemo = new WeakMap<MeshData, { key: string; box: MeshData['bbox'] }>();

/** Tight bbox of the mesh through the matrix: vertices only, so it is cheap enough to follow a gizmo drag. */
export function worldBBox(data: MeshData, m: readonly number[] | null | undefined): MeshData['bbox'] {
  if (!m || isIdentity(m)) return data.bbox;
  const key = m.join(',');
  const hit = boxMemo.get(data);
  if (hit?.key === key) return hit.box;
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < data.nVert; i++) {
    const p = applyPoint(m, data.positions[i * 3]!, data.positions[i * 3 + 1]!, data.positions[i * 3 + 2]!);
    for (let k = 0; k < 3; k++) {
      if (p[k]! < min[k]!) min[k] = p[k]!;
      if (p[k]! > max[k]!) max[k] = p[k]!;
    }
  }
  const box = { min, max };
  boxMemo.set(data, { key, box });
  return box;
}

const memo = new WeakMap<MeshData, { key: string; view: MeshData }>();

export function worldView(data: MeshData, m: readonly number[] | null | undefined): MeshData {
  if (!m || isIdentity(m)) return data;
  const key = m.join(',');
  const hit = memo.get(data);
  if (hit?.key === key) return hit.view;

  const positions = new Float32Array(data.positions.length);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < data.nVert; i++) {
    const p = applyPoint(m, data.positions[i * 3]!, data.positions[i * 3 + 1]!, data.positions[i * 3 + 2]!);
    for (let k = 0; k < 3; k++) {
      positions[i * 3 + k] = p[k]!;
      if (p[k]! < min[k]!) min[k] = p[k]!;
      if (p[k]! > max[k]!) max[k] = p[k]!;
    }
  }
  const centroids = new Float32Array(data.centroids.length);
  for (let f = 0; f < data.nTri; f++) {
    const p = applyPoint(m, data.centroids[f * 3]!, data.centroids[f * 3 + 1]!, data.centroids[f * 3 + 2]!);
    centroids.set(p, f * 3);
  }
  const nm = new Matrix3().getNormalMatrix(new Matrix4().fromArray(m as number[])).elements;
  const normals = new Float32Array(data.normals.length);
  for (let f = 0; f < data.nTri; f++) {
    const x = data.normals[f * 3]!;
    const y = data.normals[f * 3 + 1]!;
    const z = data.normals[f * 3 + 2]!;
    const nx = nm[0]! * x + nm[3]! * y + nm[6]! * z;
    const ny = nm[1]! * x + nm[4]! * y + nm[7]! * z;
    const nz = nm[2]! * x + nm[5]! * y + nm[8]! * z;
    const l = Math.hypot(nx, ny, nz) || 1;
    normals.set([nx / l, ny / l, nz / l], f * 3);
  }
  const view: MeshData = { ...data, positions, centroids, normals, bbox: { min, max }, adjacency: null };
  memo.set(data, { key, view });
  return view;
}
