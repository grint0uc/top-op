// Parsed form of GET /api/meshes/{id}/buffer plus the pure helpers the picker needs.

export interface Adjacency {
  offsets: Uint32Array; // length nTri + 1
  neighbors: Uint32Array;
}

export interface MeshData {
  nVert: number;
  nTri: number;
  positions: Float32Array; // indexed vertices, xyz
  tris: Uint32Array; // ijk per face
  normals: Float32Array; // unit face normals, xyz
  centroids: Float32Array; // per-face centroid, xyz
  bbox: { min: [number, number, number]; max: [number, number, number] };
  adjacency: Adjacency | null;
}

/** Layout (PLAN section 3): u32 n_vert, u32 n_tri, f32 xyz*n_vert, u32 ijk*n_tri, f32 nxyz*n_tri, little-endian. */
export function parseMeshBuffer(buf: ArrayBuffer): MeshData {
  const dv = new DataView(buf);
  const nVert = dv.getUint32(0, true);
  const nTri = dv.getUint32(4, true);
  const expected = 8 + nVert * 12 + nTri * 12 + nTri * 12;
  if (buf.byteLength < expected) throw new Error(`mesh buffer truncated: ${buf.byteLength} < ${expected}`);
  // typed-array views need 4-byte alignment; offset 8 + multiples of 4 always satisfy it
  const positions = new Float32Array(buf, 8, nVert * 3);
  const tris = new Uint32Array(buf, 8 + nVert * 12, nTri * 3);
  const normals = new Float32Array(buf, 8 + nVert * 12 + nTri * 12, nTri * 3);
  const centroids = new Float32Array(nTri * 3);
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nVert; i++) {
    for (let k = 0; k < 3; k++) {
      const v = positions[i * 3 + k]!;
      if (v < min[k]!) min[k] = v;
      if (v > max[k]!) max[k] = v;
    }
  }
  for (let f = 0; f < nTri; f++) {
    for (let k = 0; k < 3; k++) {
      centroids[f * 3 + k] =
        (positions[tris[f * 3]! * 3 + k]! + positions[tris[f * 3 + 1]! * 3 + k]! + positions[tris[f * 3 + 2]! * 3 + k]!) / 3;
    }
  }
  return { nVert, nTri, positions, tris, normals, centroids, bbox: { min, max }, adjacency: null };
}

/** CSR adjacency from the flat u32 pair list returned by /adjacency. */
export function buildAdjacency(pairs: Uint32Array, nTri: number): Adjacency {
  const offsets = new Uint32Array(nTri + 1);
  for (let i = 0; i < pairs.length; i++) offsets[pairs[i]! + 1]!++;
  for (let f = 0; f < nTri; f++) offsets[f + 1]! += offsets[f]!;
  const fill = offsets.slice(0, nTri);
  const neighbors = new Uint32Array(pairs.length);
  for (let i = 0; i < pairs.length; i += 2) {
    const a = pairs[i]!;
    const b = pairs[i + 1]!;
    neighbors[fill[a]!++] = b;
    neighbors[fill[b]!++] = a;
  }
  return { offsets, neighbors };
}

/**
 * Flat-face grow: BFS over the adjacency from `seed`, accepting neighbours whose normal is within
 * `angleDeg` of the seed's normal (seed-relative so a fillet cannot drift the region round a corner).
 */
export function growFlat(data: MeshData, adj: Adjacency, seed: number, angleDeg: number): number[] {
  const cosLim = Math.cos((angleDeg * Math.PI) / 180) - 1e-9;
  const n = data.normals;
  const sx = n[seed * 3]!;
  const sy = n[seed * 3 + 1]!;
  const sz = n[seed * 3 + 2]!;
  const seen = new Uint8Array(data.nTri);
  seen[seed] = 1;
  const queue = [seed];
  for (let qi = 0; qi < queue.length; qi++) {
    const f = queue[qi]!;
    for (let j = adj.offsets[f]!; j < adj.offsets[f + 1]!; j++) {
      const nb = adj.neighbors[j]!;
      if (seen[nb]) continue;
      if (sx * n[nb * 3]! + sy * n[nb * 3 + 1]! + sz * n[nb * 3 + 2]! >= cosLim) {
        seen[nb] = 1;
        queue.push(nb);
      }
    }
  }
  return queue;
}

export function selectionCentroid(data: MeshData, faceIds: readonly number[]): [number, number, number] | null {
  if (faceIds.length === 0) return null;
  let x = 0;
  let y = 0;
  let z = 0;
  for (const f of faceIds) {
    x += data.centroids[f * 3]!;
    y += data.centroids[f * 3 + 1]!;
    z += data.centroids[f * 3 + 2]!;
  }
  const n = faceIds.length;
  return [x / n, y / n, z / n];
}
