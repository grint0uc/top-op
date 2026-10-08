// STL parsing + mesh helpers for the mock backend (Node only, no dependencies).

export interface MeshGeom {
  positions: Float32Array; // merged vertices, xyz
  tris: Uint32Array; // ijk per face
  normals: Float32Array; // unit face normals (zeros for degenerate faces)
  areas: Float32Array;
}

export class StlError extends Error {}

function readBinary(buf: Buffer, n: number): Float32Array {
  const out = new Float32Array(n * 9);
  for (let f = 0; f < n; f++) {
    const o = 84 + f * 50 + 12; // skip the stored normal, recomputed from the vertices
    for (let k = 0; k < 9; k++) out[f * 9 + k] = buf.readFloatLE(o + k * 4);
  }
  return out;
}

function readAscii(text: string): Float32Array {
  const re = /vertex\s+(\S+)\s+(\S+)\s+(\S+)/g;
  const vals: number[] = [];
  for (let m = re.exec(text); m; m = re.exec(text)) vals.push(Number(m[1]), Number(m[2]), Number(m[3]));
  if (vals.length === 0 || vals.length % 9 !== 0) throw new StlError('malformed ASCII STL');
  return Float32Array.from(vals);
}

export function parseSTL(buf: Buffer): MeshGeom {
  let coords: Float32Array;
  if (buf.length >= 84 && 84 + 50 * buf.readUInt32LE(80) === buf.length) {
    coords = readBinary(buf, buf.readUInt32LE(80));
  } else if (buf.subarray(0, 5).toString('latin1').toLowerCase() === 'solid') {
    coords = readAscii(buf.toString('latin1'));
  } else {
    throw new StlError('not a valid STL file');
  }
  return fromTriangleSoup(coords);
}

/** Merge identical vertices (Map keyed on the float32 coordinates) and derive normals/areas. */
export function fromTriangleSoup(coords: Float32Array): MeshGeom {
  const nTri = coords.length / 9;
  const index = new Map<string, number>();
  const verts: number[] = [];
  const tris = new Uint32Array(nTri * 3);
  for (let f = 0; f < nTri; f++) {
    for (let c = 0; c < 3; c++) {
      const x = coords[f * 9 + c * 3]!;
      const y = coords[f * 9 + c * 3 + 1]!;
      const z = coords[f * 9 + c * 3 + 2]!;
      const key = `${x},${y},${z}`;
      let id = index.get(key);
      if (id === undefined) {
        id = verts.length / 3;
        index.set(key, id);
        verts.push(x, y, z);
      }
      tris[f * 3 + c] = id;
    }
  }
  const positions = Float32Array.from(verts);
  const normals = new Float32Array(nTri * 3);
  const areas = new Float32Array(nTri);
  for (let f = 0; f < nTri; f++) {
    const a = tris[f * 3]! * 3;
    const b = tris[f * 3 + 1]! * 3;
    const c = tris[f * 3 + 2]! * 3;
    const e1x = positions[b]! - positions[a]!;
    const e1y = positions[b + 1]! - positions[a + 1]!;
    const e1z = positions[b + 2]! - positions[a + 2]!;
    const e2x = positions[c]! - positions[a]!;
    const e2y = positions[c + 1]! - positions[a + 1]!;
    const e2z = positions[c + 2]! - positions[a + 2]!;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    const len = Math.hypot(nx, ny, nz);
    areas[f] = len / 2;
    if (len > 0) {
      normals[f * 3] = nx / len;
      normals[f * 3 + 1] = ny / len;
      normals[f * 3 + 2] = nz / len;
    }
  }
  return { positions, tris, normals, areas };
}

export function bboxOf(g: MeshGeom): [number[], number[]] {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < g.positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = g.positions[i + k]!;
      if (v < lo[k]!) lo[k] = v;
      if (v > hi[k]!) hi[k] = v;
    }
  }
  return [lo, hi];
}

export function volumeOf(g: MeshGeom): number {
  let v = 0;
  for (let f = 0; f < g.tris.length / 3; f++) {
    const a = g.tris[f * 3]! * 3;
    const b = g.tris[f * 3 + 1]! * 3;
    const c = g.tris[f * 3 + 2]! * 3;
    const p = g.positions;
    v +=
      (p[a]! * (p[b + 1]! * p[c + 2]! - p[b + 2]! * p[c + 1]!) -
        p[a + 1]! * (p[b]! * p[c + 2]! - p[b + 2]! * p[c]!) +
        p[a + 2]! * (p[b]! * p[c + 1]! - p[b + 1]! * p[c]!)) /
      6;
  }
  return Math.abs(v);
}

/** Face pairs sharing a manifold edge (exactly two faces), like trimesh.face_adjacency. */
export function adjacencyPairs(g: MeshGeom): Uint32Array {
  const nV = g.positions.length / 3;
  const edges = new Map<number, number[]>();
  for (let f = 0; f < g.tris.length / 3; f++) {
    for (let e = 0; e < 3; e++) {
      const a = g.tris[f * 3 + e]!;
      const b = g.tris[f * 3 + ((e + 1) % 3)]!;
      const key = Math.min(a, b) * nV + Math.max(a, b);
      const list = edges.get(key);
      if (list) list.push(f);
      else edges.set(key, [f]);
    }
  }
  const out: number[] = [];
  for (const list of edges.values()) if (list.length === 2) out.push(list[0]!, list[1]!);
  return Uint32Array.from(out);
}

export function isWatertight(g: MeshGeom): boolean {
  const nV = g.positions.length / 3;
  const counts = new Map<number, number>();
  for (let f = 0; f < g.tris.length / 3; f++) {
    for (let e = 0; e < 3; e++) {
      const a = g.tris[f * 3 + e]!;
      const b = g.tris[f * 3 + ((e + 1) % 3)]!;
      const key = Math.min(a, b) * nV + Math.max(a, b);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  for (const n of counts.values()) if (n !== 2) return false;
  return true;
}

/** PLAN section 3 layout: u32 n_vert, u32 n_tri, f32 xyz*n_vert, u32 ijk*n_tri, f32 nxyz*n_tri. */
export function meshBuffer(g: MeshGeom): Buffer {
  const nV = g.positions.length / 3;
  const nT = g.tris.length / 3;
  const buf = Buffer.alloc(8 + nV * 12 + nT * 12 + nT * 12);
  buf.writeUInt32LE(nV, 0);
  buf.writeUInt32LE(nT, 4);
  let o = 8;
  for (let i = 0; i < g.positions.length; i++, o += 4) buf.writeFloatLE(g.positions[i]!, o);
  for (let i = 0; i < g.tris.length; i++, o += 4) buf.writeUInt32LE(g.tris[i]!, o);
  for (let i = 0; i < g.normals.length; i++, o += 4) buf.writeFloatLE(g.normals[i]!, o);
  return buf;
}

export interface Facet {
  id: number;
  n_faces: number;
  area: number;
  normal: number[];
  centroid: number[];
  bbox: number[][];
}

/** Group faces whose normal is within angle_deg of the seed face normal, BFS over the adjacency. */
export function computeFacets(g: MeshGeom, adj: Uint32Array, angleDeg: number): Facet[] {
  const nT = g.tris.length / 3;
  const nbrs: number[][] = Array.from({ length: nT }, () => []);
  for (let i = 0; i < adj.length; i += 2) {
    nbrs[adj[i]!]!.push(adj[i + 1]!);
    nbrs[adj[i + 1]!]!.push(adj[i]!);
  }
  const cosLim = Math.cos((angleDeg * Math.PI) / 180);
  const seen = new Uint8Array(nT);
  const out: Facet[] = [];
  for (let s = 0; s < nT; s++) {
    if (seen[s]) continue;
    seen[s] = 1;
    const queue = [s];
    const sx = g.normals[s * 3]!;
    const sy = g.normals[s * 3 + 1]!;
    const sz = g.normals[s * 3 + 2]!;
    let area = 0;
    const nsum = [0, 0, 0];
    const csum = [0, 0, 0];
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let qi = 0; qi < queue.length; qi++) {
      const f = queue[qi]!;
      const a = g.areas[f]!;
      area += a;
      for (let k = 0; k < 3; k++) {
        nsum[k]! += g.normals[f * 3 + k]! * a;
        let c = 0;
        for (let v = 0; v < 3; v++) {
          const x = g.positions[g.tris[f * 3 + v]! * 3 + k]!;
          c += x / 3;
          if (x < lo[k]!) lo[k] = x;
          if (x > hi[k]!) hi[k] = x;
        }
        csum[k]! += c * a;
      }
      for (const nb of nbrs[f]!) {
        if (seen[nb]) continue;
        const dot = sx * g.normals[nb * 3]! + sy * g.normals[nb * 3 + 1]! + sz * g.normals[nb * 3 + 2]!;
        if (dot >= cosLim) {
          seen[nb] = 1;
          queue.push(nb);
        }
      }
    }
    const nl = Math.hypot(nsum[0]!, nsum[1]!, nsum[2]!) || 1;
    out.push({
      id: 0,
      n_faces: queue.length,
      area,
      normal: nsum.map((v) => v / nl),
      centroid: csum.map((v) => v / (area || 1)),
      bbox: [lo, hi],
    });
  }
  out.sort((a, b) => b.area - a.area);
  out.forEach((f, i) => (f.id = i));
  return out;
}

export function boxSTL(lo: number[], hi: number[]): Buffer {
  const [x0, y0, z0] = lo as [number, number, number];
  const [x1, y1, z1] = hi as [number, number, number];
  const v = (x: number, y: number, z: number): [number, number, number] => [x, y, z];
  const p = [
    v(x0, y0, z0), v(x1, y0, z0), v(x1, y1, z0), v(x0, y1, z0),
    v(x0, y0, z1), v(x1, y0, z1), v(x1, y1, z1), v(x0, y1, z1),
  ] as const;
  // outward-facing CCW quads split in two triangles
  const quads: [number, number, number, number][] = [
    [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7],
  ];
  const tris: [number, number, number][] = [];
  for (const [a, b, c, d] of quads) tris.push([a, b, c], [a, c, d]);
  const buf = Buffer.alloc(84 + tris.length * 50);
  buf.write('top-op mock result', 0, 'latin1');
  buf.writeUInt32LE(tris.length, 80);
  tris.forEach(([a, b, c], i) => {
    const o = 84 + i * 50;
    const pa = p[a]!;
    const pb = p[b]!;
    const pc = p[c]!;
    const e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]] as const;
    const e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]] as const;
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(n[0]!, n[1]!, n[2]!) || 1;
    n.forEach((c2, k) => buf.writeFloatLE(c2 / l, o + k * 4));
    [pa, pb, pc].forEach((pt, j) => pt.forEach((c2, k) => buf.writeFloatLE(c2, o + 12 + j * 12 + k * 4)));
  });
  return buf;
}

export interface MultipartPart {
  name: string;
  filename?: string;
  data: Buffer;
}

export function parseMultipart(body: Buffer, contentType: string): MultipartPart[] {
  const m = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType);
  const boundary = m?.[1] ?? m?.[2];
  if (!boundary) throw new StlError('multipart boundary missing');
  const delim = Buffer.from(`--${boundary}`);
  const parts: MultipartPart[] = [];
  let pos = body.indexOf(delim);
  while (pos !== -1) {
    const start = pos + delim.length;
    if (body.subarray(start, start + 2).toString() === '--') break;
    const next = body.indexOf(delim, start);
    if (next === -1) break;
    const part = body.subarray(start + 2, next - 2); // strip leading CRLF and trailing CRLF
    const split = part.indexOf('\r\n\r\n');
    if (split !== -1) {
      const head = part.subarray(0, split).toString('latin1');
      const name = /name="([^"]*)"/.exec(head)?.[1] ?? '';
      const filename = /filename="([^"]*)"/.exec(head)?.[1];
      parts.push({ name, filename, data: part.subarray(split + 4) });
    }
    pos = next;
  }
  return parts;
}
