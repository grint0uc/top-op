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
  kind: 'plane' | 'cylinder' | 'other';
  axis: number[] | null;
  radius: number | null;
  brep_face: number | null;
  /** exact triangle ids (GET /facets/{id}/faces); not part of the wire facet */
  faces: number[];
  /** lowest triangle id of the group: discovery order, used for the fake B-rep face numbering */
  seed: number;
}

/** The wire form of a facet (FacetInfo): everything but the face list. */
export function facetInfo(f: Facet): Omit<Facet, 'faces' | 'seed'> {
  const { faces: _faces, seed: _seed, ...info } = f;
  return info;
}

/** Group faces whose normal is within angle_deg of the seed face normal, BFS over the adjacency; curved strips are merged into cylinders. */
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
      kind: 'plane',
      axis: null,
      radius: null,
      brep_face: null,
      faces: [...queue].sort((a, b) => a - b),
      seed: s,
    });
  }
  const merged = mergeCurved(g, nbrs, out);
  merged.sort((a, b) => b.area - a.area);
  merged.forEach((f, i) => (f.id = i));
  return merged;
}

/** B-rep face numbers for a STEP mock: the order the faces were discovered in, which is not the area order of the ids. */
export function numberBrepFaces(facets: Facet[]): Facet[] {
  [...facets].sort((a, b) => a.seed - b.seed).forEach((f, i) => (f.brep_face = i));
  return facets;
}

/**
 * Small facets that touch each other with a smoothly turning normal (a tessellated hole or fillet) become one facet:
 * a cylinder when the strip is perpendicular to a coordinate axis (radius from a circle fit), otherwise `other`.
 */
function mergeCurved(g: MeshGeom, nbrs: number[][], facets: Facet[]): Facet[] {
  const owner = new Int32Array(g.tris.length / 3);
  facets.forEach((f, i) => f.faces.forEach((t) => (owner[t] = i)));
  const small = facets.map((f) => f.n_faces <= 4);
  const parent = facets.map((_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i]!)));
  const cosSmooth = Math.cos((12 * Math.PI) / 180);
  for (let t = 0; t < owner.length; t++) {
    for (const u of nbrs[t]!) {
      const a = owner[t]!;
      const b = owner[u]!;
      if (a === b || !small[a] || !small[b]) continue;
      const d = g.normals[t * 3]! * g.normals[u * 3]! + g.normals[t * 3 + 1]! * g.normals[u * 3 + 1]! + g.normals[t * 3 + 2]! * g.normals[u * 3 + 2]!;
      if (d >= cosSmooth) parent[find(a)] = find(b);
    }
  }
  const groups = new Map<number, number[]>();
  facets.forEach((_, i) => groups.set(find(i), [...(groups.get(find(i)) ?? []), i]));
  const out: Facet[] = [];
  for (const members of groups.values()) {
    if (members.length < 6) {
      for (const i of members) out.push(facets[i]!);
      continue;
    }
    const parts = members.map((i) => facets[i]!);
    const faces = parts.flatMap((f) => f.faces).sort((a, b) => a - b);
    const area = parts.reduce((s, f) => s + f.area, 0);
    const centroid = [0, 1, 2].map((k) => parts.reduce((s, f) => s + f.centroid[k]! * f.area, 0) / area);
    const bbox = [0, 1, 2].map((k) => Math.min(...parts.map((f) => f.bbox[0]![k]!)));
    const bbox2 = [0, 1, 2].map((k) => Math.max(...parts.map((f) => f.bbox[1]![k]!)));
    // axis: the coordinate axis the face normals are most perpendicular to
    const perp = [0, 1, 2].map((a) => faces.reduce((s, t) => s + g.areas[t]! * g.normals[t * 3 + a]! ** 2, 0) / area);
    const axis = perp.indexOf(Math.min(...perp));
    let kind: Facet['kind'] = 'other';
    let radius: number | null = null;
    let axisVec: number[] | null = null;
    if (perp[axis]! < 0.02) {
      const [u, v] = [0, 1, 2].filter((k) => k !== axis) as [number, number];
      const pts = new Set<number>();
      faces.forEach((t) => [0, 1, 2].forEach((c) => pts.add(g.tris[t * 3 + c]!)));
      radius = fitCircle([...pts].map((i) => [g.positions[i * 3 + u]!, g.positions[i * 3 + v]!]));
      kind = 'cylinder';
      axisVec = [0, 0, 0].map((_, k) => (k === axis ? 1 : 0));
    }
    out.push({
      id: 0,
      n_faces: faces.length,
      area,
      normal: [0, 0, 0],
      centroid,
      bbox: [bbox, bbox2],
      kind,
      axis: axisVec,
      radius,
      brep_face: null,
      faces,
      seed: Math.min(...parts.map((f) => f.seed)),
    });
  }
  return out;
}

/** Kasa least-squares circle through 2D points; returns the radius. */
function fitCircle(pts: number[][]): number {
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sxz = 0, syz = 0, sz = 0;
  for (const [x, y] of pts as [number, number][]) {
    const z = x * x + y * y;
    sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; sxz += x * z; syz += y * z; sz += z;
  }
  const n = pts.length;
  // solve [sxx sxy sx; sxy syy sy; sx sy n] [D E F]^T = -[sxz syz sz]^T  (x^2 + y^2 + D x + E y + F = 0)
  const m = [
    [sxx, sxy, sx],
    [sxy, syy, sy],
    [sx, sy, n],
  ];
  const b = [-sxz, -syz, -sz];
  const det3 = (a: number[][]) =>
    a[0]![0]! * (a[1]![1]! * a[2]![2]! - a[1]![2]! * a[2]![1]!) -
    a[0]![1]! * (a[1]![0]! * a[2]![2]! - a[1]![2]! * a[2]![0]!) +
    a[0]![2]! * (a[1]![0]! * a[2]![1]! - a[1]![1]! * a[2]![0]!);
  const d = det3(m);
  if (Math.abs(d) < 1e-12) return 0;
  const col = (k: number) => m.map((row, r) => row.map((v, c) => (c === k ? b[r]! : v)));
  const [D, E, F] = [det3(col(0)) / d, det3(col(1)) / d, det3(col(2)) / d];
  return Math.sqrt(Math.max(0, (D * D) / 4 + (E * E) / 4 - F));
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
