// top-op lite: marching cubes, smoothing and STL/zip writers. Exposes window.TopopMesh.
(function () {
  'use strict';
  // Standard marching-cubes triangle table (Lorensen/Bourke), one case per comma, edge ids in hex.
  const TRI_SRC = ',083,019,183981,12a,08312a,92a029,2832a8a98,3b2,0b28b0,19023b,1b219b98b,3a1ba3,0a108a8ba,3903b9ba9,98aa8b,478,430734,019847,419471731,12a847,34730412a,92a902847,2a9297273794,8473b2,b47b24204,90184723b,47b94b9b2921,3a13ba784,1ba14b1047b4,47890b9bab03,47b4b99ba,954,954083,054150,854835315,12a954,30812a495,52a542402,2a5325354348,95423b,0b208b495,05401523b,21525828b485,a3ba13954,4950818a18ba,54050b5bab03,54858aa8b,978579,930953573,078017157,153357,978957a12,a12950530573,802825857a52,2a5253357,7957893b2,95797292027b,23b018178157,b21b17715,958857a13a3b,5705097b010aba0,ba0b03a50807570,ba57b5,a65,0835a6,9015a6,1831985a6,165261,165126308,965906026,598582526328,23ba65,b08b20a65,01923b5a6,5a61929b298b,63b653513,08b0b50515b6,3b6036065059,65969bb98,5a6478,43047365a,1905a6847,a65197173794,612651478,125526304347,847905065026,739794329596269,3b2784a65,5a647242027b,01947823b5a6,9219b294b7b45a6,8473b53515b6,51b5b610b7b404b,059065036b63847,65969b4797b9,a4964a,4a649a083,a01a60640,83181686461a,149124264,308129249264,024426,832824426,a49a64b23,08228b49a4a6,3b201606461a,64161a48121b8b1,964936913b63,8b1810b61914641,3b6360064,648b68,7a678a89a,0730a709a67a,a671a7178180,a67a71173,126168189867,269291679093739,780706602,732672,23ba68a89867,20727b09767a9a7,1801781a767a23b,b21b17a61671,896867916b63136,091b67,7807063b0b60,7b6,76b,308b76,019b76,819831b76,a126b7,12a3086b7,2902a96b7,6b72a3a83a98,723627,708760620,276237019,162186198876,a76a17137,a7617a187108,03707a0a96a7,76a7a88a9,684b86,36b306046,86b846901,946963931b36,6846b82a1,12a30b06b046,4b846b0292a9,a93a32943b36463,823842462,042462,190234246438,194142246,8138618466a1,a10a06604,4634386a3039a93,a946a4,49576b,083495b76,50154076b,b76834354315,954a1276b,6b712a083495,76b54a42a402,348354325a52b76,723762549,954086062687,362376150540,628687218485158,954a16176137,16a176107870954,40a4a503a6a737a,76a7a854a48a,6956b9b89,36b063056095,0b805b01556b,6b3635531,12a95b9b8b56,0b306b09656912a,b85b56805a52025,6b36352a3a53,589528562382,956960062,158180568382628,156216,13616a386569896,a10a06950560,03856a,a56,b5a75b,b5ab75830,5b75ab190,a75ab7981831,b12b71751,08312717572b,9759279022b7,75272b592328982,25a235375,820852875a25,9015a35373a2,982921872a25752,135375,087071175,903935537,987597,5845a8ab8,5045b05abb30,01984a8aba45,ab4a45b34941314,2512852b8458,04b0b345b2b151b,0250592b5458b85,9452b3,25a352345384,5a2524420,3a235a385458019,5a2524192942,845853351,045105,845853905035,945,4b749b9ab,0834979b79ab,1ab1b414074b,3143481a474bab4,4b79b492b912,9749b791b2b1083,b74b42240,b74b42834324,29a279237749,9a7974a27870207,37a3a274a1a040a,1a2874,491417713,491417081871,403743,487,9a8ab8,30939bb9a,01a0a88ab,31ab3a,12b1b99b8,30939b1292b9,02b80b,32b,23828aa89,9a2092,23828a0181a8,1a2,138918,091,038,';
  const TRI = TRI_SRC.split(',').map((s) => Array.from(s, (ch) => parseInt(ch, 16)));
  // Corner order matches the Hex8 node order in CLAUDE.md.
  const CORNER = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
  const EDGE_V = [[0, 1], [1, 2], [3, 2], [0, 3], [4, 5], [5, 6], [7, 6], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const EDGE_AXIS = [0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2];

  // rho: element densities, C-order (nx, ny, nz). Samples sit at element centres, padded with 0.
  function marchingCubes(rho, nx, ny, nz, h, origin, iso) {
    const NX = nx + 2, NY = ny + 2, NZ = nz + 2;
    const F = new Float32Array(NX * NY * NZ);
    for (let ix = 0; ix < nx; ix++)
      for (let iy = 0; iy < ny; iy++)
        for (let iz = 0; iz < nz; iz++)
          F[((ix + 1) * NY + iy + 1) * NZ + iz + 1] = rho[(ix * ny + iy) * nz + iz];
    const coff = CORNER.map(([a, b, c]) => a * NY * NZ + b * NZ + c);
    const pos = [];
    const idx = [];
    const vmap = new Map();
    const cv = new Float32Array(8);
    const tri = [0, 0, 0];
    for (let i = 0; i < NX - 1; i++) {
      for (let j = 0; j < NY - 1; j++) {
        for (let k = 0; k < NZ - 1; k++) {
          const base = (i * NY + j) * NZ + k;
          let cube = 0;
          for (let c = 0; c < 8; c++) {
            const v = F[base + coff[c]];
            cv[c] = v;
            if (v < iso) cube |= 1 << c;
          }
          if (cube === 0 || cube === 255) continue;
          const edges = TRI[cube];
          for (let t = 0; t + 2 < edges.length; t += 3) {
            for (let m = 0; m < 3; m++) {
              const e = edges[t + m];
              const a = EDGE_V[e][0], b = EDGE_V[e][1];
              const key = (base + coff[a]) * 3 + EDGE_AXIS[e];
              let vi = vmap.get(key);
              if (vi === undefined) {
                const s = (iso - cv[a]) / (cv[b] - cv[a]);
                const A = CORNER[a], B = CORNER[b];
                vi = pos.length / 3;
                pos.push(
                  origin[0] + h * (i + A[0] + s * (B[0] - A[0]) - 0.5),
                  origin[1] + h * (j + A[1] + s * (B[1] - A[1]) - 0.5),
                  origin[2] + h * (k + A[2] + s * (B[2] - A[2]) - 0.5),
                );
                vmap.set(key, vi);
              }
              tri[m] = vi;
            }
            if (tri[0] !== tri[1] && tri[1] !== tri[2] && tri[0] !== tri[2]) idx.push(tri[0], tri[1], tri[2]);
          }
        }
      }
    }
    const positions = new Float32Array(pos);
    const indices = new Uint32Array(idx);
    if (signedVolume(positions, indices) < 0) {
      for (let t = 0; t < indices.length; t += 3) {
        const tmp = indices[t + 1];
        indices[t + 1] = indices[t + 2];
        indices[t + 2] = tmp;
      }
    }
    return { positions, indices };
  }

  function signedVolume(p, ix) {
    let v = 0;
    for (let t = 0; t < ix.length; t += 3) {
      const a = 3 * ix[t], b = 3 * ix[t + 1], c = 3 * ix[t + 2];
      v += p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1])
        - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c])
        + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
    }
    return v / 6;
  }

  // Taubin smoothing (lambda/mu pairs): Laplacian smoothing without the volume loss.
  function smooth(positions, indices, passes) {
    if (!passes) return positions;
    const nv = positions.length / 3;
    const deg = new Int32Array(nv + 1);
    for (let t = 0; t < indices.length; t++) deg[indices[t] + 1] += 2;
    for (let v = 0; v < nv; v++) deg[v + 1] += deg[v];
    const nb = new Int32Array(deg[nv]);
    const fill = deg.slice(0, nv);
    for (let t = 0; t < indices.length; t += 3) {
      for (let m = 0; m < 3; m++) {
        const a = indices[t + m], b = indices[t + ((m + 1) % 3)];
        nb[fill[a]++] = b;
        nb[fill[b]++] = a;
      }
    }
    let p = positions;
    for (let pass = 0; pass < passes; pass++) {
      for (const lam of [0.5, -0.53]) {
        const q = new Float32Array(p.length);
        for (let v = 0; v < nv; v++) {
          const s = deg[v], e = deg[v + 1], n = e - s;
          if (!n) {
            q[3 * v] = p[3 * v]; q[3 * v + 1] = p[3 * v + 1]; q[3 * v + 2] = p[3 * v + 2];
            continue;
          }
          let ax = 0, ay = 0, az = 0;
          for (let k = s; k < e; k++) {
            const w = 3 * nb[k];
            ax += p[w]; ay += p[w + 1]; az += p[w + 2];
          }
          q[3 * v] = p[3 * v] + lam * (ax / n - p[3 * v]);
          q[3 * v + 1] = p[3 * v + 1] + lam * (ay / n - p[3 * v + 1]);
          q[3 * v + 2] = p[3 * v + 2] + lam * (az / n - p[3 * v + 2]);
        }
        p = q;
      }
    }
    return p;
  }

  function triNormal(p, a, b, c) {
    const ux = p[b] - p[a], uy = p[b + 1] - p[a + 1], uz = p[b + 2] - p[a + 2];
    const vx = p[c] - p[a], vy = p[c + 1] - p[a + 1], vz = p[c + 2] - p[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    return [nx / l, ny / l, nz / l];
  }

  function stlBinary(p, ix) {
    const nt = ix.length / 3;
    const buf = new ArrayBuffer(84 + 50 * nt);
    const dv = new DataView(buf);
    const head = 'top-op lite result';
    for (let i = 0; i < head.length; i++) dv.setUint8(i, head.charCodeAt(i));
    dv.setUint32(80, nt, true);
    let o = 84;
    for (let t = 0; t < nt; t++) {
      const a = 3 * ix[3 * t], b = 3 * ix[3 * t + 1], c = 3 * ix[3 * t + 2];
      const n = triNormal(p, a, b, c);
      for (const x of n) { dv.setFloat32(o, x, true); o += 4; }
      for (const v of [a, b, c]) for (let d = 0; d < 3; d++) { dv.setFloat32(o, p[v + d], true); o += 4; }
      o += 2;
    }
    return new Uint8Array(buf);
  }

  function stlAscii(p, ix) {
    const f = (x) => String(+x.toPrecision(7));
    const out = ['solid topop_lite'];
    for (let t = 0; t < ix.length; t += 3) {
      const a = 3 * ix[t], b = 3 * ix[t + 1], c = 3 * ix[t + 2];
      const n = triNormal(p, a, b, c);
      out.push(` facet normal ${f(n[0])} ${f(n[1])} ${f(n[2])}`, '  outer loop');
      for (const v of [a, b, c]) out.push(`   vertex ${f(p[v])} ${f(p[v + 1])} ${f(p[v + 2])}`);
      out.push('  endloop', ' endfacet');
    }
    out.push('endsolid topop_lite', '');
    return out.join('\n');
  }

  let crcTable = null;
  function crc32(data) {
    if (!crcTable) {
      crcTable = new Uint32Array(256);
      for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        crcTable[n] = c >>> 0;
      }
    }
    let c = 0xffffffff;
    for (let i = 0; i < data.length; i++) c = crcTable[(c ^ data[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }

  // Single-file, stored (uncompressed) zip archive.
  function zipOne(name, data) {
    const nameBytes = new TextEncoder().encode(name);
    const crc = crc32(data);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, data.length, true);
    local.setUint32(22, data.length, true);
    local.setUint16(26, nameBytes.length, true);
    const central = new DataView(new ArrayBuffer(46));
    central.setUint32(0, 0x02014b50, true);
    central.setUint16(4, 20, true);
    central.setUint16(6, 20, true);
    central.setUint32(16, crc, true);
    central.setUint32(20, data.length, true);
    central.setUint32(24, data.length, true);
    central.setUint16(28, nameBytes.length, true);
    const cdOffset = 30 + nameBytes.length + data.length;
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, 1, true);
    end.setUint16(10, 1, true);
    end.setUint32(12, 46 + nameBytes.length, true);
    end.setUint32(16, cdOffset, true);
    const parts = [new Uint8Array(local.buffer), nameBytes, data, new Uint8Array(central.buffer), nameBytes, new Uint8Array(end.buffer)];
    const out = new Uint8Array(parts.reduce((s, x) => s + x.length, 0));
    let o = 0;
    for (const x of parts) { out.set(x, o); o += x.length; }
    return out;
  }

  window.TopopMesh = { marchingCubes, smooth, signedVolume, stlBinary, stlAscii, zipOne, crc32 };
})();
