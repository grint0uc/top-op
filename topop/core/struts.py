"""Strut (truss) post-processing of a SIMP result.

Two ways to get a bar network, then one meshing and verification path:
- layout (default): plastic minimum-volume ground-structure LP (Dorn, Gomory & Greenberg 1964)
  over nodes at the loads, supports, keep-in boundaries and sampled from the SIMP solid;
- skeleton: medial axis of the SIMP solid (rho >= 0.5), simplified to straight bars with radii
  from the distance transform. For results that are already beam-like.
Bars become capsules (hull of two spheres, so joints are spherical), unioned with the keep-in
bodies (manifold3d), trimmed to the design mesh, voxelized back onto the problem grid and solved.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy import ndimage
from scipy.optimize import linprog
from scipy.spatial import cKDTree

from topop.core.export import trim_to_design
from topop.core.fem import Assembler, von_mises
from topop.core.problem import HEX8_OFFSETS, Grid, Problem
from topop.core.solver import LinearSolver
from topop.core.voxelize import voxelize_mesh

NodeKind = Literal["sample", "load", "support", "attach"]
KIND_CODE = {"sample": 0, "load": 1, "support": 2, "attach": 3}
MAX_CLUSTERS = 16  # representative points per load / support
AREA_DROP = 1e-3  # bars below this fraction of the largest area are removed
COLLINEAR_COS = np.cos(np.radians(10.0))
VOLUME_TOL = 0.08  # one re-mesh when the strut volume misses the target by more than this


@dataclass
class StrutParams:
    mode: Literal["layout", "skeleton"] = "layout"
    sigma_allow: float = 20.0  # LP stress limit (units of the problem); sets the raw areas
    node_spacing: float | None = None  # sampled-node spacing; None -> 4 h
    max_bar_length: float | None = None  # None -> 0.4 * diagonal of the active bbox
    target_volume: float | None = None  # None -> SIMP material volume; <= 0 -> keep sigma sizing
    min_radius: float | None = None  # None -> max(1.0, 0.8 h)
    sample: Literal["solid", "active"] = "solid"  # layout nodes from rho >= rho_threshold or all
    rho_threshold: float = 0.3
    prune: float = 0.05  # drop bars thinner than this fraction of the largest area, re-solve
    max_nodes: int = 400
    max_candidates: int = 40_000
    penal: float = 3.0  # SIMP penalization for the reference compliance
    segments: int = 16  # circular segments of the capsules
    verify: bool = True


@dataclass
class StrutResult:
    mesh: trimesh.Trimesh
    nodes: np.ndarray  # (n, 3) world coordinates
    bars: np.ndarray  # (m, 3) float: node i, node j, radius
    compliance: list[float]  # per load case, voxelized struts (solid E, Emin elsewhere)
    stress_max: float  # max von Mises over the strut voxels and load cases
    volume: float  # strut volume outside the keep-ins (mesh)
    warnings: list[str] = field(default_factory=list)
    mode: str = "layout"
    target_volume: float = 0.0
    simp_volume: float = 0.0  # sum(rho) * h^3 over the free cells
    simp_compliance: list[float] = field(default_factory=list)  # penalized rho, per case
    voxel_volume: float = 0.0  # strut voxels in free cells * h^3 (what the FE check sees)
    lp_volume: float | None = None  # optimal LP volume at sigma_allow (layout only)
    watertight: bool = False
    n_bodies: int = 0
    node_kind: np.ndarray | None = None  # (n,) int8, KIND_CODE
    timings: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict:
        ratio = None
        if self.compliance and self.simp_compliance:
            ratio = float(sum(self.compliance) / max(sum(self.simp_compliance), 1e-300))
        return {
            "mode": self.mode,
            "n_nodes": len(self.nodes),
            "n_bars": len(self.bars),
            "radius_min": float(self.bars[:, 2].min()) if len(self.bars) else 0.0,
            "radius_max": float(self.bars[:, 2].max()) if len(self.bars) else 0.0,
            "volume": float(self.volume),
            "target_volume": float(self.target_volume),
            "voxel_volume": float(self.voxel_volume),
            "simp_volume": float(self.simp_volume),
            "lp_volume": self.lp_volume,
            "compliance": [float(c) for c in self.compliance],
            "simp_compliance": [float(c) for c in self.simp_compliance],
            "compliance_ratio": ratio,
            "stress_max": float(self.stress_max),
            "watertight": bool(self.watertight),
            "n_bodies": int(self.n_bodies),
            "triangles": len(self.mesh.faces),
            "warnings": list(self.warnings),
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
        }

    def to_json(self) -> dict:
        """summary + nodes + bars (what `struts.json` holds)."""
        kinds = {v: k for k, v in KIND_CODE.items()}
        kind = self.node_kind if self.node_kind is not None else np.zeros(len(self.nodes))
        return {
            **self.summary(),
            "nodes": [
                {"xyz": [float(v) for v in p], "kind": kinds[int(k)]}
                for p, k in zip(self.nodes, kind, strict=True)
            ],
            "bars": [{"i": int(b[0]), "j": int(b[1]), "radius": float(b[2])} for b in self.bars],
        }


# ---- geometry helpers ----------------------------------------------------------------------------


def voxel_surface(mask: np.ndarray, grid: Grid) -> trimesh.Trimesh:
    """Closed surface of the voxel cells of `mask` (exact cube faces, outward winding)."""
    pad = np.pad(np.asarray(mask, dtype=bool), 1)
    tris = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        eb, ec = np.eye(3, dtype=np.int64)[b], np.eye(3, dtype=np.int64)[c]
        for sign in (1, -1):
            nb = np.roll(pad, -sign, axis=a)
            cells = np.argwhere(pad & ~nb)  # padded cell index; neighbour across the face is empty
            if not len(cells):
                continue
            p = cells.copy()
            if sign > 0:
                p[:, a] += 1
            q = np.stack([p, p + eb, p + eb + ec, p + ec], axis=1)  # e_b x e_c = e_a
            if sign < 0:
                q = q[:, ::-1]
            tris += [q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
    if not tris:
        return trimesh.Trimesh()
    corners = np.concatenate(tris).reshape(-1, 3)
    verts, inv = np.unique(corners, axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    xyz = np.asarray(grid.origin, dtype=np.float64) + grid.h * (verts - 1).astype(np.float64)
    return trimesh.Trimesh(xyz, faces, process=False)


def _cells_of(points: np.ndarray, grid: Grid, slack: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(cell index (n,3) clipped to the grid, flag: within `slack` cells of the grid)."""
    idx = np.floor((points - np.asarray(grid.origin)) / grid.h).astype(np.int64)
    shape = np.asarray(grid.shape)
    inside = np.all((idx >= -slack) & (idx < shape + slack), axis=1)
    return np.clip(idx, 0, shape - 1), inside


def _centers(ijk: np.ndarray, grid: Grid) -> np.ndarray:
    return np.asarray(grid.origin, dtype=np.float64) + grid.h * (ijk + 0.5)


def cluster_points(xyz: np.ndarray, cell: float, max_clusters: int = MAX_CLUSTERS) -> list:
    """Grid clustering of points: index arrays, at most `max_clusters` (the cell grows)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return []
    cell = max(float(cell), 1e-12)
    while True:
        key = np.floor((xyz - xyz.min(0)) / cell).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        n = int(inv.max()) + 1
        if n <= max_clusters:
            return [np.flatnonzero(inv == k) for k in range(n)]
        cell *= 1.5


def _bin_sample(ijk: np.ndarray, grid: Grid, spacing: float, mask: np.ndarray) -> np.ndarray:
    """One point per `spacing` cube of the cells `ijk`: the cells' centroid when it lies in
    `mask`, else the centre of the cell nearest to it."""
    if len(ijk) == 0:
        return np.zeros((0, 3))
    xyz = _centers(ijk, grid)
    key = np.floor((xyz - xyz.min(0)) / spacing).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    cen = np.stack([np.bincount(inv, xyz[:, a], n) for a in range(3)], 1) / cnt[:, None]
    cells, inside = _cells_of(cen, grid)
    ok = inside & mask[tuple(cells.T)]
    out = cen.copy()
    if not ok.all():
        d = np.linalg.norm(xyz - cen[inv], axis=1)
        order = np.lexsort((d, inv))
        first = order[np.searchsorted(inv[order], np.arange(n))]
        out[~ok] = xyz[first][~ok]
    return out


def segments_inside(
    p0: np.ndarray, p1: np.ndarray, mask: np.ndarray, grid: Grid, step: float | None = None
) -> np.ndarray:
    """bool (m,): every sample along p0[i]->p1[i] (spacing `step`, default h/2) lies in a
    `mask` cell."""
    step = 0.5 * grid.h if step is None else float(step)
    m = len(p0)
    out = np.ones(m, dtype=bool)
    if m == 0:
        return out
    d = p1 - p0
    n = np.maximum(np.ceil(np.linalg.norm(d, axis=1) / step).astype(np.int64) + 1, 2)
    start = 0
    while start < m:  # chunks of <= ~4M samples
        cum = np.cumsum(n[start:])
        stop = start + max(1, int(np.searchsorted(cum, 4_000_000)))
        nn = n[start:stop]
        seg = np.repeat(np.arange(start, stop), nn)
        first = np.repeat(np.cumsum(nn) - nn, nn)
        t = (np.arange(seg.size) - first) / (nn.repeat(nn) - 1)
        pts = p0[seg] + t[:, None] * d[seg]
        cells, inside = _cells_of(pts, grid, slack=1)
        good = inside & mask[tuple(cells.T)]
        bad = np.bincount(seg - start, weights=~good, minlength=stop - start)
        out[start:stop] = bad == 0
        start = stop
    return out


def drop_overlapping(nodes: np.ndarray, bars: np.ndarray, tol: float) -> np.ndarray:
    """bool keep-mask: a bar passing within `tol` of another node (strictly between its ends) is
    the sum of shorter bars and is dropped (standard ground-structure reduction)."""
    keep = np.ones(len(bars), dtype=bool)
    n = len(nodes)
    chunk = max(1, int(1e6 // max(n, 1)))  # (chunk, n, 3) temporaries of ~24 MB
    for lo in range(0, len(bars), chunk):
        b = bars[lo : lo + chunk]
        a = nodes[b[:, 0]]
        d = nodes[b[:, 1]] - a
        L2 = np.einsum("ij,ij->i", d, d)
        rel = nodes[None, :, :] - a[:, None, :]
        t = np.einsum("knj,kj->kn", rel, d) / L2[:, None]
        perp = rel - t[..., None] * d[:, None, :]
        dist2 = np.einsum("knj,knj->kn", perp, perp)
        hit = (t > 1e-6) & (t < 1 - 1e-6) & (dist2 < tol * tol)
        keep[lo : lo + chunk] = ~hit.any(axis=1)
    return keep


# ---- layout LP -----------------------------------------------------------------------------------


def equilibrium_matrix(nodes: np.ndarray, bars: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    """(B (3n, m), lengths): B q = f, tension positive. Column of bar (i, j) with unit vector
    u = (x_j - x_i)/l holds -u at node i and +u at node j."""
    n, m = len(nodes), len(bars)
    if m == 0:
        return sp.csr_matrix((3 * n, 0)), np.zeros(0)
    d = nodes[bars[:, 1]] - nodes[bars[:, 0]]
    L = np.linalg.norm(d, axis=1)
    u = d / L[:, None]
    rows = np.concatenate(
        [
            (3 * bars[:, 0, None] + np.arange(3)).ravel(),
            (3 * bars[:, 1, None] + np.arange(3)).ravel(),
        ]
    )
    cols = np.concatenate([np.repeat(np.arange(m), 3)] * 2)
    vals = np.concatenate([-u.ravel(), u.ravel()])
    return sp.csr_matrix((vals, (rows, cols)), shape=(3 * n, m)), L


@dataclass
class LayoutSolution:
    areas: np.ndarray  # (m,)
    forces: np.ndarray  # (K, m) bar forces, tension positive
    rigid_forces: np.ndarray  # (K, r)
    volume: float  # sum(l a)
    lengths: np.ndarray


def solve_layout(
    nodes: np.ndarray,
    bars: np.ndarray,
    fixed: np.ndarray,
    loads: np.ndarray,
    sigma: float,
    rigid: np.ndarray | None = None,
) -> LayoutSolution:
    """Plastic minimum-volume truss, multiple load cases (HiGHS LP).

    min sum l_i a_i  s.t.  B q_k + B_r r_k = f_k (free DOFs), |q_ik| <= sigma a_i, a >= 0.
    fixed (n, 3) bool: supported DOFs; loads (K, n, 3); rigid (r, 2): zero-cost links of
    unbounded strength (keep-in bodies). ValueError if no layout carries the loads.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    bars = np.asarray(bars, dtype=np.int64).reshape(-1, 2)
    loads = np.asarray(loads, dtype=np.float64).reshape(-1, len(nodes), 3)
    rigid = np.zeros((0, 2), dtype=np.int64) if rigid is None else np.asarray(rigid).reshape(-1, 2)
    K, m, r = loads.shape[0], len(bars), len(rigid)
    B, L = equilibrium_matrix(nodes, bars)
    Br, _ = equilibrium_matrix(nodes, rigid)
    free = ~np.asarray(fixed, dtype=bool).ravel()
    A = sp.hstack([B, Br]).tocsr()[free]
    F = loads.reshape(K, -1)[:, free]
    used = np.asarray(abs(A).sum(axis=1)).ravel() > 0
    if np.any(np.abs(F[:, ~used]) > 0):
        raise ValueError("a loaded node has no candidate bar: no strut layout can carry the loads")
    A, F = A[used], F[:, used]
    nr = A.shape[0]
    mq = m + r
    if K == 1:
        return _solve_single(A, F[0], L, m, r, sigma)
    # variables: a (m) | per case: q (m), rigid (r)
    nv = m + K * mq
    cost = np.concatenate([L, np.zeros(K * mq)])
    A_eq = sp.hstack([sp.csr_matrix((K * nr, m)), sp.block_diag([A] * K, format="csr")]).tocsr()
    b_eq = F.ravel()
    eye = sp.identity(m, format="csr")
    sel = sp.hstack([eye, sp.csr_matrix((m, r))])  # q part of one case block
    ub_rows = []
    for k in range(K):
        place = [sp.csr_matrix((m, mq))] * K
        place[k] = sel
        blk = sp.hstack(place)
        ub_rows += [sp.hstack([-sigma * eye, blk]), sp.hstack([-sigma * eye, -blk])]
    A_ub = sp.vstack(ub_rows).tocsr() if m else None
    b_ub = np.zeros(2 * K * m) if m else None
    bounds = np.zeros((nv, 2))
    bounds[:, 1] = np.inf
    bounds[m:, 0] = -np.inf
    x = _linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    blocks = x[m:].reshape(K, mq)
    return LayoutSolution(
        areas=np.maximum(x[:m], 0.0),
        forces=blocks[:, :m],
        rigid_forces=blocks[:, m:],
        volume=float(cost @ x),
        lengths=L,
    )


def _linprog(cost: np.ndarray, **kw) -> np.ndarray:
    # interior point + crossover: several times faster than dual simplex on ground structures
    res = linprog(cost, method="highs-ipm", **kw)
    if res.status != 0 or res.x is None:
        raise ValueError(f"no strut layout carries the loads (LP: {res.message})")
    return res.x


def _solve_single(
    A: sp.csr_matrix, f: np.ndarray, L: np.ndarray, m: int, r: int, sigma: float
) -> LayoutSolution:
    """One load case: q = q+ - q-, a = (q+ + q-)/sigma; no inequality rows."""
    cost = np.concatenate([L / sigma, L / sigma, np.zeros(r)])
    A_eq = sp.hstack([A[:, :m], -A[:, :m], A[:, m:]]).tocsr()
    bounds = np.zeros((2 * m + r, 2))
    bounds[:, 1] = np.inf
    bounds[2 * m :, 0] = -np.inf
    x = _linprog(cost, A_eq=A_eq, b_eq=f, bounds=bounds)
    qp, qm = x[:m], x[m : 2 * m]
    return LayoutSolution(
        areas=(qp + qm) / sigma,
        forces=(qp - qm)[None],
        rigid_forces=x[2 * m :][None],
        volume=float(cost @ x),
        lengths=L,
    )


# ---- boundary nodes ------------------------------------------------------------------------------


class _NodeSet:
    def __init__(self, grid: Grid, n_cases: int):
        self.grid, self.n_cases = grid, n_cases
        self.xyz: list[np.ndarray] = []
        self.kind: list[int] = []
        self.fixed: list[np.ndarray] = []
        self.load: list[np.ndarray] = []
        self.pad: list[float] = []  # sphere radius covering the cluster's grid nodes
        self._index: dict[tuple, int] = {}

    def add(self, p: np.ndarray, kind: NodeKind, pad: float = 0.0) -> int:
        key = tuple(np.round(np.asarray(p) / (1e-6 * self.grid.h)).astype(np.int64))
        i = self._index.get(key)
        if i is None:
            i = self._index[key] = len(self.xyz)
            self.xyz.append(np.asarray(p, dtype=np.float64))
            self.kind.append(KIND_CODE[kind])
            self.fixed.append(np.zeros(3, dtype=bool))
            self.load.append(np.zeros((self.n_cases, 3)))
            self.pad.append(0.0)
        elif self.kind[i] == KIND_CODE["sample"]:
            self.kind[i] = KIND_CODE[kind]
        self.pad[i] = max(self.pad[i], pad)
        return i

    def arrays(self):
        n = len(self.xyz)
        return (
            np.asarray(self.xyz).reshape(n, 3),
            np.asarray(self.kind, dtype=np.int8),
            np.asarray(self.fixed).reshape(n, 3),
            np.stack(self.load, axis=1) if n else np.zeros((self.n_cases, 0, 3)),
            np.asarray(self.pad),
        )


def _node_xyz(grid: Grid, ids: np.ndarray) -> np.ndarray:
    ijk = np.stack(np.unravel_index(np.asarray(ids, dtype=np.int64), grid.node_shape), 1)
    return np.asarray(grid.origin, dtype=np.float64) + grid.h * ijk


def _boundary_nodes(problem: Problem, spacing: float, ns: _NodeSet) -> None:
    """Load and support cluster representatives (grid nodes nearest the cluster centroids)."""
    grid = problem.grid
    cell = 3.0 * grid.h
    for sp_ in problem.supports:
        xyz = _node_xyz(grid, sp_.nodes)
        for c in cluster_points(xyz, cell):
            pts = xyz[c]
            rep = pts[np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1))]
            i = ns.add(rep, "support")  # a bar end anywhere on a support holds: no pad
            ns.fixed[i] |= np.asarray(sp_.fix, dtype=bool)
    for ld in problem.loads:
        xyz = _node_xyz(grid, ld.nodes)
        f = np.asarray(ld.force, dtype=np.float64)
        for c in cluster_points(xyz, cell):
            pts = xyz[c]
            rep = pts[np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1))]
            pad = float(np.linalg.norm(pts - rep, axis=1).max()) + 0.5 * grid.h
            i = ns.add(rep, "load", pad)
            ns.load[i][ld.case] += f * (len(c) / len(xyz))


def _keep_bodies(problem: Problem) -> tuple[np.ndarray, int]:
    keep = np.asarray(problem.active, dtype=bool) & (problem.passive == 1)
    return ndimage.label(keep, structure=np.ones((3, 3, 3), dtype=bool))


def _attachment_nodes(problem: Problem, spacing: float, ns: _NodeSet) -> None:
    """Points on the boundary of each keep-in body facing free design cells."""
    labels, nb = _keep_bodies(problem)
    if nb == 0:
        return
    near_free = ndimage.binary_dilation(
        problem.free, structure=ndimage.generate_binary_structure(3, 1)
    )
    boundary = (labels > 0) & near_free
    for b in range(1, nb + 1):
        ijk = np.argwhere(boundary & (labels == b))
        for p in _bin_sample(ijk, problem.grid, spacing, labels == b):
            ns.add(p, "attach")


def _rigid_links(problem: Problem, nodes: np.ndarray) -> np.ndarray:
    """Zero-cost links tying together every LP node within 1.5 h of the same keep-in body."""
    labels, nb = _keep_bodies(problem)
    if nb == 0 or len(nodes) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    dist, idx = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    cells, inside = _cells_of(nodes, problem.grid)
    c = tuple(cells.T)
    near = inside & (dist[c] <= 1.5)
    body = labels[tuple(idx[:, c[0], c[1], c[2]])]
    body = np.where(near, body, 0)
    links: set[tuple[int, int]] = set()
    from scipy.sparse.csgraph import minimum_spanning_tree

    for b in range(1, nb + 1):
        members = np.flatnonzero(body == b)
        if len(members) < 2:
            continue
        pts = nodes[members]
        k = min(9, len(members))
        _, nn = cKDTree(pts).query(pts, k=k)
        for row, nbrs in enumerate(np.atleast_2d(nn)):
            for j in nbrs[1:]:
                a, bb = sorted((int(members[row]), int(members[j])))
                links.add((a, bb))
        dmat = np.linalg.norm(pts[:, None] - pts[None], axis=2) if len(pts) <= 2000 else None
        if dmat is not None:
            mst = minimum_spanning_tree(sp.csr_matrix(dmat + 1e-12 * (dmat == 0))).tocoo()
            for a, bb in zip(mst.row, mst.col, strict=True):
                links.add(tuple(sorted((int(members[a]), int(members[bb])))))
    return np.asarray(sorted(links), dtype=np.int64).reshape(-1, 2)


def _layout(
    problem: Problem, rho: np.ndarray, params: StrutParams, spacing: float, warnings: list[str]
):
    grid, h = problem.grid, problem.grid.h
    active = np.asarray(problem.active, dtype=bool)
    keepout = active & (problem.passive == -1)
    domain = active & ~keepout
    # one voxel of tolerance, diagonals included: grid nodes on edges and corners count as inside
    allowed = ndimage.binary_dilation(domain, np.ones((3, 3, 3), dtype=bool))
    allowed &= ~(problem.passive == -1)

    def sample_mask(kind: str) -> np.ndarray:
        if kind == "active":
            return problem.free.copy()
        return problem.free & (rho >= params.rho_threshold)

    def build(kind: str, s: float):
        ns = _NodeSet(grid, problem.n_cases)
        _boundary_nodes(problem, s, ns)
        _attachment_nodes(problem, s, ns)
        n_special = len(ns.xyz)
        mask = sample_mask(kind)
        spacing_s = s
        for _ in range(12):
            pts = _bin_sample(np.argwhere(mask), grid, spacing_s, mask)
            if n_special + len(pts) <= params.max_nodes or len(pts) == 0:
                break
            spacing_s *= ((n_special + len(pts)) / params.max_nodes) ** (1 / 3) * 1.05
        if n_special:
            special = np.asarray(ns.xyz)
            d, _ = cKDTree(special).query(pts) if len(pts) else (np.zeros(0), None)
            pts = pts[d > 0.5 * spacing_s]
        for p in pts:
            ns.add(p, "sample")
        return ns

    lo, hi = np.argwhere(domain).min(0), np.argwhere(domain).max(0) + 1
    diag = float(np.linalg.norm((hi - lo) * h))
    max_len = params.max_bar_length or 0.4 * diag

    def attempt(kind: str):
        ns = build(kind, spacing)
        nodes, kinds, fixed, loads, pad = ns.arrays()
        pairs = cKDTree(nodes).query_pairs(max_len, output_type="ndarray").astype(np.int64)
        pairs = pairs[segments_inside(nodes[pairs[:, 0]], nodes[pairs[:, 1]], allowed, grid)]
        pairs = pairs[drop_overlapping(nodes, pairs, 0.2 * h)]
        if len(pairs) > params.max_candidates:
            L = np.linalg.norm(nodes[pairs[:, 1]] - nodes[pairs[:, 0]], axis=1)
            pairs = pairs[np.argsort(L, kind="stable")[: params.max_candidates]]
            warnings.append(f"candidate bars capped at the {params.max_candidates} shortest")
        rigid = _rigid_links(problem, nodes)
        lp = _LP(nodes, fixed, loads, params.sigma_allow, rigid)
        pairs, sol = lp.prune(pairs, lp.solve(pairs), params.prune)
        return nodes, kinds, pad, pairs, sol, lp

    try:
        return attempt(params.sample)
    except ValueError as exc:
        if params.sample == "active":
            raise
        warnings.append(f"layout on the SIMP solid failed ({exc}); retried on the whole domain")
        return attempt("active")


@dataclass
class _LP:
    nodes: np.ndarray
    fixed: np.ndarray
    loads: np.ndarray
    sigma: float
    rigid: np.ndarray

    def solve(self, bars: np.ndarray) -> LayoutSolution:
        return solve_layout(self.nodes, bars, self.fixed, self.loads, self.sigma, self.rigid)

    def prune(
        self, bars: np.ndarray, sol: LayoutSolution, frac: float, rounds: int = 4
    ) -> tuple[np.ndarray, LayoutSolution]:
        """Drop bars thinner than `frac` x the largest and re-solve on the survivors (fewer,
        cleaner members); stops when nothing changes or the reduced LP is infeasible."""
        for _ in range(rounds):
            if not frac > 0 or not len(bars):
                break
            keep = sol.areas > frac * sol.areas.max()
            if keep.all():
                break
            try:
                sol2 = self.solve(bars[keep])
            except ValueError:
                break
            bars, sol = bars[keep], sol2
        return bars, sol


def _simplify(
    nodes: np.ndarray, kinds: np.ndarray, bars: np.ndarray, areas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop tiny bars, merge collinear chains through unloaded degree-2 nodes, drop unused nodes.

    Returns (used node ids, bars renumbered to them, areas)."""
    if len(bars) == 0 or not areas.max() > 0:
        return np.zeros(0, dtype=np.int64), bars[:0], areas[:0]
    keep = areas > AREA_DROP * areas.max()
    bars, areas = bars[keep].copy(), areas[keep].copy()
    changed = True
    while changed:
        changed = False
        deg = np.bincount(bars.ravel(), minlength=len(nodes))
        for k in np.flatnonzero((deg == 2) & (kinds == KIND_CODE["sample"])):
            at = np.flatnonzero((bars == k).any(axis=1))
            if len(at) != 2:
                continue
            e1, e2 = bars[at[0]], bars[at[1]]
            i = e1[0] if e1[1] == k else e1[1]
            j = e2[0] if e2[1] == k else e2[1]
            if i == j:
                continue
            u, v = nodes[k] - nodes[i], nodes[j] - nodes[k]
            if u @ v < COLLINEAR_COS * np.linalg.norm(u) * np.linalg.norm(v):
                continue
            bars[at[0]] = (min(i, j), max(i, j))
            areas[at[0]] = max(areas[at[0]], areas[at[1]])
            bars = np.delete(bars, at[1], axis=0)
            areas = np.delete(areas, at[1])
            changed = True
            break
    # merge duplicates created by chain merging
    bars = np.sort(bars, axis=1)
    uniq, inv = np.unique(bars, axis=0, return_inverse=True)
    areas = np.bincount(inv.ravel(), weights=areas, minlength=len(uniq))
    used = np.unique(uniq)
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return used, remap[uniq], areas


# ---- skeleton ------------------------------------------------------------------------------------

_OFFSETS = np.array(
    [
        (i, j, k)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if (i, j, k) != (0, 0, 0)
    ]
)


@dataclass
class SkeletonGraph:
    nodes: np.ndarray  # (n, 3) world
    bars: np.ndarray  # (m, 2)
    radii: np.ndarray  # (m,)
    branches: list[dict]  # {"a", "b", "length", "radius", "points"}


def _rdp(points: np.ndarray, tol: float) -> list[int]:
    """Ramer-Douglas-Peucker: indices of the kept polyline vertices."""
    if len(points) <= 2:
        return list(range(len(points)))
    a, b = points[0], points[-1]
    d = b - a
    L = np.linalg.norm(d)
    rel = points - a
    if L < 1e-12:
        dist = np.linalg.norm(rel, axis=1)
    else:
        dist = np.linalg.norm(np.cross(rel, d / L), axis=1)
    k = int(np.argmax(dist))
    if dist[k] <= tol:
        return [0, len(points) - 1]
    left = _rdp(points[: k + 1], tol)
    right = _rdp(points[k:], tol)
    return left[:-1] + [k + i for i in right]


def _anchor(skel: np.ndarray, mask: np.ndarray, anchors: np.ndarray, h: float, origin) -> None:
    """Draw a voxel line from each anchor (world point) to the nearest skeleton voxel when the
    line stays inside the (one-voxel dilated) solid: loads and supports on the solid's skin
    become skeleton endpoints instead of lying beside a medial axis that never reaches them."""
    ijk = np.argwhere(skel)
    tree = cKDTree(ijk + 0.5)
    shape = np.asarray(mask.shape)
    allowed = ndimage.binary_dilation(mask)
    for p in anchors:
        a = (p - origin) / h  # continuous cell coordinates
        _, j = tree.query(a)
        b = ijk[j] + 0.5
        n = int(np.ceil(2 * np.linalg.norm(b - a))) + 2
        pts = a[None] + np.linspace(0, 1, n)[:, None] * (b - a)[None]
        cells = np.clip(np.floor(pts).astype(np.int64), 0, shape - 1)
        if allowed[tuple(cells.T)].all():
            skel[tuple(cells.T)] = True


def _drop_short_cycles(adj: list[dict[int, float]], n: int, min_cycle: float) -> list[dict]:
    """Spanning forest of the voxel graph plus the non-tree edges closing cycles longer than
    `min_cycle`: removes voxel triangles and the ladders of even-thickness skeletons, keeps
    real loops of the structure."""
    from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree

    rows = [a for a in range(n) for b in adj[a] if b > a]
    cols = [b for a in range(n) for b in adj[a] if b > a]
    wts = [adj[a][b] for a in range(n) for b in adj[a] if b > a]
    G = sp.csr_matrix((wts, (rows, cols)), shape=(n, n))
    T = minimum_spanning_tree(G)
    T = (T + T.T).tocsr()
    parent = -np.ones(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    dist = np.zeros(n)
    seen = np.zeros(n, dtype=bool)
    for root in range(n):
        if seen[root]:
            continue
        order, pred = breadth_first_order(T, root, directed=False)
        seen[order] = True
        for v in order[1:]:
            p = pred[v]
            parent[v] = p
            depth[v] = depth[p] + 1
            dist[v] = dist[p] + T[v, p]
    out: list[dict[int, float]] = [{} for _ in range(n)]
    Tc = T.tocoo()
    for a, b, w in zip(Tc.row, Tc.col, Tc.data, strict=True):
        out[int(a)][int(b)] = float(w)
    for a, b, w in zip(rows, cols, wts, strict=True):
        if b in out[a]:
            continue
        u, v = a, b
        while depth[u] > depth[v]:
            u = parent[u]
        while depth[v] > depth[u]:
            v = parent[v]
        while u != v and u >= 0 and v >= 0:
            u, v = parent[u], parent[v]
        if u < 0 or v < 0:
            continue
        if dist[a] + dist[b] - 2 * dist[u] + w > min_cycle:
            out[a][b] = out[b][a] = w
    return out


def skeleton_graph(
    mask: np.ndarray,
    h: float = 1.0,
    origin=(0.0, 0.0, 0.0),
    protect: np.ndarray | None = None,
    prune_factor: float = 2.0,
    anchors: np.ndarray | None = None,
) -> SkeletonGraph:
    """Medial-axis graph of a voxel solid: branches between junctions/endpoints, spurs shorter
    than `prune_factor` x the distance-transform value at their junction pruned (unless their
    tip lies in `protect`), radius per branch = median distance transform - h/2."""
    from skimage.morphology import skeletonize

    mask = np.asarray(mask, dtype=bool)
    skel = skeletonize(mask).astype(bool)
    dt = ndimage.distance_transform_edt(mask) * h
    if anchors is not None and len(anchors) and skel.any():
        _anchor(skel, mask, np.asarray(anchors), h, np.asarray(origin, dtype=np.float64))
    ijk = np.argwhere(skel)
    n = len(ijk)
    origin = np.asarray(origin, dtype=np.float64)
    xyz = origin + h * (ijk + 0.5)
    empty = SkeletonGraph(np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int64), np.zeros(0), [])
    if n < 2:
        return empty
    index = -np.ones(mask.shape, dtype=np.int64)
    index[tuple(ijk.T)] = np.arange(n)
    pad = np.pad(index, 1, constant_values=-1)
    nb = np.stack([pad[tuple((ijk + 1 + o).T)] for o in _OFFSETS], axis=1)  # (n, 26)
    w = np.linalg.norm(_OFFSETS, axis=1)
    adj: list[dict[int, float]] = [
        {int(j): float(w[c]) for c, j in enumerate(row) if j >= 0} for row in nb
    ]
    adj = _drop_short_cycles(adj, n, max(8.0, 2 * np.pi * float(np.median(dt[skel])) / h + 2))
    deg = np.array([len(a) for a in adj])
    # junction clusters (degree >= 3, adjacent ones merged) and endpoints are graph keys
    key = -np.ones(n, dtype=np.int64)
    junction = deg >= 3
    n_keys = 0
    for s in np.flatnonzero(junction):
        if key[s] >= 0:
            continue
        stack = [s]
        key[s] = n_keys
        while stack:
            v = stack.pop()
            for u in adj[v]:
                if junction[u] and key[u] < 0:
                    key[u] = n_keys
                    stack.append(u)
        n_keys += 1
    for s in np.flatnonzero(deg == 1):
        key[s] = n_keys
        n_keys += 1

    branches: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def trace(start: int, nxt: int) -> None:
        path = [start, nxt]
        prev, cur = start, nxt
        while key[cur] < 0:
            step = [u for u in adj[cur] if u != prev]
            if not step:
                break
            prev, cur = cur, step[0]
            path.append(cur)
            if cur == start:
                break
        e = (min(path[0], path[1]), max(path[0], path[1]))
        e2 = (min(path[-1], path[-2]), max(path[-1], path[-2]))
        if e in seen or e2 in seen:
            return
        seen.add(e)
        seen.add(e2)
        branches.append(
            {"a": int(key[start]), "b": int(key[cur]) if key[cur] >= 0 else -1, "path": path}
        )

    for v in np.flatnonzero(key >= 0):
        for u in adj[v]:
            if key[u] == key[v]:
                continue
            trace(int(v), int(u))
    # loops without any key voxel
    on_branch = np.zeros(n, dtype=bool)
    for br in branches:
        on_branch[br["path"]] = True
    for v in np.flatnonzero((deg == 2) & ~on_branch & (key < 0)):
        if on_branch[v]:
            continue
        key[v] = n_keys
        n_keys += 1
        u = next(iter(adj[v]))
        trace(int(v), int(u))
        for br in branches[-1:]:
            on_branch[br["path"]] = True

    key_xyz = np.zeros((n_keys, 3))
    key_cnt = np.zeros(n_keys)
    key_dt = np.zeros(n_keys)
    for v in np.flatnonzero(key >= 0):
        key_xyz[key[v]] += xyz[v]
        key_cnt[key[v]] += 1
        key_dt[key[v]] = max(key_dt[key[v]], dt[tuple(ijk[v])])
    key_xyz /= np.maximum(key_cnt, 1)[:, None]
    for br in branches:
        if br["b"] < 0:
            br["b"] = br["a"]
        pts = xyz[br["path"]]
        br["length"] = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    tip_protected = np.zeros(n_keys, dtype=bool)
    if protect is not None:
        for v in np.flatnonzero(deg == 1):
            tip_protected[key[v]] = bool(protect[tuple(ijk[v])])

    def degree() -> np.ndarray:
        d = np.zeros(n_keys, dtype=np.int64)
        for br in branches:
            d[br["a"]] += 1
            d[br["b"]] += 1
        return d

    parent = np.arange(n_keys)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _ in range(8):
        changed = False
        d = degree()
        # contract short internal branches between two junctions
        for br in list(branches):
            a, b = br["a"], br["b"]
            if a != b and d[a] >= 3 and d[b] >= 3 and br["length"] < max(key_dt[a], key_dt[b]):
                parent[find(b)] = find(a)
                branches.remove(br)
                changed = True
        for br in branches:
            br["a"], br["b"] = find(br["a"]), find(br["b"])
        # prune spurs
        d = degree()
        for br in list(branches):
            a, b = br["a"], br["b"]
            for tip, base in ((a, b), (b, a)):
                spur = d[tip] == 1 and d[base] >= 3 and not tip_protected[tip]
                if spur and br["length"] < prune_factor * key_dt[base]:
                    branches.remove(br)
                    d[tip] -= 1
                    d[base] -= 1
                    changed = True
                    break
        # merge branches through degree-2 keys
        merged = True
        while merged:
            merged = False
            d = degree()
            for k in np.flatnonzero(d == 2):
                at = [br for br in branches if k in (br["a"], br["b"])]
                if len(at) != 2:
                    continue
                b1, b2 = at
                p1 = b1["path"] if b1["b"] == k else b1["path"][::-1]
                p2 = b2["path"] if b2["a"] == k else b2["path"][::-1]
                o1 = b1["a"] if b1["b"] == k else b1["b"]
                o2 = b2["b"] if b2["a"] == k else b2["a"]
                branches.remove(b1)
                branches.remove(b2)
                length = b1["length"] + b2["length"]
                branches.append({"a": o1, "b": o2, "path": p1 + p2[1:], "length": length})
                merged = changed = True
                break
        if not changed:
            break

    used_keys = sorted({br["a"] for br in branches} | {br["b"] for br in branches})
    node_list = [key_xyz[k] for k in used_keys]
    node_of = {k: i for i, k in enumerate(used_keys)}
    bars, radii, out_br = [], [], []
    for br in branches:
        pts = xyz[br["path"]]
        pts = np.concatenate([[key_xyz[br["a"]]], pts[1:-1], [key_xyz[br["b"]]]])
        inner = br["path"][1:-1] or br["path"]
        r = float(np.median([dt[tuple(ijk[v])] for v in inner])) - 0.5 * h
        r = max(r, 0.5 * h)
        keep = _rdp(pts, max(h, 0.5 * r))
        idx = [node_of[br["a"]]]
        for kpt in keep[1:-1]:
            idx.append(len(node_list))
            node_list.append(pts[kpt])
        idx.append(node_of[br["b"]])
        for i, j in itertools.pairwise(idx):
            if i != j:
                bars.append((i, j))
                radii.append(r)
        out_br.append(
            {
                "a": node_of[br["a"]],
                "b": node_of[br["b"]],
                "length": br["length"],
                "radius": r,
                "points": pts,
            }
        )
    return SkeletonGraph(
        np.asarray(node_list).reshape(-1, 3),
        np.asarray(bars, dtype=np.int64).reshape(-1, 2),
        np.asarray(radii),
        out_br,
    )


def _skeleton(problem: Problem, rho: np.ndarray, spacing: float, warnings: list[str]):
    grid, h = problem.grid, problem.grid.h
    solid = problem.free & (rho >= 0.5)
    keep = np.asarray(problem.active, dtype=bool) & (problem.passive == 1)
    ns = _NodeSet(grid, problem.n_cases)
    _boundary_nodes(problem, spacing, ns)
    bnodes, bkinds, _, bloads, bpad = ns.arrays()
    protect = keep.copy()
    for ids in [ld.nodes for ld in problem.loads] + [sp_.nodes for sp_ in problem.supports]:
        ijk = np.stack(np.unravel_index(np.asarray(ids, dtype=np.int64), grid.node_shape), 1)
        for off in HEX8_OFFSETS:  # the cells around each boundary node
            c = ijk - off
            ok = np.all((c >= 0) & (c < np.asarray(grid.shape)), axis=1)
            protect[tuple(c[ok].T)] = True
    protect = ndimage.binary_dilation(protect, np.ones((3, 3, 3), dtype=bool), iterations=2)
    sk = skeleton_graph(solid, h, grid.origin, protect=protect, anchors=bnodes)
    if len(sk.bars) == 0:
        raise ValueError("the SIMP solid (rho >= 0.5) has no skeleton; use mode 'layout'")
    nodes = np.concatenate([sk.nodes, bnodes])
    kinds = np.concatenate([np.zeros(len(sk.nodes), dtype=np.int8), bkinds])
    pad = np.concatenate([np.zeros(len(sk.nodes)), bpad])
    bars, radii = [*map(tuple, sk.bars)], list(sk.radii)
    r_typ = float(np.median(sk.radii))
    n_sk = len(sk.nodes)
    deg = np.bincount(sk.bars.ravel(), minlength=n_sk)
    linked = np.zeros(len(bnodes), dtype=bool)
    if len(bnodes):
        # skeleton tips stop about a radius short of the boundary: tie them to the nearest
        # load / support point close by
        btree = cKDTree(bnodes)
        for t in np.flatnonzero(deg == 1):
            d, b = btree.query(sk.nodes[t])
            if d <= 3 * h + bpad[b]:
                bars.append((int(t), n_sk + int(b)))
                radii.append(r_typ)
                linked[b] = True
    tree = cKDTree(sk.nodes)
    loaded = np.abs(bloads).sum(axis=(0, 2)) > 0
    supp = bkinds == KIND_CODE["support"]
    need = np.flatnonzero(loaded & ~linked)
    if supp.any() and not (linked & supp).any():  # nothing reaches a support: the nearest one
        dist = tree.query(bnodes[supp])[0]
        need = np.append(need, np.flatnonzero(supp)[np.argmin(dist)])
    for b in need:
        _, j = tree.query(bnodes[b])
        bars.append((int(j), n_sk + int(b)))
        radii.append(r_typ)
    return nodes, kinds, pad, np.asarray(bars, dtype=np.int64), np.asarray(radii)


# ---- meshing -------------------------------------------------------------------------------------


def _manifold(mesh: trimesh.Trimesh):
    import manifold3d as m3

    m = mesh
    if m.volume < 0:
        m = m.copy()
        m.invert()
    out = m3.Manifold(
        mesh=m3.Mesh(
            vert_properties=np.asarray(m.vertices, dtype=np.float32),
            tri_verts=np.asarray(m.faces, dtype=np.uint32),
        )
    )
    return out


def _to_trimesh(man) -> trimesh.Trimesh:
    mm = man.to_mesh()
    # keep manifold's own vertex indexing: merging coincident vertices of touching parts would
    # create non-manifold edges
    return trimesh.Trimesh(
        np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False
    )


def strut_mesh(
    nodes: np.ndarray,
    bars: np.ndarray,
    radii: np.ndarray,
    pads: list[tuple[np.ndarray, float]],
    keep_meshes: list[trimesh.Trimesh],
    segments: int = 16,
) -> tuple[trimesh.Trimesh, list[str]]:
    """Union of capsules (bars), spheres (pads) and the keep-in bodies. Untrimmed."""
    import manifold3d as m3

    warnings: list[str] = []
    parts = []
    spheres: dict[float, object] = {}

    def sphere(r: float, at: np.ndarray):
        key = round(float(r), 9)
        if key not in spheres:
            spheres[key] = m3.Manifold.sphere(float(r), int(segments))
        return spheres[key].translate(tuple(float(v) for v in at))

    for (i, j), r in zip(bars, radii, strict=True):
        parts.append(m3.Manifold.batch_hull([sphere(r, nodes[i]), sphere(r, nodes[j])]))
    for c, r in pads:
        parts.append(sphere(r, c))
    for k, km in enumerate(keep_meshes):
        if km is None or not len(km.faces):
            continue
        man = _manifold(km)
        if man.status() != m3.Error.NoError or man.is_empty():
            warnings.append(f"keep-in {k} is not a closed manifold; not merged into the struts")
            continue
        parts.append(man)
    if not parts:
        return trimesh.Trimesh(), ["no struts"]
    return _to_trimesh(m3.Manifold.batch_boolean(parts, m3.OpType.Add)), warnings


def _drop_slivers(mesh: trimesh.Trimesh, min_volume: float) -> trimesh.Trimesh:
    """Remove the near-zero-volume shells booleans leave behind (float32 round-off)."""
    if not len(mesh.faces) or mesh.body_count <= 1:
        return mesh
    parts = mesh.split(only_watertight=False)
    keep = [b for b in parts if abs(b.volume) > min_volume]
    if len(keep) == len(parts) or not keep:
        return mesh
    return trimesh.util.concatenate(keep)


def _keep_volume(keep_meshes: list[trimesh.Trimesh]) -> float:
    import manifold3d as m3

    parts = []
    for km in keep_meshes:
        if km is not None and len(km.faces):
            man = _manifold(km)
            if man.status() == m3.Error.NoError:
                parts.append(man)
    if not parts:
        return 0.0
    return float(m3.Manifold.batch_boolean(parts, m3.OpType.Add).volume())


# ---- verification --------------------------------------------------------------------------------


def fe_check(
    problem: Problem, density: np.ndarray, penal: float = 1.0
) -> tuple[list[float], np.ndarray]:
    """(compliance per case, von Mises (n_cases, nel_active) of solid material) at E =
    Emin + density^penal (E0 - Emin)."""
    asm = Assembler(problem)
    E0 = float(problem.material.E)
    Emin = problem.material.emin_ratio * E0
    xe = np.asarray(density, dtype=np.float64).ravel()[asm.element_ids]
    K = asm.assemble(Emin + xe**penal * (E0 - Emin))
    solver = LinearSolver("auto", prolongators=asm.prolongators, ordering=asm.band_ordering)
    U, _ = solver.solve(K, asm.F_free)
    U = np.asarray(U).reshape(asm.F_free.shape)
    comp = [float(v) for v in np.sum(asm.F_free * U, axis=0)]
    vm = von_mises(asm.element_stress(U)).reshape(asm.n_cases, -1)
    return comp, vm


def _rmin_volume(nodes: np.ndarray, bars: np.ndarray, sol: LayoutSolution, rmin: float) -> float:
    keep = sol.areas > AREA_DROP * sol.areas.max()
    L = np.linalg.norm(nodes[bars[keep, 1]] - nodes[bars[keep, 0]], axis=1)
    return float(np.pi * rmin**2 * L.sum())


def _scale_radii(r0: np.ndarray, lengths: np.ndarray, rmin: float, target: float) -> float:
    """Factor s with sum pi max(s r0, rmin)^2 l = target (0 when even rmin overshoots)."""

    def vol(s: float) -> float:
        return float(np.sum(np.pi * np.maximum(s * r0, rmin) ** 2 * lengths))

    if vol(0.0) >= target:
        return 0.0
    hi = 1.0
    while vol(hi) < target:
        hi *= 2.0
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if vol(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def generate_struts(
    problem: Problem,
    rho: np.ndarray,
    design_mesh_world: trimesh.Trimesh | None = None,
    keep_meshes_world: list[trimesh.Trimesh] | None = None,
    params: StrutParams | None = None,
) -> StrutResult:
    """Strut structure from a SIMP result. `design_mesh_world` None -> the voxel surface of the
    active domain; `keep_meshes_world` None/empty -> the voxel surface of the passive-solid cells.
    ValueError when no layout can carry the loads."""
    params = params or StrutParams()
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    grid, h = problem.grid, problem.grid.h
    rho = np.asarray(rho, dtype=np.float64)
    if rho.shape != grid.shape:
        raise ValueError(f"rho shape {rho.shape} != grid shape {grid.shape}")
    if params.mode not in ("layout", "skeleton"):
        raise ValueError(f"unknown strut mode {params.mode!r}")
    if not params.sigma_allow > 0:
        raise ValueError("sigma_allow must be > 0")
    warnings: list[str] = []
    active = np.asarray(problem.active, dtype=bool)
    keep_mask = active & (problem.passive == 1)
    spacing = float(params.node_spacing or 4.0 * h)
    rmin = float(params.min_radius if params.min_radius is not None else max(1.0, 0.8 * h))
    simp_volume = float(rho[problem.free].sum()) * h**3
    target = simp_volume if params.target_volume is None else float(params.target_volume)

    lp_volume = None
    if params.mode == "layout":
        # bars at min_radius alone must leave room for sizing: coarsen the ground structure
        # (fewer nodes -> fewer members) unless the spacing was given
        tries = 1 if params.node_spacing or not target > 0 else 3
        best = None
        for _ in range(tries):
            w: list[str] = []
            try:
                best = (_layout(problem, rho, params, spacing, w), w)
            except ValueError:
                if best is None:
                    raise
                break  # a coarser ground structure no longer connects: keep the finer one
            nodes, kinds, pad, pairs, sol, _ = best[0]
            if _rmin_volume(nodes, pairs, sol, rmin) <= 0.8 * target:
                break
            spacing *= 1.4
        (nodes, kinds, pad, pairs, sol, _), w = best
        warnings += w
        lp_volume = sol.volume
        used, bars, areas = _simplify(nodes, kinds, pairs, sol.areas)
        nodes, kinds, pad = nodes[used], kinds[used], pad[used]
        r0 = np.sqrt(areas / np.pi)
    else:
        nodes, kinds, pad, bars, r0 = _skeleton(problem, rho, spacing, warnings)
    used = np.unique(bars)  # boundary points no bar reaches get no pad either
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    nodes, kinds, pad, bars = nodes[used], kinds[used], pad[used], remap[bars]
    timings["graph"] = time.perf_counter() - t0
    if len(bars) == 0:
        raise ValueError("no struts: the layout is empty")

    lengths = np.linalg.norm(nodes[bars[:, 1]] - nodes[bars[:, 0]], axis=1)
    if target > 0:
        s = _scale_radii(r0, lengths, rmin, target)
        if s == 0.0:
            need = float(np.pi * rmin**2 * lengths.sum())
            warnings.append(
                f"bars at min_radius {rmin:g} already need about {need:.4g} > target volume "
                f"{target:.4g}: every bar is at min_radius (lower min_radius or raise node_spacing)"
            )
    else:
        s = 1.0
    keeps = [m for m in (keep_meshes_world or []) if m is not None and len(m.faces)]
    if not keeps and keep_mask.any():
        keeps = [voxel_surface(keep_mask, grid)]
    design = design_mesh_world
    if design is None or not len(design.faces):
        design = voxel_surface(active & (problem.passive != -1), grid)
    keep_vol = _keep_volume(keeps)

    def make(scale: float):
        radii = np.maximum(scale * r0, rmin)
        node_r = np.zeros(len(nodes))
        np.maximum.at(node_r, bars[:, 0], radii)
        np.maximum.at(node_r, bars[:, 1], radii)
        pads = [(nodes[i], max(pad[i], node_r[i])) for i in np.flatnonzero(pad > 0)]
        raw, w = strut_mesh(nodes, bars, radii, pads, keeps, params.segments)
        trimmed, wt = trim_to_design(raw, design)
        return radii, _drop_slivers(trimmed, 1e-3 * h**3), w + wt

    t1 = time.perf_counter()
    radii, mesh, mw = make(s)
    vol = float(mesh.volume) - keep_vol if len(mesh.faces) else 0.0
    if target > 0 and vol > 0 and abs(vol / target - 1) > VOLUME_TOL and s > 0:
        s *= float(np.sqrt(target / vol))
        radii, mesh, mw = make(s)
        vol = float(mesh.volume) - keep_vol if len(mesh.faces) else 0.0
    warnings += mw
    if target > 0 and s > 0 and vol > 0 and abs(vol / target - 1) > 0.15:
        warnings.append(
            f"strut volume {vol:.4g} misses the target {target:.4g} (trimmed or merged)"
        )
    timings["mesh"] = time.perf_counter() - t1
    watertight = bool(len(mesh.faces) and mesh.is_watertight)
    n_bodies = int(mesh.body_count) if len(mesh.faces) else 0
    if not watertight:
        warnings.append("strut mesh is not watertight")
    if n_bodies > 1:
        warnings.append(f"strut mesh has {n_bodies} separate bodies")

    compliance: list[float] = []
    simp_compliance: list[float] = []
    stress_max = float("nan")
    voxel_volume = 0.0
    if params.verify:
        t2 = time.perf_counter()
        mask, _ = voxelize_mesh(mesh, grid) if len(mesh.faces) else (np.zeros(grid.shape, bool), [])
        mask = (mask & active & (problem.passive != -1)) | keep_mask
        voxel_volume = float((mask & problem.free).sum()) * h**3
        compliance, vm = fe_check(problem, mask.astype(np.float64), 1.0)
        solid_e = mask.ravel()[np.flatnonzero(active.ravel())]
        stress_max = float(vm[:, solid_e].max(initial=0.0))
        simp_compliance, _ = fe_check(problem, rho, params.penal)
        if sum(compliance) > 1e3 * max(sum(simp_compliance), 1e-300):
            warnings.append(
                "strut compliance is orders of magnitude above SIMP: a load is not connected"
            )
        timings["verify"] = time.perf_counter() - t2
    timings["total"] = time.perf_counter() - t0
    return StrutResult(
        mesh=mesh,
        nodes=nodes,
        bars=np.column_stack([bars.astype(np.float64), radii]) if len(bars) else np.zeros((0, 3)),
        compliance=compliance,
        stress_max=stress_max,
        volume=vol,
        warnings=warnings,
        mode=params.mode,
        target_volume=target,
        simp_volume=simp_volume,
        simp_compliance=simp_compliance,
        voxel_volume=voxel_volume,
        lp_volume=lp_volume,
        watertight=watertight,
        n_bodies=n_bodies,
        node_kind=kinds,
        timings=timings,
    )
