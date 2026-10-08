"""Mesh loading and voxelization onto the common element grid.

Occupancy rule: an element is occupied iff its center is inside the solid.

Contract note: `passive == -1` (forced void) is reserved by `problem.py` but NOT produced by
`build_domain` in v1. keep_out bodies remove elements outright (`active=False`, `passive=0`).
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from scipy import ndimage

from topop.core.problem import Grid

VoxelMethod = Literal["auto", "contains", "scanline", "vote", "surface_fill"]

_MAX_PAIRS = 2_000_000  # (triangle, sample) candidates processed per chunk
_MAX_SLAB = 16_000_000  # cells per cumulative-sum slab


def load_mesh(src: str | bytes | os.PathLike, file_type: str | None = None) -> trimesh.Trimesh:
    """Load a path or raw bytes (+ file_type) into one processed Trimesh. ValueError on failure."""
    if isinstance(src, bytes | bytearray | memoryview):
        if not file_type:
            raise ValueError("file_type is required when loading a mesh from bytes")
        if len(src) == 0:
            raise ValueError("empty mesh data")
        file_obj: str | io.BytesIO = io.BytesIO(bytes(src))
    else:
        path = Path(os.fspath(src))
        if not path.is_file():
            raise ValueError(f"mesh file not found: {path}")
        file_obj = str(path)
        file_type = file_type or path.suffix
    ftype = (file_type or "").lower().lstrip(".")
    if not ftype:
        raise ValueError("cannot determine mesh file type")
    try:
        loaded = trimesh.load(file_obj, file_type=ftype, process=True)
    except Exception as exc:  # trimesh raises many exception types for bad/unsupported input
        raise ValueError(f"could not load {ftype!r} mesh: {exc}") from exc
    if isinstance(loaded, trimesh.Scene):
        if not any(isinstance(g, trimesh.Trimesh) for g in loaded.geometry.values()):
            raise ValueError("file contains no triangle meshes")
        loaded = loaded.to_mesh()
    if not isinstance(loaded, trimesh.Trimesh):
        msg = f"file does not contain a triangle mesh (got {type(loaded).__name__})"
        raise ValueError(msg)  # noqa: TRY004 - bad input file, not a programming error
    mesh = loaded
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no (non-degenerate) faces")
    return mesh


def mesh_info(mesh: trimesh.Trimesh) -> dict:
    """`MeshInfo` fields except id/name. volume is None unless watertight and consistently wound."""
    closed = bool(mesh.is_watertight)
    volume = float(abs(mesh.volume)) if closed and mesh.is_winding_consistent else None
    return {
        "n_faces": len(mesh.faces),
        "n_vertices": len(mesh.vertices),
        "bbox": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "is_watertight": closed,
        "volume": volume,
    }


def transform_matrix(t16_colmajor: Sequence[float] | np.ndarray | None) -> np.ndarray:
    """16 floats in column-major (three.js) order -> row-major (4,4). None -> identity."""
    if t16_colmajor is None:
        return np.eye(4)
    t = np.asarray(t16_colmajor, dtype=np.float64).ravel()
    if t.size != 16 or not np.all(np.isfinite(t)):
        raise ValueError("transform must be 16 finite floats (column-major)")
    return t.reshape(4, 4).T.copy()


def apply_transform(
    mesh: trimesh.Trimesh, t16_colmajor: Sequence[float] | np.ndarray | None
) -> trimesh.Trimesh:
    out = mesh.copy()
    m = transform_matrix(t16_colmajor)
    if not np.allclose(m, np.eye(4)):
        out.apply_transform(m)
    return out


# ---- voxelization ------------------------------------------------------------------------------


def _expand(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(owner, local index) for `counts[i]` consecutive items per owner i."""
    owner = np.repeat(np.arange(counts.size), counts)
    local = np.arange(owner.size) - np.repeat(np.cumsum(counts) - counts, counts)
    return owner, local


def _chunks(counts: np.ndarray, limit: int) -> Iterator[tuple[int, int]]:
    """Consecutive index ranges whose summed counts stay around `limit`."""
    if counts.size == 0:
        return
    cid = (np.cumsum(counts) - 1) // limit
    cuts = np.concatenate([[0], np.flatnonzero(np.diff(cid)) + 1, [counts.size]])
    yield from zip(cuts[:-1].tolist(), cuts[1:].tolist(), strict=True)


def covered_samples(
    u: np.ndarray, v: np.ndarray, nu: int, nv: int, eps: float = 1e-7
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield chunks (tri, i, j) of samples (i+0.5, j+0.5) of an nu x nv lattice covered by the
    2D triangles with corners (u, v) (F,3). Superset: samples within `eps` of a triangle count.

    Works row by row (span per triangle and sample row), so long thin triangles cost their area,
    not their bounding box.
    """
    j0 = np.clip(np.ceil(v.min(1) - 0.5 - eps), 0, nv).astype(np.int64)
    j1 = np.clip(np.floor(v.max(1) - 0.5 + eps), -1, nv - 1).astype(np.int64)
    rows = (j1 - j0 + 1).clip(0)
    rows[(u.max(1) < 0.5 - eps) | (u.min(1) > nu - 0.5 + eps)] = 0
    nxt = np.array([1, 2, 0])
    for s, e in _chunks(rows, _MAX_PAIRS):
        f, k = _expand(rows[s:e])
        f += s
        j = j0[f] + k
        ua, va = u[f], v[f]
        ub, vb = ua[:, nxt], va[:, nxt]
        dv = vb - va
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((j + 0.5)[:, None] - va) / dv
        ok = (dv != 0) & (t >= -1e-9) & (t <= 1 + 1e-9)
        x = ua + np.clip(np.nan_to_num(t), 0.0, 1.0) * (ub - ua)
        xl = np.where(ok, x, np.inf).min(1)
        xr = np.where(ok, x, -np.inf).max(1)
        i0 = np.clip(np.ceil(xl - 0.5 - eps), 0, nu).astype(np.int64)
        i1 = np.clip(np.floor(xr - 0.5 + eps), -1, nu - 1).astype(np.int64)
        n = (i1 - i0 + 1).clip(0)
        for s2, e2 in _chunks(n, _MAX_PAIRS):
            g, kk = _expand(n[s2:e2])
            g += s2
            yield f[g], i0[g] + kk, j[g]


def _ray_hits(
    vertices: np.ndarray, faces: np.ndarray, grid: Grid, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crossings of rays cast along +`axis` through every element-center column.

    Returns (ia, ib, k, sign): column indices along the two other axes (cyclic order), index of
    the first element whose center lies past the crossing, and -sign(normal[axis]) (+1 entering).

    A ray exactly on a shared edge/vertex is resolved by a symbolic perturbation of the column
    position: each edge function is evaluated in a canonical (lexicographic) vertex order, so both
    triangles sharing an edge see bitwise-identical values and exactly one of them owns the ray.
    """
    a, b = (axis + 1) % 3, (axis + 2) % 3
    origin = np.asarray(grid.origin, dtype=np.float64)
    h = grid.h
    na, nb, nc = grid.shape[a], grid.shape[b], grid.shape[axis]
    empty = np.zeros(0, dtype=np.int64)

    tri = vertices[faces]  # (F,3,3)
    pa, pb = tri[:, :, a], tri[:, :, b]
    area2 = (pa[:, 1] - pa[:, 0]) * (pb[:, 2] - pb[:, 0]) - (pa[:, 2] - pa[:, 0]) * (
        pb[:, 1] - pb[:, 0]
    )
    # triangles parallel to the ray are never crossed by the perturbed ray
    fids = np.flatnonzero(area2 != 0)
    if fids.size == 0:
        return empty, empty, empty, empty
    faces, pa, pb, pc = faces[fids], pa[fids], pb[fids], tri[fids, :, axis]
    sigma = np.sign(area2[fids])
    # canonical edge data per (face, edge j): edge j joins corner j -> corner (j+1)%3
    i0, i1 = np.array([0, 1, 2]), np.array([1, 2, 0])
    va, vb = faces[:, i0], faces[:, i1]  # vertex ids
    xa, xb, ya, yb = pa[:, i0], pa[:, i1], pb[:, i0], pb[:, i1]
    swap = (xa > xb) | ((xa == xb) & ((ya > yb) | ((ya == yb) & (va > vb))))
    px, py = np.where(swap, xb, xa), np.where(swap, yb, ya)
    dx, dy = np.where(swap, xa, xb) - px, np.where(swap, ya, yb) - py
    sgn = np.where(swap, -1.0, 1.0)
    w = sigma[:, None] * sgn  # interior side of each canonical edge

    out: list[tuple[np.ndarray, ...]] = []
    u, v = (pa - origin[a]) / h, (pb - origin[b]) / h
    for f, ia, ib in covered_samples(u, v, na, nb):
        cx = origin[a] + h * (ia + 0.5)
        cy = origin[b] + h * (ib + 0.5)
        e_all = dx[f] * (cy[:, None] - py[f]) - dy[f] * (cx[:, None] - px[f])  # (P,3)
        ww = w[f]
        inside = np.all((ww * e_all > 0) | ((e_all == 0) & (ww > 0)), axis=1)
        if not inside.any():
            continue
        f, ia, ib, e_all = f[inside], ia[inside], ib[inside], e_all[inside]
        # barycentric weight of corner k = signed edge function of the opposite edge (j = k+1)
        lam = (e_all * sgn[f])[:, [1, 2, 0]]
        z = (lam * pc[f]).sum(1) / lam.sum(1)
        k = np.floor((z - origin[axis]) / h - 0.5).astype(np.int64) + 1
        out.append((ia, ib, np.clip(k, 0, nc), -sigma[f].astype(np.int64)))
    if not out:
        return empty, empty, empty, empty
    return tuple(np.concatenate(c) for c in zip(*out, strict=True))  # type: ignore[return-value]


def _scanline(mesh: trimesh.Trimesh, grid: Grid, axis: int, rule: str) -> np.ndarray:
    """Occupancy from rays along `axis`. rule 'nonzero' (winding number) or 'parity'."""
    a, b = (axis + 1) % 3, (axis + 2) % 3
    na, nb, nc = grid.shape[a], grid.shape[b], grid.shape[axis]
    ia, ib, k, sign = _ray_hits(
        np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces), grid, axis
    )
    if rule == "parity":
        sign = np.ones_like(sign)
    occ = np.zeros((na, nb, nc), dtype=bool)
    order = np.argsort(ia, kind="stable")
    ia, ib, k, sign = ia[order], ib[order], k[order], sign[order]
    step = max(1, _MAX_SLAB // max(1, nb * (nc + 1)))
    for a0 in range(0, na, step):
        a1 = min(na, a0 + step)
        lo, hi = np.searchsorted(ia, [a0, a1])
        if lo == hi:
            continue
        diff = np.zeros((a1 - a0, nb, nc + 1), dtype=np.int32)
        np.add.at(diff, (ia[lo:hi] - a0, ib[lo:hi], k[lo:hi]), sign[lo:hi])
        wind = np.cumsum(diff[..., :nc], axis=2)
        occ[a0:a1] = (wind % 2 == 1) if rule == "parity" else (wind != 0)
    # occ is indexed (a, b, axis); move back to (x, y, z)
    return np.moveaxis(occ, [0, 1, 2], [a, b, axis])


def surface_points(tri: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """(points, triangle index) covering every triangle: edge samples at spacing <= s plus an
    in-plane square lattice of pitch s. Every point of a triangle is within 2*s of a sample."""
    edges = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]], 1)
    elen = np.linalg.norm(edges, axis=2)  # (F,3)
    n_e = (np.ceil(elen / s).astype(np.int64) + 1).ravel()
    own, loc = _expand(n_e)
    f, j = own // 3, own % 3
    t = loc / np.maximum(n_e[own] - 1, 1)
    pts = [tri[f, j] + t[:, None] * edges[f, j]]
    owners = [f]

    # interior lattice in a frame whose x axis is the longest edge (v0 -> v1 after rolling)
    shift = elen.argmax(1)
    roll = (np.arange(3)[None, :] + shift[:, None]) % 3
    v = np.take_along_axis(tri, roll[:, :, None], axis=1)
    L = np.linalg.norm(v[:, 1] - v[:, 0], axis=1)
    ex = (v[:, 1] - v[:, 0]) / np.maximum(L, 1e-300)[:, None]
    w = v[:, 2] - v[:, 0]
    x2 = np.einsum("ij,ij->i", w, ex)
    perp = w - x2[:, None] * ex
    H = np.linalg.norm(perp, axis=1)
    ey = perp / np.maximum(H, 1e-300)[:, None]
    nx_t = np.floor(L / s).astype(np.int64) + 1
    ny_t = np.where(H > s, np.floor(H / s).astype(np.int64) + 1, 0)
    own, loc = _expand(nx_t * ny_t)
    x = (loc % nx_t[own]) * s
    y = (loc // nx_t[own]) * s
    inside = (y * x2[own] <= H[own] * x) & (y * (L[own] - x2[own]) <= H[own] * (L[own] - x))
    own, x, y = own[inside], x[inside], y[inside]
    pts.append(v[own, 0] + x[:, None] * ex[own] + y[:, None] * ey[own])
    owners.append(own)
    return np.concatenate(pts), np.concatenate(owners)


def _surface_fill(mesh: trimesh.Trimesh, grid: Grid) -> np.ndarray:
    """Mark elements touched by the surface, then fill enclosed cavities."""
    origin = np.asarray(grid.origin, dtype=np.float64)
    pts, _ = surface_points(np.asarray(mesh.triangles, dtype=np.float64), grid.h / 2)
    idx = np.floor((pts - origin) / grid.h).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.asarray(grid.shape)), axis=1)
    shell = np.zeros(grid.shape, dtype=bool)
    shell[tuple(idx[ok].T)] = True
    return ndimage.binary_fill_holes(shell)


def voxelize_mesh(
    mesh: trimesh.Trimesh, grid: Grid, method: VoxelMethod = "auto"
) -> tuple[np.ndarray, list[str]]:
    """bool (nx,ny,nz) occupancy (element center inside the solid) + warnings.

    auto: watertight -> 'scanline' (exact winding-number scan along +z); otherwise 'vote'
    (parity scans along x, y and z, majority of 3 — robust to small holes and cracks).
    'contains' uses trimesh's ray test (slow without embree), 'surface_fill' marks surface
    voxels and fills cavities (over-estimates volume by about half a voxel layer).
    """
    warnings: list[str] = []
    if len(mesh.faces) == 0:
        return np.zeros(grid.shape, dtype=bool), ["mesh has no faces"]
    closed = bool(mesh.is_watertight)
    if method == "auto":
        method = "scanline" if closed else "vote"
    if not closed:
        warnings.append(
            f"mesh is not watertight; voxelized with '{method}', the result may be approximate"
        )
    if method == "scanline":
        rule = "nonzero" if mesh.is_winding_consistent else "parity"
        mask = _scanline(mesh, grid, 2, rule)
    elif method == "vote":
        votes = sum(_scanline(mesh, grid, ax, "parity").astype(np.int8) for ax in range(3))
        mask = votes >= 2
    elif method == "contains":
        mask = np.asarray(mesh.contains(grid.element_centers()), dtype=bool).reshape(grid.shape)
    elif method == "surface_fill":
        mask = _surface_fill(mesh, grid)
    else:
        raise ValueError(f"unknown voxelization method {method!r}")
    return mask, warnings


def _union_bounds(meshes: Sequence[trimesh.Trimesh]) -> np.ndarray:
    bb = np.stack([np.asarray(m.bounds, dtype=np.float64) for m in meshes])
    return np.stack([bb[:, 0].min(0), bb[:, 1].max(0)])


def build_domain(
    design_mesh: trimesh.Trimesh,
    ref_models: Sequence[tuple[trimesh.Trimesh, str]],
    elements_along_longest: int,
    padding: int = 1,
) -> tuple[Grid, np.ndarray, np.ndarray, list[str]]:
    """World-space meshes -> (grid, active, passive, warnings).

    Grid = union of design and keep_in bounds (keep_out never extends it).
    active = design ∪ keep_in, minus keep_out; passive = 1 on keep_in, 0 elsewhere.
    """
    for _, mode in ref_models:
        if mode not in ("keep_in", "keep_out"):
            raise ValueError(f"unknown reference-model mode {mode!r}")
    keep_in = [m for m, mode in ref_models if mode == "keep_in" and len(m.faces)]
    keep_out = [m for m, mode in ref_models if mode == "keep_out" and len(m.faces)]
    grid = Grid.from_bounds(
        _union_bounds([design_mesh, *keep_in]), elements_along_longest, padding=padding
    )
    warnings: list[str] = []
    active, w = voxelize_mesh(design_mesh, grid)
    warnings += [f"design: {s}" for s in w]
    if not active.any():
        warnings.append("design mesh produced no elements; increase the resolution")
    passive = np.zeros(grid.shape, dtype=np.int8)
    for i, m in enumerate(keep_in):
        mask, w = voxelize_mesh(m, grid)
        warnings += [f"keep_in {i}: {s}" for s in w]
        if not mask.any():
            warnings.append(f"keep_in {i} covers no element centers")
        active |= mask
        passive[mask] = 1
    for i, m in enumerate(keep_out):
        mask, w = voxelize_mesh(m, grid)
        warnings += [f"keep_out {i}: {s}" for s in w]
        if not (mask & active).any():
            warnings.append(f"keep_out {i} removes no elements")
        active &= ~mask
        passive[mask] = 0
    n_comp = ndimage.label(active, structure=np.ones((3, 3, 3), dtype=bool))[1]
    if n_comp > 1:
        warnings.append(f"active region has {n_comp} disconnected parts")
    return grid, active, passive, warnings


def node_element_count(active: np.ndarray) -> np.ndarray:
    """uint8 (nx+1,ny+1,nz+1): number of active elements touching each node (0..8)."""
    p = np.pad(np.asarray(active, dtype=np.uint8), 1)
    nx, ny, nz = active.shape
    count = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                count += p[dx : dx + nx + 1, dy : dy + ny + 1, dz : dz + nz + 1]
    return count


def domain_stats(
    grid: Grid, active: np.ndarray, passive: np.ndarray, warnings: Sequence[str] = ()
) -> dict:
    """`VoxelStats` fields."""
    active = np.asarray(active, dtype=bool)
    passive = np.asarray(passive)
    n_active = int(active.sum())
    n_nodes = int((node_element_count(active) > 0).sum())
    try:
        from topop.core.fem import Assembler

        est_bytes = int(Assembler.estimate_bytes(n_active))
    except (ImportError, AttributeError, TypeError, ValueError):  # estimator not available
        est_bytes = int(n_active * 576 * 16 * 1.5)
    return {
        "nx": grid.nx,
        "ny": grid.ny,
        "nz": grid.nz,
        "h": float(grid.h),
        "origin": [float(v) for v in grid.origin],
        "n_active": n_active,
        "n_free": int((active & (passive == 0)).sum()),
        "n_passive_solid": int((active & (passive == 1)).sum()),
        "n_passive_void": int((active & (passive == -1)).sum()),
        "n_nodes": n_nodes,
        "n_dof": 3 * n_nodes,
        "est_bytes": est_bytes,
        "est_sec_per_iter": 8e-5 * n_active,
        "warnings": list(warnings),
    }
