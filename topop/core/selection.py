"""Resolve `schemas.Selection` mappings to full-grid node ids. Pure numpy/trimesh, no pydantic.

`sel` is a plain mapping with the same keys as the schema models (`sel["kind"]`, ...).
Meshes passed to `resolve_selection` are already in world coordinates, keyed by mesh_id.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import trimesh
from scipy import sparse
from scipy.spatial import cKDTree

from topop.core.problem import Grid
from topop.core.voxelize import node_element_count, surface_points, transform_matrix

FACE_TOL = 0.75 * np.sqrt(3.0)  # max node-to-face distance, in voxel edges
OCCLUSION_SLACK = 0.25  # dist_sel may exceed dist_full by this many voxel edges
SAMPLE_PITCH = 1.0  # surface sample spacing for the candidate search, in voxel edges


def surface_nodes(grid: Grid, active: np.ndarray) -> np.ndarray:
    """Full-grid ids of nodes touching >= 1 and < 8 active elements (incl. inner holes)."""
    c = node_element_count(active).ravel()
    return np.flatnonzero((c > 0) & (c < 8)).astype(np.int64)


def active_nodes(grid: Grid, active: np.ndarray) -> np.ndarray:
    return np.flatnonzero(node_element_count(active).ravel() > 0).astype(np.int64)


def node_xyz(grid: Grid, node_ids: np.ndarray) -> np.ndarray:
    """(n,3) world coordinates of full-grid node ids."""
    idx = np.stack(np.unravel_index(np.asarray(node_ids, dtype=np.int64), grid.node_shape), 1)
    return np.asarray(grid.origin, dtype=np.float64) + grid.h * idx.astype(np.float64)


def _unit(v: Sequence[float], what: str) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = np.linalg.norm(v)
    if v.size != 3 or not np.isfinite(n) or n == 0:
        raise ValueError(f"{what} must be a non-zero 3-vector")
    return v / n


def faces_from_normal(
    mesh: trimesh.Trimesh,
    direction: Sequence[float],
    angle_deg: float,
    within: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Face ids whose unit normal is within `angle_deg` of `direction`.

    `within` ([[min],[max]]) keeps faces whose bounding box overlaps the box: a superset of
    "centroid inside", so a large triangle spanning the box is not lost (resolve_selection then
    clips the resulting nodes to the box).
    """
    d = _unit(direction, "direction")
    cos = np.cos(np.radians(float(angle_deg)))
    ok = np.asarray(mesh.face_normals) @ d >= cos - 1e-12
    if within is not None:
        lo, hi = np.asarray(within, dtype=np.float64).reshape(2, 3)
        tri = np.asarray(mesh.triangles)
        overlap = np.all((tri.min(1) <= hi) & (tri.max(1) >= lo), axis=1)
        ok &= overlap
    return np.flatnonzero(ok).astype(np.int64)


# ---- facets ------------------------------------------------------------------------------------
#
# Two phases. A: planar region growing from the largest faces; a neighbour joins when its normal
# is within angle_deg of the group's area-weighted mean normal and within 2*angle_deg of the seed
# normal, so growth stops on fillets and spheres (curved surfaces fragment into thin strips).
# B: Phase-A groups that are *strips* (see `_STRIP_AREA_RATIO`) are chained into curved regions
# through "smooth" adjacencies (mean normals within 3*angle_deg) or "regular" ones (equal steps
# between equal flat strips of a coarse tessellation, see `_regular_pairs`) and each region is
# tested against a cylinder model. Strips are split into cylinder-like and sphere-like ones by
# their discrete bending tensor sum(theta_e * len_e * e e^T) over their smooth or regular mesh
# edges: on a cylinder every bend is around the axis (rank 1), on a sphere it is isotropic. Only
# like joins like, so a capsule's barrel and caps end up in different regions. A fitted cylinder
# then sheds boundary faces that leave its surface (`_peel`) and takes back connected faces that
# lie on it (`_grab`), so the tolerance of Phase A does not blur where a fillet starts.
# Adjacent cylinders on one axis line merge (a seam or a change of tessellation density splits a
# region). Strips that are no cylinder become "other", one facet per smoothly connected patch. A
# mostly sphere-like patch that fits a sphere becomes a "sphere" and grabs the connected faces
# that lie on it (`_grab_sphere`: rows of a coarse sphere that Phase B could not chain).

_SMOOTH_FACTOR = 3.0  # strips chain when their mean normals differ by <= 3 * angle_deg
# A Phase-A group is a strip (a fragment of a curved surface) when it has a smooth neighbour
# group at least 1/ratio of its area. Strips of one surface have similar areas; a plane next to
# a fillet is much larger than the fillet's strips, so it is never chained into the fillet.
_STRIP_AREA_RATIO = 4.0
_CYL_ANISOTROPY = 0.5  # bending tensor lambda2/lambda1 at or below this -> cylinder-like strip
_CYL_RESIDUAL = 0.02  # max rms radial residual of the circle fit, relative to the radius
_CYL_MIN_GROUPS = 3  # a cylinder region needs >= 3 Phase-A groups (2 planes at 10 deg are no arc)
_CYL_MIN_TURN = 4.0  # ... and normals turning by >= 4 * angle_deg (3 planks at 6 deg are no arc)
_BIG_FLAT = 0.1  # in a region that is no cylinder, flat groups with >= 10 % of its area are planes
# Coarse tessellation: flat strips chain across steps above 3 * angle_deg up to this many degrees
# when the steps, hinge lengths and strip areas agree within _REGULAR_TOL. 37 deg: a 10-gon (36)
# is still a cylinder, an octagon (45) and anything coarser stay planes.
_REGULAR_MAX_STEP = 37.0
_REGULAR_TOL = 0.2
_MERGE_AXIS_DEG = 10.0  # adjacent cylinders merge: axes within max(3 * angle_deg, 10 deg), and
_MERGE_TOL = 0.05  # radii within 5 %, axis lines (sphere centres) within 5 % of the radius
_SPH_RESIDUAL = 0.02  # max rms radial residual of the sphere fit, relative to the radius
_SPH_INLIERS = 0.95  # share of faces whose normal is within angle_deg of (centroid - centre)
# A face this thin (altitude / sqrt(total area)) whose normal disagrees with all its neighbours
# has a noise normal (float32 STL rounding tilts it 1 deg at ~4e-6): it joins any group, unweighted.
_SLIVER = 1e-4
_FACET_CACHE: OrderedDict[tuple[int, float], _Segmentation] = OrderedDict()
_FACET_CACHE_SIZE = 8
_FACET_LOCK = threading.Lock()  # the server segments from worker threads


@dataclass(frozen=True)
class _Segmentation:
    label: np.ndarray  # (n_faces,) facet id (already ranked)
    facets: tuple[dict, ...]


def _neighbours(n: int, pairs: np.ndarray) -> tuple[list[int], list[int]]:
    """CSR face adjacency (Python lists), neighbours sorted by face id."""
    a = np.concatenate([pairs[:, 0], pairs[:, 1]])
    b = np.concatenate([pairs[:, 1], pairs[:, 0]])
    order = np.lexsort((b, a))
    indptr = np.searchsorted(a[order], np.arange(n + 1))
    return indptr.tolist(), b[order].tolist()


def _grow_planar(
    normals: np.ndarray,
    area: np.ndarray,
    unreliable: np.ndarray,
    pairs: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Phase A group label per face. Seeds by decreasing area (relative, rounded, so rigid
    transforms keep the order), ties by face id; BFS with neighbours in face-id order.
    Unreliable faces (degenerate or slivers: their normal is noise) never seed a group and join
    whichever group reaches them first, without weight."""
    n = len(area)
    ip, nb = _neighbours(n, pairs)
    cos1 = float(np.cos(np.radians(angle_deg)))
    cos2 = float(np.cos(np.radians(min(2 * angle_deg, 180.0))))
    amax = float(area.max()) or 1.0
    order = np.lexsort((np.arange(n), -np.round(area / amax, 9), unreliable))
    nrm = normals.tolist()
    ar = area.tolist()
    bad = unreliable.tolist()
    lab = [-1] * n
    g = 0
    for s in order[: n - int(unreliable.sum())].tolist():
        if lab[s] >= 0:
            continue
        lab[s] = g
        sx, sy, sz = nrm[s]
        a = ar[s]
        mx, my, mz, mlen = a * sx, a * sy, a * sz, a
        queue = [s]
        i = 0
        while i < len(queue):
            f = queue[i]
            i += 1
            for k in range(ip[f], ip[f + 1]):
                h = nb[k]
                if lab[h] >= 0:
                    continue
                if not bad[h]:
                    hx, hy, hz = nrm[h]
                    if (
                        hx * sx + hy * sy + hz * sz < cos2
                        or hx * mx + hy * my + hz * mz < cos1 * mlen
                    ):
                        continue
                    a = ar[h]
                    mx, my, mz = mx + a * hx, my + a * hy, mz + a * hz
                    mlen = (mx * mx + my * my + mz * mz) ** 0.5
                lab[h] = g
                queue.append(h)
        g += 1
    for s in order.tolist():  # left-over islands of unreliable faces
        if lab[s] < 0:
            lab[s] = g
            queue = [s]
            for f in queue:
                for k in range(ip[f], ip[f + 1]):
                    if lab[nb[k]] < 0:
                        lab[nb[k]] = g
                        queue.append(nb[k])
            g += 1
    return np.asarray(lab, dtype=np.int64)


def _bincount3(label: np.ndarray, w: np.ndarray, vec: np.ndarray, n: int) -> np.ndarray:
    return np.stack([np.bincount(label, weights=w * vec[:, i], minlength=n) for i in range(3)], 1)


def _components(n: int, edges: np.ndarray) -> np.ndarray:
    return np.asarray(
        trimesh.graph.connected_component_labels(edges.reshape(-1, 2), node_count=n),
        dtype=np.int64,
    )


class _Cylinder(NamedTuple):
    axis: np.ndarray
    center: np.ndarray  # on the axis, at the area-weighted mean axial position
    radius: float
    na: np.ndarray  # |n . axis| per fitted face
    resid: float  # rms radial residual / radius


def _fit_cylinder(mesh: trimesh.Trimesh, faces: np.ndarray, angle_deg: float) -> _Cylinder | None:
    """Axis = least eigenvector of sum(area * n n^T) (cylinder normals are perpendicular to it),
    then an algebraic (Kasa) circle fit of the vertices projected along the axis. None unless
    the normals turn by >= _CYL_MIN_TURN * angle_deg around the axis, their rms axial component is
    <= sin(angle_deg) and the rms radial residual is <= _CYL_RESIDUAL * radius."""
    n = np.asarray(mesh.face_normals)[faces]
    a = np.asarray(mesh.area_faces)[faces]
    tot = float(a.sum())
    if not tot > 0:
        return None
    w, v = np.linalg.eigh((n * a[:, None]).T @ n)
    axis = v[:, 0]
    if w[1] < tot * _min_spread(angle_deg):
        return None
    na = np.abs(n @ axis)
    if np.sqrt(np.sum(a * na**2) / tot) > np.sin(np.radians(angle_deg)):
        return None
    u = v[:, 1] - axis * (v[:, 1] @ axis)
    u /= np.linalg.norm(u)
    vv = np.cross(axis, u)
    pts = np.asarray(mesh.vertices)[np.unique(np.asarray(mesh.faces)[faces])]
    mean = pts.mean(0)
    x, y = (pts - mean) @ u, (pts - mean) @ vv
    lhs = np.stack([x, y, np.ones_like(x)], 1)
    sol, *_ = np.linalg.lstsq(lhs, -(x * x + y * y), rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r2 = cx * cx + cy * cy - sol[2]
    if not r2 > 0:
        return None
    r = float(np.sqrt(r2))
    resid = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))) / r
    if resid > _CYL_RESIDUAL:
        return None
    z = float(np.sum(a * (np.asarray(mesh.triangles_center)[faces] @ axis)) / tot)
    center = mean + cx * u + cy * vv + (z - mean @ axis) * axis
    big = np.flatnonzero(np.abs(axis) > 1e-9)
    if big.size and axis[big[0]] < 0:  # lexicographically towards +x/+y/+z
        axis = -axis
    return _Cylinder(axis, center, r, na, resid)


def _peel(
    mesh: trimesh.Trimesh,
    adj: sparse.csr_matrix,
    faces: np.ndarray,
    cyl: _Cylinder,
    unreliable: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Mask over `faces`: faces whose normal leaves the cylinder (|n . axis| above
    max(sin(angle_deg/4), 4 * area-weighted median)) and that reach the region's boundary
    through such faces, e.g. the first ring of a capsule cap, which Phase A puts into the
    barrel's strips."""
    na = cyl.na
    a = np.asarray(mesh.area_faces)[faces]
    srt = np.argsort(na)
    cum = np.cumsum(a[srt])
    med = na[srt][min(int(np.searchsorted(cum, 0.5 * cum[-1])), len(srt) - 1)]
    bad = (na > max(np.sin(np.radians(angle_deg / 4)), 4 * med)) & ~unreliable
    if not bad.any():
        return bad
    sub = adj[faces][:, faces].tocoo()
    boundary = np.diff(adj[faces].indptr) > np.bincount(sub.row, minlength=len(faces))
    keep = bad[sub.row] & bad[sub.col]
    comp = _components(len(faces), np.stack([sub.row[keep], sub.col[keep]], 1))
    hit = np.zeros(int(comp.max()) + 1, dtype=bool)
    hit[comp[bad & boundary]] = True
    return bad & hit[comp]


def _grab(
    mesh: trimesh.Trimesh,
    adj: sparse.csr_matrix,
    faces: np.ndarray,
    cyl: _Cylinder,
    free: np.ndarray,
    unreliable: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Faces of `free` (plane / other faces) connected to the cylinder that lie on it: normal
    within angle_deg/4 of perpendicular to the axis, every vertex within max(4 * rms residual,
    1e-3) * radius of the surface. Phase A puts a fillet's first strips (normal < angle_deg off
    the tangent plane) into that plane, and strip ends into a corner blend's groups; this hands
    them back."""
    tol = max(4 * cyl.resid, 1e-3) * cyl.radius
    sin_n = np.sin(np.radians(angle_deg / 4))
    normals = np.asarray(mesh.face_normals)
    seen = np.zeros(len(mesh.faces), dtype=bool)
    seen[faces] = True
    front, taken = faces, []
    while front.size:
        nb = np.unique(adj[front].indices)
        nb = nb[free[nb] & ~seen[nb]]
        seen[nb] = True
        d = np.asarray(mesh.triangles)[nb] - cyl.center
        radial = np.linalg.norm(d - (d @ cyl.axis)[..., None] * cyl.axis, axis=2)
        ok = ((np.abs(normals[nb] @ cyl.axis) <= sin_n) | unreliable[nb]) & np.all(
            np.abs(radial - cyl.radius) <= tol, axis=1
        )
        front = nb[ok]
        taken.append(front)
    return np.concatenate(taken) if taken else np.zeros(0, dtype=np.int64)


def _min_spread(angle_deg: float) -> float:
    """Variance of normals turning uniformly by _CYL_MIN_TURN * angle_deg (t^2 / 12)."""
    return float(np.radians(_CYL_MIN_TURN * angle_deg) ** 2 / 12)


def _normal_spread(second: np.ndarray, first: np.ndarray, tot: np.ndarray | float) -> np.ndarray:
    """Middle eigenvalue of the area-weighted covariance of unit normals, from sum(a n n^T)
    (...,3,3), sum(a n) (...,3) and sum(a): how far the normals turn in their second direction
    (t^2 / 12 for a uniform turn by t). The least one is ~0 for any cap (normals near a mean)."""
    tot = np.asarray(tot, dtype=np.float64)[..., None, None]
    mean = first[..., :, None] / tot
    return np.linalg.eigvalsh(second / tot - mean * np.swapaxes(mean, -1, -2))[..., 1]


def _close(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(x - y) <= _REGULAR_TOL * np.maximum(np.abs(x), np.abs(y))


def _regular_pairs(
    gp: np.ndarray,
    phi: np.ndarray,
    hinge: np.ndarray,
    bend: np.ndarray,
    g_area: np.ndarray,
    g_flat: np.ndarray,
    lo: float,
) -> np.ndarray:
    """Mask over the group pairs `gp`: steps of a coarsely tessellated curve. A pair A-B of flat
    groups at lo < phi <= _REGULAR_MAX_STEP qualifies when the chain continues on B's (or A's)
    other side, B-C, with the same step, hinge length (shared edge length) and bend sign, and
    A, B, C have the same area (equal hinges and areas: equal strip widths). A feature edge
    between two flats has no such continuation, nor has a chamfer between flats of other widths.
    `phi`, `hinge`, `bend` (+1 convex, -1 concave, 0 mixed) are per pair."""
    ok = np.zeros(len(gp), dtype=bool)
    idx = np.flatnonzero(
        (phi > lo)
        & (phi <= np.radians(_REGULAR_MAX_STEP))
        & (bend != 0)
        & g_flat[gp[:, 0]]
        & g_flat[gp[:, 1]]
    )
    # incidences (centre group, other group, pair) with both groups of about the same area,
    # grouped by centre; a centre with k incidences is the middle of k(k-1)/2 candidate chains
    c = np.concatenate([gp[idx, 0], gp[idx, 1]])
    o = np.concatenate([gp[idx, 1], gp[idx, 0]])
    p = np.concatenate([idx, idx])
    keep = _close(g_area[c], g_area[o])
    c, o, p = c[keep], o[keep], p[keep]
    srt = np.lexsort((p, c))
    c, o, p = c[srt], o[srt], p[srt]
    for off in range(1, len(c)):
        i = np.flatnonzero(c[:-off] == c[off:])
        if i.size == 0:
            break
        a, b = p[i], p[i + off]
        good = (
            _close(phi[a], phi[b])
            & _close(hinge[a], hinge[b])
            & (bend[a] == bend[b])
            & _close(g_area[o[i]], g_area[o[i + off]])
        )
        ok[a[good]] = True
        ok[b[good]] = True
    return ok


class _Sphere(NamedTuple):
    center: np.ndarray
    radius: float
    resid: float  # rms radial residual / radius
    sign: float  # +1 normals point away from the centre, -1 towards it (a pocket)


def _fit_sphere(
    mesh: trimesh.Trimesh, faces: np.ndarray, unreliable: np.ndarray, angle_deg: float
) -> _Sphere | None:
    """Algebraic least squares |x|^2 + a.x + b = 0 on the vertices. None unless the normals turn
    by >= _CYL_MIN_TURN * angle_deg both ways (`_normal_spread`), the rms radial residual is <=
    _SPH_RESIDUAL * radius and the face normals point along (centroid - centre), all outwards
    or all inwards, within angle_deg on >= _SPH_INLIERS of the reliable faces."""
    n = np.asarray(mesh.face_normals)[faces]
    a = np.asarray(mesh.area_faces)[faces]
    tot = float(a.sum())
    if not tot > 0 or _normal_spread((n * a[:, None]).T @ n, a @ n, tot) < _min_spread(angle_deg):
        return None
    pts = np.asarray(mesh.vertices)[np.unique(np.asarray(mesh.faces)[faces])]
    mean = pts.mean(0)
    q = pts - mean
    scale = float(np.sqrt(np.mean(np.einsum("ij,ij->i", q, q))))
    if not scale > 0:
        return None
    q /= scale
    lhs = np.concatenate([2 * q, np.ones((len(q), 1))], 1)
    sol, *_ = np.linalg.lstsq(lhs, np.einsum("ij,ij->i", q, q), rcond=None)
    c = sol[:3]
    r2 = float(sol[3] + c @ c)
    if not r2 > 0:
        return None
    r = np.sqrt(r2)
    resid = float(np.sqrt(np.mean((np.linalg.norm(q - c, axis=1) - r) ** 2)) / r)
    if resid > _SPH_RESIDUAL:
        return None
    center = mean + scale * c
    d = np.asarray(mesh.triangles_center)[faces] - center
    dn = np.linalg.norm(d, axis=1)
    cos = np.divide(np.einsum("ij,ij->i", d, n), dn, out=np.zeros(len(d)), where=dn > 0)
    sign = 1.0 if np.sum(a * cos) >= 0 else -1.0
    ok = (sign * cos >= np.cos(np.radians(angle_deg)))[~unreliable]
    if ok.size and ok.mean() < _SPH_INLIERS:
        return None
    return _Sphere(center, float(scale * r), resid, sign)


def _grab_sphere(
    mesh: trimesh.Trimesh,
    adj: sparse.csr_matrix,
    faces: np.ndarray,
    sph: _Sphere,
    free: np.ndarray,
    unreliable: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Faces of `free` connected to the sphere that lie on it: every vertex within
    max(4 * rms residual, 1e-3) * radius of the surface, normal within angle_deg of the radial
    direction at the centroid. On a coarse sphere the rows near the poles or steps between
    unequal bands do not chain in Phase B and end up as small planes, cylinders or patches."""
    tol = max(4 * sph.resid, 1e-3) * sph.radius
    cos_n = np.cos(np.radians(angle_deg))
    normals = np.asarray(mesh.face_normals)
    seen = np.zeros(len(mesh.faces), dtype=bool)
    seen[faces] = True
    front, taken = faces, []
    while front.size:
        nb = np.unique(adj[front].indices)
        nb = nb[free[nb] & ~seen[nb]]
        seen[nb] = True
        r = np.linalg.norm(np.asarray(mesh.triangles)[nb] - sph.center, axis=2)
        d = np.asarray(mesh.triangles_center)[nb] - sph.center
        dn = np.linalg.norm(d, axis=1)
        cos = np.divide(
            np.einsum("ij,ij->i", d, normals[nb]), dn, out=np.zeros(len(nb)), where=dn > 0
        )
        ok = np.all(np.abs(r - sph.radius) <= tol, axis=1) & (
            (sph.sign * cos >= cos_n) | unreliable[nb]
        )
        front = nb[ok]
        taken.append(front)
    return np.concatenate(taken) if taken else np.zeros(0, dtype=np.int64)


def _same_cylinder(p: _Cylinder, q: _Cylinder, angle_deg: float) -> bool:
    tol = np.cos(np.radians(max(_SMOOTH_FACTOR * angle_deg, _MERGE_AXIS_DEG)))
    if abs(float(p.axis @ q.axis)) < tol:
        return False
    if abs(p.radius - q.radius) > _MERGE_TOL * max(p.radius, q.radius):
        return False
    d = q.center - p.center
    off = max(np.linalg.norm(d - (d @ p.axis) * p.axis), np.linalg.norm(d - (d @ q.axis) * q.axis))
    return bool(off <= _MERGE_TOL * min(p.radius, q.radius))


def _same_sphere(p: _Sphere, q: _Sphere) -> bool:
    r = min(p.radius, q.radius)
    return bool(
        abs(p.radius - q.radius) <= _MERGE_TOL * max(p.radius, q.radius)
        and np.linalg.norm(p.center - q.center) <= _MERGE_TOL * r
    )


def _merge_adjacent(
    adj: np.ndarray,
    owner: np.ndarray,
    items: list[tuple[np.ndarray, Any]],
    same: Callable[[Any, Any], bool],
    refit: Callable[[np.ndarray], Any],
) -> tuple[list[tuple[np.ndarray, Any]], np.ndarray]:
    """Transitively merge items (faces, fit) that share a mesh edge, `same` their fits and refit
    as one (`refit(faces)` not None). `owner`: item id per face or -1. Pairs are visited in id
    order and a merged item keeps the lower id, so the result is deterministic."""
    if len(items) < 2:
        return items, owner
    oa, ob = owner[adj[:, 0]], owner[adj[:, 1]]
    m = (oa >= 0) & (ob >= 0) & (oa != ob)
    root = list(range(len(items)))

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    faces = [f for f, _ in items]
    fits = [g for _, g in items]
    for a, b in np.unique(np.sort(np.stack([oa[m], ob[m]], 1), axis=1), axis=0).tolist():
        ra, rb = find(a), find(b)
        if ra == rb or not same(items[a][1], items[b][1]):
            continue
        union = np.union1d(faces[ra], faces[rb])
        fit = refit(union)
        if fit is None:
            continue
        lo, hi = min(ra, rb), max(ra, rb)
        root[hi] = lo
        faces[lo], fits[lo] = union, fit
    roots = sorted({find(i) for i in range(len(items))})
    if len(roots) == len(items):
        return items, owner
    new = np.empty(len(items), dtype=np.int64)
    new[roots] = np.arange(len(roots))
    remap = new[[find(i) for i in range(len(items))]]
    owner = np.where(owner >= 0, remap[np.maximum(owner, 0)], -1)
    return [(faces[r], fits[r]) for r in roots], owner


def _find_spheres(
    mesh: trimesh.Trimesh,
    patch: np.ndarray,
    oth: np.ndarray,
    doubly: np.ndarray,
    unreliable: np.ndarray,
    angle_deg: float,
) -> tuple[list[tuple[np.ndarray, _Sphere]], np.ndarray]:
    """Sphere fits of the "other" patches (`patch` label per face, used where `oth`) and the
    sphere id per face (-1 elsewhere). Only patches with >= 4 faces, normals that turn both ways
    and at least half of their area `doubly` curved (Phase-B groups with an isotropic bending
    tensor) are fitted: any two coaxial circles lie on a sphere, so a cone, a countersink or a
    torus band (singly curved: rank-1 bending) would pass the residual and normal tests."""
    sph_of_face = np.full(len(mesh.faces), -1, dtype=np.int64)
    ids = np.flatnonzero(oth)
    if ids.size < 4:
        return [], sph_of_face
    _, lbl, cnt = np.unique(patch[ids], return_inverse=True, return_counts=True)
    lbl = lbl.ravel()
    big = cnt[lbl] >= 4
    ids, lbl = ids[big], lbl[big]
    if ids.size == 0:
        return [], sph_of_face
    _, lbl = np.unique(lbl, return_inverse=True)
    lbl = lbl.ravel()
    k = int(lbl.max()) + 1
    n = np.asarray(mesh.face_normals)[ids]
    a = np.asarray(mesh.area_faces)[ids]
    tens = np.zeros((k, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            tens[:, i, j] = tens[:, j, i] = np.bincount(
                lbl, weights=a * n[:, i] * n[:, j], minlength=k
            )
    tot = np.bincount(lbl, weights=a, minlength=k)
    turns = (_normal_spread(tens, _bincount3(lbl, a, n, k), tot) >= _min_spread(angle_deg)) & (
        np.bincount(lbl, weights=a * doubly[ids], minlength=k) >= 0.5 * tot
    )
    order = np.argsort(lbl, kind="stable")
    starts = np.searchsorted(lbl[order], np.arange(k + 1))
    spheres: list[tuple[np.ndarray, _Sphere]] = []
    for g in np.flatnonzero(turns).tolist():
        faces = ids[order[starts[g] : starts[g + 1]]]
        fit = _fit_sphere(mesh, faces, unreliable[faces], angle_deg)
        if fit is not None:
            sph_of_face[faces] = len(spheres)
            spheres.append((faces, fit))
    return spheres, sph_of_face


def _segment(mesh: trimesh.Trimesh, angle_deg: float) -> _Segmentation:
    n = len(mesh.faces)
    if n == 0:
        return _Segmentation(np.zeros(0, dtype=np.int64), ())
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    area = np.asarray(mesh.area_faces, dtype=np.float64)
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64).reshape(-1, 2)
    theta = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    smooth = np.radians(min(_SMOOTH_FACTOR * angle_deg, 180.0))

    # ---- phase A
    tri = np.asarray(mesh.triangles)
    longest = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(1)
    altitude = np.divide(2 * area, longest, out=np.zeros(n), where=longest > 0)
    # a thin face whose normal disagrees with every neighbour: its normal is rounding noise
    agree = np.full(n, np.pi)
    np.minimum.at(agree, adj.ravel(), np.repeat(theta, 2))
    unreliable = (np.linalg.norm(normals, axis=1) < 0.5) | (
        (altitude <= _SLIVER * np.sqrt(area.sum())) & (agree > np.radians(angle_deg))
    )
    lab = _grow_planar(normals, area, unreliable, adj, angle_deg)
    ng = int(lab.max()) + 1
    g_area = np.bincount(lab, weights=area, minlength=ng)
    g_sum = _bincount3(lab, area, normals, ng)
    g_len = np.linalg.norm(g_sum, axis=1)
    g_n = np.divide(g_sum, g_len[:, None], out=np.zeros_like(g_sum), where=g_len[:, None] > 0)
    g_flat = g_len >= np.cos(np.radians(angle_deg / 2)) * g_area

    # ---- phase B: smooth and regular group adjacency, strips, bending tensors
    ev = np.asarray(mesh.face_adjacency_edges, dtype=np.int64).reshape(-1, 2)
    vec = np.asarray(mesh.vertices)[ev[:, 1]] - np.asarray(mesh.vertices)[ev[:, 0]]
    elen = np.linalg.norm(vec, axis=1)
    convex_e = np.asarray(mesh.face_adjacency_convex, dtype=bool)
    la, lb = lab[adj[:, 0]], lab[adj[:, 1]]
    cross = la != lb
    keys, pinv = np.unique(
        (np.minimum(la, lb) * ng + np.maximum(la, lb))[cross], return_inverse=True
    )
    pinv = pinv.ravel()
    gp = np.stack([keys // ng, keys % ng], 1)  # sorted pairs of adjacent groups
    phi = np.arccos(np.clip(np.einsum("ij,ij->i", g_n[gp[:, 0]], g_n[gp[:, 1]]), -1, 1))
    hinge = np.bincount(pinv, weights=elen[cross], minlength=len(gp))
    signed = np.where(convex_e, elen, -elen)[cross]
    bend_sign = np.sign(np.bincount(pinv, weights=signed, minlength=len(gp)))
    reg = _regular_pairs(gp, phi, hinge, bend_sign, g_area, g_flat, smooth)
    reg_edge = np.zeros(len(adj), dtype=bool)  # mesh edges between regular pairs
    reg_edge[cross] = reg[pinv]
    gp = gp[((phi <= smooth) | reg) & (g_area[gp[:, 0]] > 0) & (g_area[gp[:, 1]] > 0)]
    nbr_area = np.zeros(ng)
    np.maximum.at(nbr_area, gp[:, 0], g_area[gp[:, 1]])
    np.maximum.at(nbr_area, gp[:, 1], g_area[gp[:, 0]])
    strip = (nbr_area > 0) & (g_area <= _STRIP_AREA_RATIO * nbr_area)

    rel = ~unreliable[adj[:, 0]] & ~unreliable[adj[:, 1]]
    bend = (theta > 0) & ((theta <= smooth) | reg_edge) & (elen > 0) & rel
    e = vec[bend] / elen[bend, None]
    w = (theta * elen)[bend]
    bcross = cross[bend]

    def per_group(c: np.ndarray) -> np.ndarray:  # edge values -> both groups (once if internal)
        return np.bincount(la[bend], weights=c, minlength=ng) + np.bincount(
            lb[bend], weights=np.where(bcross, c, 0.0), minlength=ng
        )

    tens = np.zeros((ng, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            tens[:, i, j] = tens[:, j, i] = per_group(w * e[:, i] * e[:, j])
    lam, vecs = np.linalg.eigh(tens)
    g_axis = vecs[:, :, 2]
    cyl_like = (lam[:, 2] > 0) & (lam[:, 1] <= _CYL_ANISOTROPY * lam[:, 2])
    # +1 convex, -1 concave: a convex and a concave fillet that meet tangentially (an S-curve)
    # are two cylinders, not one
    convex = convex_e[bend]
    g_convex = np.sign(per_group(np.where(convex, w, -w)))

    sa, sb = gp[:, 0], gp[:, 1]
    both = strip[sa] & strip[sb]
    same_axis = np.abs(np.einsum("ij,ij->i", g_axis[sa], g_axis[sb])) >= np.cos(smooth)
    same_cyl = cyl_like[sa] & cyl_like[sb] & same_axis & (g_convex[sa] == g_convex[sb])
    join = both & (same_cyl | (~cyl_like[sa] & ~cyl_like[sb]))
    region = _components(ng, gp[join])
    # a group no larger than a curved strip of another region next to it (e.g. a lone sliver
    # group on a sphere) is curved too, whatever its own region turns out to be
    other_strip = strip[sb] & (region[sa] != region[sb])
    absorbed = np.zeros(ng, dtype=bool)
    absorbed[sa[other_strip & (g_area[sa] <= _STRIP_AREA_RATIO * g_area[sb])]] = True
    other_strip = strip[sa] & (region[sa] != region[sb])
    absorbed[sb[other_strip & (g_area[sb] <= _STRIP_AREA_RATIO * g_area[sa])]] = True

    # ---- classify regions: face kind 0 plane (per group), 1 cylinder (per region), 2 other
    kind = np.zeros(ng, dtype=np.int8)  # per group
    cyl_of_face = np.full(n, -1, dtype=np.int64)
    cylinders: list[tuple[np.ndarray, _Cylinder]] = []
    fadj = sparse.coo_matrix(
        (np.ones(2 * len(adj), dtype=np.int8), (adj.T.ravel(), adj[:, ::-1].T.ravel())),
        shape=(n, n),
    ).tocsr()
    order = np.argsort(lab, kind="stable")
    starts = np.searchsorted(lab[order], np.arange(ng + 1))

    def faces_of(groups: np.ndarray) -> np.ndarray:
        return np.sort(np.concatenate([order[starts[g] : starts[g + 1]] for g in groups]))

    pending = [np.flatnonzero(region == r) for r in np.unique(region[strip])]
    while pending:
        groups = pending.pop()
        if len(groups) >= _CYL_MIN_GROUPS and cyl_like[groups].all():
            faces = faces_of(groups)
            fit = _fit_cylinder(mesh, faces, angle_deg)
            if fit is not None:
                drop = _peel(mesh, fadj, faces, fit, unreliable[faces], angle_deg)
                refit = _fit_cylinder(mesh, faces[~drop], angle_deg) if drop.any() else None
                if refit is not None:
                    faces, fit = faces[~drop], refit
                kind[groups] = 1
                cyl_of_face[faces] = len(cylinders)
                cylinders.append((faces, fit))
                continue
        big = g_flat[groups] & (g_area[groups] >= _BIG_FLAT * g_area[groups].sum())
        if big.any() and (~big).any():
            # e.g. two wide planes at a shallow angle chained with a fillet: keep the planes,
            # retry the rest (terminates: every split drops >= 1 group)
            rest = groups[~big]
            sub = join & np.isin(sa, rest) & np.isin(sb, rest)
            comp = _components(ng, gp[sub])[rest]
            pending.extend(rest[comp == c] for c in np.unique(comp))
            continue
        if len(groups) < _CYL_MIN_GROUPS:
            kind[groups] = np.where(g_flat[groups] & ~absorbed[groups], 0, 2)
        else:
            kind[groups] = np.where(big, 0, 2)

    face_kind = kind[lab]
    face_kind[(face_kind == 1) & (cyl_of_face < 0)] = 2  # peeled faces
    for c, (faces, fit) in enumerate(cylinders):
        extra = _grab(mesh, fadj, faces, fit, face_kind != 1, unreliable, angle_deg)
        if extra.size:
            faces = np.union1d(faces, extra)
            face_kind[extra] = 1
            cyl_of_face[extra] = c
            cylinders[c] = (faces, _fit_cylinder(mesh, faces, angle_deg) or fit)
    cylinders, cyl_of_face = _merge_adjacent(
        adj,
        cyl_of_face,
        cylinders,
        lambda p, q: _same_cylinder(p, q, angle_deg),
        lambda f: _fit_cylinder(mesh, f, angle_deg),
    )
    # "other" faces: one facet per smoothly connected patch, a sphere where one fits
    oth = face_kind == 2
    sm = oth[adj[:, 0]] & oth[adj[:, 1]] & ((theta <= smooth) | ~rel | reg_edge)
    other_comp = _components(n, adj[sm])
    doubly = (~cyl_like & (lam[:, 2] > 0))[lab]
    spheres, sph_of_face = _find_spheres(mesh, other_comp, oth, doubly, unreliable, angle_deg)
    face_kind[sph_of_face >= 0] = 3
    lost = np.zeros(len(cylinders), dtype=bool)  # cylinders that gave faces to a sphere
    grabbed = False
    for s, (faces, sph) in enumerate(spheres):
        extra = _grab_sphere(mesh, fadj, faces, sph, face_kind != 3, unreliable, angle_deg)
        if extra.size:
            grabbed = True
            lost[cyl_of_face[extra][cyl_of_face[extra] >= 0]] = True
            faces = np.union1d(faces, extra)
            face_kind[extra], sph_of_face[extra], cyl_of_face[extra] = 3, s, -1
            spheres[s] = (faces, _fit_sphere(mesh, faces, unreliable[faces], angle_deg) or sph)
    if lost.any():  # what is left of them is refitted, or "other"
        kept: list[tuple[np.ndarray, _Cylinder]] = []
        for c, (faces, fit) in enumerate(cylinders):
            if lost[c]:
                faces = np.flatnonzero(cyl_of_face == c)
                fit = _fit_cylinder(mesh, faces, angle_deg) if faces.size else None
                if fit is None:
                    face_kind[faces], cyl_of_face[faces] = 2, -1
                    continue
            cyl_of_face[faces] = len(kept)  # <= c: later cylinders keep their old ids so far
            kept.append((faces, fit))
        cylinders = kept
    spheres, sph_of_face = _merge_adjacent(
        adj,
        sph_of_face,
        spheres,
        _same_sphere,
        lambda f: _fit_sphere(mesh, f, unreliable[f], angle_deg),
    )
    oth = face_kind == 2
    if grabbed:
        sm = oth[adj[:, 0]] & oth[adj[:, 1]] & ((theta <= smooth) | ~rel | reg_edge)
        other_comp = _components(n, adj[sm])

    # final label: planes keep their group, cylinders their region, spheres and others their patch
    key = np.where(face_kind == 0, lab, np.where(face_kind == 1, ng + cyl_of_face, 0))
    key[face_kind == 3] = ng + len(cylinders) + sph_of_face[face_kind == 3]
    key[oth] = ng + len(cylinders) + len(spheres) + other_comp[oth]
    _, final = np.unique(key, return_inverse=True)
    final = final.ravel()
    nf = int(final.max()) + 1
    f_area = np.bincount(final, weights=area, minlength=nf)
    f_nf = np.bincount(final, minlength=nf)
    f_sum = _bincount3(final, area, normals, nf)
    f_len = np.linalg.norm(f_sum, axis=1)
    f_n = np.divide(f_sum, f_len[:, None], out=np.zeros_like(f_sum), where=f_len[:, None] > 0)
    safe = np.where(f_area > 0, f_area, 1.0)[:, None]
    f_c = _bincount3(final, area, np.asarray(mesh.triangles_center), nf) / safe
    f_min = np.full((nf, 3), np.inf)
    f_max = np.full((nf, 3), -np.inf)
    np.minimum.at(f_min, final, tri.min(1))
    np.maximum.at(f_max, final, tri.max(1))
    first = np.full(nf, n, dtype=np.int64)
    np.minimum.at(first, final, np.arange(n))
    f_kind = np.zeros(nf, dtype=np.int8)
    f_kind[final] = face_kind
    f_cyl = np.full(nf, -1, dtype=np.int64)
    f_cyl[final] = cyl_of_face
    f_sph = np.full(nf, -1, dtype=np.int64)
    f_sph[final] = sph_of_face

    total = max(float(f_area.sum()), np.finfo(float).tiny)
    rank_order = np.lexsort((first, -np.round(f_area / total, 9)))
    rank = np.empty(nf, dtype=np.int64)
    rank[rank_order] = np.arange(nf)
    out = []
    for r, g in enumerate(rank_order):
        k = int(f_kind[g])
        info = {
            "id": r,
            "n_faces": int(f_nf[g]),
            "area": float(f_area[g]),
            "normal": f_n[g].tolist() if k == 0 else [0.0, 0.0, 0.0],
            "centroid": f_c[g].tolist(),
            "bbox": [f_min[g].tolist(), f_max[g].tolist()],
            "kind": ("plane", "cylinder", "other", "sphere")[k],
            "axis": None,
            "radius": None,
        }
        if k == 1:
            fit = cylinders[int(f_cyl[g])][1]
            info.update(axis=fit.axis.tolist(), radius=fit.radius, centroid=fit.center.tolist())
        elif k == 3:
            sph = spheres[int(f_sph[g])][1]
            info.update(radius=sph.radius, centroid=sph.center.tolist())
        out.append(info)
    return _Segmentation(rank[final], tuple(out))


def _segmentation(mesh: trimesh.Trimesh, angle_deg: float) -> _Segmentation:
    key = (hash(mesh), float(angle_deg))  # trimesh hashes vertices + faces
    with _FACET_LOCK:
        hit = _FACET_CACHE.get(key)
        if hit is not None:
            _FACET_CACHE.move_to_end(key)
            return hit
    seg = _segment(mesh, float(angle_deg))
    seg.label.flags.writeable = False
    with _FACET_LOCK:
        _FACET_CACHE[key] = seg
        while len(_FACET_CACHE) > _FACET_CACHE_SIZE:
            _FACET_CACHE.popitem(last=False)
    return seg


def compute_facets(mesh: trimesh.Trimesh, angle_deg: float = 5.0) -> tuple[list[dict], np.ndarray]:
    """Planar / cylindrical / spherical / other face groups as `FacetInfo` dicts + face_to_facet.

    Order: area descending (rounded to 1e-9 of the total area), ties by lowest face id. Every
    criterion is invariant under rigid transforms, so ids listed for a raw mesh stay valid after
    the project transform is applied.
    """
    seg = _segmentation(mesh, angle_deg)
    facets = [
        {
            **f,
            "normal": list(f["normal"]),
            "centroid": list(f["centroid"]),
            "bbox": [list(b) for b in f["bbox"]],
            "axis": None if f["axis"] is None else list(f["axis"]),
        }
        for f in seg.facets
    ]
    return facets, seg.label.copy()


def facet_faces(mesh: trimesh.Trimesh, angle_deg: float, facet_ids: Sequence[int]) -> np.ndarray:
    """Sorted face ids (int64) of the given facets of `compute_facets(mesh, angle_deg)`."""
    seg = _segmentation(mesh, angle_deg)
    want = np.asarray(facet_ids, dtype=np.int64).ravel()
    bad = want[(want < 0) | (want >= len(seg.facets))]
    if bad.size:
        raise ValueError(f"unknown facet ids {bad.tolist()} (mesh has {len(seg.facets)} facets)")
    return np.flatnonzero(np.isin(seg.label, want)).astype(np.int64)


def _nodes_near_faces(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    grid: Grid,
    active: np.ndarray,
    within: np.ndarray | None = None,
) -> np.ndarray:
    face_ids = np.unique(np.asarray(face_ids, dtype=np.int64))
    if face_ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    if face_ids[0] < 0 or face_ids[-1] >= len(mesh.faces):
        raise ValueError(f"face ids out of range [0, {len(mesh.faces)})")
    h = grid.h
    tol = FACE_TOL * h
    cand = surface_nodes(grid, active)
    xyz = node_xyz(grid, cand)
    sel_v = np.asarray(mesh.triangles)[face_ids].reshape(-1, 3)
    lo, hi = sel_v.min(0) - tol, sel_v.max(0) + tol
    if within is not None:
        pad = 1e-9 * h
        lo, hi = np.maximum(lo, within[0] - pad), np.minimum(hi, within[1] + pad)
    near = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    cand, xyz = cand[near], xyz[near]
    if cand.size == 0:
        return cand

    # exact distances to every face (selected or not) that can lie within `tol` of a candidate
    tri = np.asarray(mesh.triangles, dtype=np.float64)
    lo, hi = xyz.min(0) - tol, xyz.max(0) + tol
    near_f = np.flatnonzero(np.all((tri.min(1) <= hi) & (tri.max(1) >= lo), axis=1))
    pi, ti, d = _pair_distances(tri[near_f], xyz, tol, SAMPLE_PITCH * h)
    is_sel = np.zeros(len(mesh.faces), dtype=bool)
    is_sel[face_ids] = True
    sel = is_sel[near_f[ti]]
    d_sel = np.full(cand.size, np.inf)
    d_full = np.full(cand.size, np.inf)
    np.minimum.at(d_sel, pi[sel], d[sel])
    np.minimum.at(d_full, pi, d)
    # reject nodes much closer to some other face (e.g. the far side of a thin wall)
    return cand[(d_sel <= tol) & (d_sel <= d_full + OCCLUSION_SLACK * h)]


def _pair_distances(
    tri: np.ndarray, pts: np.ndarray, r: float, pitch: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(point idx, triangle idx, exact distance) for all pairs closer than `r` (plus some more).

    Candidates come from a KD-tree over surface samples: every triangle point is within
    2*pitch of a sample, so querying samples within r + 2*pitch finds every triangle within r.
    """
    if len(tri) == 0 or len(pts) == 0:
        e = np.zeros(0, dtype=np.int64)
        return e, e, np.zeros(0)
    samples, owner = surface_points(tri, pitch)
    pairs = cKDTree(pts).sparse_distance_matrix(
        cKDTree(samples), r + 2 * pitch, output_type="ndarray"
    )
    key = np.unique(pairs["i"].astype(np.int64) * len(tri) + owner[pairs["j"]])
    pi, ti = key // len(tri), key % len(tri)
    d = np.linalg.norm(trimesh.triangles.closest_point(tri[ti], pts[pi]) - pts[pi], axis=1)
    return pi, ti, d


def _primitive_nodes(sel: Mapping, grid: Grid, active: np.ndarray) -> np.ndarray:
    kind = sel["kind"]
    ids = (
        surface_nodes(grid, active) if sel.get("surface_only", True) else active_nodes(grid, active)
    )
    m = transform_matrix(sel.get("transform"))
    if abs(np.linalg.det(m[:3, :3])) < 1e-300:
        raise ValueError("primitive transform is singular")
    minv = np.linalg.inv(m)
    xyz = node_xyz(grid, ids)
    local = xyz @ minv[:3, :3].T + minv[:3, 3]
    size = np.asarray(sel.get("size", (1.0, 1.0, 1.0)), dtype=np.float64)
    eps = 1e-9 * max(float(np.abs(size).max()), 1e-300)
    if kind == "box":
        inside = np.all(np.abs(local) <= np.abs(size) / 2 + eps, axis=1)
    elif kind == "sphere":
        inside = np.linalg.norm(local, axis=1) <= abs(size[0]) + eps
    elif kind == "cylinder":  # axis = local Y (three.js CylinderGeometry)
        r = np.hypot(local[:, 0], local[:, 2])
        inside = (r <= abs(size[0]) + eps) & (np.abs(local[:, 1]) <= abs(size[1]) / 2 + eps)
    else:
        raise ValueError(f"unknown primitive kind {kind!r}")
    return ids[inside]


def _mesh(meshes: Mapping[str, trimesh.Trimesh], sel: Mapping) -> trimesh.Trimesh:
    mesh_id = sel.get("mesh_id")
    if mesh_id not in meshes:
        raise ValueError(f"selection refers to unknown mesh {mesh_id!r}")
    return meshes[mesh_id]


def resolve_selection(
    sel: Mapping,
    grid: Grid,
    active: np.ndarray,
    meshes: Mapping[str, trimesh.Trimesh],
) -> np.ndarray:
    """Sorted unique full-grid node ids (int64) selected by `sel`. May be empty."""
    kind = sel.get("kind")
    active = np.asarray(active, dtype=bool)
    if kind == "faces":
        out = _nodes_near_faces(_mesh(meshes, sel), sel.get("face_ids", []), grid, active)
    elif kind == "facets":
        mesh = _mesh(meshes, sel)
        faces = facet_faces(mesh, float(sel.get("angle_deg", 5.0)), sel.get("facet_ids", []))
        out = _nodes_near_faces(mesh, faces, grid, active)
    elif kind == "normal":
        mesh = _mesh(meshes, sel)
        within = sel.get("within")
        box = None if within is None else np.asarray(within, dtype=np.float64).reshape(2, 3)
        faces = faces_from_normal(mesh, sel["direction"], float(sel.get("angle_deg", 10.0)), box)
        out = _nodes_near_faces(mesh, faces, grid, active, within=box)
    elif kind == "plane":
        n = _unit(sel["normal"], "plane normal")
        p = np.asarray(sel["point"], dtype=np.float64)
        tol = float(sel.get("tol", 0.0) or 0.0)
        # tol 0 -> a slab one voxel edge thick, i.e. the single node layer nearest the plane
        tol = tol if tol > 0 else 0.5 * grid.h * (1 + 1e-9)
        cand = surface_nodes(grid, active)
        out = cand[np.abs((node_xyz(grid, cand) - p) @ n) <= tol]
    elif kind in ("box", "sphere", "cylinder"):
        out = _primitive_nodes(sel, grid, active)
    else:
        raise ValueError(f"unknown selection kind {kind!r}")
    return np.unique(np.asarray(out, dtype=np.int64))


def resolved_preview(node_ids: np.ndarray, grid: Grid, cap: int = 5000) -> dict:
    """`ResolvedNodes` fields; evenly subsampled to `cap` points."""
    ids = np.asarray(node_ids, dtype=np.int64).ravel()
    count = int(ids.size)
    truncated = count > cap
    if truncated:
        ids = ids[np.unique(np.linspace(0, count - 1, cap).round().astype(np.int64))]
    return {"count": count, "xyz": node_xyz(grid, ids).tolist(), "truncated": truncated}
