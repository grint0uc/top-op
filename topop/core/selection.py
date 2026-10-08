"""Resolve `schemas.Selection` mappings to full-grid node ids. Pure numpy/trimesh, no pydantic.

`sel` is a plain mapping with the same keys as the schema models (`sel["kind"]`, ...).
Meshes passed to `resolve_selection` are already in world coordinates, keyed by mesh_id.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import trimesh
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


def compute_facets(mesh: trimesh.Trimesh, angle_deg: float = 5.0) -> tuple[list[dict], np.ndarray]:
    """Coplanar face groups as `FacetInfo` dicts (sorted by area, id = rank) + face_to_facet.

    Order: area descending (rounded to 1e-9 of the total area), ties by lowest face id. Both keys
    are invariant under rigid transforms, so ids listed for a raw mesh stay valid after the
    project transform is applied.
    """
    n = len(mesh.faces)
    if n == 0:
        return [], np.zeros(0, dtype=np.int64)
    adj = np.asarray(mesh.face_adjacency)
    if len(adj):
        adj = adj[np.asarray(mesh.face_adjacency_angles) <= np.radians(float(angle_deg))]
    label = np.asarray(
        trimesh.graph.connected_component_labels(adj.reshape(-1, 2), node_count=n), dtype=np.int64
    )
    ng = int(label.max()) + 1

    area = np.asarray(mesh.area_faces, dtype=np.float64)
    g_area = np.bincount(label, weights=area, minlength=ng)
    g_nf = np.bincount(label, minlength=ng)
    wn = np.stack(
        [
            np.bincount(label, weights=area * mesh.face_normals[:, i], minlength=ng)
            for i in range(3)
        ],
        1,
    )
    norm = np.linalg.norm(wn, axis=1, keepdims=True)
    # a closed curved group (e.g. a finely tessellated sphere) has no meaningful normal -> 0
    g_normal = np.divide(wn, norm, out=np.zeros_like(wn), where=norm > 1e-6 * g_area[:, None])
    tc = np.asarray(mesh.triangles_center)
    safe = np.where(g_area > 0, g_area, 1.0)[:, None]
    g_centroid = (
        np.stack([np.bincount(label, weights=area * tc[:, i], minlength=ng) for i in range(3)], 1)
        / safe
    )
    tri = np.asarray(mesh.triangles)
    g_min = np.full((ng, 3), np.inf)
    g_max = np.full((ng, 3), -np.inf)
    np.minimum.at(g_min, label, tri.min(1))
    np.maximum.at(g_max, label, tri.max(1))
    first_face = np.full(ng, n, dtype=np.int64)
    np.minimum.at(first_face, label, np.arange(n))

    total = max(float(g_area.sum()), np.finfo(float).tiny)
    order = np.lexsort((first_face, -np.round(g_area / total, 9)))
    rank = np.empty(ng, dtype=np.int64)
    rank[order] = np.arange(ng)
    facets = [
        {
            "id": int(r),
            "n_faces": int(g_nf[g]),
            "area": float(g_area[g]),
            "normal": g_normal[g].tolist(),
            "centroid": g_centroid[g].tolist(),
            "bbox": [g_min[g].tolist(), g_max[g].tolist()],
        }
        for r, g in enumerate(order)
    ]
    return facets, rank[label]


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
        facets, f2f = compute_facets(mesh, float(sel.get("angle_deg", 5.0)))
        want = np.asarray(sel.get("facet_ids", []), dtype=np.int64)
        bad = want[(want < 0) | (want >= len(facets))]
        if bad.size:
            raise ValueError(f"unknown facet ids {bad.tolist()} (mesh has {len(facets)} facets)")
        out = _nodes_near_faces(mesh, np.flatnonzero(np.isin(f2f, want)), grid, active)
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
