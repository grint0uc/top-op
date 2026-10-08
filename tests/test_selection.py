from __future__ import annotations

import numpy as np
import pytest
import trimesh

from topop.core.problem import Grid
from topop.core.selection import (
    compute_facets,
    faces_from_normal,
    node_xyz,
    resolve_selection,
    resolved_preview,
    surface_nodes,
)
from topop.core.voxelize import build_domain, load_mesh
from topop.server.schemas import (
    FacetInfo,
    FacetSelection,
    NormalSelection,
    PlaneSelection,
    PrimitiveSelection,
    ResolvedNodes,
)

BOTTOM_AREA = 80 * 60 - 4 * np.pi * 9


@pytest.fixture(scope="module")
def cant(examples_dir):
    mesh = load_mesh(examples_dir / "cantilever.stl")
    grid, active, _, _ = build_domain(mesh, [], 30)
    return mesh, grid, active, {"c": mesh}


@pytest.fixture(scope="module")
def bracket(examples_dir):
    mesh = load_mesh(examples_dir / "bracket.stl")
    grid, active, _, _ = build_domain(mesh, [], 80)
    return mesh, grid, active, {"b": mesh}


def colmajor(m: np.ndarray) -> list[float]:
    return m.ravel(order="F").tolist()


def resolve(ctx, sel) -> np.ndarray:
    _, grid, active, meshes = ctx
    if hasattr(sel, "model_dump"):
        sel = sel.model_dump()
    ids = resolve_selection(sel, grid, active, meshes)
    assert ids.dtype == np.int64
    assert np.array_equal(ids, np.unique(ids))
    return ids


def test_surface_nodes_counts(cant):
    _, grid, active, _ = cant
    nodes = surface_nodes(grid, active)
    assert nodes.size == 31 * 11 * 11 - 29 * 9 * 9

    hollow = np.ones((4, 4, 4), dtype=bool)
    hollow[1:3, 1:3, 1:3] = False
    g = Grid(origin=(0.0, 0.0, 0.0), h=1.0, shape=(4, 4, 4))
    ids = surface_nodes(g, hollow)
    assert ids.size == 125 - 1  # only the node in the middle of the cavity is untouched
    assert g.node_ids(2, 2, 2) not in ids


def test_normal_top_face(cant):
    _, grid, _, _ = cant
    ids = resolve(cant, NormalSelection(mesh_id="c", direction=[0, 0, 1]))
    assert ids.size == 31 * 11
    assert np.allclose(node_xyz(grid, ids)[:, 2], 20, atol=1e-9)
    for d, n in [([1, 0, 0], 121), ([0, -1, 0], 341), ([0, 0, -1], 341)]:
        assert resolve(cant, NormalSelection(mesh_id="c", direction=d)).size == n


def test_normal_within_clips_nodes(cant):
    _, grid, _, _ = cant
    sel = NormalSelection(mesh_id="c", direction=[0, 0, 1], within=[[50, -1, 19], [60, 21, 21]])
    xyz = node_xyz(grid, resolve(cant, sel))
    assert len(xyz) == 6 * 11
    assert xyz[:, 0].min() == pytest.approx(50) and np.allclose(xyz[:, 2], 20)


def test_faces_equals_normal(cant):
    mesh = cant[0]
    top = faces_from_normal(mesh, [0, 0, 1], 1.0)
    assert top.size == 2 and np.allclose(mesh.face_normals[top], [0, 0, 1])
    by_faces = resolve(cant, {"kind": "faces", "mesh_id": "c", "face_ids": top.tolist()})
    by_normal = resolve(cant, {"kind": "normal", "mesh_id": "c", "direction": [0, 0, 1]})
    assert np.array_equal(by_faces, by_normal)


def test_faces_on_thin_plate_skip_far_side():
    plate = trimesh.creation.box(extents=(20, 20, 1))  # exactly one element thick at h=1
    grid, active, _, _ = build_domain(plate, [], 20, padding=1)
    assert grid.h == pytest.approx(1.0) and active.sum() == 400
    ctx = (plate, grid, active, {"p": plate})
    ids = resolve(ctx, {"kind": "normal", "mesh_id": "p", "direction": [0, 0, 1]})
    assert ids.size == 21 * 21
    assert np.allclose(node_xyz(grid, ids)[:, 2], 0.5)


def test_plane(cant):
    _, grid, _, _ = cant
    ids = resolve(cant, PlaneSelection(point=[0, 0, 0], normal=[1, 0, 0]))
    assert ids.size == 11 * 11
    assert np.allclose(node_xyz(grid, ids)[:, 0], 0)
    # explicit tolerance reaches the next node layer (only its surface nodes)
    assert resolve(cant, PlaneSelection(point=[0, 0, 0], normal=[-2, 0, 0], tol=2.5)).size == 161
    with pytest.raises(ValueError):
        resolve(cant, PlaneSelection(point=[0, 0, 0], normal=[0, 0, 0]))


def test_box_surface_only(cant):
    _, grid, _, _ = cant
    t = colmajor(trimesh.transformations.translation_matrix([60, 10, 10]))
    sel = PrimitiveSelection(kind="box", transform=t, size=[5, 30, 30])
    surf = resolve(cant, sel)
    assert surf.size == 121 + 40  # x=60 face + perimeter ring of the x=58 layer
    assert set(np.unique(node_xyz(grid, surf)[:, 0])) == {58.0, 60.0}
    thin = resolve(cant, PrimitiveSelection(kind="box", transform=t, size=[1, 30, 30]))
    assert thin.size == 121
    sel.surface_only = False
    assert resolve(cant, sel).size == 2 * 121


def test_cylinder_axis_is_local_y(cant):
    _, grid, _, _ = cant
    m = trimesh.transformations.translation_matrix([30, 10, 20]) @ (
        trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])  # local Y -> world Z
    )
    sel = PrimitiveSelection(kind="cylinder", transform=colmajor(m), size=[5, 4, 0])
    xyz = node_xyz(grid, resolve(cant, sel))
    assert len(xyz) == 21  # lattice points (step 2) within r=5 on the top face
    assert np.allclose(xyz[:, 2], 20)
    assert np.all(np.hypot(xyz[:, 0] - 30, xyz[:, 1] - 10) <= 5 + 1e-9)
    # unrotated: axis along world Y through the beam, radius 3 around (30, *, 10)
    t = colmajor(trimesh.transformations.translation_matrix([30, 10, 10]))
    sel = PrimitiveSelection(kind="cylinder", transform=t, size=[3, 100, 0], surface_only=False)
    xyz = node_xyz(grid, resolve(cant, sel))
    assert np.all(np.hypot(xyz[:, 0] - 30, xyz[:, 2] - 10) <= 3 + 1e-9)
    assert len(np.unique(xyz[:, 1])) == 11


def test_sphere(cant):
    sel = PrimitiveSelection(kind="sphere", size=[3, 0, 0])  # identity transform: at the corner
    assert resolve(cant, sel).size == 7


def test_compute_facets_bracket(bracket):
    mesh = bracket[0]
    facets, f2f = compute_facets(mesh)
    for f in facets[:5]:
        FacetInfo(**f)
    assert [f["id"] for f in facets] == list(range(len(facets)))
    areas = [f["area"] for f in facets]
    # descending; near-equal areas (within 1e-9 of the total) are ordered by lowest face id
    assert np.all(np.diff(areas) <= 1e-9 * sum(areas))
    assert f2f.shape == (len(mesh.faces),) and sum(f["n_faces"] for f in facets) == len(mesh.faces)
    assert np.allclose(np.bincount(f2f, weights=mesh.area_faces), areas)
    bottom = facets[0]
    assert np.allclose(bottom["normal"], [0, 0, -1])
    assert bottom["area"] == pytest.approx(BOTTOM_AREA, rel=1e-3)
    assert np.allclose(bottom["bbox"], [[0, 0, 0], [80, 60, 0]])
    assert np.allclose(bottom["centroid"][2], 0)

    again, f2f_again = compute_facets(mesh)
    assert again == facets and np.array_equal(f2f, f2f_again)

    # ids survive a rigid transform (the project transform is applied before resolving)
    moved = mesh.copy()
    moved.apply_transform(
        trimesh.transformations.translation_matrix([3, -7, 11])
        @ trimesh.transformations.rotation_matrix(0.9, [1, 1, 0])
    )
    assert np.array_equal(compute_facets(moved)[1], f2f)
    coarse, _ = compute_facets(mesh, angle_deg=10)
    assert len(coarse) < len(facets)  # Ø6 holes (5.6° between sections) merge at 10°


def test_facet_selection_bottom(bracket):
    _, grid, _, _ = bracket
    ids = resolve(bracket, FacetSelection(mesh_id="b", facet_ids=[0]))
    xyz = node_xyz(grid, ids)
    assert ids.size > 4000
    assert np.allclose(xyz[:, 2], 0)
    assert xyz[:, 0].max() == pytest.approx(80) and xyz[:, 1].max() == pytest.approx(60)
    # the plate holes are not grabbed
    assert np.all(np.hypot(xyz[:, 0] - 20, xyz[:, 1] - 8) >= 3 - 1.0)
    with pytest.raises(ValueError):
        resolve(bracket, FacetSelection(mesh_id="b", facet_ids=[10_000]))


def test_empty_and_invalid_selections(cant):
    assert resolve(cant, {"kind": "faces", "mesh_id": "c", "face_ids": []}).size == 0
    tilted = NormalSelection(mesh_id="c", direction=[1, 1, 1], angle_deg=1)
    assert resolve(cant, tilted).size == 0
    far = colmajor(trimesh.transformations.translation_matrix([500, 0, 0]))
    assert resolve(cant, PrimitiveSelection(kind="box", transform=far)).size == 0
    with pytest.raises(ValueError):
        resolve(cant, {"kind": "faces", "mesh_id": "nope", "face_ids": [0]})
    with pytest.raises(ValueError):
        resolve(cant, {"kind": "faces", "mesh_id": "c", "face_ids": [12]})
    with pytest.raises(ValueError):
        resolve(cant, {"kind": "blob"})


def test_resolved_preview(bracket):
    _, grid, active, _ = bracket
    ids = surface_nodes(grid, active)
    assert ids.size > 5000
    prev = resolved_preview(ids, grid)
    ResolvedNodes(**prev)
    assert prev["count"] == ids.size and prev["truncated"] is True
    assert len(prev["xyz"]) == 5000
    assert np.allclose(prev["xyz"][0], node_xyz(grid, ids[:1])[0])
    assert np.allclose(prev["xyz"][-1], node_xyz(grid, ids[-1:])[0])

    small = resolved_preview(ids[:10], grid)
    assert small == {"count": 10, "xyz": node_xyz(grid, ids[:10]).tolist(), "truncated": False}
    assert resolved_preview(np.zeros(0, dtype=np.int64), grid)["count"] == 0
