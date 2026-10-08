from __future__ import annotations

import numpy as np
import pytest
import trimesh

from topop.core.problem import Grid
from topop.core.selection import (
    FACE_TOL,
    compute_facets,
    faces_from_normal,
    facet_faces,
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

HOLES = 4 * np.pi * 3**2  # four Ø6 plate holes
BOTTOM_AREA = 80 * 60 - HOLES
TOP_AREA = 80 * 60 - 10 * 60 - HOLES  # the wall stands on the plate's x < 10 strip


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


def _kinds(facets: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in facets:
        out[f["kind"]] = out.get(f["kind"], 0) + 1
    return out


def _parallel(a, b, tol_deg: float = 1.0) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)) >= np.cos(np.radians(tol_deg))


def test_compute_facets_bracket(bracket):
    mesh = bracket[0]
    facets, f2f = compute_facets(mesh)
    for f in facets:
        FacetInfo(**f)
    assert [f["id"] for f in facets] == list(range(len(facets)))
    areas = [f["area"] for f in facets]
    # descending; near-equal areas (within 1e-9 of the total) are ordered by lowest face id
    assert np.all(np.diff(areas) <= 1e-9 * sum(areas))
    assert f2f.shape == (len(mesh.faces),) and sum(f["n_faces"] for f in facets) == len(mesh.faces)
    assert np.allclose(np.bincount(f2f, weights=mesh.area_faces), areas)
    # 8 flat faces (plate bottom/top/end, wall outside/inside/top, two L-shaped sides) + 5 holes
    assert _kinds(facets) == {"plane": 8, "cylinder": 5}

    bottom, top = facets[0], facets[1]
    assert bottom["kind"] == top["kind"] == "plane"
    assert np.allclose(bottom["normal"], [0, 0, -1]) and np.allclose(top["normal"], [0, 0, 1])
    assert bottom["area"] == pytest.approx(BOTTOM_AREA, rel=1e-3)
    assert top["area"] == pytest.approx(TOP_AREA, rel=1e-3)
    assert np.allclose(bottom["bbox"], [[0, 0, 0], [80, 60, 0]])
    assert np.allclose(top["bbox"], [[10, 0, 10], [80, 60, 10]])
    assert np.allclose(bottom["centroid"][2], 0)
    for f in facets:
        if f["kind"] == "plane":
            assert f["axis"] is None and f["radius"] is None
            assert np.linalg.norm(f["normal"]) == pytest.approx(1.0)

    cyl = [f for f in facets if f["kind"] == "cylinder"]
    for f in cyl:
        assert f["normal"] == [0.0, 0.0, 0.0]
        assert np.linalg.norm(f["axis"]) == pytest.approx(1.0)
        assert max(f["axis"], key=abs) > 0  # sign: towards +x/+y/+z
    wall = [f for f in cyl if f["radius"] == pytest.approx(6, abs=0.1)]
    assert len(wall) == 1
    assert _parallel(wall[0]["axis"], [1, 0, 0])
    assert np.allclose(wall[0]["centroid"], [5, 30, 35], atol=0.05)
    plate = sorted(
        (f for f in cyl if f["radius"] == pytest.approx(3, abs=0.1)), key=lambda f: f["centroid"]
    )
    assert len(plate) == 4
    for f, (x, y) in zip(plate, [(20, 8), (20, 52), (72, 8), (72, 52)]):
        assert _parallel(f["axis"], [0, 0, 1])
        assert np.allclose(f["centroid"], [x, y, 5], atol=0.05)
        assert f["area"] == pytest.approx(2 * np.pi * 3 * 10, rel=1e-2)

    again, f2f_again = compute_facets(mesh)
    assert again == facets and np.array_equal(f2f, f2f_again)

    # ids survive a rigid transform (the project transform is applied before resolving)
    moved = mesh.copy()
    moved.apply_transform(
        trimesh.transformations.translation_matrix([3, -7, 11])
        @ trimesh.transformations.rotation_matrix(0.9, [1, 1, 0])
    )
    assert np.array_equal(compute_facets(moved)[1], f2f)
    # the holes are one cylinder each at any angle (the old pure chaining split the Ø6 holes,
    # 5.6° between sections, into 64 strips at 5° and merged them at 10°)
    coarse, f2f_coarse = compute_facets(mesh, angle_deg=10)
    assert _kinds(coarse) == {"plane": 8, "cylinder": 5}
    assert np.array_equal(f2f_coarse, f2f)


def test_facets_capsule():
    """Barrel = one cylinder; the hemispheres stay apart from it and are one 'other' each."""
    tilt = trimesh.transformations.rotation_matrix(0.7, [1, 2, 0])
    mesh = trimesh.creation.capsule(height=2.0, radius=1.0, transform=tilt)  # 64 x 32 sections
    facets, _ = compute_facets(mesh)
    for f in facets:
        FacetInfo(**f)
    assert _kinds(facets) == {"cylinder": 1, "other": 2}
    (barrel,) = [f for f in facets if f["kind"] == "cylinder"]
    assert barrel["radius"] == pytest.approx(1.0, abs=1e-3)
    assert _parallel(barrel["axis"], tilt[:3, 2], 0.1)
    assert np.allclose(barrel["centroid"], 0, atol=1e-6)
    assert barrel["area"] == pytest.approx(2 * np.pi * 1.0 * 2.0, rel=1e-2)
    # only barrel faces: every vertex lies within the straight part
    faces = facet_faces(mesh, 5.0, [barrel["id"]])
    z = (mesh.vertices[mesh.faces[faces]] @ tilt[:3, 2]).ravel()
    assert np.all(np.abs(z) <= 1.0 + 1e-9)
    for cap in (f for f in facets if f["kind"] == "other"):
        assert cap["area"] == pytest.approx(2 * np.pi, rel=2e-2)
        assert cap["normal"] == [0.0, 0.0, 0.0] and cap["axis"] is None


def test_facets_icosphere():
    mesh = trimesh.creation.icosphere(4)
    facets, _ = compute_facets(mesh)
    assert len(facets) <= 3
    biggest_triangle = mesh.area_faces.max()
    assert all(f["area"] <= biggest_triangle * (1 + 1e-9) for f in facets if f["kind"] == "plane")
    assert facets[0]["kind"] == "other" and facets[0]["area"] > 0.99 * mesh.area


def _rounded_edge_box(a=40.0, b=30.0, c=20.0, r=5.0, sections=64) -> trimesh.Trimesh:
    """Box [0,a]x[0,b]x[0,c] whose top-back edge (y=b, z=c, along x) is a tangent fillet r."""
    lower = trimesh.creation.box(extents=(a, b, c - r))
    lower.apply_translation((a / 2, b / 2, (c - r) / 2))
    front = trimesh.creation.box(extents=(a, b - r, c))
    front.apply_translation((a / 2, (b - r) / 2, c / 2))
    rod = trimesh.creation.cylinder(radius=r, height=a, sections=sections)
    rod.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    rod.apply_translation((a / 2, b - r, c - r))
    return trimesh.boolean.union([lower, front, rod], engine="manifold")


@pytest.mark.parametrize("sections", [32, 64, 128])
def test_facets_rounded_edge(sections):
    """The fillet does not chain the top and back faces together, and is one cylinder."""
    mesh = _rounded_edge_box(sections=sections)
    assert mesh.is_watertight
    facets, _ = compute_facets(mesh)
    assert _kinds(facets) == {"plane": 6, "cylinder": 1}
    planes = {tuple(np.round(f["normal"], 9)): f for f in facets if f["kind"] == "plane"}
    # 128 sections: the first fillet strip is 1.4° off the tangent planes, i.e. inside the 5°
    # tolerance of Phase A, and is handed back to the cylinder by the model fit
    assert planes[(0, 0, 1)]["area"] == pytest.approx(40 * 25, rel=1e-9)
    assert planes[(0, 1, 0)]["area"] == pytest.approx(40 * 15, rel=1e-9)
    (rnd,) = [f for f in facets if f["kind"] == "cylinder"]
    assert rnd["radius"] == pytest.approx(5, abs=0.05)
    assert _parallel(rnd["axis"], [1, 0, 0])
    assert np.allclose(rnd["centroid"], [20, 25, 15], atol=1e-6)
    assert rnd["area"] == pytest.approx(np.pi / 2 * 5 * 40, rel=1e-2)


def test_facets_s_curve():
    """A convex and a concave fillet meeting tangentially are two cylinders, not one."""
    from manifold3d import CrossSection

    def arc(cx, cy, a0, a1):
        t = np.radians(np.linspace(a0, a1, 17))[1:]
        return np.stack([cx + 10 * np.cos(t), cy + 10 * np.sin(t)], 1).tolist()

    profile = [[0, 0], [100, 0], [100, 10], [70, 10], *arc(70, 20, -90, -180), *arc(50, 20, 0, 90)]
    m = CrossSection([[*profile, [0, 30]]]).extrude(30).to_mesh()
    mesh = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts))
    facets, _ = compute_facets(mesh)
    cyl = sorted((f for f in facets if f["kind"] == "cylinder"), key=lambda f: f["centroid"])
    assert len(cyl) == 2
    for f, x in zip(cyl, (50, 70)):
        assert f["radius"] == pytest.approx(10, abs=0.1)
        assert np.allclose(f["centroid"], [x, 20, 15], atol=1e-6)
    assert sum(f["area"] for f in facets if f["kind"] == "other") < 0.1 * cyl[0]["area"]


def test_facet_faces(bracket):
    mesh = bracket[0]
    facets, f2f = compute_facets(mesh)
    (wall,) = [f for f in facets if f["kind"] == "cylinder" and f["radius"] > 5]
    faces = facet_faces(mesh, 5.0, [wall["id"]])
    assert faces.dtype == np.int64 and np.array_equal(faces, np.flatnonzero(f2f == wall["id"]))
    assert faces.size == wall["n_faces"]
    v = mesh.vertices[mesh.faces[faces]].reshape(-1, 3)
    assert np.allclose(np.hypot(v[:, 1] - 30, v[:, 2] - 35), 6, atol=1e-3)
    assert np.allclose(mesh.face_normals[faces][:, 0], 0, atol=1e-6)
    both = facet_faces(mesh, 5.0, [0, wall["id"]])
    assert np.array_equal(both, np.flatnonzero(np.isin(f2f, [0, wall["id"]])))
    assert facet_faces(mesh, 5.0, []).size == 0
    with pytest.raises(ValueError):
        facet_faces(mesh, 5.0, [len(facets)])
    # the cached segmentation is not mutated through the returned copies
    facets[0]["normal"][0] = 99.0
    f2f[:] = -1
    again, f2f_again = compute_facets(mesh)
    assert again[0]["normal"] == [0.0, 0.0, -1.0] and f2f_again.min() == 0


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


def test_facet_selection_cylinder(bracket):
    """The wall hole picked as one facet: nodes on its bore, nothing from the wall faces."""
    _, grid, _, _ = bracket
    facets, _ = compute_facets(bracket[0])
    (wall,) = [f for f in facets if f["kind"] == "cylinder" and f["radius"] > 5]
    xyz = node_xyz(grid, resolve(bracket, FacetSelection(mesh_id="b", facet_ids=[wall["id"]])))
    assert len(xyz) > 50
    r = np.hypot(xyz[:, 1] - 30, xyz[:, 2] - 35)
    assert np.all(np.abs(r - 6) <= FACE_TOL * grid.h)
    assert xyz[:, 0].min() >= -1e-9 and xyz[:, 0].max() <= 10 + 1e-9


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
