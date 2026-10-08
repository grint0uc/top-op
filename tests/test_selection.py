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
    _assert_invariant(mesh, facets, f2f)
    # the holes are one cylinder each at any angle (the old pure chaining split the Ø6 holes,
    # 5.6° between sections, into 64 strips at 5° and merged them at 10°)
    coarse, f2f_coarse = compute_facets(mesh, angle_deg=10)
    assert _kinds(coarse) == {"plane": 8, "cylinder": 5}
    assert np.array_equal(f2f_coarse, f2f)


def test_facets_capsule():
    """Barrel = one cylinder; the hemispheres stay apart from it and are one sphere each."""
    tilt = trimesh.transformations.rotation_matrix(0.7, [1, 2, 0])
    mesh = trimesh.creation.capsule(height=2.0, radius=1.0, transform=tilt)  # 64 x 32 sections
    facets, _ = compute_facets(mesh)
    for f in facets:
        FacetInfo(**f)
    assert _kinds(facets) == {"cylinder": 1, "sphere": 2}
    (barrel,) = [f for f in facets if f["kind"] == "cylinder"]
    assert barrel["radius"] == pytest.approx(1.0, abs=1e-3)
    assert _parallel(barrel["axis"], tilt[:3, 2], 0.1)
    assert np.allclose(barrel["centroid"], 0, atol=1e-6)
    assert barrel["area"] == pytest.approx(2 * np.pi * 1.0 * 2.0, rel=1e-2)
    # only barrel faces: every vertex lies within the straight part
    faces = facet_faces(mesh, 5.0, [barrel["id"]])
    z = (mesh.vertices[mesh.faces[faces]] @ tilt[:3, 2]).ravel()
    assert np.all(np.abs(z) <= 1.0 + 1e-9)
    caps = sorted(
        (f for f in facets if f["kind"] == "sphere"), key=lambda f: f["centroid"] @ tilt[:3, 2]
    )
    for cap, z in zip(caps, (-1.0, 1.0)):  # centres at -+height/2 on the axis
        assert cap["area"] == pytest.approx(2 * np.pi, rel=2e-2)
        assert cap["radius"] == pytest.approx(1.0, abs=1e-3)
        assert np.allclose(cap["centroid"], z * tilt[:3, 2], atol=1e-3)
        assert cap["normal"] == [0.0, 0.0, 0.0] and cap["axis"] is None


def test_facets_icosphere():
    mesh = trimesh.creation.icosphere(4)
    facets, _ = compute_facets(mesh)
    assert [f["kind"] for f in facets] == ["sphere"]
    (ball,) = facets
    FacetInfo(**ball)
    assert ball["n_faces"] == len(mesh.faces) and ball["area"] == pytest.approx(mesh.area)
    assert ball["radius"] == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(ball["centroid"], 0, atol=1e-9)
    assert ball["normal"] == [0.0, 0.0, 0.0] and ball["axis"] is None


@pytest.mark.parametrize("count", [[16, 16], [10, 32], [8, 16]])
def test_facets_uv_sphere(count):
    """[10, 32] has 20 deg latitude steps between bands of unequal area, which chain neither as
    smooth nor as regular strips: the fitted sphere grabs the rows that lie on it."""
    mesh = trimesh.creation.uv_sphere(radius=2.0, count=count)
    mesh.apply_translation([1, 2, 3])
    facets, _ = compute_facets(mesh)
    assert [f["kind"] for f in facets] == ["sphere"]
    assert facets[0]["radius"] == pytest.approx(2.0, abs=1e-6)
    assert np.allclose(facets[0]["centroid"], [1, 2, 3], atol=1e-6)


def test_facets_spherical_pocket_and_dome():
    """Concave (normals towards the centre) and shallow (a +-20 deg cap) spheres."""
    box = trimesh.creation.box(extents=(30, 30, 30))
    ball = trimesh.creation.icosphere(3, radius=10)
    ball.apply_translation((0, 0, 15))
    pocket = trimesh.boolean.difference([box, ball], engine="manifold")
    h = 10 * np.cos(np.radians(20))
    below = trimesh.creation.box(extents=(30, 30, 30))
    below.apply_translation((0, 0, h - 15))
    cap = trimesh.boolean.difference(
        [trimesh.creation.icosphere(4, radius=10), below], engine="manifold"
    )
    slab = trimesh.creation.box(extents=(30, 30, 10))
    slab.apply_translation((0, 0, h - 5))
    dome = trimesh.boolean.union([slab, cap], engine="manifold")
    for mesh, center in ((pocket, (0, 0, 15)), (dome, (0, 0, 0))):
        facets, _ = compute_facets(mesh)
        assert _kinds(facets) == {"plane": 6, "sphere": 1}
        (sph,) = [f for f in facets if f["kind"] == "sphere"]
        assert sph["radius"] == pytest.approx(10, rel=2e-2)
        assert np.allclose(sph["centroid"], center, atol=0.2)


def test_facets_torus_and_cone_are_no_spheres():
    """Any two coaxial circles lie on a sphere, so torus bands and cones pass a residual test;
    they are singly curved (rank-1 bending) and stay 'other'."""
    facets, _ = compute_facets(trimesh.creation.torus(major_radius=5, minor_radius=1))
    assert _kinds(facets) == {"other": 1}
    coarse, _ = compute_facets(trimesh.creation.torus(5, 1, 16, 8))  # 22.5 deg minor steps
    assert set(_kinds(coarse)) == {"other"}
    facets, _ = compute_facets(trimesh.creation.cone(radius=3, height=6))
    assert _kinds(facets) == {"other": 1, "plane": 1}
    (base,) = [f for f in facets if f["kind"] == "plane"]
    assert np.allclose(base["normal"], [0, 0, -1]) and base["area"] == pytest.approx(
        0.5 * 32 * 9 * np.sin(2 * np.pi / 32)
    )


@pytest.mark.parametrize("sections", [10, 12, 16, 24])
def test_facets_coarse_cylinder(sections):
    """Section steps above 3 * angle_deg (up to 36 deg at 10 sections) are one cylinder."""
    tilt = trimesh.transformations.rotation_matrix(0.4, [1, -1, 2])
    mesh = trimesh.creation.cylinder(radius=5, height=20, sections=sections, transform=tilt)
    facets, _ = compute_facets(mesh)
    assert _kinds(facets) == {"cylinder": 1, "plane": 2}
    (cyl,) = [f for f in facets if f["kind"] == "cylinder"]
    assert cyl["n_faces"] == 2 * sections
    assert cyl["radius"] == pytest.approx(5, abs=1e-6)
    assert _parallel(cyl["axis"], tilt[:3, 2], 1e-3)
    assert np.allclose(cyl["centroid"], 0, atol=1e-6)


@pytest.mark.parametrize("sections", [6, 8, 9])
def test_facets_polygonal_prism_stays_planar(sections):
    """40 deg steps (9 sections) and coarser are a polygonal prism, one plane per side: the
    coarse-tessellation limit is _REGULAR_MAX_STEP = 37 deg, i.e. >= 10 sections per turn."""
    mesh = trimesh.creation.cylinder(radius=5, height=20, sections=sections)
    assert _kinds(compute_facets(mesh)[0]) == {"plane": sections + 2}


def _prism(angles: np.ndarray, r: float = 5.0, h: float = 20.0) -> trimesh.Trimesh:
    """Closed prism over the polygon with vertices at `angles` on a circle (cap fans)."""
    k = len(angles)
    ring = np.stack([r * np.cos(angles), r * np.sin(angles)], 1)
    lo, hi = np.c_[ring, np.full(k, -h / 2)], np.c_[ring, np.full(k, h / 2)]
    v = np.concatenate([lo, hi, [[0, 0, -h / 2], [0, 0, h / 2]]])
    f = []
    for i in range(k):
        j = (i + 1) % k
        f += [[i, j, k + j], [i, k + j, k + i], [2 * k, j, i], [2 * k + 1, k + i, k + j]]
    return trimesh.Trimesh(v, np.array(f))


def test_facets_cylinder_seam_merges():
    """Half the barrel in 22.5 deg steps, half in 11.25: the step between the halves is neither
    smooth nor regular, so they are two regions, merged as one cylinder (same axis and radius)."""
    ang = np.concatenate([np.linspace(0, np.pi, 9)[:-1], np.linspace(np.pi, 2 * np.pi, 17)[:-1]])
    mesh = _prism(ang)
    assert mesh.is_watertight
    facets, _ = compute_facets(mesh)
    assert _kinds(facets) == {"cylinder": 1, "plane": 2}
    (cyl,) = [f for f in facets if f["kind"] == "cylinder"]
    assert cyl["n_faces"] == 2 * 24 and cyl["radius"] == pytest.approx(5, abs=1e-6)
    assert _parallel(cyl["axis"], [0, 0, 1], 1e-3)


def _extrude(profile: list, depth: float = 30.0) -> trimesh.Trimesh:
    from manifold3d import CrossSection

    m = CrossSection([profile]).extrude(depth).to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts))


def test_facets_chamfer_and_feature_edges():
    """Flats with equal steps are a tessellated curve only if they are equally wide and bend the
    same way: a chamfer, a double chamfer and a corrugated sheet stay planes."""
    box = trimesh.creation.box(extents=(40, 30, 20))
    cut = trimesh.creation.box(extents=(60, 10 * np.sqrt(2), 10 * np.sqrt(2)))
    cut.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [1, 0, 0]))
    cut.apply_translation((0, 15, 10))
    facets, _ = compute_facets(trimesh.boolean.difference([box, cut], engine="manifold"))
    assert _kinds(facets) == {"plane": 7}
    area = {tuple(np.round(f["normal"], 6)): f["area"] for f in facets}
    s = round(np.sqrt(0.5), 6)
    assert area[(0, 0, 1)] == pytest.approx(40 * 20)  # flat, chamfer, flat
    assert area[(0, s, s)] == pytest.approx(40 * 10 * np.sqrt(2))
    assert area[(0, 1, 0)] == pytest.approx(40 * 10)

    # widths 30, 8, 30 at two 20 deg steps
    d = [np.array([np.cos(np.radians(a)), -np.sin(np.radians(a))]) for a in (0, 20, 40)]
    p1 = np.array([0.0, 30.0]) + 30 * d[0]
    p2 = p1 + 8 * d[1]
    p3 = p2 + 30 * d[2]
    profile = [[0, 0], [p3[0], 0], p3.tolist(), p2.tolist(), p1.tolist(), [0, 30]]
    assert _kinds(compute_facets(_extrude(profile))[0]) == {"plane": 8}

    # 12 equal flats at +-15 deg: 30 deg steps, alternately convex and concave
    x = np.arange(13) * 10.0
    y = 5 + np.where(np.arange(13) % 2, 10 * np.tan(np.radians(15)), 0.0)
    profile = [[0, 0], [120, 0], *np.stack([x, y], 1)[::-1].tolist()]
    assert _kinds(compute_facets(_extrude(profile, 40))[0]) == {"plane": 12 + 5}


def _assert_invariant(mesh: trimesh.Trimesh, facets: list[dict], f2f: np.ndarray) -> None:
    """Same facets after a uniform scale (same ids) and a face shuffle (same partition)."""
    scaled = mesh.copy()
    scaled.apply_scale(37.5)
    s_facets, s_f2f = compute_facets(scaled)
    assert np.array_equal(s_f2f, f2f)
    assert [f["kind"] for f in s_facets] == [f["kind"] for f in facets]
    for a, b in zip(s_facets, facets):
        assert (a["radius"] is None) == (b["radius"] is None)
        if a["radius"] is not None:
            assert a["radius"] == pytest.approx(37.5 * b["radius"], rel=1e-9)
    perm = np.random.default_rng(0).permutation(len(mesh.faces))
    shuffled = trimesh.Trimesh(mesh.vertices, np.asarray(mesh.faces)[perm], process=False)
    sh_facets, sh_f2f = compute_facets(shuffled)
    back = np.empty_like(sh_f2f)
    back[perm] = sh_f2f  # facet of each original face
    pairs = np.unique(np.stack([back, f2f], 1), axis=0)
    assert len(pairs) == len(facets) == len(sh_facets)
    assert sorted((f["kind"], round(f["area"], 6)) for f in sh_facets) == sorted(
        (f["kind"], round(f["area"], 6)) for f in facets
    )


@pytest.mark.parametrize("shape", ["cylinder16", "seam", "uv_sphere", "capsule", "torus"])
def test_facets_invariant(shape):
    mesh = {
        "cylinder16": lambda: trimesh.creation.cylinder(radius=5, height=20, sections=16),
        "seam": lambda: _prism(
            np.concatenate([np.linspace(0, np.pi, 9)[:-1], np.linspace(np.pi, 2 * np.pi, 17)[:-1]])
        ),
        "uv_sphere": lambda: trimesh.creation.uv_sphere(count=[10, 32]),
        "capsule": lambda: trimesh.creation.capsule(height=2.0, radius=1.0),
        "torus": lambda: trimesh.creation.torus(5, 1, 16, 8),
    }[shape]()
    facets, f2f = compute_facets(mesh)
    moved = mesh.copy()
    moved.apply_transform(
        trimesh.transformations.translation_matrix([3, -7, 11])
        @ trimesh.transformations.rotation_matrix(0.9, [1, 1, 0])
    )
    assert np.array_equal(compute_facets(moved)[1], f2f)
    _assert_invariant(mesh, facets, f2f)


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


@pytest.mark.parametrize("sections", [12, 16, 32, 64, 128])
def test_facets_rounded_edge(sections):
    """The fillet does not chain the top and back faces together, and is one cylinder (also at
    30 and 22.5 deg steps: 3 and 4 regular strips)."""
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
    # all of the inscribed quarter polygon (pi / 2 * 5 * 40 minus 1.1 % at 12 sections)
    assert rnd["area"] == pytest.approx(sections / 4 * 10 * np.sin(np.pi / sections) * 40, rel=1e-6)


def test_facets_s_curve():
    """A convex and a concave fillet meeting tangentially are two cylinders, not one."""

    def arc(cx, cy, a0, a1):
        t = np.radians(np.linspace(a0, a1, 17))[1:]
        return np.stack([cx + 10 * np.cos(t), cy + 10 * np.sin(t)], 1).tolist()

    profile = [[0, 0], [100, 0], [100, 10], [70, 10], *arc(70, 20, -90, -180), *arc(50, 20, 0, 90)]
    facets, _ = compute_facets(_extrude([*profile, [0, 30]]))
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
