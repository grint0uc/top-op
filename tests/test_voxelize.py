from __future__ import annotations

import numpy as np
import pytest
import trimesh

from topop.core.problem import Grid
from topop.core.voxelize import (
    apply_transform,
    build_domain,
    domain_stats,
    load_mesh,
    mesh_info,
    transform_matrix,
    voxelize_mesh,
)
from topop.server.schemas import MeshInfo, VoxelStats

BRACKET_VOLUME = 75741.7


@pytest.fixture(scope="module")
def cantilever(examples_dir) -> trimesh.Trimesh:
    return load_mesh(examples_dir / "cantilever.stl")


@pytest.fixture(scope="module")
def bracket(examples_dir) -> trimesh.Trimesh:
    return load_mesh(examples_dir / "bracket.stl")


def element_at(grid: Grid, p) -> tuple[int, int, int]:
    idx = np.floor((np.asarray(p, dtype=float) - np.asarray(grid.origin)) / grid.h).astype(int)
    return tuple(int(i) for i in idx)


def box_at(lo, hi) -> trimesh.Trimesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    m = trimesh.creation.box(extents=hi - lo)
    m.apply_translation((lo + hi) / 2)
    return m


# ---- loading / info / transforms ---------------------------------------------------------------


def test_load_mesh_path_and_bytes(examples_dir, cantilever):
    raw = (examples_dir / "cantilever.stl").read_bytes()
    from_bytes = load_mesh(raw, file_type="stl")
    assert len(cantilever.faces) == len(from_bytes.faces) == 12
    assert len(cantilever.vertices) == 8  # merged
    assert np.allclose(from_bytes.bounds, [[0, 0, 0], [60, 20, 20]])
    assert len(load_mesh(str(examples_dir / "bracket.stl"), file_type=".STL").faces) > 100


def test_load_mesh_scene_is_concatenated():
    scene = trimesh.Scene([box_at((0, 0, 0), (1, 1, 1)), box_at((2, 0, 0), (3, 1, 1))])
    mesh = load_mesh(scene.export(file_type="glb"), file_type="glb")
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.faces) == 24
    assert np.allclose(mesh.bounds, [[0, 0, 0], [3, 1, 1]])


def test_load_mesh_drops_degenerate_faces():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 1, 3]])  # second face is collinear
    buf = trimesh.Trimesh(v, f, process=False).export(file_type="stl")
    assert len(load_mesh(buf, file_type="stl").faces) == 1


@pytest.mark.parametrize(
    "src, ftype",
    [(b"", "stl"), (b"solid x\nendsolid x\n", "stl"), (b"abc", None), (b"abc", "nope")],
)
def test_load_mesh_rejects_bad_input(src, ftype):
    with pytest.raises(ValueError):
        load_mesh(src, file_type=ftype)


def test_load_mesh_missing_file(tmp_path):
    with pytest.raises(ValueError):
        load_mesh(tmp_path / "missing.stl")


def test_mesh_info(cantilever):
    info = mesh_info(cantilever)
    assert set(info) == set(MeshInfo.model_fields) - {"id", "name", "source", "n_brep_faces"}
    MeshInfo(id="m", name="cantilever", **info)
    assert info["n_faces"] == 12 and info["n_vertices"] == 8
    assert info["bbox"] == [[0, 0, 0], [60, 20, 20]]
    assert info["is_watertight"] is True
    assert info["volume"] == pytest.approx(24000)

    open_mesh = cantilever.copy()
    open_mesh.update_faces(np.arange(1, 12))
    info = mesh_info(open_mesh)
    assert info["is_watertight"] is False and info["volume"] is None


def test_transform_round_trip(cantilever):
    rot = trimesh.transformations.rotation_matrix(0.7, [1, 2, 3])
    m = trimesh.transformations.translation_matrix([5, -3, 2]) @ rot
    t16 = m.ravel(order="F").tolist()  # three.js Matrix4.toArray()
    assert np.allclose(t16[12:15], [5, -3, 2])
    assert np.allclose(transform_matrix(t16), m)
    assert np.allclose(transform_matrix(None), np.eye(4))

    moved = apply_transform(cantilever, t16)
    assert np.allclose(moved.vertices, cantilever.vertices @ m[:3, :3].T + m[:3, 3])
    assert np.allclose(cantilever.bounds, [[0, 0, 0], [60, 20, 20]])  # input untouched
    back = apply_transform(moved, np.linalg.inv(m).ravel(order="F"))
    assert np.allclose(back.vertices, cantilever.vertices)
    with pytest.raises(ValueError):
        transform_matrix([1.0] * 15)


# ---- voxelization ------------------------------------------------------------------------------


def test_cantilever_volume(cantilever):
    grid = Grid.from_bounds(cantilever.bounds, 30)
    mask, warnings = voxelize_mesh(cantilever, grid)
    assert grid.h == pytest.approx(2.0)
    assert mask.shape == grid.shape and mask.dtype == bool
    assert mask.sum() * grid.h**3 == pytest.approx(24000, rel=0.02)
    assert warnings == []
    # padding layer stays empty
    assert not mask[0].any() and not mask[-1].any() and not mask[:, :, 0].any()


def test_bracket_volume_and_holes(bracket):
    grid = Grid.from_bounds(bracket.bounds, 80)
    mask, _ = voxelize_mesh(bracket, grid)
    assert mask.sum() * grid.h**3 == pytest.approx(BRACKET_VOLUME, rel=0.04)
    for x in (20, 72):
        for y in (8, 52):
            assert not mask[element_at(grid, (x, y, 5))], (x, y)
    assert not mask[element_at(grid, (5, 30, 35))]  # wall hole
    for p in [(40, 30, 5), (5, 10, 35), (5, 30, 55), (75, 2, 2)]:
        assert mask[element_at(grid, p)], p


def test_bracket_fine_grid_volume(bracket):
    grid = Grid.from_bounds(bracket.bounds, 150)
    mask, _ = voxelize_mesh(bracket, grid)
    assert mask.sum() * grid.h**3 == pytest.approx(BRACKET_VOLUME, rel=0.04)


def test_scanline_matches_contains(bracket):
    grid = Grid.from_bounds(bracket.bounds, 40)
    mask, _ = voxelize_mesh(bracket, grid, method="scanline")
    idx = np.random.default_rng(0).choice(grid.nel, 3000, replace=False)
    expected = bracket.contains(grid.element_centers()[idx])
    assert np.array_equal(mask.ravel()[idx], expected)
    vote, _ = voxelize_mesh(bracket, grid, method="vote")
    assert np.array_equal(vote, mask)


def test_scanline_handles_rays_through_edges_and_vertices():
    # element-center columns pass exactly through the face diagonals of the box triangulation
    mesh = box_at((0, 0, 0), (4, 4, 4))
    grid = Grid(origin=(-1.0, -1.0, -1.0), h=1.0, shape=(6, 6, 6))
    mask, _ = voxelize_mesh(mesh, grid)
    assert mask.sum() == 64
    assert mask[1:5, 1:5, 1:5].all()
    for ax in (1, 2):  # same box, triangulated in another orientation
        perm = np.eye(4)
        perm[:3, :3] = np.roll(np.eye(3), ax, axis=1)
        m = mesh.copy()
        m.apply_transform(perm)
        assert voxelize_mesh(m, grid)[0].sum() == 64
    # centers exactly on faces, edges and corners: ties resolve half-open, never double count
    shifted = Grid(origin=(-1.5, -1.5, -1.5), h=1.0, shape=(7, 7, 7))
    assert voxelize_mesh(mesh, shifted)[0].sum() == 64
    inverted = mesh.copy()
    inverted.invert()
    assert voxelize_mesh(inverted, grid)[0].sum() == 64


def test_overlapping_bodies_are_unioned():
    a = box_at((0, 0, 0), (4, 4, 4))
    b = box_at((2, 0, 0), (6, 4, 4))
    both = trimesh.util.concatenate([a, b])
    grid = Grid.from_bounds(both.bounds, 6)
    mask, _ = voxelize_mesh(both, grid)
    assert mask.sum() * grid.h**3 == pytest.approx(6 * 4 * 4)


def test_non_watertight_still_voxelizes(bracket):
    holed = bracket.copy()
    holed.update_faces(np.setdiff1d(np.arange(len(holed.faces)), [0, 7, 100, 400]))
    assert not holed.is_watertight
    grid = Grid.from_bounds(bracket.bounds, 80)
    for method in ("auto", "surface_fill"):
        mask, warnings = voxelize_mesh(holed, grid, method=method)
        assert mask.shape == grid.shape and mask.any()
        assert any("watertight" in w for w in warnings)
    reference, _ = voxelize_mesh(bracket, grid)
    mask, _ = voxelize_mesh(holed, grid)  # auto -> 3-axis parity vote
    assert (mask != reference).sum() < 0.005 * reference.sum()
    assert mask.sum() * grid.h**3 == pytest.approx(BRACKET_VOLUME, rel=0.04)


def test_surface_fill_covers_solid(cantilever):
    grid = Grid.from_bounds(cantilever.bounds, 30)
    exact, _ = voxelize_mesh(cantilever, grid, method="scanline")
    filled, _ = voxelize_mesh(cantilever, grid, method="surface_fill")
    assert (filled >= exact).all()
    assert filled.sum() < 1.5 * exact.sum()


# ---- domain ------------------------------------------------------------------------------------


def test_build_domain_design_only(cantilever):
    grid, active, passive, warnings = build_domain(cantilever, [], 30)
    assert grid == Grid.from_bounds(cantilever.bounds, 30)
    assert active.sum() == 3000
    assert passive.dtype == np.int8 and not passive.any()
    assert warnings == []


def test_build_domain_keep_in_extends_grid(cantilever):
    keep_in = box_at((60, 0, 0), (70, 20, 20))
    grid, active, passive, warnings = build_domain(cantilever, [(keep_in, "keep_in")], 35)
    assert grid.h == pytest.approx(2.0)
    assert np.allclose(grid.bounds, [[-2, -2, -2], [72, 22, 22]])
    assert (passive == 1).sum() == 5 * 10 * 10
    assert active.sum() == 3000 + 500
    assert active[passive == 1].all()
    assert set(np.unique(passive)) <= {0, 1}
    assert passive[element_at(grid, (65, 10, 10))] == 1
    assert passive[element_at(grid, (55, 10, 10))] == 0
    assert warnings == []

    stats = domain_stats(grid, active, passive, warnings)
    assert stats["n_passive_solid"] == 500 and stats["n_free"] == 3000


def test_build_domain_keep_out_cylinder(cantilever):
    cyl = trimesh.creation.cylinder(radius=5, height=40, sections=64)  # axis Z
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # axis Y
    cyl.apply_translation((30, 10, 10))
    keep_in = box_at((26, 0, 0), (34, 20, 4))  # overlaps the cylinder only below it
    grid, active, passive, _ = build_domain(
        cantilever, [(keep_in, "keep_in"), (cyl, "keep_out")], 30
    )
    assert grid == Grid.from_bounds(cantilever.bounds, 30)  # keep_out never extends the grid
    assert not active[element_at(grid, (30, 10, 10))]
    assert passive[element_at(grid, (30, 10, 10))] == 0
    removed = 3000 - (active & (passive == 0)).sum() - (passive == 1).sum()
    assert removed == pytest.approx(np.pi * 25 * 20 / 8, rel=0.25)
    assert (passive[~active] == 0).all()


def test_build_domain_keep_out_outside(cantilever):
    far = box_at((100, 100, 100), (110, 110, 110))
    grid, active, _, warnings = build_domain(cantilever, [(far, "keep_out")], 30)
    assert grid == Grid.from_bounds(cantilever.bounds, 30)
    assert active.sum() == 3000
    assert any("keep_out 0" in w for w in warnings)
    with pytest.raises(ValueError):
        build_domain(cantilever, [(far, "sideways")], 30)


def test_build_domain_warns_on_disconnected(cantilever):
    island = box_at((70, 0, 0), (76, 20, 20))
    *_, warnings = build_domain(cantilever, [(island, "keep_in")], 40)
    assert any("disconnected" in w for w in warnings)


def test_domain_stats(cantilever):
    grid, active, passive, _ = build_domain(cantilever, [], 30)
    stats = domain_stats(grid, active, passive, ["hello"])
    assert set(stats) == set(VoxelStats.model_fields)
    VoxelStats(**stats)
    assert (stats["nx"], stats["ny"], stats["nz"]) == (32, 12, 12)
    assert stats["n_active"] == stats["n_free"] == 3000
    assert stats["n_passive_solid"] == stats["n_passive_void"] == 0
    assert stats["n_nodes"] == 31 * 11 * 11
    assert stats["n_dof"] == 3 * stats["n_nodes"]
    assert stats["est_bytes"] > 0 and stats["est_sec_per_iter"] > 0
    assert stats["warnings"] == ["hello"]

    passive[16, 6, 6] = -1
    assert domain_stats(grid, active, passive)["n_passive_void"] == 1
