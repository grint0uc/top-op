"""build.py: project document -> domain -> Problem (shared by server, CLI and MCP)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from topop.core.problem import SymmetryPlane
from topop.core.selection import node_xyz
from topop.core.step import META_FACETS, load_step
from topop.core.voxelize import load_mesh
from topop.server.build import (
    DESIGN,
    ProblemInvalid,
    build_domain_from_project,
    build_problem,
    domain_key,
    load_project_meshes,
    params_warnings,
    resolve_sel,
    run_params,
    size_warnings,
    step_facets_to_faces,
)
from topop.server.schemas import (
    IDENTITY,
    FacetSelection,
    GridSpec,
    LoadSpec,
    MeshRef,
    NormalSelection,
    ParamsSpec,
    PlaneSelection,
    ProjectIn,
    RefModel,
    SupportSpec,
    SymmetrySpec,
    VoxelStats,
)


def resolver(key: str):
    return load_mesh(key)


def translation(dx: float, dy: float, dz: float) -> list[float]:
    t = list(IDENTITY)
    t[12:15] = [dx, dy, dz]  # column-major: translation is the last column = elements 12..14
    return t


@pytest.fixture(scope="module")
def bracket(examples_dir: Path) -> str:
    return str(examples_dir / "bracket.stl")


def bracket_project(path: str, **kw) -> ProjectIn:
    """Base plate bolted down (plane z=0), wall top pushed down (facet 6), plate end pulled +x."""
    return ProjectIn(
        name="bracket",
        design_mesh=MeshRef(path=path),
        grid=GridSpec(elements_along_longest=24),
        supports=[
            SupportSpec(id="base", selection=PlaneSelection(point=[0, 0, 0], normal=[0, 0, 1]))
        ],
        loads=[
            LoadSpec(
                id="wall-top",
                selection=FacetSelection(mesh_id=path, facet_ids=[6]),
                force=[0, 0, -1],
            ),
            LoadSpec(
                id="plate-end",
                selection=NormalSelection(
                    mesh_id=DESIGN, direction=[1, 0, 0], within=[[70, -10, -10], [90, 70, 20]]
                ),
                force=[1, 0, 0],
                case=1,
            ),
        ],
        **kw,
    )


def test_build_problem_bracket(bracket: str):
    project = bracket_project(bracket, params=ParamsSpec(volfrac=0.25, max_iter=7, solver="direct"))
    meshes = load_project_meshes(project, resolver)
    assert meshes[DESIGN] is meshes[bracket]
    built = build_problem(project, meshes)

    assert built.problem.validate() == []
    assert set(built.resolved) == {"base", "wall-top", "plate-end"}
    assert all(n.size > 0 for n in built.resolved.values())
    assert built.problem.n_cases == 2
    VoxelStats(**built.stats)
    assert built.stats["n_active"] == built.problem.n_active > 0
    h = built.grid.h
    assert np.allclose(node_xyz(built.grid, built.resolved["base"])[:, 2], 0, atol=h / 2)
    top = node_xyz(built.grid, built.resolved["wall-top"])
    assert (top[:, 2] > 60 - h).all() and (top[:, 0] < 10 + h).all()
    end = node_xyz(built.grid, built.resolved["plate-end"])
    assert (end[:, 0] > 80 - h).all() and (end[:, 2] < 10 + h).all()
    assert built.params.volfrac == 0.25 and built.params.max_iter == 7
    assert built.params.solver == "direct"
    assert built.problem.material.nu == 0.3


def test_transform_shifts_grid(bracket: str):
    plain = bracket_project(bracket)
    moved = bracket_project(bracket)
    moved.design_mesh.transform = translation(100, -20, 5)
    a = build_domain_from_project(plain, load_project_meshes(plain, resolver))
    b = build_domain_from_project(moved, load_project_meshes(moved, resolver))
    assert b.grid.shape == a.grid.shape and b.grid.h == a.grid.h
    assert np.allclose(np.subtract(b.grid.origin, a.grid.origin), [100, -20, 5])
    assert np.array_equal(a.active, b.active)
    assert domain_key(plain) != domain_key(moved)


def test_transform_is_column_major(bracket: str):
    # 90 degrees about +z, column-major: x' = -y, y' = x -> 80 x 60 footprint becomes 60 x 80
    project = bracket_project(bracket)
    project.design_mesh.transform = [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    meshes = load_project_meshes(project, resolver)
    assert np.allclose(meshes[DESIGN].bounds, [[-60, 0, 0], [0, 80, 60]])
    dom = build_domain_from_project(project, meshes)
    plain = bracket_project(bracket)
    ref = build_domain_from_project(plain, load_project_meshes(plain, resolver))
    assert dom.grid.shape == (ref.grid.shape[1], ref.grid.shape[0], ref.grid.shape[2])


def test_ref_models_and_aliases(bracket: str, examples_dir: Path):
    box = str(examples_dir / "cantilever.stl")  # 60 x 20 x 20 at the origin, overlaps the plate
    project = bracket_project(
        bracket,
        ref_models=[
            RefModel(id="keep", mesh_id=None, path=box, mode="keep_in"),
            RefModel(id="cut", path=box, mode="keep_out", transform=translation(40, 40, 0)),
            RefModel(id="dangling", name="not uploaded"),
        ],
    )
    meshes = load_project_meshes(project, resolver)
    assert {"ref:keep", "ref:cut", box, DESIGN, bracket} <= set(meshes)
    assert np.allclose(meshes["ref:cut"].bounds[0], [40, 40, 0])
    dom = build_domain_from_project(project, meshes)
    assert dom.stats["n_passive_solid"] > 0
    assert any("not uploaded" in w for w in dom.warnings)
    plain = bracket_project(bracket)
    base = build_domain_from_project(plain, load_project_meshes(plain, resolver))
    assert dom.grid.shape == base.grid.shape  # keep_in inside the bbox, keep_out never extends it


def test_not_runnable_lists_issues(bracket: str):
    project = bracket_project(bracket)
    project.supports = []
    project.loads[0].selection = FacetSelection(mesh_id=bracket, facet_ids=[100_000])
    with pytest.raises(ProblemInvalid) as err:
        build_problem(project, load_project_meshes(project, resolver))
    issues = err.value.issues
    assert "no supports defined" in issues
    assert any("unknown facet ids" in s for s in issues)
    assert isinstance(err.value, ValueError) and "no supports defined" in str(err.value)

    nowhere = bracket_project(bracket)
    nowhere.supports[0].selection = PlaneSelection(point=[0, 0, 500], normal=[0, 0, 1])
    with pytest.raises(ProblemInvalid, match="zero nodes"):
        build_problem(nowhere, load_project_meshes(nowhere, resolver))

    with pytest.raises(ValueError, match="no design mesh"):
        load_project_meshes(ProjectIn(), resolver)


def test_params_and_size_warnings():
    rp = run_params(ParamsSpec(density_every=7, dtype="float32", heaviside=True))
    assert rp.dtype == "float32" and rp.heaviside and not hasattr(rp, "density_every")
    assert size_warnings(1000) == []
    assert "slow" in size_warnings(200_000)[0]
    assert "out of memory" in size_warnings(400_000)[0]


def test_run_params_v02_fields():
    spec = ParamsSpec(
        optimizer="mma",
        symmetry=[SymmetrySpec(axis="y"), SymmetrySpec(axis="z", position=12.5)],
        stress_limit=2.5,
        stress_pnorm=10,
        overhang="-z",
    )
    rp = run_params(spec)
    assert rp.optimizer == "mma" and rp.stress_limit == 2.5 and rp.stress_pnorm == 10
    assert rp.overhang == "-z"
    assert rp.symmetry == (SymmetryPlane("y"), SymmetryPlane("z", 12.5))
    assert isinstance(rp.symmetry, tuple) and rp.symmetry[0].position is None
    plain = run_params(ParamsSpec())
    assert plain.symmetry == () and plain.stress_limit is None and plain.overhang is None
    assert plain.optimizer == "oc"


def test_params_warnings():
    assert params_warnings(ParamsSpec()) == []
    assert params_warnings(ParamsSpec(optimizer="mma", stress_limit=1.0)) == []
    (forced,) = params_warnings(ParamsSpec(stress_limit=1.0))  # optimizer defaults to oc
    assert "stress_limit" in forced and "mma" in forced
    (up,) = params_warnings(ParamsSpec(overhang="+y"))
    assert "base plate" in up and "min y" in up
    (down,) = params_warnings(ParamsSpec(overhang="-x"))
    assert "base plate" in down and "max x" in down


def test_build_problem_reports_param_notes_and_carries_the_params(bracket: str):
    project = bracket_project(
        bracket,
        params=ParamsSpec(
            stress_limit=3.0, symmetry=[SymmetrySpec(axis="y", position=30)], overhang="+z"
        ),
    )
    built = build_problem(project, load_project_meshes(project, resolver))
    assert any("mma" in w for w in built.warnings) and any(
        "base plate" in w for w in built.warnings
    )
    assert built.stats["warnings"] == built.warnings
    VoxelStats(**built.stats)
    assert built.params.symmetry == (SymmetryPlane("y", 30.0),)
    assert built.params.stress_limit == 3.0 and built.params.overhang == "+z"
    # the cached domain itself knows nothing about params
    assert not any(
        "base plate" in w
        for w in build_domain_from_project(project, load_project_meshes(project, resolver)).warnings
    )
    assert domain_key(project) == domain_key(bracket_project(bracket))


def test_resolve_sel_turns_step_facets_into_brep_faces(examples_dir: Path):
    pytest.importorskip("OCP", reason="STEP support not installed (uv sync --extra step)")
    mesh = load_step(examples_dir / "bracket.step").mesh
    key = str(examples_dir / "bracket.step")
    project = ProjectIn(design_mesh=MeshRef(path=key), grid=GridSpec(elements_along_longest=40))
    domain = build_domain_from_project(project, load_project_meshes(project, lambda _: mesh))
    facets = mesh.metadata[META_FACETS]
    hole = next(i for i, f in enumerate(facets) if abs((f.get("radius") or 0) - 6) < 1e-6)
    sel = {"kind": "facets", "mesh_id": key, "facet_ids": [hole]}
    assert step_facets_to_faces(sel, {key: mesh})["kind"] == "faces"
    assert step_facets_to_faces({"kind": "plane"}, {key: mesh}) == {"kind": "plane"}
    assert resolve_sel(sel, domain).size > 0
    assert step_facets_to_faces(sel, {key: load_mesh(examples_dir / "bracket.stl")}) is sel


def _box_project(transform: list[float], **kw) -> ProjectIn:
    return ProjectIn(
        design_mesh=MeshRef(path="box", transform=transform),
        grid=GridSpec(elements_along_longest=12),
        **kw,
    )


def test_facet_ids_are_the_raw_meshs_under_a_non_uniform_scale():
    # raw 10 x 20 x 30: the two largest facets are the x faces (20 x 30). Scaled x3 along x the
    # y faces (30 x 30) become the largest, so ranking facets on the world mesh permuted the ids.
    from topop.core.selection import compute_facets

    box = trimesh.creation.box((10.0, 20.0, 30.0))
    diag = [3.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]
    project = _box_project(diag)
    meshes = load_project_meshes(project, lambda _: box)
    assert meshes.raw["box"] is box and meshes.raw[DESIGN] is box
    domain = build_domain_from_project(project, meshes)
    listed, _ = compute_facets(box)  # what `/meshes/{id}/facets` and `topop describe` list
    world_listed, _ = compute_facets(meshes[DESIGN])
    assert abs(listed[0]["normal"][0]) == 1 and abs(world_listed[0]["normal"][1]) == 1
    h = domain.grid.h
    for fid in (0, 1):
        sign = listed[fid]["normal"][0]
        for key in ("box", DESIGN):
            nodes = resolve_sel({"kind": "facets", "mesh_id": key, "facet_ids": [fid]}, domain)
            assert nodes.size > 0
            assert np.allclose(node_xyz(domain.grid, nodes)[:, 0], 15.0 * sign, atol=h)


def test_load_cases_are_renumbered_to_the_ones_in_use(bracket: str):
    project = bracket_project(bracket)
    project.loads[1].case = 15
    built = build_problem(project, load_project_meshes(project, resolver))
    assert built.problem.n_cases == 2
    assert [ld.case for ld in built.problem.loads] == [0, 1]
    one = bracket_project(bracket)
    one.loads[0].case = one.loads[1].case = 7
    assert build_problem(one, load_project_meshes(one, resolver)).problem.n_cases == 1


_MEMORY_SCRIPT = """
import json, sys, time
from topop.server.build import ProblemInvalid, build_problem, load_project_meshes
from topop.server.schemas import GridSpec, ProjectIn
from topop.core.voxelize import load_mesh
project = ProjectIn.model_validate_json(sys.argv[1])
project.grid = GridSpec(elements_along_longest=600)
t = time.perf_counter()
try:
    build_problem(project, load_project_meshes(project, load_mesh))
    msg = None
except ProblemInvalid as exc:
    msg = str(exc)
dt = time.perf_counter() - t
with open("/proc/self/status") as f:
    peak = next(int(ln.split()[1]) * 1024 for ln in f if ln.startswith("VmHWM:"))
print(json.dumps({"msg": msg, "dt": dt, "peak": peak}))
"""


@pytest.mark.skipif(not Path("/proc/self/status").is_file(), reason="needs Linux VmHWM")
def test_too_fine_grid_is_refused_before_voxelizing(bracket: str):
    import json
    import subprocess
    import sys

    project = bracket_project(bracket)
    out = subprocess.run(
        [sys.executable, "-c", _MEMORY_SCRIPT, project.model_dump_json()],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    res = json.loads(out.stdout)
    assert res["msg"] is not None and "memory cap" in res["msg"] and "6.0 GB" in res["msg"]
    assert "elements_along_longest" in res["msg"]
    assert res["dt"] < 2.0
    assert res["peak"] < 1e9


def test_run_over_the_memory_cap_is_refused_before_resolving(bracket: str, monkeypatch):
    from topop.server import build

    def boom(*args, **kwargs):
        raise AssertionError("selections resolved")

    project = bracket_project(bracket)
    project.grid = GridSpec(elements_along_longest=200)  # ~1.2M active elements, ~13 GB
    meshes = load_project_meshes(project, resolver)
    domain = build_domain_from_project(project, meshes)  # the voxel preview still works
    assert domain.stats["est_bytes"] > 6e9
    monkeypatch.setattr(build, "resolve_project_selections", boom)
    with pytest.raises(ProblemInvalid, match="memory cap"):
        build_problem(project, meshes, domain)
