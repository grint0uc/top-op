"""Pins the WP0 contract: grid index conventions, schema round-trips, API surface, examples."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
import trimesh
from fastapi.routing import APIWebSocketRoute
from openapi_spec_validator import validate
from pydantic import BaseModel, ValidationError

from topop import __version__
from topop.core.problem import HEX8_OFFSETS, Grid
from topop.server import routes_runs, schemas
from topop.server.schemas import (
    FaceSelection,
    FacetSelection,
    LoadSpec,
    MeshRef,
    NormalSelection,
    PlaneSelection,
    PrimitiveSelection,
    Project,
    ProjectIn,
    RefModel,
    SupportSpec,
)


def load_example(examples_dir, name: str) -> trimesh.Trimesh:
    mesh = trimesh.load(examples_dir / name, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh


# ---- (a)-(c) Grid conventions ------------------------------------------------------------------


def test_from_bounds_cantilever(examples_dir):
    bounds = load_example(examples_dir, "cantilever.stl").bounds
    grid = Grid.from_bounds(bounds, elements_along_longest=30, padding=1)
    assert grid.h == pytest.approx(2.0, abs=1e-9)
    assert grid.shape == (32, 12, 12)  # 30 x 10 x 10 inner + 1 voxel padding on every side
    assert np.allclose(grid.origin, (-2.0, -2.0, -2.0), atol=1e-9)
    assert np.allclose(grid.bounds, [[-2, -2, -2], [62, 22, 22]], atol=1e-9)

    wide = Grid.from_bounds(bounds, elements_along_longest=30, padding=3)
    assert wide.shape == (36, 16, 16)
    assert wide.h == pytest.approx(2.0, abs=1e-9)


def test_element_nodes_single_element_follows_hex8_order():
    grid = Grid(origin=(0.0, 0.0, 0.0), h=1.0, shape=(1, 1, 1))
    en = grid.element_nodes()
    assert en.shape == (1, 8)
    # node id = 4*ix + 2*iy + iz on a (2,2,2) node grid
    assert en[0].tolist() == [0, 4, 6, 2, 1, 5, 7, 3]
    expected = np.ravel_multi_index(tuple(HEX8_OFFSETS.T), (2, 2, 2))
    assert np.array_equal(en[0], expected)


def test_node_ids_and_coords():
    grid = Grid(origin=(1.5, -2.0, 3.0), h=0.5, shape=(3, 4, 5))
    coords = grid.node_coords()
    assert coords.shape == (grid.n_nodes, 3)
    origin = np.asarray(grid.origin)
    assert np.allclose(coords[grid.node_ids(1, 0, 0)], origin + (0.5, 0, 0))
    assert np.allclose(coords[grid.node_ids(2, 3, 1)], origin + 0.5 * np.array([2, 3, 1]))
    # last node is the far corner of the grid
    assert np.allclose(coords[-1], grid.bounds[1])


# ---- (d) schemas -------------------------------------------------------------------------------


def all_models() -> list[type[BaseModel]]:
    return [
        m
        for _, m in inspect.getmembers(schemas, inspect.isclass)
        if issubclass(m, BaseModel) and m.__module__ == schemas.__name__
    ]


@pytest.mark.parametrize("model", all_models(), ids=lambda m: m.__name__)
def test_schema_models_instantiate_with_defaults(model):
    try:
        instance = model()
    except ValidationError as e:
        # only required fields may be missing; every other field must carry a valid default
        assert {err["type"] for err in e.errors()} == {"missing"}, e
    else:
        assert isinstance(instance, model)


def test_default_only_models():
    for model in (schemas.GridSpec, schemas.MaterialSpec, schemas.ParamsSpec, ProjectIn):
        model()


def make_full_project() -> ProjectIn:
    mid = "mesh-1"
    selections = [
        FaceSelection(mesh_id=mid, face_ids=[0, 5, 9]),
        FacetSelection(mesh_id=mid, facet_ids=[1, 2], angle_deg=7.5),
        NormalSelection(
            mesh_id=mid,
            direction=[0, 0, 1],
            angle_deg=12,
            within=[[0, 0, 0], [10, 10, 10]],
        ),
        PlaneSelection(point=[0, 0, 0], normal=[1, 0, 0], tol=0.25),
        PrimitiveSelection(kind="box", size=[1, 2, 3], surface_only=False),
        PrimitiveSelection(kind="sphere", size=[4, 4, 4]),
        PrimitiveSelection(kind="cylinder", size=[2, 5, 2]),
    ]
    loads = [
        LoadSpec(id=f"l{i}", name=f"load {i}", selection=s, force=[0, 0, -1.0], case=i % 2)
        for i, s in enumerate(selections[:4])
    ]
    supports = [
        SupportSpec(id=f"s{i}", selection=s, fix=[True, i % 2 == 0, True])
        for i, s in enumerate(selections[4:])
    ]
    return ProjectIn(
        name="roundtrip",
        design_mesh=MeshRef(mesh_id=mid),
        ref_models=[
            RefModel(id="r0", mesh_id="mesh-2", mode="keep_in"),
            RefModel(id="r1", path="cutter.stl", mode="keep_out", visible=False),
        ],
        loads=loads,
        supports=supports,
    )


def test_project_in_roundtrip_all_selection_kinds():
    project = make_full_project()
    kinds = {ld.selection.kind for ld in project.loads} | {
        sp.selection.kind for sp in project.supports
    }
    assert kinds == {"faces", "facets", "normal", "plane", "box", "sphere", "cylinder"}

    again = ProjectIn.model_validate_json(project.model_dump_json())
    assert again == project
    assert isinstance(again.loads[0].selection, FaceSelection)
    assert isinstance(again.loads[1].selection, FacetSelection)
    assert isinstance(again.loads[2].selection, NormalSelection)
    assert isinstance(again.loads[3].selection, PlaneSelection)
    assert all(isinstance(sp.selection, PrimitiveSelection) for sp in again.supports)

    full = Project(
        **project.model_dump(), id="p1", created_at="2026-01-01T00:00:00Z", updated_at="x"
    )
    assert Project.model_validate_json(full.model_dump_json()) == full


# ---- (e) API surface ---------------------------------------------------------------------------

# PLAN.md section 3, as (method, path). WebSocket route is checked separately.
PLAN_ENDPOINTS = [
    ("post", "/api/meshes"),
    ("get", "/api/meshes/{id}/buffer"),
    ("get", "/api/meshes/{id}/adjacency"),
    ("get", "/api/meshes/{id}/facets"),
    ("get", "/api/meshes/{id}/preview.png"),
    ("get", "/api/runs/{id}/preview.png"),
    ("post", "/api/projects"),
    ("put", "/api/projects/{id}"),
    ("post", "/api/projects/{id}/voxelize"),
    ("post", "/api/projects/{id}/resolve-selection"),
    ("post", "/api/runs"),
    ("post", "/api/runs/{id}/cancel"),
    ("get", "/api/runs/{id}/result.stl"),
    ("get", "/api/runs/{id}/result.vti"),
    ("get", "/api/runs/{id}/result.npz"),
    ("get", "/api/runs/{id}/project.json"),
]


def norm(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", path)


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "version": __version__}


def test_root_serves_something(client):
    # placeholder when the frontend is not built, index.html when it is
    assert client.get("/").status_code == 200


def test_openapi_valid_and_complete(client):
    res = client.get("/openapi.json")
    assert res.status_code == 200
    spec = res.json()
    validate(spec)
    assert spec["info"]["title"] == "top-op"
    assert spec["info"]["version"] == __version__

    have = {(method, norm(p)) for p, item in spec["paths"].items() for method in item}
    missing = [(m, p) for m, p in PLAN_ENDPOINTS if (m, norm(p)) not in have]
    assert not missing

    # models no route references must still reach types.gen.ts
    names = spec["components"]["schemas"]
    assert {"ProgressMsg", "StatusMsg", "WsMessage", "Selection"} <= set(names)
    assert not any(n.endswith(("-Input", "-Output")) for n in names)


def test_stream_websocket_declared():
    paths = [r.path for r in routes_runs.router.routes if isinstance(r, APIWebSocketRoute)]
    assert "/api/runs/{id}/stream" in paths


# ---- (f) example meshes ------------------------------------------------------------------------


def test_example_cantilever(examples_dir):
    mesh = load_example(examples_dir, "cantilever.stl")
    assert mesh.is_watertight
    assert np.allclose(mesh.bounds, [[0, 0, 0], [60, 20, 20]])
    assert mesh.volume == pytest.approx(60 * 20 * 20)


def test_example_bracket(examples_dir):
    mesh = load_example(examples_dir, "bracket.stl")
    assert mesh.is_watertight
    assert np.allclose(mesh.bounds, [[0, 0, 0], [80, 60, 60]])
    solid = 80 * 60 * 10 + 10 * 60 * 50
    holes = 4 * np.pi * 3**2 * 10 + np.pi * 6**2 * 10
    assert mesh.volume == pytest.approx(solid - holes, rel=0.005)
