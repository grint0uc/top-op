"""STEP import: B-rep faces as exact facets (core loader, store, API, selections, CLI)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from topop.agent import Session
from topop.cli import main
from topop.core.step import StepMesh, is_step, load_step
from topop.server.app import app
from topop.server.build import build_problem, load_project_meshes
from topop.server.schemas import ProjectIn
from topop.server.store import Store

N_BREP_FACES = 14  # 8 planes, the fillet, the Ø12 hole, four Ø6 holes


@pytest.fixture(scope="session")
def step_path(examples_dir: Path) -> Path:
    pytest.importorskip("OCP", reason="STEP support not installed (uv sync --extra step)")
    return examples_dir / "bracket.step"


@pytest.fixture(scope="session")
def bracket(step_path: Path) -> StepMesh:
    return load_step(step_path)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "data"
    monkeypatch.setenv("TOPOP_DATA_DIR", str(d))
    return d


@pytest.fixture
def api(data_dir: Path):
    with TestClient(app) as client:
        yield client


def upload(client: TestClient, path: Path) -> dict:
    res = client.post("/api/meshes", files={"file": (path.name, path.read_bytes())})
    assert res.status_code == 200, res.text
    return res.json()


def by_kind(sm: StepMesh, kind: str) -> list[dict]:
    return [f for f in sm.faces if f["kind"] == kind]


def parallel(a, b) -> bool:
    return abs(float(np.dot(a, b))) > 0.999


# ---- detection and missing dependency (no OpenCascade needed) ------------------------------------


def test_is_step_by_extension_and_header(tmp_path: Path):
    assert is_step("part.step") and is_step("PART.STP") and is_step(Path("a/b.stp"))
    assert not is_step("part.stl") and not is_step(tmp_path / "nothing.obj")
    assert is_step(b"ISO-10303-21;\nHEADER;") and is_step(b"\xef\xbb\xbf  \nISO-10303-21;")
    assert not is_step(b"solid cube\nfacet normal") and not is_step(b"")
    sniffed = tmp_path / "part.dat"  # unknown extension, STEP content
    sniffed.write_bytes(b"ISO-10303-21;\nHEADER;\n")
    assert is_step(sniffed)


def test_missing_opencascade_gives_an_install_hint(data_dir: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "OCP", None)  # `import OCP` raises ImportError
    with pytest.raises(ImportError, match=r"uv sync --extra step"):
        load_step(b"ISO-10303-21;\n")
    with pytest.raises(ValueError, match=r"pip install \"topop\[step\]\""):
        Store(data_dir).add_mesh(b"ISO-10303-21;\n", "part.step")  # reported like bad input
    with TestClient(app) as client:
        res = client.post("/api/meshes", files={"file": ("part.step", b"ISO-10303-21;\n")})
    assert res.status_code == 400 and "--extra step" in res.json()["detail"]


# ---- core loader --------------------------------------------------------------------------------


def test_bracket_tessellation_is_watertight_with_the_right_volume(bracket: StepMesh, step_path):
    cq = pytest.importorskip("cadquery")
    exact = cq.importers.importStep(str(step_path)).val().Volume()
    m = bracket.mesh
    assert m.is_watertight and m.is_winding_consistent and m.volume > 0
    assert m.volume == pytest.approx(exact, rel=0.02)
    # the STL bracket (75741.7) plus the 3 mm fillet, (1 - pi/4) * 3^2 * 60 = 116
    assert exact == pytest.approx(75741.7 + (1 - math.pi / 4) * 9 * 60, rel=2e-3)
    assert m.bounds.ravel() == pytest.approx([0, 0, 0, 80, 60, 60], abs=1e-6)
    assert len(m.faces) == len(bracket.brep_faces) and len(m.faces) > 0
    assert len(bracket.faces) == N_BREP_FACES >= 12
    assert set(np.unique(bracket.brep_faces)) == set(
        range(N_BREP_FACES)
    )  # every face has triangles
    assert [f["brep_face"] for f in bracket.faces] == list(range(N_BREP_FACES))


def test_path_and_bytes_agree(bracket: StepMesh, step_path: Path):
    again = load_step(step_path.read_bytes())
    assert np.array_equal(again.brep_faces, bracket.brep_faces)
    assert np.allclose(again.mesh.vertices, bracket.mesh.vertices)


def test_cylinders_carry_exact_radius_and_axis(bracket: StepMesh):
    cyls = by_kind(bracket, "cylinder")
    assert len(cyls) == 6
    big = [c for c in cyls if c["radius"] == pytest.approx(6)]
    assert len(big) == 1  # the Ø12 hole along X, centered at y=30, z=35
    assert parallel(big[0]["axis"], [1, 0, 0])
    assert big[0]["centroid"] == pytest.approx([5, 30, 35], abs=1e-3)
    assert big[0]["area"] == pytest.approx(2 * math.pi * 6 * 10, rel=1e-6)

    holes = [c for c in cyls if parallel(c["axis"], [0, 0, 1])]
    assert len(holes) == 4  # the Ø6 holes through the plate
    assert all(c["radius"] == pytest.approx(3) for c in holes)
    xy = sorted((round(c["centroid"][0]), round(c["centroid"][1])) for c in holes)
    assert xy == [(20, 8), (20, 52), (72, 8), (72, 52)]

    fillet = [c for c in cyls if parallel(c["axis"], [0, 1, 0])]
    assert len(fillet) == 1  # 3 mm blend between plate and wall, axis along Y
    assert fillet[0]["radius"] == pytest.approx(3) and fillet[0]["kind"] != "plane"
    assert fillet[0]["area"] == pytest.approx(math.pi / 2 * 3 * 60, rel=1e-6)
    assert all(c["normal"] == [0, 0, 0] for c in cyls)


def test_plane_normals_point_outward_and_match_the_triangles(bracket: StepMesh):
    planes = by_kind(bracket, "plane")
    assert len(planes) == 8
    m = bracket.mesh
    for f in planes:
        sel = bracket.brep_faces == f["brep_face"]
        n = (m.face_normals[sel] * m.area_faces[sel, None]).sum(0)
        n /= np.linalg.norm(n)
        assert np.dot(n, f["normal"]) > 0.999, f
        assert np.linalg.norm(f["normal"]) == pytest.approx(1)
    bottom = max(planes, key=lambda f: f["area"])
    assert bottom["area"] == pytest.approx(80 * 60 - 4 * math.pi * 9, rel=1e-9)
    assert bottom["normal"] == pytest.approx([0, 0, -1], abs=1e-9)


def test_facets_are_area_sorted_ranks_with_brep_face(bracket: StepMesh):
    facets, face_to_facet = bracket.facets()
    assert [f["id"] for f in facets] == list(range(N_BREP_FACES))
    areas = [f["area"] for f in facets]
    assert areas == sorted(areas, reverse=True)
    assert sorted(f["brep_face"] for f in facets) == list(range(N_BREP_FACES))
    assert sum(f["n_faces"] for f in facets) == len(bracket.mesh.faces)
    for f in facets:
        tri = np.flatnonzero(face_to_facet == f["id"])
        assert len(tri) == f["n_faces"]
        assert set(bracket.brep_faces[tri]) == {f["brep_face"]}
        lo, hi = np.asarray(f["bbox"])
        v = bracket.mesh.vertices[bracket.mesh.faces[tri]].reshape(-1, 3)
        assert v.min(0) == pytest.approx(lo, abs=1e-6) and v.max(0) == pytest.approx(hi, abs=1e-6)
    assert facets[0]["normal"] == pytest.approx([0, 0, -1], abs=1e-9)  # plate bottom
    json.dumps(facets)  # plain python types only


def test_npz_round_trip_keeps_everything(bracket: StepMesh):
    back = StepMesh.from_npz_bytes(bracket.to_npz_bytes())
    assert np.array_equal(back.mesh.faces, bracket.mesh.faces)
    assert np.array_equal(back.mesh.vertices, bracket.mesh.vertices)
    assert np.array_equal(back.brep_faces, bracket.brep_faces)
    assert back.faces == bracket.faces
    with pytest.raises(ValueError):
        StepMesh.from_npz_bytes(b"not an npz")


def test_tolerance_controls_the_triangle_count(step_path: Path, bracket: StepMesh):
    coarse = load_step(step_path, tolerance=1.0, angular_tolerance_deg=45)
    assert len(coarse.mesh.faces) < len(bracket.mesh.faces) / 2
    assert coarse.mesh.is_watertight and len(coarse.faces) == N_BREP_FACES
    with pytest.raises(ValueError, match="tolerance"):
        load_step(step_path, tolerance=-1)


def test_sphere_is_other_and_watertight_despite_its_poles(tmp_path: Path):
    cq = pytest.importorskip("cadquery")
    path = tmp_path / "ball.step"
    cq.exporters.export(cq.Workplane("XY").sphere(10), str(path))
    sm = load_step(path)
    assert [f["kind"] for f in sm.faces] == ["other"] and sm.faces[0]["radius"] is None
    assert sm.mesh.is_watertight and sm.mesh.is_winding_consistent
    assert sm.mesh.volume == pytest.approx(4 / 3 * math.pi * 1000, rel=0.01)
    assert not np.any(sm.mesh.area_faces == 0)


def test_unreadable_input_is_a_value_error(step_path: Path, tmp_path: Path):
    with pytest.raises(ValueError):
        load_step(b"")
    with pytest.raises(ValueError):
        load_step(b"ISO-10303-21;\nthis is not a STEP file\n")
    with pytest.raises(ValueError, match="not found"):
        load_step(tmp_path / "missing.step")


# ---- store ---------------------------------------------------------------------------------------


def test_store_caches_the_tessellation(step_path: Path, data_dir: Path, monkeypatch):
    store = Store(data_dir)
    info = store.add_mesh(step_path.read_bytes(), "bracket.step")
    assert info.source == "step" and info.n_brep_faces == N_BREP_FACES
    assert info.is_watertight and info.name == "bracket.step"
    for ext in ("step", "stl", "brep.npz", "json"):
        assert (store.mesh_dir / f"{info.id}.{ext}").is_file(), ext
    assert (store.mesh_dir / f"{info.id}.step").read_bytes() == step_path.read_bytes()

    def boom(*args, **kwargs):
        raise AssertionError("tessellated again")

    monkeypatch.setattr("topop.server.store.load_step", boom)
    fresh = Store(data_dir)  # a restarted server: nothing in memory
    mesh = fresh.get_mesh(info.id)
    assert len(mesh.faces) == info.n_faces
    facets, total = fresh.mesh_facets(info.id, 5.0)
    assert total == N_BREP_FACES and all(f["brep_face"] is not None for f in facets)
    assert fresh.mesh_info(info.id) == info
    assert fresh.add_mesh(step_path.read_bytes(), "again.step").id == info.id  # same bytes, same id
    # the cached STL is the same part (float32), for use outside top-op
    import trimesh

    stl = trimesh.load(store.mesh_dir / f"{info.id}.stl")
    assert stl.volume == pytest.approx(mesh.volume, rel=1e-5)


def test_store_retessellates_when_the_cache_is_lost(step_path: Path, data_dir: Path):
    store = Store(data_dir)
    info = store.add_mesh(step_path.read_bytes(), "bracket.step")
    (store.mesh_dir / f"{info.id}.brep.npz").unlink()
    fresh = Store(data_dir)
    assert fresh.mesh_info(info.id) == info
    assert (store.mesh_dir / f"{info.id}.brep.npz").is_file()  # rewritten


def test_store_step_facets_ignore_the_angle_and_expand_to_triangles(step_path: Path, data_dir):
    store = Store(data_dir)
    mid = store.add_mesh(step_path.read_bytes(), "bracket.step").id
    f5, n5 = store.mesh_facets(mid, 5.0)
    f60, n60 = store.mesh_facets(mid, 60.0)
    assert f5 == f60 and n5 == n60 == N_BREP_FACES
    tris = store.facet_faces(mid, [0], angle_deg=5.0)
    assert len(tris) == f5[0]["n_faces"] > 0
    mesh = store.get_mesh(mid)
    assert np.allclose(mesh.face_normals[tris], [0, 0, -1], atol=1e-6)
    both = store.facet_faces(mid, [0, 1])
    assert len(both) == f5[0]["n_faces"] + f5[1]["n_faces"]
    with pytest.raises(ValueError, match="unknown facet ids"):
        store.facet_faces(mid, [N_BREP_FACES])


def test_store_facet_faces_for_plain_meshes(examples_dir: Path, data_dir: Path):
    store = Store(data_dir)
    mid = store.add_mesh((examples_dir / "bracket.stl").read_bytes(), "bracket.stl").id
    assert store.mesh_info(mid).source == "mesh" and store.mesh_info(mid).n_brep_faces is None
    facets, _ = store.mesh_facets(mid, 5.0)
    tris = store.facet_faces(mid, [0], 5.0)
    assert len(tris) == facets[0]["n_faces"]
    assert np.allclose(store.get_mesh(mid).face_normals[tris], [0, 0, -1], atol=1e-6)
    assert all(f.get("brep_face") is None for f in facets)


# ---- API -----------------------------------------------------------------------------------------


def test_api_upload_and_facets(api: TestClient, step_path: Path):
    info = upload(api, step_path)
    assert info["source"] == "step" and info["n_brep_faces"] == N_BREP_FACES
    assert info["is_watertight"] and info["name"] == "bracket.step"
    assert api.get(f"/api/meshes/{info['id']}").json() == info

    body = api.get(f"/api/meshes/{info['id']}/facets", params={"angle_deg": 30}).json()
    assert body["n_facets_total"] == N_BREP_FACES and len(body["facets"]) == N_BREP_FACES
    facets = body["facets"]
    assert [f["id"] for f in facets] == list(range(N_BREP_FACES))
    assert all(f["brep_face"] is not None for f in facets)
    assert sorted(f["brep_face"] for f in facets) == list(range(N_BREP_FACES))
    kinds = [f["kind"] for f in facets]
    assert kinds.count("plane") == 8 and kinds.count("cylinder") == 6
    wide = next(f for f in facets if f["radius"] == pytest.approx(6))
    assert parallel(wide["axis"], [1, 0, 0])
    assert body["facets"] == api.get(f"/api/meshes/{info['id']}/facets").json()["facets"]

    buf = api.get(f"/api/meshes/{info['id']}/buffer").content
    n_vert, n_tri = np.frombuffer(buf[:8], dtype="<u4")
    assert n_tri == info["n_faces"] and n_vert == info["n_vertices"]
    assert api.get(f"/api/meshes/{info['id']}/preview.png").status_code == 200


def test_api_rejects_a_broken_step_upload(api: TestClient):
    pytest.importorskip("OCP")
    res = api.post("/api/meshes", files={"file": ("broken.step", b"ISO-10303-21;\nnope\n")})
    assert res.status_code == 400 and "STEP" in res.json()["detail"]


def bracket_case(mesh_id: str, **edit) -> dict:
    return {
        "name": "step bracket",
        "design_mesh": {"mesh_id": mesh_id},
        "grid": {"elements_along_longest": 24},
        "params": {"max_iter": 3},
        "supports": [
            {
                "id": "base",
                "selection": {"kind": "facets", "mesh_id": mesh_id, "facet_ids": [0]},
            }
        ],
        "loads": [
            {
                "id": "push",
                "selection": {
                    "kind": "normal",
                    "mesh_id": mesh_id,
                    "direction": [-1, 0, 0],
                    "angle_deg": 10,
                    "within": [[-5, -5, 40], [5, 65, 65]],
                },
                "force": [100, 0, 0],
            }
        ],
        **edit,
    }


def test_api_project_with_facet_support_voxelizes_and_resolves(api: TestClient, step_path: Path):
    mid = upload(api, step_path)["id"]
    res = api.post("/api/projects", json=bracket_case(mid))
    assert res.status_code == 200, res.text
    stats = api.post(f"/api/projects/{res.json()['id']}/voxelize")
    assert stats.status_code == 200, stats.text
    stats = stats.json()
    assert stats["n_active"] > 0
    assert not [w for w in stats["warnings"] if "resolves to 0 nodes" in w or "unknown" in w]


# ---- selections ----------------------------------------------------------------------------------


@pytest.fixture
def session(data_dir: Path, step_path: Path) -> Session:
    return Session(data_dir)


def nodes_bbox(rows: list[dict], ident: str) -> np.ndarray:
    row = next(r for r in rows if r["id"] == ident)
    assert row["n_nodes"] > 0, row
    return np.asarray(row["bbox"])


def test_facets_selection_on_step_resolves_the_exact_brep_face(session: Session, step_path: Path):
    info = session.load_mesh(step_path)
    desc = session.describe_mesh(info.id, top=0)
    assert desc["mesh"]["source"] == "step" and desc["n_facets_total"] == N_BREP_FACES
    hole = next(f for f in desc["facets"] if f["radius"] == pytest.approx(6))
    plate_bottom = desc["facets"][0]
    assert plate_bottom["brep_face"] is not None and plate_bottom["kind"] == "plane"

    p = session.create_project(
        ProjectIn.model_validate(
            bracket_case(
                info.id,
                supports=[
                    {
                        "id": "base",
                        "selection": {
                            "kind": "facets",
                            "mesh_id": "design",
                            "facet_ids": [plate_bottom["id"]],
                        },
                    }
                ],
                loads=[
                    {
                        "id": "pin",
                        "selection": {
                            "kind": "facets",
                            "mesh_id": "design",
                            "facet_ids": [hole["id"]],
                        },
                        "force": [0, 0, -50],
                    }
                ],
            )
        )
    )
    stats = session.voxel_stats(p.id)
    h = stats.h
    assert stats.n_active > 0 and not [w for w in stats.warnings if "0 nodes" in w]
    rows = session.boundaries(p.id)
    base = nodes_bbox(rows["supports"], "base")
    assert base[:, 2] == pytest.approx([0, 0], abs=h)  # all on the plate underside
    assert base[0, 0] < 0.3 * 80 and base[1, 0] > 0.9 * 80  # the whole plate, not one hole
    pin = nodes_bbox(rows["loads"], "pin")
    assert pin[0] == pytest.approx([0, 24, 29], abs=1.5 * h)  # bore of the Ø12 hole only
    assert pin[1] == pytest.approx([10, 36, 41], abs=1.5 * h)
    # the same selection through `resolve` (what the MCP tool calls)
    out = session.resolve(
        p.id, {"kind": "facets", "facet_ids": [hole["id"]], "angle_deg": 45}
    )  # angle is ignored for STEP meshes
    assert out["count"] == rows["loads"][0]["n_nodes"]


def test_facets_selection_follows_the_design_transform(session: Session, step_path: Path):
    info = session.load_mesh(step_path)
    shift = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 0, 0, 1]  # column-major translation
    case = bracket_case(info.id)
    case["design_mesh"]["transform"] = shift
    p = session.create_project(ProjectIn.model_validate(case))
    base = nodes_bbox(session.boundaries(p.id)["supports"], "base")
    h = session.voxel_stats(p.id).h
    assert (
        base[0, 0] > 100 - h and base[1, 0] < 180 + h and base[:, 2] == pytest.approx([0, 0], abs=h)
    )


def test_unknown_step_facet_id_blocks_the_project(session: Session, step_path: Path):
    info = session.load_mesh(step_path)
    case = bracket_case(info.id)
    case["supports"][0]["selection"]["facet_ids"] = [99]
    p = session.create_project(ProjectIn.model_validate(case))
    assert any("unknown facet ids [99]" in w for w in session.voxel_stats(p.id).warnings)
    project = session.get_project(p.id)
    meshes = load_project_meshes(project, session.store.get_mesh)
    from topop.server.build import ProblemInvalid

    with pytest.raises(ProblemInvalid, match="unknown facet ids"):
        build_problem(project, meshes)


# ---- CLI -----------------------------------------------------------------------------------------


def test_cli_describe_step(step_path: Path, data_dir: Path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["describe", str(step_path), "--top", "3"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "bracket.step" in out and "watertight True" in out
    assert "bbox min (0, 0, 0)  max (80, 60, 60)" in out
