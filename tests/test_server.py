"""API tests: TestClient + WebSocket against a temporary TOPOP_DATA_DIR."""

from __future__ import annotations

import asyncio
import io
import json
import os
import struct
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from topop.agent import Session
from topop.core.export import from_npz_bytes
from topop.core.voxelize import load_mesh
from topop.server import jobs
from topop.server.app import OriginMiddleware, allowed_hosts, app
from topop.server.jobs import GONE, Mailbox, RunManager, density_frame
from topop.server.schemas import ParamsSpec, Project, ProjectIn, RunExport, RunInfo, VoxelStats
from topop.server.store import RunRecord, Store, _write_atomic

PNG = b"\x89PNG\r\n\x1a\n"
TERMINAL = ("done", "error", "cancelled")


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOPOP_DATA_DIR", str(tmp_path))
    with TestClient(app) as client:
        yield client


@pytest.fixture
def cantilever(examples_dir: Path) -> Path:
    return examples_dir / "cantilever.stl"


def upload(client: TestClient, path: Path) -> dict:
    res = client.post("/api/meshes", files={"file": (path.name, path.read_bytes())})
    assert res.status_code == 200, res.text
    return res.json()


def cantilever_project(mesh_id: str, *, supports: bool = True, **params) -> dict:
    """Fixed at x=0, pushed down on the top face beyond x=50."""
    return {
        "name": "cantilever",
        "design_mesh": {"mesh_id": mesh_id},
        "grid": {"elements_along_longest": 16},
        "params": {"max_iter": 5, "density_every": 1, **params},
        "supports": [
            {
                "id": "fixed",
                "name": "wall",
                "selection": {"kind": "plane", "point": [0, 0, 0], "normal": [1, 0, 0]},
                "fix": [True, True, True],
            }
        ]
        if supports
        else [],
        "loads": [
            {
                "id": "tip",
                "name": "tip load",
                "selection": {
                    "kind": "normal",
                    "mesh_id": mesh_id,
                    "direction": [0, 0, 1],
                    "within": [[50, -10, -10], [70, 30, 30]],
                },
                "force": [0, 0, -1],
            }
        ],
    }


def create_project(client: TestClient, body: dict) -> dict:
    res = client.post("/api/projects", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def collect(ws) -> list:
    """Every message until the server closes: dicts for JSON frames, bytes for density frames."""
    out: list = []
    while True:
        msg = ws.receive()
        if msg["type"] == "websocket.close":
            return out
        if msg.get("bytes") is not None:
            out.append(msg["bytes"])
        else:
            out.append(json.loads(msg["text"]))


def statuses(msgs: list) -> list[str]:
    return [m["type"] for m in msgs if isinstance(m, dict) and m["type"] != "progress"]


# ---- meshes -------------------------------------------------------------------------------------


def test_mesh_upload_buffer_adjacency_facets_preview(api: TestClient, cantilever: Path):
    info = upload(api, cantilever)
    assert info["name"] == "cantilever.stl"
    assert info["is_watertight"] is True
    assert info["volume"] == pytest.approx(24000)
    assert np.allclose(info["bbox"], [[0, 0, 0], [60, 20, 20]])
    assert len(info["id"]) == 16
    # same bytes -> same id (the frontend remaps selections after a reload with this)
    assert upload(api, cantilever)["id"] == info["id"]
    mid = info["id"]

    res = api.get(f"/api/meshes/{mid}/buffer")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/octet-stream"
    buf = res.content
    n_vert, n_tri = struct.unpack("<2I", buf[:8])
    assert (n_vert, n_tri) == (info["n_vertices"], info["n_faces"])
    assert len(buf) == 8 + 12 * n_vert + 12 * n_tri + 12 * n_tri
    xyz = np.frombuffer(buf, "<f4", 3 * n_vert, 8).reshape(-1, 3)
    assert np.allclose(xyz.min(0), [0, 0, 0]) and np.allclose(xyz.max(0), [60, 20, 20])
    ijk = np.frombuffer(buf, "<u4", 3 * n_tri, 8 + 12 * n_vert)
    assert ijk.max() < n_vert

    adj = api.get(f"/api/meshes/{mid}/adjacency").content
    expected = load_mesh(cantilever).face_adjacency
    assert len(adj) == expected.size * 4
    assert np.array_equal(np.frombuffer(adj, "<u4").reshape(-1, 2), expected)

    facets = api.get(f"/api/meshes/{mid}/facets", params={"angle_deg": 5}).json()
    assert facets["mesh_id"] == mid
    assert facets["n_facets_total"] == len(facets["facets"]) == 6
    areas = [f["area"] for f in facets["facets"]]
    assert areas == sorted(areas, reverse=True)
    assert areas[0] == pytest.approx(60 * 20)  # the 20x20 +-x ends are the smallest
    assert areas[-1] == pytest.approx(20 * 20)

    png = api.get(f"/api/meshes/{mid}/preview.png", params={"view": "iso"})
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content.startswith(PNG)
    assert api.get(f"/api/meshes/{mid}/preview.png", params={"view": "sideways"}).status_code == 422


def test_mesh_errors(api: TestClient):
    assert api.get("/api/meshes/0123456789abcdef/buffer").status_code == 404
    assert api.get("/api/meshes/not-a-mesh-id/facets").status_code == 404
    bad = api.post("/api/meshes", files={"file": ("junk.stl", b"definitely not a mesh")})
    assert bad.status_code == 400
    assert "detail" in bad.json()


# ---- projects -----------------------------------------------------------------------------------


def test_project_crud_and_persistence(api: TestClient, cantilever: Path, tmp_path: Path):
    mid = upload(api, cantilever)["id"]
    p = create_project(api, cantilever_project(mid))
    Project.model_validate(p)
    assert p["created_at"] == p["updated_at"]
    assert api.get(f"/api/projects/{p['id']}").json() == p
    assert [q["id"] for q in api.get("/api/projects").json()] == [p["id"]]
    assert (tmp_path / "projects" / f"{p['id']}.json").is_file()

    edited = {**p, "name": "renamed", "id": "ignored", "created_at": "x"}
    res = api.put(f"/api/projects/{p['id']}", json=edited)
    assert res.status_code == 200
    q = res.json()
    assert (q["id"], q["name"], q["created_at"]) == (p["id"], "renamed", p["created_at"])
    assert q["updated_at"] >= p["updated_at"]

    assert api.get("/api/projects/nope").status_code == 404
    assert api.put("/api/projects/nope", json=edited).status_code == 404

    # a fresh server on the same data dir sees the project and the mesh
    with TestClient(app) as again:
        assert again.get(f"/api/projects/{p['id']}").json()["name"] == "renamed"
        assert again.get(f"/api/meshes/{mid}/buffer").status_code == 200


def test_voxelize_and_resolve_selection(api: TestClient, cantilever: Path):
    mid = upload(api, cantilever)["id"]
    p = create_project(api, cantilever_project(mid))

    res = api.post(f"/api/projects/{p['id']}/voxelize")
    assert res.status_code == 200, res.text
    stats = VoxelStats.model_validate(res.json())
    assert (stats.nx, stats.ny, stats.nz) == (18, 8, 8)  # 16 x 6 x 6 inner + padding
    assert stats.h == pytest.approx(3.75)
    assert stats.n_active == stats.n_free == 16 * 6 * 6
    assert stats.n_dof == 3 * stats.n_nodes == 3 * 17 * 7 * 7
    assert stats.est_bytes > 0 and stats.est_sec_per_iter > 0
    assert stats.warnings == []

    url = f"/api/projects/{p['id']}/resolve-selection"
    support = api.post(url, json=p["supports"][0]["selection"]).json()
    assert support["count"] == 7 * 7 and not support["truncated"]
    assert np.allclose(np.asarray(support["xyz"])[:, 0], 0)
    load = api.post(url, json=p["loads"][0]["selection"]).json()
    assert load["count"] > 0
    xyz = np.asarray(load["xyz"])
    assert (xyz[:, 0] >= 50).all() and (xyz[:, 2] > 17).all()

    bad = {"kind": "facets", "mesh_id": mid, "facet_ids": [99]}
    assert api.post(url, json=bad).status_code == 422
    assert api.post("/api/projects/nope/voxelize").status_code == 404

    # selections that resolve to nothing are reported by voxelize
    p2 = cantilever_project(mid)
    p2["supports"][0]["selection"]["point"] = [500, 0, 0]
    vid = create_project(api, p2)["id"]
    warnings = api.post(f"/api/projects/{vid}/voxelize").json()["warnings"]
    assert any("resolves to 0 nodes" in w for w in warnings)

    # no design mesh -> 409; unknown mesh -> 404
    empty = create_project(api, {"name": "empty"})
    assert api.post(f"/api/projects/{empty['id']}/voxelize").status_code == 409
    ghost = create_project(api, cantilever_project("feedfacefeedface"))
    assert api.post(f"/api/projects/{ghost['id']}/voxelize").status_code == 404


# ---- runs ---------------------------------------------------------------------------------------


def test_run_stream_exports_and_restart(api: TestClient, cantilever: Path, tmp_path: Path):
    mid = upload(api, cantilever)["id"]
    p = create_project(api, cantilever_project(mid))
    sem = api.app.state.runs.semaphore
    sem.acquire()  # hold the run in the queue so the stream sees every iteration
    try:
        res = api.post("/api/runs", json={"project_id": p["id"]})
        assert res.status_code == 200, res.text
        run = RunInfo.model_validate(res.json())
        assert run.status == "queued"
        assert run.stats is not None and run.stats.n_active == 16 * 6 * 6
        with api.websocket_connect(f"/api/runs/{run.id}/stream") as ws:
            first = ws.receive_json()
            assert first["type"] == "started"
            assert first["run"]["stats"]["n_active"] == 16 * 6 * 6
            sem.release()
            sem = None
            msgs = collect(ws)
    finally:
        if sem is not None:
            sem.release()

    progress = [m for m in msgs if isinstance(m, dict) and m["type"] == "progress"]
    frames = [m for m in msgs if isinstance(m, bytes)]
    assert [m["it"] for m in progress] == [1, 2, 3, 4, 5]
    assert all(m["compliance"] > 0 and 0 < m["volume"] < 1 for m in progress)
    assert len(frames) >= 5
    shape = (18, 8, 8)
    for i, frame in enumerate(frames, start=1):
        assert struct.unpack("<4I", frame[:16]) == (i, *shape)
        assert len(frame) == 16 + 18 * 8 * 8
    rho = np.frombuffer(frames[-1], np.uint8, offset=16).reshape(shape)
    assert rho[0].max() == 0 and rho[-1].max() == 0  # padding layers are inactive
    assert rho.max() > 128
    done = msgs[-1]
    assert done["type"] == "done"
    assert statuses(msgs) == ["started", "done"]  # queued -> running announcement, then done
    assert len(done["run"]["history"]) == 5
    assert done["run"]["finished_at"]

    info = api.get(f"/api/runs/{run.id}").json()
    assert info["status"] == "done" and len(info["history"]) == 5
    assert [r["id"] for r in api.get("/api/runs").json()] == [run.id]

    stl = api.get(f"/api/runs/{run.id}/result.stl", params={"threshold": 0.5, "smooth": 2})
    assert stl.status_code == 200 and stl.headers["content-type"] == "model/stl"
    mesh = trimesh.load(io.BytesIO(stl.content), file_type="stl")
    assert len(mesh.faces) > 0
    assert mesh.bounds[0][0] > -5 and mesh.bounds[1][0] < 65
    vti = api.get(f"/api/runs/{run.id}/result.vti")
    assert vti.headers["content-type"] == "application/xml"
    assert vti.content.startswith(b"<?xml") and b"density" in vti.content
    npz = api.get(f"/api/runs/{run.id}/result.npz")
    assert npz.headers["content-type"] == "application/octet-stream"
    with np.load(io.BytesIO(npz.content)) as z:
        assert z["rho"].shape == shape
    exp = RunExport.model_validate(api.get(f"/api/runs/{run.id}/project.json").json())
    assert exp.project.id == p["id"] and exp.run.status == "done"
    assert RunExport.model_validate_json(exp.model_dump_json()) == exp
    assert exp.project.loads[0].selection.kind == "normal"
    png = api.get(f"/api/runs/{run.id}/preview.png", params={"threshold": 0.5, "view": "+y"})
    assert png.status_code == 200 and png.content.startswith(PNG)
    assert api.get(f"/api/runs/{run.id}/result.stl", params={"threshold": 0}).status_code == 422

    assert (tmp_path / "runs" / f"{run.id}.npz").is_file()
    assert (tmp_path / "runs" / f"{run.id}.json").is_file()
    with TestClient(app) as again:  # restart: run record and exports come back from disk
        info2 = again.get(f"/api/runs/{run.id}").json()
        assert info2["status"] == "done" and len(info2["history"]) == 5
        assert again.get(f"/api/runs/{run.id}/result.npz").content
        assert len(again.get(f"/api/runs/{run.id}/result.stl").content) > 84
        assert again.get(f"/api/runs/{run.id}/preview.png").content.startswith(PNG)
        with again.websocket_connect(f"/api/runs/{run.id}/stream") as ws:
            replay = collect(ws)
        assert statuses(replay) == ["started", "done"]
        assert isinstance(replay[1], bytes) and struct.unpack("<I", replay[1][:4]) == (5,)


def test_finished_run_stream_replays_status(api: TestClient, cantilever: Path):
    mid = upload(api, cantilever)["id"]
    pid = create_project(api, cantilever_project(mid, max_iter=2))["id"]
    rid = api.post("/api/runs", json={"project_id": pid}).json()["id"]
    with api.websocket_connect(f"/api/runs/{rid}/stream") as ws:
        collect(ws)  # live or replayed, whichever the timing gives
    with api.websocket_connect(f"/api/runs/{rid}/stream") as ws:
        msgs = collect(ws)
    assert statuses(msgs) == ["started", "done"]
    assert msgs[0]["run"]["status"] == "done"
    assert struct.unpack("<I", msgs[1][:4]) == (2,)


def test_run_not_runnable_is_422(api: TestClient, cantilever: Path):
    mid = upload(api, cantilever)["id"]
    pid = create_project(api, cantilever_project(mid, supports=False))["id"]
    res = api.post("/api/runs", json={"project_id": pid})
    assert res.status_code == 422
    assert "no supports defined" in res.json()["detail"]
    assert api.get("/api/runs").json() == []
    assert api.post("/api/runs", json={"project_id": "nope"}).status_code == 404
    empty = create_project(api, {"name": "empty"})["id"]
    assert api.post("/api/runs", json={"project_id": empty}).status_code == 409


def test_cancel_running_run(api: TestClient, cantilever: Path):
    mid = upload(api, cantilever)["id"]
    body = cantilever_project(mid, max_iter=200, tol=1e-9, density_every=10)
    pid = create_project(api, body)["id"]
    rid = api.post("/api/runs", json={"project_id": pid}).json()["id"]
    with api.websocket_connect(f"/api/runs/{rid}/stream") as ws:
        assert ws.receive_json()["type"] == "started"
        while True:  # wait until it is actually iterating
            msg = ws.receive()
            if msg.get("text") and json.loads(msg["text"])["type"] == "progress":
                break
        assert api.post(f"/api/runs/{rid}/cancel").status_code == 200
        msgs = collect(ws)
    final = msgs[-1]
    assert final["type"] == "cancelled"
    assert final["run"]["status"] == "cancelled"
    assert 0 < len(final["run"]["history"]) < 200
    info = api.get(f"/api/runs/{rid}").json()
    assert info["status"] == "cancelled" and info["finished_at"]
    # the partial result is still exportable
    assert api.get(f"/api/runs/{rid}/result.npz").status_code == 200


def test_cancel_queued_run_and_409_before_finish(api: TestClient, cantilever: Path):
    mid = upload(api, cantilever)["id"]
    pid = create_project(api, cantilever_project(mid))["id"]
    sem = api.app.state.runs.semaphore
    sem.acquire()
    try:
        rid = api.post("/api/runs", json={"project_id": pid}).json()["id"]
        assert api.get(f"/api/runs/{rid}/result.stl").status_code == 409
        res = api.post(f"/api/runs/{rid}/cancel").json()
        assert res["status"] == "cancelled" and res["history"] == []
    finally:
        sem.release()
    assert api.get(f"/api/runs/{rid}/result.stl").status_code == 409  # never produced a result
    with api.websocket_connect(f"/api/runs/{rid}/stream") as ws:
        assert statuses(collect(ws)) == ["started", "cancelled"]
    assert api.post("/api/runs/nope/cancel").status_code == 404


def test_stream_unknown_run(api: TestClient):
    with api.websocket_connect("/api/runs/nope/stream") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error" and "not found" in msg["message"]


def test_run_error_is_reported(api: TestClient, cantilever: Path, monkeypatch):
    def boom(*args, **kwargs):
        raise MemoryError("576 active elements need about 99 GB; lower the resolution")

    monkeypatch.setattr("topop.server.jobs.optimize", boom)
    mid = upload(api, cantilever)["id"]
    pid = create_project(api, cantilever_project(mid))["id"]
    rid = api.post("/api/runs", json={"project_id": pid}).json()["id"]
    with api.websocket_connect(f"/api/runs/{rid}/stream") as ws:
        msgs = collect(ws)
    assert msgs[-1]["type"] == "error"
    assert msgs[-1]["message"] == "576 active elements need about 99 GB; lower the resolution"
    info = api.get(f"/api/runs/{rid}").json()
    assert info["status"] == "error" and "lower the resolution" in info["error"]
    assert api.get(f"/api/runs/{rid}/result.npz").status_code == 409


def test_mesh_get_info(client, examples_dir):
    with open(examples_dir / "cantilever.stl", "rb") as fh:
        up = client.post("/api/meshes", files={"file": ("cantilever.stl", fh, "model/stl")})
    assert up.status_code == 200
    mid = up.json()["id"]
    r = client.get(f"/api/meshes/{mid}")
    assert r.status_code == 200 and r.json()["n_faces"] == up.json()["n_faces"]
    assert client.get("/api/meshes/0000000000000000").status_code == 404


# ---- facet faces, STEP selections, v0.2 params ---------------------------------------------------


@pytest.fixture(scope="module")
def step_file(examples_dir: Path) -> Path:
    pytest.importorskip("OCP", reason="STEP support not installed (uv sync --extra step)")
    return examples_dir / "bracket.step"


def hole_facet(client: TestClient, mesh_id: str, radius: float = 6.0) -> dict:
    facets = client.get(f"/api/meshes/{mesh_id}/facets").json()["facets"]
    holes = [f for f in facets if f["kind"] == "cylinder" and f["radius"] == pytest.approx(radius)]
    assert len(holes) == 1, holes
    return holes[0]


@pytest.mark.parametrize("name", ["bracket.stl", "bracket.step"])
def test_facet_faces_of_the_bracket_hole(api: TestClient, examples_dir: Path, name: str):
    if name.endswith(".step"):
        pytest.importorskip("OCP", reason="STEP support not installed (uv sync --extra step)")
    info = upload(api, examples_dir / name)
    hole = hole_facet(api, info["id"])
    assert hole["axis"] is not None and hole["n_faces"] > 50
    res = api.get(f"/api/meshes/{info['id']}/facets/{hole['id']}/faces", params={"angle_deg": 5})
    assert res.status_code == 200, res.text
    ids = res.json()["face_ids"]
    assert len(ids) == hole["n_faces"] and len(set(ids)) == len(ids)
    assert 0 <= min(ids) and max(ids) < info["n_faces"]
    # the triangles really are the hole wall: centroids sit 6 from the hole axis
    buf = api.get(f"/api/meshes/{info['id']}/buffer").content
    n_vert, n_tri = struct.unpack("<2I", buf[:8])
    xyz = np.frombuffer(buf, "<f4", 3 * n_vert, 8).reshape(-1, 3).astype(float)
    tri = np.frombuffer(buf, "<u4", 3 * n_tri, 8 + 12 * n_vert).reshape(-1, 3)[ids]
    axis = np.asarray(hole["axis"])
    rel = xyz[tri].mean(1) - np.asarray(hole["centroid"])
    radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
    assert np.allclose(radial, 6.0, atol=0.05)
    # a facet that does not exist, and a mesh that does not exist
    missing = api.get(f"/api/meshes/{info['id']}/facets/9999/faces")
    assert missing.status_code == 404 and "unknown facet" in missing.json()["detail"]
    assert api.get(f"/api/meshes/{info['id']}/facets/-1/faces").status_code == 404
    assert api.get("/api/meshes/0123456789abcdef/facets/0/faces").status_code == 404


def test_step_facet_faces_ignore_angle(api: TestClient, step_file: Path):
    info = upload(api, step_file)
    hole = hole_facet(api, info["id"])
    first = api.get(f"/api/meshes/{info['id']}/facets/{hole['id']}/faces").json()
    wide = api.get(
        f"/api/meshes/{info['id']}/facets/{hole['id']}/faces", params={"angle_deg": 60}
    ).json()
    assert first == wide and len(first["face_ids"]) == hole["n_faces"]


def test_resolve_selection_facets_on_step_mesh(api: TestClient, step_file: Path):
    info = upload(api, step_file)
    assert info["source"] == "step" and info["n_brep_faces"] == 14
    hole = hole_facet(api, info["id"])
    pid = create_project(
        api,
        {
            "name": "step",
            "design_mesh": {"mesh_id": info["id"]},
            "grid": {"elements_along_longest": 40},
        },
    )["id"]
    url = f"/api/projects/{pid}/resolve-selection"
    sel = {"kind": "facets", "mesh_id": info["id"], "facet_ids": [hole["id"]]}
    res = api.post(url, json=sel)
    assert res.status_code == 200, res.text
    nodes = res.json()
    assert nodes["count"] > 0
    # the wall of the Ø12 hole: nodes lie within a voxel of the cylinder surface
    xyz = np.asarray(nodes["xyz"])
    axis, centre = np.asarray(hole["axis"]), np.asarray(hole["centroid"])
    rel = xyz - centre
    radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
    h = api.post(f"/api/projects/{pid}/voxelize").json()["h"]
    assert np.abs(radial - 6.0).max() <= h
    assert api.post(url, json={**sel, "facet_ids": [99]}).status_code == 422


def streamed(client: TestClient, project_id: str) -> tuple[str, list]:
    run = client.post("/api/runs", json={"project_id": project_id})
    assert run.status_code == 200, run.text
    rid = run.json()["id"]
    with client.websocket_connect(f"/api/runs/{rid}/stream") as ws:
        return rid, collect(ws)


def test_symmetry_overhang_run_stress_vti_and_trim(api: TestClient, cantilever: Path, tmp_path):
    mid = upload(api, cantilever)["id"]
    body = cantilever_project(
        mid, max_iter=4, symmetry=[{"axis": "y", "position": None}], overhang="+z"
    )
    pid = create_project(api, body)["id"]
    notes = api.post(f"/api/projects/{pid}/voxelize").json()["warnings"]
    assert any("overhang +z" in w and "base plate" in w and "min z" in w for w in notes)
    rid, msgs = streamed(api, pid)

    progress = [m for m in msgs if isinstance(m, dict) and m["type"] == "progress"]
    assert [m["it"] for m in progress] == [1, 2, 3, 4]
    assert all(isinstance(m["stress_max"], float) and m["stress_max"] > 0 for m in progress)
    assert all(m["constraint"] is None for m in progress)  # no stress_limit
    done = msgs[-1]
    assert done["type"] == "done"
    assert isinstance(done["run"]["history"][-1]["stress_max"], float)

    shape = (18, 8, 8)
    res = api.get(f"/api/runs/{rid}/stress")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/octet-stream"
    assert struct.unpack("<3I", res.content[:12]) == shape
    assert len(res.content) == 12 + 4 * 18 * 8 * 8
    field = np.frombuffer(res.content, "<f4", offset=12).reshape(shape)
    assert np.isfinite(field).all() and field.max() > 0
    assert field[0].max() == field[-1].max() == 0  # padding layers are inactive
    # Result.stress is recomputed at the final design; the last progress stress_max belongs to the
    # design one update earlier, which after only 4 iterations (move 0.2) can differ by 2-3x.
    sm = progress[-1]["stress_max"]
    assert 0.1 * sm < field.max() < 10 * sm

    vti = api.get(f"/api/runs/{rid}/result.vti").content
    root = ET.fromstring(vti)
    arrays = {a.attrib["Name"]: a.attrib["type"] for a in root.iter("DataArray")}
    assert arrays == {"density": "Float32", "passive": "Int8", "stress": "Float32"}
    with np.load(io.BytesIO(api.get(f"/api/runs/{rid}/result.npz").content)) as z:
        assert np.array_equal(z["stress"].astype("<f4"), field)
        rho = z["rho"]
    assert np.abs(rho - rho[:, ::-1, :]).max() < 1e-9  # mirror symmetric in y

    # trim: the AM filter leaves the density low after 4 iterations, so cut relative to its peak
    thr = 0.5 * float(rho.max())
    params = {"threshold": thr, "smooth": 0}
    raw = api.get(f"/api/runs/{rid}/result.stl", params=params)
    trimmed = api.get(f"/api/runs/{rid}/result.stl", params={**params, "trim": True})
    assert raw.status_code == trimmed.status_code == 200
    assert "x-topop-warnings" not in trimmed.headers
    raw_mesh = trimesh.load(io.BytesIO(raw.content), file_type="stl")
    cut = trimesh.load(io.BytesIO(trimmed.content), file_type="stl")
    design_lo, design_hi = np.array([0.0, 0.0, 0.0]), np.array([60.0, 20.0, 20.0])
    assert (raw_mesh.bounds[0] < design_lo - 0.5).any()  # the voxel skin pokes out of the CAD
    assert (cut.bounds[0] >= design_lo - 1e-6).all() and (cut.bounds[1] <= design_hi + 1e-6).all()
    assert cut.is_watertight and 0 < cut.volume < raw_mesh.volume
    png = api.get(f"/api/runs/{rid}/preview.png", params={"threshold": thr, "trim": True})
    assert png.status_code == 200 and png.content.startswith(PNG)
    assert "x-topop-warnings" not in png.headers

    # nothing above the threshold: the empty surface cannot be trimmed -> 200 plus a header
    empty = api.get(f"/api/runs/{rid}/result.stl", params={"threshold": 1.0, "trim": True})
    assert empty.status_code == 200 and len(empty.content) == 84
    assert "not trimmed" in empty.headers["x-topop-warnings"]
    shot = api.get(f"/api/runs/{rid}/preview.png", params={"trim": True})
    assert shot.status_code == 200 and "not trimmed" in shot.headers["x-topop-warnings"]

    with TestClient(app) as again:  # the stress field survives a restart (runs/{id}.npz)
        assert again.get(f"/api/runs/{rid}/stress").content == res.content
        assert b'Name="stress"' in again.get(f"/api/runs/{rid}/result.vti").content


def test_stress_limit_with_oc_forces_mma(api: TestClient, cantilever: Path):
    mid = upload(api, cantilever)["id"]
    body = cantilever_project(mid, max_iter=3, optimizer="oc", stress_limit=5.0, stress_pnorm=6)
    pid = create_project(api, body)["id"]
    notes = api.post(f"/api/projects/{pid}/voxelize").json()["warnings"]
    assert any("mma" in w and "stress_limit" in w for w in notes)
    rid, msgs = streamed(api, pid)
    progress = [m for m in msgs if isinstance(m, dict) and m["type"] == "progress"]
    assert [m["it"] for m in progress] == [1, 2, 3]
    assert all(isinstance(m["constraint"], float) for m in progress)  # only mma has a constraint
    assert all(isinstance(m["stress_max"], float) for m in progress)
    assert msgs[-1]["type"] == "done"
    info = api.get(f"/api/runs/{rid}").json()
    assert isinstance(info["history"][-1]["constraint"], float)
    assert any("mma" in w for w in info["stats"]["warnings"])  # the run's own stats say it too


def test_stress_endpoint_409_until_the_run_has_a_result(api: TestClient, cantilever: Path):
    mid = upload(api, cantilever)["id"]
    pid = create_project(api, cantilever_project(mid, max_iter=2))["id"]
    sem = api.app.state.runs.semaphore
    sem.acquire()
    try:
        rid = api.post("/api/runs", json={"project_id": pid}).json()["id"]
        assert api.get(f"/api/runs/{rid}/stress").status_code == 409  # queued
        api.post(f"/api/runs/{rid}/cancel")
    finally:
        sem.release()
    assert api.get(f"/api/runs/{rid}/stress").status_code == 409  # cancelled, no result
    assert api.get("/api/runs/nope/stress").status_code == 404


# ---- hardening: limits, guards, memory, processes sharing a data dir ----------------------------


def test_stalled_subscriber_holds_one_frame_and_is_dropped(monkeypatch):
    loop = asyncio.new_event_loop()
    try:
        info = RunInfo(id="r", project_id="p", status="running", created_at="t")
        project = Project.model_validate({"id": "p", "created_at": "t", "updated_at": "t"})
        rec = RunRecord(info=info, project=project)
        box = Mailbox(loop)
        rec.subscribers.append(box)
        for it in range(1, 51):  # a client that never reads
            RunManager._push_locked(rec, {"type": "progress", "it": it})
            RunManager._push_locked(rec, density_frame(it, np.full((8, 8, 8), it / 50)))
        held = box.pending()
        frames = [m for m in held if isinstance(m, bytes)]
        assert len(frames) == 1 and struct.unpack("<I", frames[0][:4]) == (50,)  # the newest
        assert [m["it"] for m in held if isinstance(m, dict)] == list(range(1, 51))  # every JSON
        assert held[-1] is frames[0]  # after the progress message it belongs to
        assert loop.run_until_complete(box.get()) == {"type": "progress", "it": 1}

        monkeypatch.setattr(jobs, "STALL_SECONDS", 0.0)  # stands for 30 s without reading
        time.sleep(0.01)
        RunManager._push_locked(rec, {"type": "progress", "it": 51})
        assert rec.subscribers == [] and box.pending() == []  # dropped, memory freed
        assert loop.run_until_complete(box.get()) is GONE
        assert not box.put({"type": "progress", "it": 52})
    finally:
        loop.close()


def test_upload_above_the_size_cap_is_413(api: TestClient, monkeypatch):
    monkeypatch.setenv("TOPOP_MAX_UPLOAD_MB", "1")
    res = api.post("/api/meshes", files={"file": ("big.stl", b"x" * (2**20 + 1))})
    assert res.status_code == 413 and "TOPOP_MAX_UPLOAD_MB" in res.json()["detail"]

    def chunked():  # no Content-Length: refused while it streams in
        yield b'--b\r\nContent-Disposition: form-data; name="file"; filename="big.stl"\r\n\r\n'
        for _ in range(20):
            yield b"x" * 2**17
        yield b"\r\n--b--\r\n"

    headers = {"content-type": "multipart/form-data; boundary=b"}
    res = api.post("/api/meshes", content=chunked(), headers=headers)
    assert res.status_code == 413 and "TOPOP_MAX_UPLOAD_MB" in res.json()["detail"]
    box = trimesh.creation.box().export(file_type="stl")
    assert api.post("/api/meshes", files={"file": ("box.stl", box)}).status_code == 200


def test_cross_site_requests_and_foreign_hosts_are_refused(api: TestClient):
    evil = api.post("/api/projects", json={"name": "x"}, headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403 and "evil.example" in evil.json()["detail"]
    assert api.put("/api/projects/x", json={}, headers={"Origin": "null"}).status_code == 403
    for origin in ("http://localhost:5173", "http://127.0.0.1:8000", "http://[::1]:8000"):
        res = api.post("/api/projects", json={"name": "ok"}, headers={"Origin": origin})
        assert res.status_code == 200, origin
    assert api.post("/api/projects", json={"name": "no origin (curl, MCP)"}).status_code == 200
    assert api.get("/api/projects", headers={"Origin": "https://evil.example"}).status_code == 200
    evil_ws = {"Origin": "https://evil.example"}  # WebSockets are not covered by CORS: checked
    with (
        pytest.raises(WebSocketDisconnect),
        api.websocket_connect("/api/runs/x/stream", headers=evil_ws),
    ):
        pass
    # DNS rebinding: only requests addressed to this machine are answered
    for host, code in (("evil.example", 400), ("localhost:8000", 200), ("[::1]:8000", 200)):
        other = TestClient(app, base_url=f"http://{host}", client=("127.0.0.1", 50000))
        assert other.get("/api/health").status_code == code, host


def test_allowed_hosts_from_the_environment(monkeypatch):
    monkeypatch.setenv("TOPOP_ALLOWED_HOSTS", "box.lan, *.corp.example,")
    hosts = allowed_hosts()
    assert hosts == ["localhost", "127.0.0.1", "[::1]", "box.lan", "*.corp.example"]
    ok = OriginMiddleware(app, hosts)._ok
    assert ok("http://box.lan:8000") and ok("https://cad.corp.example") and ok("http://[::1]")
    assert not ok("https://evil.example") and not ok("null") and not ok("file:///x")


def test_3mf_upload(api: TestClient):
    data = trimesh.creation.box(extents=[10, 20, 30]).export(file_type="3mf")
    res = api.post("/api/meshes", files={"file": ("part.3mf", data)})
    assert res.status_code == 200, res.text
    info = res.json()
    assert info["is_watertight"] and np.allclose(info["bbox"], [[-5, -10, -15], [5, 10, 15]])


def test_reupload_under_another_extension_keeps_the_stored_type(tmp_path: Path):
    data = trimesh.creation.box(extents=[1, 2, 3]).export(file_type="stl")
    store = Store(tmp_path)
    first = store.add_mesh(data, "part.stl")
    again = store.add_mesh(data, "part.ply")  # same bytes, same id
    assert again.id == first.id and again.name == "part.ply"
    assert len(Store(tmp_path).get_mesh(first.id).faces) == 12  # a fresh store still loads it
    Store(tmp_path).add_mesh(data, "copy.obj")  # not in memory: parsed as what it was stored as
    assert Store(tmp_path).mesh_info(first.id).n_faces == 12
    assert sorted(p.name for p in (tmp_path / "meshes").iterdir()) == [
        f"{first.id}.json",
        f"{first.id}.stl",
    ]


def test_stores_on_one_data_dir_see_each_others_projects_and_runs(
    api: TestClient, cantilever: Path, tmp_path: Path
):
    assert api.get("/api/projects").json() == [] and api.get("/api/runs").json() == []
    other = Session(tmp_path)  # e.g. `topop mcp` on the same TOPOP_DATA_DIR
    mesh = other.load_mesh(cantilever)
    body = ProjectIn.model_validate(cantilever_project(mesh.id, max_iter=2))
    project = other.create_project(body)
    assert api.get(f"/api/projects/{project.id}").json()["name"] == "cantilever"
    assert [p["id"] for p in api.get("/api/projects").json()] == [project.id]
    other.set_params(project.id, volfrac=0.4)  # edited in the other process: re-read
    assert api.get(f"/api/projects/{project.id}").json()["params"]["volfrac"] == 0.4

    run = other.run(project.id)
    assert run.status == "done"
    assert api.get(f"/api/runs/{run.id}").json()["status"] == "done"
    assert [r["id"] for r in api.get("/api/runs").json()] == [run.id]
    with np.load(io.BytesIO(api.get(f"/api/runs/{run.id}/result.npz").content)) as z:
        assert z["rho"].shape == (18, 8, 8)

    gui = create_project(api, {"name": "from the GUI"})["id"]  # and the other way round
    assert other.get_project(gui).name == "from the GUI"
    assert gui in {p.id for p in Store(tmp_path).list_projects()}


def test_persist_failure_still_sends_done(api: TestClient, cantilever: Path, monkeypatch):
    def full(self, rec):
        raise MemoryError("np.savez could not allocate")

    monkeypatch.setattr(Store, "persist_run", full)
    mid = upload(api, cantilever)["id"]
    rid, msgs = streamed(api, create_project(api, cantilever_project(mid, max_iter=2))["id"])
    assert msgs[-1]["type"] == "done"
    assert "result not persisted: np.savez could not allocate" in msgs[-1]["message"]
    info = api.get(f"/api/runs/{rid}").json()
    assert info["status"] == "done" and "result not persisted" in info["message"]
    assert api.get(f"/api/runs/{rid}/result.npz").status_code == 200  # still in memory


def test_finished_run_releases_its_arrays(api: TestClient, cantilever: Path):
    mid = upload(api, cantilever)["id"]
    rid, msgs = streamed(api, create_project(api, cantilever_project(mid, max_iter=2))["id"])
    assert msgs[-1]["type"] == "done"
    rec = api.app.state.store.get_run(rid)
    assert rec.rho is None and rec.built is None and rec.stress is None
    assert rec.latest_frame is None
    for url in ("result.npz", "result.vti", "stress", "preview.png"):
        assert api.get(f"/api/runs/{rid}/{url}").status_code == 200, url
    stl = api.get(f"/api/runs/{rid}/result.stl", params={"threshold": 0.2, "trim": True})
    assert stl.status_code == 200 and len(stl.content) > 84
    with api.websocket_connect(f"/api/runs/{rid}/stream") as ws:  # a late joiner
        replay = collect(ws)
    assert statuses(replay) == ["started", "done"] and struct.unpack("<I", replay[1][:4]) == (2,)


def test_result_request_in_the_finish_window_waits_for_the_result(
    api: TestClient, cantilever: Path, tmp_path: Path
):
    mid = upload(api, cantilever)["id"]
    rid, _ = streamed(api, create_project(api, cantilever_project(mid, max_iter=2))["id"])
    store = api.app.state.store
    rec = store.get_run(rid)
    npz = tmp_path / "runs" / f"{rid}.npz"
    rho, grid, active, passive = from_npz_bytes(npz.read_bytes())
    npz.unlink()  # as in `_finish` before the files exist
    store._results.clear()
    codes: list[int] = []

    def request() -> None:
        codes.append(api.get(f"/api/runs/{rid}/result.npz").status_code)

    with rec.lock:  # `_finish` sets the status, then the result, under this lock
        rec.info.status = "done"
        reader = threading.Thread(target=request)
        reader.start()
        time.sleep(0.3)
        rec.rho, rec.built = rho, SimpleNamespace(grid=grid, active=active, passive=passive)
    reader.join(30)
    assert codes == [200]


def test_base_exception_in_the_worker_fails_the_run(api: TestClient, cantilever: Path, monkeypatch):
    class Abort(BaseException):
        pass

    def abort(*args, **kwargs):
        raise Abort("worker torn down")

    monkeypatch.setattr("topop.server.jobs.optimize", abort)
    mid = upload(api, cantilever)["id"]
    pid = create_project(api, cantilever_project(mid, max_iter=2))["id"]
    rid, msgs = streamed(api, pid)
    assert msgs[-1]["type"] == "error" and "Abort" in msgs[-1]["message"]
    assert api.get(f"/api/runs/{rid}").json()["status"] == "error"
    runs = api.app.state.runs
    assert runs.semaphore.acquire(timeout=10)  # released for the next run
    runs.semaphore.release()

    def leave(*args, **kwargs):
        raise SystemExit("bye")

    monkeypatch.setattr("topop.server.jobs.optimize", leave)  # re-raised, after marking the run
    store = api.app.state.store
    rec = store.new_run(store.get_project(pid), SimpleNamespace(active=None, problem=None), None)
    with pytest.raises(SystemExit):
        runs._worker(rec, rec.built, ParamsSpec())
    assert rec.info.status == "error" and "SystemExit: bye" in rec.info.error
    assert runs.semaphore.acquire(timeout=1)
    runs.semaphore.release()


def test_too_many_queued_runs_is_429(api: TestClient, cantilever: Path, monkeypatch):
    monkeypatch.setenv("TOPOP_MAX_QUEUED", "1")
    mid = upload(api, cantilever)["id"]
    pid = create_project(api, cantilever_project(mid, max_iter=2))["id"]
    sem = api.app.state.runs.semaphore
    sem.acquire()  # a run in progress: new ones queue
    try:
        first = api.post("/api/runs", json={"project_id": pid}).json()
        assert first["status"] == "queued"
        refused = api.post("/api/runs", json={"project_id": pid})
        assert refused.status_code == 429 and "TOPOP_MAX_QUEUED=1" in refused.json()["detail"]
        api.post(f"/api/runs/{first['id']}/cancel")
        second = api.post("/api/runs", json={"project_id": pid})
        assert second.status_code == 200  # room again
        api.post(f"/api/runs/{second.json()['id']}/cancel")
        assert api.app.state.store.get_run(first["id"]).built is None  # cancelled: released
    finally:
        sem.release()


def test_store_start_removes_stale_temp_files(tmp_path: Path, monkeypatch):
    Store(tmp_path)
    stale = tmp_path / "projects" / ".abc.json.deadbeef.tmp"
    fresh = tmp_path / "runs" / ".r.npz.cafebabe.tmp"  # may be another process writing now
    stale.write_bytes(b"x")
    fresh.write_bytes(b"y")
    os.utime(stale, (time.time() - 3600,) * 2)
    Store(tmp_path)
    assert not stale.exists() and fresh.exists()

    def no_space(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", no_space)  # written data is synced before the rename
    with pytest.raises(OSError, match="No space"):
        _write_atomic(tmp_path / "x.json", b"{}")
    assert not (tmp_path / "x.json").exists() and not list(tmp_path.glob(".x.json.*"))
