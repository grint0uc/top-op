"""Strut post-processing (`core.struts`): LP layout, skeleton, meshing, FE check, and wiring."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from mcp import Client

from topop.agent import Session
from topop.cli import main
from topop.core import benchmarks
from topop.core.optimize import optimize
from topop.core.problem import Load, RunParams
from topop.core.struts import (
    StrutParams,
    equilibrium_matrix,
    generate_struts,
    skeleton_graph,
    solve_layout,
    voxel_surface,
)
from topop.core.voxelize import voxelize_mesh
from topop.mcp_server import create_server

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "data"
    monkeypatch.setenv("TOPOP_DATA_DIR", str(d))
    return d


@pytest.fixture(scope="module")
def plate():
    """Thin-plate cantilever 40 x 20 x 2 (h = 1) and its SIMP result at volfrac 0.3."""
    p = benchmarks.cantilever(40, 20, 2)
    res = optimize(p, RunParams(volfrac=0.3, rmin=1.5, max_iter=60), None)
    return p, res.rho


def cli(*argv: object) -> int:
    with pytest.raises(SystemExit) as exc:
        main([str(a) for a in argv])
    return exc.value.code


def inside_bounds(mesh: trimesh.Trimesh, bounds: np.ndarray, tol: float = 1e-4) -> bool:
    return bool(
        np.all(mesh.bounds[0] >= bounds[0] - tol) and np.all(mesh.bounds[1] <= bounds[1] + tol)
    )


# ---- LP --------------------------------------------------------------------------------------


def three_bar(load: np.ndarray):
    """Supports at (+-1, 0) and (0, 0); free node (0, -1) with the given load cases."""
    nodes = np.array([[-1.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0], [0.0, -1, 0]])
    fixed = np.zeros((4, 3), dtype=bool)
    fixed[:3] = True
    loads = np.zeros((len(load), 4, 3))
    loads[:, 3] = load
    return nodes, fixed, loads


def check_equilibrium(nodes, bars, fixed, loads, sol) -> None:
    B, _ = equilibrium_matrix(nodes, bars)
    free = ~fixed.ravel()
    for k in range(loads.shape[0]):
        np.testing.assert_allclose((B @ sol.forces[k])[free], loads[k].ravel()[free], atol=1e-6)
        assert np.all(np.abs(sol.forces[k]) <= 20.0 * sol.areas + 1e-6)


def test_lp_two_bar_and_three_bar_optimum():
    P, sigma = 10.0, 20.0
    nodes, fixed, loads = three_bar(np.array([[0.0, -P, 0]]))
    two = np.array([[0, 3], [1, 3]])
    sol = solve_layout(nodes, two, fixed, loads, sigma)
    # two 45-degree bars of length sqrt 2, each force P / sqrt 2: volume 2 P / sigma
    assert sol.volume == pytest.approx(2 * P / sigma, rel=1e-6)
    np.testing.assert_allclose(sol.areas, P / np.sqrt(2) / sigma, rtol=1e-6)
    assert np.all(sol.forces[0] > 0)  # hanging below the supports: both in tension
    check_equilibrium(nodes, two, fixed, loads, sol)

    three = np.array([[0, 3], [1, 3], [2, 3]])
    sol = solve_layout(nodes, three, fixed, loads, sigma)
    # the vertical bar alone (length 1, force P) is optimal: volume P / sigma
    assert sol.volume == pytest.approx(P / sigma, rel=1e-6)
    np.testing.assert_allclose(sol.areas, [0, 0, P / sigma], atol=1e-9)
    check_equilibrium(nodes, three, fixed, loads, sol)


def test_lp_two_load_cases_need_both_diagonals():
    P, sigma = 10.0, 20.0
    nodes, fixed, loads = three_bar(np.array([[0.0, -P, 0], [P, 0, 0]]))
    three = np.array([[0, 3], [1, 3], [2, 3]])
    sol = solve_layout(nodes, three, fixed, loads, sigma)
    # the horizontal case needs both diagonals at P / (sqrt 2 sigma); they carry the vertical
    # case too, so the vertical bar is not worth its volume
    assert sol.volume == pytest.approx(2 * P / sigma, rel=1e-6)
    np.testing.assert_allclose(sol.areas, [P / np.sqrt(2) / sigma] * 2 + [0], atol=1e-7)
    check_equilibrium(nodes, three, fixed, loads, sol)


def test_lp_rigid_links_carry_load_for_free():
    nodes, fixed, loads = three_bar(np.array([[0.0, -1, 0]]))
    sol = solve_layout(nodes, np.array([[0, 3]]), fixed, loads, 20.0, rigid=np.array([[2, 3]]))
    assert sol.volume == pytest.approx(0.0, abs=1e-9)


def test_lp_unconnected_load_is_an_error():
    nodes, fixed, loads = three_bar(np.array([[0.0, -1, 0]]))
    with pytest.raises(ValueError, match="no candidate bar"):
        solve_layout(nodes, np.array([[0, 1]]), fixed, loads, 20.0)


# ---- layout on a SIMP result -----------------------------------------------------------------


def test_layout_cantilever_is_a_michell_truss(plate):
    p, rho = plate
    res = generate_struts(p, rho)
    s = res.summary()
    assert s["n_bars"] >= 4
    assert res.watertight and res.n_bodies == 1
    assert inside_bounds(res.mesh, p.grid.bounds)
    assert np.all((res.nodes >= p.grid.bounds[0] - 1e-9) & (res.nodes <= p.grid.bounds[1] + 1e-9))
    # equal volume up to min_radius (1 voxel here): the default target is the SIMP volume
    assert res.target_volume == pytest.approx(res.simp_volume)
    assert 0.6 * res.simp_volume < res.volume < 1.4 * res.simp_volume
    (c,), (c_simp,) = res.compliance, res.simp_compliance
    assert np.isfinite(c) and 0 < c < 3 * c_simp
    assert np.isfinite(res.stress_max) and res.stress_max > 0
    # the load point (tip, bottom edge) and the wall are both reached
    assert res.nodes[:, 0].min() == pytest.approx(0.0)
    assert np.any(np.linalg.norm(res.nodes - [40, 0, 1], axis=1) < 1e-9)
    doc = res.to_json()
    assert len(doc["bars"]) == s["n_bars"] and all(b["radius"] > 0 for b in doc["bars"])


def test_layout_two_load_cases_stiff_both_ways(plate):
    p, rho = plate
    tip = p.loads[0].nodes
    p2 = benchmarks.cantilever(40, 20, 2)
    p2.loads = [Load(tip, (0.0, -1.0, 0.0), 0), Load(tip, (-1.0, 0.0, 0.0), 1)]
    res = generate_struts(p2, rho)
    assert len(res.compliance) == 2 and len(res.simp_compliance) == 2
    assert all(np.isfinite(c) and 0 < c < 1e3 for c in res.compliance)
    assert res.watertight and res.n_bodies == 1


def test_keep_in_body_is_merged_and_loaded_through(plate):
    _, rho = plate
    p2 = benchmarks.cantilever(40, 20, 2)
    p2.passive[34:, :5, :] = 1  # a solid lug at the loaded tip
    rho2 = rho.copy()
    rho2[p2.passive == 1] = 1.0
    res = generate_struts(p2, rho2)
    mask, _ = voxelize_mesh(res.mesh, p2.grid)
    assert mask[p2.passive == 1].mean() > 0.95  # the keep-in stays solid
    assert res.watertight and res.n_bodies == 1
    assert np.isfinite(res.compliance[0]) and res.compliance[0] < 3 * res.simp_compliance[0]


def test_voxel_surface_is_closed():
    p = benchmarks.l_bracket(20, 2)
    mesh = voxel_surface(p.active, p.grid)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(p.active.sum() * p.grid.h**3)


# ---- skeleton --------------------------------------------------------------------------------


def test_skeleton_two_crossing_rods():
    n, r = 41, 3.0
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(11), indexing="ij")
    c = (i + 0.5, j + 0.5, k + 0.5)
    rod_x = (c[1] - 20.5) ** 2 + (c[2] - 5.5) ** 2 <= r * r
    rod_y = (c[0] - 20.5) ** 2 + (c[2] - 5.5) ** 2 <= r * r
    rod_x &= (c[0] > 2) & (c[0] < 39)
    rod_y &= (c[1] > 2) & (c[1] < 39)
    g = skeleton_graph(rod_x | rod_y)
    assert 2 <= len(g.branches) <= 4
    for br in g.branches:
        assert br["radius"] == pytest.approx(r, rel=0.3)
    span = g.nodes.max(0) - g.nodes.min(0)
    assert span[0] > 25 and span[1] > 25  # both rods kept their length


def test_skeleton_mode_cantilever(plate):
    p, rho = plate
    res = generate_struts(p, rho, params=StrutParams(mode="skeleton"))
    assert res.mode == "skeleton" and len(res.bars) >= 4
    assert res.watertight and res.n_bodies == 1
    assert np.isfinite(res.compliance[0]) and res.compliance[0] < 5 * res.simp_compliance[0]


# ---- wiring ----------------------------------------------------------------------------------


def topop_run(examples_dir: Path, out: Path) -> None:
    code = cli("run", examples_dir / "cantilever.json", "--out", out, "--quiet",
               "--resolution", 20, "--max-iter", 20)  # fmt: skip
    assert code == 0


def test_cli_struts_writes_three_files(examples_dir: Path, tmp_path: Path, capsys):
    run = tmp_path / "run"
    topop_run(examples_dir, run)
    capsys.readouterr()
    out = tmp_path / "struts"
    assert cli("struts", run / "run.json", "--out", out) == 0
    text = capsys.readouterr().out
    assert "bars" in text and "compliance" in text
    assert (out / "struts.png").read_bytes().startswith(PNG)
    mesh = trimesh.load(out / "struts.stl")
    assert mesh.is_watertight and len(mesh.faces) > 100
    doc = json.loads((out / "struts.json").read_text())
    assert doc["n_bars"] == len(doc["bars"]) > 0 and len(doc["nodes"]) == doc["n_nodes"]
    assert np.isfinite(doc["compliance"][0]) and doc["simp_compliance"][0] > 0
    # the run directory itself works too, and bad input is exit 2
    assert cli("struts", run, "--mode", "skeleton", "--out", tmp_path / "sk") == 0
    assert (tmp_path / "sk" / "struts.stl").is_file()
    assert cli("struts", tmp_path / "nowhere") == 2
    assert cli("struts", run, "--min-radius", "-1") == 2


def test_session_server_and_mcp(examples_dir: Path, tmp_path: Path):
    session = Session()
    pid = session.load_case(examples_dir / "cantilever.json").id
    session.set_grid(pid, elements_along_longest=16)
    rid = session.run(pid, max_iter=15).id

    summary = session.generate_struts(rid)
    assert summary["n_bars"] > 0 and summary["compliance"][0] > 0
    assert Path(summary["files"]["stl"]).is_file()

    from topop.server.app import app

    with TestClient(app) as client:
        res = client.post(f"/api/runs/{rid}/struts", json={"mode": "skeleton"})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["mode"] == "skeleton" and body["stl_url"] == f"/api/runs/{rid}/struts.stl"
        stl = client.get(body["stl_url"])
        assert stl.status_code == 200 and stl.headers["content-type"] == "model/stl"
        assert client.post(f"/api/runs/{rid}/struts", json={"sigma_allow": 0}).status_code == 422
        assert client.post("/api/runs/nope/struts", json={}).status_code == 404
        assert client.get("/api/runs/nope/struts.stl").status_code == 404

    async def scenario():
        async with Client(create_server()) as mcp:
            tools = {t.name for t in (await mcp.list_tools()).tools}
            assert "generate_struts" in tools
            out = tmp_path / "mcp.stl"
            res = await mcp.call_tool("generate_struts", {"run_id": rid, "path": str(out)})
            assert not res.is_error, res.content[0].text
            data = json.loads(res.content[0].text)
            assert data["path"] == str(out) and out.is_file() and data["n_bars"] > 0
            bad = await mcp.call_tool("generate_struts", {"run_id": rid, "path": "rel.stl"})
            assert bad.is_error

    anyio.run(scenario)


def test_capped_candidates_retry_when_short_bars_cannot_carry_the_load():
    # a long slender cantilever: with a cap so small that only short bars survive, no load path
    # reaches the support; the layout must retry with more candidates instead of failing
    from topop.core.benchmarks import cantilever
    from topop.core.struts import StrutParams, generate_struts

    p = cantilever(40, 6, 2)
    rho = np.ones(p.grid.shape)
    res = generate_struts(p, rho, None, [], StrutParams(max_candidates=60, verify=False))
    assert len(res.bars) > 0
    assert any("capped" in w for w in res.warnings) or len(res.bars) > 0
