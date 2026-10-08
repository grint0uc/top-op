"""MCP server: an in-process client drives the whole workflow against the MCPServer object."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import anyio
import numpy as np
import pytest
import trimesh
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from topop.mcp_server import create_server

PNG = b"\x89PNG\r\n\x1a\n"
EXPECTED_TOOLS = {
    "load_mesh", "describe_mesh", "preview_mesh", "create_project", "get_project", "list_projects",
    "set_params", "set_grid", "set_material", "add_load", "add_support", "add_ref_model",
    "remove_load", "remove_support", "remove_ref_model", "voxel_stats", "resolve_selection", "run",
    "get_run", "cancel_run", "result_preview", "export_stl", "export_files", "export_case",
    "load_case", "facet_faces", "result_stress_summary",
}  # fmt: skip
SUPPORT = {"kind": "plane", "point": [0, 0, 0], "normal": [1, 0, 0]}
TIP = {"kind": "normal", "direction": [0, 0, 1], "within": [[48, -4, 16], [64, 24, 24]]}


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "data"
    monkeypatch.setenv("TOPOP_DATA_DIR", str(d))
    return d


class Agent:
    """Thin wrapper over the MCP client: dict results parsed, errors raised."""

    def __init__(self, client: Client):
        self.client = client

    async def raw(self, tool: str, /, **args):
        return await self.client.call_tool(tool, args)

    async def __call__(self, tool: str, /, **args) -> dict:
        res = await self.raw(tool, **args)
        text = res.content[0].text
        assert not res.is_error, f"{tool} failed: {text}"
        return json.loads(text)

    async def fails(self, tool: str, /, **args) -> str:
        res = await self.raw(tool, **args)
        assert res.is_error, f"{tool} should have failed"
        return res.content[0].text


def drive(scenario):
    """Run `scenario(agent)` against a fresh in-process server."""

    async def main():
        async with Client(create_server()) as client:
            return await scenario(Agent(client))

    return anyio.run(main)


async def setup_cantilever(agent: Agent, examples_dir: Path, **project) -> str:
    mesh = await agent("load_mesh", path=str(examples_dir / "cantilever.stl"))
    made = await agent(
        "create_project",
        name="cantilever",
        design_mesh_id=mesh["id"],
        **{"elements_along_longest": 16, "max_iter": 50, **project},
    )
    pid = made["project_id"]
    await agent("add_support", project_id=pid, selection=SUPPORT)
    await agent("add_load", project_id=pid, selection=TIP, force=[0, 0, -1])
    return pid


def test_all_tools_listed_with_manual_style_descriptions():
    async def scenario(agent: Agent):
        tools = {t.name: t for t in (await agent.client.list_tools()).tools}
        assert EXPECTED_TOOLS <= set(tools)
        assert all(t.description and len(t.description) > 30 for t in tools.values())
        for name in ("add_load", "add_support", "resolve_selection"):
            doc = tools[name].description
            for needle in ("facets", "normal", "plane", "box", "sphere", "cylinder", "COLUMN-MAJOR",
                           "local Y", "within", "pad", "voxel"):  # fmt: skip
                assert needle in doc, (name, needle)
        assert (
            "Units" in tools["load_mesh"].description or "units" in tools["load_mesh"].description
        )
        assert "150k" in tools["create_project"].description
        params = tools["set_params"].description
        for needle in ("symmetry", "stress_limit", "stress_pnorm", "overhang", "base plate",
                       "mma", "position", "forces optimizer mma"):  # fmt: skip
            assert needle in params, needle
        assert "trim" in tools["export_stl"].description
        assert "design mesh" in tools["export_stl"].description
        describe = tools["describe_mesh"].description
        for needle in ("cylinder", "radius", "axis", "B-rep", "brep_face", "hole"):
            assert needle in describe, needle
        assert "B-rep" in tools["add_support"].description  # SELECTION_HELP: STEP facet ids
        assert "stress_max" in tools["run"].description and "constraint" in tools["run"].description
        assert "mean_solid" in tools["result_stress_summary"].description
        assert "facet_id" in tools["facet_faces"].description

    drive(scenario)


def test_full_workflow(examples_dir, tmp_path):
    async def scenario(agent: Agent):
        mesh = await agent("load_mesh", path=str(examples_dir / "cantilever.stl"))
        assert len(mesh["id"]) == 16 and mesh["is_watertight"]
        assert (await agent("load_mesh", path=str(examples_dir / "cantilever.stl")))["id"] == mesh[
            "id"
        ]

        desc = await agent("describe_mesh", mesh_id=mesh["id"])
        assert desc["n_facets_total"] == 6 and len(desc["facets"]) == 6
        first = desc["facets"][0]
        assert first["area"] == pytest.approx(1200) and len(first["bbox"]) == 2

        shot = await agent.raw("preview_mesh", mesh_id=mesh["id"], view="+x")
        assert shot.content[0].type == "image"

        project = await agent("create_project", name="t", design_mesh_id=mesh["id"],
                              elements_along_longest=16, max_iter=50)  # fmt: skip
        pid = project["project_id"]
        support = await agent("add_support", project_id=pid, selection=SUPPORT, name="wall")
        assert support["support"]["id"] == "support1" and support["resolved"]["count"] == 49
        load = await agent("add_load", project_id=pid, selection=TIP, force=[0, 0, -1])
        assert load["load"]["selection"]["mesh_id"] == "design"  # defaulted
        assert load["resolved"]["count"] == 28
        assert load["resolved"]["bbox"][0][2] == pytest.approx(21.25)  # top face, h = 3.75

        stats = await agent("voxel_stats", project_id=pid)
        assert (stats["nx"], stats["ny"], stats["nz"]) == (18, 8, 8) and stats["h"] == 3.75
        assert stats["n_active"] == 576 and stats["warnings"] == []
        assert [b["n_nodes"] for b in stats["boundaries"]["loads"]] == [28]
        assert [b["n_nodes"] for b in stats["boundaries"]["supports"]] == [49]

        seen = []

        async def on_progress(progress, total, message):
            seen.append((progress, total))

        res = await agent.client.call_tool(
            "run", {"project_id": pid, "max_iter": 4}, progress_callback=on_progress
        )
        assert not res.is_error, res.content[0].text
        run = json.loads(res.content[0].text)
        assert run["status"] == "done" and run["outcome"] == "max_iter"
        assert run["iterations"] == 4 and [h["it"] for h in run["history"]] == [1, 2, 3, 4]
        assert run["compliance"]["last"] < run["compliance"]["first"]
        assert run["volume"] == pytest.approx(0.3, abs=1e-3)
        assert seen[-1] == (4, 4)  # one progress notification per iteration
        # max_iter was for this run only
        project_after = await agent("get_project", project_id=pid)
        assert project_after["params"]["max_iter"] == 50
        assert (await agent("get_run", run_id=run["run_id"]))["status"] == "done"

        preview = await agent.raw("result_preview", run_id=run["run_id"], view="iso")
        assert not preview.is_error
        image = preview.content[0]
        assert image.type == "image" and image.mime_type == "image/png"
        assert base64.b64decode(image.data).startswith(PNG)
        assert "orange" in preview.content[1].text

        out = tmp_path / "exports" / "beam.stl"
        written = await agent("export_stl", run_id=run["run_id"], path=str(out), smooth=2)
        assert Path(written["path"]) == out and written["bytes"] == out.stat().st_size
        assert len(trimesh.load(out, force="mesh").faces) == written["triangles"] >= 1

        files = await agent("export_files", run_id=run["run_id"], directory=str(tmp_path / "all"))
        assert set(files["files"]) == {"result.stl", "result.png", "result.vti", "density.npz",
                                       "run.json"} and files["errors"] == {}  # fmt: skip

        case = await agent("export_case", project_id=pid, path=str(tmp_path / "case.json"))
        reloaded = await agent("load_case", path=case["path"])
        assert reloaded["project_id"] != pid and len(reloaded["loads"]) == 1
        assert reloaded["design_mesh"]["mesh_id"] == mesh["id"]

    drive(scenario)


def test_editing_tools(examples_dir):
    async def scenario(agent: Agent):
        pid = await setup_cantilever(agent, examples_dir)
        assert (await agent("set_params", project_id=pid, volfrac=0.4, continuation=True))[
            "params"
        ]["volfrac"] == 0.4
        assert (await agent("set_grid", project_id=pid, elements_along_longest=20))["grid"][
            "elements_along_longest"
        ] == 20
        assert (await agent("set_material", project_id=pid, E=210000.0))["material"]["E"] == 210000
        assert "volfrac" in await agent.fails("set_params", project_id=pid, volfrac=2.0)
        await agent("add_load", project_id=pid, selection=TIP, force=[0, 0, -2], case=1, name="b")
        project = await agent("get_project", project_id=pid)
        assert [x["id"] for x in project["loads"]] == ["load1", "load2"]
        assert (await agent("remove_load", project_id=pid, load_id="load2"))["loads"][0][
            "id"
        ] == "load1"
        assert "no load" in await agent.fails("remove_load", project_id=pid, load_id="load2")
        assert (await agent("remove_support", project_id=pid, support_id="support1"))[
            "supports"
        ] == []
        assert "no supports defined" in await agent.fails("run", project_id=pid)
        assert [p["project_id"] for p in (await agent("list_projects"))["projects"]] == [pid]

    drive(scenario)


def test_selection_shorthand_and_ref_models(examples_dir):
    async def scenario(agent: Agent):
        pid = await setup_cantilever(agent, examples_dir, elements_along_longest=30)
        end = await agent("resolve_selection", project_id=pid,
                          selection={"kind": "box", "min": [59, -1, -1], "max": [61, 21, 21]})  # fmt: skip
        assert end["count"] == 121 and end["bbox"][0][0] == end["bbox"][1][0] == 60
        rod = await agent("resolve_selection", project_id=pid,
                          selection={"kind": "cylinder", "center": [60, 10, 10], "radius": 5,
                                     "height": 4, "axis": "x"})  # fmt: skip
        assert 0 < rod["count"] < end["count"]
        none = await agent("add_support", project_id=pid,
                           selection={"kind": "sphere", "center": [500, 0, 0], "radius": 1})  # fmt: skip
        assert none["resolved"]["count"] == 0 and "pad" in none["resolved"]["hint"]
        assert "kind" in await agent.fails("resolve_selection", project_id=pid,
                                           selection={"kind": "blob"})  # fmt: skip
        assert "unknown mesh" in (await agent.fails("resolve_selection", project_id=pid,
                                  selection={"kind": "facets", "mesh_id": "zz", "facet_ids": [0]}))  # fmt: skip

        before = (await agent("voxel_stats", project_id=pid))["n_active"]
        mesh = await agent("load_mesh", path=str(examples_dir / "cantilever.stl"))
        # a 10 x 10 x 20 block of the beam as a keep-out: scale (1/6, .5, 1), move to x = 20
        t = [1 / 6, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1, 0, 20, 5, 0, 1]
        ref = await agent("add_ref_model", project_id=pid, mesh_id=mesh["id"], mode="keep_out",
                          transform=t, name="hole")  # fmt: skip
        assert ref["ref_model"]["id"] == "ref1" and ref["selection_mesh_id"] == "ref:ref1"
        assert (await agent("voxel_stats", project_id=pid))["n_active"] < before
        unknown = await agent.fails(
            "add_ref_model", project_id=pid, mesh_id="0" * 16, mode="keep_in"
        )
        assert "not found" in unknown
        assert (await agent("remove_ref_model", project_id=pid, ref_id="ref1"))["ref_models"] == []
        assert (await agent("voxel_stats", project_id=pid))["n_active"] == before

    drive(scenario)


def test_errors_are_readable(examples_dir):
    async def scenario(agent: Agent):
        assert "not found" in await agent.fails("load_mesh", path="/no/such/file.stl")
        assert "not found" in await agent.fails("describe_mesh", mesh_id="deadbeefdeadbeef")
        assert "not found" in await agent.fails("get_project", project_id="nope")
        mesh = await agent("load_mesh", path=str(examples_dir / "cantilever.stl"))
        bad_view = await agent.fails("preview_mesh", mesh_id=mesh["id"], view="sideways")
        assert "unknown view" in bad_view and "iso" in bad_view
        pid = (await agent("create_project", name="x", design_mesh_id=mesh["id"],
                           elements_along_longest=12))["project_id"]  # fmt: skip
        msg = await agent.fails("add_load", project_id=pid, selection=TIP, force=[0, 0])
        assert "force" in msg and "invalid input" in msg
        msg = await agent.fails("run", project_id=pid)
        assert "project is not runnable" in msg and "no loads defined" in msg
        assert "not found" in await agent.fails("result_preview", run_id="missing")

    drive(scenario)


def test_set_params_v02_fields_set_and_clear(examples_dir):
    async def scenario(agent: Agent):
        pid = await setup_cantilever(agent, examples_dir)
        new = (
            await agent(
                "set_params",
                project_id=pid,
                optimizer="mma",
                symmetry=[{"axis": "y", "position": None}, {"axis": "z", "position": 10.0}],
                stress_limit=5.0,
                stress_pnorm=6.0,
                overhang="+z",
            )
        )["params"]
        assert new["optimizer"] == "mma" and new["stress_limit"] == 5.0
        assert new["stress_pnorm"] == 6.0 and new["overhang"] == "+z"
        assert new["symmetry"] == [{"axis": "y", "position": None}, {"axis": "z", "position": 10.0}]
        # omitted fields keep their value
        assert (await agent("set_params", project_id=pid, volfrac=0.4))["params"][
            "overhang"
        ] == "+z"
        notes = (await agent("voxel_stats", project_id=pid))["warnings"]
        assert any("overhang +z" in w and "base plate" in w for w in notes)
        assert not any("forced to mma" in w for w in notes)  # optimizer is already mma
        # 0 / "none" / [] remove a feature
        gone = (
            await agent("set_params", project_id=pid, stress_limit=0, overhang="none", symmetry=[])
        )["params"]
        assert gone["stress_limit"] is None and gone["overhang"] is None and gone["symmetry"] == []
        assert gone["optimizer"] == "mma" and gone["stress_pnorm"] == 6.0
        # oc + stress_limit: valid, with a note
        await agent("set_params", project_id=pid, optimizer="oc", stress_limit=2.0)
        notes = (await agent("voxel_stats", project_id=pid))["warnings"]
        assert any("forced to mma" in w for w in notes)
        # errors are readable
        assert "stress_limit" in await agent.fails("set_params", project_id=pid, stress_limit=-1.0)
        assert "axis" in await agent.fails(
            "set_params", project_id=pid, symmetry=[{"axis": "w", "position": None}]
        )
        assert "stress_pnorm" in await agent.fails("set_params", project_id=pid, stress_pnorm=1.0)
        assert (await agent("get_project", project_id=pid))["params"]["stress_limit"] == 2.0

    drive(scenario)


def test_run_summary_stress_trim_and_stress_summary(examples_dir, tmp_path):
    async def scenario(agent: Agent):
        pid = await setup_cantilever(agent, examples_dir)
        await agent("set_params", project_id=pid, symmetry=[{"axis": "y", "position": None}])
        plain = await agent("run", project_id=pid, max_iter=6)
        assert isinstance(plain["stress_max"], float) and plain["stress_max"] > 0
        assert plain["constraint"] is None  # no stress_limit
        assert all(isinstance(h["stress_max"], float) for h in plain["history"])
        assert "result_stress_summary" in plain["next"]

        s = await agent("result_stress_summary", run_id=plain["run_id"])
        assert set(s) >= {"max", "mean_solid", "location", "cell", "n_solid", "run_id"}
        assert "stress_limit" not in s
        assert s["max"] > s["mean_solid"] > 0 and s["n_solid"] > 0
        x, y, z = s["location"]
        assert 0 <= x <= 60 and -2 <= y <= 22 and -2 <= z <= 22  # inside the beam's voxel box
        # the cell holding the max, read back from density.npz
        files = await agent("export_files", run_id=plain["run_id"], directory=str(tmp_path / "f"))
        assert files["errors"] == {} and files["warnings"] == []
        with np.load(tmp_path / "f" / "density.npz") as z_:
            stress, rho = z_["stress"], z_["rho"]
            origin, h = z_["origin"], float(z_["h"])
        cell = np.unravel_index(int(np.argmax(stress)), stress.shape)
        assert list(cell) == s["cell"] and s["max"] == pytest.approx(stress.max(), rel=1e-5)
        assert np.allclose(s["location"], origin + h * (np.asarray(cell) + 0.5), rtol=1e-4)
        assert s["mean_solid"] == pytest.approx(stress[rho >= 0.5].mean(), rel=1e-4)
        assert b'Name="stress"' in (tmp_path / "f" / "result.vti").read_bytes()

        out = tmp_path / "trimmed.stl"
        saved = await agent("export_stl", run_id=plain["run_id"], path=str(out), trim=True)
        assert "warnings" not in saved and saved["triangles"] > 0
        mesh = trimesh.load(out, force="mesh")
        assert (mesh.bounds[0] >= -1e-6).all() and (
            mesh.bounds[1] <= [60.000001, 20.000001, 20.000001]
        ).all()
        raw = tmp_path / "raw.stl"
        await agent("export_stl", run_id=plain["run_id"], path=str(raw))
        poke = trimesh.load(raw, force="mesh").bounds
        assert (poke[0] < -1e-3).any() or (poke[1] > [60.001, 20.001, 20.001]).any()  # voxel skin
        look = await agent.raw("result_preview", run_id=plain["run_id"], trim=True)
        assert not look.is_error and look.content[0].type == "image"

        # with a stress limit (oc is forced to mma) the run reports the constraint
        await agent("set_params", project_id=pid, stress_limit=3.0)
        limited = await agent("run", project_id=pid, max_iter=3)
        assert isinstance(limited["constraint"], float) and isinstance(limited["stress_max"], float)
        assert any("forced to mma" in w for w in limited["warnings"])
        sl = await agent("result_stress_summary", run_id=limited["run_id"])
        assert sl["stress_limit"] == 3.0 and sl["max_over_limit"] == pytest.approx(sl["max"] / 3.0)

    drive(scenario)


def test_stress_summary_errors_are_readable():
    async def scenario(agent: Agent):
        assert "not found" in await agent.fails("result_stress_summary", run_id="missing")

    drive(scenario)


def test_facet_faces_and_facet_kinds(examples_dir):
    async def scenario(agent: Agent):
        mesh = await agent("load_mesh", path=str(examples_dir / "cantilever.stl"))
        desc = await agent("describe_mesh", mesh_id=mesh["id"])
        assert [f["kind"] for f in desc["facets"]] == ["plane"] * 6
        assert all(
            "radius" not in f and "axis" not in f and "brep_face" not in f for f in desc["facets"]
        )
        faces = await agent("facet_faces", mesh_id=mesh["id"], facet_id=0)
        assert faces["n_faces"] == desc["facets"][0]["n_faces"] == len(faces["face_ids"]) == 2
        assert max(faces["face_ids"]) < mesh["n_faces"]
        assert "unknown facet" in await agent.fails("facet_faces", mesh_id=mesh["id"], facet_id=99)
        assert "not found" in await agent.fails(
            "facet_faces", mesh_id="deadbeefdeadbeef", facet_id=0
        )

        bracket = await agent("load_mesh", path=str(examples_dir / "bracket.stl"))
        table = await agent("describe_mesh", mesh_id=bracket["id"], top=0)
        holes = [f for f in table["facets"] if f["kind"] == "cylinder"]
        big = max(holes, key=lambda f: f["radius"])
        assert big["radius"] == pytest.approx(6.0, abs=1e-3) and len(big["axis"]) == 3
        got = await agent("facet_faces", mesh_id=bracket["id"], facet_id=big["id"])
        assert got["n_faces"] == big["n_faces"] == len(got["face_ids"]) > 50

    drive(scenario)


def test_step_facets_through_mcp(examples_dir):
    pytest.importorskip("OCP", reason="STEP support not installed (uv sync --extra step)")

    async def scenario(agent: Agent):
        mesh = await agent("load_mesh", path=str(examples_dir / "bracket.step"))
        assert mesh["source"] == "step" and mesh["n_brep_faces"] == 14
        table = await agent("describe_mesh", mesh_id=mesh["id"], top=0)
        assert all(isinstance(f["brep_face"], int) for f in table["facets"])
        hole = next(f for f in table["facets"] if f.get("radius") == pytest.approx(6.0))
        got = await agent("facet_faces", mesh_id=mesh["id"], facet_id=hole["id"], angle_deg=40)
        assert got["n_faces"] == hole["n_faces"]  # angle_deg is ignored for B-rep faces
        pid = (await agent("create_project", name="step", design_mesh_id=mesh["id"],
                           elements_along_longest=40))["project_id"]  # fmt: skip
        sel = {"kind": "facets", "facet_ids": [hole["id"]]}
        resolved = await agent("resolve_selection", project_id=pid, selection=sel)
        assert resolved["count"] > 0

    drive(scenario)


def test_topop_mcp_serves_over_stdio(examples_dir):
    """`topop mcp` as a subprocess: the handshake works and stdout carries nothing but protocol."""

    async def main():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "topop.cli", "mcp"], env={**os.environ}
        )
        async with Client(params) as client:
            assert EXPECTED_TOOLS <= {t.name for t in (await client.list_tools()).tools}
            res = await client.call_tool("load_mesh", {"path": str(examples_dir / "bracket.stl")})
            assert json.loads(res.content[0].text)["n_faces"] == 1326

    anyio.run(main)
