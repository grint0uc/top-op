"""`topop describe` / `topop run` and the headless `Session` they share (no HTTP)."""

from __future__ import annotations

import json
import math
import re
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
import trimesh

from topop.agent import Session, expand_selection, parse_selection, pretty_json
from topop.cli import main
from topop.core.voxelize import transform_matrix
from topop.server.schemas import ProjectIn, RunExport

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "data"
    monkeypatch.setenv("TOPOP_DATA_DIR", str(d))
    return d


def cli(*argv: object) -> int:
    with pytest.raises(SystemExit) as exc:
        main([str(a) for a in argv])
    return exc.value.code


def case_json(examples_dir: Path, **edit) -> dict:
    case = json.loads((examples_dir / "cantilever.json").read_text())
    case["design_mesh"]["path"] = str(examples_dir / "cantilever.stl")
    case.update(edit)
    return case


# ---- topop describe ---------------------------------------------------------------------------


def test_describe_bracket_lists_plate_bottom_first(examples_dir, capsys, tmp_path):
    png = tmp_path / "bracket.png"
    assert cli("describe", examples_dir / "bracket.stl", "--top", 8, "--png", png) == 0
    out = capsys.readouterr().out
    assert "bracket.stl" in out and "watertight True" in out
    assert "bbox min (0, 0, 0)  max (80, 60, 60)" in out
    header = next(
        i for i, ln in enumerate(out.splitlines()) if ln.split()[:3] == ["id", "faces", "area"]
    )
    rows = out.splitlines()[header + 1 :]
    first = re.match(r"\s*(\d+)\s+(\d+)\s+([\d.]+)\s+\(([^)]*)\)", rows[0])
    assert first and int(first.group(1)) == 0
    assert float(first.group(3)) == pytest.approx(80 * 60 - 4 * math.pi * 9, rel=1e-3)
    assert [float(v) for v in first.group(4).split(",")] == [0.0, 0.0, -1.0]  # faces down
    assert len([r for r in rows if re.match(r"\s*\d+\s", r)]) == 8  # --top
    assert png.read_bytes().startswith(PNG)


def test_describe_missing_file_exits_2(capsys, tmp_path):
    assert cli("describe", tmp_path / "nope.stl") == 2
    assert "not found" in capsys.readouterr().err


# ---- topop run --------------------------------------------------------------------------------


def test_run_cantilever_writes_everything(examples_dir, capsys, tmp_path):
    out = tmp_path / "out"
    args = ["run", examples_dir / "cantilever.json", "--out", out, "--max-iter", 6]
    assert cli(*args, "--resolution", 16) == 0
    text = capsys.readouterr().out
    assert "grid 18x8x8" in text and "support fixed end x=0" in text and "-> 49 nodes" in text
    its = [ln for ln in text.splitlines() if re.match(r"\s*\d+\s+[\d.e+-]+\s+0\.\d+\s", ln)]
    assert len(its) == 6  # one progress line per iteration
    assert re.search(r"^status\s+done \(stopped at max_iter=6\)", text, re.MULTILINE)

    for name in ("result.stl", "result.png", "result.vti", "density.npz", "run.json"):
        assert (out / name).stat().st_size > 0, name
    assert len(trimesh.load(out / "result.stl", force="mesh").faces) >= 1
    assert (out / "result.png").read_bytes().startswith(PNG)
    assert b"<VTKFile" in (out / "result.vti").read_bytes()
    with np.load(out / "density.npz") as npz:
        assert 0 <= npz["rho"].min() and npz["rho"].max() <= 1
    export = RunExport.model_validate_json((out / "run.json").read_text())
    assert export.run.status == "done" and len(export.run.history) == 6
    assert [r.it for r in export.run.history] == [1, 2, 3, 4, 5, 6]
    assert export.project.grid.elements_along_longest == 16
    assert export.project.params.max_iter == 6
    compliance = [r.compliance for r in export.run.history]
    assert compliance[-1] < compliance[0]


def test_run_json_is_rerunnable(examples_dir, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    base = ["run", examples_dir / "cantilever.json", "--quiet"]
    assert cli(*base, "--out", first, "--max-iter", 2, "--resolution", 12) == 0
    assert cli("run", first / "run.json", "--out", second, "--max-iter", 3, "--quiet") == 0
    again = RunExport.model_validate_json((second / "run.json").read_text())
    assert len(again.run.history) == 3 and again.project.grid.elements_along_longest == 12


def test_run_with_threshold_above_every_density_still_writes_the_rest(
    examples_dir, capsys, tmp_path
):
    out = tmp_path / "out"
    args = ["run", examples_dir / "cantilever.json", "--out", out, "--max-iter", 2, "--quiet"]
    assert cli(*args, "--resolution", 12, "--threshold", 0.99) == 0
    assert "result.stl not written: no material above threshold 0.99" in capsys.readouterr().err
    assert not (out / "result.stl").exists() and (out / "run.json").exists()
    assert (out / "density.npz").exists()


def test_run_without_supports_exits_2(examples_dir, capsys, tmp_path):
    case = tmp_path / "nosupport.json"
    case.write_text(json.dumps(case_json(examples_dir, supports=[])))
    assert cli("run", case, "--out", tmp_path / "out", "--resolution", 12) == 2
    err = capsys.readouterr().err
    assert "not runnable" in err and "no supports defined" in err
    assert not (tmp_path / "out").exists()


def test_run_selection_that_misses_exits_2(examples_dir, capsys, tmp_path):
    case = case_json(examples_dir)
    case["loads"][0]["selection"]["within"] = [[500, 0, 0], [510, 10, 10]]  # far from the beam
    path = tmp_path / "miss.json"
    path.write_text(json.dumps(case))
    assert cli("run", path, "--out", tmp_path / "out", "--resolution", 12) == 2
    captured = capsys.readouterr()
    assert "resolves to 0 nodes" in captured.out  # the warning while setting up
    assert "load 0 resolves to zero nodes" in captured.err


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda c, d: c["design_mesh"].update(path="missing.stl"), "mesh file not found"),
        (lambda c, d: c["grid"].update(elements_along_longest=1), "elements_along_longest"),
        (lambda c, d: c["loads"][0]["selection"].update(kind="blob"), "kind"),
    ],
)
def test_run_bad_case_exits_2(examples_dir, capsys, tmp_path, mutate, needle):
    case = case_json(examples_dir)
    mutate(case, tmp_path)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(case))
    assert cli("run", path, "--out", tmp_path / "out") == 2
    assert needle in capsys.readouterr().err
    assert cli("run", tmp_path / "does-not-exist.json") == 2


def test_run_out_of_memory_exits_3(examples_dir, capsys, tmp_path, monkeypatch):
    def too_big(*args, **kwargs):
        raise MemoryError("1000 active elements need about 99 GB; lower the resolution")

    monkeypatch.setattr("topop.agent.optimize", too_big)
    assert cli("run", examples_dir / "cantilever.json", "--out", tmp_path / "o", "--quiet") == 3
    assert "lower the resolution" in capsys.readouterr().err


# ---- the shared Session API ---------------------------------------------------------------------


def test_bundled_cases_resolve_their_boundaries(examples_dir):
    s = Session()
    for name, loads, supports in (("cantilever", 77, 121), ("bracket", 279, 1265)):
        project = s.load_case(examples_dir / f"{name}.json")
        stats = s.voxel_stats(project.id)
        assert stats.warnings == [] and stats.n_active > 1000
        b = s.boundaries(project.id)
        assert b["loads"][0]["n_nodes"] == loads and b["supports"][0]["n_nodes"] == supports
    assert s.get_project(project.id).ref_models[0].mode == "keep_out"  # slot removes no-warning


def test_case_round_trip_keeps_paths_and_selections(examples_dir, tmp_path):
    s = Session()
    project = s.load_case(examples_dir / "bracket.json")
    dst = tmp_path / "cases" / "copy.json"
    s.save_case(project.id, dst)
    saved = json.loads(dst.read_text())
    assert saved["design_mesh"]["path"].endswith("bracket.stl")
    assert saved["design_mesh"]["mesh_id"] == project.design_mesh.mesh_id
    assert saved["loads"][0]["selection"] == project.loads[0].selection.model_dump(
        exclude_none=True
    )
    assert "id" not in saved and "created_at" not in saved
    again = s.load_case(dst)
    assert again.id != project.id
    assert again.design_mesh.mesh_id == project.design_mesh.mesh_id
    assert again.ref_models[0].transform == project.ref_models[0].transform


def test_saved_case_next_to_its_mesh_uses_a_relative_path(examples_dir, tmp_path):
    for name in ("cantilever.stl", "cantilever.json"):
        shutil.copy(examples_dir / name, tmp_path / name)
    s = Session()
    project = s.load_case(tmp_path / "cantilever.json")
    s.save_case(project.id, tmp_path / "saved.json")
    assert (
        json.loads((tmp_path / "saved.json").read_text())["design_mesh"]["path"] == "cantilever.stl"
    )
    other = Session(tmp_path / "other-store")  # a fresh store: the case alone rebuilds everything
    assert other.voxel_stats(other.load_case(tmp_path / "saved.json").id).n_active == 3000


def test_session_edit_api_validates(examples_dir):
    s = Session()
    mesh = s.load_mesh(examples_dir / "cantilever.stl")
    project = s.create_project(ProjectIn.model_validate({"design_mesh": {"mesh_id": mesh.id}}))
    assert s.set_params(project.id, volfrac=0.5, max_iter=None).params.volfrac == 0.5
    with pytest.raises(ValueError, match="unknown params"):
        s.set_params(project.id, volfrack=0.5)
    with pytest.raises(ValueError, match="volfrac"):  # pydantic bound: volfrac < 1
        s.set_params(project.id, volfrac=1.5)
    assert s.get_project(project.id).params.volfrac == 0.5  # a rejected edit changes nothing
    with pytest.raises(LookupError, match="not found"):
        s.get_project("nope")
    with pytest.raises(ValueError, match="no load"):
        s.remove_load(project.id, "x")


def test_cancelled_run_keeps_its_result_and_other_stores_can_export_it(examples_dir):
    s = Session()
    project = s.load_case(examples_dir / "cantilever.json")
    s.set_grid(project.id, elements_along_longest=12)
    stop = threading.Event()
    info = s.run(project.id, lambda r: stop.set() if r.it == 2 else None, stop)
    assert info.status == "cancelled" and [r.it for r in info.history] == [1, 2]
    assert s.run_outcome(info.id)["outcome"] == "cancelled"
    assert s.result_stl(info.id, threshold=0.3)[84:]  # the partial result is exportable
    # a second process on the same TOPOP_DATA_DIR (e.g. `topop serve`) finds the persisted run
    other = Session()
    assert other.get_run(info.id).status == "cancelled"
    assert other.result_png(info.id, threshold=0.3).startswith(PNG)
    assert other.export(info.id).project.name == "cantilever"
    with pytest.raises(ValueError, match="threshold"):
        other.result_stl(info.id, threshold=0.0)
    with pytest.raises(ValueError, match="no material above"):
        other.result_stl(info.id, threshold=1.0)


def point_set(session: Session, project_id: str, sel: dict) -> np.ndarray:
    r = session.resolve(project_id, sel, samples=10_000)
    return np.asarray(r.get("sample_xyz", [])).reshape(-1, 3)


def test_primitive_shorthand_matches_canonical_transform():
    cyl = expand_selection(
        {"kind": "cylinder", "center": [30, 10, 10], "radius": 4, "height": 20, "axis": "x"}
    )
    m = transform_matrix(cyl["transform"])
    assert np.allclose(m[:3, 3], [30, 10, 10]) and cyl["size"] == [4.0, 20.0, 4.0]
    assert abs((m[:3, :3] @ [0, 1, 0])[0]) == pytest.approx(1.0)  # local Y -> world X
    box = expand_selection({"kind": "box", "min": [0, 0, 0], "max": [10, 20, 30]})
    assert box["size"] == [10, 20, 30]
    assert np.allclose(transform_matrix(box["transform"])[:3, 3], [5, 10, 15])
    assert parse_selection({"kind": "normal", "direction": [0, 0, 1]}).mesh_id == "design"
    with pytest.raises(ValueError, match="not both"):
        expand_selection({"kind": "box", "min": [0] * 3, "max": [1] * 3, "transform": [0] * 16})
    with pytest.raises(ValueError, match="axis"):
        expand_selection({"kind": "cylinder", "center": [0] * 3, "radius": 1, "height": 1})


def test_primitive_shorthand_selects_the_expected_nodes(examples_dir):
    s = Session()
    project = s.load_case(examples_dir / "cantilever.json")
    s.set_grid(project.id, elements_along_longest=30)  # h = 2
    box = {"kind": "box", "min": [58, -1, -1], "max": [61, 21, 21]}  # the free end face
    pts = point_set(s, project.id, box)
    assert np.all((pts[:, 0] >= 58) & (pts[:, 0] <= 60))  # surface nodes inside the box
    assert (pts[:, 0] == 60).sum() == 11 * 11  # the whole free-end face
    ball = {"kind": "sphere", "center": [60, 10, 10], "radius": 3}
    ball_pts = point_set(s, project.id, ball)
    assert 0 < len(ball_pts) < len(pts)
    assert np.all(np.linalg.norm(ball_pts - [60, 10, 10], axis=1) <= 3 + 1e-9)
    # a cylinder along x around the beam axis picks the end-face nodes within its radius
    cyl = {"kind": "cylinder", "center": [60, 10, 10], "radius": 5, "height": 4, "axis": "x"}
    cyl_pts = point_set(s, project.id, cyl)
    assert len(cyl_pts) > 0 and np.all(np.hypot(cyl_pts[:, 1] - 10, cyl_pts[:, 2] - 10) <= 5 + 1e-9)
    empty = s.resolve(project.id, {"kind": "sphere", "center": [500, 0, 0], "radius": 1})
    assert empty["count"] == 0


def test_pretty_json_collapses_flat_arrays_only():
    obj = {
        "a": [1, 2, 3],
        "b": {"x": 1, "y": [[0, 1], [2, 3]]},
        "c": [{"k": 1}, {"k": 2}],
        "s": "[ ]",
    }
    text = pretty_json(obj)
    assert json.loads(text) == obj
    assert '"a": [1, 2, 3]' in text and '"y": [[0, 1], [2, 3]]' in text
    assert '{"k": 1}' in text and "\n" in text
