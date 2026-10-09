"""Headless top-op session: the one API behind `topop run`, `topop describe` and `topop mcp`.

No HTTP. A `Session` wraps the same `Store` as `topop serve` (same `TOPOP_DATA_DIR` default), so
meshes, projects and finished runs are shared between the GUI and the agent interface.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from pydantic import TypeAdapter, ValidationError

from topop.core.export import density_to_mesh, render_png, to_npz_bytes, to_stl_bytes, to_vti_bytes
from topop.core.optimize import optimize
from topop.core.problem import IterationInfo
from topop.core.selection import compute_facets, node_xyz, resolved_preview
from topop.core.step import META_FACE_TO_FACET
from topop.core.struts import StrutResult
from topop.core.voxelize import transform_matrix
from topop.server.build import (
    DESIGN,
    ProblemInvalid,
    build_problem,
    load_project_meshes,
    params_warnings,
    resolve_project_selections,
    resolve_sel,
)
from topop.server.jobs import iteration_record
from topop.server.routes_runs import DESIGN_RGB, PREVIEW_SMOOTH, RESULT_RGB
from topop.server.routes_struts import (
    StrutRequest,
    strut_json,
    struts_for_project,
    struts_for_run,
)
from topop.server.schemas import (
    FacetInfo,
    GridSpec,
    IterationRecord,
    LoadSpec,
    MaterialSpec,
    MeshInfo,
    ParamsSpec,
    Project,
    ProjectIn,
    RefModel,
    RunExport,
    RunInfo,
    Selection,
    SupportSpec,
    VoxelStats,
)
from topop.server.store import NotFoundError, RunRecord, Store, now_iso

RESULT_STATUS = {
    "converged": "done",
    "max_iter": "done",
    "cancelled": "cancelled",
    "error": "error",
}
ProgressFn = Callable[[IterationRecord], None]
# what write_outputs writes (`topop run --out`, MCP export_files)
OUTPUT_FILES = ("result.stl", "result.png", "result.vti", "density.npz", "run.json")
_SELECTION = TypeAdapter(Selection)
_ENVELOPE = {"id", "created_at", "updated_at"}
_ARRAY_LEAF = re.compile(r"\[\n\s*([^\[\]{}]*?)\n\s*\]")  # array of scalars
_ARRAY_NEST = re.compile(r"\[\n\s*([^{}]*?)\n\s*\]")  # array of (already flat) arrays
_FLAT_OBJECT = re.compile(r"\{\n\s*([^{}]*?)\n\s*\}")


class ProjectInvalid(ValueError):
    """The project is not runnable; `issues` lists why (one sentence each)."""

    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        super().__init__("project is not runnable: " + "; ".join(self.issues))


def pretty_json(obj: Any, width: int = 110) -> str:
    """`json.dumps(indent=1)` with short flat arrays/objects kept on one line (readable, compact)."""

    def squeeze(m: re.Match) -> str:
        flat = m.group(0)[0] + re.sub(r",\n\s*", ", ", m.group(1)) + m.group(0)[-1]
        return flat if len(flat) <= width else m.group(0)

    text = json.dumps(obj, indent=1)
    for pattern in (_ARRAY_LEAF, _ARRAY_NEST, _FLAT_OBJECT):
        text = pattern.sub(squeeze, text)
    return text


def explain_validation_error(exc: ValidationError) -> str:
    """One line per pydantic error: `location: message`."""
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors(include_url=False)
    )


# ---- selection shorthand ----------------------------------------------------------------------

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _colmajor(m: np.ndarray) -> list[float]:
    return np.asarray(m, dtype=np.float64).T.ravel().tolist()


def _translation(center: Any) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = np.asarray(center, dtype=np.float64).reshape(3)
    return m


def expand_selection(sel: Mapping[str, Any]) -> dict[str, Any]:
    """Selection dict with the agent shorthands expanded to the canonical `schemas.Selection`.

    faces/facets/normal: `mesh_id` defaults to "design". Primitives accept box {min, max} or
    {center, size}, sphere {center, radius}, cylinder {center, radius, height, axis: x|y|z|vector}
    instead of `transform`/`size`; mixing the shorthand with `transform` is an error.
    """
    out = dict(sel)
    kind = out.get("kind")
    if kind in ("faces", "facets", "normal"):
        out.setdefault("mesh_id", "design")
        return out
    if kind not in ("box", "sphere", "cylinder"):
        return out
    keys = ("min", "max", "center", "radius", "height", "axis")
    short = {k: out.pop(k) for k in keys if k in out}
    if not short:
        return out
    if "transform" in out:
        raise ValueError(
            f"{kind}: use either center/min/max/radius/height/axis or transform, not both"
        )
    if kind == "box":
        if "min" in short or "max" in short:
            if "min" not in short or "max" not in short or "center" in short or "size" in out:
                raise ValueError("box: give either {min, max} or {center, size}")
            lo, hi = (np.asarray(short[k], dtype=np.float64) for k in ("min", "max"))
            out["size"] = (hi - lo).tolist()
            short["center"] = ((lo + hi) / 2).tolist()
        if "center" not in short:
            raise ValueError("box: give {min, max} or {center, size}")
        out["transform"] = _colmajor(_translation(short["center"]))
    elif kind == "sphere":
        if "center" not in short or "radius" not in short:
            raise ValueError("sphere: give {center, radius}")
        r = float(short["radius"])
        out["size"] = [r, r, r]
        out["transform"] = _colmajor(_translation(short["center"]))
    else:
        if not {"center", "radius", "height", "axis"} <= short.keys():
            raise ValueError(
                "cylinder: give {center, radius, height, axis} (axis: x|y|z or vector)"
            )
        axis = short["axis"]
        vec = _AXES.get(axis.lower()) if isinstance(axis, str) else axis
        if vec is None or np.linalg.norm(np.asarray(vec, dtype=np.float64)) == 0:
            raise ValueError(
                f"cylinder axis must be 'x', 'y', 'z' or a non-zero vector, got {axis!r}"
            )
        v = np.asarray(vec, dtype=np.float64).reshape(3)
        rot = trimesh.geometry.align_vectors([0.0, 1.0, 0.0], v / np.linalg.norm(v))
        out["size"] = [float(short["radius"]), float(short["height"]), float(short["radius"])]
        out["transform"] = _colmajor(_translation(short["center"]) @ rot)
    return out


def parse_selection(sel: Mapping[str, Any] | Any) -> Any:
    """dict (shorthand allowed) or an existing selection model -> validated selection model."""
    if isinstance(sel, Mapping):
        return _SELECTION.validate_python(expand_selection(sel))
    return _SELECTION.validate_python(sel)


def _rewrite_selection_meshes(body: ProjectIn, remap: Mapping[str, str]) -> None:
    for item in [*body.loads, *body.supports]:
        mid = getattr(item.selection, "mesh_id", None)
        if mid in remap:
            item.selection.mesh_id = remap[mid]


def _unit(v: np.ndarray) -> list[float]:
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 0 else [0.0, 0.0, 0.0]


def _place_facets(
    facets: Sequence[dict], labels: np.ndarray, world: trimesh.Trimesh, m: np.ndarray
) -> list[dict]:
    """Facet rows of the raw mesh with their geometry in world space (`m`: the 4x4 placement).

    Ids, n_faces and kind stay (facet ids are always those of the raw mesh); area and bbox come
    from the facet's world triangles (exact); centroid, normal and axis are mapped; a radius is
    scaled by the transform across the axis (exact for rotations and uniform scale).
    """
    lin = m[:3, :3]
    normal_map = np.linalg.inv(lin).T
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    out = []
    for f in facets:
        lo, hi = np.searchsorted(sorted_labels, [f["id"], f["id"] + 1])
        tris = order[lo:hi]
        row = dict(f)
        if tris.size:
            xyz = world.vertices[world.faces[tris]].reshape(-1, 3)
            row["area"] = float(world.area_faces[tris].sum())
            row["bbox"] = [xyz.min(0).tolist(), xyz.max(0).tolist()]
        row["centroid"] = (lin @ np.asarray(f["centroid"], dtype=float) + m[:3, 3]).tolist()
        if any(f["normal"]):
            row["normal"] = _unit(normal_map @ np.asarray(f["normal"], dtype=float))
        if f.get("axis") is not None:
            axis = np.asarray(f["axis"], dtype=float)
            row["axis"] = _unit(lin @ axis)
            if f.get("radius") is not None:  # mean stretch of two directions across the axis
                u = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
                u /= np.linalg.norm(u)
                v = np.cross(axis, u)
                stretch = (np.linalg.norm(lin @ u) + np.linalg.norm(lin @ v)) / 2
                row["radius"] = float(f["radius"] * stretch)
        elif f.get("radius") is not None:  # sphere
            row["radius"] = float(f["radius"] * abs(np.linalg.det(lin)) ** (1 / 3))
        out.append(row)
    return out


# ---- session ----------------------------------------------------------------------------------


@dataclass
class _Outcome:
    status: str  # core status: converged | max_iter | cancelled | error
    message: str
    wall_seconds: float


class Session:
    def __init__(self, data_dir: str | os.PathLike | None = None):
        self.store = Store(data_dir)
        self._lock = threading.RLock()  # project read-modify-write
        self._run_lock = threading.Lock()  # one optimization at a time per session
        self._paths: dict[str, str] = {}  # mesh_id -> file it was loaded from
        self._outcomes: dict[str, _Outcome] = {}

    # ---- meshes -------------------------------------------------------------------------------

    def load_mesh(self, path: str | os.PathLike) -> MeshInfo:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"mesh file not found: {p}")
        info = self.store.add_mesh(p.read_bytes(), p.name)
        self._paths[info.id] = str(p.resolve())
        return info

    def mesh_info(self, mesh_id: str) -> MeshInfo:
        return self.store.mesh_info(mesh_id)  # NotFoundError

    def describe_mesh(
        self, mesh_id: str, angle_deg: float = 5.0, top: int = 30, project_id: str | None = None
    ) -> dict:
        """MeshInfo + the `top` largest facets (all if top <= 0): what names faces.

        STEP meshes list their B-rep faces instead (exact; `angle_deg` is ignored). Coordinates
        are the mesh file's own (`frame` "mesh"). With `project_id` they are WORLD coordinates
        (`frame` "world"), the frame every selection is resolved in: the mesh as that project
        places it (`mesh_id` = its design mesh or "design", a reference model's mesh or
        "ref:<id>"), so normals, centroids, bboxes, axes and radii are what `direction`, `within`
        and the primitives must use. Facet ids are the same in both frames.
        """
        angle = float(angle_deg)
        if project_id is None:
            info = self.mesh_info(mesh_id)
            facets, total = self.store.mesh_facets(mesh_id, angle)
            frame: dict[str, Any] = {"frame": "mesh"}
            shown = facets[:top] if top > 0 else facets
        else:
            project = self.store.get_project(project_id)
            key, ref = self._placement(project, mesh_id)
            raw_id = ref.mesh_id
            info = self.mesh_info(raw_id)
            facets, total = self.store.mesh_facets(raw_id, angle)
            shown = facets[:top] if top > 0 else facets
            m = transform_matrix(ref.transform)
            frame = {"frame": "world", "project_id": project.id, "placed_as": key}
            if not np.allclose(m, np.eye(4)):
                world = load_project_meshes(project, self.store.get_mesh)[key]
                raw = self.store.get_mesh(raw_id)
                labels = raw.metadata.get(META_FACE_TO_FACET)
                if labels is None:
                    labels = compute_facets(raw, angle)[1]
                shown = _place_facets(shown, np.asarray(labels), world, m)
                volume = None if info.volume is None else abs(float(world.volume))
                info = info.model_copy(update={"bbox": world.bounds.tolist(), "volume": volume})
        return {
            "mesh": info.model_dump(),
            **frame,
            "angle_deg": angle,
            "n_facets_total": total,
            "facets": [FacetInfo(**f).model_dump() for f in shown],
        }

    @staticmethod
    def _placement(project: Project, mesh_id: str) -> tuple[str, Any]:
        """(world mesh key, MeshRef/RefModel) for a mesh of the project: its design mesh id or
        "design", "ref:<id>", or the mesh id of a reference model."""
        design, refs = project.design_mesh, project.ref_models
        if design is not None and design.mesh_id and mesh_id in (DESIGN, design.mesh_id):
            return DESIGN, design
        for ref in refs:
            if ref.mesh_id and mesh_id in (f"ref:{ref.id}", ref.mesh_id):
                return f"ref:{ref.id}", ref
        names = [DESIGN] if design is not None else []
        names += [f"ref:{r.id}" for r in refs]
        raise ValueError(
            f"mesh {mesh_id!r} is not part of project {project.id} "
            f"(use its mesh id or one of {names or 'nothing: the project has no mesh'})"
        )

    def facet_faces(self, mesh_id: str, facet_id: int, angle_deg: float = 5.0) -> list[int]:
        """Triangle ids of one facet of `describe_mesh` (ValueError if the facet does not exist)."""
        return self.store.facet_faces(mesh_id, [int(facet_id)], float(angle_deg)).tolist()

    def preview_mesh(self, mesh_id: str, view: str = "iso") -> bytes:
        return render_png([(self.store.get_mesh(mesh_id), DESIGN_RGB, 1.0)], view)

    # ---- projects -----------------------------------------------------------------------------

    def _attach(self, ref: Any, base: Path, remap: dict[str, str]) -> None:
        """Upload `ref.path` (relative to `base`) and set `ref.mesh_id`; the path is kept absolute."""
        if ref.path:
            p = Path(ref.path).expanduser()
            written = ref.path
            p = p if p.is_absolute() else base / p
            if p.is_file():
                info = self.load_mesh(p)
                remap[written] = remap[str(p)] = info.id
                ref.mesh_id, ref.path = info.id, str(p.resolve())
                return
            if not ref.mesh_id:
                raise FileNotFoundError(
                    f"mesh file not found: {p} (paths in a case file are relative to the case file)"
                )
        if ref.mesh_id:
            self.store.get_mesh(ref.mesh_id)  # NotFoundError if it was never uploaded

    def _materialize(self, body: ProjectIn, base: Path | None = None) -> ProjectIn:
        """Upload every mesh given by `path` and rewrite selections that named a mesh by path."""
        body = body.model_copy(deep=True)
        base = base or Path.cwd()
        remap: dict[str, str] = {}
        if body.design_mesh is not None:
            self._attach(body.design_mesh, base, remap)
        for ref in body.ref_models:
            self._attach(ref, base, remap)
        _rewrite_selection_meshes(body, remap)
        return body

    def create_project(self, body: ProjectIn) -> Project:
        return self.store.create_project(self._materialize(body))

    def get_project(self, project_id: str) -> Project:
        return self.store.get_project(project_id)

    def list_projects(self) -> list[Project]:
        return self.store.list_projects()

    def update_project(self, project_id: str, body: ProjectIn) -> Project:
        return self.store.update_project(project_id, self._materialize(body))

    def _edit(self, project_id: str, fn: Callable[[ProjectIn], Any]) -> tuple[Project, Any]:
        with self._lock:
            body = ProjectIn.model_validate(
                self.store.get_project(project_id).model_dump(exclude=_ENVELOPE)
            )
            result = fn(body)
            body = ProjectIn.model_validate(body.model_dump())  # re-validate what fn changed
            return self.store.update_project(project_id, body), result

    @staticmethod
    def _fresh_id(prefix: str, taken: set[str]) -> str:
        n = 1
        while f"{prefix}{n}" in taken:
            n += 1
        return f"{prefix}{n}"

    @staticmethod
    def _boundary_ids(body: ProjectIn) -> set[str]:
        return {x.id for x in [*body.loads, *body.supports]}

    def add_load(self, project_id: str, load: LoadSpec) -> LoadSpec:
        """Append a load. An empty `id` is replaced by `loadN`."""

        def fn(body: ProjectIn) -> LoadSpec:
            taken = self._boundary_ids(body)
            spec = load.model_copy(update={"id": load.id or self._fresh_id("load", taken)})
            if spec.id in taken:
                raise ValueError(f"id {spec.id!r} is already used by a load or support")
            body.loads.append(spec)
            return spec

        return self._edit(project_id, fn)[1]

    def add_support(self, project_id: str, support: SupportSpec) -> SupportSpec:
        """Append a support. An empty `id` is replaced by `supportN`."""

        def fn(body: ProjectIn) -> SupportSpec:
            taken = self._boundary_ids(body)
            spec = support.model_copy(update={"id": support.id or self._fresh_id("support", taken)})
            if spec.id in taken:
                raise ValueError(f"id {spec.id!r} is already used by a load or support")
            body.supports.append(spec)
            return spec

        return self._edit(project_id, fn)[1]

    def add_ref_model(self, project_id: str, ref: RefModel) -> RefModel:
        """Append a reference body (keep_in / keep_out). An empty `id` is replaced by `refN`."""

        def fn(body: ProjectIn) -> RefModel:
            taken = {r.id for r in body.ref_models}
            spec = ref.model_copy(update={"id": ref.id or self._fresh_id("ref", taken)})
            if spec.id in taken:
                raise ValueError(f"reference model id {spec.id!r} already exists")
            self._attach(spec, Path.cwd(), {})
            body.ref_models.append(spec)
            return spec

        return self._edit(project_id, fn)[1]

    def _remove(self, project_id: str, attr: str, item_id: str, what: str) -> Project:
        def fn(body: ProjectIn) -> None:
            items = getattr(body, attr)
            keep = [x for x in items if x.id != item_id]
            if len(keep) == len(items):
                known = ", ".join(x.id for x in items) or "none"
                raise ValueError(f"no {what} with id {item_id!r} (existing: {known})")
            setattr(body, attr, keep)

        return self._edit(project_id, fn)[0]

    def remove_load(self, project_id: str, load_id: str) -> Project:
        return self._remove(project_id, "loads", load_id, "load")

    def remove_support(self, project_id: str, support_id: str) -> Project:
        return self._remove(project_id, "supports", support_id, "support")

    def remove_ref_model(self, project_id: str, ref_id: str) -> Project:
        return self._remove(project_id, "ref_models", ref_id, "reference model")

    @staticmethod
    def _merge(
        current: Any,
        spec_cls: type,
        fields: Mapping[str, Any],
        what: str,
        clear: Sequence[str] = (),
    ) -> Any:
        unknown = sorted((set(fields) | set(clear)) - set(spec_cls.model_fields))
        if unknown:
            raise ValueError(
                f"unknown {what} field(s) {unknown}; valid: {sorted(spec_cls.model_fields)}"
            )
        given = {k: v for k, v in fields.items() if v is not None}
        return spec_cls.model_validate({**current.model_dump(), **given, **dict.fromkeys(clear)})

    def set_params(self, project_id: str, clear: Sequence[str] = (), **fields: Any) -> Project:
        """Merge the given `ParamsSpec` fields (None = leave unchanged). `clear` names optional
        fields (`stress_limit`, `overhang`) to reset to None."""

        def fn(body: ProjectIn) -> None:
            body.params = self._merge(body.params, ParamsSpec, fields, "params", clear)

        return self._edit(project_id, fn)[0]

    def set_grid(self, project_id: str, **fields: Any) -> Project:
        def fn(body: ProjectIn) -> None:
            body.grid = self._merge(body.grid, GridSpec, fields, "grid")

        return self._edit(project_id, fn)[0]

    def set_material(self, project_id: str, **fields: Any) -> Project:
        def fn(body: ProjectIn) -> None:
            body.material = self._merge(body.material, MaterialSpec, fields, "material")

        return self._edit(project_id, fn)[0]

    # ---- voxelization and selections ------------------------------------------------------------

    def _domain(self, project_id: str):
        project = self.store.get_project(project_id)
        try:
            return project, self.store.get_domain(project)
        except ValueError as exc:  # no design mesh, degenerate geometry
            raise ProjectInvalid([str(exc)]) from exc

    def voxel_stats(self, project_id: str) -> VoxelStats:
        """Grid statistics; warnings include loads/supports that resolve to nothing."""
        project, domain = self._domain(project_id)
        _, _, warnings, errors = resolve_project_selections(project, domain)
        notes = params_warnings(project.params)
        return VoxelStats(
            **{**domain.stats, "warnings": [*domain.warnings, *warnings, *errors, *notes]}
        )

    def boundaries(self, project_id: str) -> dict:
        """Per load/support: resolved node count and bbox (what the optimizer will actually use)."""
        project, domain = self._domain(project_id)
        load_nodes, support_nodes, _, _ = resolve_project_selections(project, domain)

        def row(item: Any, nodes: np.ndarray) -> dict:
            out = {"id": item.id, "name": item.name, "n_nodes": int(nodes.size)}
            if nodes.size:
                xyz = node_xyz(domain.grid, nodes)
                out["bbox"] = [xyz.min(0).tolist(), xyz.max(0).tolist()]
            return out

        return {
            "loads": [
                {**row(ld, n), "force": ld.force, "case": ld.case}
                for ld, n in zip(project.loads, load_nodes, strict=True)
            ],
            "supports": [
                {**row(sp, n), "fix": sp.fix}
                for sp, n in zip(project.supports, support_nodes, strict=True)
            ],
        }

    def resolve(
        self, project_id: str, selection: Mapping[str, Any] | Any, samples: int = 8
    ) -> dict:
        """What a selection picks on the project's grid: count, bbox, centroid and a few nodes."""
        sel = parse_selection(selection)
        _, domain = self._domain(project_id)
        nodes = resolve_sel(sel.model_dump(), domain)
        out: dict[str, Any] = {"count": int(nodes.size), "h": float(domain.grid.h)}
        if nodes.size:
            xyz = node_xyz(domain.grid, nodes)
            out["bbox"] = [xyz.min(0).tolist(), xyz.max(0).tolist()]
            out["centroid"] = xyz.mean(0).tolist()
            out["sample_xyz"] = resolved_preview(nodes, domain.grid, cap=samples)["xyz"]
        return out

    # ---- running ------------------------------------------------------------------------------

    def run(
        self,
        project_id: str,
        progress: ProgressFn | None = None,
        cancel: threading.Event | None = None,
        max_iter: int | None = None,
    ) -> RunInfo:
        """Optimize synchronously; the result is stored (and persisted) like a server run.

        `max_iter` overrides params.max_iter for this run only. ProjectInvalid (a ValueError
        with `.issues`) if the project is not runnable; MemoryError above the memory cap.
        """
        project, domain = self._domain(project_id)
        if max_iter is not None:
            params = ParamsSpec.model_validate(
                {**project.params.model_dump(), "max_iter": max_iter}
            )
            project = project.model_copy(update={"params": params})
        try:
            built = build_problem(project, domain.meshes_world, domain)
        except ProblemInvalid as exc:
            raise ProjectInvalid(exc.issues) from exc
        rec = self.store.new_run(project, built, VoxelStats(**built.stats))
        with self._run_lock:
            if rec.cancel.is_set():  # cancelled while waiting for the previous run
                self._finish(rec, "cancelled", "cancelled while queued")
                return rec.snapshot()
            with rec.lock:
                rec.info.status = "running"

            def callback(info: IterationInfo, rho: np.ndarray) -> bool:
                record = iteration_record(info)
                with rec.lock:
                    rec.info.history.append(record)
                if progress is not None:
                    progress(record)
                return not self._cancelled(rec, cancel)

            t0 = time.perf_counter()
            try:
                result = optimize(
                    built.problem,
                    built.params,
                    callback,
                    cancel=lambda: self._cancelled(rec, cancel),
                )
            except MemoryError as exc:
                msg = str(exc) or "out of memory; lower the resolution"
                self._finish(rec, "error", msg, time.perf_counter() - t0, outcome="error")
                raise MemoryError(msg) from exc
            except Exception as exc:
                self._finish(rec, "error", str(exc) or type(exc).__name__, outcome="error")
                raise
            except BaseException as exc:  # a second Ctrl-C: never leave the run "running"
                self._finish(rec, "error", f"run aborted ({type(exc).__name__})", outcome="error")
                raise
            rho = result.rho if result.history else None
            self._finish(
                rec,
                RESULT_STATUS.get(result.status, "error"),
                result.message,
                time.perf_counter() - t0,
                rho=rho,
                outcome=result.status,
                stress=result.stress,
            )
        return rec.snapshot()

    @staticmethod
    def _cancelled(rec: RunRecord, extra: threading.Event | None) -> bool:
        return rec.cancel.is_set() or (extra is not None and extra.is_set())

    def _finish(
        self,
        rec: RunRecord,
        status: str,
        message: str | None,
        wall: float = 0.0,
        rho: np.ndarray | None = None,
        outcome: str = "cancelled",
        stress: np.ndarray | None = None,
    ) -> None:
        with rec.lock:
            rec.info.status = status
            rec.info.finished_at = now_iso()
            rec.message = message
            rec.info.message = message
            rec.info.outcome = outcome
            if status == "error":
                rec.info.error = message or "run failed"
            if rho is not None:
                rec.rho, rec.stress = rho, stress
        # persist, then release the arrays (exports read runs/{id}.npz back); a failure keeps
        # them in memory and is appended to the message
        self.store.finish_run(rec)
        self._outcomes[rec.info.id] = _Outcome(outcome, rec.message or "", wall)

    def cancel(self, run_id: str) -> RunInfo:
        """Stop a run after its current iteration (callable from another thread)."""
        rec = self.store.get_run(run_id)
        rec.cancel.set()
        return rec.snapshot()

    def get_run(self, run_id: str) -> RunInfo:
        return self.store.get_run(run_id).snapshot()

    def list_runs(self) -> list[RunInfo]:
        return [r.snapshot() for r in self.store.list_runs()]

    def run_outcome(self, run_id: str) -> dict:
        """How a run of this session ended: converged | max_iter | cancelled | error, message, wall s."""
        o = self._outcomes.get(run_id)
        return (
            {}
            if o is None
            else {"outcome": o.status, "message": o.message, "wall_s": o.wall_seconds}
        )

    # ---- results ------------------------------------------------------------------------------

    def _result(
        self, run_id: str
    ) -> tuple[RunRecord, tuple[np.ndarray, Any, np.ndarray, np.ndarray]]:
        rec = self.store.get_run(run_id)
        with rec.lock:  # the status and the in-memory result are set together
            status = rec.info.status
        if status not in ("done", "cancelled"):
            raise ValueError(f"run {run_id} is {status}; no result to export")
        res = self.store.run_result(rec)
        if res is None:
            raise ValueError(f"run {run_id} has no density result")
        return rec, res

    @staticmethod
    def _isosurface(rho: np.ndarray, grid: Any, threshold: float, smooth: int) -> trimesh.Trimesh:
        mesh = density_to_mesh(rho, grid, threshold, int(smooth))  # ValueError if threshold <= 0
        if not len(mesh.faces):
            raise ValueError(
                f"no material above threshold {threshold} (max density {float(np.max(rho)):.3f}); "
                "lower the threshold"
            )
        return mesh

    def export_stl(
        self, run_id: str, threshold: float = 0.5, smooth: int = 0, trim: bool = False
    ) -> tuple[bytes, list[str]]:
        """(binary STL, warnings). `trim` intersects the surface with the design mesh; if that is
        impossible (open mesh, failed boolean) the untrimmed surface is returned plus a warning."""
        rec, (rho, grid, _, _) = self._result(run_id)
        mesh = self._isosurface(rho, grid, threshold, smooth)
        mesh, warnings = self.store.trim_result(rec, mesh) if trim else (mesh, [])
        return to_stl_bytes(mesh), warnings

    def result_stl(
        self, run_id: str, threshold: float = 0.5, smooth: int = 0, trim: bool = False
    ) -> bytes:
        return self.export_stl(run_id, threshold, smooth, trim)[0]

    def render_result(
        self, run_id: str, threshold: float = 0.5, view: str = "iso", trim: bool = False
    ) -> tuple[bytes, list[str]]:
        """(PNG, warnings): the result in orange over the ghosted design mesh."""
        rec, (rho, grid, _, _) = self._result(run_id)
        design = self.store.design_world(rec)
        result = self._isosurface(rho, grid, threshold, PREVIEW_SMOOTH)
        result, warnings = self.store.trim_result(rec, result) if trim else (result, [])
        layers = [(design, DESIGN_RGB, 0.15), (result, RESULT_RGB, 1.0)]
        png = render_png([layer for layer in layers if layer[0] is not None], view)
        return png, warnings

    def result_png(
        self, run_id: str, threshold: float = 0.5, view: str = "iso", trim: bool = False
    ) -> bytes:
        return self.render_result(run_id, threshold, view, trim)[0]

    def result_vti(self, run_id: str) -> bytes:
        rec, (rho, grid, _, passive) = self._result(run_id)
        return to_vti_bytes(rho, passive, grid, self.store.run_stress(rec))

    def result_npz(self, run_id: str) -> bytes:
        rec, (rho, grid, active, passive) = self._result(run_id)
        return to_npz_bytes(rho, grid, active, passive, self.store.run_stress(rec))

    def stress_summary(self, run_id: str) -> dict:
        """Von Mises of the final design: max, mean over solid cells (rho >= 0.5) and where the max is.

        `location` is the center of the element holding the maximum, in world coordinates.
        """
        rec, (rho, grid, active, _) = self._result(run_id)
        stress = self.store.run_stress(rec)
        if stress is None:
            raise ValueError(f"run {run_id} has no stress field (it produced no result)")
        stress = np.where(active, np.nan_to_num(stress), 0.0)
        cell = np.unravel_index(int(np.argmax(stress)), stress.shape)
        solid = active & (rho >= 0.5)
        out: dict[str, Any] = {
            "run_id": run_id,
            "max": float(stress[cell]),
            "mean_solid": float(stress[solid].mean()) if solid.any() else 0.0,
            "n_solid": int(solid.sum()),
            "location": (np.asarray(grid.origin) + grid.h * (np.asarray(cell) + 0.5)).tolist(),
            "cell": [int(c) for c in cell],
        }
        limit = rec.project.params.stress_limit
        if limit is not None:
            out["stress_limit"] = float(limit)
            out["max_over_limit"] = out["max"] / limit
        return out

    # ---- struts -------------------------------------------------------------------------------

    def generate_struts(self, run_id: str, **params: Any) -> dict:
        """Explicit strut (truss) structure of a finished run (`core.struts`), verified by an FE
        solve; `params` are `StrutRequest` fields. Stored next to the run as
        runs/{id}.struts.stl / .json / .png. Returns the summary plus `files`."""
        result, paths = struts_for_run(self.store, run_id, StrutRequest(**params))
        data = strut_json(result)
        data.pop("nodes")
        data.pop("bars")
        return {"run_id": run_id, **data, "files": {k: str(v) for k, v in paths.items()}}

    def struts_from_density(
        self, project_id: str, rho: np.ndarray, grid: Any, **params: Any
    ) -> tuple[StrutResult, trimesh.Trimesh | None]:
        """(struts, world design mesh) of a density field on the project's grid (`topop struts`
        on a run directory). ProjectInvalid if the project is not runnable."""
        project, domain = self._domain(project_id)
        try:
            return struts_for_project(project, domain, rho, grid, StrutRequest(**params))
        except ProblemInvalid as exc:
            raise ProjectInvalid(exc.issues) from exc

    def export(self, run_id: str) -> RunExport:
        rec = self.store.get_run(run_id)
        return RunExport(project=rec.project, run=rec.snapshot())

    def write_outputs(
        self,
        run_id: str,
        out_dir: str | os.PathLike,
        threshold: float = 0.5,
        smooth: int = 0,
        trim: bool = False,
    ) -> dict[str, Any]:
        """result.stl, result.png (iso), result.vti, density.npz, run.json into `out_dir`.

        Returns {"files": {name: path}, "errors": {name: why}, "warnings": [..]}; a file that
        cannot be made (e.g. a threshold above every density) is reported, the others are still
        written. `trim` intersects the STL (and the preview) with the design mesh; `warnings`
        says why that was not possible.
        """
        out = Path(out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []

        def with_warnings(make: Callable[[], tuple[bytes, list[str]]]) -> bytes:
            data, ws = make()
            warnings.extend(w for w in ws if w not in warnings)
            return data

        makers = {  # keys: OUTPUT_FILES
            "result.stl": lambda: with_warnings(
                lambda: self.export_stl(run_id, threshold, smooth, trim)
            ),
            "result.png": lambda: with_warnings(
                lambda: self.render_result(run_id, threshold, "iso", trim)
            ),
            "result.vti": lambda: self.result_vti(run_id),
            "density.npz": lambda: self.result_npz(run_id),
            "run.json": lambda: pretty_json(self.export(run_id).model_dump(mode="json")).encode(),
        }
        files: dict[str, str] = {}
        errors: dict[str, str] = {}
        for name, make in makers.items():
            try:
                (out / name).write_bytes(make())
                files[name] = str(out / name)
            except ValueError as exc:
                errors[name] = str(exc)
        return {"files": files, "errors": errors, "warnings": warnings}

    # ---- case files ---------------------------------------------------------------------------

    def load_case(self, path: str | os.PathLike) -> Project:
        """Read a case file (a `ProjectIn`, or a run.json), upload its meshes, store the project.

        `design_mesh.path` / `ref_models[].path` are relative to the case file. Selection
        shorthands (see `expand_selection`) are accepted.
        """
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"case file not found: {p}")
        raw = json.loads(p.read_text())
        if isinstance(raw, dict) and isinstance(raw.get("project"), dict) and "run" in raw:
            raw = raw["project"]  # a run.json written by `topop run`
        if not isinstance(raw, dict):
            raise ValueError("a case file must be a JSON object (ProjectIn)")  # noqa: TRY004
        for key in _ENVELOPE:  # a saved Project document is also accepted as a case
            raw.pop(key, None)
        for key in ("loads", "supports"):
            for item in raw.get(key) or []:
                if isinstance(item, dict) and isinstance(item.get("selection"), dict):
                    item["selection"] = expand_selection(item["selection"])
        body = ProjectIn.model_validate(raw)
        return self.store.create_project(self._materialize(body, p.resolve().parent))

    def save_case(self, project_id: str, path: str | os.PathLike) -> str:
        """Write the project as a case file; mesh `path`s are relative to the case file."""
        project = self.store.get_project(project_id)
        body = ProjectIn.model_validate(project.model_dump(exclude=_ENVELOPE))
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        for ref in [r for r in (body.design_mesh, *body.ref_models) if r is not None]:
            src = ref.path or (self._paths.get(ref.mesh_id) if ref.mesh_id else None)
            if src:
                try:
                    rel = os.path.relpath(src, out.parent)
                except ValueError:  # different drive
                    rel = str(src)
                ref.path = str(src) if rel.startswith("..") else rel
        body_json = body.model_dump(mode="json", exclude_none=True)
        out.write_text(pretty_json(body_json) + "\n")
        return str(out)


__all__ = [
    "OUTPUT_FILES",
    "NotFoundError",
    "ProjectInvalid",
    "Session",
    "expand_selection",
    "explain_validation_error",
    "parse_selection",
    "pretty_json",
]
