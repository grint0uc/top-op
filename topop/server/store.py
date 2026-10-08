"""Server state: uploaded meshes (disk + LRU), projects (JSON on disk), runs (memory + disk)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import trimesh
from fastapi import Request

from topop.core.export import from_npz_bytes, to_npz_bytes, trim_to_design
from topop.core.problem import Grid
from topop.core.selection import compute_facets
from topop.core.selection import facet_faces as mesh_facet_faces
from topop.core.step import META_FACETS, StepMesh, facet_triangles, is_step, load_step
from topop.core.voxelize import load_mesh, mesh_info
from topop.server.build import (
    BuiltDomain,
    BuiltProblem,
    build_domain_from_project,
    domain_key,
    load_project_meshes,
)
from topop.server.schemas import MeshInfo, Project, ProjectIn, RunExport, RunInfo

log = logging.getLogger(__name__)

MESH_CACHE = 20
FACET_CACHE = 20
DOMAIN_CACHE = 8
TERMINAL = ("done", "error", "cancelled")
_MESH_ID = re.compile(r"[0-9a-f]{16}")
_SAFE_ID = re.compile(r"[0-9A-Za-z_-]{1,64}")


class NotFoundError(LookupError):
    pass


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_data_dir() -> Path:
    return Path(os.environ.get("TOPOP_DATA_DIR") or "~/.cache/topop").expanduser()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


@dataclass
class RunRecord:
    info: RunInfo
    project: Project  # snapshot taken when the run was created
    built: BuiltProblem | None = None
    rho: np.ndarray | None = None
    stress: np.ndarray | None = None  # von Mises per element of the final design (or None)
    latest_frame: bytes | None = None
    latest_frame_it: int = -1
    message: str | None = None  # final status message (StatusMsg.message)
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    subscribers: list = field(default_factory=list)  # (asyncio loop, asyncio.Queue)
    # result loaded back from runs/{id}.npz after a restart: (rho, grid, active, passive)
    _from_disk: tuple[np.ndarray, Grid, np.ndarray, np.ndarray] | None = None
    _stress_from_disk: np.ndarray | None = None

    @property
    def finished(self) -> bool:
        return self.info.status in TERMINAL

    def snapshot(self) -> RunInfo:
        with self.lock:
            return self.info.model_copy(deep=True)


class Store:
    def __init__(self, data_dir: str | os.PathLike | None = None):
        self.root = Path(data_dir).expanduser() if data_dir else default_data_dir()
        self.mesh_dir = self.root / "meshes"
        self.project_dir = self.root / "projects"
        self.run_dir = self.root / "runs"
        for d in (self.mesh_dir, self.project_dir, self.run_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._meshes: OrderedDict[str, trimesh.Trimesh] = OrderedDict()
        self._mesh_locks: dict[str, threading.Lock] = {}
        self._facets: OrderedDict[tuple[str, float], tuple[list[dict], int]] = OrderedDict()
        self._projects: dict[str, Project] | None = None
        self._domains: OrderedDict[str, tuple[str, BuiltDomain]] = OrderedDict()
        self._domain_locks: dict[str, threading.Lock] = {}
        self._runs: dict[str, RunRecord] | None = None

    # ---- meshes ---------------------------------------------------------------------------------

    def add_mesh(self, data: bytes, filename: str) -> MeshInfo:
        """Store an upload; id = sha256 prefix of the bytes, so re-uploads get the same id.

        STEP files (by extension or header) are tessellated once; the original bytes, an STL and
        `{id}.brep.npz` (exact tessellation + B-rep face table) are kept so `get_mesh` never
        tessellates again.
        """
        step = is_step(filename) or is_step(data)
        ext = "step" if step else Path(filename).suffix.lower().lstrip(".")
        if not ext:
            raise ValueError(
                f"cannot tell the file type of {filename!r}; use .stl/.obj/.3mf/.ply/.step"
            )
        mesh_id = hashlib.sha256(data).hexdigest()[:16]
        with self._lock:
            mesh = self._meshes.get(mesh_id)
        if mesh is None and step:
            mesh = self._store_step(mesh_id, data)
        elif mesh is None:
            mesh = load_mesh(data, ext)  # ValueError on bad input, before anything is written
            path = self.mesh_dir / f"{mesh_id}.{ext}"
            if not path.exists():
                _write_atomic(path, data)
        meta = {"name": filename, "file_type": ext}
        _write_atomic(self.mesh_dir / f"{mesh_id}.json", json.dumps(meta).encode())
        with self._lock:
            mesh = self._meshes.setdefault(mesh_id, mesh)
            self._meshes.move_to_end(mesh_id)
            while len(self._meshes) > MESH_CACHE:
                self._meshes.popitem(last=False)
        return self._info(mesh_id, filename, mesh)

    def _store_step(self, mesh_id: str, data: bytes) -> trimesh.Trimesh:
        cached = self._read_step_cache(mesh_id)
        if cached is not None:
            return cached.mesh
        sm = self._tessellate(data)
        self._write_step_files(mesh_id, data, sm)
        return sm.mesh

    @staticmethod
    def _tessellate(data: bytes) -> StepMesh:
        try:
            return load_step(data)
        except ImportError as exc:  # the OpenCascade extra is missing: report it like bad input
            raise ValueError(str(exc)) from exc

    def _write_step_files(self, mesh_id: str, data: bytes, sm: StepMesh) -> None:
        _write_atomic(self.mesh_dir / f"{mesh_id}.step", data)
        _write_atomic(self.mesh_dir / f"{mesh_id}.stl", bytes(sm.mesh.export(file_type="stl")))
        # last: its presence marks a complete cache
        _write_atomic(self.mesh_dir / f"{mesh_id}.brep.npz", sm.to_npz_bytes())

    def _read_step_cache(self, mesh_id: str) -> StepMesh | None:
        try:
            return StepMesh.from_npz_bytes((self.mesh_dir / f"{mesh_id}.brep.npz").read_bytes())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _info(mesh_id: str, name: str, mesh: trimesh.Trimesh) -> MeshInfo:
        facets = mesh.metadata.get(META_FACETS)
        step = {} if facets is None else {"source": "step", "n_brep_faces": len(facets)}
        return MeshInfo(id=mesh_id, name=name, **mesh_info(mesh), **step)

    def mesh_info(self, mesh_id: str) -> MeshInfo:
        name = self._mesh_meta(mesh_id).get("name", mesh_id)
        return self._info(mesh_id, name, self.get_mesh(mesh_id))

    def _mesh_meta(self, mesh_id: str) -> dict:
        if not _MESH_ID.fullmatch(mesh_id):
            raise NotFoundError(f"mesh {mesh_id} not found")
        try:
            return json.loads((self.mesh_dir / f"{mesh_id}.json").read_text())
        except (OSError, ValueError) as exc:
            raise NotFoundError(f"mesh {mesh_id} not found (upload it again)") from exc

    def _load_from_disk(self, mesh_id: str, file_type: str) -> trimesh.Trimesh:
        if file_type == "step":
            cached = self._read_step_cache(mesh_id)
            if cached is not None:
                return cached.mesh
        try:
            data = (self.mesh_dir / f"{mesh_id}.{file_type}").read_bytes()
        except OSError as exc:
            raise NotFoundError(f"mesh {mesh_id} not found (upload it again)") from exc
        if file_type != "step":
            return load_mesh(data, file_type)
        sm = self._tessellate(data)  # cache lost or damaged: tessellate the original again
        self._write_step_files(mesh_id, data, sm)
        return sm.mesh

    def get_mesh(self, mesh_id: str) -> trimesh.Trimesh:
        """The processed mesh every face-serving endpoint uses (face ids agree). Read-only."""
        with self._lock:
            mesh = self._meshes.get(mesh_id)
            if mesh is not None:
                self._meshes.move_to_end(mesh_id)
                return mesh
            lock = self._mesh_locks.setdefault(mesh_id, threading.Lock())
        with lock:
            with self._lock:
                if mesh_id in self._meshes:
                    return self._meshes[mesh_id]
            mesh = self._load_from_disk(mesh_id, self._mesh_meta(mesh_id)["file_type"])
            with self._lock:
                self._meshes[mesh_id] = mesh
                while len(self._meshes) > MESH_CACHE:
                    self._meshes.popitem(last=False)
            return mesh

    def mesh_facets(self, mesh_id: str, angle_deg: float) -> tuple[list[dict], int]:
        """(facets sorted by area, total count). Cached per (mesh, angle).

        STEP meshes list their B-rep faces (exact kind/radius/axis, `brep_face` set) and ignore
        the angle.
        """
        mesh = self.get_mesh(mesh_id)
        step_facets = mesh.metadata.get(META_FACETS)
        if step_facets is not None:
            return step_facets, len(step_facets)
        key = (mesh_id, float(angle_deg))
        with self._lock:
            hit = self._facets.get(key)
            if hit is not None:
                self._facets.move_to_end(key)
                return hit
        facets, _ = compute_facets(mesh, float(angle_deg))
        out = (facets, len(facets))
        with self._lock:
            self._facets[key] = out
            while len(self._facets) > FACET_CACHE:
                self._facets.popitem(last=False)
        return out

    def facet_faces(
        self, mesh_id: str, facet_ids: Sequence[int], angle_deg: float = 5.0
    ) -> np.ndarray:
        """Triangle ids of the given facets (ids as listed by `mesh_facets`). ValueError if unknown."""
        mesh = self.get_mesh(mesh_id)
        tris = facet_triangles(mesh, facet_ids)
        return tris if tris is not None else mesh_facet_faces(mesh, float(angle_deg), facet_ids)

    # ---- projects -------------------------------------------------------------------------------

    def _project_map(self) -> dict[str, Project]:
        with self._lock:
            if self._projects is None:
                projects: dict[str, Project] = {}
                for path in sorted(self.project_dir.glob("*.json")):
                    try:
                        p = Project.model_validate_json(path.read_bytes())
                    except (OSError, ValueError) as exc:
                        log.warning("skipping unreadable project %s: %s", path, exc)
                        continue
                    projects[p.id] = p
                self._projects = dict(sorted(projects.items(), key=lambda kv: kv[1].created_at))
            return self._projects

    def _save_project(self, p: Project) -> None:
        _write_atomic(self.project_dir / f"{p.id}.json", p.model_dump_json(indent=1).encode())

    def create_project(self, body: ProjectIn) -> Project:
        now = now_iso()
        p = Project(**body.model_dump(), id=uuid.uuid4().hex[:12], created_at=now, updated_at=now)
        with self._lock:
            self._save_project(p)
            self._project_map()[p.id] = p
        return p

    def list_projects(self) -> list[Project]:
        with self._lock:
            return list(self._project_map().values())

    def get_project(self, project_id: str) -> Project:
        with self._lock:
            p = self._project_map().get(project_id)
        if p is None:
            raise NotFoundError(f"project {project_id} not found")
        return p

    def update_project(self, project_id: str, body: ProjectIn) -> Project:
        with self._lock:
            old = self.get_project(project_id)
            fields = body.model_dump(exclude={"id", "created_at", "updated_at"})
            p = Project(**fields, id=old.id, created_at=old.created_at, updated_at=now_iso())
            self._save_project(p)
            self._project_map()[p.id] = p
        return p

    def get_domain(self, project: Project) -> BuiltDomain:
        """Voxel domain of the project, cached until a relevant field (mesh, transform, grid) changes."""
        key = domain_key(project)
        with self._lock:
            hit = self._domains.get(project.id)
            if hit is not None and hit[0] == key:
                self._domains.move_to_end(project.id)
                return hit[1]
            lock = self._domain_locks.setdefault(project.id, threading.Lock())
        with lock:
            with self._lock:
                hit = self._domains.get(project.id)
                if hit is not None and hit[0] == key:
                    return hit[1]
            meshes = load_project_meshes(project, self.get_mesh)
            domain = build_domain_from_project(project, meshes)
            with self._lock:
                self._domains[project.id] = (key, domain)
                while len(self._domains) > DOMAIN_CACHE:
                    self._domains.popitem(last=False)
            return domain

    # ---- runs -----------------------------------------------------------------------------------

    def _run_map(self) -> dict[str, RunRecord]:
        with self._lock:
            if self._runs is None:
                runs: dict[str, RunRecord] = {}
                for path in self.run_dir.glob("*.json"):
                    try:
                        exp = RunExport.model_validate_json(path.read_bytes())
                    except (OSError, ValueError) as exc:
                        log.warning("skipping unreadable run %s: %s", path, exc)
                        continue
                    runs[exp.run.id] = RunRecord(info=exp.run, project=exp.project)
                self._runs = dict(sorted(runs.items(), key=lambda kv: kv[1].info.created_at))
            return self._runs

    def new_run(self, project: Project, built: BuiltProblem, stats) -> RunRecord:
        info = RunInfo(
            id=uuid.uuid4().hex[:12],
            project_id=project.id,
            status="queued",
            created_at=now_iso(),
            stats=stats,
        )
        rec = RunRecord(info=info, project=project.model_copy(deep=True), built=built)
        with self._lock:
            self._run_map()[info.id] = rec
        return rec

    def get_run(self, run_id: str) -> RunRecord:
        with self._lock:
            rec = self._run_map().get(run_id)
        if rec is None:
            raise NotFoundError(f"run {run_id} not found")
        return rec

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            return list(self._run_map().values())

    def persist_run(self, rec: RunRecord) -> None:
        """runs/{id}.npz (density) + runs/{id}.json (RunExport) so exports survive a restart."""
        if not _SAFE_ID.fullmatch(rec.info.id):
            return
        try:
            if rec.rho is not None and rec.built is not None:
                b = rec.built
                _write_atomic(
                    self.run_dir / f"{rec.info.id}.npz",
                    to_npz_bytes(rec.rho, b.grid, b.active, b.passive, rec.stress),
                )
            exp = RunExport(project=rec.project, run=rec.snapshot())
            _write_atomic(self.run_dir / f"{rec.info.id}.json", exp.model_dump_json().encode())
        except OSError as exc:
            log.warning("could not persist run %s: %s", rec.info.id, exc)

    def run_result(self, rec: RunRecord) -> tuple[np.ndarray, Grid, np.ndarray, np.ndarray] | None:
        """(rho, grid, active, passive) of a finished run, from memory or runs/{id}.npz."""
        if rec.rho is not None and rec.built is not None:
            return rec.rho, rec.built.grid, rec.built.active, rec.built.passive
        return self._load_npz(rec)

    def _load_npz(self, rec: RunRecord) -> tuple[np.ndarray, Grid, np.ndarray, np.ndarray] | None:
        if rec._from_disk is None:
            path = self.run_dir / f"{rec.info.id}.npz"
            if not _SAFE_ID.fullmatch(rec.info.id) or not path.is_file():
                return None
            rho, grid, active, passive, stress = from_npz_bytes(path.read_bytes(), with_stress=True)
            rec._stress_from_disk = stress
            rec._from_disk = (rho, grid, active, passive)
        return rec._from_disk

    def run_stress(self, rec: RunRecord) -> np.ndarray | None:
        """Von Mises field (nx,ny,nz) of a finished run from memory or runs/{id}.npz, else None."""
        if rec.stress is not None:
            return rec.stress
        if rec.rho is None or rec.built is None:  # not in memory: a run loaded from disk
            self._load_npz(rec)
            return rec._stress_from_disk
        return None

    def design_world(self, rec: RunRecord) -> trimesh.Trimesh | None:
        """The run's design mesh in world space (for the ghosted result preview)."""
        if rec.built is not None:
            return rec.built.meshes_world.get("design")
        try:
            meshes = load_project_meshes(
                rec.project.model_copy(update={"ref_models": []}), self.get_mesh
            )
        except (NotFoundError, ValueError):
            return None
        return meshes.get("design")

    def trim_result(
        self, rec: RunRecord, mesh: trimesh.Trimesh
    ) -> tuple[trimesh.Trimesh, list[str]]:
        """`mesh` intersected with the run's world-space design mesh, plus why it could not be."""
        design = self.design_world(rec)
        if design is None:
            return mesh, ["not trimmed to the design: the design mesh is not available"]
        return trim_to_design(mesh, design)


def store_of(app) -> Store:
    """The app's store. The lifespan creates it; fall back to creating it lazily (no lifespan)."""
    store = getattr(app.state, "store", None)
    if store is None:
        store = app.state.store = Store()
    return store


def get_store(request: Request) -> Store:
    return store_of(request.app)
