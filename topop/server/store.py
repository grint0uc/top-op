"""Server state: uploaded meshes (disk + LRU), projects (JSON on disk), runs (memory + disk)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
import zipfile
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
RESULT_CACHE = 2  # finished runs whose density (from runs/{id}.npz) stays loaded
STALE_TMP_S = 600.0  # `_write_atomic` leftovers older than this are removed when a Store starts
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
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _remove_stale_tmp(directory: Path, older_than: float = STALE_TMP_S) -> None:
    """Delete `_write_atomic` temp files a crashed process left behind. Recent ones may belong to
    another process writing into the same data dir right now, so they are kept."""
    cutoff = time.time() - older_than
    for tmp in directory.glob(".*.tmp"):
        try:
            if tmp.is_file() and tmp.stat().st_mtime < cutoff:
                tmp.unlink()
        except OSError:
            pass


RunResult = tuple[np.ndarray, Grid, np.ndarray, np.ndarray]


@dataclass
class RunRecord:
    """A run of this process (live or finished) or one read back from runs/{id}.json.

    `built`, `rho`, `stress` and `latest_frame` are only held while the run is live, and after it
    finished only if persisting failed; otherwise results are read back from runs/{id}.npz.
    """

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
    subscribers: list = field(default_factory=list)  # jobs.Mailbox per WebSocket

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
            _remove_stale_tmp(d)
        self._lock = threading.RLock()
        self._meshes: OrderedDict[str, trimesh.Trimesh] = OrderedDict()
        self._mesh_locks: dict[str, threading.Lock] = {}
        self._facets: OrderedDict[tuple[str, float], tuple[list[dict], int]] = OrderedDict()
        # other processes (`topop mcp`, `topop run`) write into the same directories: projects are
        # re-read when their file changed, unknown ids and list calls look at the disk
        self._projects: dict[str, Project] = {}
        self._project_stamps: dict[str, tuple[int, ...]] = {}  # id -> `_stamp` of the file read
        self._domains: OrderedDict[str, tuple[str, BuiltDomain]] = OrderedDict()
        self._domain_locks: dict[str, threading.Lock] = {}
        self._runs: dict[str, RunRecord] = {}
        self._unreadable: dict[Path, tuple[int, ...]] = {}  # bad files, skipped until they change
        self._results: OrderedDict[str, tuple] = OrderedDict()  # run id -> from_npz_bytes(...)

    # ---- meshes ---------------------------------------------------------------------------------

    def add_mesh(self, data: bytes, filename: str) -> MeshInfo:
        """Store an upload; id = sha256 prefix of the bytes, so re-uploads get the same id.

        STEP files (by extension or header) are tessellated once; the original bytes, an STL and
        `{id}.brep.npz` (exact tessellation + B-rep face table) are kept so `get_mesh` never
        tessellates again. The same bytes uploaded again under another extension keep the file
        type they were first stored as.
        """
        step = is_step(filename) or is_step(data)
        ext = "step" if step else Path(filename).suffix.lower().lstrip(".")
        if not ext:
            raise ValueError(
                f"cannot tell the file type of {filename!r}; use .stl/.obj/.3mf/.ply/.step"
            )
        mesh_id = hashlib.sha256(data).hexdigest()[:16]
        stored = self._stored_type(mesh_id)
        file_type = stored or ext
        with self._lock:
            mesh = self._meshes.get(mesh_id)
        if file_type == "step":
            if mesh is None or stored is None:
                mesh = self._store_step(mesh_id, data)
        else:
            if mesh is None:
                mesh = load_mesh(data, file_type)  # ValueError on bad input, before any write
            if stored is None:
                _write_atomic(self.mesh_dir / f"{mesh_id}.{file_type}", data)
        meta = {"name": filename, "file_type": file_type}
        _write_atomic(self.mesh_dir / f"{mesh_id}.json", json.dumps(meta).encode())
        with self._lock:
            mesh = self._meshes.setdefault(mesh_id, mesh)
            self._meshes.move_to_end(mesh_id)
            while len(self._meshes) > MESH_CACHE:
                self._meshes.popitem(last=False)
        return self._info(mesh_id, filename, mesh)

    def _stored_type(self, mesh_id: str) -> str | None:
        """File type of an earlier upload of these bytes whose source file is still on disk."""
        try:
            file_type = self._mesh_meta(mesh_id)["file_type"]
        except (NotFoundError, KeyError, TypeError):
            return None
        source = "brep.npz" if file_type == "step" else file_type
        return file_type if (self.mesh_dir / f"{mesh_id}.{source}").is_file() else None

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

    @staticmethod
    def _stamp(path: Path) -> tuple[int, ...] | None:
        """Identity of a file version: `_write_atomic` renames a new inode into place, so the inode
        changes even when two saves land in the same mtime tick with the same size."""
        try:
            st = path.stat()
        except OSError:
            return None
        return st.st_ino, st.st_mtime_ns, st.st_size

    def _parse(self, path: Path, model: type, stamp: tuple[int, ...]):
        """`model` from a JSON file, or None (logged once per file version) if unreadable."""
        if self._unreadable.get(path) == stamp:
            return None
        try:
            return model.model_validate_json(path.read_bytes())
        except (OSError, ValueError) as exc:
            log.warning("skipping unreadable %s: %s", path, exc)
            with self._lock:
                self._unreadable[path] = stamp
            return None

    def _read_project(self, project_id: str) -> Project | None:
        """projects/{id}.json, parsed again only when the file changed (another process saved)."""
        path = self.project_dir / f"{project_id}.json"
        stamp = self._stamp(path)
        if stamp is None:
            return None
        with self._lock:
            if self._project_stamps.get(project_id) == stamp:
                return self._projects.get(project_id)
        p = self._parse(path, Project, stamp)
        if p is None or p.id != project_id:
            return None
        with self._lock:
            self._projects[project_id] = p
            self._project_stamps[project_id] = stamp
        return p

    def _save_project(self, p: Project) -> None:
        path = self.project_dir / f"{p.id}.json"
        _write_atomic(path, p.model_dump_json(indent=1).encode())
        stamp = self._stamp(path)
        with self._lock:
            self._projects[p.id] = p
            if stamp is not None:
                self._project_stamps[p.id] = stamp

    def create_project(self, body: ProjectIn) -> Project:
        now = now_iso()
        p = Project(**body.model_dump(), id=uuid.uuid4().hex[:12], created_at=now, updated_at=now)
        self._save_project(p)
        return p

    def list_projects(self) -> list[Project]:
        """Every project on disk (rescanned: other processes add projects too), oldest first."""
        for path in self.project_dir.glob("*.json"):
            if _SAFE_ID.fullmatch(path.stem):
                self._read_project(path.stem)
        with self._lock:
            projects = list(self._projects.values())
        return sorted(projects, key=lambda p: p.created_at)

    def get_project(self, project_id: str) -> Project:
        p = self._read_project(project_id) if _SAFE_ID.fullmatch(project_id) else None
        if p is None:
            with self._lock:  # saved by this process but the file is gone or unreadable
                p = self._projects.get(project_id)
        if p is None:
            raise NotFoundError(f"project {project_id} not found")
        return p

    def update_project(self, project_id: str, body: ProjectIn) -> Project:
        with self._lock:
            old = self.get_project(project_id)
            fields = body.model_dump(exclude={"id", "created_at", "updated_at"})
            p = Project(**fields, id=old.id, created_at=old.created_at, updated_at=now_iso())
            self._save_project(p)
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

    def _read_run(self, run_id: str) -> RunRecord | None:
        """A finished run from runs/{id}.json (written by this or another process), registered."""
        path = self.run_dir / f"{run_id}.json"
        stamp = self._stamp(path)
        if stamp is None:
            return None
        exp = self._parse(path, RunExport, stamp)
        if exp is None or exp.run.id != run_id:
            return None
        with self._lock:
            return self._runs.setdefault(run_id, RunRecord(info=exp.run, project=exp.project))

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
            self._runs[info.id] = rec
        return rec

    def get_run(self, run_id: str) -> RunRecord:
        with self._lock:
            rec = self._runs.get(run_id)
        if rec is None and _SAFE_ID.fullmatch(run_id):
            rec = self._read_run(run_id)
        if rec is None:
            raise NotFoundError(f"run {run_id} not found")
        return rec

    def list_runs(self) -> list[RunRecord]:
        """Runs of this process plus every finished run on disk (rescanned), oldest first."""
        for path in self.run_dir.glob("*.json"):
            with self._lock:
                known = path.stem in self._runs
            if not known and _SAFE_ID.fullmatch(path.stem):
                self._read_run(path.stem)
        with self._lock:
            runs = list(self._runs.values())
        return sorted(runs, key=lambda r: r.info.created_at)

    def persist_run(self, rec: RunRecord) -> None:
        """runs/{id}.npz (density) + runs/{id}.json (RunExport): exports survive a restart and
        other processes on the data dir see the run. Raises if a file could not be written."""
        if not _SAFE_ID.fullmatch(rec.info.id):
            return
        with rec.lock:
            rho, built, stress = rec.rho, rec.built, rec.stress
        if rho is not None and built is not None:
            _write_atomic(
                self.run_dir / f"{rec.info.id}.npz",
                to_npz_bytes(rho, built.grid, built.active, built.passive, stress),
            )
        exp = RunExport(project=rec.project, run=rec.snapshot())
        _write_atomic(self.run_dir / f"{rec.info.id}.json", exp.model_dump_json().encode())

    def finish_run(self, rec: RunRecord) -> str | None:
        """Persist a finished run, then release its arrays (exports read runs/{id}.npz back).

        If persisting fails the result stays in memory, "result not persisted: <reason>" is
        appended to the run's message, and that note is returned.
        """
        try:
            self.persist_run(rec)
        except Exception as exc:  # noqa: BLE001 - disk full, MemoryError while compressing, ...
            note = f"result not persisted: {str(exc) or type(exc).__name__}"
            log.warning("run %s: %s", rec.info.id, note)
            with rec.lock:
                msg = rec.info.message
                rec.info.message = rec.message = f"{msg}; {note}" if msg else note
            return note
        with rec.lock:
            rec.built = rec.rho = rec.stress = rec.latest_frame = None
        return None

    def _load_npz(self, run_id: str) -> tuple | None:
        """(rho, grid, active, passive, stress | None) from runs/{id}.npz; the last few cached."""
        with self._lock:
            hit = self._results.get(run_id)
            if hit is not None:
                self._results.move_to_end(run_id)
                return hit
        path = self.run_dir / f"{run_id}.npz"
        if not _SAFE_ID.fullmatch(run_id) or not path.is_file():
            return None
        try:
            res = from_npz_bytes(path.read_bytes(), with_stress=True)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            log.warning("unreadable run result %s: %s", path, exc)
            return None
        with self._lock:
            self._results[run_id] = res
            while len(self._results) > RESULT_CACHE:
                self._results.popitem(last=False)
        return res

    def run_result(self, rec: RunRecord) -> RunResult | None:
        """(rho, grid, active, passive) of a finished run, from memory or runs/{id}.npz.
        Reads the disk: call it off the event loop."""
        with rec.lock:
            rho, built = rec.rho, rec.built
        if rho is not None and built is not None:
            return rho, built.grid, built.active, built.passive
        res = self._load_npz(rec.info.id)
        return None if res is None else res[:4]

    def run_stress(self, rec: RunRecord) -> np.ndarray | None:
        """Von Mises field (nx,ny,nz) of a finished run from memory or runs/{id}.npz, else None."""
        with rec.lock:
            rho, built, stress = rec.rho, rec.built, rec.stress
        if rho is not None and built is not None:
            return stress
        res = self._load_npz(rec.info.id)
        return None if res is None else res[4]

    def design_world(self, rec: RunRecord) -> trimesh.Trimesh | None:
        """The run's design mesh in world space (for the ghosted result preview)."""
        built = rec.built
        if built is not None:
            return built.meshes_world.get("design")
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
