"""Project document (`schemas.ProjectIn`) -> voxel domain -> solvable `Problem`.

The single place that does this; the server, `topop run` and `topop mcp` all go through it.

Mesh keys: every mesh is keyed by `mesh_id` (server) or `path` (CLI case files). Selections
name meshes by that key. Two aliases are always present: `"design"` (the design mesh) and
`"ref:<RefModel.id>"` (each reference model, world space). When the design mesh and a reference
model share a key, the plain key means the design mesh (that is what the GUI's face picks refer to).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import trimesh

from topop.core.problem import Grid, Load, Material, Problem, RunParams, Support
from topop.core.selection import resolve_selection
from topop.core.voxelize import apply_transform, build_domain, domain_stats, transform_matrix
from topop.server.schemas import MaterialSpec, ParamsSpec, ProjectIn

DESIGN = "design"
SLOW_ACTIVE = 150_000
OOM_ACTIVE = 300_000

MeshResolver = Callable[[str], trimesh.Trimesh]


@dataclass
class BuiltDomain:
    grid: Grid
    active: np.ndarray  # bool (nx,ny,nz)
    passive: np.ndarray  # int8 (nx,ny,nz)
    stats: dict  # `VoxelStats` fields
    warnings: list[str]
    meshes_world: dict[str, trimesh.Trimesh]


@dataclass
class BuiltProblem(BuiltDomain):
    problem: Problem
    resolved: dict[str, np.ndarray]  # full-grid node ids per load/support id
    params: RunParams = field(default_factory=RunParams)


class ProblemInvalid(ValueError):
    """The project builds but is not runnable; `issues` lists why (Problem.validate() wording)."""

    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__("project is not runnable: " + "; ".join(self.issues))


def mesh_key(ref) -> str | None:
    """`MeshRef` / `RefModel` -> the key its mesh is stored under (mesh_id, else path)."""
    return ref.mesh_id or ref.path or None


def _world(raw: trimesh.Trimesh, t16: Sequence[float]) -> trimesh.Trimesh:
    # identity -> share the (read-only) raw mesh: same face ids, no copy of a large mesh
    if np.allclose(transform_matrix(t16), np.eye(4)):
        return raw
    return apply_transform(raw, t16)


def load_project_meshes(project: ProjectIn, resolver: MeshResolver) -> dict[str, trimesh.Trimesh]:
    """World-space meshes of the project keyed by mesh key, plus the `design` / `ref:<id>` aliases.

    `resolver(key)` returns the raw (untransformed) mesh. Meshes must be treated as read-only.
    """
    design = project.design_mesh
    dkey = mesh_key(design) if design is not None else None
    if design is None or dkey is None:
        raise ValueError("project has no design mesh")
    raw: dict[str, trimesh.Trimesh] = {}

    def get(key: str) -> trimesh.Trimesh:
        if key not in raw:
            raw[key] = resolver(key)
        return raw[key]

    out: dict[str, trimesh.Trimesh] = {}
    for ref in project.ref_models:
        key = mesh_key(ref)
        if key is None:
            continue
        world = _world(get(key), ref.transform)
        out[f"ref:{ref.id}"] = world
        out.setdefault(key, world)
    world = _world(get(dkey), design.transform)
    out[dkey] = world
    out[DESIGN] = world
    return out


def domain_key(project: ProjectIn) -> str:
    """Hash of every field the voxel domain depends on (cache key)."""
    d = project.design_mesh
    payload = {
        "design": None if d is None else [d.mesh_id, d.path, list(d.transform)],
        "refs": [[r.id, r.mesh_id, r.path, list(r.transform), r.mode] for r in project.ref_models],
        "grid": project.grid.model_dump(),
        "dtype": project.params.dtype,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def estimate_sec_per_iter(n_active: int) -> float:
    try:
        from topop.core import fem

        fn = getattr(fem, "estimate_seconds_per_iter", None)
        if fn is not None:
            return float(fn(n_active))
    except ImportError:
        pass
    return max(0.05, 2e-4 * n_active)


def estimate_bytes(n_active: int, dtype: str = "float64") -> int:
    from topop.core.fem import Assembler

    return int(Assembler.estimate_bytes(n_active, np.dtype(dtype)))


def size_warnings(n_active: int) -> list[str]:
    if n_active > OOM_ACTIVE:
        return [f"{n_active:,} active elements: likely out of memory; lower the resolution"]
    if n_active > SLOW_ACTIVE:
        return [f"{n_active:,} active elements: slow on 16 GB machines"]
    return []


def build_domain_from_project(
    project: ProjectIn, meshes_world: dict[str, trimesh.Trimesh]
) -> BuiltDomain:
    """Voxelize design + reference models onto the project grid and fill in `VoxelStats`."""
    if DESIGN not in meshes_world:
        raise ValueError("project has no design mesh")
    design = meshes_world[DESIGN]
    warnings: list[str] = []
    refs: list[tuple[trimesh.Trimesh, str]] = []
    for ref in project.ref_models:
        mesh = meshes_world.get(f"ref:{ref.id}")
        if mesh is None:
            warnings.append(f"reference model {ref.name or ref.id!r} has no mesh; ignored")
        else:
            refs.append((mesh, ref.mode))
    grid, active, passive, w = build_domain(
        design, refs, project.grid.elements_along_longest, project.grid.padding
    )
    warnings += w
    if not design.is_watertight and not any("watertight" in s for s in warnings):
        warnings.append("design mesh is not watertight; the voxelization may be approximate")
    stats = domain_stats(grid, active, passive)
    n_active = stats["n_active"]
    warnings += size_warnings(n_active)
    stats["est_bytes"] = estimate_bytes(n_active, project.params.dtype)
    stats["est_sec_per_iter"] = estimate_sec_per_iter(n_active)
    stats["warnings"] = list(warnings)
    return BuiltDomain(grid, active, passive, stats, warnings, meshes_world)


def _label(kind: str, item) -> str:
    return f"{kind} {item.name or item.id!r}"


def resolve_project_selections(
    project: ProjectIn, domain: BuiltDomain
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]]:
    """(node ids per load, per support, warnings for empty selections, selection errors)."""
    warnings: list[str] = []
    errors: list[str] = []

    def resolve(kind: str, item) -> np.ndarray:
        try:
            nodes = resolve_selection(
                item.selection.model_dump(), domain.grid, domain.active, domain.meshes_world
            )
        except ValueError as exc:
            errors.append(f"{_label(kind, item)}: {exc}")
            return np.zeros(0, dtype=np.int64)
        if nodes.size == 0:
            warnings.append(f"{_label(kind, item)} resolves to 0 nodes")
        return nodes

    load_nodes = [resolve("load", ld) for ld in project.loads]
    support_nodes = [resolve("support", sp) for sp in project.supports]
    return load_nodes, support_nodes, warnings, errors


def run_params(spec: ParamsSpec) -> RunParams:
    names = {f.name for f in dataclasses.fields(RunParams)}
    return RunParams(**{k: v for k, v in spec.model_dump().items() if k in names})


def material(spec: MaterialSpec) -> Material:
    return Material(E=spec.E, nu=spec.nu)


def build_problem(
    project: ProjectIn,
    meshes_world: dict[str, trimesh.Trimesh],
    domain: BuiltDomain | None = None,
) -> BuiltProblem:
    """Resolve every load/support and assemble the `Problem`. ProblemInvalid if not runnable."""
    if domain is None:
        domain = build_domain_from_project(project, meshes_world)
    load_nodes, support_nodes, sel_warnings, sel_errors = resolve_project_selections(
        project, domain
    )
    problem = Problem(
        grid=domain.grid,
        active=domain.active,
        passive=domain.passive,
        material=material(project.material),
        loads=[
            Load(nodes=n, force=tuple(ld.force), case=ld.case)
            for ld, n in zip(project.loads, load_nodes, strict=True)
        ],
        supports=[
            Support(nodes=n, fix=tuple(sp.fix))
            for sp, n in zip(project.supports, support_nodes, strict=True)
        ],
    )
    issues = sel_errors + problem.validate()
    if issues:
        raise ProblemInvalid(issues)
    warnings = [*domain.warnings, *sel_warnings]
    return BuiltProblem(
        grid=domain.grid,
        active=domain.active,
        passive=domain.passive,
        stats={**domain.stats, "warnings": list(warnings)},
        warnings=warnings,
        meshes_world=domain.meshes_world,
        problem=problem,
        resolved={
            **{sp.id: n for sp, n in zip(project.supports, support_nodes, strict=True)},
            **{ld.id: n for ld, n in zip(project.loads, load_nodes, strict=True)},
        },
        params=run_params(project.params),
    )
