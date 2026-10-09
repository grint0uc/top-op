"""Project document (`schemas.ProjectIn`) -> voxel domain -> solvable `Problem`.

The single place that does this; the server, `topop run` and `topop mcp` all go through it.

Mesh keys: every mesh is keyed by `mesh_id` (server) or `path` (CLI case files). Selections
name meshes by that key. Two aliases are always present: `"design"` (the design mesh) and
`"ref:<RefModel.id>"` (each reference model, world space). When the design mesh and a reference
model share a key, the plain key means the design mesh (that is what the GUI's face picks refer to).

Facet ids are listed on the raw (untransformed) mesh (`/meshes/{id}/facets`, `topop describe`), so
`facets` selections are turned into triangle ids on the raw mesh and resolved on the world mesh
(triangle ids survive the transform; facet ids do not under a non-uniform scale).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import trimesh

from topop.core.problem import (
    Grid,
    Load,
    Material,
    Problem,
    RunParams,
    Support,
    SymmetryPlane,
)
from topop.core.selection import facet_faces, resolve_selection
from topop.core.step import facet_triangles
from topop.core.voxelize import (
    apply_transform,
    build_domain,
    domain_grid,
    domain_stats,
    transform_matrix,
)
from topop.server.schemas import MaterialSpec, ParamsSpec, ProjectIn

DESIGN = "design"
SLOW_ACTIVE = 150_000
OOM_ACTIVE = 300_000
# peak bytes per grid cell of voxelization + stats + selections (measured 21-25 B on 8-16M cells,
# closed and open meshes): grids above memory_cap_bytes / this are refused before voxelizing
VOXEL_BYTES_PER_CELL = 50

MeshResolver = Callable[[str], trimesh.Trimesh]


class ProjectMeshes(dict[str, trimesh.Trimesh]):
    """World-space meshes by key (a plain dict to every caller) + `raw`: the untransformed mesh
    under the same keys, which facet ids refer to."""

    def __init__(self, *args, raw: dict[str, trimesh.Trimesh] | None = None, **kw):
        super().__init__(*args, **kw)
        self.raw: dict[str, trimesh.Trimesh] = dict(raw or {})


@dataclass
class BuiltDomain:
    grid: Grid
    active: np.ndarray  # bool (nx,ny,nz)
    passive: np.ndarray  # int8 (nx,ny,nz)
    stats: dict  # `VoxelStats` fields
    warnings: list[str]
    meshes_world: dict[str, trimesh.Trimesh]
    # untransformed meshes under the keys of `meshes_world` (empty: facets resolve on the world mesh)
    meshes_raw: dict[str, trimesh.Trimesh] = field(default_factory=dict, kw_only=True)


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


def load_project_meshes(project: ProjectIn, resolver: MeshResolver) -> ProjectMeshes:
    """World-space meshes of the project keyed by mesh key, plus the `design` / `ref:<id>` aliases;
    `.raw` holds the untransformed mesh under every key.

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

    out = ProjectMeshes()
    for ref in project.ref_models:
        key = mesh_key(ref)
        if key is None:
            continue
        world = _world(get(key), ref.transform)
        out[f"ref:{ref.id}"] = world
        out.raw[f"ref:{ref.id}"] = raw[key]
        if key not in out:
            out[key], out.raw[key] = world, raw[key]
    world = _world(get(dkey), design.transform)
    out[dkey] = out[DESIGN] = world
    out.raw[dkey] = out.raw[DESIGN] = raw[dkey]
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


def estimate_bytes(n_active: int, dtype: str = "float64", n_cases: int = 1) -> int:
    from topop.core.fem import Assembler

    return int(Assembler.estimate_bytes(n_active, np.dtype(dtype), n_cases))


def memory_cap(project: ProjectIn) -> int:
    return int(run_params(project.params).memory_cap_bytes)


def _check_voxel_memory(project: ProjectIn, grid: Grid) -> None:
    """ProblemInvalid when voxelizing `grid` alone would exceed the memory cap."""
    cap = memory_cap(project)
    n_cells = grid.nel
    need = VOXEL_BYTES_PER_CELL * n_cells
    if need <= cap:
        return
    nx, ny, nz = grid.shape
    msg = (
        f"grid {nx} x {ny} x {nz} = {n_cells / 1e6:.0f}M cells needs about "
        f"{need / 1e9:.1f} GB to voxelize, above the {cap / 1e9:.1f} GB memory cap; "
        f"lower elements_along_longest ({project.grid.elements_along_longest})"
    )
    raise ProblemInvalid([msg])


def _check_solve_memory(project: ProjectIn, n_active: int, n_cases: int) -> int:
    """Estimated run bytes; ProblemInvalid when they exceed the memory cap."""
    cap = memory_cap(project)
    need = estimate_bytes(n_active, project.params.dtype, n_cases)
    if need > cap:
        lo, hi = 0, n_active  # largest element count that fits; n_active scales with eal^3
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if estimate_bytes(mid, project.params.dtype, n_cases) <= cap:
                lo = mid
            else:
                hi = mid - 1
        eal = project.grid.elements_along_longest
        fits = max(4, int(0.95 * eal * (lo / n_active) ** (1 / 3)))  # margin: surface effects
        msg = (
            f"{n_active:,} active elements ({n_cases} load case(s)) need about "
            f"{need / 1e9:.1f} GB, above the {cap / 1e9:.1f} GB memory cap; lower "
            f"elements_along_longest ({eal}) to about {fits}"
        )
        raise ProblemInvalid([msg])
    return need


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
    eal, padding = project.grid.elements_along_longest, project.grid.padding
    _check_voxel_memory(project, domain_grid(design, refs, eal, padding))
    grid, active, passive, w = build_domain(design, refs, eal, padding)
    warnings += w
    if not design.is_watertight and not any("watertight" in s for s in warnings):
        warnings.append("design mesh is not watertight; the voxelization may be approximate")
    stats = domain_stats(grid, active, passive)
    n_active = stats["n_active"]
    warnings += size_warnings(n_active)
    stats["est_bytes"] = estimate_bytes(n_active, project.params.dtype)
    stats["est_sec_per_iter"] = estimate_sec_per_iter(n_active)
    stats["warnings"] = list(warnings)
    raw = getattr(meshes_world, "raw", {})
    return BuiltDomain(grid, active, passive, stats, warnings, meshes_world, meshes_raw=raw)


def step_facets_to_faces(sel: Mapping, meshes: Mapping[str, trimesh.Trimesh]) -> Mapping:
    """A `facets` selection on a STEP-sourced mesh -> the equivalent `faces` selection.

    Facet ids of a STEP mesh are B-rep faces (`/facets` table), not angle-grouped triangles, so
    `core.selection` must never see them. Any other selection is returned unchanged.
    """
    if sel.get("kind") != "facets":
        return sel
    mesh = meshes.get(sel.get("mesh_id"))
    tris = None if mesh is None else facet_triangles(mesh, sel.get("facet_ids", []))
    if tris is None:
        return sel
    return {"kind": "faces", "mesh_id": sel["mesh_id"], "face_ids": tris.tolist()}


def facets_to_faces(
    sel: Mapping, meshes_world: Mapping[str, trimesh.Trimesh], meshes_raw: Mapping
) -> Mapping:
    """A `facets` selection -> the `faces` selection of the same triangles, facet ids taken on
    the RAW mesh (where they were listed). Without a raw mesh for the key: `step_facets_to_faces`
    (facets of other meshes then resolve on the world mesh). Other selections are unchanged."""
    if sel.get("kind") != "facets":
        return sel
    key = sel.get("mesh_id")
    raw, world = meshes_raw.get(key), meshes_world.get(key)
    if raw is None or world is None or len(raw.faces) != len(world.faces):
        return step_facets_to_faces(sel, meshes_world)
    ids = sel.get("facet_ids", [])
    tris = facet_triangles(raw, ids)
    if tris is None:
        tris = facet_faces(raw, float(sel.get("angle_deg", 5.0)), ids)
    return {"kind": "faces", "mesh_id": key, "face_ids": tris}


def resolve_sel(sel: Mapping, domain: BuiltDomain) -> np.ndarray:
    """`core.selection.resolve_selection` on the domain; facet ids are those listed for the raw
    mesh (STEP: B-rep faces). The one entry point to use."""
    meshes = domain.meshes_world
    sel = facets_to_faces(sel, meshes, domain.meshes_raw)
    return resolve_selection(sel, domain.grid, domain.active, meshes)


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
            nodes = resolve_sel(item.selection.model_dump(), domain)
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
    fields = {k: v for k, v in spec.model_dump().items() if k in names}
    fields["symmetry"] = tuple(SymmetryPlane(s.axis, s.position) for s in spec.symmetry)
    return RunParams(**fields)


def params_warnings(spec: ParamsSpec) -> list[str]:
    """Notes about parameter combinations that do not do what they look like they do.

    Not part of the cached domain (`BuiltDomain.warnings`): params do not change the voxels.
    """
    out: list[str] = []
    if spec.stress_limit is not None and spec.optimizer == "oc":
        out.append(
            "stress_limit is set: the optimizer is forced to mma (oc has no stress constraint)"
        )
    if spec.overhang is not None:
        sign, axis = spec.overhang[0], spec.overhang[1]
        face = "min" if sign == "+" else "max"
        out.append(
            f"overhang {spec.overhang}: the base plate is the domain's {face} {axis} face "
            "(the part grows from it; no support structures are generated)"
        )
    return out


def material(spec: MaterialSpec) -> Material:
    return Material(E=spec.E, nu=spec.nu)


def build_problem(
    project: ProjectIn,
    meshes_world: dict[str, trimesh.Trimesh],
    domain: BuiltDomain | None = None,
) -> BuiltProblem:
    """Resolve every load/support and assemble the `Problem`. ProblemInvalid if not runnable.

    Load cases are renumbered to the ones in use (sorted), so cases {0, 15} become {0, 1}.
    A run that would exceed the memory cap is refused before anything is resolved.
    """
    if domain is None:
        domain = build_domain_from_project(project, meshes_world)
    case_index = {c: i for i, c in enumerate(sorted({ld.case for ld in project.loads}))}
    n_cases = max(1, len(case_index))
    est = _check_solve_memory(project, int(domain.stats["n_active"]), n_cases)
    load_nodes, support_nodes, sel_warnings, sel_errors = resolve_project_selections(
        project, domain
    )
    problem = Problem(
        grid=domain.grid,
        active=domain.active,
        passive=domain.passive,
        material=material(project.material),
        loads=[
            Load(nodes=n, force=tuple(ld.force), case=case_index[ld.case])
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
    warnings = [*domain.warnings, *sel_warnings, *params_warnings(project.params)]
    return BuiltProblem(
        grid=domain.grid,
        active=domain.active,
        passive=domain.passive,
        stats={**domain.stats, "est_bytes": est, "warnings": list(warnings)},
        warnings=warnings,
        meshes_world=domain.meshes_world,
        meshes_raw=domain.meshes_raw,
        problem=problem,
        resolved={
            **{sp.id: n for sp, n in zip(project.supports, support_nodes, strict=True)},
            **{ld.id: n for ld, n in zip(project.loads, load_nodes, strict=True)},
        },
        params=run_params(project.params),
    )
