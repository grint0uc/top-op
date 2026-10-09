"""MCP server (stdio) over `topop.agent.Session`: drive the whole workflow without the GUI.

Every tool returns plain JSON-able dicts (or an `Image`); failures raise `ToolError` so the model
sees the real message. Tool docstrings are the agent's manual.
"""

import contextlib
import functools
import inspect
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import anyio
import anyio.to_thread
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from topop.agent import (
    OUTPUT_FILES,
    ProjectInvalid,
    Session,
    explain_validation_error,
    parse_selection,
    pretty_json,
)
from topop.server.schemas import (
    IDENTITY,
    LoadSpec,
    MeshRef,
    Project,
    ProjectIn,
    RefModel,
    RunInfo,
    SupportSpec,
    SymmetrySpec,
)

INSTRUCTIONS = """\
top-op: 3D topology optimization (voxel SIMP compliance minimisation) of a mesh you name regions on.
Loop: load_mesh -> describe_mesh (+ preview_mesh) -> create_project -> add_support / add_load
(select regions by facet id, face normal, plane or primitive) -> voxel_stats (node counts, warnings,
size) -> run (try a coarse grid or few iterations first) -> result_preview -> adjust -> export_stl.
Optional design rules via set_params: symmetry (mirror planes), stress_limit (von Mises, forces the
mma optimizer; check it with result_stress_summary) and overhang (additive-manufacturing build
direction). export_stl(trim=true) clips the result to the CAD surface. STEP files load like meshes;
their facets are the exact B-rep faces, and a hole is the facet of kind "cylinder" with its radius.
Units: whatever the mesh is in; E, forces and lengths must be consistent (compliance is in those units).
Size: keep active elements <= ~150k on a 16 GB machine (voxel_stats shows n_active, memory and time
estimates); 30-40 elements along the longest side to debug a setup, 60-100 for a real result.
Always check that a load/support resolved to a sensible node count before running.
"""

SELECTION_HELP = """\
SELECTION: a JSON object; "kind" picks the shape. Coordinates are in the mesh's own units, in WORLD
space: the mesh as the project places it (design_transform applied). describe_mesh(mesh_id,
project_id=...) lists the facets in that frame; without project_id its normals/centroids/bboxes are the
raw file's, which only agree while the design_transform is the identity. Facet ids agree in both frames.
 facets   {"kind":"facets","facet_ids":[0,3]}   facet ids from describe_mesh: coplanar groups and cylinders
          (same angle_deg, default 5); for STEP meshes the ids are exact B-rep faces. A hole = the facet
          with kind "cylinder" and the radius you want; facet_faces(mesh_id, facet_id) lists its triangles
 normal   {"kind":"normal","direction":[0,0,1],"angle_deg":10,"within":[[xmin,ymin,zmin],[xmax,ymax,zmax]]}
          surface faces pointing within angle_deg of direction ("within" optionally clips to a box)
 plane    {"kind":"plane","point":[0,0,0],"normal":[1,0,0],"tol":0}   surface nodes on the plane x=0
          (tol 0 = the one node layer nearest the plane, +-h/2); needs no mesh
 box      {"kind":"box","min":[x,y,z],"max":[x,y,z]}   or "center":[..] with "size":[sx,sy,sz]
 sphere   {"kind":"sphere","center":[x,y,z],"radius":r}
 cylinder {"kind":"cylinder","center":[x,y,z],"radius":r,"height":h,"axis":"z"}   axis is x|y|z or a
          vector; the height runs along the axis, centred on `center`
 faces    {"kind":"faces","face_ids":[...]}   raw triangle ids (what the GUI produces); avoid
mesh_id (faces/facets/normal) defaults to "design", the project's design mesh ("ref:<id>" = a reference model).
Primitives pick the SURFACE grid nodes inside them ("surface_only": false adds interior nodes).
Canonical primitive form (what is stored): "transform" = 16 floats COLUMN-MAJOR (three.js
Matrix4.toArray(); translation = elements 12,13,14) plus "size": box = unit cube centred at the origin
scaled by size; sphere radius = size[0]; cylinder axis = local Y, radius size[0], height size[1].
PITFALL (within): grid nodes lie up to h/2 outside the true surface and `within` clips strictly on node
coordinates, so pad every box derived from the mesh bbox by one voxel h on each side (h is in voxel_stats;
about longest_side / elements_along_longest) and re-check after changing the resolution.
Check the returned node count/bbox: 0 nodes means the selection missed."""


def _round(obj: Any, sig: int = 6) -> Any:
    """Floats to `sig` significant digits (keeps tool output short); containers recursed."""
    if isinstance(obj, float):
        return float(f"{obj:.{sig}g}")
    if isinstance(obj, dict):
        return {k: _round(v, sig) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_round(v, sig) for v in obj]
    return obj


def _dump(model: Any) -> dict:
    return model.model_dump(mode="json")


def _absolute(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        raise ValueError(
            f"path must be absolute: {path!r} would be resolved against the MCP server's working "
            f"directory ({Path.cwd()}), not yours"
        )
    return p


def _output_path(path: str, suffix: str, overwrite: bool) -> Path:
    """An absolute `path` ending in `suffix` that is new, or an existing regular file of that type
    when `overwrite` is set. Anything else is refused (ValueError)."""
    p = _absolute(path)
    if p.suffix.lower() != suffix:
        raise ValueError(f"{p}: the file name must end in {suffix}")
    if p.is_symlink():
        raise ValueError(f"{p} is a symbolic link; refusing to write through it")
    if p.exists():
        if not p.is_file():
            raise ValueError(f"{p} exists and is not a regular file")
        if not overwrite:
            raise ValueError(f"{p} already exists; pass overwrite=true to replace it")
    return p


def _brief(p: Project) -> dict:
    """Project summary without the (possibly long) selections."""
    return {
        "project_id": p.id,
        "name": p.name,
        "design_mesh": _dump(p.design_mesh) if p.design_mesh else None,
        "ref_models": [
            {"id": r.id, "name": r.name, "mesh_id": r.mesh_id, "mode": r.mode} for r in p.ref_models
        ],
        "grid": _dump(p.grid),
        "material": _dump(p.material),
        "params": _dump(p.params),
        "loads": [
            {"id": x.id, "name": x.name, "case": x.case, "force": x.force, "kind": x.selection.kind}
            for x in p.loads
        ],
        "supports": [
            {"id": x.id, "name": x.name, "fix": x.fix, "kind": x.selection.kind} for x in p.supports
        ],
    }


def _render(result: Any) -> Any:
    """dict results become compact JSON text (the SDK's default puts every number on its own line)."""
    return pretty_json(_round(result), width=220) if isinstance(result, dict) else result


def _guard(fn: Callable) -> Callable:
    """Expected failures become ToolError (the model sees the message); crashes stay crashes."""

    def explain(exc: BaseException) -> ToolError:
        if isinstance(exc, ProjectInvalid):
            return ToolError(
                "project is not runnable: "
                + "; ".join(exc.issues)
                + ". Fix with add_load/add_support/remove_*; voxel_stats shows what each resolves to."
            )
        if isinstance(exc, ValidationError):
            return ToolError(f"invalid input: {explain_validation_error(exc)}")
        if isinstance(exc, MemoryError):
            return ToolError(f"out of memory: {exc}. Lower elements_along_longest (set_grid).")
        text = str(exc.args[0]) if isinstance(exc, LookupError) and exc.args else str(exc)
        return ToolError(text or type(exc).__name__)

    expected = (ValueError, LookupError, OSError, MemoryError)
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return _render(await fn(*args, **kwargs))
            except ToolError:
                raise
            except expected as exc:
                raise explain(exc) from exc

        return awrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return _render(fn(*args, **kwargs))
        except ToolError:
            raise
        except expected as exc:
            raise explain(exc) from exc

    return wrapper


def summarize_run(session: Session, info: RunInfo) -> dict:
    """Run result for the model: outcome, convergence numbers, thinned history (<= 12 its: all)."""
    hist = info.history
    keep = [r for i, r in enumerate(hist) if i == 0 or r.it % 10 == 0 or i == len(hist) - 1]
    if len(hist) <= 12:  # short runs: every iteration
        keep = hist
    out: dict[str, Any] = {
        "run_id": info.id,
        "project_id": info.project_id,
        "status": info.status,
        **session.run_outcome(info.id),
        "iterations": len(hist),
        "history": [_dump(r) for r in keep],
    }
    if hist:
        out["compliance"] = {"first": hist[0].compliance, "last": hist[-1].compliance}
        out["volume"] = hist[-1].volume
        out["change"] = hist[-1].change
        out["stress_max"] = hist[-1].stress_max  # von Mises of the last evaluated design
        out["constraint"] = hist[
            -1
        ].constraint  # stress constraint g (<= 0 satisfied), null if none
    if info.stats:
        s = info.stats
        out["grid"] = {"shape": [s.nx, s.ny, s.nz], "h": s.h, "n_active": s.n_active}
        out["warnings"] = s.warnings
    if info.error:
        out["error"] = info.error
    if info.status == "done":
        out["next"] = (
            "result_preview(run_id) to look at it; result_stress_summary(run_id) for the stress "
            "field; export_stl(run_id, path) to save it."
        )
    return _round(out)


def create_server(session: Session | None = None) -> MCPServer:
    """The MCP server; one `Session` (default store, `TOPOP_DATA_DIR` respected) per process."""
    session = session or Session()
    server = MCPServer("top-op", instructions=INSTRUCTIONS)

    def tool(extra: str = ""):
        def deco(fn: Callable) -> Callable:
            doc = inspect.cleandoc(fn.__doc__ or "")
            server.add_tool(
                _guard(fn),
                name=fn.__name__,
                description=f"{doc}\n\n{extra}" if extra else doc,
                structured_output=False,
            )
            return fn

        return deco

    # ---- meshes ---------------------------------------------------------------------------------

    @tool()
    def load_mesh(path: str) -> dict:
        """Load an STL/OBJ/3MF/PLY file, or a STEP file (.step/.stp, needs the optional STEP extra), from
        disk. Returns its mesh_id plus faces, bbox, volume and whether it is watertight (a
        non-watertight design voxelizes poorly); STEP meshes also report source "step" and
        n_brep_faces. The id is a content hash, so loading the same file again returns the same id.
        Units are whatever the file is in (STEP: as OpenCascade reports them, usually mm)."""
        return _round(_dump(session.load_mesh(path)))

    @tool()
    def describe_mesh(
        mesh_id: str, angle_deg: float = 5.0, top: int = 30, project_id: str | None = None
    ) -> dict:
        """Mesh info plus the facet table that lets you name faces without seeing them. Each row has
        id, n_faces, area, kind, unit normal, centroid and bbox [[min],[max]]. kind is "plane" (a group
        of coplanar triangles, neighbours within angle_deg of each other; has a normal), "cylinder"
        (a hole, boss or round fillet: has `axis` (unit vector) and `radius`; normal [0,0,0]) or
        "other" (curved, normal [0,0,0]). To select a hole, take the cylinder facet with the right
        radius and axis. Ids are ranks by area, largest first, stable for a given angle_deg. STEP
        meshes list their exact B-rep faces instead (angle_deg is ignored, each row also has
        `brep_face`, the face index in the file, and ids never renumber). `top` rows are returned
        (0 = all). Select a facet with {"kind":"facets","facet_ids":[id],"angle_deg":<same>}. Combine
        with preview_mesh to check what you picked.
        FRAME: selections (`direction`, `within`, plane `point`, primitives) are WORLD coordinates,
        the mesh as the project places it (design_transform / reference-model transform applied).
        Without project_id the geometry here is the raw file's ("frame": "mesh"); pass project_id
        to get it in world coordinates ("frame": "world"; mesh_id may then also be "design" or
        "ref:<id>"). Use the world table whenever the project has a design_transform. Facet ids are
        the same in both frames."""
        data = session.describe_mesh(mesh_id, angle_deg, top, project_id)
        optional = ("axis", "radius", "brep_face")
        data["facets"] = [
            {k: v for k, v in f.items() if k not in optional or v is not None}
            for f in data["facets"]
        ]
        return _round(data)

    @tool()
    def facet_faces(mesh_id: str, facet_id: int, angle_deg: float = 5.0) -> dict:
        """Triangle ids (the mesh's face indices, as drawn by preview_mesh and used by the raw
        {"kind":"faces"} selection) that make up one facet of describe_mesh. facet_id is the `id`
        column; use the same angle_deg as for describe_mesh (STEP meshes ignore it: the facet is the
        B-rep face). Returns n_faces and face_ids. You rarely need this: selecting
        {"kind":"facets","facet_ids":[facet_id]} does the same server-side. Unknown ids raise."""
        faces = session.facet_faces(mesh_id, facet_id, angle_deg)
        return {
            "mesh_id": mesh_id,
            "facet_id": facet_id,
            "angle_deg": angle_deg,
            "n_faces": len(faces),
            "face_ids": faces,
        }

    @tool()
    def preview_mesh(mesh_id: str, view: str = "iso") -> Image:
        """PNG render of a mesh so you can look at it. view: iso, +x, -x, +y, -y, +z, -z. The image
        shows the bbox frame and an X(red) Y(green) Z(blue) axis triad; the bbox size and min corner
        are printed in the top-left corner."""
        return Image(data=session.preview_mesh(mesh_id, view), format="png")

    # ---- projects -------------------------------------------------------------------------------

    @tool()
    def create_project(
        name: str,
        design_mesh_id: str,
        elements_along_longest: int = 60,
        volfrac: float = 0.3,
        penal: float = 3.0,
        rmin: float = 2.0,
        max_iter: int = 100,
        E: float = 1.0,
        nu: float = 0.3,
        design_transform: list[float] | None = None,
    ) -> dict:
        """Create a project around a loaded design mesh. The design volume is voxelized into cubes;
        elements_along_longest sets the resolution (4..600; 30-40 to debug, 60-100 for results; keep
        n_active <= ~150k on 16 GB, see voxel_stats). volfrac = target material fraction of the free
        volume, penal = SIMP penalty (3), rmin = density-filter radius in VOXELS (>= 1; 1.5-3; features
        thinner than about 2*rmin voxels cannot appear), max_iter, E and nu (Young's modulus,
        Poisson ratio; E only scales compliance). design_transform (optional, 16 floats column-major,
        default identity) moves/rotates/scales the mesh. Returns the project summary; add supports
        and loads next."""
        spec = ProjectIn.model_validate(
            {
                "name": name,
                "design_mesh": MeshRef(
                    mesh_id=design_mesh_id, transform=design_transform or list(IDENTITY)
                ),
                "grid": {"elements_along_longest": elements_along_longest},
                "material": {"E": E, "nu": nu},
                "params": {
                    "volfrac": volfrac,
                    "penal": penal,
                    "rmin": rmin,
                    "max_iter": max_iter,
                },
            }
        )
        return _brief(session.create_project(spec))

    @tool()
    def get_project(project_id: str) -> dict:
        """The full project document: design mesh, reference models, grid, material, params and every
        load/support with its selection JSON."""
        return _dump(session.get_project(project_id))

    @tool()
    def list_projects() -> dict:
        """Projects in the store (shared with `topop serve`): id, name, counts. Use to resume work."""
        return {
            "projects": [
                {
                    "project_id": p.id,
                    "name": p.name,
                    "updated_at": p.updated_at,
                    "n_loads": len(p.loads),
                    "n_supports": len(p.supports),
                    "n_ref_models": len(p.ref_models),
                }
                for p in session.list_projects()
            ]
        }

    @tool()
    def set_params(
        project_id: str,
        volfrac: float | None = None,
        penal: float | None = None,
        rmin: float | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        move: float | None = None,
        heaviside: bool | None = None,
        continuation: bool | None = None,
        solver: Literal["auto", "amg", "direct"] | None = None,
        dtype: Literal["float64", "float32"] | None = None,
        optimizer: Literal["oc", "mma"] | None = None,
        symmetry: list[SymmetrySpec] | None = None,
        stress_limit: float | None = None,
        stress_pnorm: float | None = None,
        overhang: Literal["+x", "-x", "+y", "-y", "+z", "-z", "none"] | None = None,
    ) -> dict:
        """Change optimizer parameters; omitted ones keep their value. volfrac (0..1) target volume
        fraction of free elements; penal (1..6) SIMP penalty; rmin >= 1 filter radius in voxels;
        max_iter (1..2000); tol = stop when the largest density change per iteration is below it
        (0.01); move = OC move limit (0.2); heaviside = projection for crisper edges; continuation
        = ramp penal 1 -> penal over 20 iterations (helps avoid local minima); solver auto|amg|
        direct; dtype float32 halves memory.
        optimizer: "oc" (default, volume constraint only, fastest) or "mma" (handles the stress
        constraint; slower per iteration).
        symmetry: mirror planes, e.g. [{"axis":"y","position":null}]; position is a world coordinate
        (null = centre of the design's bbox, snapped to the nearest voxel boundary or centre). The
        design variables are tied across each plane, so the result is exactly mirror-symmetric. Use it
        only when loads, supports and domain are symmetric too (otherwise the design is a compromise
        and run warns); pass [] to remove all planes.
        stress_limit: von Mises limit in the units of E (pass 0 to remove it). Sets a p-norm
        aggregated stress constraint on the design and forces optimizer mma; the design gets heavier
        and stiffer where stress concentrates. Check it afterwards: run reports `stress_max` and
        `constraint` (g <= 0 means satisfied, within 0.01) and result_stress_summary gives the field's
        max and where it is. Stresses are voxel stresses: sharp corners overshoot, so give some margin.
        stress_pnorm (4..256, default 64): FINAL exponent of the p-norm continuation, which starts
        at min(8, stress_pnorm) and doubles up to it as the run settles; while the constraint is
        active the move limit is capped at 0.1 (0.05 near the limit). Rarely needs changing.
        overhang: additive-manufacturing build direction ("+z" = printed upward): a 45-degree
        overhang filter keeps every material voxel supported from below. The base plate is the
        domain's min face along that axis for "+x/+y/+z" and its max face for "-x/-y/-z" (the part
        must be able to grow from there; no support structures are generated, so keep the supports
        or loads sensible); pass "none" to remove it. voxel_stats and run echo these as warnings.
        Returns the new params."""
        clear = [
            name
            for name, off in (("stress_limit", stress_limit == 0), ("overhang", overhang == "none"))
            if off
        ]
        p = session.set_params(
            project_id,
            clear=clear,
            stress_limit=None if stress_limit == 0 else stress_limit,
            stress_pnorm=stress_pnorm,
            overhang=None if overhang == "none" else overhang,
            optimizer=optimizer,
            symmetry=None if symmetry is None else [s.model_dump() for s in symmetry],
            volfrac=volfrac,
            penal=penal,
            rmin=rmin,
            max_iter=max_iter,
            tol=tol,
            move=move,
            heaviside=heaviside,
            continuation=continuation,
            solver=solver,
            dtype=dtype,
        )
        return {"project_id": p.id, "params": _dump(p.params)}

    @tool()
    def set_grid(
        project_id: str, elements_along_longest: int | None = None, padding: int | None = None
    ) -> dict:
        """Change the voxel resolution: elements_along_longest (4..600) cubes along the longest
        bbox side (h = side / that), padding = empty voxels around the model (default 1). Selections
        that use `within` boxes must be re-checked afterwards (h changed): see voxel_stats.
        Cost grows with the cube of the resolution; check n_active, est_bytes and est_sec_per_iter
        in voxel_stats (<= ~150k active elements on 16 GB)."""
        p = session.set_grid(
            project_id, elements_along_longest=elements_along_longest, padding=padding
        )
        return {"project_id": p.id, "grid": _dump(p.grid)}

    @tool()
    def set_material(project_id: str, E: float | None = None, nu: float | None = None) -> dict:
        """Isotropic material: Young's modulus E (> 0, in the unit system of your forces and
        lengths; it only scales compliance) and Poisson ratio nu (0..0.5)."""
        p = session.set_material(project_id, E=E, nu=nu)
        return {"project_id": p.id, "material": _dump(p.material)}

    @tool(SELECTION_HELP)
    def add_load(
        project_id: str, selection: dict, force: list[float], case: int = 0, name: str = ""
    ) -> dict:
        """Apply a force to the surface nodes a selection resolves to. force is the TOTAL force
        vector [fx,fy,fz] in your unit system, split equally over the selected nodes. Loads with
        the same `case` act together; compliance is summed over cases (use different cases for load
        scenarios that never occur at the same time). Returns the stored load (with its id) and what
        the selection resolved to."""
        sel = parse_selection(selection)
        stored = session.add_load(
            project_id,
            LoadSpec(id="", name=name, selection=sel, force=force, case=case),
        )
        return {
            "load": _dump(stored),
            "resolved": _resolved(project_id, stored.selection),
        }

    @tool(SELECTION_HELP)
    def add_support(
        project_id: str, selection: dict, fix: list[bool] | None = None, name: str = ""
    ) -> dict:
        """Fix the surface nodes a selection resolves to. fix = [x,y,z] flags for the constrained
        translations (default [true,true,true] = fully clamped; [false,false,true] = only z is held,
        a roller). A project needs at least one support and one load, and the supports must stop all
        rigid-body motion."""
        sel = parse_selection(selection)
        stored = session.add_support(
            project_id,
            SupportSpec(id="", name=name, selection=sel, fix=fix or [True, True, True]),
        )
        return {
            "support": _dump(stored),
            "resolved": _resolved(project_id, stored.selection),
        }

    def _resolved(project_id: str, selection: Any) -> dict:
        try:
            out = _round(session.resolve(project_id, selection))
        except (ValueError, LookupError) as exc:
            return {"error": str(exc.args[0]) if isinstance(exc, LookupError) else str(exc)}
        if out["count"] == 0:
            out["hint"] = (
                "selects 0 nodes: check the coordinates, and pad `within` boxes by one voxel h"
            )
        return out

    @tool()
    def add_ref_model(
        project_id: str,
        mesh_id: str,
        mode: Literal["keep_in", "keep_out"],
        transform: list[float] | None = None,
        name: str = "",
    ) -> dict:
        """Add a reference body (a loaded mesh) as a passive region. keep_in = forced solid (e.g. a
        bolt boss that must exist; it also extends the domain); keep_out = forced void (clearance
        for another part; it never extends the domain). transform = 16 floats COLUMN-MAJOR (three.js
        Matrix4.toArray(): translation in elements 12,13,14, e.g. [1,0,0,0, 0,1,0,0, 0,0,1,0, tx,ty,tz,1];
        a uniform scale s puts s,s,s on the diagonal); default identity. Reference bodies are
        voxelized like the design, so thin ones need a fine enough grid; voxel_stats warns when one
        covers/removes nothing. Returns the reference model with its id."""
        stored = session.add_ref_model(
            project_id,
            RefModel(
                id="", name=name, mesh_id=mesh_id, mode=mode, transform=transform or list(IDENTITY)
            ),
        )
        return {
            "ref_model": _dump(stored),
            "selection_mesh_id": f"ref:{stored.id}",
        }

    @tool()
    def remove_load(project_id: str, load_id: str) -> dict:
        """Remove a load by id (ids are in the add_load result and in get_project)."""
        return _brief(session.remove_load(project_id, load_id))

    @tool()
    def remove_support(project_id: str, support_id: str) -> dict:
        """Remove a support by id (ids are in the add_support result and in get_project)."""
        return _brief(session.remove_support(project_id, support_id))

    @tool()
    def remove_ref_model(project_id: str, ref_id: str) -> dict:
        """Remove a reference model by id."""
        return _brief(session.remove_ref_model(project_id, ref_id))

    # ---- checking -------------------------------------------------------------------------------

    @tool()
    def voxel_stats(project_id: str) -> dict:
        """Voxelize the project (cached) and report: grid shape nx,ny,nz and voxel size h, n_active
        elements (free / passive solid / passive void), nodes, dofs, estimated memory (bytes) and
        seconds per iteration, plus warnings (non-watertight mesh, disconnected parts, too many
        elements, loads/supports that resolve to nothing, and notes on parameters such as the
        mma optimizer being forced by a stress_limit or where the overhang base plate is). `boundaries` lists, per load and support,
        the number of grid nodes it resolved to and their bbox: check these before running."""
        stats = session.voxel_stats(project_id)
        return _round({**_dump(stats), "boundaries": session.boundaries(project_id)})

    @tool(SELECTION_HELP)
    def resolve_selection(project_id: str, selection: dict) -> dict:
        """Dry-run a selection on the project's grid without storing it: returns the number of
        grid nodes it picks, their bbox, centroid, a few sample coordinates and the voxel size h."""
        return _round(session.resolve(project_id, selection))

    # ---- running --------------------------------------------------------------------------------

    @tool()
    async def run(project_id: str, max_iter: int | None = None, ctx: Context | None = None) -> dict:
        """Run the optimization and wait for it (minutes for large grids; progress notifications are
        sent per iteration). max_iter overrides params.max_iter for this run only (e.g. 4 for a quick
        smoke test). Returns status (done | cancelled | error), outcome (converged | max_iter),
        compliance first/last (should fall and flatten), volume (should sit at volfrac), change (largest
        density change of the last iteration; converged when below tol), stress_max (max von Mises of
        the last evaluated design, units of E) and constraint (the stress constraint value when a
        stress_limit is set: <= 0 satisfied, null otherwise; a run that ends with constraint > 0.01
        needs more iterations or a higher limit), wall seconds, and the history every 10th iteration
        plus the last (with stress_max and constraint per iteration). With symmetry/overhang/
        stress_limit set the run is slower and may need 100+ iterations to settle. Raises with the issue list if the project is not
        runnable (no loads/supports, a selection that resolves to 0 nodes, ...)."""
        cancel = threading.Event()
        total = max_iter or session.get_project(project_id).params.max_iter

        def progress(r) -> None:
            if ctx is None:
                return
            msg = f"it {r.it}: compliance {r.compliance:.4g}, volume {r.volume:.3f}"
            if r.stress_max is not None:
                msg += f", stress_max {r.stress_max:.4g}"
            # best effort: never fail a run because a notification could not be sent
            with contextlib.suppress(Exception):
                anyio.from_thread.run(ctx.report_progress, r.it, total, msg)

        def work() -> RunInfo:
            return session.run(project_id, progress, cancel, max_iter)

        try:
            info = await anyio.to_thread.run_sync(work, abandon_on_cancel=True)
        except anyio.get_cancelled_exc_class():
            cancel.set()  # the client gave up: stop after the current iteration
            raise
        return summarize_run(session, info)

    @tool()
    def get_run(run_id: str) -> dict:
        """Status and history summary of a run (also works while another call is still running it)."""
        return summarize_run(session, session.get_run(run_id))

    @tool()
    def cancel_run(run_id: str) -> dict:
        """Ask a running optimization to stop after its current iteration; the result so far is kept."""
        return summarize_run(session, session.cancel(run_id))

    # ---- results --------------------------------------------------------------------------------

    @tool()
    def result_preview(
        run_id: str, threshold: float = 0.5, view: str = "iso", trim: bool = False
    ) -> list[Image | str]:
        """PNG render of a finished run: the optimized material in ORANGE over the original design
        ghosted in light grey. threshold (0..1) is the density cut (0.5 default; raise it to see
        only the dense core). view: iso, +x, -x, +y, -y, +z, -z. Look from several views to judge
        the load paths. trim=true shows the result clipped to the design surface, as
        export_stl(trim=true) would write it."""
        png, warnings = session.render_result(run_id, threshold, view, trim)
        caption = (
            f"run {run_id}, view {view}, density threshold {threshold}: orange = optimized material, "
            "grey = original design"
        )
        notes = [f"warning: {w}" for w in warnings]
        return [Image(data=png, format="png"), "\n".join([caption, *notes])]

    @tool()
    def export_stl(
        run_id: str,
        path: str,
        threshold: float = 0.5,
        smooth: int = 0,
        trim: bool = False,
        overwrite: bool = False,
    ) -> dict:
        """Write the result as a binary STL (marching-cubes iso-surface of the density at
        threshold; smooth = Laplacian smoothing iterations, 0-10, more shrinks thin members).
        trim=true intersects the surface with the design mesh (a boolean in the design's world
        space): the part never pokes outside the CAD surface and keeps its exact faces (flat
        mounting faces, hole walls) wherever material reaches them, instead of the stair-stepped
        voxel skin. It needs a watertight design mesh; if that or the boolean fails, the untrimmed
        STL is written and the reason is returned under "warnings".
        path: ABSOLUTE (this server's working directory is not yours), ending in .stl; parent
        directories are created. An existing file is only replaced with overwrite=true (and only
        a regular .stl file). Returns path, byte size and triangle count."""
        out = _output_path(path, ".stl", overwrite)
        data, warnings = session.export_stl(run_id, threshold, smooth, trim)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        result = {"path": str(out), "bytes": len(data), "triangles": (len(data) - 84) // 50}
        return {**result, "warnings": warnings} if warnings else result

    @tool()
    def export_files(
        run_id: str,
        directory: str,
        threshold: float = 0.5,
        smooth: int = 0,
        trim: bool = False,
        overwrite: bool = False,
    ) -> dict:
        """Write everything into a directory: result.stl, result.png (iso), result.vti (ParaView
        density + passive mask + von Mises "stress" cell data), density.npz (numpy arrays, with a
        "stress" key) and run.json (project + run record; `topop run run.json` re-runs it
        headlessly). trim=true clips result.stl (and the png) to the design surface, see
        export_stl. directory: ABSOLUTE (this server's working directory is not yours), created
        if missing; if any of those five files already exists nothing is written unless
        overwrite=true. Returns the written paths and, under "errors", any file that could not be
        made and, under "warnings", why a trim was skipped."""
        out = _absolute(directory)
        if out.exists() and not out.is_dir():
            raise ValueError(f"{out} exists and is not a directory")
        for name in OUTPUT_FILES:
            _output_path(str(out / name), Path(name).suffix, overwrite)
        return session.write_outputs(run_id, out, threshold, smooth, trim)

    @tool()
    def result_stress_summary(run_id: str) -> dict:
        """Von Mises stress of a finished run's final design (units of E): `max`, `mean_solid` (mean
        over solid voxels, density >= 0.5), `location` = [x, y, z] of the element centre holding the
        max (world coordinates, mesh units) with its grid `cell`, and, when the project has a
        stress_limit, `stress_limit` and `max_over_limit`. The field is the density-weighted voxel
        stress, so grey voxels count less and sharp inner corners overshoot; use it to see whether
        a stress_limit held and where the hot spot is, then look at that place with result_preview.
        The full field is in result.vti ("stress") and density.npz."""
        return session.stress_summary(run_id)

    @tool()
    def generate_struts(
        run_id: str,
        mode: Literal["layout", "skeleton"] = "layout",
        sigma_allow: float = 20.0,
        target_volume: float | None = None,
        node_spacing: float | None = None,
        min_radius: float | None = None,
        max_bar_length: float | None = None,
        sample: Literal["solid", "active"] = "solid",
        path: str | None = None,
        overwrite: bool = False,
    ) -> dict:
        """Turn a finished run into an explicit strut (truss) structure: round bars with spherical
        joints, unioned with the keep-in bodies, clipped to the design, then voxelized and FE-solved.
        mode "layout" (default): minimum-volume truss over a ground structure of nodes at the
        loads, supports, keep-in boundaries and sampled from the SIMP solid (sample "active": the
        whole domain), every node pair within max_bar_length (default 0.4 x domain diagonal),
        node_spacing default 4 voxels (coarsened automatically while min-radius bars would exceed
        the volume). mode "skeleton": the medial axis of the SIMP solid, for results that already
        look like beams. Radii are scaled to target_volume (default: the SIMP material volume;
        <= 0 keeps the sigma_allow sizing), never below min_radius (default max(1, 0.8 h)).
        Returns n_bars, radii, volume vs simp_volume, compliance per load case vs simp_compliance
        (compliance_ratio < 1 = stiffer than SIMP), stress_max, watertight, n_bodies, warnings, and
        `files` (struts.stl / .json / .png in the data dir; GET /api/runs/{id}/struts.stl serves
        the same STL). path: optional ABSOLUTE .stl to also write the mesh to (overwrite=true to
        replace)."""
        out = _output_path(path, ".stl", overwrite) if path else None
        data = session.generate_struts(
            run_id,
            mode=mode,
            sigma_allow=sigma_allow,
            target_volume=target_volume,
            node_spacing=node_spacing,
            min_radius=min_radius,
            max_bar_length=max_bar_length,
            sample=sample,
        )
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(Path(data["files"]["stl"]).read_bytes())
            data["path"] = str(out)
        return data

    @tool()
    def export_case(project_id: str, path: str, overwrite: bool = False) -> dict:
        """Save the project as a case file (JSON) that `topop run` and load_case read. Mesh files
        are referenced by path relative to the case file when their source file is known.
        path: ABSOLUTE (this server's working directory is not yours), ending in .json; an
        existing file is only replaced with overwrite=true (and only a regular .json file)."""
        out = _output_path(path, ".json", overwrite)
        return {"path": session.save_case(project_id, out)}

    @tool()
    def load_case(path: str) -> dict:
        """Load a case file (a ProjectIn JSON, or a run.json) as a new project. Mesh `path`s are
        relative to the case file and are uploaded; selection shorthands are accepted. Returns the
        project summary."""
        return _brief(session.load_case(path))

    return server


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
