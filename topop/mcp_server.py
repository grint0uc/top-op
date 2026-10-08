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
)

INSTRUCTIONS = """\
top-op: 3D topology optimization (voxel SIMP compliance minimisation) of a mesh you name regions on.
Loop: load_mesh -> describe_mesh (+ preview_mesh) -> create_project -> add_support / add_load
(select regions by facet id, face normal, plane or primitive) -> voxel_stats (node counts, warnings,
size) -> run (try a coarse grid or few iterations first) -> result_preview -> adjust -> export_stl.
Units: whatever the mesh is in; E, forces and lengths must be consistent (compliance is in those units).
Size: keep active elements <= ~150k on a 16 GB machine (voxel_stats shows n_active, memory and time
estimates); 30-40 elements along the longest side to debug a setup, 60-100 for a real result.
Always check that a load/support resolved to a sensible node count before running.
"""

SELECTION_HELP = """\
SELECTION: a JSON object; "kind" picks the shape. Coordinates are in the mesh's own units.
 facets   {"kind":"facets","facet_ids":[0,3]}   coplanar groups from describe_mesh (same angle_deg, default 5)
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
    if info.stats:
        s = info.stats
        out["grid"] = {"shape": [s.nx, s.ny, s.nz], "h": s.h, "n_active": s.n_active}
        out["warnings"] = s.warnings
    if info.error:
        out["error"] = info.error
    if info.status == "done":
        out["next"] = "result_preview(run_id) to look at it; export_stl(run_id, path) to save it."
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
        """Load an STL/OBJ/3MF/PLY file from disk. Returns its mesh_id plus faces, bbox, volume and
        whether it is watertight (a non-watertight design voxelizes poorly). The id is a content hash,
        so loading the same file again returns the same id. Units are whatever the file is in."""
        return _round(_dump(session.load_mesh(path)))

    @tool()
    def describe_mesh(mesh_id: str, angle_deg: float = 5.0, top: int = 30) -> dict:
        """Mesh info plus the facet table that lets you name faces without seeing them. A facet is
        a group of coplanar triangles (neighbours within angle_deg of each other); ids are ranks by
        area, largest first, stable for a given angle_deg. Each row has id, n_faces, area, unit
        normal (a closed curved group has normal [0,0,0]), centroid and bbox [[min],[max]]. `top` rows
        are returned (0 = all). Select a facet with {"kind":"facets","facet_ids":[id],"angle_deg":<same>}.
        Combine with preview_mesh to check what you picked."""
        return _round(session.describe_mesh(mesh_id, angle_deg, top))

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
    ) -> dict:
        """Change optimizer parameters; omitted ones keep their value. volfrac (0..1) target volume
        fraction of free elements; penal (1..6) SIMP penalty; rmin >= 1 filter radius in voxels;
        max_iter (1..2000); tol = stop when the largest density change per iteration is below it
        (0.01); move = OC move limit (0.2); heaviside = projection for crisper edges; continuation
        = ramp penal 1 -> penal over 20 iterations (helps avoid local minima); solver auto|amg|
        direct; dtype float32 halves memory. Returns the new params."""
        p = session.set_params(
            project_id,
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
        elements, loads/supports that resolve to nothing). `boundaries` lists, per load and support,
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
        density change of the last iteration; converged when below tol), wall seconds, and the history
        every 10th iteration plus the last. Raises with the issue list if the project is not
        runnable (no loads/supports, a selection that resolves to 0 nodes, ...)."""
        cancel = threading.Event()
        total = max_iter or session.get_project(project_id).params.max_iter

        def progress(r) -> None:
            if ctx is None:
                return
            msg = f"it {r.it}: compliance {r.compliance:.4g}, volume {r.volume:.3f}"
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
    def result_preview(run_id: str, threshold: float = 0.5, view: str = "iso") -> list[Image | str]:
        """PNG render of a finished run: the optimized material in ORANGE over the original design
        ghosted in light grey. threshold (0..1) is the density cut (0.5 default; raise it to see
        only the dense core). view: iso, +x, -x, +y, -y, +z, -z. Look from several views to judge
        the load paths."""
        png = session.result_png(run_id, threshold, view)
        caption = (
            f"run {run_id}, view {view}, density threshold {threshold}: orange = optimized material, "
            "grey = original design"
        )
        return [Image(data=png, format="png"), caption]

    @tool()
    def export_stl(run_id: str, path: str, threshold: float = 0.5, smooth: int = 0) -> dict:
        """Write the result as a binary STL (marching-cubes iso-surface of the density at
        threshold; smooth = Laplacian smoothing iterations, 0-10, more shrinks thin members).
        Parent directories are created. Returns path, byte size and triangle count."""
        data = session.result_stl(run_id, threshold, smooth)
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return {"path": str(out), "bytes": len(data), "triangles": (len(data) - 84) // 50}

    @tool()
    def export_files(run_id: str, directory: str, threshold: float = 0.5, smooth: int = 0) -> dict:
        """Write everything into a directory: result.stl, result.png (iso), result.vti (ParaView
        density + passive mask), density.npz (numpy arrays) and run.json (project + run record;
        `topop run run.json` re-runs it headlessly). Returns the written paths and, under
        "errors", any file that could not be made."""
        return session.write_outputs(run_id, directory, threshold, smooth)

    @tool()
    def export_case(project_id: str, path: str) -> dict:
        """Save the project as a case file (JSON) that `topop run` and load_case read. Mesh files
        are referenced by path relative to the case file when their source file is known."""
        return {"path": session.save_case(project_id, path)}

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
