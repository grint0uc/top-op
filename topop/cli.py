from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

EXIT_INVALID = 2
EXIT_MEMORY = 3
EXIT_INTERRUPTED = 130


LOOPBACK = ("localhost", "127.0.0.1", "::1", "[::1]")
WILDCARD = ("0.0.0.0", "::", "[::]", "")


def _allow_host(host: str) -> None:
    """Let the app answer requests addressed to `--host` (it only answers localhost by default)."""
    if host in LOOPBACK:
        return
    if host in WILDCARD:
        print(
            f"note: listening on every interface, but only requests addressed to localhost are "
            f"answered; list the names clients use in TOPOP_ALLOWED_HOSTS (now "
            f"{os.environ.get('TOPOP_ALLOWED_HOSTS') or 'unset'})",
            file=sys.stderr,
        )
        return
    name = f"[{host}]" if ":" in host else host
    extra = os.environ.get("TOPOP_ALLOWED_HOSTS", "")
    os.environ["TOPOP_ALLOWED_HOSTS"] = f"{extra},{name}" if extra else name


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    _allow_host(args.host)  # before the app module reads the allowed hosts
    if args.dev:
        print(
            "dev mode: run `cd web && npm run dev` (Vite proxies /api to this server)",
            file=sys.stderr,
        )
        uvicorn.run(
            "topop.server.app:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=["topop"],
        )
    else:
        from topop.server.app import app

        uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _err(command: str, message: str) -> None:
    print(f"topop {command}: {message}", file=sys.stderr)


def _explain(exc: BaseException) -> str:
    from topop.agent import explain_validation_error

    return explain_validation_error(exc) if isinstance(exc, ValidationError) else str(exc)


# ---- describe -----------------------------------------------------------------------------------


def _vec(v, fmt: str) -> str:
    return "(" + ", ".join(format(x, fmt) for x in v) + ")"


def format_description(data: dict) -> str:
    """MeshInfo block + fixed-width facet table.

    Planes show their normal; cylinders their radius and axis (a hole is a cylinder facet); STEP
    meshes also show the B-rep face index. Cells that do not apply stay blank.
    """
    m = data["mesh"]
    lo, hi = m["bbox"]
    size = [b - a for a, b in zip(lo, hi, strict=True)]
    volume = "n/a (not watertight)" if m["volume"] is None else f"{m['volume']:.6g}"
    what = (
        "B-rep faces"
        if m.get("source") == "step"
        else f"coplanar groups and cylinders, angle {data['angle_deg']:g} deg"
    )
    lines = [
        f"mesh {m['name']}  id {m['id']}",
        (
            f"  faces {m['n_faces']}  vertices {m['n_vertices']}  watertight {m['is_watertight']}"
            f"  volume {volume}"
        ),
        f"  bbox min {_vec(lo, '.6g')}  max {_vec(hi, '.6g')}  size {_vec(size, '.6g')}",
        f"facets ({what}, by area): showing {len(data['facets'])} of {data['n_facets_total']}",
        (
            f"{'id':>4} {'faces':>6} {'area':>11}  {'normal':<24}  {'kind':<8}  {'radius':>8}  "
            f"{'axis':<24}  {'brep':>4}  {'centroid':<30}  bbox min .. max"
        ),
    ]
    for f in data["facets"]:
        kind = f.get("kind", "other")
        normal = _vec(f["normal"], "+.3f") if any(f["normal"]) else ""
        radius = "" if f.get("radius") is None else format(f["radius"], ".5g")
        axis = "" if kind == "plane" or not f.get("axis") else _vec(f["axis"], "+.3f")
        brep = "" if f.get("brep_face") is None else str(f["brep_face"])
        lines.append(
            f"{f['id']:>4} {f['n_faces']:>6} {f['area']:>11.5g}  {normal:<24}  {kind:<8}  "
            f"{radius:>8}  {axis:<24}  {brep:>4}  {_vec(f['centroid'], '.5g'):<30}  "
            f"{_vec(f['bbox'][0], '.5g')} .. {_vec(f['bbox'][1], '.5g')}"
        )
    return "\n".join(lines)


def _describe(args: argparse.Namespace) -> int:
    from topop.agent import Session
    from topop.server.store import NotFoundError

    try:
        session = Session()
        info = session.load_mesh(args.mesh)
        data = session.describe_mesh(info.id, args.angle, args.top)
        png = session.preview_mesh(info.id, args.view) if args.png else None
    except (ValueError, OSError, NotFoundError) as exc:
        _err("describe", _explain(exc))
        return EXIT_INVALID
    print(format_description(data))
    if args.png and png is not None:
        Path(args.png).expanduser().write_bytes(png)
        print(f"preview ({args.view}) written to {args.png}")
    return 0


# ---- run ----------------------------------------------------------------------------------------


def _print_setup(case: str, project, stats, bounds: dict) -> None:
    mb = stats.est_bytes / 1e6
    print(f"case {project.name}  ({case})")
    print(
        f"grid {stats.nx}x{stats.ny}x{stats.nz}  h {stats.h:.4g}  active {stats.n_active:,} "
        f"(free {stats.n_free:,}, solid {stats.n_passive_solid:,}, void {stats.n_passive_void:,})"
        f"  nodes {stats.n_nodes:,}  est {mb:,.0f} MB, {stats.est_sec_per_iter:.2f} s/iter"
    )
    for ld in bounds["loads"]:
        print(
            f"load    {ld['name'] or ld['id']}: force {_vec(ld['force'], 'g')} case {ld['case']}"
            f" -> {ld['n_nodes']} nodes"
        )
    for sp in bounds["supports"]:
        axes = "".join(a for a, f in zip("xyz", sp["fix"], strict=True) if f) or "none"
        print(f"support {sp['name'] or sp['id']}: fixes {axes} -> {sp['n_nodes']} nodes")
    extras = _feature_summary(project.params)
    if extras:
        print(f"features    {extras}")
    for w in stats.warnings:
        print(f"warning: {w}")


def _feature_summary(params) -> str:
    """Non-default v0.2 options of the project, for the setup block."""
    parts = []
    if params.optimizer == "mma" or params.stress_limit is not None:  # a limit forces mma
        parts.append("optimizer mma")
    if params.symmetry:
        planes = ", ".join(
            s.axis if s.position is None else f"{s.axis}={s.position:g}" for s in params.symmetry
        )
        parts.append(f"symmetry {planes}")
    if params.stress_limit is not None:
        from topop.core.optimize import STRESS_P_START

        # stress_pnorm is the final exponent of the p-continuation
        p_end = params.stress_pnorm
        p_start = min(STRESS_P_START, p_end)
        sched = f"{p_start:g}->{p_end:g}" if p_start < p_end else f"{p_end:g}"
        parts.append(f"stress_limit {params.stress_limit:g} (p {sched})")
    if params.overhang is not None:
        parts.append(f"overhang {params.overhang}")
    return "; ".join(parts)


def _install_sigint(cancel: threading.Event):
    def handler(signum, frame):
        cancel.set()
        print(
            "\ninterrupted: stopping after this iteration (Ctrl-C again to abort)", file=sys.stderr
        )
        signal.signal(signal.SIGINT, signal.default_int_handler)

    try:
        return signal.signal(signal.SIGINT, handler)
    except ValueError:  # not the main thread
        return None


def _run(args: argparse.Namespace) -> int:
    from topop.agent import ProjectInvalid, Session
    from topop.server.store import NotFoundError

    t0 = time.perf_counter()
    session = Session()
    try:
        project = session.load_case(args.case)
        pid = project.id
        if args.resolution is not None:
            session.set_grid(pid, elements_along_longest=args.resolution)
        overrides = _param_overrides(args)
        if overrides:
            session.set_params(pid, **overrides)
        stats = session.voxel_stats(pid)
        bounds = session.boundaries(pid)
        project = session.get_project(pid)
    except (ValueError, OSError, NotFoundError) as exc:
        _report_invalid(exc)
        return EXIT_INVALID
    with_constraint = project.params.stress_limit is not None
    if not args.quiet:
        _print_setup(args.case, project, stats, bounds)
        head = f"{'it':>4} {'compliance':>13} {'volume':>8} {'change':>8} {'t_iter':>8}"
        head += f" {'stress_max':>11}" + (f" {'constraint':>10}" if with_constraint else "")
        print(head, flush=True)

    def progress(r) -> None:
        if not args.quiet:
            line = f"{r.it:>4} {r.compliance:>13.6e} {r.volume:>8.4f} {r.change:>8.4f} {r.t_iter:>7.2f}s"
            line += f" {'-' if r.stress_max is None else format(r.stress_max, '11.4e'):>11}"
            if with_constraint:
                line += f" {'-' if r.constraint is None else format(r.constraint, '+10.4f'):>10}"
            print(line, flush=True)

    cancel = threading.Event()
    previous = _install_sigint(cancel)
    try:
        info = session.run(pid, progress, cancel, args.max_iter)
    except ProjectInvalid as exc:
        _report_invalid(exc)
        return EXIT_INVALID
    except MemoryError as exc:
        _err("run", f"out of memory: {exc}")
        return EXIT_MEMORY
    except Exception as exc:  # noqa: BLE001 - last-resort report, exit code 1
        _err("run", f"run failed: {_explain(exc)}")
        return 1
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)

    return _finish_run(session, info, args, time.perf_counter() - t0)


def _param_overrides(args: argparse.Namespace) -> dict:
    """`topop run` flags -> `ParamsSpec` fields (only the ones given)."""
    fields: dict = {}
    if args.optimizer is not None:
        fields["optimizer"] = args.optimizer
    if args.stress_limit is not None:
        fields["stress_limit"] = args.stress_limit
    if args.overhang is not None:
        fields["overhang"] = args.overhang
    if args.symmetry:
        fields["symmetry"] = args.symmetry
    return fields


def _symmetry_arg(text: str) -> dict:
    """`y` or `y=12.5` -> {"axis": "y", "position": None | 12.5}."""
    axis, sep, pos = text.partition("=")
    axis = axis.strip().lower()
    if axis not in ("x", "y", "z"):
        raise argparse.ArgumentTypeError(f"{text!r}: expected x, y or z, optionally =POSITION")
    try:
        position = float(pos) if sep else None
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r}: position {pos!r} is not a number") from None
    return {"axis": axis, "position": position}


def _threshold_arg(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0 < value <= 1:  # also rejects nan
        raise argparse.ArgumentTypeError(f"{text}: must be in (0, 1]")
    return value


def _smooth_arg(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text}: must be >= 0")
    return value


def _report_invalid(exc: BaseException) -> None:
    from topop.agent import ProjectInvalid

    if isinstance(exc, ProjectInvalid):
        _err("run", "the project is not runnable:")
        for issue in exc.issues:
            print(f"  - {issue}", file=sys.stderr)
    else:
        _err("run", f"invalid case: {_explain(exc)}")


def _finish_run(session, info, args: argparse.Namespace, wall: float) -> int:
    status = info.status
    outcome = session.run_outcome(info.id)
    if status == "error" or not info.history:
        _err("run", f"run {status}: {info.error or outcome.get('message') or 'no iterations ran'}")
        return 1
    out = Path(args.out) if args.out else Path.cwd() / f"{Path(args.case).stem}-out"
    try:
        result = session.write_outputs(info.id, out, args.threshold, args.smooth, args.trim)
    except OSError as exc:
        _err("run", f"could not write to {out}: {exc}")
        return 1
    except MemoryError as exc:
        _err("run", f"out of memory while writing the results: {exc}")
        return EXIT_MEMORY
    written = result["files"]
    for name, why in result["errors"].items():
        _err("run", f"{name} not written: {why}")
    for why in result["warnings"]:
        _err("run", f"warning: {why}")
    last = info.history[-1]
    detail = outcome.get("message") or status
    print(f"status      {status} ({detail})  run {info.id}")
    print(
        f"iterations  {len(info.history)}   compliance {last.compliance:.6e}   volume {last.volume:.4f}"
    )
    print(_stress_line(session, info, last))
    if args.trim and "result.stl" in written and not result["warnings"]:
        print("trim        result.stl intersected with the design mesh")
    print(f"wall time   {wall:.1f} s total, {outcome.get('wall_s', 0.0):.1f} s optimizing")
    print(f"files       {out}/ : {', '.join(written)}")
    print(f"rerun       topop run {Path(out) / 'run.json'}")
    if "result.stl" not in written:  # the one output a script cannot do without
        _err("run", "failed: result.stl was not written")
        return 1
    return EXIT_INTERRUPTED if status == "cancelled" else 0


def _stress_line(session, info, last) -> str:
    """Max von Mises of the final design and, with a stress limit, whether the constraint held."""
    from topop.core.optimize import STRESS_FEAS_TOL

    try:
        summary = session.stress_summary(info.id)
    except ValueError:
        summary = None
    peak = summary["max"] if summary else last.stress_max
    if peak is None:
        return "stress      not available"
    where = ""
    if summary:
        where = (
            f" at {_vec(summary['location'], '.4g')}, mean over solid {summary['mean_solid']:.4e}"
        )
    line = f"stress      max {peak:.4e} von Mises{where}"
    limit = session.get_project(info.project_id).params.stress_limit
    if limit is None:
        return line
    ok = last.constraint is not None and last.constraint <= STRESS_FEAS_TOL
    g = "n/a" if last.constraint is None else f"{last.constraint:+.4f}"
    verdict = "satisfied" if ok else "NOT satisfied"
    return f"{line}\nconstraint  stress <= {limit:g}: {verdict} (g = {g}, max/limit {peak / limit:.2f})"


# ---- struts -------------------------------------------------------------------------------------

STRUT_FILES = ("struts.stl", "struts.png", "struts.json")


def _run_files(target: str) -> tuple[Path, Path]:
    """RUN_DIR or RUN_DIR/run.json -> (run.json, density.npz)."""
    p = Path(target).expanduser()
    run_json = p / "run.json" if p.is_dir() else p
    npz = run_json.parent / "density.npz"
    for f in (run_json, npz):
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found (expected the output directory of `topop run`)")
    return run_json, npz


def _struts(args: argparse.Namespace) -> int:

    from topop.agent import ProjectInvalid, Session, pretty_json
    from topop.core.export import from_npz_bytes, to_stl_bytes
    from topop.server.routes_struts import strut_json, struts_png
    from topop.server.store import NotFoundError

    t0 = time.perf_counter()
    params = {
        "mode": args.mode,
        "sigma_allow": args.sigma,
        "node_spacing": args.spacing,
        "target_volume": args.volume,
        "min_radius": args.min_radius,
        "max_bar_length": args.max_length,
        "sample": args.sample,
    }
    try:
        run_json, npz = _run_files(args.run)
        rho, grid, _, _ = from_npz_bytes(npz.read_bytes())
        session = Session()
        project = session.load_case(run_json)
    except (ValueError, OSError, NotFoundError) as exc:
        _err("struts", _explain(exc))
        return EXIT_INVALID
    try:
        result, design = session.struts_from_density(project.id, rho, grid, **params)
    except ProjectInvalid as exc:
        _report_invalid(exc)
        return EXIT_INVALID
    except ValidationError as exc:
        _err("struts", f"invalid options: {_explain(exc)}")
        return EXIT_INVALID
    except ValueError as exc:
        _err("struts", f"failed: {exc}")
        return 1
    out = Path(args.out).expanduser() if args.out else run_json.parent
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / "struts.stl").write_bytes(to_stl_bytes(result.mesh))
        (out / "struts.png").write_bytes(struts_png(result, design))
        (out / "struts.json").write_text(pretty_json(strut_json(result)) + "\n")
    except OSError as exc:
        _err("struts", f"could not write to {out}: {exc}")
        return 1
    s = result.summary()
    for w in result.warnings:
        _err("struts", f"warning: {w}")
    print(
        f"struts      mode {s['mode']}: {s['n_bars']} bars, {s['n_nodes']} nodes, radius "
        f"{s['radius_min']:.4g}..{s['radius_max']:.4g}"
    )
    print(
        f"volume      {s['volume']:.6g} (target {s['target_volume']:.6g}, voxels "
        f"{s['voxel_volume']:.6g}, SIMP {s['simp_volume']:.6g})"
    )
    comp = ", ".join(f"{c:.6e}" for c in s["compliance"])
    simp = ", ".join(f"{c:.6e}" for c in s["simp_compliance"])
    ratio = s["compliance_ratio"]
    print(f"compliance  {comp} (SIMP {simp}; ratio {'n/a' if ratio is None else f'{ratio:.3f}'})")
    print(f"stress      max {s['stress_max']:.4e} von Mises")
    print(
        f"mesh        watertight {s['watertight']}, {s['n_bodies']} body(ies), "
        f"{s['triangles']} triangles"
    )
    print(f"wall time   {time.perf_counter() - t0:.1f} s")
    print(f"files       {out}/ : {', '.join(STRUT_FILES)}")
    return 0


# ---- mcp ----------------------------------------------------------------------------------------


def _mcp(args: argparse.Namespace) -> int:
    from topop.mcp_server import main as mcp_main

    mcp_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="topop", description="3D topology optimization tool")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="serve the API and the built frontend")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--dev", action="store_true", help="auto-reload; use with `npm run dev`")
    p_serve.set_defaults(func=_serve)

    p_run = sub.add_parser(
        "run",
        help="run a case file headless",
        description="Run a case (ProjectIn JSON with mesh `path`s) and write result files. "
        "Exit codes: 0 done, 1 failed (also: result.stl could not be written), 2 invalid "
        "case/project/arguments, 3 out of memory, 130 interrupted.",
    )
    p_run.add_argument("case", metavar="CASE.json")
    p_run.add_argument("--out", metavar="DIR", default=None, help="default: ./<case>-out")
    p_run.add_argument("--max-iter", type=int, default=None, help="override params.max_iter")
    p_run.add_argument(
        "--resolution", type=int, default=None, help="override grid.elements_along_longest"
    )
    p_run.add_argument(
        "--threshold",
        type=_threshold_arg,
        default=0.5,
        help="STL iso level in (0, 1] (default 0.5)",
    )
    p_run.add_argument(
        "--smooth", type=_smooth_arg, default=0, help="STL Laplacian smoothing iterations (>= 0)"
    )
    p_run.add_argument("--quiet", action="store_true", help="no per-iteration lines")
    p_run.add_argument(
        "--trim",
        action="store_true",
        help="intersect result.stl with the design mesh (exact CAD skin)",
    )
    p_run.add_argument(
        "--optimizer", choices=("oc", "mma"), default=None, help="override params.optimizer"
    )
    p_run.add_argument(
        "--stress-limit",
        type=float,
        metavar="SIGMA",
        default=None,
        help="von Mises limit (units of E); forces the mma optimizer",
    )
    p_run.add_argument(
        "--overhang",
        choices=("+x", "-x", "+y", "-y", "+z", "-z"),
        default=None,
        metavar="+z",
        help="additive-manufacturing build direction (45 deg overhang filter; base plate on the "
        "domain face opposite it). Write negative ones as --overhang=-z",
    )
    p_run.add_argument(
        "--symmetry",
        type=_symmetry_arg,
        action="append",
        metavar="AXIS[=POS]",
        help="mirror the design about the plane AXIS=POS (default: domain center); repeatable",
    )
    p_run.set_defaults(func=_run)

    p_describe = sub.add_parser("describe", help="print the facet table of a mesh")
    p_describe.add_argument("mesh", metavar="MESH")
    p_describe.add_argument(
        "--angle", type=float, default=5.0, help="coplanarity angle, degrees (STEP: ignored)"
    )
    p_describe.add_argument("--top", type=int, default=30, help="facets to list (0 = all)")
    p_describe.add_argument("--png", metavar="OUT.png", default=None, help="also render a preview")
    p_describe.add_argument("--view", default="iso", help="iso, +x, -x, +y, -y, +z or -z")
    p_describe.set_defaults(func=_describe)

    p_struts = sub.add_parser(
        "struts",
        help="turn a `topop run` result into an explicit strut (truss) structure",
        description="Read RUN_DIR/run.json and density.npz (what `topop run --out RUN_DIR` "
        "writes), rebuild the problem, generate struts (layout: minimum-volume truss LP over a "
        "ground structure; skeleton: medial axis of the SIMP solid), verify them by FE and write "
        "struts.stl, struts.png and struts.json. Exit codes: 0 done, 1 failed, 2 invalid input.",
    )
    p_struts.add_argument("run", metavar="RUN_DIR_OR_RUN_JSON")
    p_struts.add_argument("--out", metavar="DIR", default=None, help="default: the run directory")
    p_struts.add_argument("--mode", choices=("layout", "skeleton"), default="layout")
    p_struts.add_argument(
        "--sigma", type=float, default=20.0, help="LP stress limit, units of E (default 20)"
    )
    p_struts.add_argument(
        "--spacing", type=float, default=None, help="layout node spacing (default 4 voxels)"
    )
    p_struts.add_argument(
        "--volume",
        type=float,
        default=None,
        help="strut volume (default: the SIMP material volume; <= 0: sigma sizing)",
    )
    p_struts.add_argument(
        "--min-radius", type=float, default=None, help="default max(1, 0.8 voxel)"
    )
    p_struts.add_argument(
        "--max-length", type=float, default=None, help="longest bar (default 0.4 x diagonal)"
    )
    p_struts.add_argument(
        "--sample",
        choices=("solid", "active"),
        default="solid",
        help="layout nodes from the SIMP solid or the whole design domain",
    )
    p_struts.set_defaults(func=_struts)

    p_mcp = sub.add_parser("mcp", help="MCP server (stdio) over the same API")
    p_mcp.set_defaults(func=_mcp)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
