from __future__ import annotations

import argparse
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


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

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
    """MeshInfo block + fixed-width facet table."""
    m = data["mesh"]
    lo, hi = m["bbox"]
    size = [b - a for a, b in zip(lo, hi, strict=True)]
    volume = "n/a (not watertight)" if m["volume"] is None else f"{m['volume']:.6g}"
    lines = [
        f"mesh {m['name']}  id {m['id']}",
        (
            f"  faces {m['n_faces']}  vertices {m['n_vertices']}  watertight {m['is_watertight']}"
            f"  volume {volume}"
        ),
        f"  bbox min {_vec(lo, '.6g')}  max {_vec(hi, '.6g')}  size {_vec(size, '.6g')}",
        (
            f"facets (coplanar groups, angle {data['angle_deg']:g} deg, by area): "
            f"showing {len(data['facets'])} of {data['n_facets_total']}"
        ),
        f"{'id':>4} {'faces':>6} {'area':>11}  {'normal':<24}  {'centroid':<30}  bbox min .. max",
    ]
    for f in data["facets"]:
        lines.append(
            f"{f['id']:>4} {f['n_faces']:>6} {f['area']:>11.5g}  {_vec(f['normal'], '+.3f'):<24}  "
            f"{_vec(f['centroid'], '.5g'):<30}  {_vec(f['bbox'][0], '.5g')} .. {_vec(f['bbox'][1], '.5g')}"
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
    for w in stats.warnings:
        print(f"warning: {w}")


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
        stats = session.voxel_stats(pid)
        bounds = session.boundaries(pid)
        project = session.get_project(pid)
    except (ValueError, OSError, NotFoundError) as exc:
        _report_invalid(exc)
        return EXIT_INVALID
    if not args.quiet:
        _print_setup(args.case, project, stats, bounds)
        print(f"{'it':>4} {'compliance':>13} {'volume':>8} {'change':>8} {'t_iter':>8}", flush=True)

    def progress(r) -> None:
        if not args.quiet:
            line = f"{r.it:>4} {r.compliance:>13.6e} {r.volume:>8.4f} {r.change:>8.4f} {r.t_iter:>7.2f}s"
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
        result = session.write_outputs(info.id, out, args.threshold, args.smooth)
    except OSError as exc:
        _err("run", f"could not write to {out}: {exc}")
        return 1
    written = result["files"]
    for name, why in result["errors"].items():
        _err("run", f"{name} not written: {why}")
    last = info.history[-1]
    detail = outcome.get("message") or status
    print(f"status      {status} ({detail})  run {info.id}")
    print(
        f"iterations  {len(info.history)}   compliance {last.compliance:.6e}   volume {last.volume:.4f}"
    )
    print(f"wall time   {wall:.1f} s total, {outcome.get('wall_s', 0.0):.1f} s optimizing")
    print(f"files       {out}/ : {', '.join(written)}")
    print(f"rerun       topop run {Path(out) / 'run.json'}")
    return EXIT_INTERRUPTED if status == "cancelled" else 0


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
        "Exit codes: 0 done, 1 failed, 2 invalid case/project, 3 out of memory, 130 interrupted.",
    )
    p_run.add_argument("case", metavar="CASE.json")
    p_run.add_argument("--out", metavar="DIR", default=None, help="default: ./<case>-out")
    p_run.add_argument("--max-iter", type=int, default=None, help="override params.max_iter")
    p_run.add_argument(
        "--resolution", type=int, default=None, help="override grid.elements_along_longest"
    )
    p_run.add_argument("--threshold", type=float, default=0.5, help="STL iso level (default 0.5)")
    p_run.add_argument("--smooth", type=int, default=0, help="STL Laplacian smoothing iterations")
    p_run.add_argument("--quiet", action="store_true", help="no per-iteration lines")
    p_run.set_defaults(func=_run)

    p_describe = sub.add_parser("describe", help="print the facet table of a mesh")
    p_describe.add_argument("mesh", metavar="MESH")
    p_describe.add_argument("--angle", type=float, default=5.0, help="coplanarity angle, degrees")
    p_describe.add_argument("--top", type=int, default=30, help="facets to list (0 = all)")
    p_describe.add_argument("--png", metavar="OUT.png", default=None, help="also render a preview")
    p_describe.add_argument("--view", default="iso", help="iso, +x, -x, +y, -y, +z or -z")
    p_describe.set_defaults(func=_describe)

    p_mcp = sub.add_parser("mcp", help="MCP server (stdio) over the same API")
    p_mcp.set_defaults(func=_mcp)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
