from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

NOT_IMPLEMENTED = "not implemented yet"


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


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"topop {args.command}: {NOT_IMPLEMENTED}", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="topop", description="3D topology optimization tool")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="serve the API and the built frontend")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--dev", action="store_true", help="auto-reload; use with `npm run dev`")
    p_serve.set_defaults(func=_serve)

    p_run = sub.add_parser("run", help="run a case file headless")
    p_run.add_argument("case", metavar="CASE.json")
    p_run.add_argument("--out", metavar="DIR", default=None)
    p_run.set_defaults(func=_not_implemented)

    p_describe = sub.add_parser("describe", help="print the facet table of a mesh")
    p_describe.add_argument("mesh", metavar="MESH")
    p_describe.set_defaults(func=_not_implemented)

    p_mcp = sub.add_parser("mcp", help="MCP server (stdio) over the same API")
    p_mcp.set_defaults(func=_not_implemented)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
