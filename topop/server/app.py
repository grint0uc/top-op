from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, TypeAdapter
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from topop import __version__
from topop.server import routes_meshes, routes_projects, routes_runs, routes_struts
from topop.server.jobs import RunManager
from topop.server.schemas import Selection, WsMessage
from topop.server.store import Store

STATIC_DIR = Path(__file__).parent / "static"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")
STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})
DEFAULT_MAX_UPLOAD_MB = 200.0


def allowed_hosts() -> list[str]:
    """Host names the server answers to: this machine, plus TOPOP_ALLOWED_HOSTS (comma-separated,
    `*.example.com` patterns, `*` = any) for serving beyond localhost."""
    extra = os.environ.get("TOPOP_ALLOWED_HOSTS", "")
    return [*LOCAL_HOSTS, *(h.strip() for h in extra.split(",") if h.strip())]


def max_upload_bytes() -> int:
    """TOPOP_MAX_UPLOAD_MB (default 200) as bytes: the largest request body accepted."""
    try:
        mb = float(os.environ.get("TOPOP_MAX_UPLOAD_MB") or DEFAULT_MAX_UPLOAD_MB)
    except ValueError:
        mb = DEFAULT_MAX_UPLOAD_MB
    return int(mb * 1024 * 1024)


def _host_allowed(host: str, patterns: list[str]) -> bool:
    return any(
        p == "*" or host == p or (p.startswith("*.") and host.endswith(p[1:])) for p in patterns
    )


def _is_test_client(scope: Scope) -> bool:
    """Starlette's TestClient (peer "testclient", Host "testserver"); no socket has that peer."""
    client = scope.get("client")
    host = Headers(scope=scope).get("host")
    return bool(client) and client[0] == "testclient" and host == "testserver"


class LocalHostMiddleware(TrustedHostMiddleware):
    """TrustedHostMiddleware (a DNS-rebinding guard) that also lets the in-process TestClient in."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and _is_test_client(scope):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


class OriginMiddleware:
    """403 for state-changing requests and WebSocket handshakes sent by a page of another site
    (CSRF): a present `Origin` must name an allowed host. Requests without one (curl, MCP, the
    same-origin GUI's GETs) pass."""

    def __init__(self, app: ASGIApp, allowed: list[str]):
        self.app = app
        self.allowed = allowed

    def _ok(self, origin: str) -> bool:
        try:
            url = urlsplit(origin)
            host = url.hostname
        except ValueError:
            return False
        if url.scheme not in ("http", "https") or not host:
            return False  # "null" (sandboxed frames, file://) and anything odd
        return _host_allowed(f"[{host}]" if ":" in host else host, self.allowed)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        checked = scope["type"] == "websocket" or (
            scope["type"] == "http" and scope["method"] in STATE_CHANGING
        )
        origin = Headers(scope=scope).get("origin") if checked else None
        if origin is None or self._ok(origin):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":  # closing before accept rejects the handshake (403)
            await send({"type": "websocket.close", "code": 1008, "reason": "origin not allowed"})
            return
        detail = f"cross-site request refused: Origin {origin} is not this machine"
        await JSONResponse({"detail": detail}, status_code=403)(scope, receive, send)


class BodyLimitMiddleware:
    """413 for request bodies above TOPOP_MAX_UPLOAD_MB: refused up front by Content-Length, or
    aborted while a chunked body streams in, before it is all read."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = max_upload_bytes()
        detail = f"request body larger than {limit / 2**20:g} MB (TOPOP_MAX_UPLOAD_MB)"
        declared = Headers(scope=scope).get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)
            return
        seen = 0

        async def counted() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:  # FastAPI re-raises HTTPExceptions from body parsing
                    raise HTTPException(413, detail)
            return message

        await self.app(scope, counted, send)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Fresh store (data dir from TOPOP_DATA_DIR) and run manager; cancel running jobs on exit."""
    store = Store()
    runs = RunManager(store)
    app.state.store, app.state.runs = store, runs
    try:
        yield
    finally:
        await asyncio.to_thread(runs.shutdown)
        app.state.store = app.state.runs = None


app = FastAPI(
    title="top-op",
    version=__version__,
    # one schema per model (no Foo-Input / Foo-Output split) so types.gen.ts names stay stable
    separate_input_output_schemas=False,
    lifespan=lifespan,
)


# outermost first: Host, then Origin, then body size
app.add_middleware(BodyLimitMiddleware)
app.add_middleware(OriginMiddleware, allowed=allowed_hosts())
app.add_middleware(LocalHostMiddleware, allowed_hosts=allowed_hosts())


class Health(BaseModel):
    status: str
    version: str


app.include_router(routes_meshes.router)
app.include_router(routes_projects.router)
app.include_router(routes_runs.router)
app.include_router(routes_struts.router)


@app.get("/api/health", response_model=Health, tags=["meta"])
async def health() -> Health:
    return Health(status="ok", version=__version__)


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:

    @app.get("/", include_in_schema=False)
    async def placeholder() -> PlainTextResponse:
        return PlainTextResponse("top-op API is running; frontend not built (make build). /docs")


def custom_openapi() -> dict[str, Any]:
    """Add the models no route references (WebSocket messages, Selection) to components.schemas
    so openapi-typescript emits them."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    for name, annotation in (("WsMessage", WsMessage), ("Selection", Selection)):
        js = TypeAdapter(annotation).json_schema(ref_template="#/components/schemas/{model}")
        components.update(js.pop("$defs", {}))
        components[name] = js
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi  # type: ignore[method-assign]
