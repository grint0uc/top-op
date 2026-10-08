from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, TypeAdapter

from topop import __version__
from topop.server import routes_meshes, routes_projects, routes_runs
from topop.server.schemas import Selection, WsMessage

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="top-op",
    version=__version__,
    # one schema per model (no Foo-Input / Foo-Output split) so types.gen.ts names stay stable
    separate_input_output_schemas=False,
)


class Health(BaseModel):
    status: str
    version: str


app.include_router(routes_meshes.router)
app.include_router(routes_projects.router)
app.include_router(routes_runs.router)


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
