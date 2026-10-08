from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Response, WebSocket

from topop.server.schemas import ErrorResponse, RunCreate, RunExport, RunInfo

router = APIRouter(prefix="/api", tags=["runs"])

NOT_FOUND = {404: {"model": ErrorResponse}}


def _binary(media_type: str, description: str) -> dict:
    return {
        200: {
            "description": description,
            "content": {media_type: {"schema": {"type": "string", "format": "binary"}}},
        },
        **NOT_FOUND,
    }


@router.post(
    "/runs",
    response_model=RunInfo,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Start a run for a project",
)
async def create_run(body: RunCreate) -> RunInfo:
    raise HTTPException(501, "not implemented")


@router.websocket("/runs/{id}/stream")
async def run_stream(websocket: WebSocket, id: str) -> None:
    """Text frames: ProgressMsg / StatusMsg JSON. Binary frames: [u32 it][u32 nx][u32 ny][u32 nz]
    [u8 rho*255 ...] (CLAUDE.md). Not part of OpenAPI; message types are in components.schemas."""
    await websocket.accept()
    await websocket.close(code=1011, reason="not implemented")


@router.post(
    "/runs/{id}/cancel", response_model=RunInfo, responses=NOT_FOUND, summary="Cancel a run"
)
async def cancel_run(id: str) -> RunInfo:
    raise HTTPException(501, "not implemented")


@router.get(
    "/runs/{id}/result.stl",
    response_class=Response,
    responses=_binary("model/stl", "Isosurface of the density field"),
    summary="Export the result as STL",
)
async def result_stl(
    id: str,
    threshold: float = Query(0.5, ge=0, le=1),
    smooth: int = Query(3, ge=0, le=50),
) -> Response:
    raise HTTPException(501, "not implemented")


@router.get(
    "/runs/{id}/result.vti",
    response_class=Response,
    responses=_binary("application/xml", "VTK ImageData with the density field"),
    summary="Export the density field as VTI",
)
async def result_vti(id: str) -> Response:
    raise HTTPException(501, "not implemented")


@router.get(
    "/runs/{id}/result.npz",
    response_class=Response,
    responses=_binary("application/octet-stream", "NumPy archive with the density field"),
    summary="Export the density field as NPZ",
)
async def result_npz(id: str) -> Response:
    raise HTTPException(501, "not implemented")


@router.get(
    "/runs/{id}/project.json",
    response_model=RunExport,
    responses=NOT_FOUND,
    summary="Project + run record (reloadable, re-runnable headlessly)",
)
async def run_project(id: str) -> RunExport:
    raise HTTPException(501, "not implemented")


@router.get("/runs", response_model=list[RunInfo], summary="List runs")
async def list_runs() -> list[RunInfo]:
    raise HTTPException(501, "not implemented")


@router.get("/runs/{id}", response_model=RunInfo, responses=NOT_FOUND, summary="Run status")
async def get_run(id: str) -> RunInfo:
    raise HTTPException(501, "not implemented")


@router.get(
    "/runs/{id}/preview.png",
    response_class=Response,
    responses=_binary("image/png", "Offscreen render of the result"),
    summary="PNG render of the result so an agent can look at it",
)
async def run_preview(
    id: str, threshold: float = Query(0.5, ge=0, le=1), view: str = Query("iso")
) -> Response:
    raise HTTPException(501, "not implemented")
