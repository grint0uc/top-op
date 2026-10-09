from __future__ import annotations

import asyncio
import contextlib
import struct
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response, WebSocket
from starlette.websockets import WebSocketDisconnect

from topop.core.export import (
    density_to_mesh,
    render_png,
    to_npz_bytes,
    to_stl_bytes,
    to_vti_bytes,
)
from topop.core.problem import Grid
from topop.server.build import ProblemInvalid, build_problem
from topop.server.jobs import CLOSE, GONE, Mailbox, RunManager, get_runs, max_queued, runs_of
from topop.server.schemas import (
    ErrorResponse,
    RunCreate,
    RunExport,
    RunInfo,
    StatusMsg,
    VoxelStats,
)
from topop.server.store import NotFoundError, RunRecord, Store, get_store, store_of

router = APIRouter(prefix="/api", tags=["runs"])

NOT_FOUND = {404: {"model": ErrorResponse}}
DESIGN_RGB = (0.75, 0.75, 0.78)
RESULT_RGB = (0.95, 0.55, 0.15)
PREVIEW_SMOOTH = 3
WARNINGS_HEADER = "X-Topop-Warnings"
TRIM_HEADER_DOC = {
    WARNINGS_HEADER: {
        "description": "Why `trim=true` left the result untrimmed (joined with '; '); absent if fine",
        "schema": {"type": "string"},
    }
}

StoreDep = Annotated[Store, Depends(get_store)]
RunsDep = Annotated[RunManager, Depends(get_runs)]


async def _run(store: Store, run_id: str) -> RunRecord:
    """The run record (may read runs/{id}.json written by another process); 404 if unknown."""
    try:
        return await asyncio.to_thread(store.get_run, run_id)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


async def _result(
    store: Store, run_id: str
) -> tuple[RunRecord, tuple[np.ndarray, Grid, np.ndarray, np.ndarray]]:
    """(record, (rho, grid, active, passive)); 409 unless the run finished with a result."""
    rec = await _run(store, run_id)
    with rec.lock:  # `_finish` sets the status and the in-memory result together
        status = rec.info.status
    if status not in ("done", "cancelled"):
        raise HTTPException(409, f"run {run_id} is {status}; no result to export")
    res = await asyncio.to_thread(store.run_result, rec)
    if res is None:
        raise HTTPException(409, f"run {run_id} has no density result")
    return rec, res


def _attachment(data: bytes, media_type: str, filename: str) -> Response:
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(data, media_type=media_type, headers=headers)


def _binary(media_type: str, description: str, headers: dict | None = None) -> dict:
    ok: dict = {
        "description": description,
        "content": {media_type: {"schema": {"type": "string", "format": "binary"}}},
    }
    if headers:
        ok["headers"] = headers
    return {200: ok, **NOT_FOUND}


def _warning_headers(warnings: list[str]) -> dict[str, str]:
    """`X-Topop-Warnings` (latin-1 safe, single line) or no header at all."""
    if not warnings:
        return {}
    text = "; ".join(" ".join(w.split()) for w in warnings)
    return {WARNINGS_HEADER: text.encode("latin-1", "replace").decode("latin-1")}


@router.post(
    "/runs",
    response_model=RunInfo,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
    summary="Start a run for a project",
)
async def create_run(body: RunCreate, store: StoreDep, runs: RunsDep) -> RunInfo:
    _check_queue(runs)
    try:
        project = await asyncio.to_thread(store.get_project, body.project_id)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    def build():
        domain = store.get_domain(project)
        return build_problem(project, domain.meshes_world, domain)

    try:
        built = await asyncio.to_thread(build)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ProblemInvalid as exc:
        raise HTTPException(422, str(exc)) from exc
    except ValueError as exc:  # no design mesh, degenerate geometry
        raise HTTPException(409, str(exc)) from exc
    _check_queue(runs)  # again: other requests may have queued runs while this one was built
    rec = store.new_run(project, built, VoxelStats(**built.stats))
    runs.start(rec.info.id, built, project.params)
    return rec.snapshot()


def _check_queue(runs: RunManager) -> None:
    """429 when TOPOP_MAX_QUEUED runs already wait (each holds its assembled problem in memory)."""
    limit, waiting = max_queued(), runs.queued()
    if waiting >= limit:
        raise HTTPException(
            429,
            f"{waiting} runs are already queued (TOPOP_MAX_QUEUED={limit}); "
            "wait for one to finish or cancel one",
        )


@router.websocket("/runs/{id}/stream")
async def run_stream(websocket: WebSocket, id: str) -> None:
    """Text frames: ProgressMsg / StatusMsg JSON. Binary frames: [u32 it][u32 nx][u32 ny][u32 nz]
    [u8 rho*255 ...] (CLAUDE.md). Not part of OpenAPI; message types are in components.schemas.

    A client that falls behind gets every JSON message but only the newest density frame; one that
    reads nothing for `jobs.STALL_SECONDS` while messages wait is disconnected."""
    await websocket.accept()
    store, runs = store_of(websocket.app), runs_of(websocket.app)
    try:
        rec = await asyncio.to_thread(store.get_run, id)
    except NotFoundError as exc:
        await websocket.send_json(StatusMsg(type="error", message=str(exc)).model_dump(mode="json"))
        await websocket.close(code=1008)
        return
    box = Mailbox(asyncio.get_running_loop())
    info, frame, final = await asyncio.to_thread(runs.subscribe, rec, box)
    watcher = asyncio.create_task(_watch_disconnect(websocket, box))
    try:
        await websocket.send_json(StatusMsg(type="started", run=info).model_dump(mode="json"))
        if frame is not None:
            await websocket.send_bytes(frame)
        if final is not None:
            await websocket.send_json(final)
        else:
            while (msg := await box.get()) is not CLOSE:
                if msg is GONE:  # disconnected, or dropped for not reading
                    return
                if isinstance(msg, bytes):
                    await websocket.send_bytes(msg)
                else:
                    await websocket.send_json(msg)
        await websocket.close()
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass  # client went away
    finally:
        watcher.cancel()
        runs.unsubscribe(rec, box)


async def _watch_disconnect(websocket: WebSocket, box: Mailbox) -> None:
    """Drain client frames (ignored) so a disconnect is noticed while the run is quiet."""
    with contextlib.suppress(Exception):
        while (await websocket.receive())["type"] != "websocket.disconnect":
            pass
    box.drop()


@router.post(
    "/runs/{id}/cancel", response_model=RunInfo, responses=NOT_FOUND, summary="Cancel a run"
)
async def cancel_run(id: str, runs: RunsDep) -> RunInfo:
    try:
        return await asyncio.to_thread(runs.cancel, id)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get(
    "/runs/{id}/result.stl",
    response_class=Response,
    responses=_binary("model/stl", "Isosurface of the density field", TRIM_HEADER_DOC),
    summary="Export the result as STL (trim=true: intersected with the design mesh)",
)
async def result_stl(
    id: str,
    store: StoreDep,
    threshold: float = Query(0.5, ge=0, le=1),
    smooth: int = Query(3, ge=0, le=50),
    trim: bool = Query(False),
) -> Response:
    rec, (rho, grid, _, _) = await _result(store, id)

    def work() -> tuple[bytes, list[str]]:
        mesh = density_to_mesh(rho, grid, threshold, smooth)
        mesh, warnings = store.trim_result(rec, mesh) if trim else (mesh, [])
        return to_stl_bytes(mesh), warnings

    try:
        data, warnings = await asyncio.to_thread(work)
    except ValueError as exc:  # threshold 0
        raise HTTPException(422, str(exc)) from exc
    res = _attachment(data, "model/stl", f"{id}.stl")
    res.headers.update(_warning_headers(warnings))
    return res


@router.get(
    "/runs/{id}/result.vti",
    response_class=Response,
    responses=_binary("application/xml", "VTK ImageData with the density field"),
    summary="Export the density field as VTI",
)
async def result_vti(id: str, store: StoreDep) -> Response:
    rec, (rho, grid, _, passive) = await _result(store, id)
    stress = await asyncio.to_thread(store.run_stress, rec)
    data = await asyncio.to_thread(to_vti_bytes, rho, passive, grid, stress)
    return _attachment(data, "application/xml", f"{id}.vti")


@router.get(
    "/runs/{id}/result.npz",
    response_class=Response,
    responses=_binary("application/octet-stream", "NumPy archive with the density field"),
    summary="Export the density field as NPZ",
)
async def result_npz(id: str, store: StoreDep) -> Response:
    rec, (rho, grid, active, passive) = await _result(store, id)
    stress = await asyncio.to_thread(store.run_stress, rec)
    data = await asyncio.to_thread(to_npz_bytes, rho, grid, active, passive, stress)
    return _attachment(data, "application/octet-stream", f"{id}.npz")


@router.get(
    "/runs/{id}/stress",
    response_class=Response,
    responses={
        **_binary(
            "application/octet-stream",
            "[u32 nx][u32 ny][u32 nz][f32 von Mises per element], little-endian, C-order, "
            "0 on inactive elements",
        ),
        409: {"model": ErrorResponse},
    },
    summary="Von Mises stress field of the final design",
)
async def result_stress(id: str, store: StoreDep) -> Response:
    rec, (_, _, active, _) = await _result(store, id)
    stress = await asyncio.to_thread(store.run_stress, rec)
    if stress is None:
        raise HTTPException(409, f"run {id} has no stress field")

    def pack() -> bytes:
        field = np.where(np.asarray(active, dtype=bool), np.nan_to_num(stress), 0.0)
        head = struct.pack("<3I", *field.shape)
        return head + np.ascontiguousarray(field, dtype="<f4").tobytes()

    return Response(await asyncio.to_thread(pack), media_type="application/octet-stream")


@router.get(
    "/runs/{id}/project.json",
    response_model=RunExport,
    responses=NOT_FOUND,
    summary="Project + run record (reloadable, re-runnable headlessly)",
)
async def run_project(id: str, store: StoreDep) -> RunExport:
    rec = await _run(store, id)
    return RunExport(project=rec.project, run=rec.snapshot())


@router.get("/runs", response_model=list[RunInfo], summary="List runs")
async def list_runs(store: StoreDep) -> list[RunInfo]:
    return [rec.snapshot() for rec in await asyncio.to_thread(store.list_runs)]


@router.get("/runs/{id}", response_model=RunInfo, responses=NOT_FOUND, summary="Run status")
async def get_run(id: str, store: StoreDep) -> RunInfo:
    return (await _run(store, id)).snapshot()


@router.get(
    "/runs/{id}/preview.png",
    response_class=Response,
    responses=_binary("image/png", "Offscreen render of the result", TRIM_HEADER_DOC),
    summary="PNG render of the result so an agent can look at it",
)
async def run_preview(
    id: str,
    store: StoreDep,
    threshold: float = Query(0.5, ge=0, le=1),
    view: str = Query("iso"),
    trim: bool = Query(False),
) -> Response:
    rec, (rho, grid, _, _) = await _result(store, id)

    def work() -> tuple[bytes, list[str]]:
        design = store.design_world(rec)
        result = density_to_mesh(rho, grid, threshold, PREVIEW_SMOOTH)
        result, warnings = store.trim_result(rec, result) if trim else (result, [])
        layers = [(design, DESIGN_RGB, 0.15), (result, RESULT_RGB, 1.0)]
        return render_png([layer for layer in layers if layer[0] is not None], view), warnings

    try:
        png, warnings = await asyncio.to_thread(work)
    except ValueError as exc:  # unknown view, threshold 0
        raise HTTPException(422, str(exc)) from exc
    return Response(png, media_type="image/png", headers=_warning_headers(warnings))
