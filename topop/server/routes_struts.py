"""Strut post-processing of a finished run (`core.struts`): POST /api/runs/{id}/struts computes and
stores it, GET /api/runs/{id}/struts.stl downloads the mesh.

Request/response models live here, not in `schemas.py` (frozen contract); they reach OpenAPI
through the routes. The helpers are shared with `topop.agent.Session` (CLI, MCP).
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from topop.core.export import render_png, to_stl_bytes
from topop.core.problem import Grid
from topop.core.struts import StrutParams, StrutResult, generate_struts
from topop.server.build import BuiltDomain, ProblemInvalid, build_problem
from topop.server.routes_runs import DESIGN_RGB, NOT_FOUND, RESULT_RGB, StoreDep, _binary, _result
from topop.server.schemas import ErrorResponse, Project, ProjectIn
from topop.server.store import _SAFE_ID, NotFoundError, Store

router = APIRouter(prefix="/api", tags=["struts"])


class StrutRequest(BaseModel):
    """Strut generation parameters (`core.struts.StrutParams`); lengths in mesh units."""

    mode: Literal["layout", "skeleton"] = Field(
        "layout",
        description="layout: ground-structure truss LP; skeleton: medial axis of the SIMP solid",
    )
    sigma_allow: float = Field(20.0, gt=0, description="LP stress limit (units of E)")
    node_spacing: float | None = Field(None, gt=0, description="sampled-node spacing; null: 4 h")
    max_bar_length: float | None = Field(
        None, gt=0, description="longest candidate bar; null: 0.4 x domain diagonal"
    )
    target_volume: float | None = Field(
        None, description="strut volume; null: the SIMP material volume; <= 0: sigma_allow sizing"
    )
    min_radius: float | None = Field(None, gt=0, description="null: max(1, 0.8 h)")
    sample: Literal["solid", "active"] = Field(
        "solid", description="layout nodes from the SIMP solid (rho >= 0.3) or the whole domain"
    )


class StrutSummary(BaseModel):
    run_id: str
    mode: str
    n_nodes: int
    n_bars: int
    radius_min: float
    radius_max: float
    volume: float = Field(description="strut volume outside the keep-ins (mesh)")
    target_volume: float
    voxel_volume: float = Field(description="strut voxels in the free cells x h^3")
    simp_volume: float = Field(description="sum of the SIMP densities over the free cells x h^3")
    lp_volume: float | None = Field(None, description="optimal LP volume at sigma_allow (layout)")
    compliance: list[float] = Field(description="per load case, voxelized struts")
    simp_compliance: list[float] = Field(description="per load case, the SIMP density")
    compliance_ratio: float | None = None
    stress_max: float | None = Field(None, description="max von Mises over the strut voxels")
    watertight: bool
    n_bodies: int
    triangles: int
    warnings: list[str]
    timings: dict[str, float]
    stl_url: str


def strut_params(req: StrutRequest, penal: float = 3.0) -> StrutParams:
    return StrutParams(**req.model_dump(), penal=penal)


def keep_in_meshes(project: Project | ProjectIn, meshes_world: dict) -> list[trimesh.Trimesh]:
    return [
        meshes_world[f"ref:{r.id}"]
        for r in project.ref_models
        if r.mode == "keep_in" and f"ref:{r.id}" in meshes_world
    ]


def struts_for_project(
    project: Project | ProjectIn,
    domain: BuiltDomain,
    rho: np.ndarray,
    grid: Grid,
    req: StrutRequest,
) -> tuple[StrutResult, trimesh.Trimesh | None]:
    """(struts, world design mesh) of `rho` on the project's problem. ValueError when the
    density does not belong to this grid or no layout carries the loads."""
    built = build_problem(project, domain.meshes_world, domain)
    if built.grid.shape != grid.shape or not np.allclose(
        [*built.grid.origin, built.grid.h], [*grid.origin, grid.h]
    ):
        raise ValueError(
            f"the density grid {grid.shape} does not match the project's grid {built.grid.shape}"
        )
    design = built.meshes_world.get("design")
    result = generate_struts(
        built.problem,
        rho,
        design,
        keep_in_meshes(project, built.meshes_world),
        strut_params(req, built.params.penal),
    )
    return result, design


def struts_png(result: StrutResult, design: trimesh.Trimesh | None, view: str = "iso") -> bytes:
    layers = [(design, DESIGN_RGB, 0.15), (result.mesh, RESULT_RGB, 1.0)]
    return render_png([layer for layer in layers if layer[0] is not None], view)


def strut_json(result: StrutResult) -> dict:
    """`StrutResult.to_json()` with NaN (no verification) as null."""
    data = result.to_json()
    if isinstance(data.get("stress_max"), float) and math.isnan(data["stress_max"]):
        data["stress_max"] = None
    return data


def stored_paths(store: Store, run_id: str) -> dict[str, Path]:
    if not _SAFE_ID.fullmatch(run_id):
        raise NotFoundError(f"run {run_id} not found")
    return {ext: store.run_dir / f"{run_id}.struts.{ext}" for ext in ("stl", "json", "png")}


def struts_for_run(
    store: Store, run_id: str, req: StrutRequest
) -> tuple[StrutResult, dict[str, Path]]:
    """Compute the struts of a finished run and store struts.{stl,json,png} next to its result.
    NotFoundError / ValueError (no result, not runnable, no layout) as raised."""
    rec = store.get_run(run_id)
    with rec.lock:
        status = rec.info.status
    if status not in ("done", "cancelled"):
        raise ValueError(f"run {run_id} is {status}; no result to post-process")
    res = store.run_result(rec)
    if res is None:
        raise ValueError(f"run {run_id} has no density result")
    rho, grid, _, _ = res
    domain = store.get_domain(rec.project)
    try:
        result, design = struts_for_project(rec.project, domain, rho, grid, req)
    except ProblemInvalid as exc:
        raise ValueError(str(exc)) from exc
    paths = stored_paths(store, run_id)
    paths["stl"].write_bytes(to_stl_bytes(result.mesh))
    paths["json"].write_text(json.dumps({"run_id": run_id, **strut_json(result)}))
    paths["png"].write_bytes(struts_png(result, design))
    return result, paths


@router.post(
    "/runs/{id}/struts",
    response_model=StrutSummary,
    responses={**NOT_FOUND, 409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Turn a finished run into an explicit strut (truss) structure, verified by FE",
)
async def create_struts(id: str, body: StrutRequest, store: StoreDep) -> StrutSummary:
    await _result(store, id)  # 404 / 409 like the other exports

    try:
        result, _ = await asyncio.to_thread(struts_for_run, store, id, body)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    data = strut_json(result)
    data = {k: v for k, v in data.items() if k in StrutSummary.model_fields}
    return StrutSummary(**data, run_id=id, stl_url=f"/api/runs/{id}/struts.stl")


@router.get(
    "/runs/{id}/struts.stl",
    response_class=Response,
    responses=_binary("model/stl", "Strut mesh made by POST /api/runs/{id}/struts"),
    summary="Download the strut structure of a run (POST /api/runs/{id}/struts first)",
)
async def struts_stl(id: str, store: StoreDep) -> Response:
    try:
        path = stored_paths(store, id)["stl"]
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, f"run {id} has no struts; POST /api/runs/{id}/struts first")
    data = await asyncio.to_thread(path.read_bytes)
    headers = {"Content-Disposition": f'attachment; filename="{id}-struts.stl"'}
    return Response(data, media_type="model/stl", headers=headers)
