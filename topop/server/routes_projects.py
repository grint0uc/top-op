from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from topop.core import selection as core_selection
from topop.server.build import (
    BuiltDomain,
    params_warnings,
    resolve_project_selections,
    resolve_sel,
)
from topop.server.schemas import (
    ErrorResponse,
    Project,
    ProjectIn,
    ResolvedNodes,
    Selection,
    VoxelStats,
)
from topop.server.store import NotFoundError, Store, get_store

router = APIRouter(prefix="/api", tags=["projects"])

NOT_FOUND = {404: {"model": ErrorResponse}}

StoreDep = Annotated[Store, Depends(get_store)]


async def _project(store: Store, project_id: str) -> Project:
    """The project, re-read if another process (`topop mcp`) saved it; 404 if unknown."""
    try:
        return await asyncio.to_thread(store.get_project, project_id)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


async def project_domain(store: Store, project: Project) -> BuiltDomain:
    """Cached voxel domain; missing mesh -> 404, nothing to voxelize -> 409."""
    try:
        return await asyncio.to_thread(store.get_domain, project)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/projects", response_model=Project, summary="Create a project")
async def create_project(body: ProjectIn, store: StoreDep) -> Project:
    return await asyncio.to_thread(store.create_project, body)


@router.get("/projects", response_model=list[Project], summary="List projects")
async def list_projects(store: StoreDep) -> list[Project]:
    return await asyncio.to_thread(store.list_projects)


@router.get("/projects/{id}", response_model=Project, responses=NOT_FOUND, summary="Get a project")
async def get_project(id: str, store: StoreDep) -> Project:
    return await _project(store, id)


@router.put(
    "/projects/{id}", response_model=Project, responses=NOT_FOUND, summary="Replace a project"
)
async def update_project(id: str, body: Project, store: StoreDep) -> Project:
    try:
        return await asyncio.to_thread(store.update_project, id, body)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post(
    "/projects/{id}/voxelize",
    response_model=VoxelStats,
    responses=NOT_FOUND,
    summary="Voxelize the project and report grid statistics",
)
async def voxelize_project(id: str, store: StoreDep) -> VoxelStats:
    project = await _project(store, id)
    domain = await project_domain(store, project)
    # loads/supports that resolve to nothing at this resolution are worth a warning here
    _, _, warnings, errors = await asyncio.to_thread(resolve_project_selections, project, domain)
    notes = params_warnings(project.params)
    return VoxelStats(
        **{**domain.stats, "warnings": [*domain.warnings, *warnings, *errors, *notes]}
    )


@router.post(
    "/projects/{id}/resolve-selection",
    response_model=ResolvedNodes,
    responses=NOT_FOUND,
    summary="Resolve a selection to full-grid nodes (preview capped at 5000 points)",
)
async def resolve_selection(id: str, body: Selection, store: StoreDep) -> ResolvedNodes:
    project = await _project(store, id)
    domain = await project_domain(store, project)

    def work() -> ResolvedNodes:
        nodes = resolve_sel(body.model_dump(), domain)
        return ResolvedNodes(**core_selection.resolved_preview(nodes, domain.grid))

    try:
        return await asyncio.to_thread(work)
    except ValueError as exc:  # unknown mesh / facet ids, degenerate direction
        raise HTTPException(422, str(exc)) from exc
