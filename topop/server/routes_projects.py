from __future__ import annotations

from fastapi import APIRouter, HTTPException

from topop.server.schemas import (
    ErrorResponse,
    Project,
    ProjectIn,
    ResolvedNodes,
    Selection,
    VoxelStats,
)

router = APIRouter(prefix="/api", tags=["projects"])

NOT_FOUND = {404: {"model": ErrorResponse}}


@router.post("/projects", response_model=Project, summary="Create a project")
async def create_project(body: ProjectIn) -> Project:
    raise HTTPException(501, "not implemented")


@router.get("/projects", response_model=list[Project], summary="List projects")
async def list_projects() -> list[Project]:
    raise HTTPException(501, "not implemented")


@router.get("/projects/{id}", response_model=Project, responses=NOT_FOUND, summary="Get a project")
async def get_project(id: str) -> Project:
    raise HTTPException(501, "not implemented")


@router.put(
    "/projects/{id}", response_model=Project, responses=NOT_FOUND, summary="Replace a project"
)
async def update_project(id: str, body: Project) -> Project:
    raise HTTPException(501, "not implemented")


@router.post(
    "/projects/{id}/voxelize",
    response_model=VoxelStats,
    responses=NOT_FOUND,
    summary="Voxelize the project and report grid statistics",
)
async def voxelize_project(id: str) -> VoxelStats:
    raise HTTPException(501, "not implemented")


@router.post(
    "/projects/{id}/resolve-selection",
    response_model=ResolvedNodes,
    responses=NOT_FOUND,
    summary="Resolve a selection to full-grid nodes (preview capped at 5000 points)",
)
async def resolve_selection(id: str, body: Selection) -> ResolvedNodes:
    raise HTTPException(501, "not implemented")
