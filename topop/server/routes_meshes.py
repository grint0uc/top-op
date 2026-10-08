from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile

from topop.server.schemas import ErrorResponse, MeshFacets, MeshInfo

router = APIRouter(prefix="/api", tags=["meshes"])

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
    "/meshes",
    response_model=MeshInfo,
    responses={400: {"model": ErrorResponse}},
    summary="Upload a mesh (STL/OBJ/3MF, multipart field `file`)",
)
async def upload_mesh(file: Annotated[UploadFile, File()]) -> MeshInfo:
    raise HTTPException(501, "not implemented")


@router.get(
    "/meshes/{id}/buffer",
    response_class=Response,
    responses=_binary(
        "application/octet-stream",
        "[u32 n_vert][u32 n_tri][f32 xyz*n_vert][u32 ijk*n_tri][f32 nxyz per face], little-endian",
    ),
    summary="Render buffer for the viewport",
)
async def mesh_buffer(id: str) -> Response:
    raise HTTPException(501, "not implemented")


@router.get(
    "/meshes/{id}/adjacency",
    response_class=Response,
    responses=_binary(
        "application/octet-stream", "u32 pairs of adjacent face ids (trimesh.face_adjacency)"
    ),
    summary="Face adjacency for flat-face grow",
)
async def mesh_adjacency(id: str) -> Response:
    raise HTTPException(501, "not implemented")


@router.get(
    "/meshes/{id}/facets",
    response_model=MeshFacets,
    responses=NOT_FOUND,
    summary="Coplanar face groups, area-sorted (agents pick by id)",
)
async def mesh_facets(id: str, angle_deg: float = Query(5.0, ge=0, le=90)) -> MeshFacets:
    raise HTTPException(501, "not implemented")


@router.get(
    "/meshes/{id}/preview.png",
    response_class=Response,
    responses=_binary("image/png", "Offscreen render of the mesh"),
    summary="PNG render so an agent can look at the model",
)
async def mesh_preview(id: str, view: str = Query("iso")) -> Response:
    raise HTTPException(501, "not implemented")
