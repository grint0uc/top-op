from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Annotated

import numpy as np
import trimesh
from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel

from topop.core.export import render_png
from topop.server.schemas import ErrorResponse, FacetInfo, MeshFacets, MeshInfo
from topop.server.store import NotFoundError, Store, get_store

router = APIRouter(prefix="/api", tags=["meshes"])

NOT_FOUND = {404: {"model": ErrorResponse}}
MAX_FACETS = 300
MESH_RGB = (0.75, 0.75, 0.78)

StoreDep = Annotated[Store, Depends(get_store)]


class FacetFaces(BaseModel):
    """Triangle ids (of GET /meshes/{id}/buffer) that make up one facet."""

    face_ids: list[int]


async def _with_mesh[T](store: Store, mesh_id: str, fn: Callable[[trimesh.Trimesh], T]) -> T:
    """Run fn(mesh) in a worker thread; unknown mesh -> 404."""

    def work() -> T:
        return fn(store.get_mesh(mesh_id))

    try:
        return await asyncio.to_thread(work)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


def _octets(data: bytes) -> Response:
    return Response(data, media_type="application/octet-stream")


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
async def upload_mesh(file: Annotated[UploadFile, File()], store: StoreDep) -> MeshInfo:
    data = await file.read()
    try:
        return await asyncio.to_thread(store.add_mesh, data, file.filename or "mesh.stl")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get(
    "/meshes/{id}",
    response_model=MeshInfo,
    responses={404: {"model": ErrorResponse}},
    summary="Mesh info by id",
)
async def mesh_get(id: str, store: StoreDep) -> MeshInfo:
    try:
        return await asyncio.to_thread(store.mesh_info, id)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get(
    "/meshes/{id}/buffer",
    response_class=Response,
    responses=_binary(
        "application/octet-stream",
        "[u32 n_vert][u32 n_tri][f32 xyz*n_vert][u32 ijk*n_tri][f32 nxyz per face], little-endian",
    ),
    summary="Render buffer for the viewport",
)
async def mesh_buffer(id: str, store: StoreDep) -> Response:
    def pack(mesh: trimesh.Trimesh) -> bytes:
        verts = np.asarray(mesh.vertices, dtype="<f4")
        faces = np.asarray(mesh.faces, dtype="<u4")
        normals = np.asarray(mesh.face_normals, dtype="<f4")
        head = np.array([len(verts), len(faces)], dtype="<u4")
        return b"".join(np.ascontiguousarray(a).tobytes() for a in (head, verts, faces, normals))

    return _octets(await _with_mesh(store, id, pack))


@router.get(
    "/meshes/{id}/adjacency",
    response_class=Response,
    responses=_binary(
        "application/octet-stream", "u32 pairs of adjacent face ids (trimesh.face_adjacency)"
    ),
    summary="Face adjacency for flat-face grow",
)
async def mesh_adjacency(id: str, store: StoreDep) -> Response:
    def pack(mesh: trimesh.Trimesh) -> bytes:
        return np.ascontiguousarray(mesh.face_adjacency, dtype="<u4").tobytes()

    return _octets(await _with_mesh(store, id, pack))


@router.get(
    "/meshes/{id}/facets",
    response_model=MeshFacets,
    responses=NOT_FOUND,
    summary="Coplanar face groups, area-sorted (agents pick by id)",
)
async def mesh_facets(
    id: str, store: StoreDep, angle_deg: float = Query(5.0, ge=0, le=90)
) -> MeshFacets:
    try:
        facets, total = await asyncio.to_thread(store.mesh_facets, id, angle_deg)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return MeshFacets(
        mesh_id=id,
        angle_deg=angle_deg,
        facets=[FacetInfo(**f) for f in facets[:MAX_FACETS]],
        n_facets_total=total,
    )


@router.get(
    "/meshes/{id}/facets/{facet_id}/faces",
    response_model=FacetFaces,
    responses=NOT_FOUND,
    summary="Triangle ids of one facet (GUI highlight; STEP meshes ignore angle_deg)",
)
async def mesh_facet_faces(
    id: str, facet_id: int, store: StoreDep, angle_deg: float = Query(5.0, ge=0, le=90)
) -> FacetFaces:
    try:
        faces = await asyncio.to_thread(store.facet_faces, id, [facet_id], angle_deg)
    except NotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:  # facet id outside the table
        raise HTTPException(404, str(exc)) from exc
    return FacetFaces(face_ids=faces.tolist())


@router.get(
    "/meshes/{id}/preview.png",
    response_class=Response,
    responses=_binary("image/png", "Offscreen render of the mesh"),
    summary="PNG render so an agent can look at the model",
)
async def mesh_preview(id: str, store: StoreDep, view: str = Query("iso")) -> Response:
    try:
        png = await _with_mesh(store, id, lambda m: render_png([(m, MESH_RGB, 1.0)], view))
    except ValueError as exc:  # unknown view
        raise HTTPException(422, str(exc)) from exc
    return Response(png, media_type="image/png")
