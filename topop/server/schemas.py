"""API contract. pydantic v2. Generates OpenAPI -> web/src/api/types.gen.ts. See CLAUDE.md for conventions."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Transform = Annotated[list[float], Field(min_length=16, max_length=16)]  # column-major (three.js)
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Bool3 = Annotated[list[bool], Field(min_length=3, max_length=3)]

IDENTITY: list[float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


class MeshInfo(BaseModel):
    id: str
    name: str
    n_faces: int
    n_vertices: int
    bbox: list[Vec3]  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    is_watertight: bool
    volume: float | None = None
    source: Literal["mesh", "step"] = "mesh"
    n_brep_faces: int | None = None  # STEP only


class FaceSelection(BaseModel):
    """Explicit triangle ids (what the GUI produces by clicking/painting)."""

    kind: Literal["faces"] = "faces"
    mesh_id: str
    face_ids: list[int]


class FacetSelection(BaseModel):
    """Coplanar facet ids from GET /api/meshes/{id}/facets (agent-friendly, stable per angle_deg)."""

    kind: Literal["facets"] = "facets"
    mesh_id: str
    facet_ids: list[int]
    angle_deg: float = 5.0


class NormalSelection(BaseModel):
    """All surface faces whose normal is within angle_deg of `direction`, optionally clipped to a world bbox."""

    kind: Literal["normal"] = "normal"
    mesh_id: str
    direction: Vec3
    angle_deg: float = 10.0
    # [[xmin,ymin,zmin],[xmax,ymax,zmax]], clips strictly on grid-node coordinates. Grid nodes
    # sit up to h/2 outside the surface, so pad a box derived from the mesh bbox by one voxel h.
    within: list[Vec3] | None = None


class PlaneSelection(BaseModel):
    """Surface grid nodes within `tol` of the plane (no mesh needed). Agent-friendly: 'fix the plane x=0'."""

    kind: Literal["plane"] = "plane"
    point: Vec3
    normal: Vec3
    tol: float = 0.0  # 0 -> band of ±h/2 (the single node layer nearest the plane)


class PrimitiveSelection(BaseModel):
    kind: Literal["box", "sphere", "cylinder"]
    transform: Transform = Field(default_factory=lambda: list(IDENTITY))
    size: Vec3 = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    surface_only: bool = True


Selection = Annotated[
    FaceSelection | FacetSelection | NormalSelection | PlaneSelection | PrimitiveSelection,
    Field(discriminator="kind"),
]


class FacetInfo(BaseModel):
    id: int
    n_faces: int
    area: float
    normal: Vec3  # [0,0,0] when not planar
    centroid: Vec3
    bbox: list[Vec3]
    kind: Literal["plane", "cylinder", "other"] = "other"
    axis: Vec3 | None = None  # cylinder axis direction
    radius: float | None = None  # cylinder radius
    brep_face: int | None = None  # B-rep face index when the mesh came from STEP


class MeshFacets(BaseModel):
    mesh_id: str
    angle_deg: float
    facets: list[FacetInfo]  # sorted by area desc, capped at 300
    n_facets_total: int


class MeshRef(BaseModel):
    mesh_id: str | None = None  # server: required
    path: str | None = None  # CLI case files only: load from disk
    transform: Transform = Field(default_factory=lambda: list(IDENTITY))


class RefModel(BaseModel):
    id: str
    name: str = ""
    mesh_id: str | None = None  # server: required
    path: str | None = None  # CLI case files only
    transform: Transform = Field(default_factory=lambda: list(IDENTITY))
    mode: Literal["keep_in", "keep_out"] = "keep_out"
    visible: bool = True


class GridSpec(BaseModel):
    elements_along_longest: int = Field(default=60, ge=4, le=600)
    padding: int = Field(default=1, ge=0, le=10)


class MaterialSpec(BaseModel):
    E: float = Field(default=1.0, gt=0)
    nu: float = Field(default=0.3, ge=0, lt=0.5)


class SymmetrySpec(BaseModel):
    axis: Literal["x", "y", "z"]
    position: float | None = None  # world coordinate of the mirror plane; None -> domain center


class ParamsSpec(BaseModel):
    volfrac: float = Field(default=0.3, gt=0, lt=1)
    penal: float = Field(default=3.0, ge=1, le=6)
    rmin: float = Field(default=2.0, ge=1.0)
    max_iter: int = Field(default=100, ge=1, le=2000)
    tol: float = Field(default=0.01, gt=0)
    move: float = Field(default=0.2, gt=0, le=1)
    heaviside: bool = False
    continuation: bool = False
    solver: Literal["auto", "amg", "direct"] = "auto"
    dtype: Literal["float64", "float32"] = "float64"
    density_every: int = Field(default=1, ge=1)  # send a density frame every N iterations
    optimizer: Literal["oc", "mma"] = "oc"
    symmetry: list[SymmetrySpec] = Field(default_factory=list)
    stress_limit: float | None = Field(default=None, gt=0)  # von Mises limit, units of E
    stress_pnorm: float = Field(default=8.0, ge=2, le=40)
    overhang: Literal["+x", "-x", "+y", "-y", "+z", "-z"] | None = None  # AM build direction


class LoadSpec(BaseModel):
    id: str
    name: str = ""
    selection: Selection
    force: Vec3  # total force
    case: int = Field(default=0, ge=0)


class SupportSpec(BaseModel):
    id: str
    name: str = ""
    selection: Selection
    fix: Bool3 = Field(default_factory=lambda: [True, True, True])


class ProjectIn(BaseModel):
    name: str = "untitled"
    design_mesh: MeshRef | None = None
    ref_models: list[RefModel] = Field(default_factory=list)
    grid: GridSpec = Field(default_factory=GridSpec)
    material: MaterialSpec = Field(default_factory=MaterialSpec)
    params: ParamsSpec = Field(default_factory=ParamsSpec)
    loads: list[LoadSpec] = Field(default_factory=list)
    supports: list[SupportSpec] = Field(default_factory=list)


class Project(ProjectIn):
    id: str
    created_at: str
    updated_at: str


class VoxelStats(BaseModel):
    nx: int
    ny: int
    nz: int
    h: float
    origin: Vec3
    n_active: int
    n_free: int
    n_passive_solid: int
    n_passive_void: int
    n_nodes: int
    n_dof: int
    est_bytes: int
    est_sec_per_iter: float
    warnings: list[str] = Field(default_factory=list)


class ResolvedNodes(BaseModel):
    count: int
    xyz: list[Vec3]  # capped at 5000 points for preview
    truncated: bool = False


class RunCreate(BaseModel):
    project_id: str


RunStatus = Literal["queued", "running", "done", "error", "cancelled"]


class IterationRecord(BaseModel):
    it: int
    compliance: float
    volume: float
    change: float
    t_iter: float
    stress_max: float | None = None
    constraint: float | None = None


class RunInfo(BaseModel):
    id: str
    project_id: str
    status: RunStatus
    created_at: str
    finished_at: str | None = None
    history: list[IterationRecord] = Field(default_factory=list)
    stats: VoxelStats | None = None
    error: str | None = None
    outcome: Literal["converged", "max_iter", "cancelled", "error"] | None = None
    message: str | None = None  # e.g. "converged after 37 iterations"


# ---- WebSocket messages (JSON text frames). Density frames are BINARY frames, see CLAUDE.md. ----
class ProgressMsg(IterationRecord):
    type: Literal["progress"] = "progress"


class StatusMsg(BaseModel):
    type: Literal["started", "done", "error", "cancelled"]
    message: str | None = None
    run: RunInfo | None = None


WsMessage = Annotated[ProgressMsg | StatusMsg, Field(discriminator="type")]


class RunExport(BaseModel):
    """GET /api/runs/{id}/project.json: everything needed to reload or re-run headlessly."""

    project: Project
    run: RunInfo


class ErrorResponse(BaseModel):
    detail: str
