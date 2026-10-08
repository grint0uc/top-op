"""STEP (ISO 10303-21) import through OpenCascade, with the B-rep faces kept as facets.

Optional: needs `OCP` (`uv sync --extra step`, or `pip install "topop[step]"`). Everything except
`load_step` also works without it, so a tessellation cached by the server loads without OpenCascade.
Lengths are whatever OpenCascade returns for the file, i.e. millimetres unless the STEP reader is
configured otherwise; nothing is rescaled here.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

INSTALL_HINT = (
    "STEP import needs OpenCascade: run `uv sync --extra step` "
    '(or `pip install "topop[step]"`) and retry'
)
META_FACETS = "step_facets"  # mesh.metadata: FacetInfo dicts, area-sorted, id = rank
META_FACE_TO_FACET = "step_facet_of_face"  # mesh.metadata: facet id of every triangle

_HEADER = b"ISO-10303-21"
_EXTENSIONS = {".step", ".stp"}
_BOM = b"\xef\xbb\xbf \t\r\n"


def is_step(src: str | bytes | os.PathLike) -> bool:
    """Bytes: starts with the `ISO-10303-21` header. Path: .step/.stp, or an existing file with it."""
    if isinstance(src, bytes | bytearray | memoryview):
        return bytes(src[:64]).lstrip(_BOM).startswith(_HEADER)
    path = Path(os.fspath(src))
    if path.suffix.lower() in _EXTENSIONS:
        return True
    try:
        with path.open("rb") as fh:
            return fh.read(64).lstrip(_BOM).startswith(_HEADER)
    except OSError:
        return False


@dataclass
class StepMesh:
    mesh: trimesh.Trimesh  # merged vertices, no degenerate triangles (what `load_mesh` returns)
    brep_faces: np.ndarray  # (n_triangles,) B-rep face index of every triangle
    # one dict per B-rep face (list index = brep_face): brep_face, kind plane|cylinder|other, area,
    # normal (outward, plane only), centroid, bbox, axis + radius (cylinder only)
    faces: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        facets, face_to_facet = self.facets()
        self.mesh.metadata[META_FACETS] = facets
        self.mesh.metadata[META_FACE_TO_FACET] = face_to_facet

    def facets(self) -> tuple[list[dict], np.ndarray]:
        """B-rep faces as `FacetInfo` dicts (area descending, id = rank) + triangle -> facet id.

        Same shape as `selection.compute_facets`. Ties (relative area to 1e-9) go to the lower
        B-rep face index, so ids depend only on the file.
        """
        n = len(self.faces)
        count = np.bincount(self.brep_faces, minlength=n)
        area = np.array([f["area"] for f in self.faces], dtype=np.float64)
        total = max(float(area.sum()), np.finfo(float).tiny)
        order = np.lexsort((np.arange(n), -np.round(area / total, 9)))
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        keys = ("area", "normal", "centroid", "bbox", "kind", "axis", "radius", "brep_face")
        facets = [
            {"id": r, "n_faces": int(count[b]), **{k: self.faces[b][k] for k in keys}}
            for r, b in enumerate(order.tolist())
        ]
        return facets, rank[self.brep_faces]

    def to_npz_bytes(self) -> bytes:
        """Exact tessellation + face table (float64, unlike an STL), so reloads keep ids aligned."""
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            vertices=np.asarray(self.mesh.vertices, dtype=np.float64),
            triangles=np.asarray(self.mesh.faces, dtype=np.int64),
            brep_faces=self.brep_faces.astype(np.int32),
            faces_json=np.array(json.dumps(self.faces)),
        )
        return buf.getvalue()

    @classmethod
    def from_npz_bytes(cls, data: bytes) -> StepMesh:
        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                mesh = trimesh.Trimesh(z["vertices"], z["triangles"], process=False)
                brep = np.asarray(z["brep_faces"], dtype=np.int64)
                faces = json.loads(str(z["faces_json"]))
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"unreadable STEP tessellation cache: {exc}") from exc
        if len(brep) != len(mesh.faces) or (len(brep) and brep.max() >= len(faces)):
            raise ValueError("STEP tessellation cache is inconsistent")
        return cls(mesh, brep, faces)


def facet_triangles(
    mesh: trimesh.Trimesh, facet_ids: Sequence[int] | np.ndarray
) -> np.ndarray | None:
    """Sorted triangle ids of the B-rep faces `facet_ids` of a STEP-sourced mesh; None for any
    other mesh. ValueError on ids outside the facet table."""
    facets = mesh.metadata.get(META_FACETS)
    if facets is None:
        return None
    want = np.asarray(facet_ids, dtype=np.int64).ravel()
    bad = want[(want < 0) | (want >= len(facets))]
    if bad.size:
        raise ValueError(f"unknown facet ids {bad.tolist()} (mesh has {len(facets)} facets)")
    return np.flatnonzero(np.isin(mesh.metadata[META_FACE_TO_FACET], want)).astype(np.int64)


# ---- OpenCascade --------------------------------------------------------------------------------


def _require_occ() -> None:
    try:
        import OCP  # noqa: F401
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc


def _read_shape(src: str | bytes | os.PathLike):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    tmp: str | None = None
    if isinstance(src, bytes | bytearray | memoryview):
        if len(src) == 0:
            raise ValueError("empty STEP data")
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as fh:
            fh.write(bytes(src))  # OpenCascade reads from a path
            tmp = fh.name
        path = tmp
    else:
        p = Path(os.fspath(src))
        if not p.is_file():
            raise ValueError(f"STEP file not found: {p}")
        path = str(p)
    try:
        reader = STEPControl_Reader()
        if reader.ReadFile(path) != IFSelect_RetDone:
            raise ValueError("could not read STEP file (not valid ISO 10303-21?)")
        if reader.TransferRoots() == 0 or reader.NbShapes() == 0:
            raise ValueError("STEP file contains no shapes")
        return reader.OneShape()
    except ValueError:
        raise
    except Exception as exc:  # OpenCascade raises Standard_Failure subclasses for bad input
        raise ValueError(f"could not read STEP file: {exc}") from exc
    finally:
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)


def _diagonal(shape) -> float:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    if box.IsVoid():
        raise ValueError("STEP shape is empty")
    x0, y0, z0, x1, y1, z1 = box.Get()
    return float(np.linalg.norm([x1 - x0, y1 - y0, z1 - z0]))


def _face_properties(face, nodes: np.ndarray) -> dict:
    """B-rep face -> table row (everything but the triangles)."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_REVERSED

    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    c = props.CentreOfMass()
    surface = BRepAdaptor_Surface(face)
    gtype = surface.GetType()
    kind, normal, axis, radius = "other", [0.0, 0.0, 0.0], None, None
    if gtype == GeomAbs_Plane:
        kind = "plane"
        ax = surface.Plane().Position()
        d = ax.Direction()
        sign = (1.0 if ax.Direct() else -1.0) * (
            -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
        )
        normal = [sign * d.X(), sign * d.Y(), sign * d.Z()]
    elif gtype == GeomAbs_Cylinder:
        kind = "cylinder"
        cyl = surface.Cylinder()
        d = cyl.Axis().Direction()
        axis, radius = [d.X(), d.Y(), d.Z()], float(cyl.Radius())
    return {
        "kind": kind,
        "area": float(props.Mass()),
        "normal": [float(x) + 0.0 for x in normal],  # + 0.0: no -0.0
        "centroid": [c.X(), c.Y(), c.Z()],
        "bbox": [nodes.min(0).tolist(), nodes.max(0).tolist()] if len(nodes) else _occ_bbox(face),
        "axis": axis,
        "radius": radius,
    }


def _occ_bbox(face) -> list[list[float]]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(face, box)
    x0, y0, z0, x1, y1, z1 = box.Get()
    return [[x0, y0, z0], [x1, y1, z1]]


def _merge_vertices(
    vertices: np.ndarray, triangles: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Weld vertices closer than `eps` (face meshes share edge nodes only up to rounding)."""
    pairs = cKDTree(vertices).query_pairs(eps, output_type="ndarray")
    n = len(vertices)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_groups, label = connected_components(graph, directed=False)
    first = np.full(n_groups, n, dtype=np.int64)
    np.minimum.at(first, label, np.arange(n))
    return vertices[first], label[triangles]


def load_step(
    src: str | bytes | os.PathLike,
    tolerance: float | None = None,
    angular_tolerance_deg: float = 5.0,
) -> StepMesh:
    """Read a STEP file (path, or the file's bytes) and tessellate every B-rep face.

    `tolerance` is the chordal deviation in model units; None (default) picks 1e-3 of the bbox
    diagonal, at least 0.01. Units are those OpenCascade reports for the file (mm by default) and
    are not converted. ImportError (with an install hint) if OpenCascade is missing; ValueError if
    the file cannot be read or has no surface.
    """
    _require_occ()
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape

    shape = _read_shape(src)
    diag = _diagonal(shape)
    tol = max(1e-3 * diag, 0.01) if tolerance is None else float(tolerance)
    if tol <= 0:
        raise ValueError("tolerance must be positive")
    BRepMesh_IncrementalMesh(shape, tol, False, np.radians(float(angular_tolerance_deg)), True)

    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)  # unique faces, in a file-determined order
    if face_map.Extent() == 0:
        raise ValueError("STEP file contains no faces")

    vert_parts: list[np.ndarray] = []
    tri_parts: list[np.ndarray] = []
    owner_parts: list[np.ndarray] = []
    rows: list[dict] = []
    offset = 0
    for i in range(1, face_map.Extent() + 1):
        face = TopoDS.Face_s(face_map.FindKey(i))
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face, loc)
        nodes = np.zeros((0, 3))
        if poly is not None and poly.NbTriangles() > 0:
            trsf = loc.Transformation()
            nodes = np.array(
                [
                    (p.X(), p.Y(), p.Z())
                    for p in (poly.Node(k).Transformed(trsf) for k in range(1, poly.NbNodes() + 1))
                ]
            )
            tris = np.array(
                [poly.Triangle(k).Get() for k in range(1, poly.NbTriangles() + 1)], dtype=np.int64
            )
            tris -= 1
            if (face.Orientation() == TopAbs_REVERSED) != trsf.IsNegative():
                tris = tris[:, [0, 2, 1]]  # keep triangles counter-clockwise seen from outside
            vert_parts.append(nodes)
            tri_parts.append(tris + offset)
            owner_parts.append(np.full(len(tris), i - 1, dtype=np.int64))
            offset += len(nodes)
        rows.append({"brep_face": i - 1, **_face_properties(face, nodes)})
    if not tri_parts:
        raise ValueError("tessellation produced no triangles")

    vertices, triangles = _merge_vertices(
        np.concatenate(vert_parts), np.concatenate(tri_parts), 1e-7 * max(diag, 1e-9)
    )
    owner = np.concatenate(owner_parts)
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    keep = mesh.nondegenerate_faces()
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    owner = owner[keep]
    if len(mesh.faces) == 0:
        raise ValueError("mesh has no (non-degenerate) faces")
    return StepMesh(mesh, owner, rows)
