"""Result export (iso-surface STL, VTI, NPZ) and a headless numpy renderer for previews."""

from __future__ import annotations

import base64
import io
import warnings
from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import trimesh

from topop.core.problem import Grid
from topop.core.voxelize import covered_samples

# ---- iso-surface ---------------------------------------------------------------------------------


def density_to_mesh(
    rho: np.ndarray, grid: Grid, threshold: float = 0.5, smooth_iters: int = 0
) -> trimesh.Trimesh:
    """Marching-cubes iso-surface of the element densities in world coordinates.

    Densities live at element centers. With one zero layer of padding, padded index j is element
    j-1 whose center is origin + h*(j - 1 + 0.5), so world = origin + h*(v - 0.5).
    """
    from skimage.measure import marching_cubes

    rho = np.nan_to_num(np.asarray(rho, dtype=np.float32))
    if rho.shape != grid.shape:
        raise ValueError(f"rho shape {rho.shape} != grid shape {grid.shape}")
    if not threshold > 0:
        raise ValueError("threshold must be > 0")
    if rho.size == 0 or not float(rho.max()) > threshold:
        return trimesh.Trimesh()
    padded = np.pad(rho, 1, constant_values=0.0)
    with warnings.catch_warnings():  # skimage vs numpy>=2.5 shape-setter deprecation noise
        warnings.simplefilter("ignore", DeprecationWarning)
        verts, faces, _, _ = marching_cubes(padded, level=float(threshold), allow_degenerate=False)
    verts = np.asarray(grid.origin, dtype=np.float64) + grid.h * (verts.astype(np.float64) - 0.5)
    mesh = trimesh.Trimesh(verts, faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    if smooth_iters > 0 and len(mesh.faces):
        trimesh.smoothing.filter_laplacian(mesh, iterations=int(smooth_iters))
    return mesh


def to_stl_bytes(mesh: trimesh.Trimesh) -> bytes:
    """Binary STL."""
    return trimesh.exchange.stl.export_stl(mesh)


def _closed(mesh: trimesh.Trimesh | None) -> bool:
    return (
        mesh is not None
        and len(mesh.faces) > 0
        and mesh.is_watertight
        and mesh.is_winding_consistent
    )


def trim_to_design(
    result: trimesh.Trimesh, design: trimesh.Trimesh
) -> tuple[trimesh.Trimesh, list[str]]:
    """result ∩ design (manifold boolean), in the same (world) coordinates.

    The exported part then never pokes outside the CAD surface and keeps the exact CAD skin
    where the density stayed full. Never raises: if an input is not a closed, consistently
    oriented surface, the intersection is empty or the engine fails, returns `result`
    unchanged plus a warning.
    """
    for name, mesh in (("result", result), ("design", design)):
        if not _closed(mesh):
            return result, [f"not trimmed to the design: the {name} mesh is not watertight"]
    a, b = result, design
    if a.volume < 0:  # inside-out but closed: manifold needs outward normals
        a = a.copy()
        a.invert()
    if b.volume < 0:
        b = b.copy()
        b.invert()
    try:
        out = trimesh.boolean.intersection([a, b], engine="manifold")
    except Exception as exc:  # noqa: BLE001 - engine errors have no common base class
        return result, [f"not trimmed to the design: boolean failed ({type(exc).__name__}: {exc})"]
    if not _closed(out) or not out.volume > 0:
        return result, ["not trimmed to the design: the intersection is empty or not watertight"]
    return out, []


# ---- VTK ImageData -------------------------------------------------------------------------------


def _vtk_b64(arr: np.ndarray) -> str:
    """VTK inline binary, uncompressed: base64(UInt64 byte count + raw little-endian data)."""
    raw = np.ascontiguousarray(arr).tobytes()
    return base64.b64encode(np.uint64(len(raw)).tobytes() + raw).decode("ascii")


def to_vti_bytes(
    rho: np.ndarray, passive: np.ndarray, grid: Grid, stress: np.ndarray | None = None
) -> bytes:
    """VTK XML ImageData with CellData 'density' (Float32), 'passive' (Int8) and, when given,
    'stress' (Float32, von Mises at the element centers)."""
    if np.shape(rho) != grid.shape or np.shape(passive) != grid.shape:
        raise ValueError("rho/passive shape does not match grid")
    if stress is not None and np.shape(stress) != grid.shape:
        raise ValueError("stress shape does not match grid")
    # VTK cell order is x-fastest, i.e. Fortran order of our [ix, iy, iz] arrays
    dens = np.asarray(rho, dtype="<f4").ravel(order="F")
    pas = np.asarray(passive, dtype=np.int8).ravel(order="F")
    stress_xml = ""
    if stress is not None:
        sig = np.nan_to_num(np.asarray(stress, dtype="<f4")).ravel(order="F")
        stress_xml = (
            '        <DataArray type="Float32" Name="stress" NumberOfComponents="1" '
            f'format="binary" RangeMin="{float(sig.min(initial=0))!r}" '
            f'RangeMax="{float(sig.max(initial=0))!r}">\n'
            f"          {_vtk_b64(sig)}\n        </DataArray>\n"
        )
    nx, ny, nz = grid.shape
    ox, oy, oz = (repr(float(v)) for v in grid.origin)
    h = repr(float(grid.h))
    ext = f"0 {nx} 0 {ny} 0 {nz}"
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
        f'  <ImageData WholeExtent="{ext}" Origin="{ox} {oy} {oz}" Spacing="{h} {h} {h}" '
        'Direction="1 0 0 0 1 0 0 0 1">\n'
        f'    <Piece Extent="{ext}">\n'
        "      <PointData>\n      </PointData>\n"
        '      <CellData Scalars="density">\n'
        '        <DataArray type="Float32" Name="density" NumberOfComponents="1" '
        f'format="binary" RangeMin="{float(dens.min(initial=0))!r}" '
        f'RangeMax="{float(dens.max(initial=0))!r}">\n'
        f"          {_vtk_b64(dens)}\n        </DataArray>\n"
        '        <DataArray type="Int8" Name="passive" NumberOfComponents="1" format="binary">\n'
        f"          {_vtk_b64(pas)}\n        </DataArray>\n"
        f"{stress_xml}"
        "      </CellData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n"
    )
    return xml.encode("ascii")


# ---- NPZ -----------------------------------------------------------------------------------------


def to_npz_bytes(
    rho: np.ndarray,
    grid: Grid,
    active: np.ndarray,
    passive: np.ndarray,
    stress: np.ndarray | None = None,
) -> bytes:
    """Density archive; `stress` (von Mises per element) is stored under the key 'stress'."""
    extra = {} if stress is None else {"stress": np.asarray(stress, dtype=np.float64)}
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        rho=np.asarray(rho, dtype=np.float64),
        active=np.asarray(active, dtype=bool),
        passive=np.asarray(passive, dtype=np.int8),
        origin=np.asarray(grid.origin, dtype=np.float64),
        h=np.float64(grid.h),
        shape=np.asarray(grid.shape, dtype=np.int64),
        **extra,
    )
    return buf.getvalue()


@overload
def from_npz_bytes(
    data: bytes, with_stress: Literal[False] = False
) -> tuple[np.ndarray, Grid, np.ndarray, np.ndarray]: ...
@overload
def from_npz_bytes(
    data: bytes, with_stress: Literal[True]
) -> tuple[np.ndarray, Grid, np.ndarray, np.ndarray, np.ndarray | None]: ...
def from_npz_bytes(data: bytes, with_stress: bool = False) -> tuple:
    """Inverse of `to_npz_bytes`: (rho, grid, active, passive), plus the stress array (None when
    the archive has none) as a fifth item with `with_stress=True`."""
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        grid = Grid(
            origin=tuple(float(v) for v in z["origin"]),
            h=float(z["h"]),
            shape=tuple(int(v) for v in z["shape"]),
        )
        out = (z["rho"], grid, z["active"], z["passive"])
        if not with_stress:
            return out
        return (*out, z["stress"] if "stress" in z.files else None)


# ---- headless renderer ---------------------------------------------------------------------------

# camera position direction (from the target) and screen-up vector per view
VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "iso": ((-1.0, -1.0, 1.0), (0.0, 0.0, 1.0)),
    "+x": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "-x": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "+y": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "-y": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "+z": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "-z": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}
BACKGROUND = np.array([0.97, 0.97, 0.98])


def _camera(view: str) -> np.ndarray:
    """(3,3) rows = screen right, screen up, toward-camera."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(VIEWS)}")
    d, up = (np.asarray(v, dtype=np.float64) for v in VIEWS[view])
    back = d / np.linalg.norm(d)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(back, right), back])


def _rasterize(
    tri_px: np.ndarray, depth: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer triangles. tri_px (F,3,2) pixel coords, depth (F,3), larger = closer.

    Returns (zbuf (H*W,), visible face per pixel (H*W,), -1 = empty).
    """
    n_pix = width * height
    zbuf = np.full(n_pix, -np.inf)
    owner = np.full(n_pix, -1, dtype=np.int64)
    x, y = tri_px[..., 0], tri_px[..., 1]
    area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    fids = np.flatnonzero(np.abs(area) > 1e-12)
    if fids.size == 0:
        return zbuf, owner
    x, y, area = x[fids], y[fids], area[fids]
    # depth as an affine function of the pixel position: barycentric(k) = a*px + b*py + c
    i1, i2 = np.array([1, 2, 0]), np.array([2, 0, 1])
    xa, ya, xb, yb = x[:, i1], y[:, i1], x[:, i2], y[:, i2]
    dz = depth[fids] / area[:, None]
    za, zb, zc = ((ya - yb) * dz).sum(1), ((xb - xa) * dz).sum(1), ((xa * yb - xb * ya) * dz).sum(1)

    cand_pix, cand_z, cand_f = [], [], []
    for f, px, py in covered_samples(x, y, width, height, eps=1e-6):
        z = za[f] * (px + 0.5) + zb[f] * (py + 0.5) + zc[f]
        pix = py * width + px
        np.maximum.at(zbuf, pix, z)
        cand_pix.append(pix)
        cand_z.append(z)
        cand_f.append(f)
    if not cand_pix:
        return zbuf, owner
    pix, z, f = (np.concatenate(c) for c in (cand_pix, cand_z, cand_f))
    win = z >= zbuf[pix]
    owner[pix[win]] = fids[f[win]]
    return zbuf, owner


def _draw_line(
    img: np.ndarray,
    front: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    d0: float,
    d1: float,
    rgb: Sequence[float],
    a: float,
) -> None:
    """Blend a 1-px line into img (H,W,3), hidden where the surface depth `front` is closer."""
    h, w = img.shape[:2]
    n = int(np.ceil(np.abs(p1 - p0).max())) + 1
    t = np.linspace(0.0, 1.0, max(n, 2))
    pts = np.floor(p0[None] + t[:, None] * (p1 - p0)[None]).astype(np.int64)
    d = d0 + t * (d1 - d0)
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts, d = pts[ok], d[ok]
    vis = d >= front[pts[:, 1], pts[:, 0]]
    py, px = pts[vis, 1], pts[vis, 0]
    img[py, px] = (1 - a) * img[py, px] + a * np.asarray(rgb)


def render_png(
    layers: Sequence[tuple[trimesh.Trimesh, tuple[float, float, float], float]],
    view: str = "iso",
    size: tuple[int, int] = (900, 700),
    bounds: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> bytes:
    """Orthographic flat-shaded PNG of (mesh, rgb 0..1, alpha) layers. No GPU, no display."""
    from PIL import Image, ImageDraw

    width, height = int(size[0]), int(size[1])
    cam = _camera(view)
    layers = [(m, c, a) for m, c, a in layers if m is not None and len(m.faces)]
    if bounds is None:
        if layers:
            bb = np.stack([np.asarray(m.bounds) for m, _, _ in layers])
            bounds = np.stack([bb[:, 0].min(0), bb[:, 1].max(0)])
        else:
            bounds = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    bounds = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
    corners = np.array(
        [[bounds[i, 0], bounds[j, 1], bounds[k, 2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    )
    center = bounds.mean(0)
    sc = (corners - center) @ cam[:2].T
    margin = 0.08
    extent = np.maximum(np.abs(sc).max(0), 1e-12)
    scale = min(
        width * (1 - 2 * margin) / (2 * extent[0]), height * (1 - 2 * margin) / (2 * extent[1])
    )

    def project(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = (np.asarray(p, dtype=np.float64) - center) @ cam.T
        px = np.stack([width / 2 + scale * q[..., 0], height / 2 - scale * q[..., 1]], -1)
        return px, q[..., 2]

    # camera-relative light: the three faces visible in the iso view get distinct shades
    light = 0.12 * cam[0] + 0.24 * cam[1] + 0.96 * cam[2]
    light /= np.linalg.norm(light)
    n_pix = width * height
    depths, colours, alphas = [], [], []
    for mesh, rgb, alpha in layers:
        px, dz = project(mesh.vertices)
        lam = np.abs(np.asarray(mesh.face_normals) @ light)  # two-sided Lambert
        shade = (0.35 + 0.65 * lam)[:, None] * np.asarray(rgb, dtype=np.float64)[None]
        zbuf, owner = _rasterize(px[mesh.faces], dz[mesh.faces], width, height)
        col = np.zeros((n_pix, 3))
        hit = owner >= 0
        col[hit] = shade[owner[hit]]
        depths.append(zbuf)
        colours.append(col)
        alphas.append(float(np.clip(alpha, 0.0, 1.0)))

    img = np.tile(BACKGROUND, (n_pix, 1))
    front = np.full(n_pix, -np.inf)
    if layers:
        # later layers win on coincident surfaces (e.g. a result lying on the design boundary)
        bias = 1e-6 * float(np.ptp(corners @ cam[2]) or 1.0)
        dstack = np.stack(depths) + bias * np.arange(len(layers))[:, None]  # (L, n_pix)
        cstack = np.stack(colours)
        astack = np.asarray(alphas)
        order = np.argsort(dstack, axis=0)  # back to front
        for rank in range(len(layers)):
            li = order[rank]
            d = np.take_along_axis(dstack, li[None], 0)[0]
            hit = np.isfinite(d)
            a = astack[li][:, None]
            c = cstack[li, np.arange(n_pix)]
            img[hit] = (a * c + (1 - a) * img)[hit]
            front = np.where(hit, d, front)
        # outline silhouettes and depth discontinuities so flat faces stay readable
        f2 = front.reshape(height, width)
        span = float(np.ptp(sc) * scale) or 1.0
        fin = np.where(np.isfinite(f2), f2 * scale, -1e6)
        edge = np.zeros((height, width), dtype=bool)
        edge[:, 1:] |= np.abs(np.diff(fin, axis=1)) > 0.02 * span
        edge[1:, :] |= np.abs(np.diff(fin, axis=0)) > 0.02 * span
        img = img.reshape(height, width, 3)
        img[edge] *= 0.45
    img = img.reshape(height, width, 3)

    # bbox frame, depth-tested against the surfaces
    cpx, cdz = project(corners)
    front2 = front.reshape(height, width)
    for i in range(8):
        for j in range(i + 1, 8):
            if (i ^ j).bit_count() == 1:
                _draw_line(img, front2, cpx[i], cpx[j], cdz[i], cdz[j], (0.4, 0.4, 0.45), 0.6)
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    # axis triad, bottom-left
    o = np.array([42.0, height - 42.0])
    for k, (lbl, rgb) in enumerate(
        [("X", (220, 50, 50)), ("Y", (40, 160, 60)), ("Z", (50, 90, 220))]
    ):
        v = np.array([cam[0, k], -cam[1, k]])
        if np.hypot(*v) < 1e-6:
            draw.ellipse([o[0] - 3, o[1] - 3, o[0] + 3, o[1] + 3], outline=rgb, width=2)
            draw.text((float(o[0]) + 5, float(o[1]) + 3), lbl, fill=rgb)
            continue
        tip = o + 30 * v
        lbl_at = tip + 7 * v / np.hypot(*v) - (3, 5)
        draw.line([(float(o[0]), float(o[1])), (float(tip[0]), float(tip[1]))], fill=rgb, width=2)
        draw.text((float(lbl_at[0]), float(lbl_at[1])), lbl, fill=rgb)
    dims = bounds[1] - bounds[0]
    draw.text(
        (8, 6),
        f"view {view}   bbox {dims[0]:.4g} x {dims[1]:.4g} x {dims[2]:.4g}   "
        f"min ({bounds[0, 0]:.4g}, {bounds[0, 1]:.4g}, {bounds[0, 2]:.4g})",
        fill=(60, 60, 70),
    )
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
