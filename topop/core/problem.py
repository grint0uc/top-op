"""Problem definition shared by core, server and CLI. Pure numpy. See CLAUDE.md for index conventions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Hex8 local node offsets in the order (-,-,-) (+,-,-) (+,+,-) (-,+,-) (-,-,+) (+,-,+) (+,+,+) (-,+,+)
HEX8_OFFSETS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.int64,
)


@dataclass(frozen=True)
class Grid:
    """Regular cubic voxel grid. Element (ix,iy,iz) spans [origin + h*ix, origin + h*(ix+1)] etc."""

    origin: tuple[float, float, float]
    h: float
    shape: tuple[int, int, int]

    @property
    def nx(self) -> int:
        return self.shape[0]

    @property
    def ny(self) -> int:
        return self.shape[1]

    @property
    def nz(self) -> int:
        return self.shape[2]

    @property
    def nel(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def node_shape(self) -> tuple[int, int, int]:
        return (self.nx + 1, self.ny + 1, self.nz + 1)

    @property
    def n_nodes(self) -> int:
        a, b, c = self.node_shape
        return a * b * c

    @property
    def bounds(self) -> np.ndarray:
        o = np.asarray(self.origin, dtype=np.float64)
        return np.stack([o, o + self.h * np.asarray(self.shape, dtype=np.float64)])

    def node_coords(self) -> np.ndarray:
        """(n_nodes, 3) float64, rows in C-order flat node id order."""
        ii, jj, kk = np.meshgrid(*[np.arange(n) for n in self.node_shape], indexing="ij")
        idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(np.float64)
        return np.asarray(self.origin, dtype=np.float64) + self.h * idx

    def element_centers(self) -> np.ndarray:
        """(nel, 3) float64, rows in C-order flat element id order."""
        ii, jj, kk = np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij")
        idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(np.float64)
        return np.asarray(self.origin, dtype=np.float64) + self.h * (idx + 0.5)

    def element_nodes(self) -> np.ndarray:
        """(nel, 8) int64 full-grid node ids per element, HEX8_OFFSETS order, C-order element ids."""
        ii, jj, kk = np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij")
        base = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)  # (nel, 3)
        corners = base[:, None, :] + HEX8_OFFSETS[None, :, :]  # (nel, 8, 3)
        return np.ravel_multi_index(
            (corners[..., 0], corners[..., 1], corners[..., 2]), self.node_shape
        ).astype(np.int64)

    def node_ids(self, ix, iy, iz) -> np.ndarray:
        return np.ravel_multi_index((ix, iy, iz), self.node_shape)

    def element_ids(self, ix, iy, iz) -> np.ndarray:
        return np.ravel_multi_index((ix, iy, iz), self.shape)

    @classmethod
    def from_bounds(cls, bounds: np.ndarray, elements_along_longest: int, padding: int = 1) -> Grid:
        """Fit a grid around (2,3) bounds; `padding` empty voxels on every side."""
        bounds = np.asarray(bounds, dtype=np.float64)
        extent = bounds[1] - bounds[0]
        h = float(extent.max() / elements_along_longest)
        if not np.isfinite(h) or h <= 0:
            raise ValueError("degenerate bounds")
        n_inner = np.maximum(np.ceil(extent / h - 1e-9).astype(int), 1)
        shape = tuple(int(n + 2 * padding) for n in n_inner)
        # center the model inside the (possibly slightly larger) grid
        slack = h * np.asarray(n_inner) - extent
        origin = bounds[0] - padding * h - slack / 2
        return cls(origin=tuple(float(v) for v in origin), h=h, shape=shape)


@dataclass(frozen=True)
class Material:
    E: float = 1.0
    nu: float = 0.3
    emin_ratio: float = 1e-9  # Emin = emin_ratio * E


@dataclass
class Load:
    nodes: np.ndarray  # int64 full-grid node ids
    force: tuple[float, float, float]  # TOTAL force, split equally over nodes
    case: int = 0


@dataclass
class Support:
    nodes: np.ndarray  # int64 full-grid node ids
    fix: tuple[bool, bool, bool] = (True, True, True)


@dataclass
class Problem:
    grid: Grid
    active: np.ndarray  # bool (nx,ny,nz)
    passive: np.ndarray  # int8 (nx,ny,nz): 0 free, 1 solid, -1 void
    material: Material = field(default_factory=Material)
    loads: list[Load] = field(default_factory=list)
    supports: list[Support] = field(default_factory=list)

    @property
    def free(self) -> np.ndarray:
        """bool (nx,ny,nz): design variables (active and not passive)."""
        return self.active & (self.passive == 0)

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    @property
    def n_cases(self) -> int:
        return 1 + max((ld.case for ld in self.loads), default=0)

    def active_node_mask(self) -> np.ndarray:
        """bool (n_nodes,): nodes touched by at least one active element."""
        en = self.grid.element_nodes()[self.active.ravel()]
        mask = np.zeros(self.grid.n_nodes, dtype=bool)
        mask[en.ravel()] = True
        return mask

    def validate(self) -> list[str]:
        """Return human-readable problems. Empty list == runnable."""
        issues: list[str] = []
        if self.active.shape != self.grid.shape or self.passive.shape != self.grid.shape:
            return ["active/passive shape does not match grid"]
        if self.n_active == 0:
            return ["no active elements (voxelization produced nothing)"]
        if not self.loads:
            issues.append("no loads defined")
        if not self.supports:
            issues.append("no supports defined")
        nm = self.active_node_mask()
        for i, ld in enumerate(self.loads):
            if ld.nodes.size == 0:
                issues.append(f"load {i} resolves to zero nodes")
            elif not nm[ld.nodes].all():
                issues.append(f"load {i} has nodes not attached to active elements")
        for i, sp in enumerate(self.supports):
            if sp.nodes.size == 0:
                issues.append(f"support {i} resolves to zero nodes")
            elif not nm[sp.nodes].all():
                issues.append(f"support {i} has nodes not attached to active elements")
            if not any(sp.fix):
                issues.append(f"support {i} fixes no DOF")
        if self.free.sum() == 0:
            issues.append("no free design elements (everything is passive)")
        return issues


SolverKind = Literal["auto", "amg", "direct"]
Axis = Literal["x", "y", "z"]
Direction = Literal["+x", "-x", "+y", "-y", "+z", "-z"]


@dataclass(frozen=True)
class SymmetryPlane:
    """Mirror densities about the plane `axis = position` (world units).

    position None -> center of the active region's bounding box. The plane is snapped to the
    nearest element boundary or element center so that mirrored cells land on the grid.
    """

    axis: Axis
    position: float | None = None


@dataclass
class RunParams:
    volfrac: float = 0.3
    penal: float = 3.0
    rmin: float = 2.0  # filter radius in voxels
    max_iter: int = 100
    tol: float = 0.01  # stop when max |delta rho| < tol
    move: float = 0.2
    heaviside: bool = False
    continuation: bool = False  # penal 1 -> penal over the first 20 iterations
    solver: SolverKind = "auto"
    dtype: Literal["float64", "float32"] = "float64"
    memory_cap_bytes: int = 6_000_000_000
    optimizer: Literal["oc", "mma"] = "oc"  # oc: volume constraint only; mma: any constraints
    symmetry: tuple[SymmetryPlane, ...] = ()
    # von Mises stress constraint (same units as E); None -> compliance-only. Forces optimizer=mma.
    stress_limit: float | None = None
    stress_pnorm: float = 8.0  # p-norm aggregation exponent
    # Additive-manufacturing overhang filter (Langelaar 2017, 45 deg): build direction, or None.
    overhang: Direction | None = None


@dataclass
class IterationInfo:
    it: int
    compliance: float
    volume: float  # fraction of free elements
    change: float
    t_iter: float
    stress_max: float | None = None  # max element von Mises (when computed)
    constraint: float | None = None  # stress constraint value g <= 0 (when active)


RunStatus = Literal["converged", "max_iter", "cancelled", "error"]


@dataclass
class Result:
    rho: np.ndarray  # (nx,ny,nz) float64, 0 on inactive elements
    history: list[IterationInfo]
    status: RunStatus
    message: str = ""
    stress: np.ndarray | None = None  # (nx,ny,nz) von Mises at element centers, 0 on inactive


# callback(info, rho) -> False to cancel. rho is the physical (filtered/projected) density, full grid.
ProgressCallback = Callable[[IterationInfo, np.ndarray], bool]
