"""Reference problems on full boxes (h=1, origin 0) for regression tests and timings."""

from __future__ import annotations

import numpy as np

from topop.core.problem import Grid, Load, Material, Problem, RunParams, Support


def _plane_nodes(grid: Grid, ix=None, iy=None, iz=None) -> np.ndarray:
    axes = [
        np.arange(n) if v is None else np.atleast_1d(v)
        for v, n in zip((ix, iy, iz), grid.node_shape)
    ]
    a, b, c = np.meshgrid(*axes, indexing="ij")
    return grid.node_ids(a.ravel(), b.ravel(), c.ravel()).astype(np.int64)


def _box(nelx: int, nely: int, nelz: int, E: float, nu: float) -> Problem:
    grid = Grid(origin=(0.0, 0.0, 0.0), h=1.0, shape=(nelx, nely, nelz))
    return Problem(
        grid=grid,
        active=np.ones(grid.shape, dtype=bool),
        passive=np.zeros(grid.shape, dtype=np.int8),
        material=Material(E=E, nu=nu),
        supports=[Support(_plane_nodes(grid, ix=0))],
    )


def cantilever(nelx: int = 60, nely: int = 20, nelz: int = 4, E: float = 1.0, nu: float = 0.3):
    """Liu & Tovar top3d cantilever: x=0 clamped, total (0,-1,0) on the free-end bottom edge."""
    p = _box(nelx, nely, nelz, E, nu)
    p.loads = [Load(_plane_nodes(p.grid, ix=nelx, iy=0), (0.0, -1.0, 0.0))]
    return p


def cantilever_params() -> RunParams:
    return RunParams(volfrac=0.3, penal=3, rmin=1.5, max_iter=100, tol=0.01)


def tip_loaded_beam(
    nelx: int = 40, nely: int = 4, nelz: int = 4, E: float = 1.0, nu: float = 0.3
) -> Problem:
    """x=0 clamped, total (0,-1,0) split equally over the nodes of the x=nelx face."""
    p = _box(nelx, nely, nelz, E, nu)
    p.loads = [Load(_plane_nodes(p.grid, ix=nelx), (0.0, -1.0, 0.0))]
    return p
