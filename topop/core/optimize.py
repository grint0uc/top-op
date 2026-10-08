"""SIMP compliance minimization with OC updates on the free cells of a voxel problem."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial

import numpy as np

from topop.core.fem import Assembler, rigid_body_modes
from topop.core.filters import DensityFilter, heaviside
from topop.core.problem import IterationInfo, Problem, ProgressCallback, Result, RunParams
from topop.core.solver import LinearSolver

BETA_MAX = 64.0
CONTINUATION_ITERS = 20
# loose CG rtol while the design moves a lot (change >= 0.1), 1e-6 once change <= 0.02
ADAPTIVE_TOL: float | None = 1e-4


def _linear_volume(v0: float, dv: np.ndarray, x0: np.ndarray, x: np.ndarray) -> float:
    return v0 + float(dv @ (x - x0))


def oc_update(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    volfrac: float,
    move: float,
    volume: Callable[[np.ndarray], float] | None = None,
) -> np.ndarray:
    """Optimality criteria step x * sqrt(-dc / (lam dv)), clipped to the move limit and [0, 1].

    lam is found by bisection (in log space) so that volume(x_new) == volfrac;
    `volume` defaults to mean(x_new).
    """
    x = np.asarray(x, dtype=np.float64)
    if volume is None:
        volume = np.mean
    lo_x = np.maximum(0.0, x - move)
    hi_x = np.minimum(1.0, x + move)
    ratio = np.maximum(-np.asarray(dc, dtype=np.float64), 0.0) / np.maximum(dv, 1e-300)
    scale = float(ratio.mean()) if ratio.any() else 1.0

    def step(lam: float) -> np.ndarray:
        return np.clip(x * np.sqrt(ratio / lam), lo_x, hi_x)

    lo, hi = np.log(scale) - 40.0, np.log(scale) + 40.0  # volume(step) decreases with lam
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        x_new = step(np.exp(mid))
        v = volume(x_new)
        if abs(v - volfrac) < 1e-9 or hi - lo < 1e-12:
            break
        if v > volfrac:
            lo = mid
        else:
            hi = mid
    return x_new


def optimize(
    problem: Problem,
    params: RunParams,
    callback: ProgressCallback | None = None,
    x0: np.ndarray | None = None,
    cancel: Callable[[], bool] | None = None,
    *,
    solver_options: dict | None = None,
) -> Result:
    """SIMP + OC. `solver_options` are extra `LinearSolver` keyword arguments (tests, benchmarks)."""
    issues = problem.validate()
    if issues:
        raise ValueError("invalid problem: " + "; ".join(issues))
    dtype = np.dtype(params.dtype)
    need = Assembler.estimate_bytes(problem.n_active, dtype)
    if need > params.memory_cap_bytes:
        raise MemoryError(
            f"{problem.n_active} active elements need about {need / 1e9:.2f} GB, above the "
            f"{params.memory_cap_bytes / 1e9:.2f} GB cap; lower the resolution"
        )

    shape = problem.grid.shape
    active = np.asarray(problem.active, dtype=bool)
    free = problem.free
    solid = active & (problem.passive == 1)
    void = active & (problem.passive == -1)
    n_free = int(free.sum())
    E0 = float(problem.material.E)
    Emin = problem.material.emin_ratio * E0

    asm = Assembler(problem, dtype=dtype)
    eids = asm.element_ids
    filt = DensityFilter(shape, params.rmin, active)
    opts = {
        "prolongators": asm.prolongators,
        "ordering": asm.band_ordering,
        "adaptive_tol": ADAPTIVE_TOL,
        **(solver_options or {}),
    }
    solver = LinearSolver(params.solver, **opts)
    rigid = (
        rigid_body_modes(asm.node_coords_compressed)[asm.free]
        if solver.needs_rigid_modes(asm.n_free)
        else None
    )

    x = np.zeros(shape)
    x[free] = params.volfrac
    x[solid] = 1.0
    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape != tuple(shape):
            raise ValueError(f"x0 must have the grid shape {shape}, got {x0.shape}")
        x[free] = np.clip(x0[free], 0.0, 1.0)

    beta = 1.0 if params.heaviside else 0.0

    def physical(x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        xt = filt.apply(x)
        dproj = None
        if params.heaviside:
            xt, dproj = heaviside(xt, beta)
        xp = np.clip(xt, 0.0, 1.0)
        xp[solid] = 1.0
        xp[void] = 0.0
        xp[~active] = 0.0
        return xp, dproj

    def through_filter(g_phys: np.ndarray, dproj: np.ndarray | None) -> np.ndarray:
        g = np.where(free, g_phys, 0.0)  # passive cells' physical density does not move
        if dproj is not None:
            g = g * dproj
        return filt.apply_adjoint(g)[free]

    def projected_volume(x: np.ndarray, x_free: np.ndarray) -> float:
        xt = x.copy()
        xt[free] = x_free
        return float(physical(xt)[0][free].mean())

    dv_phys = free / n_free  # d mean(x_phys[free]) / d x_phys
    dv_linear = through_filter(dv_phys, None) if not params.heaviside else None

    xp, dproj = physical(x)
    U = None
    history: list[IterationInfo] = []
    status, message = "max_iter", ""
    beta_since, change = 0, 1.0
    dc_full = np.zeros(problem.grid.nel)
    for it in range(1, params.max_iter + 1):
        if cancel is not None and cancel():
            status, message = "cancelled", f"cancelled before iteration {it}"
            break
        t0 = time.perf_counter()
        if (
            params.heaviside
            and beta < BETA_MAX
            and ((beta_since >= 20 and change < 0.05) or beta_since >= 40)
        ):
            beta, beta_since = 2 * beta, 0
            xp, dproj = physical(x)
        beta_since += 1
        p = params.penal
        if params.continuation:
            p = 1.0 + (params.penal - 1.0) * min(1.0, (it - 1) / CONTINUATION_ITERS)

        xe = xp.ravel()[eids]
        K = asm.assemble(Emin + xe**p * (E0 - Emin))
        U, _ = solver.solve(K, asm.F_free, x0=U, rigid_modes=rigid, change=change)
        c = float(np.sum(asm.F_free * U))
        if not np.isfinite(c):
            status, message = "error", f"non-finite compliance at iteration {it}"
            break
        # dc/dx_phys = -p x^(p-1) (E0 - Emin) u_e^T k_e u_e
        dc_full[eids] = -p * xe ** (p - 1) * (E0 - Emin) * asm.element_energies(U)
        dc = through_filter(dc_full.reshape(shape), dproj)
        dv = dv_linear if dv_linear is not None else through_filter(dv_phys, dproj)

        xf = x[free]
        if params.heaviside:
            volume = partial(projected_volume, x)
        else:
            # mean(H x) over free cells is linear in x: this is exact, no filtering per bisection
            volume = partial(_linear_volume, float(xp[free].mean()), dv, xf)
        # OC oscillates on a sharp projection unless the step shrinks as beta grows
        move = params.move / np.sqrt(beta) if params.heaviside else params.move
        x_new = oc_update(xf, dc, dv, params.volfrac, move, volume=volume)
        change = float(np.abs(x_new - xf).max())
        x[free] = x_new
        xp, dproj = physical(x)
        info = IterationInfo(it, c, float(xp[free].mean()), change, time.perf_counter() - t0)
        history.append(info)

        if callback is not None:
            ret = callback(info, xp)
            if ret is not None and not ret:
                status, message = "cancelled", f"cancelled after iteration {it}"
                break
        settled = (not params.continuation or p >= params.penal) and (
            not params.heaviside or beta >= BETA_MAX
        )
        if change < params.tol and settled:
            status, message = "converged", f"converged after {it} iterations"
            break
    else:
        if params.max_iter > 0:
            message = f"stopped at max_iter={params.max_iter}"

    return Result(rho=xp, history=history, status=status, message=message)
