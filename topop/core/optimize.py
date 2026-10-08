"""SIMP compliance minimization on the free cells of a voxel problem (OC or MMA updates).

Physical density chain: density filter -> AM overhang filter (optional, Langelaar 2017) ->
Heaviside projection (optional) -> passive cells fixed. Optional mirror symmetry of the design
variables and a p-norm von Mises stress constraint (qp-relaxation, adjoint sensitivities).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from topop.core.fem import Assembler, rigid_body_modes, von_mises
from topop.core.filters import DensityFilter, heaviside
from topop.core.mma import MMA
from topop.core.problem import (
    IterationInfo,
    Problem,
    ProgressCallback,
    Result,
    RunParams,
    SymmetryPlane,
)
from topop.core.solver import LinearSolver

log = logging.getLogger(__name__)

BETA_MAX = 64.0
CONTINUATION_ITERS = 20
# loose CG rtol while the design moves a lot (change >= 0.1), 1e-6 once change <= 0.02
ADAPTIVE_TOL: float | None = 1e-4
# Relaxed ("qp") stress rho_phys^STRESS_Q * sigma_vm(solid), Bruggi 2008 / Le et al. 2010: it
# vanishes in void cells (whose solid-material stress is meaningless) and equals the solid
# stress in solid cells. Used for Result.stress, IterationInfo.stress_max and the constraint.
STRESS_Q = 0.5
STRESS_NORM_ALPHA = 0.5  # adaptive p-norm normalization c_k (Le et al. 2010, eq. 15)
SYMMETRY_WARN_FRACTION = 0.05
RETRY_RESIDUAL = 1e-3  # re-solve from zero when a warm-started solve ends above this residual
MMA_FEAS_TOL = 1e-3  # MMA converges only once every constraint g_i <= this
STRESS_FEAS_TOL = 0.01


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


def _am_filter(active: np.ndarray, direction):
    try:
        from topop.core.filters import AMFilter
    except ImportError as exc:
        raise NotImplementedError("overhang filter not available") from exc
    return AMFilter(active, direction)


@dataclass
class Responses:
    """Responses at one physical density; gradients are with respect to x[free]."""

    compliance: float
    dc: np.ndarray
    volume: float  # mean physical density over the free cells
    dv: np.ndarray
    stress: np.ndarray  # (n_cases, nel_active) relaxed von Mises rho^q sigma_vm(solid)
    stress_max: float
    stress_pnorm: float | None = None  # (sum_ce stress^p)^(1/p) over cases and active cells
    dpn: np.ndarray | None = None  # its gradient


class SimpModel:
    """SIMP model of one problem: the density chain, FE solves and sensitivities.

    `physical(x)` stores the intermediates (projection slope, AM filter state) that
    `backprop` uses, so gradients refer to the design last passed to `physical`.
    """

    def __init__(self, problem: Problem, params: RunParams, solver_options: dict | None = None):
        self.problem = problem
        self.params = params
        self.shape = tuple(problem.grid.shape)
        self.active = np.asarray(problem.active, dtype=bool)
        self.free = problem.free
        self.solid = self.active & (problem.passive == 1)
        self.void = self.active & (problem.passive == -1)
        self.n_free = int(self.free.sum())
        self.E0 = float(problem.material.E)
        self.Emin = problem.material.emin_ratio * self.E0

        self.asm = asm = Assembler(problem, dtype=np.dtype(params.dtype))
        self.eids = asm.element_ids
        self.filt = DensityFilter(self.shape, params.rmin, self.active)
        self.am = _am_filter(self.active, params.overhang) if params.overhang else None
        opts = {
            "prolongators": asm.prolongators,
            "ordering": asm.band_ordering,
            "adaptive_tol": ADAPTIVE_TOL,
            **(solver_options or {}),
        }
        self.solver = LinearSolver(params.solver, **opts)
        self.rigid = (
            rigid_body_modes(asm.node_coords_compressed)[asm.free]
            if self.solver.needs_rigid_modes(asm.n_free)
            else None
        )
        self.beta = 1.0 if params.heaviside else 0.0
        self.U: np.ndarray | None = None  # last displacement (warm start)
        self.L: np.ndarray | None = None  # last stress adjoint (warm start)
        self._dproj: np.ndarray | None = None
        self._g_full = np.zeros(problem.grid.nel)
        self.dv_phys = self.free / self.n_free  # d mean(x_phys[free]) / d x_phys
        # mean(x_phys[free]) is linear in x without projection and AM filter
        self.linear_volume = not params.heaviside and self.am is None
        self.dv_linear = self.backprop(self.dv_phys) if self.linear_volume else None

    def physical(self, x: np.ndarray) -> np.ndarray:
        xt = self.filt.apply(x)
        if self.am is not None:
            xt[self.solid] = 1.0  # passive solids support the material printed on them
            xt[self.void] = 0.0
            xt = self.am.apply(xt)
        dproj = None
        if self.params.heaviside:
            xt, dproj = heaviside(xt, self.beta)
        xp = np.clip(xt, 0.0, 1.0)
        xp[self.solid] = 1.0
        xp[self.void] = 0.0
        xp[~self.active] = 0.0
        self._dproj = dproj
        return xp

    def backprop(self, g_phys: np.ndarray) -> np.ndarray:
        """d/dx_phys (full grid) -> d/dx[free] through the chain of the last `physical` call."""
        g = np.where(self.free, g_phys, 0.0)  # passive cells' physical density does not move
        if self._dproj is not None:
            g = g * self._dproj
        if self.am is not None:
            g = np.where(self.free, self.am.backprop(g), 0.0)
        return self.filt.apply_adjoint(g)[self.free]

    def projected_volume(self, x: np.ndarray, x_free: np.ndarray) -> float:
        """Exact volume of x with x[free] = x_free (overwrites the chain state)."""
        xt = x.copy()
        xt[self.free] = x_free
        return float(self.physical(xt)[self.free].mean())

    def solve(self, K, F: np.ndarray, x0: np.ndarray | None, change: float | None) -> np.ndarray:
        U, info = self.solver.solve(K, F, x0=x0, rigid_modes=self.rigid, change=change)
        if x0 is not None and info.residual > max(RETRY_RESIDUAL, 10 * info.rtol):
            # a warm-started multigrid CG can break down on extreme density contrast (the
            # preconditioner loses definiteness); a cold start recovers
            U, info = self.solver.solve(K, F, x0=None, rigid_modes=self.rigid, change=change)
        return U

    def _to_free(self, g_active: np.ndarray) -> np.ndarray:
        self._g_full[self.eids] = g_active
        return self.backprop(self._g_full.reshape(self.shape))

    def evaluate(
        self,
        xp: np.ndarray,
        penal: float,
        change: float | None = None,
        *,
        pnorm: float | None = None,
    ) -> Responses:
        """Solve at physical density `xp` (from `physical`); `pnorm` adds the stress p-norm."""
        asm, E0, Emin = self.asm, self.E0, self.Emin
        xe = xp.ravel()[self.eids]
        K = asm.assemble(Emin + xe**penal * (E0 - Emin))
        U = self.U = self.solve(K, asm.F_free, self.U, change)
        c = float(np.sum(asm.F_free * U))
        energies, sigma = asm.element_energies_and_stress(U)
        # dc/dx_phys = -p x^(p-1) (E0 - Emin) u_e^T k_e u_e
        dE = penal * xe ** (penal - 1) * (E0 - Emin)
        dc = self._to_free(-dE * energies)
        dv = self.dv_linear if self.dv_linear is not None else self.backprop(self.dv_phys)
        vm = von_mises(sigma)  # (n_cases, nel)
        rho_q = xe**STRESS_Q
        s = rho_q[None, :] * vm
        r = Responses(c, dc, float(xp[self.free].mean()), dv, s, float(s.max(initial=0.0)))
        if pnorm is None:
            return r

        # sigma_PN = (sum s^p)^(1/p), s = rho^q vm; scaled by max(s) against overflow
        p, q = float(pnorm), STRESS_Q
        smax = r.stress_max if r.stress_max > 0 else 1.0
        pn = smax * float(np.sum((s / smax) ** p)) ** (1.0 / p)
        t = (s / pn) ** (p - 1)  # d pn / d s
        # direct term d s / d rho = q rho^(q-1) vm (finite: t ~ rho^(q(p-1)), q p > 1)
        drho = np.divide(q * s, xe[None, :], out=np.zeros_like(s), where=xe[None, :] > 0)
        direct = np.sum(t * drho, axis=0)
        # adjoint: K lam = d pn / d U, d pn / d rho += -lam^T (dK/d rho) u
        rhs = asm.von_mises_gradient(sigma, t * rho_q[None, :])
        rhs = rhs.reshape(asm.F_free.shape)
        L = self.L = self.solve(K, rhs, self.L, change)
        indirect = -dE * asm.element_cross_energies(U, L)
        r.stress_pnorm = pn
        r.dpn = self._to_free(direct + indirect)
        return r

    def stress_field(self, xp: np.ndarray, penal: float) -> np.ndarray:
        """Full-grid relaxed von Mises (max over load cases) of `xp`: one more FE solve."""
        xe = xp.ravel()[self.eids]
        K = self.asm.assemble(self.Emin + xe**penal * (self.E0 - self.Emin))
        U = self.solve(K, self.asm.F_free, self.U, None)
        vm = von_mises(self.asm.element_stress(U)).reshape(-1, self.eids.size)
        out = np.zeros(self.problem.grid.nel)
        out[self.eids] = xe**STRESS_Q * vm.max(axis=0)
        return out.reshape(self.shape)


class Symmetry:
    """Mirror maps of the free cells for the symmetry planes, and the averaging projector.

    Each plane is snapped to the nearest element boundary or element center along its axis
    (position None -> center of the active region's bounding box). Cells whose mirror is not a
    free cell (outside the grid, inactive or passive) map to themselves. `apply` averages over
    the orbits of all mirror maps (connected components of the mirror pairs): an orthogonal
    projection also when an asymmetric domain leaves some mirrors missing.
    """

    def __init__(self, problem: Problem, planes: tuple[SymmetryPlane, ...]):
        grid = problem.grid
        shape = tuple(grid.shape)
        active = np.asarray(problem.active, dtype=bool)
        free_ids = np.flatnonzero(problem.free.ravel())
        pos = np.full(grid.nel, -1, dtype=np.int64)
        pos[free_ids] = np.arange(free_ids.size)
        ijk = np.stack(np.unravel_index(np.arange(grid.nel), shape))
        self.maps: list[np.ndarray] = []
        self.unmatched: list[float] = []  # fraction of active cells without an active mirror
        self.planes: list[tuple[str, float]] = []  # (axis, snapped position)
        for plane in planes:
            if plane.axis not in ("x", "y", "z"):
                raise ValueError(f"symmetry axis must be x, y or z, got {plane.axis!r}")
            a = "xyz".index(plane.axis)
            if plane.position is None:
                idx = np.flatnonzero(active.any(axis=tuple(k for k in range(3) if k != a)))
                s2 = int(idx[0] + idx[-1] + 1)  # twice the plane index, element units
            else:
                s2 = int(np.rint(2.0 * (plane.position - grid.origin[a]) / grid.h))
            m = ijk.copy()
            m[a] = s2 - 1 - ijk[a]  # mirror index along the axis
            inside = (m[a] >= 0) & (m[a] < shape[a])
            m[a] = np.clip(m[a], 0, shape[a] - 1)
            mirror = np.where(inside, np.ravel_multi_index(tuple(m), shape), -1)
            has = inside & active.ravel()[mirror]
            n_act = int(active.sum())
            self.unmatched.append(float((active.ravel() & ~has).sum()) / max(n_act, 1))
            mf = mirror[free_ids]
            mf = np.where(mf >= 0, pos[np.maximum(mf, 0)], -1)
            self.maps.append(np.where(mf >= 0, mf, np.arange(free_ids.size)))
            self.planes.append((plane.axis, grid.origin[a] + 0.5 * s2 * grid.h))
        n = free_ids.size
        pairs = sp.coo_matrix(
            (
                np.ones(n * len(self.maps)),
                (np.tile(np.arange(n), len(self.maps)), np.concatenate(self.maps)),
            ),
            shape=(n, n),
        )
        _, self.orbit = connected_components(pairs, directed=False)
        self.orbit_size = np.bincount(self.orbit).astype(np.float64)

    def warning(self) -> str:
        bad = [
            f"{ax}={p:g} ({100 * f:.0f} % of active cells unmatched)"
            for (ax, p), f in zip(self.planes, self.unmatched)
            if f > SYMMETRY_WARN_FRACTION
        ]
        if not bad:
            return ""
        return "warning: asymmetric domain for symmetry plane " + ", ".join(bad)

    def apply(self, v: np.ndarray) -> np.ndarray:
        """Average a free-cell vector over each orbit of mirror images."""
        sums = np.bincount(self.orbit, weights=v, minlength=self.orbit_size.size)
        return (sums / self.orbit_size)[self.orbit]


def problem_rows(
    r: Responses, params: RunParams, c0: float, ck: float | None
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """MMA form (f0, df0, g, dg) on the free variables: compliance / c0, then the constraints
    g_0 = volume / volfrac - 1 and, with a stress limit, g_1 = sigma_PN ck / limit - 1."""
    g = [r.volume / params.volfrac - 1.0]
    dg = [r.dv / params.volfrac]
    if params.stress_limit is not None:
        g.append(r.stress_pnorm * ck / params.stress_limit - 1.0)
        dg.append(r.dpn * (ck / params.stress_limit))
    return r.compliance / c0, r.dc / c0, np.array(g), np.stack(dg)


def optimize(
    problem: Problem,
    params: RunParams,
    callback: ProgressCallback | None = None,
    x0: np.ndarray | None = None,
    cancel: Callable[[], bool] | None = None,
    *,
    solver_options: dict | None = None,
) -> Result:
    """SIMP with OC (volume constraint only) or MMA (volume and optional stress constraint).

    `solver_options` are extra `LinearSolver` keyword arguments (tests, benchmarks).
    """
    issues = problem.validate()
    if issues:
        raise ValueError("invalid problem: " + "; ".join(issues))
    if params.optimizer not in ("oc", "mma"):
        raise ValueError(f"unknown optimizer {params.optimizer!r}")
    if params.stress_limit is not None and not params.stress_limit > 0:
        raise ValueError("stress_limit must be positive")
    need = Assembler.estimate_bytes(problem.n_active, np.dtype(params.dtype))
    if need > params.memory_cap_bytes:
        raise MemoryError(
            f"{problem.n_active} active elements need about {need / 1e9:.2f} GB, above the "
            f"{params.memory_cap_bytes / 1e9:.2f} GB cap; lower the resolution"
        )

    model = SimpModel(problem, params, solver_options)
    shape, free, solid = model.shape, model.free, model.solid
    sym = Symmetry(problem, tuple(params.symmetry)) if params.symmetry else None
    warn = sym.warning() if sym is not None else ""
    if warn:
        log.warning(warn)
    stress_on = params.stress_limit is not None
    mma = None
    if params.optimizer == "mma" or stress_on:
        mma = MMA(model.n_free, 2 if stress_on else 1, 0.0, 1.0, params.move)

    x = np.zeros(shape)
    x[free] = params.volfrac
    x[solid] = 1.0
    if x0 is not None:
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape != tuple(shape):
            raise ValueError(f"x0 must have the grid shape {shape}, got {x0.shape}")
        x[free] = np.clip(x0[free], 0.0, 1.0)
    if sym is not None:
        x[free] = sym.apply(x[free])

    xp = model.physical(x)
    history: list[IterationInfo] = []
    status, message = "max_iter", ""
    beta_since, change = 0, 1.0
    c0 = None  # MMA objective scale: first compliance
    ck = None  # adaptive stress normalization
    mma_it = 0
    for it in range(1, params.max_iter + 1):
        if cancel is not None and cancel():
            status, message = "cancelled", f"cancelled before iteration {it}"
            break
        t0 = time.perf_counter()
        if (
            params.heaviside
            and model.beta < BETA_MAX
            and ((beta_since >= 20 and change < 0.05) or beta_since >= 40)
        ):
            model.beta, beta_since = 2 * model.beta, 0
            xp = model.physical(x)
            mma_it = 0  # the projection changed: restart the MMA asymptotes
        beta_since += 1
        p = params.penal
        if params.continuation:
            p = 1.0 + (params.penal - 1.0) * min(1.0, (it - 1) / CONTINUATION_ITERS)

        r = model.evaluate(xp, p, change, pnorm=params.stress_pnorm if stress_on else None)
        c = r.compliance
        if not np.isfinite(c):
            status, message = "error", f"non-finite compliance at iteration {it}"
            break
        if sym is not None:
            r.dc, r.dv = sym.apply(r.dc), sym.apply(r.dv)
            r.dpn = sym.apply(r.dpn) if r.dpn is not None else None
        if stress_on:
            ratio = r.stress_max / r.stress_pnorm if r.stress_pnorm > 0 else 1.0
            ck = ratio if ck is None else STRESS_NORM_ALPHA * ratio + (1 - STRESS_NORM_ALPHA) * ck

        xf = x[free]
        g_stress, feasible = None, True
        if mma is None:  # OC: compliance and the volume row only (unscaled: OC is scale-free)
            if model.linear_volume:
                # mean(H x) over free cells is linear in x: exact, no filtering per bisection
                volume = partial(_linear_volume, float(xp[free].mean()), r.dv, xf)
            else:
                volume = partial(model.projected_volume, x)
            # OC oscillates on a sharp projection unless the step shrinks as beta grows
            move = params.move / np.sqrt(model.beta) if params.heaviside else params.move
            x_new = oc_update(xf, r.dc, r.dv, params.volfrac, move, volume=volume)
        else:
            if c0 is None:
                c0 = abs(c) if c != 0 else 1.0
            f0, df0, g, dg = problem_rows(r, params, c0, ck)
            g_stress = float(g[1]) if stress_on else None
            mma_it += 1
            x_new = mma.update(mma_it, xf, f0, df0, g, dg)
            feasible = g[0] <= MMA_FEAS_TOL and (g_stress is None or g_stress <= STRESS_FEAS_TOL)
        if sym is not None:
            x_new = sym.apply(x_new)
        change = float(np.abs(x_new - xf).max())
        x[free] = x_new
        xp = model.physical(x)
        info = IterationInfo(
            it,
            c,
            float(xp[free].mean()),
            change,
            time.perf_counter() - t0,
            stress_max=r.stress_max,
            constraint=g_stress,
        )
        history.append(info)

        if callback is not None:
            ret = callback(info, xp)
            if ret is not None and not ret:
                status, message = "cancelled", f"cancelled after iteration {it}"
                break
        settled = (not params.continuation or p >= params.penal) and (
            not params.heaviside or model.beta >= BETA_MAX
        )
        if change < params.tol and settled and feasible:
            status, message = "converged", f"converged after {it} iterations"
            break
    else:
        if params.max_iter > 0:
            message = f"stopped at max_iter={params.max_iter}"

    if warn:
        message = f"{warn}; {message}" if message else warn
    # the returned design moved since the last solve: its stress needs one more solve
    stress = model.stress_field(xp, p) if history and status != "error" else None
    return Result(rho=xp, history=history, status=status, message=message, stress=stress)
