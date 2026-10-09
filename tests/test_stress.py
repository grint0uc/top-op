from __future__ import annotations

import dataclasses
import functools
import time
from typing import ClassVar

import numpy as np
import pytest

from topop.core import optimize as opt
from topop.core.benchmarks import cantilever, cantilever_params, l_bracket
from topop.core.fem import Assembler, von_mises
from topop.core.optimize import (
    STRESS_Q,
    SimpModel,
    StressControl,
    Symmetry,
    optimize,
    stress_step_target,
)
from topop.core.problem import (
    Grid,
    Load,
    Material,
    Problem,
    Result,
    RunParams,
    Support,
    SymmetryPlane,
)
from topop.core.solver import LinearSolver


def bar(shape=(12, 2, 3), h=0.5, F=7.0, consistent=True) -> Problem:
    """Bar along x: base plane x=0 held in x only, plus 3 DOFs against the rigid modes
    (node (0,0,0) in y and z, node (0,ny,0) in z), so Poisson contraction is free.
    Tip traction F along +x on the x=nx face: consistent nodal forces (trapezoid weights) or
    the equal split of a single Load."""
    grid = Grid(origin=(1.0, -1.0, 2.0), h=h, shape=shape)
    nx, ny, nz = shape
    p = Problem(grid, np.ones(shape, bool), np.zeros(shape, np.int8), Material(E=210.0, nu=0.3))
    jj, kk = np.meshgrid(np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    base = grid.node_ids(np.zeros_like(jj), jj, kk).ravel()
    p.supports = [
        Support(base, (True, False, False)),
        Support(np.array([grid.node_ids(0, 0, 0)]), (True, True, True)),
        Support(np.array([grid.node_ids(0, ny, 0)]), (True, False, True)),
    ]
    tip = grid.node_ids(np.full_like(jj, nx), jj, kk).ravel()
    if consistent:
        wy = np.where((jj == 0) | (jj == ny), 0.5, 1.0)
        wz = np.where((kk == 0) | (kk == nz), 0.5, 1.0)
        w = (wy * wz).ravel()
        p.loads = [Load(np.array([n]), (F * wi / w.sum(), 0.0, 0.0)) for n, wi in zip(tip, w)]
    else:
        p.loads = [Load(tip, (F, 0.0, 0.0))]
    return p


def solve(p: Problem, E_e=None) -> tuple[Assembler, np.ndarray]:
    asm = Assembler(p)
    E = np.full(asm.n_elements, p.material.E) if E_e is None else E_e
    U, _ = LinearSolver("direct").solve(asm.assemble(E), asm.F_free)
    return asm, U


# ---- stress recovery --------------------------------------------------------------------------


def test_uniaxial_bar_stress_is_F_over_A():
    F, h, shape = 7.0, 0.5, (12, 2, 3)
    A = shape[1] * h * shape[2] * h
    asm, U = solve(bar(shape, h, F))
    sig = asm.element_stress(U[:, 0])
    assert sig.shape == (asm.n_elements, 6)
    assert np.allclose(sig[:, 0], F / A, rtol=2e-2)
    assert np.abs(sig[:, 0] / (F / A) - 1).max() < 1e-8  # constant strain: exact
    assert np.abs(sig[:, 1:]).max() < 1e-8 * F / A
    assert np.allclose(von_mises(sig), F / A, rtol=1e-8)
    # equal nodal split (not a consistent traction): exact only away from the loaded end
    asm, U = solve(bar(shape, h, F, consistent=False))
    sig = asm.element_stress(U[:, 0]).reshape(*shape, 6)
    mid = sig[shape[0] // 3 : 2 * shape[0] // 3]
    assert np.allclose(mid[..., 0], F / A, rtol=2e-2)
    assert np.abs(mid[..., 1:]).max() < 2e-2 * F / A


def test_element_stress_scaling_and_cases():
    p = bar((6, 2, 2), consistent=False)
    p.loads.append(Load(p.loads[0].nodes, (0.0, 0.0, 1.0), case=1))
    rng = np.random.default_rng(0)
    E_e = rng.uniform(1.0, 300.0, 24)
    asm, U = solve(p, E_e)
    both = asm.element_stress(U)
    assert both.shape == (2, asm.n_elements, 6)
    assert np.allclose(both[1], asm.element_stress(U[:, 1]))
    scaled = asm.element_stress(U, E_e)
    assert np.allclose(scaled, both * (E_e / p.material.E)[None, :, None])
    energies, sigma = asm.element_energies_and_stress(U)
    assert np.allclose(energies, asm.element_energies(U))
    assert np.allclose(sigma, both)


def test_von_mises_known_states():
    s = np.zeros((5, 6))
    s[0, 0] = -3.0  # uniaxial
    s[1, 3] = 2.0  # pure shear
    s[2, :3] = 5.0  # hydrostatic
    s[3] = [1.0, -2.0, 0.5, 0.3, -0.7, 1.1]
    s[4] = [10.0, 10.0, 0.0, 0.0, 0.0, 0.0]  # equibiaxial
    vm = von_mises(s)
    assert vm[:3] == pytest.approx([3.0, 2.0 * np.sqrt(3.0), 0.0])
    assert vm[4] == pytest.approx(10.0)
    T = np.array([[1.0, 0.3, 1.1], [0.3, -2.0, -0.7], [1.1, -0.7, 0.5]])
    e = np.linalg.eigvalsh(T)
    ref = np.sqrt(0.5 * ((e[0] - e[1]) ** 2 + (e[1] - e[2]) ** 2 + (e[2] - e[0]) ** 2))
    assert vm[3] == pytest.approx(ref)
    assert von_mises(s.reshape(5, 1, 6)).shape == (5, 1)


def test_von_mises_gradient_and_cross_energies_match_finite_differences():
    p = bar((4, 2, 2), consistent=False)
    asm = Assembler(p)
    rng = np.random.default_rng(1)
    U = rng.normal(size=asm.n_free)
    w = rng.uniform(0.5, 2.0, asm.n_elements)

    def f(u):
        return float(w @ von_mises(asm.element_stress(u)))

    g = asm.von_mises_gradient(asm.element_stress(U), w)
    fd = np.array([(f(U + 1e-6 * e) - f(U - 1e-6 * e)) / 2e-6 for e in np.eye(asm.n_free)])
    assert np.allclose(g, fd, rtol=1e-5, atol=1e-8 * np.abs(fd).max())
    V = rng.normal(size=asm.n_free)
    K = asm.assemble(np.ones(asm.n_elements))
    assert asm.element_cross_energies(U, V).sum() == pytest.approx(U @ (K @ V))
    UV = np.stack([U, V], axis=1)
    assert asm.element_cross_energies(UV, UV[:, ::-1]).sum() == pytest.approx(2 * U @ (K @ V))


# ---- stress constraint gradient ---------------------------------------------------------------


@pytest.mark.parametrize("heaviside", [False, True])
def test_stress_constraint_gradient_matches_central_differences(heaviside):
    # g = sigma_PN * c / limit - 1 with the normalization c frozen, through filter + projection,
    # two load cases (the p-norm runs over cases and elements), passive cells
    p = cantilever(6, 4, 2)
    top = p.grid.node_ids(*np.meshgrid(6, 4, range(3), indexing="ij")).ravel()
    p.loads.append(Load(top, (0.3, 1.0, 0.0), case=1))
    p.passive[2, 1, 0] = 1
    p.passive[4, 3, 1] = -1
    prm = dataclasses.replace(cantilever_params(), heaviside=heaviside, stress_limit=0.7)
    model = SimpModel(p, prm)
    model.beta = 4.0 if heaviside else 0.0
    rng = np.random.default_rng(3)
    x = np.zeros(p.grid.shape)
    x[p.free] = rng.uniform(0.15, 0.85, int(p.free.sum()))
    x[p.passive == 1] = 1.0
    ck, limit = 0.8, prm.stress_limit

    def g(xx):
        r = model.evaluate(model.physical(xx), 3.0, pnorm=prm.stress_pnorm)
        return r.stress_pnorm * ck / limit - 1.0, r.dpn * ck / limit, r

    _, dg, r = g(x)
    assert r.stress_pnorm >= r.stress_max > 0
    fd = np.empty_like(dg)
    for k, e in enumerate(np.flatnonzero(p.free.ravel())):
        xx = x.copy()
        xx.ravel()[e] += 1e-6
        gp = g(xx)[0]
        xx.ravel()[e] -= 2e-6
        fd[k] = (gp - g(xx)[0]) / 2e-6
    rel = np.abs(fd - dg) / np.abs(dg)
    assert np.mean(rel < 1e-3) >= 0.95, np.sort(rel)[-5:]
    assert np.median(rel) < 1e-5


# ---- optimizer integration --------------------------------------------------------------------


def test_result_stress_and_history_fields():
    p = cantilever(16, 6, 2)
    p.active[6:9, 2:4, :] = False
    res = optimize(p, dataclasses.replace(cantilever_params(), max_iter=10))
    assert res.stress is not None and res.stress.shape == p.grid.shape
    assert np.all(res.stress[~p.active] == 0.0) and res.stress.min() >= 0.0
    assert all(h.stress_max is not None and h.stress_max > 0 for h in res.history)
    assert all(h.constraint is None for h in res.history)
    # relaxed stress rho^q * sigma_vm(solid) of the returned design (one more solve)
    asm = Assembler(p)
    xe = res.rho.ravel()[asm.element_ids]
    U, _ = LinearSolver("direct").solve(asm.assemble(1e-9 + xe**3 * (1 - 1e-9)), asm.F_free)
    vm = von_mises(asm.element_stress(U[:, 0]))
    assert np.allclose(res.stress.ravel()[asm.element_ids], xe**STRESS_Q * vm, rtol=1e-6)


def test_stress_limit_lowers_the_peak_stress():
    p = cantilever(24, 8, 2)
    base = RunParams(volfrac=0.4, rmin=1.5, max_iter=60, optimizer="mma")
    free_run = optimize(p, base)
    s0 = free_run.history[-1].stress_max
    # default max_iter: the volume stays on target throughout, so the stress comes down at a
    # bounded rate per step (docs/STRESS.md) instead of being bought with extra volume early on
    res = optimize(
        p, dataclasses.replace(base, optimizer="oc", stress_limit=0.85 * s0, max_iter=100)
    )
    assert all(h.constraint is not None for h in res.history)  # MMA forced
    assert res.history[-1].stress_max < 0.97 * s0
    assert res.history[-1].constraint < 0.1
    assert all(abs(h.volume - 0.4) < 5e-3 for h in res.history[10:])
    assert abs(res.history[-1].volume - 0.4) < 1e-3


# ---- automatic conditioning of the stress path (docs/STRESS.md) -------------------------------


def p_schedule(params: RunParams, changes) -> list[float]:
    ctl = StressControl(params, 10)
    return [ctl.begin(ch) for ch in changes]


def test_pnorm_continuation_schedule():
    every, settle = opt.STRESS_P_EVERY, opt.STRESS_P_SETTLE
    p = p_schedule(RunParams(stress_limit=1.0), [1.0] * (4 * every + 5))
    assert p[:every] == [8.0] * every
    assert p[every : 2 * every] == [16.0] * every
    assert p[2 * every : 3 * every] == [32.0] * every
    assert set(p[3 * every :]) == {64.0}  # default stress_pnorm: the final exponent
    # a settled design (change < 0.02) doubles p as soon as the asymptotes have had `settle`
    # iterations since the last restart
    p = p_schedule(RunParams(stress_limit=1.0), [0.01] * (3 * settle + 2))
    assert p == [8] * settle + [16] * settle + [32] * settle + [64, 64]
    # stress_pnorm is the final exponent, reached exactly; below 8 it is also the start
    p = p_schedule(RunParams(stress_limit=1.0, stress_pnorm=100), [0.01] * (5 * settle + 3))
    assert p[::settle] == [8, 16, 32, 64, 100, 100] and p[-1] == 100
    assert p_schedule(RunParams(stress_limit=1.0, stress_pnorm=12), [0.01] * 12)[-1] == 12
    assert set(p_schedule(RunParams(stress_limit=1.0, stress_pnorm=4), [0.01] * 12)) == {4}


def test_pnorm_never_doubles_while_damped_or_right_after_a_restart():
    every, settle = opt.STRESS_P_EVERY, opt.STRESS_P_SETTLE
    ctl = StressControl(RunParams(stress_limit=1.0), 10)
    assert [ctl.begin(1.0) for _ in range(every)] == [8.0] * every
    ctl.damp_left = 2  # oscillation damping in progress: neither trigger doubles p
    assert ctl.begin(1.0) == 8.0 and ctl.begin(0.001) == 8.0
    ctl.damp_left = 0
    assert ctl.begin(1.0) == 16.0
    # a projection change restarts the asymptotes; p waits `settle` iterations after it
    for _ in range(settle):
        ctl.begin(1.0)
    ctl.mma.xold1 = ctl.mma.low = np.zeros(10)  # pretend MMA has a history
    ctl.restart()
    assert ctl.n_restarts == 1 and ctl.mma.xold1 is None and ctl.mma.low is None
    assert [ctl.begin(0.001) for _ in range(settle + 1)] == [16.0] * settle + [32.0]


def test_stress_step_target_has_a_relative_and_an_absolute_decrease():
    assert stress_step_target(-0.3) == 0.0 and stress_step_target(0.0) == 0.0
    assert stress_step_target(6.0) == pytest.approx(0.95 * 6.0)  # far: 5 % per step
    assert stress_step_target(0.2) == pytest.approx(0.18)  # near: at least 0.02 per step
    assert stress_step_target(0.01) == 0.0  # never asks for more than feasibility
    g = np.linspace(1e-4, 8.0, 500)
    t = np.array([stress_step_target(v) for v in g])
    assert np.all((t >= 0) & (t < g) & (t <= 0.95 * g + 1e-15))
    assert np.all((g - t >= opt.STRESS_MIN_DECREASE - 1e-12) | (t == 0.0))


def test_move_limit_schedule_and_oscillation_damping():
    ctl = StressControl(RunParams(stress_limit=1.0), 200)  # default move 0.2
    ctl.begin(1.0)
    ctl.g = 3.0
    assert ctl.move() == pytest.approx(opt.STRESS_MOVE[0])  # warm-up
    ctl.it = opt.STRESS_WARMUP + 1
    assert ctl.move() == pytest.approx(opt.STRESS_MOVE[1])  # constraint near-active
    ctl.g = -0.5
    assert ctl.move() == pytest.approx(opt.STRESS_MOVE[0])  # far from active
    assert StressControl(RunParams(stress_limit=1.0, move=0.02), 5).move() == 0.02
    # alternating steps on most moving variables halve the move for OSC_DAMP_ITERS iterations
    rng = np.random.default_rng(0)
    dx = rng.choice([-0.05, 0.05], 200)
    dx[:20] = 0.0  # not moving: ignored
    ctl.track(dx)
    ctl.track(dx)  # same direction: no oscillation
    assert ctl.damp_left == 0 and not ctl.notes
    flipped = dx.copy()
    flipped[:120] *= -1  # 100 of the 180 moving variables flip
    ctl.track(flipped)
    assert ctl.damp_left == opt.OSC_DAMP_ITERS
    assert ctl.move() == pytest.approx(0.5 * opt.STRESS_MOVE[0])
    assert len(ctl.notes) == 1 and "oscillation" in ctl.notes[0]
    ctl.track(-flipped)
    assert len(ctl.notes) == 1  # reported once


class SpyControl(StressControl):
    """StressControl that records its instances, each step's input and each step passed to
    `track`; with `noise` > 0 it perturbs MMA's step (any later change of the step, as the
    symmetry projection makes, must reach `track`)."""

    made: ClassVar[list[SpyControl]] = []
    noise = 0.0

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.inputs: list[np.ndarray] = []
        self.tracked: list[np.ndarray] = []
        self.rng = np.random.default_rng(0)
        SpyControl.made.append(self)

    def step(self, xf, r, c0):
        self.inputs.append(xf.copy())
        x_new, g = super().step(xf, r, c0)
        if self.noise:
            x_new = np.clip(x_new + self.rng.uniform(-self.noise, self.noise, x_new.size), 0, 1)
        return x_new, g

    def track(self, dx):
        self.tracked.append(dx.copy())
        super().track(dx)


@pytest.fixture
def spy(monkeypatch) -> type[SpyControl]:
    SpyControl.made, SpyControl.noise = [], 0.0
    monkeypatch.setattr(opt, "StressControl", SpyControl)
    return SpyControl


def test_oscillation_detector_sees_the_applied_symmetric_step(spy):
    spy.noise = 0.01  # an asymmetric step that the symmetry projection then averages
    p = cantilever(12, 6, 4)
    planes = (SymmetryPlane("z"),)
    params = dataclasses.replace(cantilever_params(), max_iter=4, symmetry=planes, stress_limit=1.0)
    optimize(p, params)
    (ctl,) = spy.made
    sym = Symmetry(p, planes)
    assert len(ctl.inputs) == len(ctl.tracked) == 4
    for k in range(3):
        applied = ctl.inputs[k + 1] - ctl.inputs[k]
        assert np.array_equal(ctl.tracked[k], applied)
        assert np.allclose(applied, sym.apply(applied), rtol=0, atol=1e-15)


def test_projection_restart_resets_the_stress_path_asymptotes(spy):
    # heaviside: beta doubles at iteration 41 at the latest (earlier once change < 0.05), which
    # restarts the MMA asymptotes through StressControl.restart
    p = l_bracket(16, 2)
    base = RunParams(volfrac=0.5, rmin=1.5, max_iter=50, heaviside=True, optimizer="mma")
    s0 = float(optimize(p, base).stress.max())
    limit = 0.8 * s0
    res = optimize(p, dataclasses.replace(base, stress_limit=limit))
    ctl = spy.made[-1]
    assert ctl.n_restarts >= 1
    assert ctl.p == ctl.p_max  # the continuation still completed
    assert all(np.isfinite(h.compliance) and np.isfinite(h.constraint) for h in res.history)
    assert np.isfinite(res.rho).all() and np.isfinite(res.stress).all()
    # feasible, or close to fully stressed at the target volume after 50 iterations
    assert res.history[-1].constraint <= opt.STRESS_FEAS_TOL or (
        float(res.stress.max()) <= 1.15 * limit and res.history[-1].constraint < 0.15
    )
    assert abs(res.history[-1].volume - 0.5) < 5e-3


# ---- L-bracket benchmark ----------------------------------------------------------------------


@functools.cache
def free_l_bracket(n: int, volfrac: float) -> Result:
    """Compliance-only MMA run: the reference peak stress (at the re-entrant corner)."""
    params = RunParams(volfrac=volfrac, rmin=1.5, max_iter=100, optimizer="mma")
    return optimize(l_bracket(n, 4), params)


def corner_density(p: Problem, rho: np.ndarray, arm: int, w: int = 4) -> float:
    """Mean density of the active cells in the w x w (x, y) box around the re-entrant corner."""
    sl = (slice(arm - w // 2, arm + w // 2), slice(arm - w // 2, arm + w // 2))
    return float(rho[sl][p.active[sl]].mean())


def test_l_bracket_benchmark_geometry():
    p = l_bracket(40, 4)
    assert p.grid.shape == (40, 40, 4) and p.validate() == []
    assert p.active.sum() == (40 * 40 - 24 * 24) * 4
    assert not p.active[16:, 16:].any() and p.active[:16].all() and p.active[:, :16].all()
    support = np.stack(np.unravel_index(p.supports[0].nodes, p.grid.node_shape), axis=1)
    assert np.all(support[:, 1] == 40) and set(support[:, 0]) == set(range(17))
    load = np.stack(np.unravel_index(p.loads[0].nodes, p.grid.node_shape), axis=1)
    assert set(load[:, 0]) == {37, 38, 39, 40} and np.all(load[:, 1] == 16)
    assert p.loads[0].force == (0.0, -1.0, 0.0)


@pytest.mark.slow
def test_l_bracket_stress_constraint_moves_material_off_the_corner():
    # compliance + volume 0.5 + stress <= 0.7 x the unconstrained peak (at the re-entrant
    # corner), with the explicit p = 16 and move 0.05 that were needed before the automatic
    # conditioning (docs/STRESS.md)
    p = l_bracket(40, 4)
    arm = 16
    base = RunParams(volfrac=0.5, rmin=1.5, max_iter=100, optimizer="mma")
    free_run = free_l_bracket(40, 0.5)
    s0 = float(free_run.stress.max())
    peak = np.unravel_index(np.argmax(free_run.stress), p.grid.shape)
    assert abs(peak[0] + 0.5 - arm) <= 1 and abs(peak[1] + 0.5 - arm) <= 1  # at the corner
    limit = 0.7 * s0
    res = optimize(
        p,
        dataclasses.replace(base, max_iter=200, move=0.05, stress_pnorm=16, stress_limit=limit),
    )
    s1 = float(res.stress.max())
    assert s1 <= 1.1 * limit, f"max stress {s1:.4f} = {s1 / limit:.3f} x limit"
    assert abs(res.history[-1].volume - 0.5) < 1e-3
    assert corner_density(p, res.rho, arm) < corner_density(p, free_run.rho, arm)
    assert corner_density(p, res.rho, arm, w=6) < corner_density(p, free_run.rho, arm, w=6) - 0.01


# User sets only volfrac, rmin and stress_limit = 0.7 x the unconstrained peak; move (0.2),
# stress_pnorm (64) and max_iter (100) keep their defaults except where noted. vf 0.3 is
# near-infeasible at this limit (docs/STRESS.md): the unconstrained design's inner flange already
# carries 1.3 x the limit along the whole vertical arm, and the constrained optimum at exactly
# 30 % volume is fully stressed at about 1.12 x; it gets 200 iterations and a 1.15 bound.
L_BRACKET_CASES = [
    pytest.param(40, 0.5, 100, 1.10, id="40-vf0.5"),
    pytest.param(40, 0.3, 200, 1.15, id="40-vf0.3"),
    pytest.param(60, 0.4, 100, 1.10, id="60-vf0.4"),
]


@pytest.mark.slow
@pytest.mark.parametrize(("n", "volfrac", "max_iter", "bound"), L_BRACKET_CASES)
def test_l_bracket_stress_limit_at_default_parameters(n, volfrac, max_iter, bound):
    limit = 0.7 * float(free_l_bracket(n, volfrac).stress.max())
    params = RunParams(volfrac=volfrac, rmin=1.5, max_iter=max_iter, stress_limit=limit)
    t0 = time.perf_counter()
    res = optimize(l_bracket(n, 4), params)
    wall = time.perf_counter() - t0
    s1 = float(res.stress.max())
    print(
        f"l_bracket({n}) vf={volfrac}: {len(res.history)} its, {wall:.0f} s, "
        f"max stress {s1 / limit:.3f} x limit, volume {res.history[-1].volume:.4f}"
    )
    assert s1 <= bound * limit, f"max stress {s1 / limit:.3f} x limit"
    assert abs(res.history[-1].volume - volfrac) < 1e-3
    # stable: no two consecutive compliance changes above 10 % in the last 20 iterations
    c = np.array([h.compliance for h in res.history[-21:]])
    big = np.abs(np.diff(c)) / c[:-1] > 0.10
    assert not np.any(big[1:] & big[:-1])
    assert all(h.constraint is not None for h in res.history)
