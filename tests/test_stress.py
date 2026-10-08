from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from topop.core.benchmarks import cantilever, cantilever_params, l_bracket
from topop.core.fem import Assembler, von_mises
from topop.core.optimize import STRESS_Q, SimpModel, optimize
from topop.core.problem import Grid, Load, Material, Problem, RunParams, Support
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
    res = optimize(p, dataclasses.replace(base, optimizer="oc", stress_limit=0.85 * s0))
    assert all(h.constraint is not None for h in res.history)  # MMA forced
    assert res.history[-1].stress_max < 0.97 * s0
    assert res.history[-1].constraint < 0.1


# ---- L-bracket benchmark ----------------------------------------------------------------------


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
    # corner). p = 16 and move 0.05: with p = 8 or move >= 0.1 MMA oscillates and ends 12-18 %
    # above the limit (the peak at the corner is a near-singularity on this 40 x 40 x 4 mesh).
    p = l_bracket(40, 4)
    arm = 16
    base = RunParams(volfrac=0.5, rmin=1.5, max_iter=100, optimizer="mma")
    free_run = optimize(p, base)
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
