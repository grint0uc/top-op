from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from topop.core.benchmarks import cantilever, cantilever_params
from topop.core.mma import MMA
from topop.core.optimize import optimize

BEAM_COEF = np.array([61.0, 37.0, 19.0, 7.0, 1.0])
BEAM_OPT = np.array([6.016, 5.309, 4.494, 3.502, 2.153])


def run_beam(mma: MMA, x: np.ndarray, max_iter: int = 100) -> tuple[np.ndarray, int]:
    for k in range(1, max_iter + 1):
        g = np.array([np.sum(BEAM_COEF / x**3) - 1.0])
        dg = (-3.0 * BEAM_COEF / x**4)[None, :]
        x_new = mma.update(k, x, 0.0624 * x.sum(), np.full(5, 0.0624), g, dg)
        done = np.abs(x_new - x).max() < 1e-6
        x = x_new
        if done:
            break
    return x, k


@pytest.mark.parametrize("move", [0.2, 0.5, 1.0])
def test_svanberg_beam_converges_to_known_optimum(move):
    # Svanberg 1987, cantilever beam of 5 hollow square segments
    x, its = run_beam(MMA(5, 1, 1.0, 10.0, move), np.full(5, 5.0))
    assert its < 40
    assert np.allclose(x, BEAM_OPT, rtol=1e-2)
    assert 0.0624 * x.sum() == pytest.approx(1.340, abs=1e-3)
    assert np.sum(BEAM_COEF / x**3) - 1.0 < 1e-5


def test_mma_respects_bounds_move_limit_and_history():
    mma = MMA(5, 1, 1.0, 10.0, 0.05)
    x = np.full(5, 5.0)
    g = np.array([np.sum(BEAM_COEF / x**3) - 1.0])
    x1 = mma.update(1, x, 0.0624 * x.sum(), np.full(5, 0.0624), g, (-3 * BEAM_COEF / x**4)[None])
    assert np.all(np.abs(x1 - x) <= 0.05 * 9.0 + 1e-12)  # move is relative to xmax - xmin
    assert mma.xold1 is not None and np.array_equal(mma.xold1, x)
    assert np.all(mma.low < x) and np.all(mma.upp > x)
    mma.reset()
    assert mma.xold1 is None and mma.low is None
    # infeasible start far from the optimum, tight bounds: x stays inside [xmin, xmax]
    x, _ = run_beam(MMA(5, 1, 1.0, 3.0, 0.5), np.full(5, 1.5), max_iter=30)
    assert x.min() >= 1.0 and x.max() <= 3.0


def test_mma_more_constraints_than_variables():
    # min (x0-1)^2 + (x1-2)^2 s.t. x0 + x1 <= 1, x0 - x1 <= 0.5, -x0 <= 0 (m=3 > n=2):
    # optimum at x = (0, 1), where the active bound x0 >= 0 has a zero multiplier (slow tail)
    mma = MMA(2, 3, -5.0, 5.0, 0.5)
    x = np.array([0.5, 0.0])
    for k in range(1, 200):
        f0 = (x[0] - 1) ** 2 + (x[1] - 2) ** 2
        df0 = np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])
        g = np.array([x[0] + x[1] - 1, x[0] - x[1] - 0.5, -x[0]])
        dg = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 0.0]])
        x_new = mma.update(k, x, f0, df0, g, dg)
        if np.abs(x_new - x).max() < 1e-8:
            break
        x = x_new
    assert np.allclose(x, [0.0, 1.0], atol=1e-3)


def test_mma_cantilever_matches_oc():
    p = cantilever(20, 8, 4)
    base = dataclasses.replace(cantilever_params(), max_iter=100)
    oc = optimize(p, base)
    res = optimize(p, dataclasses.replace(base, optimizer="mma"))
    c_oc, c_mma = oc.history[-1].compliance, res.history[-1].compliance
    assert abs(c_mma - c_oc) / c_oc < 0.05
    assert abs(res.history[-1].volume - 0.3) < 1e-3
    assert abs(res.rho.mean() - 0.3) < 1e-3
    assert res.rho.min() >= 0.0 and res.rho.max() <= 1.0
    c = np.array([h.compliance for h in res.history])
    assert c[-1] < 0.2 * c[0]


def test_mma_runs_with_projection_and_passives():
    p = cantilever(16, 6, 2)
    p.passive[6:9, 4:6, :] = 1
    p.passive[2:4, 2:4, :] = -1
    prm = dataclasses.replace(cantilever_params(), max_iter=50, optimizer="mma", heaviside=True)
    res = optimize(p, prm)
    assert np.all(res.rho[p.passive == 1] == 1.0) and np.all(res.rho[p.passive == -1] == 0.0)
    assert abs(res.rho[p.free].mean() - 0.3) < 5e-3
    c = np.array([h.compliance for h in res.history])
    assert np.all(np.isfinite(c)) and c[-1] < 0.3 * c[0]
