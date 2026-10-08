from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from topop.core.benchmarks import cantilever, cantilever_params
from topop.core.optimize import oc_update, optimize
from topop.core.problem import Load, RunParams


def small_params(**kw) -> RunParams:
    return dataclasses.replace(cantilever_params(), **{"max_iter": 30, **kw})


@pytest.fixture(scope="module")
def small_run():
    calls = []

    def cb(info, rho):
        calls.append((info, rho.copy()))
        return True

    res = optimize(cantilever(20, 8, 4), small_params(), cb)
    return res, calls


def test_small_cantilever_converges_sensibly(small_run):
    res, calls = small_run
    assert res.status == "max_iter" and len(res.history) == 30
    c = np.array([h.compliance for h in res.history])
    assert np.all(c[5:][1:] <= c[5:][:-1] * 1.02)
    assert c[-1] < 0.2 * c[0]
    assert abs(res.history[-1].volume - 0.3) < 1e-3
    assert abs(res.rho.mean() - 0.3) < 1e-3  # full box: every element is free
    assert res.rho.shape == (20, 8, 4)
    assert res.rho.min() >= 0.0 and res.rho.max() <= 1.0
    assert [info.it for info, _ in calls] == list(range(1, 31))
    assert all(info.t_iter > 0 for info, _ in calls)
    assert np.array_equal(calls[-1][1], res.rho)


def test_rho_is_zero_outside_active_and_bounded():
    p = cantilever(20, 8, 4)
    p.active[8:14, 3:6, :] = False  # a hole through the beam
    res = optimize(p, small_params(max_iter=8))
    assert np.all(res.rho[~p.active] == 0.0)
    assert res.rho.min() >= 0.0 and res.rho.max() <= 1.0
    assert abs(res.history[-1].volume - 0.3) < 1e-3
    assert abs(res.rho[p.active].mean() - 0.3) < 1e-3


def test_passive_solid_and_void_are_kept():
    p = cantilever(20, 8, 4)
    p.passive[10:13, 5:8, :] = 1
    p.passive[3:6, 2:5, :] = -1
    seen = []
    res = optimize(p, small_params(max_iter=10), lambda info, rho: seen.append(rho.copy()))
    for rho in [*seen, res.rho]:
        assert np.all(rho[p.passive == 1] == 1.0)
        assert np.all(rho[p.passive == -1] == 0.0)
    # passives are excluded from the volume constraint
    assert abs(res.rho[p.free].mean() - 0.3) < 1e-3
    assert res.history[-1].volume == pytest.approx(res.rho[p.free].mean())


def test_cancel_via_callback_and_via_cancel_flag():
    p = cantilever(12, 6, 2)
    res = optimize(p, small_params(), lambda info, rho: info.it < 3)
    assert res.status == "cancelled" and len(res.history) == 3
    n = [0]

    def cancel():
        n[0] += 1
        return n[0] > 2

    res = optimize(p, small_params(), cancel=cancel)
    assert res.status == "cancelled" and len(res.history) == 2
    assert res.rho.shape == p.grid.shape


def test_converged_status_and_x0():
    p = cantilever(12, 6, 2)
    first = optimize(p, small_params(max_iter=60, tol=0.05))
    assert first.status == "converged"
    assert first.history[-1].change < 0.05
    again = optimize(p, small_params(max_iter=5), x0=first.rho)
    assert again.history[0].compliance < 0.5 * first.history[0].compliance


def test_projection_and_continuation_run():
    p = cantilever(16, 6, 2)
    plain = optimize(p, small_params(max_iter=60))
    res = optimize(p, small_params(max_iter=60, heaviside=True, continuation=True))
    c = np.array([h.compliance for h in res.history])
    assert np.all(np.isfinite(c))
    assert c[-1] < c[20]  # penal reaches 3 at iteration 21; compliance rises until then
    assert abs(res.history[-1].volume - 0.3) < 1e-3  # exact projected volume in the bisection
    assert abs(res.rho.mean() - 0.3) < 1e-3
    assert res.rho.min() >= 0.0 and res.rho.max() <= 1.0

    def grey(r):
        return float(np.mean(4 * r * (1 - r)))

    assert grey(res.rho) < grey(plain.rho)


def test_multiple_load_cases():
    p = cantilever(16, 6, 2)
    top = p.grid.node_ids(*np.meshgrid(16, 6, range(3), indexing="ij")).ravel()
    p.loads.append(Load(top, (0.0, 1.0, 0.0), case=1))
    res = optimize(p, small_params(max_iter=6))
    assert res.history[-1].compliance < res.history[0].compliance


@pytest.mark.parametrize("solver", ["amg", "direct"])
def test_float32_and_solver_choice_agree_with_float64(solver):
    p = cantilever(12, 6, 2)
    ref = optimize(p, small_params(max_iter=5, solver="direct"))
    res = optimize(p, small_params(max_iter=5, solver=solver, dtype="float32"))
    c_ref = [h.compliance for h in ref.history]
    c = [h.compliance for h in res.history]
    assert np.allclose(c, c_ref, rtol=1e-3)


def test_invalid_problem_and_memory_cap():
    p = cantilever(8, 4, 2)
    p.loads = []
    with pytest.raises(ValueError, match="no loads"):
        optimize(p, small_params())
    with pytest.raises(MemoryError, match="lower the resolution"):
        optimize(cantilever(8, 4, 2), small_params(memory_cap_bytes=1000))


def test_oc_update_volume_move_and_bounds():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0.1, 0.6, n)
    dc = -rng.exponential(1.0, n) * 1e-7  # tiny scale: bisection must not depend on units
    dv = np.full(n, 1.0 / n)
    for volfrac, move in [(0.3, 0.2), (0.38, 0.05), (0.45, 1.0), (0.2, 0.2)]:
        x_new = oc_update(x, dc, dv, volfrac, move)
        assert abs(x_new.mean() - volfrac) < 1e-4
        assert np.all(np.abs(x_new - x) <= move + 1e-12)
        assert x_new.min() >= 0.0 and x_new.max() <= 1.0
    # out of reach within the move limit: the closest bound
    assert np.allclose(oc_update(x, dc, dv, 0.9, 0.05), np.minimum(1.0, x + 0.05))
    # a custom (here weighted) volume measure is honoured
    w = rng.uniform(0.5, 1.5, n)
    x_new = oc_update(x, dc, w / w.sum(), 0.4, 0.2, volume=lambda xn: w @ xn / w.sum())
    assert abs(w @ x_new / w.sum() - 0.4) < 1e-4
    # larger sensitivity -> the element grows relative to the others
    dc2 = dc.copy()
    dc2[0] *= 100
    assert oc_update(x, dc2, dv, 0.5, 0.2)[0] >= oc_update(x, dc, dv, 0.5, 0.2)[0]
