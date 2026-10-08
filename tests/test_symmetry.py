from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

from topop.core.benchmarks import cantilever, cantilever_params
from topop.core.optimize import Symmetry, optimize
from topop.core.problem import Load, SymmetryPlane


def corner_loaded_cantilever():
    """20x10x4 cantilever loaded at one top corner of the free end only: asymmetric in y and z."""
    p = cantilever(20, 10, 4)
    p.loads = [Load(np.array([p.grid.node_ids(20, 10, 0)]), (0.0, -1.0, 0.0))]
    return p


def params(**kw):
    return dataclasses.replace(cantilever_params(), **{"max_iter": 40, **kw})


@pytest.fixture(scope="module")
def plain_run():
    return optimize(corner_loaded_cantilever(), params())


@pytest.mark.parametrize("optimizer", ["oc", "mma"])
def test_y_symmetry_makes_the_design_mirror_symmetric(plain_run, optimizer):
    p = corner_loaded_cantilever()
    seen = []
    res = optimize(
        p,
        params(optimizer=optimizer, symmetry=(SymmetryPlane("y"),)),
        lambda info, rho: seen.append(rho.copy()),
    )
    for rho in [*seen, res.rho]:
        assert np.abs(rho - rho[:, ::-1, :]).max() < 1e-9
    assert abs(res.history[-1].volume - 0.3) < 1e-3
    assert res.message.startswith(("stopped", "converged"))  # no asymmetry warning
    # without symmetry the corner load gives a clearly asymmetric design
    assert np.abs(plain_run.rho - plain_run.rho[:, ::-1, :]).max() > 0.3
    # constraining the design costs stiffness
    assert res.history[-1].compliance > plain_run.history[-1].compliance


def test_two_planes_and_snapping():
    p = corner_loaded_cantilever()
    # 5.2 snaps to the nearest element boundary or center (y = 5.0, the box center)
    res = optimize(p, params(max_iter=15, symmetry=(SymmetryPlane("y", 5.2), SymmetryPlane("z"))))
    assert np.abs(res.rho - res.rho[:, ::-1, :]).max() < 1e-9
    assert np.abs(res.rho - res.rho[:, :, ::-1]).max() < 1e-9
    sym = Symmetry(p, (SymmetryPlane("y", 5.2), SymmetryPlane("x", 7.3)))
    assert sym.planes == [("y", 5.0), ("x", 7.5)]  # 7.3 -> element center 7.5


def test_symmetry_maps_are_involutions_and_respect_the_domain():
    p = cantilever(9, 6, 3)
    p.active[0:2, 0:2, :] = False  # the y-mirror of these cells (rows 4, 5) loses its partner
    p.passive[6, 1, :] = 1  # passive cells are never averaged
    sym = Symmetry(p, (SymmetryPlane("x"), SymmetryPlane("y")))
    free_ids = np.flatnonzero(p.free.ravel())
    for m, axis in zip(sym.maps, (0, 1)):
        assert np.array_equal(m[m], np.arange(m.size))
        ijk = np.stack(np.unravel_index(free_ids, p.grid.shape))
        paired = m != np.arange(m.size)
        mirrored = np.stack(np.unravel_index(free_ids[m], p.grid.shape))
        n = p.grid.shape[axis]
        # x: active bbox is 0..8 -> plane at 4.5 (center of cell 4); y: 0..5 -> plane at 3.0
        assert np.all(ijk[axis][paired] + mirrored[axis][paired] == n - 1)
    assert not paired[np.flatnonzero(free_ids == p.grid.element_ids(6, 4, 0))[0]]  # mirror passive
    v = np.random.default_rng(0).random(free_ids.size)
    w = sym.apply(v)
    assert w.sum() == pytest.approx(v.sum())
    assert np.allclose(sym.apply(w), w)  # a projection


def test_asymmetric_domain_warns(caplog):
    p = cantilever(12, 6, 2)
    with caplog.at_level(logging.WARNING, logger="topop.core.optimize"):
        res = optimize(p, params(max_iter=3, symmetry=(SymmetryPlane("x", 3.0),)))
    assert res.message.startswith("warning: asymmetric domain for symmetry plane x=3")
    assert "50 %" in res.message
    assert any("asymmetric domain" in r.message for r in caplog.records)
    # cells without a mirror keep evolving: the design is not forced to be symmetric about x=3
    assert abs(res.history[-1].volume - 0.3) < 1e-3
