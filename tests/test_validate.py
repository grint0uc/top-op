"""Problem.validate / warnings: rigid-body and lost-force checks (found by review, not by tests)."""

import numpy as np

from topop.core.benchmarks import cantilever
from topop.core.problem import Load, Support


def _box(nelx=6, nely=4, nelz=4):
    p = cantilever(nelx, nely, nelz)
    g = p.grid
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in g.node_shape], indexing="ij")
    return p, g, ii, jj, kk


def test_cantilever_is_valid():
    p, *_ = _box()
    assert p.validate() == []
    assert p.warnings() == []


def test_roller_only_support_leaves_rigid_modes():
    p, g, ii, jj, kk = _box()
    bottom = g.node_ids(ii[:, :, 0], jj[:, :, 0], kk[:, :, 0]).ravel()
    p.supports = [Support(nodes=bottom, fix=(False, False, True))]
    p.loads = [Load(nodes=np.array([g.node_ids(6, 2, 4)]), force=(1.0, 0.0, 0.0))]
    issues = p.validate()
    assert any("rigid-body" in s and "rank 3/6" in s for s in issues), issues


def test_hinge_line_support_leaves_one_rotation():
    p, g, ii, jj, kk = _box()
    edge = g.node_ids(ii[0, :, 0], jj[0, :, 0], kk[0, :, 0]).ravel()  # x=0, z=0 line along y
    p.supports = [Support(nodes=edge, fix=(True, True, True))]
    p.loads = [Load(nodes=np.array([g.node_ids(6, 2, 4)]), force=(0.0, 0.0, -1.0))]
    issues = p.validate()
    assert any("rank 5/6" in s for s in issues), issues


def test_load_entirely_on_fixed_dofs_is_rejected():
    p, g, ii, jj, kk = _box()
    face = g.node_ids(ii[0], jj[0], kk[0]).ravel()
    p.loads = [Load(nodes=face[:4], force=(0.0, 0.0, -1.0))]  # on the clamped face
    assert any("acts only on fixed DOFs" in s for s in p.validate())
    p.loads = [Load(nodes=np.array([g.node_ids(6, 2, 4)]), force=(0.0, 0.0, 0.0))]
    assert any("zero total force" in s for s in p.validate())


def test_partially_lost_force_is_a_warning_not_an_error():
    p, g, ii, jj, kk = _box()
    top = g.node_ids(ii[:, :, -1], jj[:, :, -1], kk[:, :, -1]).ravel()  # shares the x=0 edge
    p.loads = [Load(nodes=top, force=(0.0, 0.0, -1.0))]
    assert p.validate() == []
    w = p.warnings()
    assert len(w) == 1 and "lost to the supports" in w[0], w


def test_active_node_mask_matches_the_element_table():
    rng = np.random.default_rng(0)
    p, g, *_ = _box(7, 5, 3)
    p.active = rng.random(g.shape) > 0.4
    expected = np.zeros(g.n_nodes, dtype=bool)
    expected[g.element_nodes()[p.active.ravel()].ravel()] = True
    assert np.array_equal(p.active_node_mask(), expected)


def test_validate_is_cheap_on_a_big_grid():
    # it used to build (nel, 8) int64 connectivity and two full node-coordinate tables
    import time
    import tracemalloc

    from topop.core.problem import Grid, Problem

    g = Grid((0.0, 0.0, 0.0), 1.0, (200, 200, 200))  # 8M cells
    active = np.zeros(g.shape, dtype=bool)
    active[:, 90:110, 90:110] = True
    ii, jj, kk = np.meshgrid(np.arange(1), np.arange(90, 111), np.arange(90, 111), indexing="ij")
    clamp = g.node_ids(ii, jj, kk).ravel()
    tip = np.array([g.node_ids(200, 100, 110)])
    p = Problem(g, active, np.zeros(g.shape, np.int8), loads=[Load(tip, (0.0, 0.0, -1.0))],
                supports=[Support(clamp)])  # fmt: skip
    tracemalloc.start()  # numpy reports its buffers to tracemalloc
    try:
        t = time.perf_counter()
        assert p.validate() == []
        dt = time.perf_counter() - t
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert dt < 2.0
    assert peak < 16 * g.nel, peak / g.nel  # was > 200 bytes per cell
