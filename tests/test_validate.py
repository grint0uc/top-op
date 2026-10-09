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
