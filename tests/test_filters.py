from __future__ import annotations

import numpy as np
import pytest

from topop.core.filters import DensityFilter, cone_kernel, heaviside

SHAPE = (11, 9, 7)
RMINS = [1.0, 1.5, 2.0, 2.6, 3.4]  # <= 2: ndimage.convolve, > 2: FFT


def irregular_active(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    active = rng.random(SHAPE) > 0.3
    active[:3, :3, :] = False  # an inactive corner block
    return active


def test_cone_kernel():
    k = cone_kernel(1.5)
    assert k.shape == (5, 5, 5)
    assert k[2, 2, 2] == pytest.approx(1.5)
    assert k[3, 2, 2] == pytest.approx(0.5)
    assert k[3, 3, 3] == 0.0  # sqrt(3) > 1.5
    assert np.allclose(k, k[::-1, ::-1, ::-1])


@pytest.mark.parametrize("rmin", RMINS)
def test_constant_field_is_preserved_on_active_cells(rmin):
    active = irregular_active()
    filt = DensityFilter(SHAPE, rmin, active)
    assert filt.use_fft == (rmin > 2)
    out = filt.apply(np.full(SHAPE, 0.37))
    assert np.abs(out[active] - 0.37).max() < 1e-10
    assert np.all(out[~active] == 0.0)


@pytest.mark.parametrize("rmin", RMINS)
def test_inactive_cells_stay_zero_and_do_not_leak(rmin):
    active = irregular_active(1)
    filt = DensityFilter(SHAPE, rmin, active)
    x = np.where(active, 0.0, 1.0)  # mass only on inactive cells
    assert np.all(filt.apply(x) == 0.0)
    assert np.all(filt.apply_adjoint(np.ones(SHAPE))[~active] == 0.0)


@pytest.mark.parametrize("rmin", RMINS)
def test_apply_and_adjoint_are_adjoint(rmin):
    rng = np.random.default_rng(2)
    filt = DensityFilter(SHAPE, rmin, irregular_active(2))
    for _ in range(3):
        x, y = rng.random(SHAPE), rng.standard_normal(SHAPE)
        lhs = np.vdot(filt.apply(x), y)
        rhs = np.vdot(x, filt.apply_adjoint(y))
        assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-12)


def test_filter_matches_explicit_weights():
    rng = np.random.default_rng(3)
    shape, rmin = (5, 4, 3), 1.5
    active = rng.random(shape) > 0.25
    x = rng.random(shape)
    out = DensityFilter(shape, rmin, active).apply(x)
    idx = np.argwhere(active)
    for i in idx[:: max(1, len(idx) // 10)]:
        d = np.linalg.norm(idx - i, axis=1)
        w = np.maximum(0.0, rmin - d)
        assert out[tuple(i)] == pytest.approx(w @ x[tuple(idx.T)] / w.sum(), rel=1e-12)


def test_heaviside_values_and_derivative():
    x = np.linspace(0.0, 1.0, 41)
    for beta in (1.0, 4.0, 32.0):
        y, dy = heaviside(x, beta)
        assert y[0] == pytest.approx(0.0, abs=1e-14)
        assert y[-1] == pytest.approx(1.0, abs=1e-14)
        assert y[20] == pytest.approx(0.5, abs=1e-14)
        assert np.all(np.diff(y) > 0)
        eps = 1e-6
        fd = (heaviside(x + eps, beta)[0] - heaviside(x - eps, beta)[0]) / (2 * eps)
        assert np.allclose(dy, fd, rtol=1e-6, atol=1e-6)
    y, dy = heaviside(x, 8.0, eta=0.3)
    fd = (heaviside(x + 1e-6, 8.0, 0.3)[0] - heaviside(x - 1e-6, 8.0, 0.3)[0]) / 2e-6
    assert np.allclose(dy, fd, rtol=1e-6, atol=1e-6)
    assert y[0] == pytest.approx(0.0, abs=1e-14) and y[-1] == pytest.approx(1.0, abs=1e-14)
