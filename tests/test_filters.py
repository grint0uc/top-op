from __future__ import annotations

import math
import time

import numpy as np
import pytest

from topop.core.filters import AMFilter, DensityFilter, cone_kernel, heaviside

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


# ---------------------------------------------------------------- AM overhang filter (Langelaar)

DIRECTIONS = ["+x", "-x", "+y", "-y", "+z", "-z"]
AM_SHAPE = (6, 5, 7)
# smin(a, b) - min(a, b) lies in [0, sqrt(eps)/2]: Langelaar's +sqrt(eps) offset makes
# smin(0, 0) = 0 exact but lets a solid cell over 5 solid supporters (smax = 5^(1/Q) > 1) come
# out at up to 1 + sqrt(eps)/2, and a void cell on top of solid at up to sqrt(eps)/2.
SMIN_TOL = 0.5 * math.sqrt(1e-4)


def am_active(seed: int) -> np.ndarray:
    active = np.random.default_rng(seed).random(AM_SHAPE) > 0.2
    assert active.mean() >= 0.7
    return active


def orient(a: np.ndarray, direction: str) -> np.ndarray:
    """Map an array built for '+z' (base plate at iz = 0) to build `direction`."""
    ax = "xyz".index(direction[1])
    out = np.moveaxis(a, 2, ax)
    return np.flip(out, ax) if direction[0] == "-" else out


def back_to_z(a: np.ndarray, direction: str) -> np.ndarray:
    """Inverse of orient()."""
    ax = "xyz".index(direction[1])
    a = np.flip(a, ax) if direction[0] == "-" else a
    return np.moveaxis(a, ax, 2)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_am_filter_gradient_matches_finite_differences(direction):
    rng = np.random.default_rng(100 + DIRECTIONS.index(direction))
    active = am_active(DIRECTIONS.index(direction))
    x = rng.random(AM_SHAPE)
    w = rng.standard_normal(AM_SHAPE)
    filt = AMFilter(active, direction)
    filt.apply(x)
    grad = filt.backprop(w)
    assert np.all(grad[~active] == 0.0)
    cells = rng.choice(np.flatnonzero(active), 30, replace=False)
    h = 1e-6
    fd = np.empty(len(cells))
    for i, c in enumerate(cells):
        xp, xm = x.copy(), x.copy()
        xp.flat[c] += h
        xm.flat[c] -= h
        fd[i] = (np.vdot(w, filt.apply(xp)) - np.vdot(w, filt.apply(xm))) / (2 * h)
    ad = grad.flat[cells]
    assert np.abs(fd - ad).max() / np.abs(ad).max() < 1e-5


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_am_filter_solid_block_on_base_plate_is_unchanged(direction):
    shape = (8, 7, 9)
    x = np.zeros(shape)
    x[1:-1, 2:6, :5] = 1.0  # block standing on the +z base plate, void around and above
    x = orient(x, direction)
    active = np.ones(x.shape, dtype=bool)
    out = AMFilter(active, direction).apply(x)
    assert np.abs(out - x).max() <= SMIN_TOL + 1e-12
    # the deviation is purely the smooth-min offset: with eps -> 1e-12 it is below 1e-6
    out = AMFilter(active, direction, eps=1e-12).apply(x)
    assert np.abs(out - x).max() < 1e-6
    out = AMFilter(active, direction).apply(np.ones(x.shape))
    assert np.abs(out - 1.0).max() <= SMIN_TOL + 1e-12


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_am_filter_removes_overhanging_plate_beyond_45_degrees(direction):
    shape, col, top = (9, 7, 7), (2, 3), 4
    x = np.zeros(shape)
    x[col[0], col[1], :top] = 1.0  # supporting column
    x[:, :, top] = 1.0  # one-voxel plate floating above the base plate
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    in_cone = np.hypot(ii - col[0], jj - col[1]) <= 1.0  # column top + its 4 face neighbours
    xo = orient(x, direction)
    out = back_to_z(AMFilter(np.ones(xo.shape, dtype=bool), direction).apply(xo), direction)
    plate = out[:, :, top]
    assert plate[in_cone].min() > 0.9
    assert plate[~in_cone].max() < 0.1
    assert out[col[0], col[1], :top].min() > 0.9


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_am_filter_45_degree_staircase_is_unchanged(direction):
    shape = (11, 4, 8)
    x = np.zeros(shape)
    for k in range(shape[2]):
        x[k : k + 2, :, k] = 1.0  # two cells wide, shifted by one cell per layer
    xo = orient(x, direction)
    out = back_to_z(AMFilter(np.ones(xo.shape, dtype=bool), direction).apply(xo), direction)
    assert out[x == 1.0].min() > 0.95
    assert out[x == 0.0].max() < 0.05


def test_am_filter_q_correction_keeps_half_density():
    # smax(0.5, ..., 0.5) = 0.5 over 5 supporters: interior of a uniform 0.5 field is unchanged
    shape = (12, 12, 5)
    out = AMFilter(np.ones(shape, dtype=bool), "+z").apply(np.full(shape, 0.5))
    for k in range(shape[2]):
        assert np.abs(out[k : -k or None, k : -k or None, k] - 0.5).max() < 1e-12


def test_am_filter_direction_pairs_are_consistent():
    rng = np.random.default_rng(7)
    active = am_active(7)
    x, w = rng.random(AM_SHAPE), rng.standard_normal(AM_SHAPE)

    def run(x, active, w, direction):
        filt = AMFilter(active, direction)
        return filt.apply(x), filt.backprop(w)

    for ax, name in enumerate("xyz"):
        ref = run(x, active, w, "+" + name)
        mir = run(np.flip(x, ax), np.flip(active, ax), np.flip(w, ax), "-" + name)
        for a, b in zip(ref, mir):
            assert np.abs(a - np.flip(b, ax)).max() < 1e-12
    for name, perm in (("x", (2, 1, 0)), ("y", (0, 2, 1))):
        ref = run(x, active, w, "+" + name)
        tr = run(*(np.transpose(a, perm) for a in (x, active, w)), "+z")
        for a, b in zip(ref, tr):
            assert np.abs(a - np.transpose(b, perm)).max() < 1e-12


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_am_filter_inactive_cells_stay_zero(direction):
    rng = np.random.default_rng(11)
    active = am_active(11)
    filt = AMFilter(active, direction)
    for x in (np.ones(AM_SHAPE), rng.random(AM_SHAPE), np.where(active, 0.0, 1.0)):
        out = filt.apply(x)
        assert np.all(out[~active] == 0.0)
        assert np.all(filt.backprop(rng.standard_normal(AM_SHAPE))[~active] == 0.0)
    # mass placed only on inactive cells never supports anything
    assert np.abs(filt.apply(np.where(active, 0.0, 1.0))).max() == 0.0


def test_am_filter_deep_void_has_no_underflow_or_nan():
    # void decays layer by layer to exactly 0; smax must stay finite (v^P alone would underflow)
    shape = (4, 4, 600)
    x = np.zeros(shape)
    x[:, :, 0] = 1.0
    x[1, 1, 300] = 1.0  # isolated floating voxel far above the base
    filt = AMFilter(np.ones(shape, dtype=bool), "+z")
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        out = filt.apply(x)
        grad = filt.backprop(np.ones(shape))
    assert np.all(np.isfinite(grad))
    assert out[1, 1, 300] < SMIN_TOL + 1e-12
    assert np.all(out[:, :, 100:300] == 0.0)  # smin(0, 0) == 0 exactly, no artificial floor


def test_am_filter_timing_60_cubed():
    rng = np.random.default_rng(12)
    shape = (60, 60, 60)
    filt = AMFilter(rng.random(shape) > 0.1, "+z")
    x, w = rng.random(shape), rng.standard_normal(shape)
    t0 = time.perf_counter()
    filt.apply(x)
    g = filt.backprop(w)
    dt = time.perf_counter() - t0
    print(f"AMFilter 60^3 apply + backprop: {dt:.3f} s")
    assert np.all(np.isfinite(g))
    assert dt <= 3.0  # target 1.5 s; slack so CI does not flake
