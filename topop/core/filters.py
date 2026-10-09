"""Density filter (cone kernel, normalized over active cells), Heaviside projection, AM filter."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage, signal

from topop.core.problem import Direction

FFT_MIN_RMIN = 2.0  # rmin above this -> FFT convolution, else direct ndimage.convolve


def cone_kernel(rmin: float) -> np.ndarray:
    r = math.ceil(rmin)
    d = np.arange(-r, r + 1, dtype=np.float64)
    dist = np.sqrt(d[:, None, None] ** 2 + d[None, :, None] ** 2 + d[None, None, :] ** 2)
    return np.maximum(0.0, rmin - dist)


class DensityFilter:
    """x_tilde = conv(x * active) / Hs on active cells, Hs = conv(active); 0 elsewhere."""

    def __init__(self, shape: tuple[int, int, int], rmin: float, active: np.ndarray):
        if rmin <= 0:
            raise ValueError("rmin must be positive")
        self.shape = tuple(shape)
        self.rmin = float(rmin)
        self.active = np.asarray(active, dtype=bool).reshape(self.shape)
        self.kernel = cone_kernel(rmin)
        self.use_fft = rmin > FFT_MIN_RMIN
        self._mask = self.active.astype(np.float64)
        hs = self._conv(self._mask)
        self.Hs = np.where(self.active, hs, 1.0)  # active cells always see their own weight rmin

    def _conv(self, x: np.ndarray) -> np.ndarray:
        if self.use_fft:
            return signal.fftconvolve(x, self.kernel, mode="same")
        return ndimage.convolve(x, self.kernel, mode="constant", cval=0.0)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(self.shape)
        out = np.where(self.active, self._conv(x * self._mask) / self.Hs, 0.0)
        if self.use_fft:  # FFT round-off: ~1e-17 below 0 next to void, above 1 inside solid
            np.clip(out, 0.0, 1.0, out=out)
        return out

    def apply_adjoint(self, y: np.ndarray) -> np.ndarray:
        """H^T y = A conv(A y / Hs) (the cone kernel is symmetric). Linear: never clipped."""
        y = np.asarray(y, dtype=np.float64).reshape(self.shape)
        return np.where(self.active, self._conv(y * self._mask / self.Hs), 0.0)


def heaviside(x_tilde: np.ndarray, beta: float, eta: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(tanh(b*eta) + tanh(b*(x-eta))) / (tanh(b*eta) + tanh(b*(1-eta))), Wang et al. 2011."""
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    t = np.tanh(beta * (np.asarray(x_tilde) - eta))
    return (np.tanh(beta * eta) + t) / den, beta * (1 - t * t) / den


AM_SUPPORT_N = 5  # cell below + its 4 face neighbours: 45 deg self-support on a cubic grid
AM_XI0 = 0.5  # Langelaar's Q correction makes smax(0.5, ..., 0.5) == 0.5
_AM_DIRECTIONS = ("+x", "-x", "+y", "-y", "+z", "-z")


class AMFilter:
    """Langelaar's additive-manufacturing overhang filter (SMO 55, 2017), 45 deg rule.

    Printed density, layer by layer from the base plate (first layer fully supported):
    xi_i = smin(x_i, smax(xi over the support set of i in the layer below)).
    Inactive cells are 0 as input, as support and as output.
    """

    def __init__(
        self, active: np.ndarray, direction: Direction, P: float = 40.0, eps: float = 1e-4
    ):
        if direction not in _AM_DIRECTIONS:
            raise ValueError(f"direction must be one of {_AM_DIRECTIONS}, got {direction!r}")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.P = float(P)
        self.Q = self.P + math.log(AM_SUPPORT_N) / math.log(AM_XI0)
        if self.Q <= 0:
            raise ValueError("P too small: Q = P + log(5)/log(0.5) must be positive")
        self.eps = float(eps)
        self.direction = direction
        self.active = np.asarray(active, dtype=bool)
        if self.active.ndim != 3:
            raise ValueError("active must be a 3D (nx, ny, nz) array")
        self.shape = self.active.shape
        self._axis = "xyz".index(direction[1])
        self._flip = direction[0] == "-"
        # Internal frame: build axis first (contiguous layers), base plate at layer 0.
        self._act = self._to_canon(self.active)
        # Base plate = first layer holding an active cell (padding layers below it are empty).
        has = self._act.reshape(self._act.shape[0], -1).any(axis=1)
        self._k0 = int(np.argmax(has)) if has.any() else 0
        self._xi: np.ndarray | None = None
        self._dxi_dx: np.ndarray | None = None
        self._dxi_dn: np.ndarray | None = None  # dxi/dsmax * dsmax/dN without the R^(P-1) factor

    def _to_canon(self, a: np.ndarray) -> np.ndarray:
        c = np.moveaxis(a, self._axis, 0)
        return np.ascontiguousarray(c[::-1] if self._flip else c)

    def _from_canon(self, c: np.ndarray) -> np.ndarray:
        c = c[::-1] if self._flip else c
        return np.array(np.moveaxis(c, 0, self._axis), order="C")  # always a fresh copy

    @staticmethod
    def _support_slices(pad: np.ndarray) -> tuple[np.ndarray, ...]:
        # center, -a, +a, -b, +b views of a 1-cell zero-padded layer (out of grid = 0)
        return (pad[1:-1, 1:-1], pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:])

    def _support_powers(self, below: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each cell: m = max supporter, R^(P-1) per supporter (5,A,B), T = sum R^P.

        smax = (sum v^P)^(1/Q) is evaluated as m^(P/Q) T^(1/Q) with R = v/m: exact, with no 0/0
        in the gradient when every v^P underflows (v < 1e-8 at P = 40; void decays to that within
        a few layers). T >= 1 unless m == 0, where T == 0 and smax == 0 exactly.
        """
        pad = np.zeros((below.shape[0] + 2, below.shape[1] + 2))
        np.maximum(below, 0.0, out=pad[1:-1, 1:-1])  # guards R^(P-1) against x slightly < 0
        n = np.stack(self._support_slices(pad))
        m = n.max(axis=0)
        r = n / np.where(m > 0.0, m, 1.0)
        rp1 = r ** (self.P - 1.0)
        return m, rp1, (rp1 * r).sum(axis=0)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(self.shape)
        act = self._act
        xa = np.where(act, self._to_canon(x), 0.0)
        P, Q, eps = self.P, self.Q, self.eps
        sq_eps = math.sqrt(eps)
        xi = np.empty_like(xa)
        dxi_dx = np.zeros_like(xa)
        dxi_dn = np.zeros_like(xa)
        k0 = self._k0
        xi[: k0 + 1] = xa[: k0 + 1]
        dxi_dx[: k0 + 1] = act[: k0 + 1]
        for k in range(k0 + 1, xa.shape[0]):
            m, _, t = self._support_powers(xi[k - 1])
            s = m ** (P / Q) * t ** (1.0 / Q)
            d = xa[k] - s
            root = np.sqrt(d * d + eps)
            xi[k] = np.where(act[k], 0.5 * (xa[k] + s - root + sq_eps), 0.0)
            q = d / root
            dxi_dx[k] = np.where(act[k], 0.5 * (1.0 - q), 0.0)
            # dsmax/dv_j = (P/Q) m^(P/Q-1) T^(1/Q-1) R_j^(P-1); R_j^(P-1) is recomputed in backprop
            ds_dn = (P / Q) * m ** (P / Q - 1.0) * np.where(t > 0.0, t, 1.0) ** (1.0 / Q - 1.0)
            dxi_dn[k] = np.where(act[k], 0.5 * (1.0 + q) * ds_dn, 0.0)
        self._xi, self._dxi_dx, self._dxi_dn = xi, dxi_dx, dxi_dn
        return self._from_canon(xi)

    def backprop(self, g_out: np.ndarray) -> np.ndarray:
        """df/dx for the last apply(), given df/dxi (full grid). Accumulates top-down."""
        if self._xi is None or self._dxi_dx is None or self._dxi_dn is None:
            raise RuntimeError("AMFilter.backprop called before apply")
        xi, dxi_dx, dxi_dn = self._xi, self._dxi_dx, self._dxi_dn
        g = self._to_canon(np.asarray(g_out, dtype=np.float64).reshape(self.shape)).copy()
        dx = np.empty_like(g)
        n_a, n_b = g.shape[1:]
        k0 = self._k0
        for k in range(g.shape[0] - 1, k0, -1):
            dx[k] = g[k] * dxi_dx[k]
            w = g[k] * dxi_dn[k]
            _, rp1, _ = self._support_powers(xi[k - 1])
            gpad = np.zeros((n_a + 2, n_b + 2))
            for view, contrib in zip(self._support_slices(gpad), rp1 * w):
                view += contrib  # transpose of the support gather
            g[k - 1] += np.where(xi[k - 1] > 0.0, gpad[1:-1, 1:-1], 0.0)
        dx[: k0 + 1] = g[: k0 + 1] * dxi_dx[: k0 + 1]
        return self._from_canon(dx)
