"""Density filter (cone kernel, normalized over active cells) and Heaviside projection."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage, signal

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
        return np.where(self.active, self._conv(x * self._mask) / self.Hs, 0.0)

    def apply_adjoint(self, y: np.ndarray) -> np.ndarray:
        """H^T y = A conv(A y / Hs) (the cone kernel is symmetric)."""
        y = np.asarray(y, dtype=np.float64).reshape(self.shape)
        return np.where(self.active, self._conv(y * self._mask / self.Hs), 0.0)


def heaviside(x_tilde: np.ndarray, beta: float, eta: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(tanh(b*eta) + tanh(b*(x-eta))) / (tanh(b*eta) + tanh(b*(1-eta))), Wang et al. 2011."""
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    t = np.tanh(beta * (np.asarray(x_tilde) - eta))
    return (np.tanh(beta * eta) + t) / den, beta * (1 - t * t) / den
