"""Linear solves for K_free U = F_free: sparse direct for small systems, AMG-preconditioned CG."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from topop.core.problem import SolverKind

# SuperLU is single-threaded: ~0.5 s at 19k DOFs (60x20x4) but ~6 s at 41k (40x30x10, AMG ~1 s)
DIRECT_MAX_DOFS = 20_000


@dataclass
class SolveInfo:
    kind: str
    iterations: int  # CG iterations summed over load cases (0 for direct)
    residual: float  # max over cases of ||F - K U|| / ||F||
    time: float


def _pyamg():
    try:
        import pyamg
    except ImportError:
        return None
    return pyamg


class LinearSolver:
    def __init__(
        self,
        kind: SolverKind = "auto",
        tol: float = 1e-6,
        maxiter: int = 2000,
        reuse: int = 1,
        direct_max_dofs: int = DIRECT_MAX_DOFS,
    ):
        if kind not in ("auto", "amg", "direct"):
            raise ValueError(f"unknown solver kind {kind!r}")
        self.kind = kind
        self.tol = tol
        self.maxiter = maxiter
        self.reuse = max(1, int(reuse))  # rebuild the AMG hierarchy every `reuse` solves
        self.direct_max_dofs = direct_max_dofs
        self.last_ml = None
        self._ml_age = 0

    def select(self, n_free: int) -> Literal["amg", "direct"]:
        kind = self.kind
        if kind == "auto":
            kind = "direct" if n_free < self.direct_max_dofs else "amg"
        if kind == "amg" and _pyamg() is None:
            warnings.warn("pyamg not available, falling back to a direct solve", stacklevel=3)
            kind = "direct"
        return kind

    def solve(
        self,
        K: sp.spmatrix,
        F: np.ndarray,
        x0: np.ndarray | None = None,
        rigid_modes: np.ndarray | None = None,
    ) -> tuple[np.ndarray, SolveInfo]:
        t0 = time.perf_counter()
        F = np.asarray(F)
        Fm = F.reshape(F.shape[0], -1)
        kind = self.select(K.shape[0])
        if kind == "direct":
            U, its = self._direct(K, Fm), 0
        else:
            X0 = None if x0 is None else np.asarray(x0).reshape(Fm.shape)
            U, its = self._amg(K, Fm, X0, rigid_modes)
        R = Fm - K @ U
        fn = np.linalg.norm(Fm, axis=0)
        res = np.linalg.norm(R, axis=0) / np.where(fn > 0, fn, 1.0)
        info = SolveInfo(kind, its, float(res.max(initial=0.0)), time.perf_counter() - t0)
        return U.reshape(F.shape), info

    def _direct(self, K: sp.spmatrix, F: np.ndarray) -> np.ndarray:
        # K is SPD: symmetric mode without pivoting, minimum degree on A^T + A
        lu = sla.splu(
            sp.csc_matrix(K),
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
        return lu.solve(np.asarray(F, dtype=K.dtype))

    def _amg(
        self, K: sp.spmatrix, F: np.ndarray, X0: np.ndarray | None, B: np.ndarray | None
    ) -> tuple[np.ndarray, int]:
        pyamg = _pyamg()
        if self.last_ml is None or self._ml_age >= self.reuse:
            if B is not None:
                B = np.asarray(B, dtype=K.dtype)
            self.last_ml = pyamg.smoothed_aggregation_solver(K, B=B, symmetry="symmetric")
            self._ml_age = 0
        self._ml_age += 1
        M = self.last_ml.aspreconditioner(cycle="V")
        U = np.zeros(F.shape, dtype=K.dtype)
        total = 0
        for c in range(F.shape[1]):
            b = F[:, c].astype(K.dtype)
            if not np.any(b):
                continue
            count = [0]

            def cb(_xk, count=count):
                count[0] += 1

            x0 = None if X0 is None else X0[:, c].astype(K.dtype)
            x, status = sla.cg(K, b, x0=x0, rtol=self.tol, maxiter=self.maxiter, M=M, callback=cb)
            if status > 0:
                warnings.warn(
                    f"CG did not reach rtol={self.tol} in {self.maxiter} iterations",
                    stacklevel=3,
                )
            U[:, c] = x
            total += count[0]
        return U, total
