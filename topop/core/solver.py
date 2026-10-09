"""Linear solves for K_free U = F_free.

- direct: banded Cholesky (LAPACK pbtrf, buffer reused across solves) when the bandwidth in the
  given ordering is small, else SuperLU.
- amg: multigrid-preconditioned CG. With geometric `prolongators` (trilinear interpolation between
  voxel grids, see `Assembler.prolongators`): Galerkin coarse operators refreshed every solve,
  Jacobi-scaled Chebyshev smoothing and row-block threaded SpMV (scipy releases the GIL). Without
  them: pyamg smoothed aggregation, whose aggregates/prolongators are kept for up to `reuse` solves
  while the operators and smoothers are refreshed from the new matrix.

The geometric V-cycle is SPD only while the Chebyshev upper bound covers lambda_max(D^-1 A) of
every level: modes above (1 + lo) * lmax are amplified and CG breaks down. lmax is a fresh Lanczos
estimate on every refresh (`GeometricMG._lmax`). If CG still breaks down, `solve` restarts from
zero, then retries with Gershgorin bounds (SPD by construction), then Jacobi-PCG; `SolveInfo.method`
says which one produced the result.

Reusing a whole hierarchy (stale fine/coarse operators) does not work for SIMP: void elements
change stiffness by up to 1e7 between iterations and CG needs 600 to >2000 iterations; reusing
only the prolongators costs a few extra CG iterations. See docs/PERF.md.
"""

from __future__ import annotations

import itertools
import os
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.linalg import lapack

from topop.core.problem import SolverKind

try:  # in-place, GIL-free CSR kernel (y += A x); falls back to slicing + scipy's matvec
    from scipy.sparse._sparsetools import csr_matvec as _csr_matvec
except ImportError:  # pragma: no cover
    _csr_matvec = None

# SuperLU is single-threaded: ~0.5 s at 19k DOFs (60x20x4) but ~6 s at 41k (40x30x10, AMG ~1 s).
# `select` keeps this split; "direct" itself prefers the banded Cholesky when it is cheap.
DIRECT_MAX_DOFS = 20_000
# auto: banded Cholesky from the first solve below this bandwidth (it beats geometric MG there
# already at the first SIMP iterations; crossover measured on thin beams, PERF.md v0.3) ...
BAND_AUTO_BW = 410
# ... and later, when multigrid CG needs more iterations per solve (min over BAND_SWITCH_WINDOW
# solves) than the factorization costs: band ~ BAND_US_PER_BW * bw - BAND_US_OFFSET microseconds
# per unknown (LAPACK pbtrf, measured at bw 335-677), GMG ~ GMG_US_PER_IT per unknown and CG
# iteration (setup not counted, so the switch is conservative). GMG's count grows as void regions
# form (10 -> 20-50 on an 80x16x8 beam, bw 491), so thin parts end up on the band.
BAND_US_PER_BW, BAND_US_OFFSET = 0.0285, 3.2
GMG_US_PER_IT = 0.7
BAND_SWITCH_WINDOW = 2
# "direct" uses the band (instead of SuperLU) up to this n bw^2 and band storage
BAND_MAX_WORK = 4.0e10
BAND_MAX_BYTES = 512 * 2**20
BAND_MAX_DOFS = 170_000  # auto never estimates the band above this many unknowns (512 MB at bw 380)
# rebuild the SA aggregation when CG needs more than this times the count right after a rebuild
REBUILD_ITER_RATIO = 2.0
DEFAULT_REUSE = 20
# adaptive CG tolerance: `tol` for change <= CHANGE_TIGHT, `adaptive_tol` for change >= CHANGE_LOOSE,
# geometric in between
CHANGE_TIGHT, CHANGE_LOOSE = 0.02, 0.1
COARSE_DENSE_MAX = 3000
PAR_MIN_NNZ = 200_000  # below this a level runs serially (thread hand-off costs ~50 us)
# Chebyshev upper bound = LMAX_BOOST * (largest Ritz value + its residual) after LANCZOS_STEPS
# Lanczos steps. That estimate was >= 0.967 lambda_max(D^-1 A) on every level of the L-bracket
# stress run (0.937 at 6 steps, 0.998 at 10); the cycle stays SPD down to 1 / 1.21 = 0.826.
LANCZOS_STEPS = 8
LMAX_BOOST = 1.1
# fallback chain of a broken-down geometric-MG CG, in escalation order (SolveInfo.method)
GMG_METHODS = ("gmg", "gmg-restart", "gmg-safe", "jacobi-pcg")
JACOBI_MAXITER_FACTOR = 5  # Jacobi-PCG (last resort) gets this many times `maxiter`
# a direct solve whose relative residual exceeds this came from a (numerically) singular K:
# LAPACK/SuperLU happily factor a matrix with a free rigid-body mode and return garbage
DIRECT_MAX_RESIDUAL = 1e-6
SINGULAR_MESSAGE = "stiffness matrix is singular or badly conditioned: check supports"


@dataclass
class SolveInfo:
    kind: str
    iterations: int  # CG iterations summed over load cases (0 for direct)
    residual: float  # max over cases of ||F - K U|| / ||F||
    time: float
    method: str = ""  # band | splu | gmg | sa | a GMG fallback (GMG_METHODS)
    rtol: float = 0.0  # CG tolerance used (0 for direct)
    setup_time: float = 0.0  # factorization / hierarchy refresh part of `time`


def _pyamg():
    try:
        import pyamg
    except ImportError:
        return None
    return pyamg


def default_threads() -> int:
    env = os.environ.get("TOPOP_THREADS")
    if env:
        return max(1, int(env))
    return max(1, min(8, os.cpu_count() or 1))


_POOLS: dict[int, ThreadPoolExecutor] = {}


def _pool(n: int) -> ThreadPoolExecutor:
    if n not in _POOLS:
        _POOLS[n] = ThreadPoolExecutor(n, thread_name_prefix="topop-solve")
    return _POOLS[n]


def _resolve(v):
    return v() if callable(v) else v


class _BlasThreads:
    """Pin OpenBLAS (numpy's and scipy's copies) to `n` threads inside a `with` block.

    The solves call BLAS only on small operands (dots, norms, coarse/banded Cholesky) between
    threaded sparse kernels; a multithreaded OpenBLAS then spends milliseconds per call waking
    its pool (6 ms for an 18k-element dot product on the CI VM). Best effort: does nothing when
    no OpenBLAS is loaded (e.g. Accelerate on macOS) or its symbols are not found.
    """

    def __init__(self, n: int = 1):
        import threading

        self.n = n
        self._fns: list[tuple] | None = None
        self._saved: list[int] = []
        self._depth = 0
        self._lock = threading.Lock()

    def _discover(self) -> list[tuple]:
        import ctypes

        paths: set[str] = set()
        try:
            with open("/proc/self/maps") as f:
                for line in f:
                    parts = line.split(None, 5)
                    if len(parts) == 6 and "openblas" in os.path.basename(parts[5]).lower():
                        paths.add(parts[5].strip())
        except OSError:
            try:  # macOS
                libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
                libc._dyld_get_image_name.restype = ctypes.c_char_p
                for i in range(libc._dyld_image_count()):
                    name = libc._dyld_get_image_name(i)
                    if name and b"openblas" in os.path.basename(name).lower():
                        paths.add(name.decode())
            except (OSError, AttributeError):
                pass
        fns = []
        for path in sorted(paths):
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            for pre in ("scipy_openblas_", "openblas_"):
                for suf in ("64_", ""):
                    get = getattr(lib, f"{pre}get_num_threads{suf}", None)
                    put = getattr(lib, f"{pre}set_num_threads{suf}", None)
                    if get is not None and put is not None:
                        get.restype = ctypes.c_int
                        put.argtypes = [ctypes.c_int]
                        fns.append((get, put))
                        break
                else:
                    continue
                break
        return fns

    def __enter__(self):
        with self._lock:
            self._enter()
        return self

    def _enter(self):
        if self._depth == 0:
            if self._fns is None:
                try:
                    self._fns = self._discover()
                except Exception:  # noqa: BLE001 -- purely an optimization
                    self._fns = []
            self._saved = [get() for get, _ in self._fns]
            for _, put in self._fns:
                put(self.n)
        self._depth += 1

    def __exit__(self, *exc):
        with self._lock:
            self._depth -= 1
            if self._depth == 0:
                for (_, put), n in zip(self._fns, self._saved, strict=True):
                    put(n)
        return False


_BLAS = _BlasThreads(int(os.environ.get("TOPOP_BLAS_THREADS", "1")))


class _Blocks:
    """Row blocks of a CSR matrix with ~equal nnz; kernels run one task per block."""

    def __init__(self, A: sp.csr_matrix, threads: int):
        if A.indptr.dtype != A.indices.dtype:
            A.indptr = A.indptr.astype(A.indices.dtype)
        self.A = A
        n, nnz = A.shape[0], A.nnz
        self.threads = threads
        k = 1 if threads <= 1 or nnz < PAR_MIN_NNZ else min(4 * threads, max(2, nnz // 50_000))
        b = np.searchsorted(A.indptr, np.linspace(0, nnz, k + 1)[1:-1])
        b = np.unique(np.concatenate([[0], b, [n]]))
        self.blocks = [(int(lo), int(hi)) for lo, hi in itertools.pairwise(b) if hi > lo] or [
            (0, n)
        ]

    def run(self, fn: Callable[[int, int], None]) -> None:
        if len(self.blocks) == 1:
            fn(*self.blocks[0])
        else:
            list(_pool(self.threads).map(lambda ab: fn(*ab), self.blocks))

    def matvec_block(self, x: np.ndarray, y: np.ndarray, a: int, b: int) -> None:
        """y[a:b] = A[a:b] @ x (x, y of the matrix dtype)."""
        A = self.A
        y[a:b] = 0
        if _csr_matvec is not None:
            _csr_matvec(b - a, A.shape[1], A.indptr[a : b + 1], A.indices, A.data, x, y[a:b])
        else:  # pragma: no cover
            y[a:b] = A[a:b] @ x

    def matvec(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.run(lambda a, b: self.matvec_block(x, y, a, b))
        return y

    def matvec64(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """float64 y = A @ x for a float32 A, converting one block of A.data at a time."""
        A = self.A

        def block(a, b):
            s, e = A.indptr[a], A.indptr[b]
            y[a:b] = 0
            if _csr_matvec is not None:
                ptr = A.indptr[a : b + 1] - s
                data = A.data[s:e].astype(np.float64)
                _csr_matvec(b - a, A.shape[1], ptr, A.indices[s:e], data, x, y[a:b])
            else:  # pragma: no cover
                y[a:b] = A[a:b].astype(np.float64) @ x

        self.run(block)
        return y


def _csr_rows(M: sp.csr_matrix, lo: int, hi: int) -> sp.csr_matrix:
    s, e = M.indptr[lo], M.indptr[hi]
    return sp.csr_matrix(
        (M.data[s:e], M.indices[s:e], M.indptr[lo : hi + 1] - s), shape=(hi - lo, M.shape[1])
    )


def _jacobi_gershgorin(A: sp.csr_matrix) -> float:
    """Upper bound of lambda_max(D^-1 A): the smaller max row sum of D^-1 |A| and D^-1/2 |A| D^-1/2
    (both similar to D^-1 A up to signs, and any induced norm bounds the spectral radius)."""
    d = A.diagonal().astype(np.float64)
    absA = abs(A).astype(np.float64)
    s = 1.0 / np.sqrt(np.where(d > 0, d, np.inf))
    g1 = absA @ np.ones(A.shape[0]) * s * s
    g2 = s * (absA @ s)
    return max(float(min(g1.max(initial=0.0), g2.max(initial=0.0))), 1e-12)


def _galerkin(R: sp.csr_matrix, A: sp.csr_matrix, P: sp.csr_matrix, threads: int) -> sp.csr_matrix:
    """R A P as (R_j A) P over row blocks j of R in threads (scipy's SpGEMM releases the GIL).

    Blocking by coarse rows keeps the large intermediate per thread; stacking a full A P
    (22M nnz at 100k elements) costs more than the products themselves.
    """
    blk = _Blocks(R, threads)
    if len(blk.blocks) == 1:
        return (R @ A @ P).tocsr()
    parts = list(_pool(threads).map(lambda ab: (_csr_rows(R, *ab) @ A) @ P, blk.blocks))
    return sp.vstack(parts, format="csr")


class GeometricMG:
    """V-cycle preconditioner on fixed prolongators; `update(K)` refreshes the operators.

    Level 0 stays in K's dtype (float32 runs smooth in float32), coarser levels are float64.
    Smoother: Chebyshev of `degree` on D^-1 A over [lo * lmax, lmax], lmax = 1.1 * Lanczos estimate
    of lambda_max (`make_safe`: Gershgorin bound). Same polynomial before and after the coarse
    correction and R = P^T, so the cycle is symmetric; it is positive definite while every level's
    lambda_max(D^-1 A) <= (1 + lo) * lmax.
    """

    def __init__(
        self,
        prolongators: Sequence[sp.csr_matrix],
        degree: int = 2,
        lo: float = 0.1,
        threads: int | None = None,
    ):
        self.P = [sp.csr_matrix(P) for P in prolongators]
        self.R = [P.T.tocsr() for P in self.P]
        self._cast: dict = {}  # level-0 transfer operators in a float32 run's dtype
        self.degree = int(degree)
        self.lo = float(lo)
        self.threads = default_threads() if threads is None else max(1, int(threads))
        self._start: dict[int, np.ndarray] = {}  # fixed random Lanczos start vector per level
        self.levels: list[dict] = []
        self.coarse = None

    @property
    def n_levels(self) -> int:
        return len(self.P) + 1

    def update(self, K: sp.csr_matrix) -> None:
        th = self.threads
        levels = []
        A = K
        for lvl, (P, R) in enumerate(zip(self.P, self.R)):
            if P.dtype != A.dtype:  # level 0 of a float32 run
                key = (lvl, A.dtype.str)
                if key not in self._cast:
                    self._cast[key] = (P.astype(A.dtype), R.astype(A.dtype))
                P, R = self._cast[key]
            blk = _Blocks(A, th)
            d = A.diagonal()
            dinv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(A.dtype)
            L = {"A": blk, "P": _Blocks(P, th), "R": _Blocks(R, th), "dinv": dinv}
            n = A.shape[0]
            for name in ("x", "r", "y", "d", "dn", "b"):
                L[name] = np.empty(n, dtype=A.dtype)
            L["lmax"] = LMAX_BOOST * self._lmax(lvl, blk, dinv, L)
            levels.append(L)
            A = _galerkin(R, A, P, th)
            if A.dtype != np.float64:
                A = A.astype(np.float64)
        self.levels = levels
        self.coarse = self._factor_coarse(A)
        self.coarse_n = A.shape[0]

    def _lmax(self, lvl: int, blk: _Blocks, dinv: np.ndarray, L: dict) -> float:
        """lambda_max(D^-1 A) from above in practice: largest Ritz value + its residual norm.

        Lanczos on D^-1/2 A D^-1/2 (three-term recurrence, no stored basis) from a fixed random
        start. Not warm-started: the previous solve's dominant vector is often localized on a
        void/solid interface that has since moved, and iterating from it found half of
        lambda_max on the L-bracket stress run (solver breakdown, see tests/test_solver.py).
        """
        n = dinv.size
        steps = min(LANCZOS_STEPS, n)
        if steps == 0:
            return 1.0
        q = self._start.get(lvl)
        if q is None or q.size != n or q.dtype != dinv.dtype:
            v = np.random.default_rng(lvl).standard_normal(n)
            q = self._start[lvl] = (v / np.linalg.norm(v)).astype(dinv.dtype)
        s = np.sqrt(dinv)
        # the V-cycle's level buffers are free until its first call; no temporaries
        w, y, bufs = L["d"], L["y"], (L["x"], L["r"])
        axpy = sl.blas.get_blas_funcs("axpy", (y,))
        alpha, beta = np.zeros(steps), np.zeros(steps)
        q_prev = None
        for j in range(steps):
            np.multiply(s, q, out=w)
            blk.matvec(w, y)
            y *= s
            alpha[j] = a = float(q @ y)
            axpy(q, y, a=-a)
            if q_prev is not None:
                axpy(q_prev, y, a=-beta[j - 1])
            beta[j] = b = float(np.linalg.norm(y))
            if not b > 1e-10 * abs(a):  # invariant subspace: the Ritz values are exact
                steps = j + 1
                break
            # bufs[j % 2] holds q_prev (already used) from step 2 on
            q_prev, q = q, np.multiply(y, 1.0 / b, out=bufs[j % 2])
        if steps == 1:
            return max(alpha[0] + beta[0], 1e-12)
        vals, vecs = sl.eigh_tridiagonal(alpha[:steps], beta[: steps - 1])
        i = int(np.argmax(vals))
        return max(float(vals[i] + abs(beta[steps - 1] * vecs[-1, i])), 1e-12)

    def make_safe(self) -> None:
        """Chebyshev bounds from Gershgorin's theorem until the next `update`: never below
        lambda_max, so the cycle is SPD by construction (1.3-2x loose: weaker smoothing)."""
        for L in self.levels:
            L["lmax"] = _jacobi_gershgorin(L["A"].A)

    @staticmethod
    def _factor_coarse(A: sp.csr_matrix):
        n = A.shape[0]
        if n == 0:
            return ("empty", None)
        if n <= COARSE_DENSE_MAX:
            M = A.toarray()
            try:
                return ("chol", sl.cho_factor(M, lower=False, check_finite=False))
            except np.linalg.LinAlgError:
                return ("pinv", sl.pinvh(M, check_finite=False))
        return ("splu", sla.splu(sp.csc_matrix(A)))

    def _coarse_solve(self, b: np.ndarray) -> np.ndarray:
        kind, f = self.coarse
        if kind == "chol":
            return sl.cho_solve(f, b, check_finite=False)
        if kind == "pinv":
            return f @ b
        if kind == "splu":
            return f.solve(b)
        return b * 0

    def _smooth(self, L: dict, x: np.ndarray, b: np.ndarray, zero_start: bool) -> None:
        A, dinv, r, y, d, dn = L["A"], L["dinv"], L["r"], L["y"], L["d"], L["dn"]
        lmax = L["lmax"]
        lmin = self.lo * lmax
        theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
        sigma = theta / delta
        rho = 1.0 / sigma
        dt = dinv.dtype.type

        def first(a, c):
            if zero_start:
                np.multiply(dinv[a:c], b[a:c], out=r[a:c])
                x[a:c] = 0
            else:
                A.matvec_block(x, y, a, c)
                np.subtract(b[a:c], y[a:c], out=r[a:c])
                r[a:c] *= dinv[a:c]
            np.multiply(r[a:c], dt(1.0 / theta), out=d[a:c])

        A.run(first)
        for _ in range(self.degree - 1):
            rho_new = 1.0 / (2.0 * sigma - rho)
            c1, c2 = dt(rho_new * rho), dt(2.0 * rho_new / delta)

            def step(a, c, c1=c1, c2=c2, d=d, dn=dn):
                A.matvec_block(d, y, a, c)
                y[a:c] *= dinv[a:c]
                r[a:c] -= y[a:c]
                x[a:c] += d[a:c]
                np.multiply(d[a:c], c1, out=dn[a:c])
                dn[a:c] += c2 * r[a:c]

            A.run(step)
            d, dn = dn, d
            rho = rho_new
        x += d

    def _vcycle(self, lvl: int, b: np.ndarray) -> np.ndarray:
        if lvl == len(self.levels):
            return self._coarse_solve(b)
        L = self.levels[lvl]
        x, res = L["x"], L["b"]
        self._smooth(L, x, b, zero_start=True)
        A = L["A"]

        def residual(a, c):
            A.matvec_block(x, res, a, c)
            np.subtract(b[a:c], res[a:c], out=res[a:c])

        A.run(residual)
        R = L["R"]
        bc = np.empty(R.A.shape[0], dtype=res.dtype)
        R.matvec(res, bc)
        nxt = self.levels[lvl + 1]["A"].A.dtype if lvl + 1 < len(self.levels) else np.float64
        xc = self._vcycle(lvl + 1, bc.astype(nxt, copy=False))
        xc = xc.astype(x.dtype, copy=False)
        P = L["P"]

        def prolong(a, c):
            if _csr_matvec is not None:
                M = P.A
                _csr_matvec(c - a, M.shape[1], M.indptr[a : c + 1], M.indices, M.data, xc, x[a:c])
            else:  # pragma: no cover
                x[a:c] += P.A[a:c] @ xc

        P.run(prolong)
        self._smooth(L, x, b, zero_start=False)
        return x

    def __call__(self, r: np.ndarray) -> np.ndarray:
        if not self.levels:
            return self._coarse_solve(np.asarray(r, dtype=np.float64))
        b0 = np.asarray(r, dtype=self.levels[0]["r"].dtype)  # read-only in the cycle
        return np.array(self._vcycle(0, b0), dtype=np.float64)  # copy out of level buffers


class _BandCholesky:
    """Banded Cholesky on a fixed sparsity pattern; the (bw+1, n) LAPACK buffer is reused."""

    def __init__(self, K: sp.csr_matrix, perm: np.ndarray | None):
        n = K.shape[0]
        rows = np.repeat(np.arange(n), np.diff(K.indptr))
        cols = K.indices.astype(np.int64)
        if perm is not None:
            inv = np.empty(n, dtype=np.int64)
            inv[perm] = np.arange(n)
            rows, cols = inv[rows], inv[cols]
        up = np.flatnonzero(rows <= cols)
        self.bw = int((cols[up] - rows[up]).max(initial=0))
        self.n = n
        self.perm = perm
        self.src = up
        self.pos = (self.bw + rows[up] - cols[up]) + (self.bw + 1) * cols[up]
        self.ab = None

    @staticmethod
    def estimate(K: sp.csr_matrix, perm: np.ndarray | None) -> tuple[int, float, int]:
        """(bandwidth, n * bw^2, band bytes) of K in ordering `perm`."""
        n = K.shape[0]
        rows = np.repeat(np.arange(n), np.diff(K.indptr))
        cols = K.indices
        if perm is not None:
            inv = np.empty(n, dtype=np.int64)
            inv[perm] = np.arange(n)
            rows, cols = inv[rows], inv[cols]
        bw = int(np.abs(cols - rows).max(initial=0))
        return bw, float(n) * bw * bw, 8 * n * (bw + 1)

    def solve(self, K: sp.csr_matrix, F: np.ndarray) -> np.ndarray | None:
        if self.ab is None:
            self.ab = np.zeros((self.bw + 1, self.n), order="F")
        ab = self.ab
        ab[:] = 0.0
        ab.reshape(-1, order="F")[self.pos] = K.data[self.src]
        c, info = lapack.dpbtrf(ab, lower=0, overwrite_ab=1)
        if info != 0:
            return None
        rhs = F if self.perm is None else F[self.perm]
        x, info = lapack.dpbtrs(c, np.asarray(rhs, dtype=np.float64), lower=0)
        if info != 0:
            return None
        if self.perm is None:
            return x
        out = np.empty_like(x)
        out[self.perm] = x
        return out


class LinearSolver:
    """K U = F for the free DOFs, see the module docstring.

    `prolongators` (list or zero-argument callable) enables geometric multigrid for "amg";
    `ordering` (permutation or callable) is a bandwidth-reducing order for the banded Cholesky.
    `solve(..., change=)` loosens the CG tolerance while the design still moves a lot when
    `adaptive_tol` (the loose rtol) is set: rtol goes geometrically from `tol` at change <= 0.02
    to `adaptive_tol` at change >= 0.1 (change = the optimizer's last max |delta x|).
    """

    def __init__(
        self,
        kind: SolverKind = "auto",
        tol: float = 1e-6,
        maxiter: int = 2000,
        reuse: int = DEFAULT_REUSE,
        direct_max_dofs: int = DIRECT_MAX_DOFS,
        *,
        prolongators: Sequence[sp.spmatrix] | Callable[[], Sequence[sp.spmatrix]] | None = None,
        ordering: np.ndarray | Callable[[], np.ndarray | None] | None = None,
        adaptive_tol: float | None = None,
        threads: int | None = None,
        mg_degree: int = 2,
    ):
        if kind not in ("auto", "amg", "direct"):
            raise ValueError(f"unknown solver kind {kind!r}")
        self.kind = kind
        self.tol = tol
        self.maxiter = maxiter
        # SA: keep aggregates/prolongators for up to `reuse` solves (operators always refreshed)
        self.reuse = max(1, int(reuse))
        self.direct_max_dofs = direct_max_dofs
        self.prolongators = prolongators
        self.ordering = ordering
        self.adaptive_tol = adaptive_tol
        self.threads = default_threads() if threads is None else max(1, int(threads))
        self.mg_degree = mg_degree
        self.last_ml = None
        self.last_method = ""
        self._ml_age = 0
        self._ml_base_its: int | None = None
        self._ml_rebuild = False
        self._mg: GeometricMG | None = None
        self._band: _BandCholesky | None = None
        self._band_key = None
        self._plan_key = None
        self._plan: str | None = None
        self._switch_its: float | None = None  # auto: GMG -> band above this many CG its per solve
        self._gmg_its: list[int] = []

    def select(self, n_free: int) -> Literal["amg", "direct"]:
        kind = self.kind
        if kind == "auto":
            kind = "direct" if n_free < self.direct_max_dofs else "amg"
        if kind == "amg" and self.prolongators is None and _pyamg() is None:
            warnings.warn("pyamg not available, falling back to a direct solve", stacklevel=3)
            kind = "direct"
        return kind

    def needs_rigid_modes(self, n_free: int) -> bool:
        """True when the solve will use pyamg SA (the only method that uses rigid body modes)."""
        return self.prolongators is None and self.select(n_free) == "amg"

    # -- method choice -------------------------------------------------------------------------

    def _band_plan(self, K: sp.spmatrix) -> tuple[int, float, int]:
        key = (K.shape, K.nnz, id(getattr(K, "indices", None)))
        if self._band_key != key:
            perm = _resolve(self.ordering)
            if perm is not None and (
                len(perm) != K.shape[0] or np.array_equal(perm, np.arange(len(perm)))
            ):
                perm = None
            self._band_perm = perm
            self._band_est = _BandCholesky.estimate(sp.csr_matrix(K), perm)
            self._band_key = key
            self._band = None
        return self._band_est

    def method(self, K: sp.spmatrix) -> str:
        """band | splu | gmg | sa for this matrix (cached per sparsity pattern).

        auto: banded Cholesky when the bandwidth is below BAND_AUTO_BW (both it and multigrid
        scale linearly in n there, see PERF.md), else geometric MG when prolongators are known,
        else the old split (direct below `direct_max_dofs`, pyamg SA above). An auto GMG plan
        turns into "band" once CG needs more iterations than the band would cost (`solve`).
        """
        key = (K.shape, K.nnz, id(getattr(K, "indices", None)), self.kind)
        if self._plan_key == key:
            return self._plan
        n = K.shape[0]
        inf = float("inf")
        # the band estimate is O(nnz); bw >= 3 * (cross-section nodes) rules out big systems
        if self.kind == "direct" or (self.kind == "auto" and n <= BAND_MAX_DOFS):
            bw, work, nbytes = self._band_plan(K)
        else:
            bw, work, nbytes = inf, inf, inf
        if self.kind == "auto" and bw <= BAND_AUTO_BW and nbytes <= BAND_MAX_BYTES:
            plan = "band"
        elif self.kind != "direct" and self.prolongators is not None:
            plan = "gmg"
        elif self.select(n) == "direct":  # also the fallback when pyamg is missing
            if work == inf:
                bw, work, nbytes = self._band_plan(K)
            plan = "band" if work <= BAND_MAX_WORK and nbytes <= BAND_MAX_BYTES else "splu"
        else:
            plan = "sa"
        self._switch_its = None
        self._gmg_its = []
        if self.kind == "auto" and plan == "gmg" and nbytes <= BAND_MAX_BYTES:
            self._switch_its = (BAND_US_PER_BW * bw - BAND_US_OFFSET) / GMG_US_PER_IT
        self._plan_key, self._plan = key, plan
        return plan

    def rtol_for(self, change: float | None) -> float:
        if self.adaptive_tol is None or change is None or self.adaptive_tol <= self.tol:
            return self.tol
        t = float(np.clip((change - CHANGE_TIGHT) / (CHANGE_LOOSE - CHANGE_TIGHT), 0.0, 1.0))
        return self.tol * (self.adaptive_tol / self.tol) ** t

    # -- solve ---------------------------------------------------------------------------------

    def solve(
        self,
        K: sp.spmatrix,
        F: np.ndarray,
        x0: np.ndarray | None = None,
        rigid_modes: np.ndarray | None = None,
        *,
        change: float | None = None,
    ) -> tuple[np.ndarray, SolveInfo]:
        with _BLAS:
            return self._solve(K, F, x0, rigid_modes, change)

    def _solve(self, K, F, x0, rigid_modes, change) -> tuple[np.ndarray, SolveInfo]:
        t0 = time.perf_counter()
        F = np.asarray(F)
        Fm = F.reshape(F.shape[0], -1)
        if not (sp.issparse(K) and K.format == "csr"):
            K = sp.csr_matrix(K)
        method = self.method(K)
        rtol = 0.0
        its = 0
        setup = 0.0
        if method == "band":
            if self._band is None:
                self._band = _BandCholesky(K, self._band_perm)
            U = self._band.solve(K, Fm)
            if U is None:  # not positive definite (e.g. an unsupported island): let SuperLU try
                method = "splu"
        if method == "splu":
            U = self._direct(K, Fm)
        if method in ("band", "splu"):
            setup = time.perf_counter() - t0
            kind = "direct"
        else:
            kind = "amg"
            rtol = self.rtol_for(change)
            X0 = None if x0 is None else np.asarray(x0).reshape(Fm.shape)
            if method == "gmg":
                U, its, setup, method = self._gmg(K, Fm, X0, rtol)
                self._maybe_switch_to_band(its)
            else:
                U, its, setup = self._amg(K, Fm, X0, rigid_modes, rtol)
        self.last_method = method
        res = self._residual(K, Fm, U)
        if kind == "direct" and not res <= DIRECT_MAX_RESIDUAL:
            raise np.linalg.LinAlgError(f"{SINGULAR_MESSAGE} (relative residual {res:.1e})")
        info = SolveInfo(kind, its, res, time.perf_counter() - t0, method, rtol, setup)
        return U.reshape(F.shape), info

    def _maybe_switch_to_band(self, its: int) -> None:
        if self._switch_its is None:
            return
        self._gmg_its.append(its)
        recent = self._gmg_its[-BAND_SWITCH_WINDOW:]
        if len(recent) == BAND_SWITCH_WINDOW and min(recent) > self._switch_its:
            self._plan, self._switch_its, self._mg = "band", None, None  # frees the hierarchy

    def _residual(self, K: sp.spmatrix, Fm: np.ndarray, U: np.ndarray) -> float:
        out = 0.0
        blk = self._mg.levels[0]["A"] if self._mg is not None and self._mg.levels else None
        if blk is None or blk.A is not K:
            blk = _Blocks(K, self.threads)
        y = np.empty(K.shape[0], dtype=np.float64)
        for c in range(Fm.shape[1]):
            f = Fm[:, c]
            fn = float(np.linalg.norm(f))
            u = np.asarray(U[:, c], dtype=np.float64)
            r = f - (blk.matvec(u, y) if K.dtype == np.float64 else blk.matvec64(u, y))
            rel = float(np.linalg.norm(r)) / (fn if fn > 0 else 1.0)
            out = max(out, rel if np.isfinite(rel) else float("inf"))  # max() drops a NaN
        return out

    def _direct(self, K: sp.spmatrix, F: np.ndarray) -> np.ndarray:
        # K is SPD: symmetric mode without pivoting, minimum degree on A^T + A
        try:
            lu = sla.splu(
                sp.csc_matrix(K, dtype=np.float64),
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
        except RuntimeError as exc:  # "Factor is exactly singular"
            raise np.linalg.LinAlgError(f"{SINGULAR_MESSAGE} ({exc})") from exc
        return lu.solve(np.asarray(F, dtype=np.float64))

    def _pcg(
        self,
        matvec: Callable[[np.ndarray], np.ndarray],
        M: Callable[[np.ndarray], np.ndarray],
        b: np.ndarray,
        x0: np.ndarray | None,
        rtol: float,
        maxiter: int | None = None,
    ) -> tuple[np.ndarray, int, str, float]:
        """(x, iterations, status, |r|/|b|); status converged | maxiter | breakdown, the last when
        p.Ap <= 0 or r.z <= 0 (or NaN): K or M is not SPD and CG cannot continue."""
        maxiter = self.maxiter if maxiter is None else maxiter
        bnorm = float(np.linalg.norm(b))
        if bnorm == 0.0:
            return np.zeros_like(b), 0, "converged", 0.0
        x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=np.float64)
        r = b - matvec(x) if x0 is not None else b.copy()
        stop = rtol * bnorm
        rn = float(np.linalg.norm(r))
        if rn <= stop:
            return x, 0, "converged", rn / bnorm
        z = M(r)
        p = z.copy()
        rz = float(r @ z)
        for it in range(1, maxiter + 1):
            q = matvec(p)
            pq = float(p @ q)
            if not (pq > 0 and rz > 0):
                return x, it, "breakdown", rn / bnorm
            alpha = rz / pq
            x += alpha * p
            r -= alpha * q
            rn = float(np.linalg.norm(r))
            if rn <= stop:
                return x, it, "converged", rn / bnorm
            z = M(r)
            rz_new = float(r @ z)
            p *= rz_new / rz
            p += z
            rz = rz_new
        return x, maxiter, "maxiter", rn / bnorm

    def _gmg(
        self, K: sp.spmatrix, F: np.ndarray, X0: np.ndarray | None, rtol: float
    ) -> tuple[np.ndarray, int, float, str]:
        t0 = time.perf_counter()
        if self._mg is None:
            self._mg = GeometricMG(
                _resolve(self.prolongators), degree=self.mg_degree, threads=self.threads
            )
        mg = self._mg
        mg.update(K)
        setup = time.perf_counter() - t0
        blk = mg.levels[0]["A"] if mg.levels else _Blocks(K, self.threads)
        yb = np.empty(K.shape[0], dtype=np.float64)

        def matvec(v):  # float64 accumulation also for float32 storage
            if K.dtype == np.float64:
                return blk.matvec(v, yb).copy()
            return blk.matvec64(v, yb).copy()

        U = np.zeros(F.shape, dtype=np.float64)
        total, level = 0, 0
        for c in range(F.shape[1]):
            b = np.asarray(F[:, c], dtype=np.float64)
            x0 = None if X0 is None else np.asarray(X0[:, c], dtype=np.float64)
            U[:, c], its, lvl = self._gmg_case(K, matvec, mg, b, x0, rtol)
            total += its
            level = max(level, lvl)
        return U, total, setup, GMG_METHODS[level]

    def _gmg_case(self, K, matvec, mg: GeometricMG, b, x0, rtol) -> tuple[np.ndarray, int, int]:
        """One load case. CG that breaks down (or stalls) escalates through GMG_METHODS: restart
        from zero, Gershgorin smoother bounds, Jacobi-PCG. (x, its, index into GMG_METHODS)."""
        x, its, status, res = self._pcg(matvec, mg, b, x0, rtol)
        if status == "converged":
            return x, its, 0
        best = (res, x)
        if status == "breakdown" and x0 is not None:  # same V-cycle, Krylov space from zero
            x, n, status, res = self._pcg(matvec, mg, b, None, rtol)
            its += n
            if status == "converged":
                return x, its, 1
            best = min(best, (res, x), key=lambda t: t[0])
        what = "broke down" if status == "breakdown" else f"did not converge in {self.maxiter} its"
        warnings.warn(
            f"multigrid CG {what} (rel. residual {best[0]:.1e}); retrying with Gershgorin smoother "
            "bounds",
            stacklevel=5,
        )
        mg.make_safe()
        x, n, status, res = self._pcg(matvec, mg, b, None, rtol)
        its += n
        if status == "converged":
            return x, its, 2
        best = min(best, (res, x), key=lambda t: t[0])
        warnings.warn(
            f"multigrid CG with Gershgorin bounds: {status}; falling back to Jacobi-PCG",
            stacklevel=5,
        )
        d = K.diagonal().astype(np.float64)
        dinv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        maxiter = JACOBI_MAXITER_FACTOR * self.maxiter
        x, n, status, res = self._pcg(matvec, lambda r: dinv * r, b, None, rtol, maxiter)
        its += n
        if status != "converged":
            res, x = min(best, (res, x), key=lambda t: t[0])
            warnings.warn(
                f"CG did not reach rtol={rtol} (best rel. residual {res:.1e})", stacklevel=5
            )
        return x, its, 3

    def _amg(
        self,
        K: sp.spmatrix,
        F: np.ndarray,
        X0: np.ndarray | None,
        B: np.ndarray | None,
        rtol: float,
    ) -> tuple[np.ndarray, int, float]:
        t0 = time.perf_counter()
        pyamg = _pyamg()
        rebuild = (
            self.last_ml is None
            or self._ml_age >= self.reuse
            or self._ml_rebuild
            or self.last_ml.levels[0].A.shape != K.shape
        )
        if rebuild:
            if B is not None:
                B = np.asarray(B, dtype=K.dtype)
            self.last_ml = pyamg.smoothed_aggregation_solver(K, B=B, symmetry="symmetric")
            self._ml_age = 0
            self._ml_base_its = None
            self._ml_rebuild = False
        else:
            _sa_refresh(self.last_ml, K)
        self._ml_age += 1
        setup = time.perf_counter() - t0
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
            x, status = sla.cg(K, b, x0=x0, rtol=rtol, maxiter=self.maxiter, M=M, callback=cb)
            if status > 0:
                warnings.warn(
                    f"CG did not reach rtol={rtol} in {self.maxiter} iterations",
                    stacklevel=3,
                )
            U[:, c] = x
            total += count[0]
        per_case = total / max(1, F.shape[1])
        if self._ml_base_its is None:
            self._ml_base_its = max(per_case, 1.0)
        elif per_case > REBUILD_ITER_RATIO * self._ml_base_its:
            self._ml_rebuild = True
        return U, total, setup


def _sa_refresh(ml, K: sp.spmatrix) -> None:
    """Keep the SA prolongators, recompute the Galerkin operators and smoothers for `K`."""
    from pyamg.relaxation.smoothing import change_smoothers

    A = K
    for lvl, level in enumerate(ml.levels):
        if lvl == 0:
            level.A = K
        else:
            prev = ml.levels[lvl - 1]
            A = prev.R @ prev.A @ prev.P
            fmt = level.A.format
            A = A.asformat(fmt) if fmt != "bsr" else sp.bsr_matrix(A, blocksize=level.A.blocksize)
            level.A = A
    for attr in ("P", "LU", "LU_Map", "L"):  # pyamg's coarse solvers cache their factorization
        if hasattr(ml.coarse_solver, attr):
            delattr(ml.coarse_solver, attr)
    gs = ("block_gauss_seidel", {"sweep": "symmetric"})
    change_smoothers(ml, gs, gs)
