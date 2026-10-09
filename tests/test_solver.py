"""Geometric-multigrid CG: the V-cycle must stay SPD on SIMP stiffness (topop/core/solver.py).

Regression: on the L-bracket stress run (`l_bracket(40, 4)`, MMA, stress_limit 0.2425, p-norm 16,
move 0.05, rmin 1.5) the Chebyshev bound came from a power iteration warm-started from the last
solve's dominant vector. That vector was localized on a solid/void interface which then moved, and
4 iterations from it gave lambda_max(D^-1 A) = 1.70 on level 1 where the true value was 3.45. The
smoother amplified the missed modes, M K had eigenvalues down to -3.9 and CG broke down after 2-3
iterations at relative residual 0.17-0.33 (solves 160-164 of the run; cold restarts as well).
`data/lbracket_gmg_breakdown.npz` holds the designs (E_e) of that sequence and the failing
right-hand side / warm-start pairs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from topop.core.benchmarks import cantilever, l_bracket
from topop.core.fem import Assembler
from topop.core.optimize import ADAPTIVE_TOL
from topop.core.solver import (
    BAND_AUTO_BW,
    BAND_MAX_BYTES,
    GMG_METHODS,
    GeometricMG,
    LinearSolver,
    _jacobi_gershgorin,
)

DATA = Path(__file__).parent / "data" / "lbracket_gmg_breakdown.npz"


def simp_design(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "simp":
        return 1e-9 + (1 - 1e-9) * rng.random(n) ** 3
    return np.where(rng.random(n) < 0.4, 1.0, 1e-9)  # solid / void, extreme contrast


def true_lmax(A: sp.spmatrix) -> float:
    """lambda_max(D^-1 A) through the similar D^-1/2 A D^-1/2."""
    s = 1.0 / np.sqrt(A.diagonal())
    B = sp.diags(s) @ sp.csr_matrix(A, dtype=np.float64) @ sp.diags(s)
    return float(sla.eigsh(B, k=1, which="LA", return_eigenvectors=False, tol=1e-8)[0])


def level_operators(K: sp.csr_matrix, mg: GeometricMG) -> list[sp.csr_matrix]:
    out = [sp.csr_matrix(K, dtype=np.float64)]
    for P in mg.P[: len(mg.levels) - 1]:
        out.append((P.T @ out[-1] @ P).tocsr())
    return out


# ---- (a) the V-cycle is a symmetric positive definite preconditioner ----------------------------


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("kind", ["simp", "solid-void"])
def test_vcycle_is_spd_on_cantilever(kind, dtype):
    asm = Assembler(cantilever(20, 8, 4), dtype=dtype)
    rng = np.random.default_rng(11)
    K = asm.assemble(simp_design(asm.n_elements, kind, rng))
    mg = GeometricMG(asm.prolongators(min_dofs=100), threads=2)
    mg.update(K)
    assert len(mg.levels) >= 3
    n = asm.n_free
    sym_tol = 1e-10 if dtype == np.float64 else 1e-5

    r1, r2 = rng.standard_normal(n), rng.standard_normal(n)
    z1, z2 = mg(r1), mg(r2)
    assert z1 @ r2 == pytest.approx(z2 @ r1, rel=sym_tol)
    assert r1 @ z1 > 0 and r2 @ z2 > 0

    # the whole spectrum: M K ~ C M C^T (K = C^T C) has its eigenvalues in (0, 1] for a V-cycle
    # with A-norm non-expansive smoothers and an exact coarsest solve
    M = np.column_stack([mg(e) for e in np.eye(n)])
    assert np.abs(M - M.T).max() <= sym_tol * np.abs(M).max()
    C = sl.cholesky(K.toarray().astype(np.float64))
    lam = sl.eigvalsh(C @ (0.5 * (M + M.T)) @ C.T)
    assert lam.min() > 1e-3, f"M K has eigenvalue {lam.min():.3g}"
    assert lam.max() <= 1.0 + 1e-3


@pytest.mark.parametrize("kind", ["simp", "solid-void"])
def test_chebyshev_bounds_cover_lambda_max(kind):
    asm = Assembler(cantilever(20, 8, 4))
    K = asm.assemble(simp_design(asm.n_elements, kind, np.random.default_rng(12)))
    mg = GeometricMG(asm.prolongators(min_dofs=100), threads=2)
    mg.update(K)
    for L, A in zip(mg.levels, level_operators(K, mg), strict=True):
        lam = true_lmax(A)
        assert lam <= L["lmax"] <= 1.3 * lam  # covered, and tight enough to smooth well
    mg.make_safe()
    for L, A in zip(mg.levels, level_operators(K, mg), strict=True):
        assert L["lmax"] == pytest.approx(_jacobi_gershgorin(A.tocsr()))
        assert L["lmax"] >= true_lmax(A)


# ---- (b) the recorded L-bracket failure -----------------------------------------------------------


@pytest.fixture(scope="module")
def lbracket():
    d = dict(np.load(DATA))
    return Assembler(l_bracket(40, 4)), d


def optimizer_like_solver(asm: Assembler) -> LinearSolver:
    return LinearSolver(
        "auto", prolongators=asm.prolongators, ordering=asm.band_ordering, adaptive_tol=ADAPTIVE_TOL
    )


def test_lbracket_breakdown_sequence_is_solved(lbracket):
    asm, d = lbracket
    s = optimizer_like_solver(asm)
    # the preconditioner history: two refreshes per design (displacement, stress adjoint); the
    # old warm-started estimate depended only on this K sequence, not on the right-hand sides
    for E in d["E_history"]:
        K = asm.assemble(E)
        for _ in range(2):
            s.solve(K, asm.F_free)
    K = asm.assemble(d["E"])
    assert s.method(K) == "gmg"
    cases = [(asm.F_free[:, 0], d["x0_u"]), (d["rhs_adjoint"], d["x0_adjoint"])]
    for F, x0 in cases:  # old code: breakdown after 3 iterations at residual 0.167 on the first
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            U, info = s.solve(K, F, x0=x0, change=float(d["change"]))
        assert info.method == "gmg"  # no fallback needed
        assert info.residual <= info.rtol
        assert info.iterations <= 30
        ref = sla.spsolve(sp.csc_matrix(K), F)
        assert np.linalg.norm(U - ref) <= 1e-3 * np.linalg.norm(ref)


@pytest.mark.parametrize("which", ["E", "E_outlier"])
def test_lbracket_bounds_cover_lambda_max(lbracket, which):
    # "E": level 1 was estimated at 1.70 (true 3.45); "E_outlier" (solve 62 of the run): a
    # localized level-0 mode at 4.13 above the bulk 3.43, missed by the old estimate (3.37)
    asm, d = lbracket
    K = asm.assemble(d[which])
    mg = GeometricMG(asm.prolongators(), threads=2)
    mg.update(K)
    for L, A in zip(mg.levels, level_operators(K, mg), strict=True):
        lam = true_lmax(A)
        assert lam <= L["lmax"] <= 1.3 * lam


# ---- (c) warm-started solves on changing designs --------------------------------------------------


def test_warm_started_solves_on_changing_designs_reach_rtol():
    asm = Assembler(cantilever(24, 12, 6))
    s = LinearSolver("amg", prolongators=lambda: asm.prolongators(min_dofs=200))
    rng = np.random.default_rng(7)
    x = rng.random(asm.n_elements)
    U = lam = None
    for _ in range(10):
        # E stays SPD-positive in [1e-9, 1]; 5 % of the elements flip solid <-> void, which
        # moves the localized top modes of D^-1 K that trapped the old warm-started estimate
        x = np.clip(x * np.exp(0.3 * rng.standard_normal(x.size)), 0.0, 1.0)
        flip = rng.random(x.size) < 0.05
        x[flip] = 1.0 - np.round(x[flip])
        K = asm.assemble(1e-9 + (1 - 1e-9) * x**3)
        g = rng.standard_normal(asm.n_free)  # a stress-adjoint-like second right-hand side
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            U, info = s.solve(K, asm.F_free, x0=U)
            assert info.method == "gmg" and info.residual <= info.rtol
            lam, info = s.solve(K, g, x0=lam)
            assert info.method == "gmg" and info.residual <= info.rtol


# ---- fallbacks inside solve ---------------------------------------------------------------------


@pytest.fixture
def small():
    asm = Assembler(cantilever(20, 8, 4))
    K = asm.assemble(simp_design(asm.n_elements, "simp", np.random.default_rng(3)) + 1e-3)
    s = LinearSolver("amg", prolongators=lambda: asm.prolongators(min_dofs=100))
    ref = sla.spsolve(sp.csc_matrix(K), asm.F_free[:, 0])
    return asm, K, s, ref


def test_breakdown_restarts_from_zero_with_the_same_preconditioner(small, monkeypatch):
    asm, K, s, ref = small
    U0, _ = s.solve(K, asm.F_free)
    real, calls = s._pcg, []

    def first_breaks(*args, **kw):
        x, its, status, res = real(*args, **kw)
        calls.append(args[3] is not None)  # warm-started?
        return x, its, ("breakdown" if len(calls) == 1 else status), res

    monkeypatch.setattr(s, "_pcg", first_breaks)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        U, info = s.solve(K, asm.F_free, x0=U0 * 0.9)
    assert calls == [True, False]
    assert info.method == "gmg-restart" and info.residual <= info.rtol
    assert np.allclose(U[:, 0], ref, rtol=1e-4, atol=1e-5 * np.abs(ref).max())


def test_indefinite_vcycle_falls_back_to_safe_bounds(small, monkeypatch):
    asm, K, s, ref = small
    # a gross underestimate of lambda_max: the smoother amplifies, the V-cycle is indefinite
    monkeypatch.setattr(GeometricMG, "_lmax", lambda self, *a: 0.3)
    with pytest.warns(UserWarning, match="Gershgorin"):
        U, info = s.solve(K, asm.F_free, x0=np.ones(asm.n_free))
    assert info.method == "gmg-safe" and info.residual <= info.rtol
    assert np.allclose(U[:, 0], ref, rtol=1e-4, atol=1e-5 * np.abs(ref).max())


def test_jacobi_pcg_is_the_last_resort(small, monkeypatch):
    asm, K, s, ref = small
    monkeypatch.setattr(GeometricMG, "_lmax", lambda self, *a: 0.3)
    monkeypatch.setattr(GeometricMG, "make_safe", lambda self: None)  # stays indefinite
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        U, info = s.solve(K, asm.F_free)
    messages = [str(w.message) for w in caught]
    assert len(messages) == 2 and "Gershgorin" in messages[0] and "Jacobi-PCG" in messages[1]
    assert info.method == "jacobi-pcg" and info.residual <= info.rtol
    assert s.last_method == info.method == GMG_METHODS[-1]
    assert np.allclose(U[:, 0], ref, rtol=1e-4, atol=1e-5 * np.abs(ref).max())


def test_stalled_cg_escalates_and_returns_the_best_attempt(small):
    asm, K, _, _ = small
    s = LinearSolver("amg", maxiter=3, prolongators=lambda: asm.prolongators(min_dofs=100))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, info = s.solve(K, asm.F_free)
    messages = [str(w.message) for w in caught]
    assert "did not converge in 3 its" in messages[0] and "did not reach rtol" in messages[-1]
    assert info.method == "jacobi-pcg" and info.iterations == 3 + 3 + 15
    assert info.rtol < info.residual < 1.0  # reported honestly, from the best (multigrid) attempt


# ---- method choice (auto) -----------------------------------------------------------------------


def auto_solver(asm: Assembler) -> LinearSolver:
    return LinearSolver("auto", prolongators=asm.prolongators, ordering=asm.band_ordering)


def test_auto_takes_the_band_for_slender_and_multigrid_for_compact_parts():
    slender = Assembler(cantilever(60, 20, 4))
    s = auto_solver(slender)
    assert s.method(slender.assemble(np.ones(slender.n_elements))) == "band"
    assert s._band_est[0] == 335 <= BAND_AUTO_BW  # one 21x5-node slab per x station
    compact = Assembler(cantilever(30, 30, 30))
    s = auto_solver(compact)
    assert s.method(compact.assemble(np.ones(compact.n_elements))) == "gmg"
    # a cube's bandwidth is bounded below by its cross-section (n / diameter): 3 * 31^2 + ...,
    # in any ordering (RCM gives 8189), so the band never pays off and is never switched to
    bw, _, nbytes = s._band_est
    assert bw == 3 * (31 * 31 + 31 + 1) + 2
    assert nbytes > BAND_MAX_BYTES and s._switch_its is None


def test_auto_switches_to_the_band_when_multigrid_cg_gets_expensive(monkeypatch):
    import topop.core.solver as solver_mod

    asm = Assembler(cantilever(40, 16, 8))  # bandwidth 491: multigrid first
    K = asm.assemble(simp_design(asm.n_elements, "simp", np.random.default_rng(5)) + 1e-3)
    s = auto_solver(asm)
    assert s.method(K) == "gmg"
    assert 12 < s._switch_its < 20  # the band costs as much as ~15 CG iterations here
    monkeypatch.setattr(solver_mod, "GMG_US_PER_IT", 1e3)  # every CG iteration "costs" more
    s = auto_solver(asm)
    U1, i1 = s.solve(K, asm.F_free)
    assert i1.method == "gmg" and s.method(K) == "gmg"  # one expensive solve is not enough
    _, i2 = s.solve(K, asm.F_free)
    assert i2.method == "gmg" and s.method(K) == "band" and s._mg is None
    U3, i3 = s.solve(K, asm.F_free)
    assert i3.method == "band" and i3.kind == "direct" and i3.residual < 1e-10
    assert np.allclose(U3, U1, rtol=1e-4, atol=1e-5 * np.abs(U3).max())
    s = LinearSolver("amg", prolongators=asm.prolongators, ordering=asm.band_ordering)
    for _ in range(3):  # an explicit "amg" stays on multigrid
        _, info = s.solve(K, asm.F_free)
    assert info.method == "gmg"


@pytest.mark.parametrize("path", ["band", "splu"])
def test_singular_stiffness_is_an_error_not_garbage(path, monkeypatch):
    # roller-only support (z on the bottom face) bypassing validate(): K has 3 free rigid-body
    # modes and the x load drives one of them. The factorizations used to "succeed".
    import topop.core.solver as solver_mod

    p = cantilever(8, 4, 4)
    g = p.grid
    ii, jj = np.meshgrid(np.arange(9), np.arange(5), indexing="ij")
    bottom = g.node_ids(ii, jj, np.zeros_like(ii)).ravel()
    p.supports = [type(p.supports[0])(nodes=bottom, fix=(False, False, True))]
    p.loads = [type(p.loads[0])(nodes=p.loads[0].nodes, force=(1.0, 0.0, -1.0))]
    if path == "splu":
        monkeypatch.setattr(solver_mod, "BAND_MAX_WORK", 0.0)
    A = Assembler(p)
    K = A.assemble(np.ones(A.n_elements))
    s = LinearSolver("direct", ordering=A.band_ordering)
    with pytest.raises(np.linalg.LinAlgError, match="singular or badly conditioned"):
        s.solve(K, A.F_free)
    # a consistent load (orthogonal to the free modes) still solves: u is defined up to them
    p.loads = [type(p.loads[0])(nodes=p.loads[0].nodes, force=(0.0, 0.0, -1.0))]
    A = Assembler(p)
    U, info = LinearSolver("direct").solve(A.assemble(np.ones(A.n_elements)), A.F_free)
    assert info.residual < 1e-9 and np.isfinite(U).all()
