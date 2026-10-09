from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from topop.core.benchmarks import cantilever, tip_loaded_beam
from topop.core.fem import (
    Assembler,
    elasticity_matrix,
    estimate_seconds_per_iter,
    hex8_center_stress_matrix,
    hex8_stiffness,
    hex8_strain_matrix,
    interpolation_1d,
    rigid_body_modes,
)
from topop.core.problem import Grid, Load, Material, Problem, Support
from topop.core.solver import _BLAS, GeometricMG, LinearSolver


def box_problem(shape, h=1.0, origin=(0.0, 0.0, 0.0), nu=0.3, active=None) -> Problem:
    grid = Grid(origin=origin, h=h, shape=shape)
    act = np.ones(shape, dtype=bool) if active is None else active
    return Problem(grid, act, np.zeros(shape, dtype=np.int8), Material(nu=nu))


def naive_K(asm: Assembler, E_e: np.ndarray) -> sp.csr_matrix:
    iK = np.repeat(asm.edof, 24, axis=1).ravel()
    jK = np.tile(asm.edof, (1, 24)).ravel()
    sK = (E_e[:, None] * asm.KE_h.ravel()[None, :]).ravel()
    K = sp.coo_matrix((sK, (iK, jK)), shape=(asm.n_dof, asm.n_dof)).tocsr()
    return K[asm.free][:, asm.free]


# ---- element ----------------------------------------------------------------------------------


@pytest.mark.parametrize("nu", [0.0, 0.3, 0.45])
def test_ke_properties(nu):
    KE = hex8_stiffness(nu)
    assert KE.shape == (24, 24)
    assert np.allclose(KE, KE.T, atol=1e-15)
    eig = np.linalg.eigvalsh(KE)
    assert eig.min() >= -1e-12
    assert np.linalg.matrix_rank(KE, tol=1e-10 * eig.max()) == 18
    assert np.abs(KE.sum(axis=1)).max() < 1e-12
    # diagonal entry is exact for trilinear shape functions: (lambda + 4 mu) / 9 on the unit cube
    lam, mu = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))
    assert KE[0, 0] == pytest.approx((lam + 4 * mu) / 9, rel=1e-12)


def test_rigid_modes_are_nullspace_of_ke():
    corners = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=float,
    )
    B = rigid_body_modes(corners * 3.0 + 7.0)
    assert B.shape == (24, 6)
    assert np.linalg.matrix_rank(B) == 6
    assert np.abs(hex8_stiffness(0.3) @ B).max() < 1e-12


# ---- assembler --------------------------------------------------------------------------------


def test_single_element_matches_scaled_ke():
    prob = box_problem((1, 1, 1), h=2.5, origin=(1.0, -1.0, 0.5))
    asm = Assembler(prob)
    assert asm.n_dof == 24 and asm.n_free == 24 and asm.n_elements == 1
    E0, Emin, rho, p = 7.0, 7e-9, 0.6, 3.0
    E_e = np.array([Emin + rho**p * (E0 - Emin)])
    K = asm.assemble(E_e).toarray()
    edof = asm.edof[0]
    assert np.allclose(K[np.ix_(edof, edof)], E_e[0] * 2.5 * hex8_stiffness(0.3), rtol=1e-12)


def test_assembly_matches_naive_coo_on_irregular_domain():
    rng = np.random.default_rng(1)
    shape = (6, 5, 4)
    prob = box_problem(shape, h=0.7, active=rng.random(shape) > 0.35)
    nodes = np.flatnonzero(prob.active_node_mask())
    prob.supports = [Support(nodes[:7], (True, False, True)), Support(nodes[20:23])]
    prob.loads = [Load(nodes[-4:], (0.0, -1.0, 0.0))]
    asm = Assembler(prob)
    assert asm.n_free == asm.n_dof - 7 * 2 - 3 * 3
    for _ in range(2):  # the CSR object is reused; refresh twice
        E_e = rng.random(asm.n_elements) + 1e-3
        K = asm.assemble(E_e)
        K.check_format(full_check=True)
        assert abs(K - naive_K(asm, E_e)).max() < 1e-13
        assert abs(K - K.T).max() < 1e-13


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_symmetric_map_matches_full_assembly(dtype):
    # upper-triangle map + mirror vs the full 576-entry map, with partially fixed nodes and holes
    rng = np.random.default_rng(8)
    shape = (7, 5, 4)
    prob = box_problem(shape, h=0.6, active=rng.random(shape) > 0.3)
    nodes = np.flatnonzero(prob.active_node_mask())
    prob.supports = [Support(nodes[:9], (True, False, True)), Support(nodes[30:34])]
    prob.loads = [Load(nodes[-3:], (0.0, 0.0, 1.0))]
    sym = Assembler(prob, dtype=dtype)
    full = Assembler(prob, dtype=dtype, symmetric_map=False)
    assert sym.symmetric_map and not full.symmetric_map
    assert sym._P.nnz < 0.55 * full._P.nnz  # 300 of 576 entries per element (+ padding)
    assert sym._mirror_dst.size == (sym._K.nnz - sym.n_free) // 2
    tol = 1e-12 if dtype == np.float64 else 1e-6  # float32: summation order of the rounding
    for _ in range(2):  # the CSR object is reused and refreshed in place
        E = 1e-9 + rng.random(sym.n_elements) ** 3
        Ks, Kf = sym.assemble(E), full.assemble(E)
        assert np.array_equal(Ks.indptr, Kf.indptr) and np.array_equal(Ks.indices, Kf.indices)
        assert np.abs(Ks.data - Kf.data).max() <= tol * np.abs(Kf.data).max()
        assert abs(Ks - Ks.T).max() == 0  # mirrored: exactly symmetric


def test_patch_test_constant_strain_is_exact():
    # 2x2x2 elements, one interior node; prescribe u = A x on every boundary node
    prob = box_problem((2, 2, 2), h=0.5, origin=(0.3, -0.2, 1.0), nu=0.3)
    asm = Assembler(prob)  # no supports: K is the full stiffness
    K = asm.assemble(np.ones(asm.n_elements)).toarray()
    A = np.array([[1e-3, 2e-4, -3e-4], [5e-4, -2e-3, 1e-4], [-4e-4, 3e-4, 1.5e-3]])
    X = asm.node_coords_compressed
    u_exact = (X @ A.T).ravel()
    interior = asm.node_map[prob.grid.node_ids(1, 1, 1)]
    i = 3 * interior + np.arange(3)
    b = np.setdiff1d(np.arange(asm.n_dof), i)
    u_i = np.linalg.solve(K[np.ix_(i, i)], -K[np.ix_(i, b)] @ u_exact[b])
    assert np.allclose(u_i, u_exact[i], rtol=0, atol=1e-14)
    # the full field is in equilibrium at the interior node and the energy is eps:D:eps * V
    assert np.abs((K @ u_exact)[i]).max() < 1e-14
    eps = 0.5 * (A + A.T)
    ev = np.array([eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[0, 1], 2 * eps[1, 2], 2 * eps[0, 2]])
    energy = ev @ elasticity_matrix(0.3) @ ev * 0.5**3
    assert np.allclose(asm.element_energies(u_exact), energy, rtol=1e-10)


def test_element_stress_of_linear_fields():
    # u = A x: every element has the strain sym(A) and the stress E D eps at its center,
    # independent of h; a rigid rotation (antisymmetric A) gives zero stress
    shape, h = (3, 2, 2), 0.7
    grid = Grid(origin=(0.3, -0.2, 1.0), h=h, shape=shape)
    prob = Problem(grid, np.ones(shape, bool), np.zeros(shape, np.int8), Material(E=5.0, nu=0.25))
    asm = Assembler(prob)  # no supports: every DOF is free
    X = asm.node_coords_compressed
    A = np.array([[1e-3, 2e-4, -3e-4], [5e-4, -2e-3, 1e-4], [-4e-4, 3e-4, 1.5e-3]])
    eps = 0.5 * (A + A.T)
    ev = np.array([eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[0, 1], 2 * eps[1, 2], 2 * eps[0, 2]])
    sig = asm.element_stress((X @ A.T).ravel())
    assert np.allclose(sig, 5.0 * elasticity_matrix(0.25) @ ev, rtol=1e-12, atol=1e-15)
    W = A - A.T
    assert np.abs(asm.element_stress((X @ W.T).ravel())).max() < 1e-14
    assert np.allclose(hex8_center_stress_matrix(0.25, h), sig_matrix_ref(0.25, h))


def sig_matrix_ref(nu: float, h: float) -> np.ndarray:
    # D B at the center from the mean of B over the 8 Gauss points (B is linear in xi)
    g = 1 / np.sqrt(3)
    B = sum(hex8_strain_matrix(np.array(xi)) for xi in itertools.product((-g, g), repeat=3)) / 8
    return elasticity_matrix(nu) @ B / h


def test_multi_case_loads_and_energies():
    prob = box_problem((4, 2, 2))
    g = prob.grid
    prob.supports = [
        Support(g.node_ids(*np.meshgrid(0, range(3), range(3), indexing="ij")).ravel())
    ]
    tip = g.node_ids(*np.meshgrid(4, range(3), range(3), indexing="ij")).ravel()
    prob.loads = [
        Load(tip, (0.0, -2.0, 0.0), case=0),
        Load(tip[:3], (0.0, 0.0, 1.0), case=2),
        Load(tip[3:], (0.5, 0.0, 0.0), case=2),
    ]
    asm = Assembler(prob)
    assert asm.n_cases == 3
    assert asm.F.shape == (asm.n_dof, 3)
    assert asm.F_free.shape == (asm.n_free, 3)
    assert np.allclose(asm.F.reshape(-1, 3, 3).sum(axis=0).T, [[0, -2, 0], [0, 0, 0], [0.5, 0, 1]])
    assert np.allclose(asm.F[3 * asm.node_map[tip[0]] + 1, 0], -2.0 / 9)
    E_e = np.full(asm.n_elements, 0.8)
    K = asm.assemble(E_e)
    U, info = LinearSolver("direct").solve(K, asm.F_free)
    assert U.shape == (asm.n_free, 3) and info.residual < 1e-10
    assert np.allclose(U[:, 1], 0.0)
    # sum_e E_e u_e^T k_e u_e == sum_cases F . U
    assert np.sum(E_e * asm.element_energies(U)) == pytest.approx(np.sum(asm.F_free * U))


@pytest.mark.parametrize("kind", ["direct", "amg"])
def test_solvers_agree(kind):
    prob = tip_loaded_beam(12, 3, 3)
    asm = Assembler(prob)
    rng = np.random.default_rng(0)
    K = asm.assemble(1e-3 + rng.random(asm.n_elements) ** 3)
    B = rigid_body_modes(asm.node_coords_compressed)[asm.free]
    U, info = LinearSolver(kind, tol=1e-10).solve(K, asm.F_free, rigid_modes=B)
    assert info.kind == kind
    ref = sla.spsolve(sp.csc_matrix(K), asm.F_free[:, 0])
    assert np.allclose(U[:, 0], ref, rtol=1e-6, atol=1e-8 * np.abs(ref).max())


def test_amg_hierarchy_reuse_and_missing_pyamg_fallback(monkeypatch):
    asm = Assembler(tip_loaded_beam(12, 3, 3))
    K = asm.assemble(np.ones(asm.n_elements))
    solver = LinearSolver("amg", reuse=2)
    U, _ = solver.solve(K, asm.F_free)
    ml = solver.last_ml
    _, info = solver.solve(K, asm.F_free, x0=U)
    assert solver.last_ml is ml and info.iterations <= 1
    solver.solve(K, asm.F_free)
    assert solver.last_ml is not ml

    import topop.core.solver as solver_mod

    monkeypatch.setattr(solver_mod, "_pyamg", lambda: None)
    with pytest.warns(UserWarning, match="pyamg not available"):
        _, info = LinearSolver("amg").solve(K, asm.F_free)
    assert info.kind == "direct" and info.residual < 1e-10


def test_tip_loaded_beam_matches_timoshenko():
    E, nu, P = 1.0, 0.3, 1.0
    L, b = 40.0, 4.0
    prob = tip_loaded_beam(40, 4, 4, E=E, nu=nu)
    asm = Assembler(prob)
    K = asm.assemble(np.full(asm.n_elements, E))
    U, _ = LinearSolver("direct").solve(K, asm.F_free)
    v = asm.expand(U[:, 0]).reshape(-1, 3)[:, 1]
    tip = asm.node_map[prob.loads[0].nodes]
    delta = -v[tip].mean()
    inertia = b**4 / 12
    G, kappa = E / (2 * (1 + nu)), 5 / 6
    timoshenko = P * L**3 / (3 * E * inertia) + P * L / (kappa * G * b * b)  # 1007.8
    # full-integration hex8 with 4 elements over the depth is slightly too stiff (shear locking)
    # and the clamped face adds Poisson restraint: measured delta = 964.9, ratio 0.957
    # (80x8x8 of the same beam scaled 2x gives 0.983, i.e. it converges to Timoshenko)
    assert delta == pytest.approx(964.89, rel=1e-3)
    assert abs(delta / timoshenko - 1) < 0.10


def test_estimate_bytes_monotone_and_sane():
    sizes = [1, 10, 1_000, 10_000, 100_000, 1_000_000]
    est = [Assembler.estimate_bytes(n, np.float64) for n in sizes]
    assert all(a < b for a, b in itertools.pairwise(est))
    assert all(
        Assembler.estimate_bytes(n, np.float32) < Assembler.estimate_bytes(n, np.float64)
        for n in sizes[2:]
    )
    assert 0.5e9 < Assembler.estimate_bytes(100_000, np.float64) < 6e9
    assert Assembler.estimate_bytes(1_000_000, np.float64) > 6e9  # refused at the default cap


def test_estimate_seconds_per_iter_monotone():
    sizes = [1, 1_000, 5_000, 20_000, 100_000, 250_000, 1_000_000]
    est = [estimate_seconds_per_iter(n) for n in sizes]
    assert all(e > 0 for e in est)
    assert all(a <= b for a, b in itertools.pairwise(est))
    assert estimate_seconds_per_iter(100_000) < 10


# ---- geometric multigrid, banded Cholesky, solver policies ------------------------------------


def test_interpolation_1d():
    for n in (2, 3, 4, 7):
        P, nc = interpolation_1d(n)
        assert P.shape == (n + 1, nc + 1) and nc == (n + 1) // 2
        assert np.allclose(P.sum(axis=1), 1.0)
        xc = 2.0 * np.arange(nc + 1)  # coarse nodes sit at even fine positions
        assert np.allclose(P @ xc, np.arange(n + 1))  # linear fields are reproduced
    P, nc = interpolation_1d(1)  # a single cell is not coarsened
    assert nc == 1 and np.allclose(P.toarray(), np.eye(2))


def test_prolongators_partition_of_unity_and_linear_fields():
    rng = np.random.default_rng(4)
    shape = (9, 6, 5)
    active = rng.random(shape) > 0.2
    prob = box_problem(shape, h=0.5, active=active)
    nodes = np.flatnonzero(prob.active_node_mask())
    prob.supports = [Support(nodes[:15], (True, False, True))]
    asm = Assembler(prob)
    Ps = asm.prolongators(min_dofs=20)
    assert len(Ps) >= 2 and asm.prolongators(min_dofs=20) is Ps  # cached
    assert Ps[0].shape[0] == asm.n_free
    for fine, coarse in itertools.pairwise(Ps):
        assert fine.shape[1] == coarse.shape[0]
    for P in Ps:
        assert np.allclose(P.sum(axis=1), 1.0)  # interpolation weights of every unknown
        assert np.diff(P.tocsc().indptr).min() > 0  # every coarse unknown is used
    # a linear displacement field on the coarse grid is interpolated exactly to the free DOFs
    P0 = Ps[0]
    u_fine = np.zeros(asm.n_dof)
    X = asm.node_coords_compressed
    u_fine[0::3], u_fine[1::3], u_fine[2::3] = X[:, 0] + 2 * X[:, 1], X[:, 2], 3 * X[:, 0]
    # coarse DOF values: evaluate the same field at the coarse nodes (spacing 2h)
    grid = prob.grid
    cn_shape = tuple((n + 1) // 2 + 1 for n in shape)
    cols = np.unique(P0.indices)
    used_full = _coarse_full_dofs(asm, shape)
    cnode, cax = np.divmod(used_full, 3)
    ci = np.stack(np.unravel_index(cnode, cn_shape), axis=1)
    Xc = np.asarray(grid.origin) + 2 * grid.h * ci
    vals = np.choose(cax, [Xc[:, 0] + 2 * Xc[:, 1], Xc[:, 2], 3 * Xc[:, 0]])
    assert cols.size == vals.size
    assert np.allclose(P0 @ vals, u_fine[asm.free])


def _coarse_full_dofs(asm, shape):
    """Recompute the coarse unknowns' full-grid DOF ids exactly as `prolongators` does."""
    import scipy.sparse as sps

    ops = [interpolation_1d(n)[0] for n in shape]
    Pn = sps.kron(sps.kron(ops[0], ops[1]), ops[2], format="csr")
    sub = Pn[asm.node_ids[asm.free_dofs // 3]]
    cdof = 3 * sub.indices + np.repeat(asm.free_dofs % 3, np.diff(sub.indptr))
    return np.unique(cdof)


def test_band_ordering_puts_the_longest_axis_slowest():
    prob = tip_loaded_beam(30, 4, 3)
    g = prob.grid
    # same beam along y: natural (x-slowest) order has a much wider band
    grid_y = Grid(origin=(0.0, 0.0, 0.0), h=1.0, shape=(4, 30, 3))
    prob_y = box_problem((4, 30, 3))
    prob_y.supports = [
        Support(grid_y.node_ids(*np.meshgrid(range(5), 0, range(4), indexing="ij")).ravel())
    ]
    prob_y.loads = [
        Load(
            grid_y.node_ids(*np.meshgrid(range(5), 30, range(4), indexing="ij")).ravel(),
            (1.0, 0, 0),
        )
    ]
    from topop.core.solver import _BandCholesky

    for p in (prob, prob_y):
        asm = Assembler(p)
        perm = asm.band_ordering()
        assert np.array_equal(np.sort(perm), np.arange(asm.n_free))
        K = asm.assemble(np.ones(asm.n_elements))
        bw_nat = _BandCholesky.estimate(K, None)[0]
        bw = _BandCholesky.estimate(K, perm)[0]
        assert bw <= bw_nat
        assert bw <= 3 * (5 * 4 + 4 + 1) + 2  # one node slab of the 4x3 cross-section
    assert g.shape == (30, 4, 3)


def ring_problem(radius: int, width: int, thickness: int) -> Problem:
    n = 2 * radius + 2
    shape = (n, n, thickness)
    c = np.arange(n) - n / 2 + 0.5
    r = np.hypot(*np.meshgrid(c, c, indexing="ij"))
    active = ((r < radius) & (r > radius - width))[:, :, None] & np.ones(shape, dtype=bool)
    prob = box_problem(shape, active=active)
    ids = np.flatnonzero(prob.active_node_mask())
    ix = np.unravel_index(ids, prob.grid.node_shape)[0]
    prob.supports = [Support(ids[ix <= 2])]
    prob.loads = [Load(ids[ix >= n - 2], (1.0, 0.0, 0.0))]
    return prob


def test_band_ordering_takes_rcm_on_rings_and_the_axis_sweep_on_beams():
    from topop.core.solver import _BandCholesky

    for prob, rcm_wins in ((ring_problem(16, 3, 3), True), (tip_loaded_beam(30, 4, 3), False)):
        asm = Assembler(prob)
        K = asm.assemble(np.ones(asm.n_elements))
        ijk = np.stack(np.unravel_index(asm.node_ids, prob.grid.node_shape), axis=1)
        extent = ijk.max(axis=0) - ijk.min(axis=0)
        axes = np.argsort(-extent, kind="stable")
        node = asm.free_dofs // 3
        sweep = np.lexsort([asm.free_dofs % 3] + [ijk[node, a] for a in axes[::-1]])
        bw_sweep = _BandCholesky.estimate(K, sweep)[0]
        perm = asm.band_ordering()
        assert np.array_equal(np.sort(perm), np.arange(asm.n_free))
        bw = _BandCholesky.estimate(K, perm)[0]
        if rcm_wins:  # around a ring every axis sweep cuts it twice; RCM follows the loop
            assert bw < 0.7 * bw_sweep
        else:
            assert bw == bw_sweep


@pytest.mark.parametrize("method", ["band", "band-perm", "splu", "gmg", "sa"])
def test_all_methods_agree(method):
    prob = tip_loaded_beam(16, 4, 4)
    asm = Assembler(prob)
    rng = np.random.default_rng(0)
    E = 1e-6 + rng.random(asm.n_elements) ** 3
    K = asm.assemble(E)
    ref = sla.spsolve(sp.csc_matrix(K), asm.F_free[:, 0])
    kw = {}
    if method == "band-perm":
        kw = {"ordering": asm.band_ordering}
    if method == "gmg":
        kw = {"prolongators": lambda: asm.prolongators(min_dofs=40)}
    kind = {"band": "direct", "band-perm": "direct", "splu": "direct", "gmg": "amg", "sa": "amg"}
    solver = LinearSolver(kind[method], tol=1e-10, **kw)
    if method == "splu":
        import topop.core.solver as solver_mod

        solver.method = lambda K: "splu"  # force SuperLU
        assert solver_mod.BAND_MAX_WORK > 0
    B = rigid_body_modes(asm.node_coords_compressed)[asm.free] if method == "sa" else None
    U, info = solver.solve(K, asm.F_free, rigid_modes=B)
    assert info.method == method.split("-")[0]
    assert info.kind == kind[method]
    assert np.allclose(U[:, 0], ref, rtol=1e-6, atol=1e-8 * np.abs(ref).max())
    if method == "gmg":
        assert len(solver._mg.levels) >= 2 and 0 < info.iterations < 60
        assert info.residual < 1e-9


def test_auto_picks_band_for_thin_and_gmg_for_large_systems():
    small = Assembler(cantilever(20, 8, 2))
    s = LinearSolver("auto", prolongators=small.prolongators, ordering=small.band_ordering)
    assert s.method(small.assemble(np.ones(small.n_elements))) == "band"
    cube = Assembler(cantilever(14, 14, 14))  # 9450 free DOFs but bandwidth 725
    s = LinearSolver("auto", prolongators=cube.prolongators, ordering=cube.band_ordering)
    assert s.method(cube.assemble(np.ones(cube.n_elements))) == "gmg"
    s = LinearSolver("auto", ordering=cube.band_ordering)  # no geometry: old split, exact solve
    assert s.method(cube.assemble(np.ones(cube.n_elements))) == "band"
    big = Assembler(cantilever(36, 14, 14))  # 21k free DOFs, wide band
    s = LinearSolver("auto", prolongators=big.prolongators, ordering=big.band_ordering)
    K = big.assemble(np.ones(big.n_elements))
    assert s.method(K) == "gmg"
    assert not s.needs_rigid_modes(big.n_free)
    assert LinearSolver("amg").needs_rigid_modes(big.n_free)
    _, info = s.solve(K, big.F_free)
    assert info.method == "gmg" and info.residual < 1e-6 and info.rtol == 1e-6


def test_gmg_multiple_cases_warm_start_and_float32():
    prob = tip_loaded_beam(16, 4, 4)
    g = prob.grid
    side = g.node_ids(*np.meshgrid(16, range(5), 4, indexing="ij")).ravel()
    prob.loads.append(Load(side, (0.0, 0.0, 1.0), case=1))
    prob.loads.append(Load(side, (1.0, 0.0, 0.0), case=2))
    asm = Assembler(prob)
    rng = np.random.default_rng(1)
    E = 1e-9 + rng.random(asm.n_elements) ** 3
    K = asm.assemble(E)
    ref = np.column_stack([sla.spsolve(sp.csc_matrix(K), asm.F_free[:, c]) for c in range(3)])
    s = LinearSolver("amg", prolongators=lambda: asm.prolongators(min_dofs=40))
    U, info = s.solve(K, asm.F_free)
    assert U.shape == (asm.n_free, 3) and info.residual < 1e-6
    assert np.allclose(U, ref, rtol=1e-4, atol=1e-5 * np.abs(ref).max())
    _, again = s.solve(K, asm.F_free, x0=U)  # warm start from the solution
    assert again.iterations <= 3
    # float32 storage: float32 hierarchy, float64 CG still reaches the tolerance
    asm32 = Assembler(prob, dtype=np.float32)
    K32 = asm32.assemble(E)
    s32 = LinearSolver("amg", prolongators=lambda: asm32.prolongators(min_dofs=40))
    U32, info32 = s32.solve(K32, asm32.F_free)
    assert U32.dtype == np.float64 and info32.residual < 1e-6
    assert np.allclose(U32, ref, rtol=1e-3, atol=1e-4 * np.abs(ref).max())


def test_geometric_mg_is_a_symmetric_preconditioner():
    asm = Assembler(tip_loaded_beam(16, 4, 4))
    rng = np.random.default_rng(2)
    K = asm.assemble(1e-9 + rng.random(asm.n_elements) ** 3)
    mg = GeometricMG(asm.prolongators(min_dofs=40), threads=2)
    mg.update(K)
    x, y = rng.standard_normal(asm.n_free), rng.standard_normal(asm.n_free)
    assert x @ mg(y) == pytest.approx(y @ mg(x), rel=1e-8)
    assert x @ mg(x) > 0


def test_adaptive_tolerance_policy():
    s = LinearSolver("amg", tol=1e-6, adaptive_tol=1e-3)
    assert s.rtol_for(None) == 1e-6
    assert s.rtol_for(0.2) == pytest.approx(1e-3)
    assert s.rtol_for(0.1) == pytest.approx(1e-3)
    assert s.rtol_for(0.02) == pytest.approx(1e-6)
    assert s.rtol_for(0.005) == pytest.approx(1e-6)
    assert 1e-6 < s.rtol_for(0.05) < 1e-3
    assert LinearSolver("amg", tol=1e-6).rtol_for(0.2) == 1e-6


def test_sa_partial_reuse_refreshes_operators():
    asm = Assembler(tip_loaded_beam(12, 3, 3))
    rng = np.random.default_rng(3)
    s = LinearSolver("amg", reuse=5, tol=1e-8)
    B = rigid_body_modes(asm.node_coords_compressed)[asm.free]
    s.solve(asm.assemble(np.ones(asm.n_elements)).copy(), asm.F_free, rigid_modes=B)
    ml = s.last_ml
    E = 1e-6 + rng.random(asm.n_elements) ** 3  # very different stiffness, same pattern
    K = asm.assemble(E)
    U, info = s.solve(K, asm.F_free, rigid_modes=B)
    assert s.last_ml is ml  # aggregates kept ...
    assert s.last_ml.levels[0].A is K  # ... operators refreshed
    ref = sla.spsolve(sp.csc_matrix(K), asm.F_free[:, 0])
    assert np.allclose(U[:, 0], ref, rtol=1e-5, atol=1e-7 * np.abs(ref).max())
    assert info.iterations < 100


def test_blas_thread_limiter_restores_state():
    fns = None
    with _BLAS:
        fns = _BLAS._fns
        assert all(get() == _BLAS.n for get, _ in fns)
        with _BLAS:  # re-entrant
            pass
        assert all(get() == _BLAS.n for get, _ in fns)
    assert [get() for get, _ in fns] == _BLAS._saved


def test_gmg_on_irregular_domain_with_roller_supports():
    rng = np.random.default_rng(5)
    shape = (14, 9, 7)
    active = np.ones(shape, dtype=bool)
    active[4:10, 3:6, :] = False  # a hole through the part
    active[rng.random(shape) < 0.05] = False  # and scattered missing voxels
    prob = box_problem(shape, h=0.5, active=active)
    g = prob.grid
    nm = prob.active_node_mask()
    x0 = g.node_ids(*np.meshgrid(0, range(10), range(8), indexing="ij")).ravel()
    y0 = g.node_ids(*np.meshgrid(range(15), 0, range(8), indexing="ij")).ravel()
    tip = g.node_ids(*np.meshgrid(14, range(10), range(8), indexing="ij")).ravel()
    prob.supports = [
        Support(x0[nm[x0]], (True, False, False)),
        Support(y0[nm[y0]], (False, True, True)),
    ]
    prob.loads = [Load(tip[nm[tip]], (0.3, -1.0, 0.2))]
    asm = Assembler(prob)
    K = asm.assemble(1e-9 + rng.random(asm.n_elements) ** 3)
    ref = sla.spsolve(sp.csc_matrix(K), asm.F_free[:, 0])
    s = LinearSolver("amg", tol=1e-9, prolongators=lambda: asm.prolongators(min_dofs=100))
    U, info = s.solve(K, asm.F_free)
    assert info.method == "gmg" and len(s._mg.levels) >= 2
    assert np.allclose(U[:, 0], ref, rtol=1e-5, atol=1e-6 * np.abs(ref).max())
