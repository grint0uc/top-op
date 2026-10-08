from __future__ import annotations

import itertools

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from topop.core.benchmarks import tip_loaded_beam
from topop.core.fem import Assembler, elasticity_matrix, hex8_stiffness, rigid_body_modes
from topop.core.problem import Grid, Load, Material, Problem, Support
from topop.core.solver import LinearSolver


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
