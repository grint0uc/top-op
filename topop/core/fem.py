"""Hex8 element, rigid body modes and the voxel-grid assembler (active elements only)."""

from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp

from topop.core.problem import HEX8_OFFSETS, Problem

# natural coordinates (+-1) of the 8 nodes, HEX8_OFFSETS order
_SIGNS = (2 * HEX8_OFFSETS - 1).astype(np.float64)

# (8, 8) index of the node-pair offset HEX8_OFFSETS[b] - HEX8_OFFSETS[a] in the 27-neighbour
# stencil, lexicographic over (dx, dy, dz) in {-1,0,1}^3 (== increasing full node id offset)
_PAIR_OFFSET = (HEX8_OFFSETS[None, :, :] - HEX8_OFFSETS[:, None, :] + 1) @ np.array([9, 3, 1])


def elasticity_matrix(nu: float) -> np.ndarray:
    """Isotropic D for E=1, Voigt order (xx, yy, zz, xy, yz, zx), engineering shear."""
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1 / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[[0, 1, 2], [0, 1, 2]] += 2 * mu
    D[[3, 4, 5], [3, 4, 5]] = mu
    return D


def hex8_strain_matrix(xi: np.ndarray) -> np.ndarray:
    """(6, 24) B at natural point `xi` for the unit cube (x = (xi + 1) / 2)."""
    f = 1 + _SIGNS * np.asarray(xi, dtype=np.float64)  # (8, 3)
    dN = np.empty((8, 3))
    dN[:, 0] = _SIGNS[:, 0] * f[:, 1] * f[:, 2]
    dN[:, 1] = _SIGNS[:, 1] * f[:, 0] * f[:, 2]
    dN[:, 2] = _SIGNS[:, 2] * f[:, 0] * f[:, 1]
    dN *= 2 / 8  # N = prod(1 + s xi) / 8, d/dx = 2 d/dxi
    B = np.zeros((6, 24))
    x, y, z = (slice(k, 24, 3) for k in range(3))
    B[0, x] = dN[:, 0]
    B[1, y] = dN[:, 1]
    B[2, z] = dN[:, 2]
    B[3, x], B[3, y] = dN[:, 1], dN[:, 0]
    B[4, y], B[4, z] = dN[:, 2], dN[:, 1]
    B[5, x], B[5, z] = dN[:, 2], dN[:, 0]
    return B


def hex8_stiffness(nu: float) -> np.ndarray:
    """(24, 24) stiffness of the unit cube for E=1; a cube of edge h has h * KE."""
    D = elasticity_matrix(nu)
    g = 1 / np.sqrt(3)
    KE = np.zeros((24, 24))
    for xi in itertools.product((-g, g), repeat=3):
        B = hex8_strain_matrix(np.array(xi))
        KE += B.T @ D @ B / 8  # weight 1, det J = 1/8
    return (KE + KE.T) / 2


def rigid_body_modes(coords: np.ndarray) -> np.ndarray:
    """(3n, 6): 3 translations and 3 rotations (about the centroid) for nodes at `coords`."""
    c = np.asarray(coords, dtype=np.float64)
    c = c - c.mean(axis=0)
    n = c.shape[0]
    B = np.zeros((n, 3, 6))
    B[:, [0, 1, 2], [0, 1, 2]] = 1.0
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    B[:, 0, 3], B[:, 1, 3] = -y, x  # about z
    B[:, 1, 4], B[:, 2, 4] = -z, y  # about x
    B[:, 0, 5], B[:, 2, 5] = z, -x  # about y
    return B.reshape(3 * n, 6)


class Assembler:
    """Global stiffness on the compressed active node set, restricted to free DOFs.

    The CSR pattern of K_free and a sparse map P with K_free.data = P @ E_e are built once;
    `assemble` only refreshes the data array. The returned matrix object is reused.
    """

    def __init__(self, problem: Problem, dtype=np.float64):
        grid = problem.grid
        self.problem = problem
        self.dtype = np.dtype(dtype)
        self.h = float(grid.h)
        self.KE = hex8_stiffness(problem.material.nu)
        self.KE_h = self.h * self.KE

        active = np.asarray(problem.active, dtype=bool)
        self.element_ids = np.flatnonzero(active.ravel())
        nel = self.element_ids.size
        ijk = np.stack(np.unravel_index(self.element_ids, grid.shape), axis=1)
        corners = ijk[:, None, :] + HEX8_OFFSETS[None, :, :]
        full_nodes = np.ravel_multi_index(tuple(np.moveaxis(corners, 2, 0)), grid.node_shape)

        used = np.zeros(grid.n_nodes, dtype=bool)
        used[full_nodes] = True
        self.node_ids = np.flatnonzero(used)  # compressed -> full, increasing
        n_nodes = self.node_ids.size
        self.node_map = np.full(grid.n_nodes, -1, dtype=np.int64)
        self.node_map[self.node_ids] = np.arange(n_nodes)
        enodes = self.node_map[full_nodes]  # (nel, 8)
        self.edof = (3 * enodes[:, :, None] + np.arange(3)).reshape(nel, 24)
        self.n_dof = 3 * n_nodes
        self.node_coords_compressed = np.asarray(grid.origin, dtype=np.float64) + self.h * np.stack(
            np.unravel_index(self.node_ids, grid.node_shape), axis=1
        )

        # free DOFs
        free = np.ones(self.n_dof, dtype=bool)
        for s in problem.supports:
            cn = self.node_map[np.asarray(s.nodes, dtype=np.int64)]
            cn = cn[cn >= 0]
            for axis, fixed in enumerate(s.fix):
                if fixed:
                    free[3 * cn + axis] = False
        self.free = free
        self.free_dofs = np.flatnonzero(free)
        self.n_free = self.free_dofs.size
        free_map = np.full(self.n_dof, -1, dtype=np.int64)
        free_map[self.free_dofs] = np.arange(self.n_free)

        # loads
        self.n_cases = problem.n_cases
        F = np.zeros((self.n_dof, self.n_cases))
        for ld in problem.loads:
            cn = self.node_map[np.asarray(ld.nodes, dtype=np.int64)]
            if cn.size == 0 or (cn < 0).any():
                raise ValueError("load references nodes outside the active domain")
            for axis in range(3):
                np.add.at(F[:, ld.case], 3 * cn + axis, ld.force[axis] / cn.size)
        self.F = F
        self.F_free = np.ascontiguousarray(F[free])

        self._build_pattern(enodes, free, free_map)

    def _build_pattern(self, enodes: np.ndarray, free: np.ndarray, free_map: np.ndarray) -> None:
        # Node n couples to the nodes sharing an element with it ("slots", sorted by node id ==
        # stencil order). Free row 3n+ai holds, per slot, the free axes of the neighbour, so every
        # element-matrix entry's CSR position is a sum of small gathered tables -- no sort.
        nel = enodes.shape[0]
        n_nodes = self.n_dof // 3
        coupled = np.zeros((n_nodes, 27), dtype=bool)
        coupled[enodes[:, :, None], _PAIR_OFFSET[None, :, :]] = True
        cnt = coupled.sum(axis=1)
        slot_start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        slot_of = np.cumsum(coupled, axis=1) - 1 + slot_start[:, None]
        full_ids = self.node_ids[:, None] + _stencil_full_offsets(self.problem.grid)[None, :]
        nbr = self.node_map[full_ids[coupled]]  # neighbour node of every slot

        free3 = free.reshape(n_nodes, 3)
        w = free3.sum(axis=1)[nbr]  # free columns contributed by each slot
        row_nnz = np.add.reduceat(w, slot_start)  # per node (identical for its 3 rows)
        cw = np.cumsum(w) - w
        col_before = cw - cw[slot_start][np.repeat(np.arange(n_nodes), cnt)]
        axis_before = np.cumsum(free3, axis=1) - free3

        row_len = row_nnz[self.free_dofs // 3]
        indptr = np.zeros(self.n_free + 1, dtype=np.int64)
        np.cumsum(row_len, out=indptr[1:])
        nnz = int(indptr[-1])
        idx_dtype = np.int32 if max(nnz, self.n_dof) < 2**31 - 1 else np.int64
        fm = free_map[3 * nbr[:, None] + np.arange(3)]
        cols_by_node = fm[fm >= 0]  # node n's column list, concatenated over nodes
        node_col_start = np.concatenate([[0], np.cumsum(row_nnz)[:-1]])
        shift = np.repeat(node_col_start[self.free_dofs // 3] - indptr[:-1], row_len)
        indices = cols_by_node[np.arange(nnz) + shift].astype(idx_dtype)
        del shift, fm, cols_by_node

        row_start = np.full(self.n_dof, -1, dtype=np.int64)
        row_start[self.free_dofs] = indptr[:-1]
        rs = row_start.astype(idx_dtype)[3 * enodes[:, :, None] + np.arange(3)]  # (nel, 8, 3)
        cb = col_before.astype(idx_dtype)[slot_of[enodes[:, :, None], _PAIR_OFFSET[None]]]
        ab = axis_before.astype(idx_dtype)[enodes]  # (nel, 8, 3)
        # entry (e, a, ai, b, aj) == KE_h[3a+ai, 3b+aj]
        tgt = (rs[:, :, :, None, None] + cb[:, :, None, :, None] + ab[:, None, None, :, :]).reshape(
            nel, 576
        )
        valid = ((rs >= 0)[:, :, :, None, None] & free3[enodes][:, None, None, :, :]).reshape(
            nel, 576
        )
        del rs, cb, ab
        counts = valid.sum(axis=1)
        tgt = tgt[valid]
        vals = np.tile(self.KE_h.ravel().astype(self.dtype), (nel, 1))[valid]
        del valid
        # column e of P holds element e's entries: K.data = P @ E_e (CSC matvec == scatter-add)
        p_dtype = np.int32 if max(nnz, tgt.size) < 2**31 - 1 else np.int64
        p_indptr = np.zeros(nel + 1, dtype=p_dtype)
        np.cumsum(counts, out=p_indptr[1:])
        self._P = sp.csc_matrix((vals, tgt.astype(p_dtype, copy=False), p_indptr), shape=(nnz, nel))
        del vals, tgt
        self._K = sp.csr_matrix(
            (np.zeros(nnz, dtype=self.dtype), indices, indptr.astype(idx_dtype)),
            shape=(self.n_free, self.n_free),
        )
        self._K.has_sorted_indices = True

    @property
    def n_elements(self) -> int:
        return int(self.element_ids.size)

    def assemble(self, E_e: np.ndarray) -> sp.csr_matrix:
        """K restricted to free DOFs for per-active-element moduli E_e (h scaling included)."""
        E_e = np.asarray(E_e, dtype=self.dtype)
        if E_e.shape != (self.n_elements,):
            raise ValueError(f"E_e must have shape ({self.n_elements},), got {E_e.shape}")
        self._K.data[:] = self._P @ E_e
        return self._K

    def expand(self, U_free: np.ndarray) -> np.ndarray:
        """Free-DOF vector(s) -> all compressed DOFs, zeros on fixed DOFs."""
        U_free = np.asarray(U_free)
        U = np.zeros((self.n_dof, *U_free.shape[1:]), dtype=U_free.dtype)
        U[self.free] = U_free
        return U

    def element_energies(self, U_free: np.ndarray) -> np.ndarray:
        """(nel_active,) u_e^T (h KE) u_e summed over load cases."""
        U = self.expand(U_free)
        if U.ndim == 1:
            U = U[:, None]
        out = np.zeros(self.n_elements)
        for c in range(U.shape[1]):
            ue = U[self.edof, c].astype(np.float64)
            out += np.einsum("ij,ij->i", ue @ self.KE_h, ue)
        return out

    @staticmethod
    def estimate_bytes(nel_active: int, dtype=np.float64) -> int:
        """Rough peak bytes of assembly + solve for `nel_active` elements."""
        item = np.dtype(dtype).itemsize
        n = int(nel_active)
        entries = 576 * n
        nodes = n + 3 * n ** (2 / 3) + 8  # box-like domain; thin domains have more nodes
        nnz = 243 * nodes  # 81 nonzeros per DOF row in the interior
        setup = entries * (8 + 8 + 2 * (4 + item))  # positions, masks and the two copies of P
        persistent = entries * (4 + item) + 4 * nnz * (4 + item)  # P, K, AMG hierarchy ~3x K
        vectors = 40 * 3 * nodes * 8
        return int(max(setup, persistent) + vectors + 50_000_000)


def _stencil_full_offsets(grid) -> np.ndarray:
    ny, nz = grid.node_shape[1], grid.node_shape[2]
    d = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    return d @ np.array([ny * nz, nz, 1])
