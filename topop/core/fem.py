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


MG_COARSE_DOFS = 1000  # geometric MG coarsens until a level has at most this many unknowns


def interpolation_1d(n: int) -> tuple[sp.csr_matrix, int]:
    """(n+1, nc+1) linear interpolation from coarse nodes (spacing 2h) to the n+1 fine nodes.

    Axes with a single cell are not coarsened (identity, nc = n).
    """
    if n < 2:
        return sp.identity(n + 1, format="csr"), n
    nc = (n + 1) // 2
    i = np.arange(n + 1)
    ev, od = i[i % 2 == 0], i[i % 2 == 1]
    rows = np.concatenate([ev, od, od])
    cols = np.concatenate([ev // 2, od // 2, od // 2 + 1])
    vals = np.concatenate([np.ones(ev.size), np.full(2 * od.size, 0.5)])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n + 1, nc + 1)), nc


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
        self._prolongators: list[sp.csr_matrix] | None = None

    def _build_pattern(self, enodes: np.ndarray, free: np.ndarray, free_map: np.ndarray) -> None:
        # Node n couples to the nodes sharing an element with it ("slots", sorted by node id ==
        # stencil order). Free row 3n+ai holds, per slot, the free axes of the neighbour, so every
        # element-matrix entry's CSR position is a sum of small gathered tables -- no sort.
        # The assembly map P (K entries x elements, CSR) gives K entry q with node offset d
        # exactly _STENCIL_MULT[d] slots, one per corner pair (a, b) with that offset
        # (_PAIR_RANK); slots of missing elements stay (0, 0.0). Element and row loops run in
        # chunks so no temporary exceeds a few MB (large temporaries are fresh mmaps per call).
        nel = enodes.shape[0]
        n_nodes = self.n_dof // 3
        coupled = np.zeros((n_nodes, 27), dtype=bool)
        for lo, hi in _chunks(nel, _ELEM_CHUNK):
            coupled[enodes[lo:hi, :, None], _PAIR_OFFSET[None, :, :]] = True
        cnt = coupled.sum(axis=1)
        slot_start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        slot_of = np.cumsum(coupled, axis=1) - 1 + slot_start[:, None]
        full_ids = self.node_ids[:, None] + _stencil_full_offsets(self.problem.grid)[None, :]
        nbr = self.node_map[full_ids[coupled]]  # neighbour node of every slot
        mult = _STENCIL_MULT[np.nonzero(coupled)[1]]  # P slots per K entry of every slot
        del full_ids

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
        fm = free_map[3 * nbr[:, None] + np.arange(3)]
        keep = fm >= 0
        cols_by_node = fm[keep]  # node n's column list, concatenated over nodes
        mult_by_node = np.broadcast_to(mult[:, None], fm.shape)[keep].astype(np.int8)
        del fm, keep, nbr, mult
        node_col_start = np.concatenate([[0], np.cumsum(row_nnz)[:-1]])
        # gather index of K entry q of free row r: node_col_start[node(r)] + (q - indptr[r])
        p_len = np.empty(nnz, dtype=np.int8)
        idx64 = max(self.n_dof, 8 * nnz) >= 2**31 - 1
        idx_dtype = np.int64 if idx64 else np.int32
        indices = np.empty(nnz, dtype=idx_dtype)
        row_node = self.free_dofs // 3
        for lo, hi in _chunks(self.n_free, _ROW_CHUNK):
            a, b = indptr[lo], indptr[hi]
            g = np.arange(a, b) + np.repeat(
                node_col_start[row_node[lo:hi]] - indptr[lo:hi], row_len[lo:hi]
            )
            indices[a:b] = cols_by_node[g]
            p_len[a:b] = mult_by_node[g]
        del cols_by_node, mult_by_node
        p_indptr = np.zeros(nnz + 1, dtype=idx_dtype)
        np.cumsum(p_len, out=p_indptr[1:], dtype=idx_dtype)
        del p_len
        nnz_p = int(p_indptr[-1])
        p_indices = np.zeros(nnz_p, dtype=idx_dtype)
        p_data = np.zeros(nnz_p, dtype=self.dtype)

        row_start = np.full(self.n_dof, -1, dtype=idx_dtype)
        row_start[self.free_dofs] = indptr[:-1]
        col_before = col_before.astype(idx_dtype)
        axis_before = axis_before.astype(idx_dtype)
        ke = self.KE_h.astype(self.dtype).reshape(8, 3, 8, 3)
        rank = _PAIR_RANK.astype(idx_dtype)[None, :, None, :, None]
        for lo, hi in _chunks(nel, _ELEM_CHUNK):
            en = enodes[lo:hi]
            m = hi - lo
            rs = row_start[3 * en[:, :, None] + np.arange(3)]  # (m, 8, 3)
            cb = col_before[slot_of[en[:, :, None], _PAIR_OFFSET[None]]]  # (m, 8, 8)
            ab = axis_before[en]  # (m, 8, 3)
            # entry (e, a, ai, b, aj) == KE_h[3a+ai, 3b+aj]
            q = rs[:, :, :, None, None] + cb[:, :, None, :, None] + ab[:, None, None, :, :]
            valid = (rs >= 0)[:, :, :, None, None] & free3[en][:, None, None, :, :]
            pos = p_indptr[q[valid]] + np.broadcast_to(rank, valid.shape)[valid]
            p_indices[pos] = np.broadcast_to(
                np.arange(lo, hi, dtype=idx_dtype)[:, None, None, None, None], valid.shape
            )[valid]
            p_data[pos] = np.broadcast_to(ke[None], (m, 8, 3, 8, 3))[valid]
        self._P = sp.csr_matrix((p_data, p_indices, p_indptr), shape=(nnz, nel))
        self._K = sp.csr_matrix(
            (np.zeros(nnz, dtype=self.dtype), indices, indptr.astype(idx_dtype)),
            shape=(self.n_free, self.n_free),
        )
        self._K.has_sorted_indices = True
        self._P_blocks = None

    @property
    def n_elements(self) -> int:
        return int(self.element_ids.size)

    def assemble(self, E_e: np.ndarray) -> sp.csr_matrix:
        """K restricted to free DOFs for per-active-element moduli E_e (h scaling included)."""
        E_e = np.asarray(E_e, dtype=self.dtype)
        if E_e.shape != (self.n_elements,):
            raise ValueError(f"E_e must have shape ({self.n_elements},), got {E_e.shape}")
        if self._P_blocks is None:
            from topop.core.solver import _Blocks, default_threads

            self._P_blocks = _Blocks(self._P, default_threads())
        self._P_blocks.matvec(E_e, self._K.data)
        return self._K

    def expand(self, U_free: np.ndarray) -> np.ndarray:
        """Free-DOF vector(s) -> all compressed DOFs, zeros on fixed DOFs."""
        U_free = np.asarray(U_free)
        U = np.zeros((self.n_dof, *U_free.shape[1:]), dtype=U_free.dtype)
        U[self.free] = U_free
        return U

    def prolongators(self, min_dofs: int = MG_COARSE_DOFS) -> list[sp.csr_matrix]:
        """Geometric MG transfer operators: trilinear interpolation from grids of spacing 2h, 4h...

        P[0] maps the unknowns of the next coarser level to the free DOFs (rows in free-DOF
        order); P[l] maps level l+1 to level l. Coarse unknowns are the coarse-grid DOFs whose
        interpolation reaches at least one unknown of the finer level, so supports and inactive
        regions are carried through the Galerkin products. Built once and cached.
        """
        if self._prolongators is not None:
            return self._prolongators
        shape = tuple(self.problem.grid.shape)
        nodes = self.node_ids[self.free_dofs // 3]  # full-grid node of every unknown
        axes = self.free_dofs % 3
        out: list[sp.csr_matrix] = []
        n = nodes.size
        while n > min_dofs:
            ops = [interpolation_1d(k) for k in shape]
            cshape = tuple(nc for _, nc in ops)
            if cshape == shape:
                break
            Pn = sp.kron(sp.kron(ops[0][0], ops[1][0]), ops[2][0], format="csr")
            sub = Pn[nodes]
            cdof = 3 * sub.indices.astype(np.int64) + np.repeat(axes, np.diff(sub.indptr))
            used, col = np.unique(cdof, return_inverse=True)
            idx = np.int32 if max(sub.nnz, used.size) < 2**31 - 1 else np.int64
            P = sp.csr_matrix(
                (sub.data, col.astype(idx), sub.indptr.astype(idx)), shape=(n, used.size)
            )
            out.append(P)
            nodes, axes, shape, n = used // 3, used % 3, cshape, used.size
        self._prolongators = out
        return out

    def band_ordering(self) -> np.ndarray:
        """Permutation of the free DOFs ordering nodes by (longest axis, middle, shortest).

        Gives the smallest bandwidth of the axis-sweep orderings for the banded Cholesky
        (reverse Cuthill-McKee is ~2x wider on these grids).
        """
        ijk = np.stack(np.unravel_index(self.node_ids, self.problem.grid.node_shape), axis=1)
        extent = ijk.max(axis=0) - ijk.min(axis=0)
        order = np.argsort(-extent, kind="stable")  # slowest first
        node = self.free_dofs // 3
        keys = [self.free_dofs % 3] + [ijk[node, a] for a in order[::-1]]
        return np.lexsort(keys)

    def element_energies(self, U_free: np.ndarray) -> np.ndarray:
        """(nel_active,) u_e^T (h KE) u_e summed over load cases."""
        U = self.expand(U_free)
        if U.ndim == 1:
            U = U[:, None]
        out = np.zeros(self.n_elements)
        for c in range(U.shape[1]):
            u = U[:, c]
            for lo, hi in _chunks(self.n_elements, 32768):  # temporaries stay a few MB
                ue = u[self.edof[lo:hi]].astype(np.float64)
                out[lo:hi] += np.einsum("ij,ij->i", ue @ self.KE_h, ue)
        return out

    @staticmethod
    def estimate_bytes(nel_active: int, dtype=np.float64) -> int:
        """Peak bytes of a run for `nel_active` elements of a box-like domain.

        Assembly map (576 entries per element plus boundary padding, CSR) + K_free + edof +
        multigrid level 1 + work vectors + interpreter, x1.12 for allocator slack and the
        Galerkin/pattern temporaries. Calibrated on peak RSS (docs/PERF.md: 2-6 % at 100k and
        250k elements); thin domains have more nodes per element and need somewhat more.
        """
        item = np.dtype(dtype).itemsize
        n = int(nel_active)
        nodes = n + 3 * n ** (2 / 3) + 8
        nnz = 228 * nodes  # K_free nonzeros (81 per interior DOF row)
        entries = 583 * n
        persistent = entries * (4 + item) + nnz * (8 + item)  # map, its row pointers, K
        persistent += 192 * n + 900 * nodes + 0.124 * nnz * 12  # edof, vectors, MG level 1
        return int(1.12 * persistent + 160_000_000)  # interpreter + chunk temporaries


# (active elements, seconds per SIMP iteration averaged over a run) on full-box cantilevers,
# 4-core CI VM, see docs/PERF.md "Time estimate"
_SEC_PER_ITER = np.array(
    [[0, 0.01], [4_800, 0.15], [12_000, 0.4], [30_000, 0.85], [100_000, 2.0], [250_000, 4.6]]
)


def estimate_seconds_per_iter(nel_active: int) -> float:
    """Seconds per SIMP iteration (whole iteration, average over a run) for a box-like domain.

    Piecewise-linear fit of measurements on the 4-core CI VM (proxy for an M1, see PERF.md);
    extrapolated linearly above the last point.
    """
    n = float(max(0, nel_active))
    x, y = _SEC_PER_ITER[:, 0], _SEC_PER_ITER[:, 1]
    if n > x[-1]:
        return float(y[-1] + (n - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]))
    return float(np.interp(n, x, y))


_ELEM_CHUNK = 4096  # elements per pattern-building chunk (~20 MB of temporaries)
_ROW_CHUNK = 32768
_STENCIL = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
# elements that can share a node pair with this offset: 8, 4, 2, 1 for 0..3 nonzero components
_STENCIL_MULT = 2 ** (3 - np.count_nonzero(_STENCIL, axis=1))
_PAIR_RANK = np.zeros((8, 8), dtype=np.int64)  # rank of corner pair (a, b) among same-offset pairs
for _s in range(27):
    _ab = np.argwhere(_PAIR_OFFSET == _s)
    _PAIR_RANK[_ab[:, 0], _ab[:, 1]] = np.arange(len(_ab))


def _chunks(n: int, size: int):
    for lo in range(0, n, size):
        yield lo, min(n, lo + size)


def _stencil_full_offsets(grid) -> np.ndarray:
    ny, nz = grid.node_shape[1], grid.node_shape[2]
    d = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    return d @ np.array([ny * nz, nz, 1])
