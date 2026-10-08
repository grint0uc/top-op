# SIMP loop performance (WP C2)

Measured on the CI container (4 vCPU x86_64, 15 GB, OpenBLAS, Python 3.13, numpy 2.5, scipy 1.18,
pyamg 5.3), used as a proxy for the target M1 MacBook Pro 16 GB. The container is shared with
other jobs (load average 2-7 during these runs), so single timings scatter by up to ~2x; the
tables give typical values, and iteration counts are the reliable comparison. Fresh memory costs
~4 s/GB here (page faults): every temporary above glibc's 32 MB mmap threshold pays that again on
each call, so the loop now avoids large temporaries.

Benchmarks: `cantilever(nx, ny, nz)` full-box cantilevers from `topop.core.benchmarks` with
`cantilever_params()` (volfrac 0.3, penal 3, rmin 1.5). "s/it" is the optimizer's `t_iter`
(the whole iteration: assembly, solve, sensitivities, filter, OC), mean over iterations 2-6
unless stated otherwise. Scripts lived in the session scratchpad; the numbers below are copied
from their output.

## Result

| case | elements | before: s/it | after: s/it | before: setup / peak RSS | after: setup / peak RSS |
|---|---:|---:|---:|---:|---:|
| 60x20x4 | 4 800 | 0.57-0.60 (SuperLU) | **0.135-0.147** (banded Cholesky) | 0.3 s / 0.28 GB | 0.1-0.5 s / 0.25 GB |
| 50x50x40 | 100 000 | 14.7-20.1 (SA-AMG rebuilt every it) | **1.5** (its 2-6), 1.9 (2-12), 2.1 (2-30) | 7.8 s / 2.41 GB | 5.0 s / 1.53 GB |
| 80x56x56 | 250 880 | 40-53 | **3.8** (its 2-6), 4.3 (2-10) | 16 s / 5.48 GB | 13.7 s / 3.56 GB (float32: 2.76 GB) |

The regression benchmark (60x20x4, 100 iterations) takes 13.8 s instead of ~58 s and ends at
compliance 96.78056 (reference 96.78772, -7e-5). With `solver="amg"` it ends at 96.78620.

Targets: <= 0.2 s/it at 4 800 elements, <= 8 s/it at 100k (PLAN: 6 s on the M1), <= 20 s/it and
<= 8 GB at 250k. "Before" is the code at commit dcc132a run under the same conditions.

Default changes: `optimize` hands the solver the grid's geometric prolongators and a bandwidth
ordering, so "auto" uses a banded Cholesky for thin small problems and geometric-multigrid CG
otherwise; the CG tolerance is adaptive (`optimize.ADAPTIVE_TOL = 1e-4`); `LinearSolver(reuse=20)`
for the pyamg fallback. Threads: `TOPOP_THREADS` (default min(8, cores)) for the sparse kernels,
`TOPOP_BLAS_THREADS` (default 1) for OpenBLAS during solves.

## What was tried, in order

### 1. AMG hierarchy reuse

Reusing the whole pyamg smoothed-aggregation hierarchy (fine operator included) as a stale
preconditioner fails for SIMP. On the 60x20x4 run (18 900 free DOFs, CG rtol 1e-6, saved density
states):

| hierarchy built at -> used at | stale hierarchy: CG its | fresh hierarchy: CG its |
|---|---:|---:|
| it 1 -> it 2 | 649 | 22 |
| it 1 -> it 5 | 2000 (no convergence) | 26 |
| it 5 -> it 10, 10 -> 20, 20 -> 30, 50 -> 75 | 2000 | 27-40 |
| it 75 -> it 100 | 926 | 40 |

Void elements change stiffness by up to 1e7 between iterations, so a preconditioner built for
the old matrix is far from the new inverse. Reusing only the *prolongators* (aggregation and
smoothed P) and recomputing the Galerkin operators and smoothers on the new matrix works: 23-48
CG iterations vs 22-41 fresh, even with P from iteration 1 used at iteration 100. That is what
`LinearSolver(reuse=...)` now means for the pyamg path (default 20 solves, or a rebuild when CG
needs more than 2x the iterations seen right after the last rebuild).

The Galerkin product with SA's smoothed prolongator (26 nnz per row) is itself expensive at 100k
elements: A P 2.3-2.6 s + R (AP) 0.7-1.0 s single-threaded (1.6-2.2 s / 0.3 s with 4 threads),
on top of ~0.3 s per CG iteration (pyamg's Gauss-Seidel holds the GIL, so it cannot be threaded).
That led to step 2b.

### 2. Cheaper hierarchy

pyamg SA options on 60x20x4 (setup s, CG its at it 1 / 20 / 100):

| options | setup | CG its |
|---|---:|---|
| defaults (symmetric strength, jacobi smoothing, improve_candidates 4x GS) | 0.31-0.40 | 21 / 33 / 40 |
| improve_candidates=None | 0.20-0.24 | 22 / 35 / 41 |
| smooth weighting='local' | 0.20-0.27 | 28 / 38 / 41 |
| both | 0.13-0.21 | 28 / 37 / 41 |
| both + strength theta=0 | 0.19-0.23 | 28 / 37 / 41 |
| improve_candidates=None, max_coarse=300 (2 levels) | 0.27-1.0 | 9 / 14 / 17, but 2.5-6 s per solve |

At 100k elements the default SA setup is 14.7 s: 7.8 s prolongator smoothing (5.3 s of it the
spectral radius estimate), 2.1 s candidate improvement, 2.0 s strength. `improve_candidates=None`
plus `weighting='local'` brings it to 7.3 s. None of this helps enough once the hierarchy is
partially reused, because the Galerkin refresh then dominates. Not measured further: rootnode,
W-cycles, a float32 SA hierarchy and `ml.solve(accel='cg')`, all superseded by 2b.

### 2b. Geometric multigrid (new default for "amg" when `optimize` drives the solve)

On a voxel grid the coarse spaces are known: `Assembler.prolongators()` builds trilinear
interpolation from grids of spacing 2h, 4h, ... (3.4 nnz per row, rows restricted to the free
DOFs, coarse unknowns = coarse DOFs that reach a fine unknown), once per problem. Each solve
refreshes the Galerkin operators R A P (fixed P), so supports, inactive regions and void
stiffness are all carried algebraically. CG iterations at the same tolerance (1e-6), with
symmetric Gauss-Seidel or Jacobi-scaled Chebyshev smoothing on [0.1, 1.1] x lambda_max(D^-1 A):

| smoother | 60x20x4: it 1 / 20 / 100 | 50x50x40: it 1 / 10 / 40 | level-0 SpMVs per CG it |
|---|---|---|---:|
| SA (pyamg defaults), for reference | 21 / 33 / 40 | 11-19 (author) | ~6 (serial GS) |
| symmetric GS (pyamg kernel, serial) | 9 / 34 / 37 | 11 / 32 / 33 | ~6 (serial) |
| damped Jacobi 0.6, 1 sweep | 14 / 43 / 63 | | 3 |
| damped Jacobi 0.6, 2 sweeps | 11 / 36 / 58 | | 5 |
| Chebyshev deg 2 | 10 / 34 / 42 | 13 / 36 / 37 | 5 |
| Chebyshev deg 2, lower bound 0.1 (default) | | 12 / 28 / 29 | 5 |
| Chebyshev deg 3, lower bound 0.1 | 8 / 29 / 35 | 10 / 26 / 25 | 7 |

pyamg's own `chebyshev` (unscaled, on A) diverges on these matrices (400+ iterations). Chebyshev
degree 2 needs only SpMVs, which run in threads: scipy's `csr_matvec` releases the GIL, and row
blocks of ~equal nnz give 38.5 -> 13.7 ms per SpMV at 100k (4 threads, 2.8x). A V-cycle is 81 ms
at 100k, a CG iteration ~0.1 s. The Galerkin product is threaded by coarse row blocks,
(R_j A) P: 0.35-0.42 s -> 0.14 s at 100k (stacking a full A P costs more than the products,
mostly page faults). The coarsest level (<= 1000 unknowns) is a dense Cholesky.

### 3. Adaptive CG tolerance

`LinearSolver(adaptive_tol=loose)` with `solve(..., change=)`: rtol goes geometrically from 1e-6
at change <= 0.02 to `loose` at change >= 0.1. On 60x20x4 (100 its, geometric MG; the run never
gets below change 0.04):

| policy | total CG its | final compliance | vs SuperLU reference 96.78772 |
|---|---:|---:|---:|
| fixed 1e-6 | 2489 | 96.78451 | -3.3e-5 |
| loose 1e-5 | 2029 | 96.78830 | +0.6e-5 |
| loose 1e-4 (default) | 1541 | 96.78620 | -1.6e-5 |
| loose 1e-3 | 1059 | 96.78322 | -4.6e-5 |

50x50x40, 30 iterations: 688 / 376 / 205 CG iterations for fixed 1e-6 / loose 1e-4 / loose 1e-3,
final compliance 0.8155087 / 0.8155103 / 0.8155314. 1e-4 halves the CG work and moves the
result by 2e-6; 1e-3 saves another 45 % of CG iterations but only ~10 % of the iteration time
(the hierarchy refresh and the rest of the iteration are fixed costs), so the default is 1e-4.

### 4. Small problems (< 20k free DOFs)

SuperLU (`MMD_AT_PLUS_A`, `SymmetricMode`, `diag_pivot_thresh=0`; the previous code) takes
0.50-0.59 s per factorization at 18 900 DOFs, independent of BLAS threads. Alternatives:

- **Banded Cholesky** (LAPACK `dpbtrf`/`dpbtrs` through `scipy.linalg.lapack`), the band copied
  from K.data through a precomputed index map into a reused Fortran-order buffer: 0.105-0.13 s
  (fill 0.01, factor 0.10, solve 0.01), residual 1e-12. A fresh 51 MB buffer per call costs
  another 0.2 s here, hence the reuse. Cost ~ n bw^2: the ordering matters, and sweeping the
  longest axis slowest beats reverse Cuthill-McKee on these grids:

  | grid | n | natural bw | RCM bw | longest-axis-slowest bw |
  |---|---:|---:|---:|---:|
  | 60x20x4 | 18 900 | 335 | 629 | 335 |
  | 20x60x4 | 18 300 | 935 | 614 | 320 |
  | 4x20x60 | 15 372 | 4031 | 518 | 269 |
  | 17x17x17 | 16 524 | 1031 | 2651 | 974 |

  `Assembler.band_ordering()` provides it. One OpenBLAS thread (0.105 s) is as fast as two and
  faster than four (0.11-0.17 s).
- **Stale factorization as CG preconditioner** (banded factor of iteration k used at k+1):
  7-128 CG iterations at ~10 ms each, vs 0.12 s for a fresh factorization. Not worth it
  (agrees with the earlier finding).
- **AMG-CG** at this size: 0.26-0.47 s per solve without setup, slower than both.
- **CHOLMOD**: `scikit-sparse` 0.5.0 exists on PyPI but builds from source and needs
  `cholmod.h` (libsuitesparse), which is not installed here; not wired in.

Banded Cholesky vs geometric MG, whole iteration (mean of iterations 2-8, `solver="direct"` vs
`"amg"`):

| grid | n | bw (axis order) | n bw^2 | band | geometric MG |
|---|---:|---:|---:|---:|---:|
| 60x20x4 | 18 900 | 335 | 2.1e9 | **0.155 s** | 0.182 s |
| 120x20x4 | 37 800 | 335 | 4.2e9 | **0.281 s** | 0.402 s |
| 200x20x4 | 63 000 | 335 | 7.1e9 | **0.495 s** | 0.589 s |
| 80x16x8 | 36 720 | 491 | 8.9e9 | 0.439 s | **0.353 s** |
| 60x30x6 | 39 060 | 677 | 1.8e10 | 0.741 s | **0.371 s** |
| 40x24x12 | 39 000 | 1019 | 4.1e10 | 8.52 s | **0.337 s** |

The band costs ~n bw^2 / 1.5e10 s and multigrid ~9.5 us per unknown, so the crossover is a
bandwidth (~380) rather than a size: "auto" takes the band below `BAND_AUTO_BW = 380` (and
512 MB of band storage), otherwise geometric MG. "direct" uses the band up to n bw^2 = 4e10,
then SuperLU.

**BLAS threads.** With the pool threads running, OpenBLAS calls on small operands became very
slow under load (an 18 900-element dot product took 6 ms, a 576x576 Cholesky 0.74 s), because
OpenBLAS's spinning workers compete with the solver's threads. `LinearSolver.solve` pins every
loaded OpenBLAS (numpy's and scipy's copies, found through /proc/self/maps or dyld) to
`TOPOP_BLAS_THREADS` (default 1) for the duration of the solve and restores it afterwards.

### 5. Multiple load cases

One hierarchy refresh per solve; each case runs its own CG with the same V-cycle and warm-starts
from that case's previous displacement (`test_gmg_multiple_cases_warm_start_and_float32`).

### 6. Memory and assembly

- Assembly map as CSR (K entries x elements) instead of CSC (elements x K entries): K.data is a
  threaded gather written in place, 0.04-0.06 s at 100k, instead of a scatter into a fresh
  195 MB array (0.15-0.9 s depending on page faults). Each K entry with node offset d gets
  exactly 8/4/2/1 slots (one per corner pair with that offset), so the map is built without a
  sort; 1.2 % padding at 100k.
- Pattern building and element energies run in chunks (no temporary above a few MB);
  `np.tile` of the element matrix is gone.
- `params.dtype == "float32"`: K, the assembly map and the level-0 hierarchy are float32, CG runs
  in float64 with SpMVs that convert one row block of K at a time, so it still reaches 1e-6;
  coarse levels are float64. Peak RSS 1.19 GB at 100k and 2.76 GB at 250k (float64: 1.52 / 3.55).

Peak RSS (`resource.getrusage`, 4 iterations) vs the recalibrated `Assembler.estimate_bytes`:

| elements | float64 measured / estimate | float32 measured / estimate |
|---:|---|---|
| 12 000 | 0.32 / 0.33 GB | 0.31 / 0.29 GB |
| 30 000 | 0.66 / 0.58 GB | 0.60 / 0.47 GB |
| 100 000 | 1.52 / 1.55 GB | 1.19 / 1.18 GB |
| 250 880 | 3.55 / 3.62 GB | 2.76 / 2.69 GB |

The previous estimate gave 2.46 GB at 100k. 1M elements estimate 13.8 GB, refused at the default
6 GB cap. Setup at 100k is 5.0 s, mostly first-touch page faults of the 0.8 GB assembly map
(about 2 s/GB less on a native machine). A symmetric map (upper triangle plus a mirror gather)
would save ~17 % of the peak and ~1 s of setup at 100k; not done.

### 7. Time estimate

`fem.estimate_seconds_per_iter(nel_active)`: piecewise-linear in active elements through these
points (whole iteration; run averages are a bit above iterations 2-6 because CG needs more
iterations once void regions form), extrapolated linearly beyond 250k:

| elements | case | its 2-6 | its 2-N | estimate |
|---:|---|---:|---:|---:|
| 4 800 | 60x20x4 (band) | 0.147 | 0.142 (N=12) | 0.15 |
| 12 000 | 30x20x20 (MG) | 0.33 | 0.38 (N=12) | 0.40 |
| 30 000 | 40x30x25 (MG) | 0.61 | 0.81 (N=12) | 0.85 |
| 100 000 | 50x50x40 (MG) | 1.53 | 1.87 (N=12), 2.1 (N=30) | 2.0 |
| 250 880 | 80x56x56 (MG) | 3.77 | 4.31 (N=10) | 4.6 |

Thin parts that take the banded path and compact parts that take MG differ by up to ~2x at the
same element count (see the crossover table). `tests/test_perf.py` checks 60x20x4 <= 0.25 s and
50x50x40 <= 10 s per iteration (iterations 2-6), the estimate within 2x of both, and peak RSS
at 100k within 30 % of `estimate_bytes` for both dtypes. The server previously assumed
8e-5 s per element (8 s at 100k).

## Not done / next

- Symmetric assembly map (see 6); CHOLMOD when available (see 4).
- An M1 has 8 cores (4 performance): the sparse kernels use up to 8 threads (`TOPOP_THREADS`), and
  macOS wheels use Accelerate, so the OpenBLAS pinning is a no-op there. Re-run
  `tests/test_perf.py` there to recalibrate the estimate.
