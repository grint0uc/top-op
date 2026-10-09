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

## v0.2 note: Chebyshev bound and CG breakdown

Stress-constrained runs (adjoint solves with very different right-hand sides, warm-started) exposed CG
breakdowns: the smoother's upper eigenvalue bound came from a 4-step power iteration restarted from the
previous solve's vector, which gets stuck on a moving solid/void boundary and underestimated λmax by up to
2x, making the V-cycle indefinite. The bound is now 8 Lanczos steps from a fixed random start on every
refresh (≥ 0.967x the true λmax over 170 recorded matrices; the cycle needs ≥ 0.826x). Cost: about +5 % per
iteration at 100k elements (1.37-1.39 → 1.44-1.50 s). Fallbacks inside `solve`: restart from zero →
Gershgorin bounds (`gmg-safe`) → Jacobi-PCG, each tagged in `SolveInfo.method`. The replayed failure is
`tests/data/lbracket_gmg_breakdown.npz` (`tests/test_solver.py`).

## v0.3: peak-RSS measurement, symmetric assembly map, band vs multigrid

Same 4-vCPU container. Other jobs came and went during this work (0 to ~2.5 busy cores); every
benchmark recorded that foreign load (system busy time minus its own CPU time), and the numbers
below are from runs with less than 0.3 foreign cores unless marked. Scripts lived in the session
scratchpad.

### 1. The float32 peak "only inside the full suite"

`test_peak_memory_matches_estimate` read the child's peak from `ru_maxrss`. On Linux that value
survives `exec`: `exec_mmap` folds the high-water RSS of the address space being replaced into the
process's `signal->maxrss`, and `subprocess` starts children with vfork, so the replaced address
space is the *parent's*. The child reported max(own peak, pytest's peak so far). Instrumented child
in a full `uv run pytest -q` (327 passed): `ru_maxrss` was 2.138 GB at its first line, before
importing topop, for both dtypes, equal to the pytest process's `ru_maxrss` at the spawn; its own
`VmHWM` was 1.33 GB (float64) / 1.09 GB (float32), tracemalloc peak 1.10 / 0.87 GB; every solve
`gmg`, 4 threads, no `TOPOP_*`/`OMP_*`/`OPENBLAS_*`/`MKL_*` variables. A pytest plugin logging the
pytest process's `ru_maxrss` after every test (one run instead of bisecting files) shows where the
2.1 GB came from: 0.85 GB after the server/CLI/MCP/export tests, 2.09 GB after
`test_seconds_per_iteration_and_estimate[50x50x40]`, which runs the 100k optimization in-process
just before the memory test. `tests/test_stress.py tests/test_solver.py` alone leave pytest well
below 1.2 GB, hence no reproduction. Minimal reproduction: a parent touches and frees 2 GB, then
spawns `python -c` printing its own numbers: `ru_maxrss` 2.03 GB, `VmHWM` 0.01 GB. Not an
environment variable, a fallback, a leftover process or transparent huge pages (THP is `madvise`,
no compaction stalls recorded).

Fix (test only): the child reports `VmHWM` from `/proc/self/status` (`ru_maxrss` where there is no
procfs), the float32 lower bound is 0.7 again, and the test asserts that every solve was `gmg`.

### 2. Symmetric assembly map

`Assembler(..., symmetric_map=True)` (default): P has slots only for K entries with col >= row.
Which element-matrix entries those are depends only on the corner pair (`fem._UPPER`, 300 of 576),
because free DOFs are ordered by (node, axis) and node ids grow with the stencil offset. Rows of
the lower triangle are empty in P, so the threaded `P @ E_e` writes 0 there, and a threaded gather
`K.data[dst] = K.data[src]` fills them. The int32 index pairs come from the slot tables while the
pattern is built, with no sort. K is now exactly symmetric. `symmetric_map=False` keeps the full
map; `tests/test_fem.py` checks that both agree to 1e-12.

| | 100k full | 100k symmetric | 250k full | 250k symmetric |
|---|---:|---:|---:|---:|
| map + mirror indices | 797 MB | 558 MB | 1999 MB | 1401 MB |
| peak RSS (VmHWM, 4-7 its) | 1.56 GB | **1.32-1.34 GB (-15 %)** | 3.64-3.66 GB | **3.02-3.05 GB (-17 %)** |
| `Assembler` build, fresh process | 2.4-2.5 s (6.5 s once) | 2.6-3.2 s | 12.6-14.1 s | **6.9-8.1 s** |
| `assemble()`, min of 15 | 33 ms | 54 ms | 85 ms | 140 ms |
| setup to first iteration (`optimize`) | 3.6-6.4 s | 4.9-6.1 s | 15.1-15.2 s | 9.0-14.9 s |
| s/it, its 2-6 (BLAS pinned, quiet box) | 1.43-1.53 (mean 1.47) | 1.43-1.53 (mean 1.48, **+0.4 %**) | 3.46-3.50 (mean 3.48) | 3.54-3.55 (mean 3.55, **+1.9 %**) |

CG iteration counts are identical (the same K up to rounding), so the per-iteration cost of the
symmetric map is the extra assembly time: +21 ms at 100k and +55 ms at 250k, about 1.5 % of an
iteration. The mirror is 23 ms at 100k with int64 indices and 31 ms with int32, using 8 chunks in
the pool. int32 is kept because it saves 48 MB; sub-chunking did not help. At 100k the build is not
faster: the extra row-loop work cancels the smaller allocations. At 250k it halves, because 600 MB
less is touched for the first time.

`estimate_bytes` recalibrated to (305 map entries x (4 + item) + K + 4 B per K entry of mirror +
edof, vectors, MG level 1) x 1.15 + 200 MB:

| elements | float64 measured / estimate | float32 measured / estimate |
|---:|---|---|
| 12 000 | 0.31 / 0.34 GB | 0.31 / 0.31 GB |
| 30 000 | 0.68 / 0.55 GB | 0.64 / 0.48 GB |
| 100 000 | 1.32 / 1.35 GB | 1.09 / 1.10 GB |
| 250 880 | 3.05 / 3.06 GB | 2.48 / 2.44 GB |

The 30k runs peak above the estimate. Mid-size temporaries stay on the glibc heap below the 32 MB
mmap threshold; the old formula was also -12 / -22 % there. 1M elements now estimate 11.5 GB,
still refused at the 6 GB default cap.

### 3. The 12k-30k "gap"

**Main cause: OpenBLAS outside the solve.** A profile of 30x20x20 under foreign load found
`optimize._linear_volume` taking 4.5 of 9.3 s. It is one `dv @ (x - x0)` per OC bisection step,
~31 per iteration. OpenBLAS threads `ddot` above n = 10 000, and with the cores busy every such
call waits for its workers. Median of 200 calls under load:

| n | 4 800 | 9 000 | 12 000 | 30 000 | 100 000 |
|---|---:|---:|---:|---:|---:|
| default threads | 1.7 us | 2.3 us | 8.0 ms | 8.0 ms | 8.0 ms |
| pinned to 1 (`solver._BLAS`) | 1.7 us | 2.3 us | 3.4 us | 5.7 us | 26 us |

`LinearSolver.solve` pins OpenBLAS only for the duration of a solve. This is why the gap starts at
~10k free elements (60x20x4 has 4 800 and never pays it) and why it comes and goes with load.
30x20x20, 8 iterations, ~2.4 foreign cores: 0.62-0.76 s/it as shipped, 0.30-0.32 s/it with the
whole `optimize` call inside `with solver._BLAS:`. On a quiet box the difference is ~10 %.

Fixed where the BLAS calls are, not with a loop-wide pin (a process-wide setting held for the whole
run): the OC volume bisection uses `einsum` instead of a `ddot`, and the Assembler's per-iteration
chunked GEMMs (`element_energies`, `element_energies_and_stress`, `element_stress`,
`von_mises_gradient`, `element_cross_energies`, and `assemble`) run under `fem._blas_pinned`,
i.e. `with solver._BLAS:` (re-entrant), ~8 ms -> 0.74 ms per call under load. 30x20x20, 12
iterations, ~1.5 foreign cores: 0.307 s/it with only the solve pinned, 0.285-0.292 s/it with the
Assembler methods pinned, the same as with the whole `optimize` call pinned (0.282-0.289).

**(a) Reverse Cuthill-McKee.** Bandwidth in DOFs: axis sweep / RCM on the node graph:

| part | n | sweep | RCM | | part | n | sweep | RCM |
|---|---:|---:|---:|---|---|---:|---:|---:|
| 60x20x4 | 18 900 | **335** | 629 | | 20x20x20 | 26 460 | **1391** | 3659 |
| 80x16x8 | 36 720 | **491** | 917 | | 30x30x30 | 86 490 | **2981** | 8189 |
| 60x30x6 | 39 060 | **677** | 1301 | | bracket res 40 | 32 169 | 2837 | **1871** |
| L-bracket 40x4 | 16 320 | **620** | 779 | | bracket res 60 | 109 590 | 6206 | **4355** |
| ring R40 w5 t4 | 21 540 | 665 | **350** | | ring R30 w4 t6 | 18 774 | 740 | **428** |

Compact parts cannot get near the band's range in any ordering. A graph with N nodes and diameter D
has bandwidth >= (N - 1) / D, so 21^3 nodes with D = 20 need >= 463 nodes (~1390 DOFs). A
20x20x20 box on the band takes 1.34 s/it vs 0.18 s/it with GMG. RCM is 2-3x wider than the sweep
on boxes and beams, but ~2x narrower on closed loops. `band_ordering()` now takes the narrower of
the two, compared on the node graph (11-70 ms in the band-eligible range).

**(b) Crossover.** Iterations 2-8, s/it, band / GMG (`solver="direct"` / `"amg"`):

| part | bw | band | GMG | GMG CG its |
|---|---:|---:|---:|---|
| 120x20x4 | 335 | **0.25** | 0.35 | 6-15 |
| ring R40 w5 t4 | 350 | **0.15** | 0.47 | 13-40 |
| 120x20x5 | 401 | **0.35-0.36** | 0.38-0.39 | 6-15 |
| 100x26x4 | 425 | 0.34-0.35 | **0.29-0.30** | 5-14 |
| ring R30 w4 t6 | 428 | **0.17** | 0.38 | 14-28 |
| 100x20x6 | 467 | 0.43 | **0.35** | 6-14 |
| 80x16x8 | 491 | 0.38 | **0.31** | 6-10 |
| 80x24x6 | 551 | 0.51 | **0.31** | 6-14 |
| L-bracket 40x4 | 620 | 0.24 | **0.17** | 6-11 |
| 60x30x6 | 677 | 0.65 | **0.29** | 5-14 |

Band time is ~n bw^2 / 2.4e10 s (1.8e10 at bw 335 up to 2.8e10 at 677), or 0.0285 bw - 3.2 us per
unknown. GMG costs 0.6-1.0 us per unknown per CG iteration, and its CG count grows as voids form.
Over whole runs, s/it for iterations 2-50:

| part | bw | auto (new) | band | GMG |
|---|---:|---|---:|---:|
| 80x16x8 | 491 | **0.384** (GMG for 11 solves, then band) | 0.376 | 0.609 (late CG its 22-51) |
| 80x24x6 | 551 | **0.486** (GMG for 13 solves, then band) | 0.506 | 0.554 |
| L-bracket 40x4 | 620 | 0.253 (stays GMG) | 0.230 | 0.245 |

New `auto` policy: band from the start when bw <= `BAND_AUTO_BW = 410`, the beam crossover at
early iterations. Otherwise GMG, switching for good to the band once two consecutive solves need
more CG iterations than (0.0285 bw - 3.2) / 0.7, provided the band fits `BAND_MAX_BYTES`. That is
~15 iterations at bw 491 and ~24 at 700. Compact parts never qualify, because their band storage
is above 512 MB or the break-even is above 50 iterations. The switch counts iterations, not
seconds, so the choice is deterministic; the hierarchy is freed when it happens. `solver="amg"`
never switches.

**(c) Multigrid on small problems.** One solve at change 0.01, min of 3:

| variant | 30x20x20 (12k): solve / CG its | 40x30x25 (30k) |
|---|---|---|
| default (coarse <= 1000 unknowns; 3 / 4 levels) | 315 ms / 11 | 558 ms / 14 |
| coarse <= 300 (4 / 5 levels) | 310 ms / 13 | 571 ms / 15 |
| coarse <= 3000 (dense Cholesky of 972 / 2376) | 315 ms / 11 | 1154 ms / 10 |
| coarse <= 8000 (2 levels, SuperLU of 5 808) | 586 ms / 7 | 1143 ms / 10 |
| float32 level-0 hierarchy, float64 CG | 314 ms / 11 | 530 ms / 14 |
| 1 / 2 / 4 threads | 274 / 269 / 307 ms | 795 / 555 / 546 ms |

At 12k the setup is 83-93 ms: the level-0 Galerkin product is 43 ms and level-0 Lanczos 14 ms. A
V-cycle is 14-16 ms, and a level-0 SpMV is 1.3 ms with 4 threads or 2.1 ms with 1. A
single-threaded CG iteration costs about 8 level-0 SpMVs, so there is no large overhead left to
cut. A larger coarse solve costs more than the iterations it saves. Defaults are kept.

**Before / after** (HEAD `fem.py`/`solver.py` vs this version, same tree otherwise; 30
iterations, s/it its 2-30, foreign load ~2.4 cores during these pairs, so both columns include
OpenBLAS stalls):

| part | elements | before | after |
|---|---:|---|---|
| 30x20x20 (quiet) | 12 000 | 0.323 (GMG) | 0.333 (GMG) |
| bracket res 40 | 9 340 | 0.322 (GMG) | 0.322 (GMG) |
| ring R40 w5 t4 | 4 688 | 0.480 (GMG) | **0.244** (band, RCM) |
| ring R30 w4 t6 | 4 200 | 0.488 (GMG) | **0.278** (GMG for 2 solves, then band) |
| 80x16x8 | 10 240 | 1.134 (GMG) | **0.899** (GMG, then band) |

Quiet-box run averages of the current code: 4 800 (band) 0.129, bracket 9 340 0.309, 12k 0.326,
30k 0.666, bracket 33k 0.687, 100k 1.74 (N=21), 250k 4.14 (N=11) s/it.

**Target** (no size between 5k and 100k above 0.35 s/it). On a quiet box, or with OpenBLAS pinned
for the whole loop, the target holds up to ~12k elements for compact parts, and for thin parts and
rings on the band. It is not reachable at 30k-100k with this design. Multigrid costs 0.6-1 us per
unknown and CG iteration at 10-25 iterations, which gives 0.67 s/it at 30k and 1.7-1.9 s/it at
100k. The band costs n bw^2, and compact bandwidths are bounded by the cross-section. Closing that
would need a nested-dissection sparse Cholesky (CHOLMOD, not installed) or a substantially
stronger preconditioner.

### 4. Time estimate

`estimate_seconds_per_iter` points are now 4 800: 0.14, 12k: 0.34, 30k: 0.70, 100k: 1.9,
250k: 4.4 s. These are run averages of the measurements above, rounded up for longer runs. They
assume no foreign load; under load, compact parts above 10k free elements pay the OpenBLAS stalls
until `optimize.py` pins BLAS.
