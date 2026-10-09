"""Time and memory budgets of the SIMP loop (numbers and method in docs/PERF.md).

Budgets are generous against the targets (0.2 s / 8 s per iteration) so a loaded CI box does not
flake; the estimates used by the server must stay within a factor 2 of what is measured here.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import numpy as np
import pytest

from topop.core.benchmarks import cantilever, cantilever_params
from topop.core.fem import Assembler, estimate_seconds_per_iter
from topop.core.optimize import optimize


def mean_iteration_time(shape: tuple[int, int, int], n_iter: int = 6) -> tuple[float, int]:
    """Mean wall time of iterations 2..n_iter (iteration 1 also warms caches and buffers)."""
    p = cantilever(*shape)
    res = optimize(p, dataclasses.replace(cantilever_params(), max_iter=n_iter))
    assert len(res.history) == n_iter
    return float(np.mean([h.t_iter for h in res.history[1:]])), p.n_active


@pytest.mark.slow
@pytest.mark.parametrize(("shape", "budget"), [((60, 20, 4), 0.25), ((50, 50, 40), 10.0)])
def test_seconds_per_iteration_and_estimate(shape, budget):
    t, nel = mean_iteration_time(shape)
    assert t <= budget, f"{shape}: {t:.3f} s/iteration > {budget}"
    est = estimate_seconds_per_iter(nel)
    assert est / 2 <= t <= 2 * est, f"{shape}: measured {t:.3f} s, estimate {est:.3f} s"


# Peak RSS of the child itself: VmHWM (high-water mark of this process's address space). Not
# ru_maxrss: on Linux exec() records the replaced address space's high-water mark into the new
# program's ru_maxrss, and subprocess vforks, so the child reported the *pytest process's* peak
# (2.14-2.19 GB inside the full suite, after the in-process 50x50x40 timing test above) for a run
# that peaks at 1.1-1.3 GB. `rss0`, ru_maxrss before any allocation, shows the inherited value.
_RSS_SCRIPT = """
import dataclasses, json, resource, sys
from topop.core import solver
from topop.core.benchmarks import cantilever, cantilever_params
from topop.core.optimize import optimize
scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
methods = set()
real = solver.LinearSolver.solve
def solve(self, *a, **k):
    U, info = real(self, *a, **k)
    methods.add(info.method)
    return U, info
solver.LinearSolver.solve = solve
shape, dtype = tuple(json.loads(sys.argv[1])), sys.argv[2]
p = cantilever(*shape)
optimize(p, dataclasses.replace(cantilever_params(), max_iter=3, dtype=dtype))
try:
    with open("/proc/self/status") as f:
        peak = next(int(ln.split()[1]) * 1024 for ln in f if ln.startswith("VmHWM:"))
except (OSError, StopIteration):  # not Linux: ru_maxrss (may include the parent's peak)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
print(json.dumps({"rss": peak, "rss0": rss0, "nel": p.n_active, "methods": sorted(methods),
                  "threads": solver.default_threads()}))
"""


@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_peak_memory_matches_estimate(dtype):
    out = subprocess.run(
        [sys.executable, "-c", _RSS_SCRIPT, json.dumps([50, 50, 40]), dtype],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    m = json.loads(out.stdout.strip().splitlines()[-1])
    est = Assembler.estimate_bytes(m["nel"], np.dtype(dtype))
    # a memory guard may overestimate; it must not underestimate
    assert 0.7 * m["rss"] <= est <= 2.0 * m["rss"], (
        f"RSS {m['rss'] / 1e9:.2f} GB, est {est / 1e9:.2f} ({m})"
    )
    assert m["methods"] == ["gmg"]  # a fallback (Jacobi-PCG, SA) would change the footprint
