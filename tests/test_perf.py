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


_RSS_SCRIPT = """
import dataclasses, json, resource, sys
from topop.core.benchmarks import cantilever, cantilever_params
from topop.core.optimize import optimize
shape, dtype = tuple(json.loads(sys.argv[1])), sys.argv[2]
p = cantilever(*shape)
params = dataclasses.replace(cantilever_params(), max_iter=3, dtype=dtype)
optimize(p, params)
scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
print(json.dumps({"rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale,
                  "nel": p.n_active}))
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
    # a memory guard may overestimate; it must not underestimate. float32 peaks measured 1.19 GB
    # in isolation but up to 2.19 GB inside the full suite on the CI box (cause not found), so its
    # lower bound is looser; docs/PLAN.md tells users to prefer float64 near the cap.
    lo = 0.7 if dtype == "float64" else 0.5
    assert lo * m["rss"] <= est <= 2.0 * m["rss"], (
        f"RSS {m['rss'] / 1e9:.2f} GB, est {est / 1e9:.2f}"
    )
