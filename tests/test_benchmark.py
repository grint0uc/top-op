from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest

from topop.core.benchmarks import cantilever, cantilever_params
from topop.core.optimize import optimize

REF = Path(__file__).resolve().parent / "data" / "cantilever_60x20x4.json"


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["auto", "amg"])
def test_cantilever_60x20x4_matches_reference_and_is_two_bar(solver):
    # auto: banded Cholesky (exact); amg: geometric MG-CG with the adaptive tolerance (1e-4 while
    # the design moves) -- both must land within 1 % of the SuperLU reference, the test allows 3 %
    nelx, nely, nelz = 60, 20, 4
    params = dataclasses.replace(cantilever_params(), solver=solver)
    t0 = time.perf_counter()
    res = optimize(cantilever(nelx, nely, nelz), params)
    wall = time.perf_counter() - t0
    c = res.history[-1].compliance

    if not REF.exists():  # first run generates the reference; commit the file
        REF.parent.mkdir(parents=True, exist_ok=True)
        ref = {
            "problem": f"cantilever({nelx}, {nely}, {nelz})",
            "params": "cantilever_params()",
            "compliance": c,
            "iterations": len(res.history),
            "status": res.status,
            "volume": res.history[-1].volume,
            "seconds_per_iteration_when_generated": wall / len(res.history),
        }
        REF.write_text(json.dumps(ref, indent=2) + "\n")
    ref = json.loads(REF.read_text())
    assert c == pytest.approx(ref["compliance"], rel=0.03)
    assert c == pytest.approx(ref["compliance"], rel=0.01)  # solver policy budget (PERF.md)
    assert abs(res.history[-1].volume - params.volfrac) < 1e-3

    # Two-chord truss: material concentrates in the top and bottom thirds. Measured over the full
    # span (top/middle 2.1, bottom/middle 2.1). Restricted to x > nelx/2 the ratios are only
    # 1.49 / 1.31: the top chord ends near x = 44 and runs diagonally down to the load at the
    # bottom edge of the free end, so that half also holds the diagonal and a thin top chord.
    bottom, middle, top = (res.rho[:, idx, :].mean() for idx in np.array_split(np.arange(nely), 3))
    assert top > 1.5 * middle and bottom > 1.5 * middle
    assert res.rho[:, 0, :].mean() > 0.8 and res.rho[: nelx * 2 // 3, -1, :].mean() > 0.8
