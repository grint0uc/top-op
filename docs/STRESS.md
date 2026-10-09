# Stress constraint

**Constraint.** Per active voxel and load case the relaxed von Mises stress `s = rho^q sigma_vm(u; E0)`,
q = 0.5 (qp-relaxation, Bruggi 2008, Le et al. 2010): the solid-material stress at the element center
times sqrt(density), so void cells read 0 and solid cells their real stress (`Result.stress`,
`IterationInfo.stress_max` and the constraint all use it). The max is aggregated by the p-norm
`sigma_PN = (sum s^p)^(1/p)` over cells and load cases with Le's adaptive normalization
`c_k = a max(s) / sigma_PN + (1 - a) c_(k-1)`, so that `c_k sigma_PN` tracks the max. The row is
`g = c_k sigma_PN / stress_limit - 1 <= 0` (`IterationInfo.constraint`), with adjoint sensitivities
through the filter / overhang / projection chain. A stress limit forces MMA (rows: volume, stress).

**What to set.** `stress_limit` in the units of E (stress = E x strain: E in MPa gives MPa, whatever
the length unit), chosen relative to the max stress of an unconstrained run. `stress_pnorm` is
optional: the start of the p continuation when below 8, its cap when above 64. `move` only caps the
step. 100 iterations suffice when the limit is reachable at the volume. Nothing else is needed.

**Automatic conditioning** (`StressControl`, `topop/core/optimize.py`; no-limit runs are unchanged):

| measure | setting |
|---|---|
| hard volume | MMA artificial-variable cost c = 1e6 on the volume row, 1e4 on the stress row |
| step target | each MMA step asks for `g_new <= 0.95 g` while g > 0, not `g_new <= 0` |
| p continuation | `min(stress_pnorm, 8)`, doubled every 15 iterations or when change < 0.02, up to `max(stress_pnorm, 64)`; c_k and the asymptotes restart at each doubling |
| normalization, moves | a = 0.3; move `min(move, 0.1)` for 10 iterations and while g <= -0.1, else `min(move, 0.05)` |
| MMA, damping | asymptotes 0.2 / 1.1 / 0.6; dx reversing on > 40 % of the moving variables halves the move for 5 iterations (noted once in `Result.message`) |

Why: the grey start's relaxed stress ~ volfrac^(q - penal) is 7-19x the limit. A step that must reach
g <= 0 cannot, so the subproblem is infeasible, the stress multiplier sits at c and the stress alone
drives the design: with equal costs MMA buys stress with volume (+17 % at vf 0.3), with a hard volume
row it stays grey. The 0.95 target keeps the subproblem feasible, so compliance still shapes the layout.

**Measured** (`test_stress.py::test_l_bracket_stress_limit_at_default_parameters`: the user sets only
volfrac, rmin = 1.5, stress_limit = 0.7 x the unconstrained peak; shared 4-core VM):

| l_bracket | volfrac | iterations | wall | max stress / limit | volume |
|---|---|---|---|---|---|
| 40 x 40 x 4 | 0.5 | 100 | 40 s | 1.009 | 0.4999 |
| 40 x 40 x 4 | 0.3 | 200 | 100 s | 1.135 (near-infeasible, see caveats) | 0.3000 |
| 60 x 60 x 4 | 0.4 | 100 | 77 s | 1.013 | 0.4000 |

Ablation on l_bracket(40), one measure off (max stress / limit, volume, compliance; vf 0.5 at 120 its,
vf 0.3 at 150; all on: 1.003, 0.500, 45.2 and 1.143, 0.300, 89.0). Volume row c = 1e4: vf 0.3 ends at
volume **0.329** (1.002). No step target: 1.023, compliance **56.1** and **1.605**, 157. p capped at
16: 1.024, 46.7 and **1.264**. No move caps or a = 0.5: same end points, but one-iteration stress jumps
of +0.93 / +0.51 x limit at vf 0.3 (all on: +0.22). Move as MMA box x +- move: 1.064, 55.9 and 1.722.
Neutral: default asymptotes, no damping, objective scaling, clamping after the step; no asymptote
restart costs 3 % compliance at vf 0.5. Before this change (200 its): 1.152 and 1.048 at volume
**0.349**. GCMMA-style inner iterations were not needed: no case oscillates late.

**Caveats.**
- Singular points: voxel stresses at re-entrant corners, clamp edges and small load patches grow with
  refinement (the optimizer greys corner cells). Set limits relative to an unconstrained run, with margin.
- Feasibility is not guaranteed: if the volume cannot carry the load at the limit, the run ends fully
  stressed above it, volume exact (`constraint` > 0, `max_iter`). l_bracket(40) vf 0.3: the unconstrained
  inner flange already carries 1.3x the limit along the whole arm; the limit is met at ~0.33 volume.
- One-iteration lag: `IterationInfo.stress_max` / `constraint` belong to the design evaluated at the
  start of that iteration; `Result.stress` is recomputed for the returned (updated) design.
- `c_k sigma_PN` estimates the max (the final max can differ from `constraint` by a few percent).
  No local (per-cell) constraints or per-region limits.
