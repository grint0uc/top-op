# Changelog

All notable changes to top-op. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may change the case-file
and API format). The release workflow uses the section of the tagged version as the release notes.
0.1.0 and 0.2.0 were not tagged releases; their links point at the last commit of each.

## [Unreleased]

## [0.3.0] - 2026-11-27

First public release.

### Added
- Strut generation (`topop struts`, MCP `generate_struts`, `POST /api/runs/{id}/struts`): turns a run into an
  explicit truss of round bars with spherical joints, either a minimum-volume layout LP over a ground structure
  (HiGHS, multiple load cases) or the medial axis of the SIMP solid; unioned with keep-in bodies, clipped to the
  design and verified by FE against the SIMP result. The layout LP retries with more candidate bars when the
  capped set is infeasible.
- GUI redesigned as a CAD-style workbench: ribbon with workflow tabs (Model, Boundary Conditions, Domain,
  Optimize, Results), feature tree, property pages, status bar, light and dark themes.
- top-op lite (`lite/`): a single-page in-browser version (STL import, face selection, keep boxes, SIMP in a
  Web Worker, STL export) for trying the method without installing anything.
- Release housekeeping: MIT license, package metadata, contributing guide, issue and PR templates, tag-driven
  release workflow.

### Changed
- Stress constraint is robust at default settings: per-step constraint target with a floor, automatic
  p-continuation from 8 to `stress_pnorm` (now the final exponent, default 64), move caps while the constraint
  is active, a hard volume row, continuation gated on damping.
- Performance: symmetric assembly map (15-17 % lower peak memory), bandwidth ordering by RCM or axis sweep,
  automatic switch from multigrid to the banded solver, calibrated memory and time estimates.

### Fixed
- Validation: supports that leave rigid-body modes free and loads lost to fixed DOFs are rejected; a singular
  stiffness raises instead of returning garbage; input models reject unknown fields and NaN; face ids, angles
  and the load-case index are bounded.
- Core: facets resolved on the raw mesh (non-uniform design scale no longer permutes them), load cases
  renumbered, memory guard checked before voxelizing, inside-out meshes repaired on load, versioned NPZ and
  STEP caches, filter clipping, BLAS threads pinned inside the assembler only.
- Server, CLI and MCP: bounded WebSocket queues, store refresh across processes, upload size cap, host and
  origin checks, queued-run limit, finished runs released from memory, atomic writes; MCP exports require
  absolute paths and never overwrite silently; meaningful CLI exit codes; `describe_mesh(project_id)` reports
  world-space geometry.
- CI stability fixes in the Playwright suite and the history regression test.

## [0.2.0] - 2026-10-08

### Added
- MMA optimizer (`optimizer: "mma"`), alongside OC.
- Symmetry planes: densities mirrored every iteration.
- Von Mises stress recovery and a p-norm aggregated stress constraint (`stress_limit`, `stress_pnorm`);
  `stress_max` per iteration, stress field in the VTI and NPZ exports and `GET /api/runs/{id}/stress`.
- Langelaar 45 degree overhang filter for additive manufacturing, all six build directions.
- STEP import (optional `step` extra): B-rep faces as exact facets with their kind, cached tessellation,
  example `bracket.step`.
- Facets: planar region growing, cylinder detection for holes and fillets (`kind`, `radius`, `axis`), sphere
  detection, `facet_faces` endpoint and MCP tool.
- Trim to CAD: intersect the result with the design mesh (`--trim`, `trim=true`).
- Frontend controls for optimizer, symmetry, stress and overhang; stress colormap; exact facet highlighting;
  design-mesh transform; STEP upload.
- CLI flags `--optimizer`, `--stress-limit`, `--overhang`, `--symmetry`, `--trim`; MCP `set_params` fields and
  `result_stress_summary`.
- GitHub Actions: CI on every push and pull request (lint, fast tests, static-bundle check, Playwright) and a
  weekly slow suite.

### Fixed
- CG breakdown on warm-started solves (Lanczos bound for the Chebyshev smoother, staged fallbacks).
- Overhang base plate is the first active layer, not padding.

## [0.1.0] - 2026-10-08

### Added
- Core: scan-line voxelizer, hex8 FEM with CSR pattern assembly, density filter, Heaviside projection, OC
  optimizer, cantilever benchmark.
- Solver performance: geometric-multigrid PCG, banded Cholesky for small problems, adaptive CG tolerance,
  in-place assembly, memory and time estimates.
- Selections (faces, facets, normal, plane, box, sphere, cylinder) resolved to grid nodes; STL, VTI and NPZ
  export; headless PNG renderer.
- FastAPI server: project, mesh and run store, threaded run manager with WebSocket density streaming, exports
  and previews.
- Three.js frontend: picking, painting, primitives with a gizmo, query selections, density view, project
  import and export; Playwright specs against a mock and the real server.
- Agent interface: headless `Session`, `topop run` and `topop describe`, MCP server, example case files.

[0.3.0]: https://github.com/grint0uc/top-op/releases/tag/v0.3.0
[0.2.0]: https://github.com/grint0uc/top-op/tree/45998e7
[0.1.0]: https://github.com/grint0uc/top-op/tree/eff2cfa
