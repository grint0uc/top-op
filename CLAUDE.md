# top-op — conventions

3D topology optimization tool. Python core (numpy/scipy) + FastAPI server + Three.js (React/TS/Vite) frontend.
Full design: `docs/PLAN.md`. Read it before touching anything non-trivial.

## Layout
- `topop/core/` — pure numerics. **No imports from `topop.server` or any web lib.** Contract: `topop/core/problem.py`.
- `topop/server/` — FastAPI. Contract: `topop/server/schemas.py` (pydantic → OpenAPI → `web/src/api/types.gen.ts`).
- `web/` — frontend. Vanilla Three.js viewport class, React only for panels, zustand store.
- `tests/` — pytest. `web/e2e/` — Playwright.
- `examples/` — generated STLs (`examples/make_examples.py`), committed.

## Hard conventions (every module must agree)
- Element grid shape `(nx, ny, nz)`, arrays indexed `[ix, iy, iz]`, flat ids are **C-order** (`np.ravel_multi_index`).
- Node grid `(nx+1, ny+1, nz+1)`, same C-order flat ids. Node `(ix,iy,iz)` sits at `origin + h*(ix,iy,iz)`.
- Load/Support node ids are **full-grid** node ids. `fem.py` compresses to active nodes internally.
- DOF id = `3*node + axis` (axis 0,1,2 = x,y,z).
- Hex8 local node order (natural coords): `(-,-,-) (+,-,-) (+,+,-) (-,+,-) (-,-,+) (+,-,+) (+,+,+) (-,+,+)` — see `Grid.element_nodes()`.
- `active[ix,iy,iz]` bool: element exists. `passive` int8: 0 free, 1 forced solid, -1 forced void. Passive elements are assembled but excluded from the volume constraint and never updated.
- Transforms over the wire: 16 floats, **column-major** (three.js `Matrix4.toArray()`); server: `np.asarray(t).reshape(4,4).T` gives the row-major matrix.
- Primitives: box = unit cube centered at origin scaled by `size`; sphere radius `size[0]`; cylinder axis = local **Y**, radius `size[0]`, height `size[1]` (three.js `CylinderGeometry` convention). Then `transform` applied.
- Density frames over WebSocket (binary): `u32 it, u32 nx, u32 ny, u32 nz, u8[nx*ny*nz]` little-endian, C-order, value = round(rho*255). Inactive elements = 0.
- Selections (`schemas.Selection`): `faces` (GUI clicks), `facets` (coplanar groups from `/facets`), `normal` (direction + angle), `plane` (grid nodes near a plane), `box|sphere|cylinder`. All resolve server-side to full-grid node ids in `core/selection.py`. Agents (Claude) use the non-`faces` kinds.
- Agent interface: `topop run case.json` (case = `ProjectIn` JSON with `path` instead of `mesh_id`), `topop describe mesh.stl` (facet table), `topop mcp` (MCP server over the same API). Everything the GUI can do must be reachable through these.
- v0.2 params (`RunParams`/`ParamsSpec`): `optimizer` oc|mma, `symmetry` planes (densities mirrored each iteration), `stress_limit`+`stress_pnorm` (p-norm von Mises constraint, forces mma; p continues 8 → `stress_pnorm`, default 64; see docs/STRESS.md), `overhang` build direction (Langelaar AM filter, 45°). `Result.stress` / `IterationInfo.stress_max` carry von Mises. Facets carry `kind` plane|cylinder|other (+axis/radius) and `brep_face` for STEP input.
- Units: none enforced. Whatever the mesh is in. Compliance reported in those units.

## Commands
- `uv sync` — install. `uv run pytest -q` — core+server tests. `uv run pytest -q -m "not slow"` in CI loops.
- `cd web && npm ci && npm run build` — frontend into `topop/server/static/` (committed; rebuild and commit it whenever `web/` changes). `npm run dev` — Vite with proxy to :8000.
- `uv run topop serve` — serve built frontend + API on :8000. `uv run topop run case.json` — headless.
- `make test`, `make e2e`, `make build`, `make dev`.

## Style
- Python 3.12+, type hints, `ruff` (line length 100). Comments only where logic is non-obvious. No docstring essays.
- TS strict. No `any` without a comment. No new UI/chart libraries without asking.
- Tests are the spec: a function without a test is unfinished. Mark >10 s tests `@pytest.mark.slow`.
- Agents: own only the files named in your brief. Do not edit `problem.py` / `schemas.py` / `CLAUDE.md`; if the contract is wrong, say so in your report instead.
