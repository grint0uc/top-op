# Contributing to top-op

Bug reports, example parts that break things, and pull requests are all welcome. For anything larger
than a fix, open an issue first so we can agree on the approach. The design is in
[docs/PLAN.md](docs/PLAN.md); the conventions every module relies on are in [CLAUDE.md](CLAUDE.md).

## Setup

You need [uv](https://docs.astral.sh/uv/) and, for frontend work, Node 22.

```
git clone https://github.com/grint0uc/top-op
cd top-op
make install          # uv sync --all-groups + npm ci in web/
uv sync --extra step  # optional: STEP import, needed for the STEP tests
make dev              # API with reload on :8000 + Vite dev server with a proxy to it
```

The end-to-end tests need a Playwright Chromium: `cd web && npx playwright install chromium` once.

## Tests and lint

```
make test-fast   # uv run pytest -q -m "not slow"   (the usual loop)
make test        # everything, including tests marked slow (> 10 s)
make lint        # ruff check, ruff format --check, web typecheck
make check       # lint + test-fast: run this before every commit
make e2e         # Playwright: UI on the mock API, and UI + real server on :8765
```

CI runs `make lint`, the fast tests, a check that the committed frontend bundle matches a fresh build,
and both Playwright projects. The slow and perf tests run weekly.

Rules of thumb:

- **Tests are the spec.** A new function or behavior comes with a test. Mark tests that take more
  than about 10 s with `@pytest.mark.slow`.
- Python 3.12+, type hints, ruff with line length 100. Comment only where the logic is not obvious.
- TypeScript is strict; no `any` without a comment saying why. Please ask before adding a UI or chart
  library.
- The built frontend in `topop/server/static/` is committed so that `pip install` gives a working GUI.
  If you change anything in `web/`, run `make build` and commit the result, or CI's static check fails.

## How the code is split

- `topop/core/` is pure numerics (numpy/scipy). It must not import from `topop/server/` or any web
  library. Its contract is `topop/core/problem.py`.
- `topop/server/` is the FastAPI app. Its contract is `topop/server/schemas.py`, which generates the
  OpenAPI schema and from it `web/src/api/types.gen.ts` (`make types`).
- `web/` is the frontend: a plain Three.js viewport class, React only for the panels, a zustand store.
- Everything the GUI can do must also be possible from `topop run`, `topop describe` and the MCP server
  (`topop mcp`), so scripts and Claude sessions are never second-class.

## Changing a contract

`problem.py` and `schemas.py` are shared by every part of the code base, and much of top-op was built
by several people (and AI agents) working in parallel on separate files. The convention that keeps
that working: **change only the files your task is about**, and treat the two contract files as
owned by the maintainer. If a contract is wrong or missing something, say so in the issue or PR
description and propose the change on its own (contract change, `make types`, docs) rather than
folding it into a feature. The grid ordering, DOF numbering, transform layout and wire formats in
`CLAUDE.md` are hard conventions; code that disagrees with them is a bug.

## Commits and pull requests

- Keep commits focused. The subject line says what changed in plain words, capitalized, no trailing
  period, optionally with an area prefix: `Solver: Lanczos bound for the Chebyshev smoother`,
  `CI: cross-machine tolerance for the history regression`. Add a body when the why is not obvious.
- Add a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for user-visible changes.
- Fill in the PR template; make sure `make check` passes.

## Releases (maintainer)

Bump `__version__` in `topop/__init__.py`, rename the "Unreleased" section of `CHANGELOG.md` to the
version and date, commit, then tag and push: `git tag v0.3.1 && git push origin v0.3.1`. The release
workflow checks that the tag matches the version, builds the wheel and sdist, and publishes a GitHub
release with the changelog section as its notes.

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE).
