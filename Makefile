.PHONY: install test test-fast lint check static-check ci build dev e2e e2e-mock e2e-real types

install:
	uv sync --all-groups
	cd web && npm ci

test:
	uv run pytest -q

test-fast:
	uv run pytest -q -m "not slow"

lint:
	uv run ruff check .
	uv run ruff format --check .
	cd web && npm run typecheck

# pre-commit gate; `lint` already ends with the web typecheck (`cd web && npm run typecheck`)
check: lint test-fast

# CI gate: the committed topop/server/static must be what a fresh build produces (Vite names assets by content hash, so
# identical sources give identical output; verified by building twice). Builds into a temp dir, leaves the tree alone.
static-check:
	tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	(cd web && npx vite build --outDir "$$tmp" --emptyOutDir --logLevel warn) && \
	{ diff -rq "$$tmp" topop/server/static || { echo 'run `make build` and commit topop/server/static'; exit 1; }; }

# What .github/workflows/ci.yml runs after the toolchain setup (needs `make install` and the Playwright browser, which
# `playwright install` provides in CI; never run here): lint (ruff + web typecheck), fast tests on 2 threads,
# committed-frontend check, both Playwright projects.
ci:
	$(MAKE) lint
	TOPOP_THREADS=2 $(MAKE) test-fast
	$(MAKE) static-check
	$(MAKE) e2e

# Vite writes to ../topop/server/static (see web/vite.config.ts); that directory is committed
build:
	cd web && npm run build

dev:
	trap 'kill 0' INT TERM; \
	uv run uvicorn topop.server.app:app --reload --host 127.0.0.1 --port 8000 & \
	(cd web && npm run dev) & \
	wait

# Playwright, two projects (web/playwright.config.ts): `chromium` = UI against the mock backend (web/mock),
# `real` = UI + `uv run topop serve` on :8765 with a fresh data dir. Never runs `playwright install`.
e2e:
	cd web && npx playwright test

e2e-mock:
	cd web && npx playwright test --project=chromium

e2e-real:
	cd web && npx playwright test --project=real

# regenerate web/src/api/types.gen.ts from the live FastAPI schema
types:
	uv run python -m topop.server.openapi > openapi.json
	cd web && npm run types
