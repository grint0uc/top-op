.PHONY: install test test-fast lint build dev e2e types

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

# Vite writes to ../topop/server/static (see web/vite.config.ts)
build:
	cd web && npm run build

dev:
	trap 'kill 0' INT TERM; \
	uv run uvicorn topop.server.app:app --reload --host 127.0.0.1 --port 8000 & \
	(cd web && npm run dev) & \
	wait

e2e:
	cd web && npx playwright test

# regenerate web/src/api/types.gen.ts from the live FastAPI schema
types:
	uv run python -m topop.server.openapi > openapi.json
	cd web && npm run types
