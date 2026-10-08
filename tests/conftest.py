from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from topop.server.app import app


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
