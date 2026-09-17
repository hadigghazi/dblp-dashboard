import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Settings are read at import time, so point them at a fresh fixture before importing the app.
_tmp = Path(tempfile.mkdtemp(prefix="dblp-api-test-"))
os.environ.update({
    "DATA_DIR": str(_tmp / "data"),
    "CACHE_DIR": str(_tmp / "cache"),
    "WATCH_SECONDS": "100000",
    "DUCKDB_MEMORY": "1GB",
    "DUCKDB_THREADS": "2",
})

from tests.make_fixture import make  # noqa: E402

make(_tmp / "data")


@pytest.fixture(scope="session")
def serving():
    from app.serving import serving as s
    s.ensure()
    assert s.status["state"] == "ready", s.status
    return s


@pytest.fixture(scope="session")
def client(serving):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def data_dir():
    return _tmp / "data"
