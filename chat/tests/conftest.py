"""Shared fixtures: a synthetic dump, the leaderboard store built over it, and a context."""
import httpx
import pytest
from fastapi.testclient import TestClient

from chat import agent, budget, config, data, server, store
from chat.llm import FakeClient
from tests import make_serving


@pytest.fixture(scope="session")
def dump(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    models = tmp_path_factory.mktemp("models")
    path = make_serving.make(cache)
    config.CACHE_DIR = cache
    config.MODELS_DIR = models
    config.TMP_DIR = models / "tmp"
    return {"serving": path, "cache": cache, "models": models}


@pytest.fixture(scope="session")
def loaded(dump):
    assert data.pool.load(dump["serving"]), data.pool.error
    store_meta = store.attach(data.pool.connection(), data.pool.meta)
    return {"store_meta": store_meta, **dump}


@pytest.fixture
def ctx(loaded):
    # no upstream services in the tests: a client pointed at a dead port exercises the fallbacks
    return agent.Ctx(data.pool, httpx.Client(timeout=0.25, transport=httpx.HTTPTransport(retries=0)),
                     loaded["store_meta"])


@pytest.fixture
def ledger(dump):
    return budget.Ledger(path=dump["models"] / "budget-test.json")


@pytest.fixture
def client(loaded, dump, monkeypatch):
    monkeypatch.setattr(config, "TOKEN", "")
    monkeypatch.setattr(budget, "ledger", budget.Ledger(path=dump["models"] / "budget-server.json"))
    server.state.client = FakeClient(
        script=[{"tool_calls": [{"id": "c1", "name": "dataset_facts", "arguments": {}}]}],
        answer="The snapshot holds 26 publications.")
    server.state.store_meta = loaded["store_meta"]
    server.state.error = None
    return TestClient(server.app)


