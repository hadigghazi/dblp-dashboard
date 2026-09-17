"""The HTTP front the dashboard talks to. Runs after test_pipeline has trained a model into the
same temporary MODELS_DIR (pytest runs files alphabetically; both share the session env)."""
import pytest

from tests import test_pipeline as tp  # noqa: F401  (sets env, builds the serving db)
from ml import cli, data, model as M


@pytest.fixture(scope="module")
def client():
    # make sure a model exists, whatever order the files ran in
    try:
        M.load()
    except FileNotFoundError:
        con, _ = data.connect()
        try:
            assert cli.train(con, tp.Args()) == 0
        finally:
            con.close()
    from fastapi.testclient import TestClient
    from ml.server import app
    with TestClient(app) as c:
        yield c


def test_status_is_a_model_card(client):
    s = client.get("/ml/status").json()
    assert s["available"] is True
    assert s["dump"]["fingerprint"] == "testfp0001"
    assert 0 <= s["assignment"]["top1_accuracy"] <= 1
    assert s["clustering"]["b3_f1"] is not None and s["clustering_bin_like"]["b3_f1"] is not None
    assert s["feature_importance"] and "feature" in s["feature_importance"][0]
    assert set(s["thresholds"]) >= {"cluster_bin", "assign"}


def test_largest_bins_listing(client):
    b = client.get("/ml/bins", params={"top": 5}).json()["bins"]
    assert [x["name"] for x in b] == ["Bin One", "Bin Two"]
    assert b[0]["papers"] == 10 and b[0]["numbered_pages"] == 2
    assert client.get("/ml/bins", params={"q": "Two"}).json()["bins"][0]["name"] == "Bin Two"


def test_split_is_computed_then_cached(client):
    first = client.get("/ml/bin", params={"key": "homepages/bin/0"}).json()
    assert first["bin"]["name"] == "Bin One" and first["cached"] is False
    assert first["papers"] == 10 and first["clusters"]
    assert first["model"]["test_metrics"]["clustering"]["b3_f1"] is not None
    again = client.get("/ml/bin", params={"key": "homepages/bin/0"}).json()
    assert again["cached"] is True and again["summary"] == first["summary"]


def test_split_errors_are_clean(client):
    assert client.get("/ml/bin", params={"key": "homepages/00/1"}).status_code == 404   # not a bin
    assert client.get("/ml/bin", params={"key": "homepages/nope"}).status_code == 404


def test_on_demand_cap_is_enforced(client):
    from ml import server
    out = client.get("/ml/bin", params={"key": "homepages/bin/1", "max_papers": 600}).json()
    assert out["papers"] <= server.MAX_ON_DEMAND
