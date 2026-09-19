"""The HTTP front the dashboard talks to. Runs after test_pipeline and test_links have trained their
models into the same temporary MODELS_DIR (pytest runs files alphabetically; all share the session env)."""
import pytest

from tests import test_pipeline as tp  # noqa: F401  (sets env, builds the serving db)
from tests import test_links as tl, test_venues as tv
from ml import cli, data, model as M
from ml.links import cli as lcli, graph as G, model as LM
from ml.venues import cli as vcli, model as VM, store as VS


@pytest.fixture(scope="module")
def client():
    # make sure every model exists, whatever order the files ran in
    con, meta = data.connect()
    try:
        try:
            M.load()
        except FileNotFoundError:
            assert cli.train(con, tp.Args()) == 0
        try:
            LM.load()
        except FileNotFoundError:
            G.attach_store(con, meta, build_if_missing=True)
            assert lcli.train(con, meta, tl.Args()) == 0
        try:
            VM.load()
        except FileNotFoundError:
            VS.attach_store(con, meta, build_if_missing=True)
            assert vcli.train(con, meta, tv.Args()) == 0
    finally:
        con.close()
    from fastapi.testclient import TestClient
    from ml.server import app
    with TestClient(app) as c:
        yield c


def test_health_reports_every_model(client):
    h = client.get("/ml/health").json()
    assert h == {"ok": True, "model": True, "links": True, "venues": True}


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


def test_links_status_is_a_model_card(client):
    s = client.get("/ml/links/status").json()
    assert s["available"] is True and s["dump"]["fingerprint"] == "testfp0001"
    assert 0 <= s["test"]["pooled"]["roc_auc"] <= 1 and s["test"]["ranking"]["mrr"] is not None
    assert set(s["baselines"]) == {"cn", "jaccard", "aa", "ra", "pa"}
    assert s["calibration"] and s["feature_importance"]
    assert s["dataset"]["test"]["new_links_origin"]["new_links"] > 0


def test_links_are_suggested_then_cached(client):
    con, _ = data.connect()
    try:
        keys = [r[0] for r in con.execute("""
            SELECT key FROM s.persons WHERE page_kind = 'regular' ORDER BY key LIMIT 40""").fetchall()]
    finally:
        con.close()
    first = next((o for o in (client.get("/ml/links", params={"key": k, "top": 5}).json() for k in keys)
                  if o.get("suggestions")), None)
    assert first is not None and first["cached"] is False
    assert len(first["suggestions"]) <= 5 and first["suggestions"][0]["via"]
    again = client.get("/ml/links", params={"key": first["author"]["key"], "top": 5}).json()
    assert again["cached"] is True and again["suggestions"] == first["suggestions"]


def test_links_errors_are_clean(client):
    assert client.get("/ml/links", params={"key": "homepages/bin/0"}).status_code == 404   # a bin
    assert client.get("/ml/links", params={"key": "homepages/nope"}).status_code == 404


def test_venues_status_is_a_model_card(client):
    s = client.get("/ml/venues/status").json()
    assert s["available"] is True and s["dump"]["fingerprint"] == "testfp0001"
    r = s["test"]["rankers"]
    assert set(r) == {"model", "naive_bayes", "centroid", "history", "popularity"}
    assert 0 <= r["model"]["acc@1"] <= 1 and s["test"]["covered"]["share"] > 0
    assert s["calibration"] and s["feature_importance"] and s["serving_statistics"]["series"] > 0


def test_venues_are_suggested_for_a_title_then_cached(client):
    first = client.get("/ml/venues", params={"title": "Spectral clustering of graph embeddings", "top": 5}).json()
    assert first["cached"] is False and first["suggestions"][0]["sid"] == "conf/bbb", first["suggestions"]
    assert first["related"] and first["suggestions"][0]["came_true"] is not None
    again = client.get("/ml/venues", params={"title": "spectral clustering of graph embeddings ", "top": 5}).json()
    assert again["cached"] is True and again["suggestions"] == first["suggestions"]
    with_authors = client.get("/ml/venues", params={"title": "Spectral clustering of graph embeddings",
                                                    "authors": "homepages/link/1/0,homepages/nope", "top": 5}).json()
    assert [a["key"] for a in with_authors["query"]["authors"]] == ["homepages/link/1/0"]


def test_venues_for_a_paper(client):
    con, _ = data.connect()
    try:
        key = con.execute("SELECT key FROM s.pubs WHERE key LIKE 'journals/ccc/l%' AND year >= 2015 "
                          "ORDER BY key LIMIT 1").fetchone()[0]
    finally:
        con.close()
    out = client.get("/ml/venues/paper", params={"key": key}).json()
    assert out["actual"]["sid"] == "journals/ccc" and out["actual"]["rank"] is not None
    assert client.get("/ml/venues/paper", params={"key": "conf/nope/x"}).status_code == 404
    assert client.get("/ml/venues", params={"title": "ab"}).status_code == 422


def test_a_retrain_into_the_same_directory_is_noticed(client):
    """A retrain on the same dump rewrites the same model directory; the watcher must still see it,
    and cached answers from the previous model must not be served for the new one."""
    import os
    import time
    from ml import server
    st = server.links
    assert st.available and not st.newer_model_exists()
    before = st.cache_path("homepages/link/0/0", 5)
    metrics = st.model_dir / "metrics.json"
    later = time.time() + 10
    os.utime(metrics, (later, later))
    assert st.newer_model_exists()
    st.load()
    assert not st.newer_model_exists()
    assert st.cache_path("homepages/link/0/0", 5) != before
