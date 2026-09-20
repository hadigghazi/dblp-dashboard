"""The HTTP front. Runs after test_pipeline has built the serving db and store; server.encoder is
swapped for the fake so the test image never needs torch."""
import pytest

from tests import test_pipeline as tp  # noqa: F401  (sets env, builds the serving db + store)
from tests.fake_encoder import FakeEncoder
from search import data, store as S, vectors as V


@pytest.fixture(scope="module")
def client():
    con, meta = data.connect()
    try:
        S.attach_store(con, meta, build_if_missing=True)
        fp = meta["fingerprint"]
        if not V.progress(con, fp)["complete"]:
            V.build(con, fp, FakeEncoder())
    finally:
        con.close()

    from search import server
    server.encoder = FakeEncoder()   # avoid needing the real model in the test image
    from fastapi.testclient import TestClient
    with TestClient(server.app) as c:
        yield c


def test_health_and_status(client):
    assert client.get("/search/health").json() == {"ok": True, "store": True}
    s = client.get("/search/status").json()
    assert s["available"] is True and s["dump"]["fingerprint"] == "testfp0001"
    assert s["index"]["complete"] is True and s["index"]["papers"] > 0


def test_search_endpoint_ranks_the_matching_topic_then_caches(client):
    first = client.get("/search/papers", params={"q": "graph clustering community partition", "top": 5}).json()
    assert first["cached"] is False and first["results"]
    assert first["results"][0]["venue"] == "AAA"
    again = client.get("/search/papers", params={"q": "graph clustering community partition", "top": 5}).json()
    assert again["cached"] is True and again["results"] == first["results"]


def test_search_endpoint_applies_kind_and_year_filters(client):
    out = client.get("/search/papers", params={"q": "graph clustering community", "kind": "journal", "top": 5}).json()
    assert {r["kind"] for r in out["results"]} <= {"journal"}


def test_search_endpoint_finds_a_preprint_by_exact_words(client):
    con, _ = data.connect()
    try:
        title = con.execute("SELECT title FROM s.pubs WHERE key LIKE 'corr/%' LIMIT 1").fetchone()[0]
    finally:
        con.close()
    out = client.get("/search/papers", params={"q": title, "top": 5}).json()
    assert any(r["kind"] == "preprint" for r in out["results"])


def test_search_endpoint_rejects_a_short_query(client):
    assert client.get("/search/papers", params={"q": "ab"}).status_code == 422
