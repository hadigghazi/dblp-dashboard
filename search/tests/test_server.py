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


def test_words_only_search_is_available_for_comparison(client):
    """`dense=false` exists so the paraphrase test can run the same query with and without the
    embeddings; without it there is no way to attribute a difference to them."""
    both = client.get("/search/papers", params={"q": "graph learning", "top": 5}).json()
    words = client.get("/search/papers", params={"q": "graph learning", "top": 5, "dense": "false"}).json()
    assert words["dense_available"] is False
    assert "as asked" in (words.get("dense_error") or "")
    assert both.get("dense_candidates", 0) >= words.get("dense_candidates", 0)
    assert all("dense" not in (r.get("sources") or []) for r in words["results"])


def test_the_embeddings_can_be_asked_on_their_own(client):
    """Fusing a good ranking with one that found nothing is not the same as asking the good one, and
    a paraphrase query is where that difference shows: BM25 finds nothing, so RRF spends half its
    slots on noise. Measuring it needs this switch."""
    dense_only = client.get("/search/papers",
                            params={"q": "graph learning", "top": 5, "sparse": "false"}).json()
    assert dense_only["sparse_used"] is False
    assert dense_only["sparse_candidates"] == 0
    assert all(r["sources"] == ["dense"] for r in dense_only["results"]), \
        "no word match may leak in, not even the exact-word top-up"


def test_a_cached_answer_belongs_to_the_ranker_that_made_it(client, monkeypatch):
    """Changing a fusion threshold and re-measuring scored the OLD ranker for twenty seconds, because
    the cache key described the query and not the ranking."""
    from search import config
    first = client.get("/search/papers", params={"q": "graph learning", "top": 5}).json()
    assert client.get("/search/papers", params={"q": "graph learning", "top": 5}).json()["cached"]
    monkeypatch.setattr(config, "COVERAGE_FLOOR", config.COVERAGE_FLOOR + 0.05)
    again = client.get("/search/papers", params={"q": "graph learning", "top": 5}).json()
    assert not again["cached"], "a different ranker must not be served from the old ranker's cache"
    assert first["query"] == again["query"]


def test_find_reports_where_a_record_sits_in_each_ranking(client):
    """Whether the right paper is deep in the list or absent from it decides between a reranker and
    a new embedding model - one is free, the other is a day of re-embedding."""
    first = client.get("/search/papers", params={"q": "graph learning", "top": 3}).json()
    key = first["results"][0]["key"]
    located = client.get("/search/papers",
                         params={"q": "graph learning", "top": 3, "find": key}).json()["find"]
    assert located["key"] == key and located["known"] is True
    assert located["fused_rank"] == 1
    assert located["dense_candidates"] >= 1

    missing = client.get("/search/papers",
                         params={"q": "graph learning", "top": 3, "find": "conf/nope/nothing"}).json()
    assert missing["find"]["known"] is False and missing["find"]["fused_rank"] is None


def test_a_failing_query_encoder_degrades_to_words(client, monkeypatch):
    """With an API-backed model the query is a network call: it can fail while the index is fine."""
    from search import server

    class Broken:
        def encode_query(self, text):
            raise RuntimeError("the embedding endpoint is unreachable")

    monkeypatch.setattr(server, "encoder", Broken())
    out = client.get("/search/papers", params={"q": "graph learning", "top": 5, "refresh": "true"}).json()
    assert out["results"], "words must still answer"
    assert out["dense_available"] is False
    assert "unreachable" in out["dense_error"]


def test_status_names_the_embedding_model_in_use(client):
    body = client.get("/search/status").json()
    assert body["embeddings"]["model"] and body["embeddings"]["dimensions"]
    assert body["embeddings"]["vectors"].endswith(".float16")


def test_the_warm_up_can_actually_reach_the_index(client):
    """It could not, for a while, and nothing said so out loud: the warm-up thread attached the dump
    but not the search store, so reading how far the index got raised every time. A failed warm-up is
    only logged, so the cache silently stayed cold and the first real search after every deploy paid
    the whole vector load while somebody waited."""
    from search import server as SV

    assert SV.state.fingerprint, "the fixture should have a store attached"
    assert SV.state.warm_now() is True, "the warm-up must reach the index rather than raise"


def test_the_warm_up_is_a_no_op_without_a_store():
    """No store yet is a normal state on a fresh machine, not a failure."""
    from search import server as SV

    cold = SV.State()
    assert cold.warm_now() is False
