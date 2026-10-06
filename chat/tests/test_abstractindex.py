"""
Dewey's own abstract index, from a two-file stand-in for OpenAlex's snapshot: the fetch keeps only
the works dblp links to that have an abstract, the build files each under its dblp record, and the
search ranks with BM25 over the whole index.
"""
import json

import duckdb
import httpx
import pytest

from chat import abstractindex as AI, agent, config, content, data, dblpqa_rag as RAG, paperids

LONG = {
    "10.1109/7": "Graph neural networks forecast traffic flow on road networks from roadside sensor data, "
                 "and the forecasts stay accurate an hour ahead on two large city road networks.",
    "10.1109/8": "Graph embeddings make the retrieval of similar items fast and accurate, which we show on "
                 "product search and on citation recommendation with millions of items in each.",
    "10.1109/6": "We train graph learning models on billions of edges with a new partitioning scheme that "
                 "keeps each machine's memory small and the communication between machines low.",
    "10.5555/elsewhere": "A paper on a subject dblp does not index at all, with an abstract long enough to be "
                         "kept if only dblp linked to its DOI, which it does not.",
}


def inverted(text):
    index = {}
    for i, word in enumerate(text.split()):
        index.setdefault(word, []).append(i)
    return json.dumps(index)


def write(path, rows, extra=False):
    con = duckdb.connect()
    con.execute("CREATE TABLE w (id VARCHAR, doi VARCHAR, abstract_inverted_index VARCHAR, title VARCHAR"
                + (", cited_by_count INTEGER" if extra else "") + ")")
    con.executemany(f"INSERT INTO w VALUES (?, ?, ?, ?{', ?' if extra else ''})", rows)
    con.execute(f"COPY w TO '{path}' (FORMAT parquet)")
    con.close()
    return (str(path), len(rows), path.stat().st_size)


@pytest.fixture
def snapshot(loaded, tmp_path, monkeypatch):
    """A connection with the paper ids attached, models in a folder of their own, and two snapshot files."""
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    con, meta = data.connect(loaded["serving"])
    paperids.build(con, meta)
    assert paperids.attach(con, meta)
    a = write(tmp_path / "works-a.parquet", [
        ("W7", "https://doi.org/10.1109/7", inverted(LONG["10.1109/7"]), "OPENALEX'S OWN TITLE"),
        ("W8", "https://doi.org/10.1109/8", inverted(LONG["10.1109/8"]), "Graph embeddings for retrieval"),
        ("W99", "https://doi.org/10.5555/elsewhere", inverted(LONG["10.5555/elsewhere"]), "Not in dblp"),
        ("W9", "https://doi.org/10.1109/9", None, "Graph transformers in practice"),
        ("W1", "https://doi.org/10.1109/1", inverted("No abstract available."), "Cloud systems for analysis"),
        ("W0", None, inverted(LONG["10.1109/7"]), "A work with no DOI"),
    ])
    # a later file with a column the first does not have: files are read by column name
    b = write(tmp_path / "works-b.parquet", [
        ("W6", "https://doi.org/10.1109/6", inverted(LONG["10.1109/6"]), "Graph learning at scale", 12),
    ], extra=True)
    AI.forget()
    yield con, [a, b]
    con.close()
    AI.forget()


def test_the_fetch_keeps_only_dblps_works_with_an_abstract_and_resumes(snapshot):
    con, files = snapshot
    lines = []
    got = AI.fetch(con, files=files, date="2099-01-01", batch=1, out=lines.append)
    assert got["complete"] and got["batches_done"] == 2
    kept = dict(con.execute(f"SELECT doi, openalex FROM read_parquet('{AI.parts_dir('2099-01-01')}/*.parquet')").fetchall())
    # not in dblp, no abstract, no DOI: dropped; the placeholder is kept here and dropped by the build
    assert kept == {"10.1109/7": "W7", "10.1109/8": "W8", "10.1109/1": "W1", "10.1109/6": "W6"}
    assert any(line.startswith("  plan:") for line in lines)
    # a second fetch finds every batch written and fetches nothing
    lines.clear()
    assert AI.fetch(con, files=files, date="2099-01-01", batch=1, out=lines.append)["works_kept"] == 4
    assert not any(line.startswith("  batch ") for line in lines)


def test_the_build_files_abstracts_under_dblp_keys_and_bm25_finds_them(snapshot):
    con, files = snapshot
    AI.fetch(con, files=files, date="2099-01-01", batch=1, out=lambda *_: None)
    meta = AI.build(con, out=lambda *_: None)
    assert meta["documents"] == 3 and meta["short_dropped"] == 1 and meta["snapshot"] == "2099-01-01"
    assert AI.info()["searchable"] == 3
    hits = AI.search("How are traffic flows on road networks forecast from sensor data?")
    assert hits[0]["key"] == "journals/bbb/p7"
    assert hits[0]["title"] == "Graph neural networks for traffic"       # dblp's title, not OpenAlex's
    assert hits[0]["abstract"].startswith("Graph neural networks forecast traffic")
    assert hits[0]["doi"] == "10.1109/7"
    assert [h["key"] for h in AI.search("partitioning scheme billions of edges")][:1] == ["conf/aaa/p6"]
    found = AI.lookup(["journals/bbb/p8", "conf/aaa/p1", "no/such/key"])
    assert set(found) == {"journals/bbb/p8"}                              # p1's placeholder was dropped
    assert "citation recommendation" in found["journals/bbb/p8"]["abstract"]


def test_an_incomplete_fetch_is_not_built_unless_asked(snapshot):
    con, files = snapshot
    AI.fetch(con, files=files, date="2099-01-01", batch=1, limit=1, out=lambda *_: None)
    with pytest.raises(RuntimeError, match="incomplete"):
        AI.build(con, out=lambda *_: None)
    assert AI.build(con, partial=True, out=lambda *_: None)["documents"] == 2


def test_without_an_index_the_search_is_empty(snapshot, monkeypatch):
    assert AI.search("graph neural networks") == [] and AI.lookup(["journals/bbb/p7"]) == {}
    assert AI.info() is None
    monkeypatch.setattr(AI, "tantivy", None)
    assert AI.opened() == (None, None)


def test_abstracts_lose_markup_and_a_leading_label():
    assert AI.clean("<jats:p>Abstract: We study graphs.</jats:p>") == "We study graphs."
    assert AI.clean("ABSTRACT We study graphs.") == "We study graphs."
    assert AI.clean("Abstraction layers are studied.") == "Abstraction layers are studied."


@pytest.fixture
def built(snapshot):
    con, files = snapshot
    AI.fetch(con, files=files, date="2099-01-01", batch=1, out=lambda *_: None)
    AI.build(con, out=lambda *_: None)
    return snapshot


QUESTION = "How are traffic flows on road networks forecast from sensor data?"


def test_v3_searches_the_index_and_reads_dblps_own_hits_from_it(built, loaded, monkeypatch):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "searchapi":
            return httpx.Response(200, json={"results": [{"key": "conf/aaa/p6", "title": "Graph learning at scale"},
                                                         {"key": "journals/bbb/p8", "title": "Graph embeddings for retrieval"}]})
        return httpx.Response(500)

    monkeypatch.setattr(config, "SEARCH_URL", "http://searchapi")
    monkeypatch.setattr(config, "CONTENT_RULE_VERSION", 2)
    monkeypatch.setattr(config, "CONTENT_LOCAL_INDEX", False)
    assert content.wanted_sources() == ["dblp-search", "openalex-search", "openalex-semantic"], "v1 and v2: no index"
    monkeypatch.setattr(config, "CONTENT_LOCAL_INDEX", True)
    monkeypatch.setattr(config, "CONTENT_SEARCH", "live")
    assert content.wanted_sources() == ["dewey-index", "dblp-search", "openalex-search", "openalex-semantic"]
    monkeypatch.setattr(config, "CONTENT_SEARCH", "local")
    monkeypatch.setattr(config, "CONTENT_FALLBACK", False)
    ctx = agent.Ctx(data.pool, httpx.Client(transport=httpx.MockTransport(handler)), loaded["store_meta"])
    out = content.search_abstracts(ctx, question=QUESTION)
    assert out["meta"]["sources"] == {"dewey-index": "ok", "dblp-search": "ok"} and out["meta"]["index"] == "2099-01-01"
    assert out["rows"][0]["key"] == "journals/bbb/p7" and out["rows"][0]["found_by"] == "dewey-index"
    rows = {r["key"]: r for r in out["rows"]}
    # dblp's title search found these two; their abstracts come from the index, not from OpenAlex
    assert rows["conf/aaa/p6"]["abstract"].startswith("We train graph learning")
    assert rows["journals/bbb/p8"]["found_by"] == "dblp-search" and "citation" in rows["journals/bbb/p8"]["abstract"]
    assert not [c for c in calls if "searchapi" not in c], "local only: no outside service is asked"
    assert out["note"].startswith("Abstracts come from Dewey's own copy")
    assert "from Dewey's abstract index and dblp's title search" in out["summary"]


def test_the_index_is_a_ranker_of_its_own_beside_the_frozen_pool(built, tmp_path, monkeypatch):
    rows = [{"id": "q1", "question": QUESTION, "dblp_key": "journals/bbb/p7", "answer": "with graph neural networks"}]
    pools = {"q1": {"candidates": {"conf/aaa/p6": {"title": "Graph learning at scale", "dblp_rank": 1}},
                    "aliases": ["journals/bbb/p7"]}}
    oracle = {"q1": {"title": "Graph neural networks for traffic", "abstract": "THE ABSTRACT THE QUESTION WAS WRITTEN FROM"}}
    before = RAG.pool_sha(pools, rows)
    found = RAG.index_pools(rows, pools, oracle, tmp_path)
    assert found["q1"]["in_index"] and found["_meta"]["snapshot"] == "2099-01-01"
    # the source is shown with the oracle's abstract, as in the pool
    assert found["q1"]["candidates"]["journals/bbb/p7"]["abstract"].startswith("THE ABSTRACT")
    rankings, report = {}, {"rankers": {}}
    RAG._add_index(rows, pools, rankings, report, found)
    assert rankings["dewey-index"]["q1"][0] == "journals/bbb/p7"
    assert report["rankers"]["dewey-index"]["recall@1"] == 1.0 and report["index"]["source_in_index"] == 1.0
    assert RAG.pool_sha(pools, rows) == before, "the pool and its fingerprint are untouched"
    ctx = RAG.rag_contexts(rows, pools, rankings, "dewey-index", k=5)["q1"]
    assert ctx["source_in_context"] and "THE ABSTRACT" in ctx["abstract"]
    # kept: with the index gone, the same candidates come back from the file
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "elsewhere")
    AI.forget()
    again = RAG.index_pools(rows, {"q1": dict(pools["q1"])}, oracle, tmp_path)
    assert list(again["q1"]["candidates"]) == list(found["q1"]["candidates"])
