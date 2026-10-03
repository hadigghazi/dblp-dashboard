"""
Realistic retrieval for DBLP-QA, without the network or a model.

The rankers decide which abstracts a model sees, so they are tested on their own; the pool builder is
run against fake Semantic Scholar, OpenAlex and dblp-search servers; and a rag run is checked to split
its scores by whether retrieval found the source paper, which is the analysis the study turns on.
"""
import json
import math

import httpx

from chat import config, dblpqa as DQ, dblpqa_rag as RAG

from tests.test_dblpqa import CSV, Scripted

# questions with words BM25 can use (single letters are dropped like stop words)
POOL_CSV = """id,question,answer,dblp_key,semantic_scholar_id
qa1,What is alpha sensing?,Alpha sensing is the first thing.,conf/x/A1,1
qa2,What is beta learning?,Beta learning is the second thing.,conf/x/B2,2
qa3,What is gamma search?,Gamma search is the third thing.,conf/x/C3,3
"""


def test_bm25_prefers_the_document_with_the_rare_query_terms():
    docs = {"d1": "compressive sensing recovers sparse signals from few measurements",
            "d2": "a survey of signals and systems",
            "d3": "deep learning for images"}
    assert RAG.bm25_rank("What is compressive sensing?", docs)[0] == "d1"
    # question words and stop words carry no weight: nothing here matches, so the order is the key order
    assert RAG.bm25_rank("What is the?", docs) == ["d1", "d2", "d3"]


def test_bm25_normalises_for_length():
    short = "graph neural networks"
    padded = "graph neural networks " + " ".join(f"filler{i}" for i in range(200))
    assert RAG.bm25_rank("graph neural networks", {"long": padded, "short": short})[0] == "short"


def test_rank_fusion_rewards_agreement():
    assert RAG.rrf([["a", "b", "c"], ["b", "a", "c"]])[-1] == "c"
    assert RAG.rrf([["x", "b"], ["y", "b"]])[0] == "b", "second in both beats first in only one"


def test_the_source_is_found_under_any_of_its_keys():
    assert RAG.source_rank(["x", "corr/abs-1", "conf/a/1"], {"conf/a/1", "corr/abs-1"}) == 2
    assert RAG.source_rank(["x", "y"], {"z"}) is None


def test_recall_and_mrr():
    got = RAG.retrieval_metrics([1, 3, None, 6])
    assert got["recall@1"] == 0.25 and got["recall@3"] == 0.5 and got["recall@5"] == 0.5
    assert got["recall@10"] == 0.75 and got["ranked_at_all"] == 0.75
    assert math.isclose(got["mrr@10"], round((1 + 1 / 3 + 1 / 6) / 4, 3))


class FakeEmbeddings:
    """A vector per text from a few words, so 'dense' has a meaning the test can predict."""
    WORDS = ("alpha", "beta", "gamma", "first", "second", "third")

    def get(self, texts, out=print):
        got = {}
        for t in texts:
            words = t.lower().replace("?", " ").replace(".", " ").split()
            vec = [words.count(w) + 1e-3 for w in self.WORDS]
            norm = math.sqrt(sum(x * x for x in vec))
            got[t] = [x / norm for x in vec]
        return got

    def cost(self):
        return 0.0


def fake_web(requests):
    """dblp search, Semantic Scholar search and batch, OpenAlex - each question has its paper in the
    pool, but qa2's only under its published version's key and qa3's not at all."""
    long = lambda s: (s + " ") * 25

    def handler(request):
        requests.append(str(request.url))
        url, host = request.url, request.url.host
        if host == "searchapi":
            q = url.params["q"]
            hits = {"What is alpha sensing?": [{"key": "conf/x/A1", "title": "On alpha sensing"},
                                               {"key": "conf/x/Z9", "title": "Z"}],
                    "What is beta learning?": [{"key": "conf/x/Z9", "title": "Z"}],
                    "What is gamma search?": [{"key": "conf/x/Z9", "title": "Z"}]}[q]
            return httpx.Response(200, json={"results": hits})
        if host == "api.semanticscholar.org" and url.path.endswith("/paper/search"):
            q = url.params["query"]
            data = {"What is alpha sensing?": [],
                    "What is beta learning?": [{"title": "On beta learning", "abstract": None,
                                    "externalIds": {"DBLP": "journals/y/B2", "DOI": "10.1/b"}},
                                   {"title": "no dblp key", "externalIds": {}}],
                    "What is gamma search?": [{"title": "Other", "abstract": "unrelated text here",
                                    "externalIds": {"DBLP": "conf/x/Q7"}}]}[q]
            return httpx.Response(200, json={"data": data})
        if host == "api.semanticscholar.org" and url.path.endswith("/paper/batch"):
            ids = json.loads(request.content)["ids"]
            fields = url.params["fields"]
            if fields == "externalIds":        # the benchmark's own papers
                return httpx.Response(200, json=[{"externalIds": {"DBLP": k}} for k in
                                                 ("conf/x/A1", "journals/y/B2", "conf/x/C3")][:len(ids)])
            if fields == "abstract":           # pool papers by DOI: withheld
                return httpx.Response(200, json=[{"abstract": None} for _ in ids])
            return httpx.Response(200, json=[{"title": t, "abstract": long(a), "externalIds": {}} for t, a in
                                             (("On alpha sensing", "Alpha sensing is the first thing."),
                                              ("On beta learning", "Beta learning is the second thing."),
                                              ("On gamma search", "Gamma search is the third thing."))][:len(ids)])
        if host == "api.openalex.org":
            return httpx.Response(200, json={"results": [
                {"doi": "https://doi.org/10.1/b", "abstract_inverted_index": {"withheld": [0], "elsewhere": [1]}}]})
        return httpx.Response(404)
    return handler


def test_the_pool_is_built_from_both_searches_with_the_source_under_any_key(tmp_path, monkeypatch):
    monkeypatch.setattr(RAG.time, "sleep", lambda _s: None)
    monkeypatch.setattr(config, "SEARCH_URL", "http://searchapi")
    requests = []
    http = httpx.Client(transport=httpx.MockTransport(fake_web(requests)))
    rows = DQ.parse(POOL_CSV)
    pools, rankings, report = RAG.prepare(rows, out=lambda *_: None, cache_dir=tmp_path, http=http,
                                          embeddings=FakeEmbeddings())

    a = pools["qa1"]["candidates"]
    assert a["conf/x/A1"]["dblp_rank"] == 1 and a["conf/x/A1"]["abstract"].startswith("Alpha sensing is the first")
    b = pools["qa2"]["candidates"]
    assert set(b) == {"conf/x/Z9", "journals/y/B2"}, "a Semantic Scholar hit with no dblp key is dropped"
    assert "journals/y/B2" in pools["qa2"]["aliases"], "the published version counts as the source"
    assert b["journals/y/B2"]["abstract"].startswith("Beta learning is the second"), \
        "the source is shown with the oracle's abstract, not OpenAlex's"
    assert report["pool"]["source_in_pool"] == round(2 / 3, 3)
    assert report["rankers"]["bm25"]["ranks"] == {"qa1": 1, "qa2": 1, "qa3": None}
    assert report["rankers"]["s2-search"]["ranks"]["qa1"] is None, "Semantic Scholar found nothing for qa1"
    assert (tmp_path / "retrieval.json").exists()

    before = len(requests)
    RAG.prepare(rows, out=lambda *_: None, cache_dir=tmp_path, http=http, embeddings=FakeEmbeddings())
    searches = [u for u in requests[before:] if "search" in u]
    assert not searches, "pools are cached: a second run searches nothing"


def test_a_pooled_paper_without_an_abstract_gets_one_from_openalex(tmp_path, monkeypatch):
    monkeypatch.setattr(RAG, "ids_from_dblp",
                        lambda keys: {"conf/x/Z9": {"doi": "10.1/b"}} if "conf/x/Z9" in keys else {})
    entries = {"conf/x/Z9": {"title": "Z"}}
    cache = {}
    http = httpx.Client(transport=httpx.MockTransport(fake_web([])))
    RAG.fetch_pool_abstracts(entries, cache, http, out=lambda *_: None)
    assert entries["conf/x/Z9"]["abstract"] == "withheld elsewhere" and cache["conf/x/Z9"] == "withheld elsewhere"


def test_dois_and_arxiv_ids_come_from_the_dblp_links(tmp_path):
    import duckdb
    parquet = tmp_path / "dblp.parquet"
    con = duckdb.connect()
    con.execute("COPY (SELECT * FROM (VALUES ('conf/x/A1', ['https://doi.org/10.1145/123.456']), "
                "('conf/x/B2', ['https://example.org/paper']), "
                "('journals/corr/abs-2101-00001', ['https://arxiv.org/abs/2101.00001v2'])) t(key, ee)) "
                f"TO '{parquet.as_posix()}' (FORMAT PARQUET)")
    con.close()
    got = RAG.ids_from_dblp(["conf/x/A1", "conf/x/B2", "journals/corr/abs-2101-00001", "nope"], parquet)
    assert got == {"conf/x/A1": {"doi": "10.1145/123.456"}, "journals/corr/abs-2101-00001": {"arxiv": "2101.00001"}}
    assert RAG.ids_from_dblp(["conf/x/A1"], tmp_path / "missing.parquet") == {}
    assert RAG._s2_id({"arxiv": "2101.00001"}) == "ARXIV:2101.00001" and RAG._s2_id({}) is None


class Reader(Scripted):
    """Answers with the first retrieved abstract - right when retrieval put the source first."""

    def complete(self, messages, model, tools=None, temperature=None, extra=None):
        user = messages[-1]["content"]
        if messages[0]["content"] == DQ.RAG_SYSTEM:
            lines = user.split("Abstracts:\n")[1].split("\n\n")[0].split("\n")
            return {"content": lines[1] if len(lines) > 1 else "I do not know.", "usage": {"input_tokens": 50, "output_tokens": 5}}
        return super().complete(messages, model, tools, temperature, extra)


def test_a_rag_run_splits_its_scores_by_whether_the_source_was_retrieved(tmp_path):
    rows = DQ.parse(CSV)
    runs = tmp_path / "runs"
    quiet = lambda *_: None
    DQ.run_condition(Scripted(), ["m1"], "judge", rows, "sha", "closed-book",
                     out_dir=runs / "20260101T000000Z-closed-book", out=quiet, reuse_controls=False)
    oracle = {r["id"]: {"abstract": r["answer"], "source": "x"} for r in rows}
    DQ.run_condition(Scripted(), ["m1"], "judge", rows, "sha", "oracle", contexts=oracle,
                     out_dir=runs / "20260102T000000Z-oracle", out=quiet, reuse_controls=False)

    pools = {"qa1": {"aliases": ["conf/x/A1"], "candidates": {
                 "conf/x/A1": {"title": "On A", "abstract": "A is the first thing."},
                 "conf/x/Z9": {"title": "Z", "abstract": "Z is unrelated."}}},
             "qa2": {"aliases": ["conf/x/B2"], "candidates": {
                 "conf/x/Z9": {"title": "Z", "abstract": "Z is unrelated."}}},
             "qa3": {"aliases": ["conf/x/C3"], "candidates": {}}}
    rankings = {"bm25": {"qa1": ["conf/x/A1", "conf/x/Z9"], "qa2": ["conf/x/Z9"], "qa3": []}}
    contexts = RAG.rag_contexts(rows, pools, rankings, "bm25", k=5)
    assert contexts["qa1"]["source_in_context"] and contexts["qa1"]["source_rank"] == 1
    assert not contexts["qa2"]["source_in_context"] and contexts["qa3"]["abstract"] == "(no papers were retrieved)"

    got = DQ.run_condition(Reader(), ["m1"], "judge", rows, "sha", "rag-bm25", contexts=contexts,
                           out_dir=runs / "20260103T000000Z-rag-bm25", out=quiet, reuse_controls=False)
    m1 = got["results"]["m1"]
    assert got["questions"] == 3, "a rag run asks every question, found or not"
    assert got["answer_prompt"] == DQ.RAG_SYSTEM
    assert m1["vs_closed_book"]["better"] == 1 and m1["vs_oracle"]["against_run"].endswith("oracle")
    hit, miss = m1["by_retrieval"]["source retrieved"], m1["by_retrieval"]["source missed"]
    assert hit["questions"] == 1 and hit["judge_score"]["mean"] == 2.0
    assert miss["questions"] == 2 and miss["vs_closed_book"]["worse"] == 0
    saved = [json.loads(x) for x in (runs / "20260103T000000Z-rag-bm25" / "answers.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    assert saved[0]["retrieved"] == ["conf/x/A1", "conf/x/Z9"] and saved[0]["context_source"] == "bm25@5"
    lines = []
    DQ.print_summary(got["results"], "rag-bm25", out=lines.append)
    assert any("source missed: 2 questions" in line for line in lines)
