"""
Realistic retrieval for DBLP-QA, without the network or a model.

The rankers decide which abstracts a model sees, so they are tested on their own; the pool builder is
run against fake dblp-search, Semantic Scholar and OpenAlex servers and a tiny dblp parquet; the pool
is checked to stay frozen once built; and rag runs are checked to split their scores by whether
retrieval found the source paper, and - for selective retrieval - to pair with plain RAG on the same
pool, which are the analyses the study turns on.
"""
import json
import math

import httpx
import pytest

from chat import config, dblpqa as DQ, dblpqa_rag as RAG

from tests.test_dblpqa import CSV, Scripted

# questions with words BM25 can use (single letters are dropped like stop words)
POOL_CSV = """id,question,answer,dblp_key,semantic_scholar_id
qa1,What is alpha sensing?,Alpha sensing is the first thing.,conf/x/A1,1
qa2,What is beta learning?,Beta learning is the second thing.,conf/x/B2,2
qa3,What is gamma search?,Gamma search is the third thing.,conf/x/C3,3
"""

DBLP_RECORDS = [
    ("conf/x/A1", "inproceedings", "On alpha sensing", ["https://doi.org/10.9/a1"]),
    ("conf/x/C3", "inproceedings", "On gamma search", ["https://doi.org/10.9/C3"]),
    ("conf/x/T1", "article", "A rather long title about delta things", []),
    ("journals/corr/abs-2101-00001", "article", "Epsilon preprint with a long enough title",
     ["https://arxiv.org/abs/2101.00001v2"]),
    ("conf/y/E5", "inproceedings", "Epsilon preprint with a long enough title", []),
    ("homepages/x/y", "www", "A rather long title about delta things", []),
]

quiet = lambda *_: None


def make_parquet(path, records=DBLP_RECORDS):
    import duckdb
    con = duckdb.connect()
    con.execute("CREATE TABLE t (key VARCHAR, type VARCHAR, title VARCHAR, ee VARCHAR[])")
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", [list(r) for r in records])
    con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()
    return path


# --------------------------------------------------------------------------- rankers

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


def test_openalex_queries_are_or_ed_and_never_read_as_operators():
    assert RAG.openalex_query("What is Alpha AND Beta-sensing?", semantic=False) == \
        {"search": "alpha OR beta OR sensing"}, "upper-case AND would be an operator; a plain AND of a whole question finds nothing"
    assert RAG.openalex_query("What is the?", semantic=False) is None
    assert RAG.openalex_query("What is  alpha?", semantic=True) == {"search.semantic": "What is alpha?"}


# --------------------------------------------------------------------------- dblp records

def test_dois_and_arxiv_ids_come_from_the_dblp_links(tmp_path):
    parquet = make_parquet(tmp_path / "dblp.parquet")
    got = RAG.ids_from_dblp(["conf/x/A1", "conf/x/T1", "journals/corr/abs-2101-00001", "nope"], parquet)
    assert got == {"conf/x/A1": {"doi": "10.9/a1"}, "journals/corr/abs-2101-00001": {"arxiv": "2101.00001"}}
    assert RAG.ids_from_dblp(["conf/x/A1"], tmp_path / "missing.parquet") == {}
    assert RAG._s2_id({"arxiv": "2101.00001"}) == "ARXIV:2101.00001" and RAG._s2_id({}) is None


def test_openalex_works_are_mapped_to_dblp_by_doi_arxiv_and_title(tmp_path):
    parquet = make_parquet(tmp_path / "dblp.parquet")
    works = [{"id": "W1", "doi": "10.9/c3", "title": "whatever"},
             {"id": "W2", "doi": "10.48550/arxiv.2101.00001", "title": "x"},
             {"id": "W3", "doi": None, "title": "A Rather Long Title About Delta Things!"},
             {"id": "W4", "doi": None, "title": "Epsilon preprint with a long enough title"},
             {"id": "W5", "doi": "10.5/elsewhere", "title": "Short"}]
    assert RAG.map_to_dblp(works, parquet) == {
        "W1": "conf/x/C3",                       # DOI, whatever its case in dblp
        "W2": "journals/corr/abs-2101-00001",    # arXiv DOI to the CoRR record, version suffix and all
        "W3": "conf/x/T1",                       # title - the home page with that title is not a paper
        "W4": "conf/y/E5"}                       # a title shared with a preprint: the published version
    assert RAG.map_to_dblp(works, tmp_path / "missing.parquet") == {}


# --------------------------------------------------------------------------- the pool

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


def fake_web(log, s2_refuses=()):
    """dblp search, Semantic Scholar and OpenAlex. qa1's paper is found by dblp search, qa2's only by
    Semantic Scholar under its published version's key, qa3's only by OpenAlex's semantic search."""
    long = lambda s: (s + " ") * 25

    def handler(request):
        log.append(request.url)
        url, host = request.url, request.url.host
        if host == "searchapi":
            hits = {"What is alpha sensing?": [{"key": "conf/x/A1", "title": "On alpha sensing"},
                                               {"key": "conf/x/Z9", "title": "Z"}],
                    "What is beta learning?": [{"key": "conf/x/Z9", "title": "Z"}],
                    "What is gamma search?": [{"key": "conf/x/Z9", "title": "Z"}]}[url.params["q"]]
            return httpx.Response(200, json={"results": hits})
        if host == "api.semanticscholar.org" and url.path.endswith("/paper/search"):
            q = url.params["query"]
            if q in s2_refuses:
                return httpx.Response(429)
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
            params = url.params
            if "search.semantic" in params:
                results = [] if params["search.semantic"] != "What is gamma search?" else [
                    {"id": "https://openalex.org/W3", "doi": "https://doi.org/10.9/C3",
                     "display_name": "On gamma search", "abstract_inverted_index": {"OpenAlex": [0], "text": [1]}},
                    {"id": "https://openalex.org/W9", "doi": "https://doi.org/10.5/elsewhere",
                     "display_name": "Biology of things", "abstract_inverted_index": None},
                    {"id": "https://openalex.org/W4", "doi": None,
                     "display_name": "A Rather Long Title About Delta Things!",
                     "abstract_inverted_index": {"delta": [0], "things": [1]}}]
                return httpx.Response(200, json={"results": results})
            if "search" in params:
                results = [] if params["search"] != "alpha OR sensing" else [
                    {"id": "https://openalex.org/W5", "doi": "https://doi.org/10.48550/arXiv.2101.00001",
                     "display_name": "Epsilon preprint with a long enough title", "abstract_inverted_index": None}]
                return httpx.Response(200, json={"results": results})
            return httpx.Response(200, json={"results": [      # abstracts by DOI
                {"doi": "https://doi.org/10.1/b", "abstract_inverted_index": {"withheld": [0], "elsewhere": [1]}}]})
        return httpx.Response(404)
    return handler


@pytest.fixture
def web(tmp_path, monkeypatch):
    monkeypatch.setattr(RAG.time, "sleep", lambda _s: None)
    monkeypatch.setattr(config, "SEARCH_URL", "http://searchapi")
    monkeypatch.setattr(RAG, "PARQUET", make_parquet(tmp_path / "dblp.parquet"))
    log, refuses = [], set()
    http = httpx.Client(transport=httpx.MockTransport(fake_web(log, refuses)))
    prep = lambda **kw: RAG.prepare(DQ.parse(POOL_CSV), out=quiet, cache_dir=tmp_path / "cache", http=http,
                                    embeddings=FakeEmbeddings(), **kw)
    return {"log": log, "refuses": refuses, "prepare": prep}


def test_the_pool_joins_four_searches_and_finds_the_source_under_any_key(web):
    pools, rankings, report = web["prepare"]()

    a = pools["qa1"]["candidates"]
    assert a["conf/x/A1"]["dblp_rank"] == 1 and a["conf/x/A1"]["abstract"].startswith("Alpha sensing is the first")
    assert a["journals/corr/abs-2101-00001"]["oa_rank"] == 1, "OpenAlex keyword search, mapped by arXiv id"
    b = pools["qa2"]["candidates"]
    assert set(b) == {"conf/x/Z9", "journals/y/B2"}, "a Semantic Scholar hit with no dblp key is dropped"
    assert "journals/y/B2" in pools["qa2"]["aliases"], "the published version counts as the source"
    assert b["journals/y/B2"]["abstract"].startswith("Beta learning is the second"), \
        "the source is shown with the oracle's abstract, not OpenAlex's"
    c = pools["qa3"]["candidates"]
    assert c["conf/x/C3"]["oas_rank"] == 1 and c["conf/x/T1"]["oas_rank"] == 2, \
        "OpenAlex semantic results ranked among the dblp records only (the biology paper is dropped)"
    assert c["conf/x/T1"]["abstract"] == "delta things"

    assert report["pool"]["source_in_pool"] == 1.0 and report["pool"]["complete"]
    assert report["rankers"]["openalex-semantic"]["ranks"] == {"qa1": None, "qa2": None, "qa3": 1}
    assert report["rankers"]["bm25"]["ranks"]["qa2"] == 1
    assert report["rankers"]["s2-search"]["ranks"]["qa1"] is None, "Semantic Scholar found nothing for qa1"
    assert any(u.host == "api.openalex.org" and u.params.get("search") == "alpha OR sensing" for u in web["log"])

    before = len(web["log"])
    _, _, frozen = web["prepare"](frozen=True)
    assert len(web["log"]) == before, "a frozen pool searches nothing"
    assert frozen["pool"]["sha256"] == report["pool"]["sha256"]


def test_a_refused_search_leaves_the_pool_incomplete_until_a_rerun_finishes_it(web):
    web["refuses"].add("What is beta learning?")
    _, _, report = web["prepare"]()
    assert not report["pool"]["complete"] and report["pool"]["failed"]["s2-search"] == 1
    with pytest.raises(SystemExit, match="missing searches"):
        web["prepare"](frozen=True)
    web["prepare"](frozen=True, allow_incomplete=True)

    web["refuses"].clear()
    before = len(web["log"])
    pools, _, report = web["prepare"]()
    again = web["log"][before:]
    assert report["pool"]["complete"] and "journals/y/B2" in pools["qa2"]["candidates"]
    assert len([u for u in again if u.path.endswith("/paper/search")]) == 1, "only the refused search is redone"
    assert not [u for u in again if u.host == "searchapi"]
    assert not [u for u in again if u.host == "api.openalex.org" and ("search" in u.params or "search.semantic" in u.params)]


def test_a_pooled_paper_without_an_abstract_gets_one_from_openalex(monkeypatch):
    monkeypatch.setattr(RAG, "ids_from_dblp",
                        lambda keys: {"conf/x/Z9": {"doi": "10.1/b"}} if "conf/x/Z9" in keys else {})
    entries = {"conf/x/Z9": {"title": "Z"}}
    cache = {}
    http = httpx.Client(transport=httpx.MockTransport(fake_web([])))
    RAG.fetch_pool_abstracts(entries, cache, http, out=quiet)
    assert entries["conf/x/Z9"]["abstract"] == "withheld elsewhere" and cache["conf/x/Z9"] == "withheld elsewhere"


# --------------------------------------------------------------------------- answering from the pool

class Reader(Scripted):
    """Answers with the first abstract it is given - right when that is the source."""

    def complete(self, messages, model, tools=None, temperature=None, extra=None):
        user = messages[-1]["content"]
        if messages[0]["content"] in (DQ.RAG_SYSTEM, DQ.PERMISSIVE_SYSTEM):
            lines = user.split("Abstracts:\n")[1].split("\n\n")[0].split("\n")
            return {"content": lines[1] if len(lines) > 1 else "I do not know.",
                    "usage": {"input_tokens": 50, "output_tokens": 5}}
        return super().complete(messages, model, tools, temperature, extra)


class Gatekeeper(Reader):
    """Gives the gate's verdict per question from a script."""

    def __init__(self, verdicts):
        super().__init__()
        self.verdicts, self.gate_calls = verdicts, 0

    def complete(self, messages, model, tools=None, temperature=None, extra=None):
        if messages[0]["content"] == RAG.GATE_SYSTEM:
            self.gate_calls += 1
            question = messages[1]["content"].split("Question: ")[1].split("\n")[0]
            return {"content": self.verdicts[question], "usage": {"input_tokens": 300, "output_tokens": 8}}
        return super().complete(messages, model, tools, temperature, extra)


SMALL_POOLS = {"qa1": {"aliases": ["conf/x/A1"], "candidates": {
                   "conf/x/A1": {"title": "On A", "abstract": "A is the first thing."},
                   "conf/x/Z9": {"title": "Z", "abstract": "Z is unrelated."}}},
               "qa2": {"aliases": ["conf/x/B2"], "candidates": {
                   "conf/x/Z9": {"title": "Z", "abstract": "Z is unrelated."}}},
               "qa3": {"aliases": ["conf/x/C3"], "candidates": {}}}


def baselines(runs, rows):
    DQ.run_condition(Scripted(), ["m1"], "judge", rows, "sha", "closed-book",
                     out_dir=runs / "20260101T000000Z-closed-book", out=quiet, reuse_controls=False)
    oracle = {r["id"]: {"abstract": r["answer"], "source": "x"} for r in rows}
    DQ.run_condition(Scripted(), ["m1"], "judge", rows, "sha", "oracle", contexts=oracle,
                     out_dir=runs / "20260102T000000Z-oracle", out=quiet, reuse_controls=False)


def test_a_rag_run_splits_its_scores_by_whether_the_source_was_retrieved(tmp_path):
    rows = DQ.parse(CSV)
    runs = tmp_path / "runs"
    baselines(runs, rows)
    rankings = {"bm25": {"qa1": ["conf/x/A1", "conf/x/Z9"], "qa2": ["conf/x/Z9"], "qa3": []}}
    contexts = RAG.rag_contexts(rows, SMALL_POOLS, rankings, "bm25", k=5)
    assert contexts["qa1"]["source_in_context"] and contexts["qa1"]["source_rank"] == 1
    assert not contexts["qa2"]["source_in_context"] and contexts["qa3"]["abstract"] == "(no papers were retrieved)"

    got = DQ.run_condition(Reader(), ["m1"], "judge", rows, "sha", "rag-bm25", contexts=contexts,
                           out_dir=runs / "20260103T000000Z-rag-bm25", out=quiet, reuse_controls=False,
                           notes={"pool_sha256": "P1"})
    m1 = got["results"]["m1"]
    assert got["questions"] == 3, "a rag run asks every question, found or not"
    assert got["answer_prompt"] == DQ.RAG_SYSTEM and got["pool_sha256"] == "P1"
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


def test_the_rag_modes_have_their_own_prompts():
    assert DQ.rag_parts("rag-openalex-semantic-gated") == ("rag-openalex-semantic", "gated")
    assert DQ.rag_parts("rag-bm25") == ("rag-bm25", "plain")
    assert RAG.condition_name("bm25", "permissive") == "rag-bm25-permissive"
    assert DQ.messages_for("rag-bm25-permissive", "q", "ctx")[0]["content"] == DQ.PERMISSIVE_SYSTEM
    assert DQ.messages_for("rag-bm25", "q", "ctx")[0]["content"] == DQ.RAG_SYSTEM
    assert DQ.messages_for("rag-bm25-gated", "q", None) == DQ.messages_for("closed-book", "q"), \
        "a question whose gate kept nothing is asked exactly as closed-book"


def test_the_gate_keeps_only_what_addresses_the_question_and_remembers_its_verdicts(tmp_path):
    rows = DQ.parse(CSV)
    rankings = {"bm25": {"qa1": ["conf/x/Z9", "conf/x/A1"], "qa2": ["conf/x/Z9"], "qa3": []}}
    contexts = RAG.rag_contexts(rows, SMALL_POOLS, rankings, "bm25", k=5)
    client = Gatekeeper({"What is A?": '{"relevant": [2]}', "What is B?": 'Sure: {"relevant": []}'})
    gated, info = RAG.gate_contexts(client, rows, SMALL_POOLS, contexts, cache_dir=tmp_path, out=quiet)
    assert gated["qa1"]["kept"] == ["conf/x/A1"] and gated["qa1"]["gate_kept_source"]
    assert gated["qa1"]["abstract"] == "[1] On A\nA is the first thing.", "renumbered, the dropped one gone"
    assert gated["qa2"]["abstract"] is None and gated["qa3"]["abstract"] is None
    assert info["passed_nothing"] == 2 and info["unreadable"] == 0
    assert client.gate_calls == 2, "nothing retrieved for qa3: no gate call"

    again = Gatekeeper({})
    regated, _ = RAG.gate_contexts(again, rows, SMALL_POOLS, contexts, cache_dir=tmp_path, out=quiet)
    assert again.gate_calls == 0 and regated == gated, "cached: every answer model gets the same context"


def test_an_unreadable_gate_verdict_keeps_everything(tmp_path):
    rows = DQ.parse(CSV)
    rankings = {"bm25": {"qa1": ["conf/x/Z9", "conf/x/A1"], "qa2": ["conf/x/Z9"], "qa3": []}}
    contexts = RAG.rag_contexts(rows, SMALL_POOLS, rankings, "bm25", k=5)
    client = Gatekeeper({"What is A?": "the second one, I think", "What is B?": '{"relevant": [7]}'})
    gated, info = RAG.gate_contexts(client, rows, SMALL_POOLS, contexts, cache_dir=tmp_path, out=quiet)
    assert gated["qa1"]["kept"] == ["conf/x/Z9", "conf/x/A1"] and gated["qa2"]["kept"] == ["conf/x/Z9"]
    assert info["unreadable"] == 2


def test_a_gated_run_is_paired_with_plain_rag_on_the_same_pool_only(tmp_path):
    rows = DQ.parse(CSV)
    runs = tmp_path / "runs"
    baselines(runs, rows)
    # the source is second, behind an unrelated paper: plain RAG answers from the wrong abstract
    rankings = {"bm25": {"qa1": ["conf/x/Z9", "conf/x/A1"], "qa2": ["conf/x/Z9"], "qa3": []}}
    contexts = RAG.rag_contexts(rows, SMALL_POOLS, rankings, "bm25", k=5)
    DQ.run_condition(Reader(), ["m1"], "judge", rows, "sha", "rag-bm25", contexts=contexts,
                     out_dir=runs / "20260103T000000Z-rag-bm25", out=quiet, reuse_controls=False,
                     notes={"pool_sha256": "P1"})
    client = Gatekeeper({"What is A?": '{"relevant": [2]}', "What is B?": '{"relevant": []}'})
    gated, info = RAG.gate_contexts(client, rows, SMALL_POOLS, contexts, cache_dir=tmp_path, out=quiet)

    got = DQ.run_condition(client, ["m1"], "judge", rows, "sha", "rag-bm25-gated", contexts=gated,
                           out_dir=runs / "20260104T000000Z-rag-bm25-gated", out=quiet, reuse_controls=False,
                           notes={"pool_sha256": "P1", "gate": info})
    m1 = got["results"]["m1"]
    assert m1["vs_plain"]["against_run"].endswith("-rag-bm25") and m1["vs_plain"]["better"] == 1
    hit, miss = m1["by_retrieval"]["source retrieved"], m1["by_retrieval"]["source missed"]
    assert hit["gate_kept_source"] == 1 and miss["gate_passed_nothing"] == 2
    assert got["gate"]["passed_nothing"] == 2
    saved = [json.loads(x) for x in (runs / "20260104T000000Z-rag-bm25-gated" / "answers.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    assert saved[0]["kept"] == ["conf/x/A1"] and saved[0]["gate_kept_source"] is True

    other = DQ.run_condition(client, ["m1"], "judge", rows, "sha", "rag-bm25-gated", contexts=gated,
                             out_dir=runs / "20260105T000000Z-rag-bm25-gated", out=quiet, reuse_controls=False,
                             notes={"pool_sha256": "P2"})
    assert other["results"]["m1"]["vs_plain"] is None, "a plain run on another pool is no comparison"
    lines = []
    DQ.print_summary(other["results"], "rag-bm25-gated", out=lines.append)
    assert any("no plain RAG run on this pool" in line for line in lines)


def test_a_saturated_semantic_scholar_is_given_up_on_instead_of_waited_for(web, monkeypatch):
    monkeypatch.setattr(RAG, "S2_GIVE_UP", 2)
    web["refuses"].update({"What is alpha sensing?", "What is beta learning?", "What is gamma search?"})
    _, _, report = web["prepare"]()
    s2_calls = [u for u in web["log"] if u.path.endswith("/paper/search")]
    assert len(s2_calls) == 2 * RAG.S2_ATTEMPTS, "the third question's search is not even tried"
    assert report["pool"]["failed"]["s2-search"] == 3 and report["pool"]["failed"]["openalex-semantic"] == 0
