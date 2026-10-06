"""
The abstract search, without the network: dblp's title search and OpenAlex are fake servers, the
dump is the synthetic one every tool test uses.

What matters is what the model is given and what it costs: the right papers, ranked the way the
study ranked them, with abstracts; never a paper dblp does not have; a slow or rationed source cut
off and said so rather than waited for; and a result that is the same when asked again.
"""
import json
import threading
import time

import httpx
import pytest

from chat import agent, config, content, data, dblpqa as DQ, dblpqa_dewey as DW, dblpqa_rag as RAG, paperids
from chat import tools as T


ABSTRACTS = {
    "10.1109/7": "Graph neural networks forecast traffic flow on road networks from sensor data.",
    "10.1109/6": "We train graph learning models on billions of edges with a new partitioning scheme.",
    "10.1109/9": "Transformers applied to graphs, measured on many practical workloads.",
    "10.1109/8": "Graph embeddings make retrieval of similar items fast and accurate.",
}


def inverted(text):
    index = {}
    for i, word in enumerate(text.split()):
        index.setdefault(word, []).append(i)
    return index


def fake_web(log, slow=()):
    def handler(request):
        url = request.url
        log.append(url)
        if url.host == "searchapi":
            return httpx.Response(200, json={"results": [{"key": "conf/aaa/p6", "title": "Graph learning at scale"},
                                                         {"key": "conf/aaa/p9", "title": "Graph transformers in practice"}]})
        if url.host == "api.openalex.org":
            params = url.params
            if "search.semantic" in params:
                if "semantic" in slow:
                    time.sleep(3)
                return httpx.Response(200, json={"results": [
                    {"id": "https://openalex.org/W8", "doi": None, "display_name": "Graph Embeddings for Retrieval",
                     "abstract_inverted_index": inverted(ABSTRACTS["10.1109/8"])},
                    {"id": "https://openalex.org/W99", "doi": "https://doi.org/10.5555/elsewhere",
                     "display_name": "A paper that dblp does not have at all", "abstract_inverted_index": None}]})
            if "search" in params:
                return httpx.Response(200, json={"results": [
                    {"id": "https://openalex.org/W7", "doi": "https://doi.org/10.1109/7",
                     "display_name": "Graph neural networks for traffic",
                     "abstract_inverted_index": inverted(ABSTRACTS["10.1109/7"])}]})
            if "filter" in params:
                wanted = params["filter"].removeprefix("doi:").split("|")
                return httpx.Response(200, json={"results": [
                    {"doi": f"https://doi.org/{d}", "abstract_inverted_index": inverted(ABSTRACTS[d])}
                    for d in wanted if d in ABSTRACTS]})
        return httpx.Response(404)
    return handler


@pytest.fixture(scope="module")
def indexed(loaded):
    """The paper-id index built from the synthetic dump and attached to the shared connection."""
    con, meta = data.connect(loaded["serving"])
    try:
        paperids.build(con, meta)
    finally:
        con.close()
    assert paperids.attach(data.pool.connection(), data.pool.meta)["dois"] != "0"
    return loaded


@pytest.fixture
def web(indexed, monkeypatch):
    monkeypatch.setattr(config, "SEARCH_URL", "http://searchapi")
    monkeypatch.setattr(content, "_semantic_last", [0.0])
    log = []

    def make(slow=()):
        http = httpx.Client(transport=httpx.MockTransport(fake_web(log, slow)))
        return agent.Ctx(data.pool, http, indexed["store_meta"])
    return {"log": log, "ctx": make}


# --------------------------------------------------------------------------- the index

def test_the_paper_id_index_maps_dois_both_ways(indexed):
    cur = data.pool.cursor()
    by_doi, by_arxiv = paperids.keys_for(cur, ["10.1109/7", "10.9999/none"])
    assert by_doi == {"10.1109/7": "journals/bbb/p7"} and by_arxiv == {}
    ids = paperids.ids_for(cur, ["journals/bbb/p7", "conf/aaa/p3"])
    assert ids == {"journals/bbb/p7": {"doi": "10.1109/7"}}, "a record with no DOI link has no ids"


def test_openalex_works_map_to_dblp_by_doi_then_title(indexed):
    works = [{"id": "W1", "doi": "10.1109/7", "title": "whatever it is called there"},
             {"id": "W2", "doi": None, "title": "Graph Embeddings for Retrieval!"},
             {"id": "W3", "doi": "10.5555/x", "title": "A paper that dblp does not have at all"},
             {"id": "W4", "doi": None, "title": "Short"}]
    assert content.map_works(data.pool.cursor(), works) == {"W1": "journals/bbb/p7", "W2": "journals/bbb/p8"}


# --------------------------------------------------------------------------- the tool

def test_the_tool_pools_three_searches_ranks_with_bm25_and_numbers_five(web):
    ctx = web["ctx"]()
    out = T.call(ctx, "search_abstracts", {"question": "How do graph neural networks forecast traffic?"})
    assert not out.get("refused"), out
    rows = out["rows"]
    assert [r["n"] for r in rows] == [1, 2, 3, 4] and rows[0]["key"] == "journals/bbb/p7"
    assert rows[0]["found_by"] == "openalex-search" and rows[0]["abstract"].startswith("Graph neural networks forecast")
    by_key = {r["key"]: r for r in rows}
    assert by_key["conf/aaa/p6"]["abstract"].startswith("We train graph learning"), \
        "dblp's own hits get their abstracts by DOI"
    assert by_key["journals/bbb/p8"]["found_by"] == "openalex-semantic", "mapped by title"
    assert by_key["journals/bbb/p8"]["venue"] and by_key["journals/bbb/p8"]["year"] == 2022, "from dblp's record"
    assert out["meta"]["sources"] == {"dblp-search": "ok", "openalex-search": "ok", "openalex-semantic": "ok"}
    assert out["meta"]["candidates"] == 4, "the work dblp does not have is dropped"
    assert not [u for u in web["log"] if "semanticscholar" in str(u)]
    assert "[n]" in out["note"]

    before = len(web["log"])
    again = T.call(web["ctx"](), "search_abstracts", {"question": "How do graph neural networks forecast traffic?"})
    assert len(web["log"]) == before and again["meta"]["cached"], "asked again: the cache, no request"
    assert [r["key"] for r in again["rows"]] == [r["key"] for r in rows]


def test_a_slow_source_is_cut_off_and_said_so_and_not_cached(web, monkeypatch):
    monkeypatch.setattr(config, "CONTENT_DEADLINE", 3.0)
    ctx = web["ctx"](slow=("semantic",))
    t0 = time.time()
    out = T.call(ctx, "search_abstracts", {"question": "graph transformers practical workloads slow"})
    assert time.time() - t0 < 3.5, "the tool keeps to its own deadline"
    assert out["meta"]["sources"]["openalex-semantic"] == "timeout"
    assert out["rows"], "the other two sources still answer"
    payload = {"name": "search_abstracts", "result": out}
    assert not agent._complete_search(payload), "an answer built on this is not cached"
    before = len(web["log"])
    T.call(web["ctx"](), "search_abstracts", {"question": "graph transformers practical workloads slow"})
    assert len(web["log"]) > before, "an incomplete search is searched again, not served from the cache"


def test_a_spent_openalex_allowance_degrades_to_dblp_titles(web, monkeypatch):
    monkeypatch.setattr(config, "OPENALEX_CALLS_PER_DAY", 0)
    out = T.call(web["ctx"](), "search_abstracts", {"question": "graph learning scale rationed"})
    assert out["meta"]["degraded"] and "Degraded" in out["note"]
    assert not [u for u in web["log"] if u.host == "api.openalex.org"]
    assert {r["key"] for r in out["rows"]} == {"conf/aaa/p6", "conf/aaa/p9"}


def test_the_frozen_pool_is_ranked_with_the_agents_query_and_nothing_is_searched(web):
    ctx = web["ctx"]()
    ctx.frozen_pool = {"candidates": {"journals/bbb/p8": {"title": "Graph embeddings for retrieval",
                                                          "abstract": ABSTRACTS["10.1109/8"]},
                                      "conf/aaa/p6": {"title": "Graph learning at scale", "abstract": ABSTRACTS["10.1109/6"]}}}
    out = T.call(ctx, "search_abstracts", {"question": "fast retrieval of similar items"})
    assert [r["key"] for r in out["rows"]] == ["journals/bbb/p8", "conf/aaa/p6"] and out["meta"]["frozen"]
    assert web["log"] == []


def test_known_papers_get_their_abstracts_by_key(web):
    out = T.call(web["ctx"](), "search_abstracts", {"question": "what does it say", "keys": ["journals/bbb/p7"]})
    assert out["rows"][0]["abstract"].startswith("Graph neural networks forecast")
    bad = T.call(web["ctx"](), "search_abstracts", {"question": "x", "keys": ["conf/zzz/Made1"]})
    assert bad["refused"]


def test_a_third_search_in_one_answer_is_refused(web):
    ctx = web["ctx"]()
    ctx.turn = {"content_calls": 2, "lock": threading.Lock()}
    out = T.call(ctx, "search_abstracts", {"question": "anything"})
    assert out["refused"] and web["log"] == []


def test_the_switch_puts_the_old_rule_back(ctx, monkeypatch):
    names = lambda: [s["function"]["name"] for s in T.schemas()]
    assert "search_abstracts" in names()
    prompt = agent.system_prompt(ctx)
    assert "9. Questions about what papers say" in prompt and "neither dblp nor the abstracts" in prompt
    monkeypatch.setattr(config, "CONTENT_TOOL", False)
    assert "search_abstracts" not in names()
    old = agent.system_prompt(ctx)
    assert "dblp does not have (citations, abstracts" in old and "9. Questions" not in old
    assert T.call(ctx, "search_abstracts", {"question": "x"})["refused"]


def test_five_long_abstracts_reach_the_model_whole():
    payload = {"summary": "x", "rows": [{"n": i, "abstract": "w " * 990} for i in range(1, 6)]}
    text = agent._trim(payload, T.max_chars("search_abstracts", agent.MAX_RESULT_CHARS))
    assert len(json.loads(text)["rows"]) == 5, "not cut: valid JSON with all five"
    assert T.max_chars("top_authors", agent.MAX_RESULT_CHARS) == agent.MAX_RESULT_CHARS


# --------------------------------------------------------------------------- Dewey on the benchmark

class DeweyAndJudge:
    """Dewey's router (searches abstracts with the question as written, then answers citing [1]) and
    the judge (2 for the gold answer or an answer that cites, 0 otherwise), in one scripted client."""

    def __init__(self):
        self.calls = 0

    def configured(self):
        return True

    def complete(self, messages, model, tools=None, temperature=None, extra=None):
        self.calls += 1
        usage = {"input_tokens": 100, "output_tokens": 10}
        if messages[0]["content"] == DQ.JUDGE_SYSTEM:
            user = messages[-1]["content"]
            truth = user.split("Ground truth: ")[1].split("\n")[0]
            candidate = user.split("Answer to grade: ")[1]
            score = 2 if candidate == truth or candidate.endswith("[1].") else 0
            return {"content": json.dumps({"score": score, "reason": "x"}), "usage": usage}
        if not any(m["role"] == "tool" for m in messages):
            question = messages[-1]["content"]
            return {"content": "", "usage": usage, "model": model, "tool_calls": [
                {"id": "c1", "name": "search_abstracts", "arguments": {"question": question}}]}
        found = json.loads(next(m["content"] for m in messages if m["role"] == "tool"))
        rows = found.get("rows") or []
        text = f"It is about {rows[0]['title']} [1]." if rows else "No abstract addresses it."
        return {"content": text, "usage": usage, "model": model, "tool_calls": []}

    def stream(self, messages, model, temperature=None):
        # the content writer (version 2): answers from the tool result, as the router would
        found = json.loads(next(m["content"] for m in messages if m["role"] == "tool"))
        rows = found.get("rows") or []
        yield "token", f"It is about {rows[0]['title']} [1]." if rows else "No abstract addresses it."
        yield "usage", {"input_tokens": 100, "output_tokens": 10}


DEWEY_CSV = """id,question,answer,dblp_key,semantic_scholar_id
qa1,How do graph neural networks forecast traffic?,They learn traffic flow on road networks from sensor data.,journals/bbb/p7,1
qa2,What partitioning scheme scales graph learning?,A new scheme splits billions of edges.,conf/x/B2,2
qa3,What is quantum error correction?,It protects quantum information from noise.,conf/x/C3,3
"""


def test_dewey_runs_the_benchmark_on_the_frozen_pool_and_records_what_it_did(indexed, tmp_path, monkeypatch):
    rows = DQ.parse(DEWEY_CSV)
    pools = {"qa1": {"aliases": ["journals/bbb/p7"], "candidates": {
                 "journals/bbb/p7": {"title": "Graph neural networks for traffic", "abstract": ABSTRACTS["10.1109/7"]},
                 "conf/aaa/p6": {"title": "Graph learning at scale", "abstract": ABSTRACTS["10.1109/6"]}}},
             "qa2": {"aliases": ["conf/x/B2"], "candidates": {
                 "conf/aaa/p6": {"title": "Graph learning at scale", "abstract": ABSTRACTS["10.1109/6"]}}},
             "qa3": {"aliases": ["conf/x/C3"], "candidates": {}}}
    monkeypatch.setattr(DQ, "study_dir", lambda: tmp_path)
    monkeypatch.setattr(RAG, "prepare", lambda rows, **kw: (pools, {}, {"pool": {"sha256": "P1"}, "rankers": {}}))
    monkeypatch.setattr(DQ, "fetch_abstracts", lambda rows, **kw: {
        "qa1": {"title": "Graph neural networks for traffic"}, "qa2": {"title": None}, "qa3": {"title": None}})
    ctx = agent.Ctx(data.pool, httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
                    indexed["store_meta"])
    got = DW.run(ctx, DeweyAndJudge(), rows, "sha", variant="dewey-frozen", judge_model="judge", out=lambda *_: None)

    run = next(d for d in (tmp_path / "runs").iterdir() if d.name.endswith("-dewey-frozen"))
    recs = {json.loads(x)["id"]: json.loads(x) for x in (run / "answers.jsonl").read_text(encoding="utf-8").splitlines()}
    first = recs["qa1"]
    assert first["source_in_context"] and first["source_rank"] == 1 and first["content_calls"] == 1
    assert first["query_verbatim"] and first["cited"] == [1] and first["invalid_citations"] == []
    assert first["writer"] == config.MODEL_FAST and first["score"] == 2 and first["model"] == "dewey"
    assert not recs["qa2"]["source_in_context"] and recs["qa3"]["answer"] == "No abstract addresses it."
    entry = got["results"]["dewey"]
    assert entry["called_the_abstract_search"] == 3 and entry["source_in_context"] == 1
    assert got["agent"]["content_tool"] and got["pool_sha256"] == "P1"
    assert (tmp_path / "dewey-ledger.json").exists(), "Dewey's spending goes to its own ledger"
    assert ctx.frozen_pool is None, "the frozen pool never outlives the run"


def test_source_hits_and_citations():
    payloads = [{"name": "search_abstracts", "result": {"rows": [
        {"n": 1, "key": "conf/a/X", "title": "Other", "abstract": "a"},
        {"n": 2, "key": "corr/abs-1", "title": "The Source Paper", "abstract": "b"}]}}]
    hits = DW.source_hits(payloads, {"conf/a/S"}, RAG.norm_title("The source paper."))
    assert hits["source_in_context"] and hits["source_rank"] == 2, "found under its title, as a preprint"
    none = DW.source_hits(payloads, {"conf/a/S"}, None)
    assert not none["source_listed"] and none["retrieved"] == ["conf/a/X", "corr/abs-1"]
    cites = DW.citations("A [1] and B [2] and C [7].", payloads)
    assert cites == {"cited": [1, 2, 7], "invalid": [7]}


# --------------------------------------------------------------------------- version 2

LONG = ("A long enough abstract about graph learning at scale, written so that a fallback source has "
        "something worth returning: partitioning, sampling and training on billions of edges.")


def test_v2_fills_missing_abstracts_from_semantic_scholar_then_crossref(indexed, monkeypatch):
    monkeypatch.setattr(config, "SEARCH_URL", "http://searchapi")
    monkeypatch.setattr(config, "CONTENT_FALLBACK", True)
    monkeypatch.setattr(content, "_semantic_last", [0.0])
    log = []
    base = fake_web(log)

    def handler(request):
        url = request.url
        if url.host == "api.openalex.org" and "filter" in url.params:
            log.append(url)
            return httpx.Response(200, json={"results": []})        # OpenAlex has neither abstract
        if url.host == "api.semanticscholar.org":
            log.append(url)
            ids = json.loads(request.content)["ids"]
            return httpx.Response(200, json=[{"abstract": LONG} if i == "DOI:10.1109/6" else None for i in ids])
        if url.host == "api.crossref.org":
            log.append(url)
            return httpx.Response(200, json={"message": {"abstract": f"<jats:p>Abstract {LONG} Crossref.</jats:p>"}})
        return base(request)

    ctx = agent.Ctx(data.pool, httpx.Client(transport=httpx.MockTransport(handler)), indexed["store_meta"])
    out = T.call(ctx, "search_abstracts", {"question": "graph learning at scale with fallback abstracts"})
    rows = {r["key"]: r for r in out["rows"]}
    assert rows["conf/aaa/p6"]["abstract"] == LONG, "Semantic Scholar's abstract, by DOI"
    assert rows["conf/aaa/p9"]["abstract"].startswith("A long enough") and rows["conf/aaa/p9"]["abstract"].endswith("Crossref."), \
        "Crossref's, with its markup and its 'Abstract' heading removed"
    assert [u.path for u in log if u.host == "api.crossref.org"] == ["/works/10.1109/9"], "only what is still missing"
    assert "search once more" in out["note"]

    monkeypatch.setattr(config, "CONTENT_FALLBACK", False)
    monkeypatch.setattr(config, "CONTENT_RULE_VERSION", 1)
    v1 = T.call(agent.Ctx(data.pool, httpx.Client(transport=httpx.MockTransport(handler)), indexed["store_meta"]),
                "search_abstracts", {"question": "graph learning at scale with fallback abstracts"})
    assert {r["key"]: r for r in v1["rows"]}["conf/aaa/p6"]["abstract"] == "(no abstract available)", \
        "version 1 has no fallback, and its own cache entry"
    assert "search once more" not in v1["note"]


def test_v2_answers_built_on_abstracts_are_written_by_the_stronger_model(ctx, ledger, monkeypatch):
    from chat.llm import FakeClient
    monkeypatch.setitem(T.HANDLERS, "search_abstracts", lambda ctx, **kw: {
        "summary": "1 abstract", "columns": ["n", "title", "abstract"],
        "rows": [{"n": 1, "title": "T", "key": "conf/x/T", "abstract": "A"}],
        "meta": {"sources": {"dblp-search": "ok"}}})
    call = {"id": "c1", "name": "search_abstracts", "arguments": {"question": "What is X?"}}

    monkeypatch.setattr(config, "CONTENT_WRITER", "gpt-4.1")
    client = FakeClient(script=[{"tool_calls": [call]}, {"content": "the router's draft"}], answer="X is T [1].")
    out = agent.answer(ctx, client, "What is X?", ledger=ledger)
    assert out["answer"].strip() == "X is T [1]." and out["model"] == "gpt-4.1"
    assert client.calls[-1]["kind"] == "stream" and client.calls[-1]["model"] == "gpt-4.1"

    plain = FakeClient(script=[{"tool_calls": [{"id": "c2", "name": "top_authors", "arguments": {"limit": 1}}]},
                               {"content": "Ada Alpha."}])
    assert agent.answer(ctx, plain, "who has most papers?", ledger=ledger)["model"] == config.MODEL_FAST, \
        "no abstracts read: the router's answer stands"

    monkeypatch.setattr(config, "CONTENT_WRITER", "")
    v1 = FakeClient(script=[{"tool_calls": [call]}, {"content": "the router's answer [1]."}])
    out = agent.answer(ctx, v1, "What is X?", ledger=ledger)
    assert out["answer"] == "the router's answer [1]." and out["model"] == config.MODEL_FAST, "version 1"


def test_the_v2_prompt_searches_before_declining_and_searches_again():
    v1, v2 = agent.CONTENT_RULES[1], agent.CONTENT_RULES[2]
    assert "never call such a question outside the data" in v2 and "search once more" in v2
    assert "outside the data" not in v1, "version 1 is kept as it was"


def test_a_variant_sets_its_own_content_settings_and_puts_them_back(indexed, tmp_path, monkeypatch):
    rows = DQ.parse(DEWEY_CSV)
    pools = {"qa1": {"aliases": ["journals/bbb/p7"], "candidates": {
        "journals/bbb/p7": {"title": "Graph neural networks for traffic", "abstract": ABSTRACTS["10.1109/7"]}}},
             "qa2": {"aliases": ["conf/x/B2"], "candidates": {}}, "qa3": {"aliases": ["conf/x/C3"], "candidates": {}}}
    monkeypatch.setattr(DQ, "study_dir", lambda: tmp_path)
    monkeypatch.setattr(RAG, "prepare", lambda rows, **kw: (pools, {}, {"pool": {"sha256": "P1"}, "rankers": {}}))
    monkeypatch.setattr(DQ, "fetch_abstracts", lambda rows, **kw: {q: {"title": None} for q in ("qa1", "qa2", "qa3")})
    before = {n: getattr(config, n) for n in DW.V2}
    ctx = agent.Ctx(data.pool, httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
                    indexed["store_meta"])
    got = DW.run(ctx, DeweyAndJudge(), rows, "sha", variant="dewey-v2-frozen", judge_model="judge", out=lambda *_: None)
    assert {n: getattr(config, n) for n in DW.V2} == before, "settings restored"
    assert got["agent"]["content_rule_version"] == 2 and got["agent"]["content_writer"] == "gpt-4.1"
    run = next(d for d in (tmp_path / "runs").iterdir() if d.name.endswith("-dewey-v2-frozen"))
    first = json.loads((run / "answers.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["writer"] == "gpt-4.1" and first["source_in_context"]
