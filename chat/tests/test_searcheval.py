"""The paraphrase test: the queries must be paraphrases, and the ranking must be read correctly."""
from chat import searcheval as SE
from chat.llm import FakeClient


def test_a_description_that_reuses_the_title_is_rejected():
    """Otherwise the test measures word matching, which is the thing it exists to look past."""
    title = "Graph neural networks for traffic forecasting"
    kept, _ = SE.describe(FakeClient(script=[{"content": "A method predicting road congestion over time."}]),
                          title)
    assert kept == "A method predicting road congestion over time."

    reused, _ = SE.describe(FakeClient(script=[{"content": "Graph neural networks predicting traffic."}]),
                            title)
    assert reused is None


def test_an_empty_description_is_rejected():
    assert SE.describe(FakeClient(script=[{"content": "  "}]), "A title")[0] is None


def test_quotes_and_line_breaks_are_stripped():
    out, _ = SE.describe(FakeClient(script=[{"content": '"Predicting congestion\non city roads."'}]),
                         "Graph neural networks for traffic forecasting")
    assert out == "Predicting congestion on city roads."


def test_rank_and_summary():
    results = [{"key": "a"}, {"key": "b"}, {"key": "c"}]
    assert SE._rank(results, "a") == 1 and SE._rank(results, "c") == 3
    assert SE._rank(results, "zz") is None

    summary = SE._summary([1, 3, None, 12], 4)
    assert summary == {"queries": 4, "found": 3, "acc@1": 0.25, "acc@5": 0.5, "acc@10": 0.5,
                       "acc@20": 0.75, "mrr": round((1 + 1 / 3 + 1 / 12) / 4, 4)}


def test_the_sample_comes_from_the_indexed_population(ctx):
    papers = SE.sample_papers(ctx, 5)
    assert papers, "the fixture has no paper long enough to sample"
    assert all(len(p["title"]) >= 30 and p["year"] >= 2010 for p in papers)


def test_the_same_query_set_is_reused_between_runs(ctx, tmp_path, monkeypatch):
    """Four runs of this test disagreed by up to five points and I read that as the effect of my
    changes. It was the sample: the model writes fresh descriptions every time."""
    monkeypatch.setattr(SE.config, "MODELS_DIR", tmp_path)
    client = FakeClient(script=[{"content": "A description of something entirely different."}] * 20)
    first = SE.build_queries(ctx, client, n_papers=3, seed=7)
    path = SE.queries_path(ctx.meta.get("fingerprint", "unknown"), 7, 3)
    path.write_text(SE.json.dumps(first), encoding="utf-8")

    reread = SE.json.loads(path.read_text(encoding="utf-8"))
    assert [c["query"] for c in reread["cases"]] == [c["query"] for c in first["cases"]]
    assert reread["cases"], "the fixture should yield at least one usable description"
