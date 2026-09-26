"""
Every tool, against a fixture whose answers are written down.

These are the tests that matter most: a wrong number here becomes a confident wrong sentence in the
product, and the language model cannot catch it.
"""
from chat import tools as T
from tests.make_serving import EXPECTED, PAPERS


def call(ctx, tool, **args):
    """`tool` rather than `name`: several tools take a `name` argument of their own."""
    return T.call(ctx, tool, args)


def test_every_tool_has_a_handler_and_a_schema():
    assert set(T.HANDLERS) == {s["name"] for s in T.SPECS}
    for spec in T.SPECS:
        fn = spec["schema"]["function"]
        assert fn["description"] and len(fn["description"]) > 40    # the router reads these
        assert fn["parameters"]["additionalProperties"] is False


def test_dataset_facts(ctx):
    out = call(ctx, "dataset_facts")
    values = {r["measure"]: r["value"] for r in out["rows"]}
    assert values["publications"] == EXPECTED["papers_total"]
    assert values["author pages"] == 9
    assert values["disambiguation bins"] == 1
    assert "2026-09-01" in out["note"]


def test_resolve_author_finds_the_page_and_warns_about_a_bin(ctx):
    out = call(ctx, "resolve_author", name="Ada")
    assert out["rows"][0]["key"] == "homepages/a/Ada"
    assert out["rows"][0]["papers"] == EXPECTED["top_author"][1]

    binned = call(ctx, "resolve_author", name="Sam Same")
    assert "disambiguation bin" in binned["note"]
    assert "share this base name" in binned["note"]


def test_resolve_author_empty_is_not_zero(ctx):
    out = call(ctx, "resolve_author", name="Nobody At All")
    assert out["rows"] == []
    assert "No author page" in out["summary"]


def test_author_profile_and_papers(ctx):
    profile = call(ctx, "author_profile", key="homepages/a/Ada")
    assert profile["meta"]["stats"]["papers"] == 10
    assert profile["meta"]["coauthors"] == EXPECTED["ada_coauthors"]
    assert any(v["sid"] == "conf/aaa" for v in profile["meta"]["top_venues"])
    assert profile["link"] == {"page": "authors", "key": "homepages/a/Ada"}

    recent = call(ctx, "author_papers", key="homepages/a/Ada", frm=2020)
    assert all(r["year"] >= 2020 for r in recent["rows"])
    journals = call(ctx, "author_papers", key="homepages/a/Ada", sid="journals/bbb")
    assert {r["venue"] for r in journals["rows"]} == {"BBB Journal"}


def test_author_papers_needs_a_key_not_a_name(ctx):
    out = call(ctx, "author_papers", key="Ada Alpha")
    assert out["rows"] == []
    assert "resolve_author" in out["note"]


def test_top_authors_global_and_scoped(ctx):
    top = call(ctx, "top_authors", limit=3)
    assert (top["rows"][0]["name"], top["rows"][0]["papers"]) == EXPECTED["top_author"]
    assert "bins are excluded" in top["note"].lower()

    scoped = call(ctx, "top_authors", sid="conf/ccc", limit=5)
    assert scoped["rows"][0]["name"] in {"Cleo Gamma", "Dan Delta"}
    assert all(r["papers"] >= 1 for r in scoped["rows"])

    coauthors = call(ctx, "top_authors", metric="coauthors", limit=3)
    assert coauthors["rows"][0]["coauthors"] >= 1


def test_top_venues(ctx):
    out = call(ctx, "top_venues", limit=3)
    assert out["rows"][0]["papers"] >= out["rows"][-1]["papers"]
    journals = call(ctx, "top_venues", kind="journal", limit=5)
    assert {r["kind"] for r in journals["rows"]} == {"journal"}
    windowed = call(ctx, "top_venues", frm=2020, to=2025, limit=5, min_papers=1)
    assert windowed["rows"], "a windowed leaderboard must still return rows"


def test_most_shared_names(ctx):
    out = call(ctx, "most_shared_names", limit=5)
    assert out["rows"][0]["base_name"] == "Sam Same"
    assert out["rows"][0]["people"] == EXPECTED["sam_numbered"]


def test_count_papers_filters(ctx):
    everything = call(ctx, "count_papers")
    assert everything["meta"]["total"] == EXPECTED["papers_total"]

    recent = call(ctx, "count_papers", frm=2020)
    expected = sum(1 for p in PAPERS if p[1] >= 2020)
    assert recent["meta"]["total"] == expected

    preprints = call(ctx, "count_papers", kind="preprint")
    assert preprints["meta"]["total"] == sum(1 for p in PAPERS if p[3])

    by_author = call(ctx, "count_papers", author_key="homepages/a/Ada")
    assert by_author["meta"]["total"] == 10

    big = call(ctx, "count_papers", min_authors=3)
    assert big["meta"]["total"] == sum(1 for p in PAPERS if len(p[2]) >= 3)

    unidentified = call(ctx, "count_papers", with_unidentified_author=True)
    assert unidentified["meta"]["total"] == EXPECTED["sam_bin_records"]


def test_timeseries_and_terms(ctx):
    series = call(ctx, "papers_timeseries", metric="papers", frm=2010, to=2025)
    assert series["rows"][0]["year"] == 2010
    assert sum(r["value"] for r in series["rows"]) == sum(
        1 for p in PAPERS if not p[3] and 2010 <= p[1] <= 2025)

    authors = call(ctx, "papers_timeseries", metric="mean_authors")
    assert authors["meta"]["peak"]["value"] >= 1

    unknown = call(ctx, "papers_timeseries", metric="citations")
    assert unknown["refused"] is True

    terms = call(ctx, "title_terms", terms=["graph", "cloud"])
    early = next(r for r in terms["rows"] if r["year"] == 2010)
    late = next(r for r in terms["rows"] if r["year"] == 2024)
    assert early["cloud"] > 0 and early["graph"] == 0
    assert late["graph"] > 0


def test_rising_words(ctx):
    out = call(ctx, "rising_words", old_from=2010, new_from=2021, min_titles=1, limit=5)
    assert out["rows"], "the fixture has enough titles for a comparison"
    assert "change_x" in out["columns"]


def test_namesakes(ctx):
    out = call(ctx, "namesakes", name="Sam Same")
    kinds = [r["page_kind"] for r in out["rows"]]
    assert kinds.count("numbered") == EXPECTED["sam_numbered"]
    assert "disambiguation" in kinds
    assert "unassigned records" in out["summary"]


def test_coauthors_and_pairs(ctx):
    out = call(ctx, "coauthors", key="homepages/a/Ada")
    names = {r["name"]: r["papers_together"] for r in out["rows"]}
    assert names["Ben Beta"] == 4          # three papers plus the 2024 preprint

    filtered = call(ctx, "coauthors", key="homepages/a/Ada", sid="conf/ccc")
    assert set(filtered["rows"][0].keys()) >= {"key", "name"}
    assert all(r["name"] in {"Cleo Gamma", "Dan Delta"} for r in filtered["rows"])

    pair = call(ctx, "pair_papers", key_a="homepages/a/Ada", key_b="homepages/b/Ben")
    assert len(pair["rows"]) == EXPECTED["ada_and_ben_together"]
    none = call(ctx, "pair_papers", key_a="homepages/a/Ada", key_b="homepages/d/Dan")
    assert len(none["rows"]) == 1          # the 2018 three-author paper


def test_authors_in_both(ctx):
    out = call(ctx, "authors_in_both", sid_a="conf/aaa", sid_b="journals/bbb")
    assert {r["name"] for r in out["rows"]} >= EXPECTED["in_both_aaa_and_bbb"]


def test_venue_tools(ctx):
    resolved = call(ctx, "resolve_venue", name="AAA")
    assert resolved["rows"][0]["sid"] == "conf/aaa"
    missing = call(ctx, "resolve_venue", name="Nonexistent Venue")
    assert missing["rows"] == []

    profile = call(ctx, "venue_profile", sid="conf/aaa")
    assert profile["meta"]["series"]["kind"] == "conference"
    assert profile["meta"]["top_authors"][0]["name"] == "Ada Alpha"


def test_paper_detail_resolves_authors(ctx):
    out = call(ctx, "paper_detail", title="Graph neural networks for traffic")
    assert out["meta"]["paper"]["year"] == 2020
    assert out["rows"][0]["name"] == "Ada Alpha"
    assert out["rows"][0]["page_kind"] == "regular"

    binned = call(ctx, "paper_detail", title="Secure graph release")
    kinds = {r["page_kind"] for r in binned["rows"]}
    assert "disambiguation" in kinds


def test_search_papers_falls_back_when_the_service_is_down(ctx):
    out = call(ctx, "search_papers", q="graph kernels")
    assert out["rows"], "the exact-word fallback must still answer"
    assert "exact-word" in out["summary"] or "word match" in out["summary"]


def test_model_tools_refuse_when_upstream_is_unreachable(ctx):
    assert call(ctx, "predict_venue", title="Graph learning for traffic")["refused"] is True
    assert call(ctx, "predict_coauthors", author_key="homepages/a/Ada")["refused"] is True
    assert call(ctx, "model_cards")["refused"] is True


def test_docs_lookup(ctx):
    out = call(ctx, "docs_lookup", question="what is a disambiguation bin?")
    assert "Disambiguation bins" in [r["topic"] for r in out["rows"]]
    limits = call(ctx, "docs_lookup", question="most cited paper impact factor")
    assert any("citation" in r["text"].lower() for r in limits["rows"])


def test_run_sql_is_guarded(ctx):
    ok = call(ctx, "run_sql", sql="SELECT count(*) AS n FROM pubs")
    assert ok["rows"][0]["n"] == EXPECTED["papers_total"]
    assert ok["meta"]["sql"].startswith("SELECT")

    for bad in ["DROP TABLE pubs", "SELECT 1; DROP TABLE pubs",
                "SELECT * FROM read_parquet('/etc/passwd')", "INSERT INTO pubs VALUES (1)"]:
        assert call(ctx, "run_sql", sql=bad).get("refused") is True, bad


def test_unknown_tool_and_bad_arguments_are_reported(ctx):
    assert T.call(ctx, "no_such_tool", {})["refused"] is True
    assert T.call(ctx, "author_profile", {"nonsense": 1})["refused"] is True


# --------------------------------------------------------------------------- the parquet mount
# `src` in the serving database is a VIEW over dblp.parquet, so any query touching it needs that file
# mounted. The fixture has src as a plain table, which hid this on the way in: the live store build
# skipped its "dataset facts" step because the container had no /data mount. These two tests pin the
# behaviour rather than the mount.
class _NoSrcCursor:
    """A cursor that fails exactly as an unreachable parquet does."""

    def __init__(self, inner):
        self.inner = inner

    def execute(self, sql, params=None):
        if "s.src" in sql:
            raise RuntimeError("IO Error: No files found that match the pattern '/data/parquet/dblp.parquet'")
        return self.inner.execute(sql, params) if params else self.inner.execute(sql)

    @property
    def description(self):
        return self.inner.description

    def fetchall(self):
        return self.inner.fetchall()


def test_the_leaderboard_store_never_depends_on_the_parquet():
    from chat import store
    for name, sql in store.STEPS:
        assert "s.src" not in sql, f"the '{name}' step would fail without the parquet mounted"


def test_paper_detail_falls_back_to_the_registry_without_the_parquet(ctx):
    class Blind(type(ctx)):
        def cursor(self):
            return _NoSrcCursor(super().cursor())

    blind = Blind(ctx.pool, ctx.http, ctx.store_meta)
    out = T.call(blind, "paper_detail", {"title": "Graph neural networks for traffic"})
    assert out["meta"]["paper"]["year"] == 2020
    assert [r["name"] for r in out["rows"]] == ["Ada Alpha"]
    assert "registry" in out["note"]


def test_dataset_facts_still_works_without_the_parquet(ctx):
    class Blind(type(ctx)):
        def cursor(self):
            return _NoSrcCursor(super().cursor())

    out = T.call(Blind(ctx.pool, ctx.http, ctx.store_meta), "dataset_facts", {})
    values = {r["measure"]: r["value"] for r in out["rows"]}
    assert values["publications"] == EXPECTED["papers_total"]
