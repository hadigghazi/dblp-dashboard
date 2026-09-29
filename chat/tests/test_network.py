"""
The network tools, against a centrality run whose answers are written down.

Three ways these tools could be wrong while reading fine, each with a test:

  * **refusing forever.** The first version looked the dump's fingerprint up through an attribute the
    context does not have, so every call would have said "not measured yet" - indistinguishable from
    the honest version of that message. The first test here is the one that catches it.
  * **answering from another snapshot.** A run for a different dump has different ids and a different
    graph; it must be refused, not used.
  * **mixing up central and prolific.** A rank is reported, never just a raw score, and the note says
    which kind of ranking it is.
"""
import json

import duckdb
import pytest

from chat import config, tools as T

FINGERPRINT = "testfp000001"   # make_serving's default

# (key, name, records, degree, core, pagerank, eigenvector, betweenness, closeness, in_largest)
# Cleo is the bridge: few co-authors, highest betweenness. Ada is the hub: most co-authors. Fay sits
# outside the largest component, so three of her measures are not defined.
AUTHORS = [
    ("homepages/a/Ada", "Ada Alpha", 12, 5, 3, 0.30, 0.50, 0.010, 0.40, True),
    ("homepages/b/Ben", "Ben Beta", 6, 3, 2, 0.20, 0.40, 0.020, 0.35, True),
    ("homepages/c/Cleo", "Cleo Gamma", 4, 2, 2, 0.15, 0.30, 0.090, 0.45, True),
    ("homepages/d/Dan", "Dan Delta", 3, 2, 1, 0.10, 0.20, 0.005, 0.30, True),
    ("homepages/f/Fay", "Fay Zeta", 2, 1, 1, 0.05, None, None, None, False),
]
LARGEST = sum(1 for a in AUTHORS if a[9])


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: (-(values[i] if values[i] is not None else -1), i))
    ranks = [0] * len(values)
    for position, i in enumerate(order, start=1):
        ranks[i] = position if values[i] is not None else 0
    return ranks


def _write_run(models, fingerprint):
    """A centrality run in the layout `ml.network.cli centrality` writes."""
    out = models / f"centrality-{fingerprint}"
    out.mkdir(parents=True, exist_ok=True)
    columns = list(zip(*AUTHORS))
    ranks = {m: _rank(list(columns[i])) for m, i in
             (("degree", 3), ("eigenvector", 6), ("betweenness", 7), ("closeness", 8))}
    con = duckdb.connect()
    con.execute("""CREATE TABLE c (person_id BIGINT, key VARCHAR, name VARCHAR, records BIGINT,
                   degree BIGINT, core INTEGER, pagerank DOUBLE, eigenvector DOUBLE,
                   betweenness DOUBLE, closeness DOUBLE, rank_degree BIGINT, rank_eigenvector BIGINT,
                   rank_betweenness BIGINT, rank_closeness BIGINT, in_largest_component BOOLEAN)""")
    for i, a in enumerate(AUTHORS):
        con.execute("INSERT INTO c VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [i + 1, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8],
                     ranks["degree"][i], ranks["eigenvector"][i], ranks["betweenness"][i],
                     ranks["closeness"][i], a[9]])
    con.execute(f"COPY c TO '{(out / 'dblp-coauthor.centrality.parquet').as_posix()}' (FORMAT parquet)")
    con.close()
    (out / "metrics.json").write_text(json.dumps({
        "nodes": len(AUTHORS), "edges": 6, "average_degree": 2.4, "max_degree": 5,
        "components": 2, "largest_component_nodes": LARGEST,
        "largest_component_share": LARGEST / len(AUTHORS),
        "approx_global_clustering_coefficient": 0.71,
        "largest_component_diameter": {"lower": 3, "upper": 3},
        "parameters": {"betweenness": {"samples": 4096}, "closeness": {"samples": 1024}},
        "closeness_spot_check": {"sampled_authors": 200, "matching_convention": "c",
                                 "by_convention": {"c": {"p90_relative_error": 0.0075}}},
    }), encoding="utf-8")
    return out


@pytest.fixture
def run(loaded):
    out = _write_run(config.MODELS_DIR, FINGERPRINT)
    yield out
    for f in out.iterdir():
        f.unlink()
    out.rmdir()


def call(ctx, tool, **args):
    return T.call(ctx, tool, args)


def test_the_tools_find_the_run_for_this_dump(ctx, run):
    """If the fingerprint lookup is wrong, every tool refuses with a message that reads exactly like
    the honest "not measured yet". This is the test that tells the two apart."""
    got = call(ctx, "network_shape")
    assert not got.get("refused"), got
    assert got["rows"], got


def test_the_network_shape_reports_the_small_world_facts(ctx, run):
    got = call(ctx, "network_shape")
    values = {r["measure"]: r["value"] for r in got["rows"]}
    assert values["Separate groups that never connect"] == "2"
    assert values["Share of authors in the single largest group"] == "80.0%"
    assert values["Degrees of separation across that group"].startswith("3 hops")
    assert "small-world" in got["note"]


def test_the_bridge_outranks_the_hub_on_betweenness(ctx, run):
    """Cleo has the fewest co-authors among the connected authors and the highest betweenness; Ada is
    the opposite. A tool that quietly ranked by degree would put Ada first."""
    between = call(ctx, "central_authors", metric="betweenness", limit=5)
    degree = call(ctx, "central_authors", metric="degree", limit=5)
    assert between["rows"][0]["name"] == "Cleo Gamma"
    assert degree["rows"][0]["name"] == "Ada Alpha"
    assert "not output" in between["note"] or "not the most prolific" in between["note"]


def test_a_sampled_measure_says_it_was_sampled(ctx, run):
    note = call(ctx, "central_authors", metric="betweenness")["note"]
    assert "4,096 sampled sources" in note and "not computed exactly" in note
    closeness = call(ctx, "central_authors", metric="closeness")["note"]
    assert "0.8%" in closeness, "the measured error, not a promised one, goes in the note"


def test_an_author_outside_the_main_component_is_not_ranked_rather_than_last(ctx, run):
    between = call(ctx, "central_authors", metric="betweenness", limit=10)
    assert "Fay Zeta" not in [r["name"] for r in between["rows"]]
    fay = call(ctx, "author_centrality", key="homepages/f/Fay")
    meanings = {r["measure"]: r["meaning"] for r in fay["rows"]}
    assert "outside the largest connected component" in meanings["betweenness"]
    assert "more central than" in meanings["degree"], "degree is defined for everybody"


def test_an_author_gets_a_rank_and_a_percentile_not_a_bare_score(ctx, run):
    got = call(ctx, "author_centrality", key="homepages/c/Cleo")
    rows = {r["measure"]: r for r in got["rows"]}
    assert rows["betweenness"]["rank"] == f"1 of {LARGEST:,}"
    assert "more central than" in rows["betweenness"]["meaning"]
    assert "Cleo Gamma" in got["summary"]


def test_an_unknown_author_is_refused_with_a_reason(ctx, run):
    got = call(ctx, "author_centrality", key="homepages/z/Nobody")
    assert got.get("refused")
    assert "resolve_author" in got.get("instead", ""), "the refusal should say what to do instead"


def test_a_run_for_another_snapshot_is_not_used(ctx, loaded):
    """Different dump, different renumbering, different graph. Answering from it would attach one
    snapshot's ranks to another snapshot's people."""
    stale = _write_run(config.MODELS_DIR, "someotherdump")
    try:
        got = call(ctx, "network_shape")
        assert got.get("refused"), "a run for another dump must not answer for this one"
    finally:
        for f in stale.iterdir():
            f.unlink()
        stale.rmdir()


def test_an_unknown_measure_is_refused_not_interpolated(ctx, run):
    """The measure name goes into SQL, so anything outside the known list has to stop before it."""
    got = call(ctx, "central_authors", metric="degree; DROP TABLE x")
    assert got.get("refused")


def test_prolific_and_central_are_different_tools(ctx, run):
    """The routing depends on the descriptions, so they have to say which question is which."""
    central = T.spec("central_authors")["description"]
    assert "top_authors" in central and "not by" in central.lower()


def test_rows_are_keyed_by_column_like_every_other_tool(ctx, run):
    """The assistant panel renders a cell as row[column]. A tool returning lists would show a table of
    blank cells - which two of these did, before this test."""
    for tool, args in (("network_shape", {}), ("central_authors", {"metric": "degree"}),
                       ("author_centrality", {"key": "homepages/a/Ada"})):
        got = call(ctx, tool, **args)
        assert got["rows"], tool
        for row in got["rows"]:
            assert isinstance(row, dict), tool
            assert set(got["columns"]) <= set(row), tool
