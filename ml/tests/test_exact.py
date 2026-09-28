"""
The two shortcuts, against a graph whose betweenness can be worked out on paper.

A lollipop: a triangle with a tail. Node 30 joins the triangle to the tail and node 40 sits in the
middle of everything, so the ordering is 40 first, then 30 and 50, then 60, then the three leaves at
zero. Nothing about that ranking is a matter of opinion, which is what makes it a test.

What is being checked is not the graph algorithms - the library computes those - but the three
comparisons this module exists to make, and the claim each one supports:

  * sampling the sources, against exact, on the *same* graph: the estimate can be trusted;
  * exact on a subgraph, against the full-graph estimate for the *same authors*: shrinking the graph
    changes the answer;
  * ego betweenness, likewise: the cheapest shortcut changes it further.

On a seven-node graph the first comparison should be near perfect, because sixty-four sampled
sources out of seven is effectively exhaustive. If that one ever drops, the wiring is wrong.
"""
import json

import numpy as np
import pytest

from tests import test_pipeline as tp  # noqa: F401  (sets env)
from tests.test_centrality import write_members  # noqa: E402  (the export's own file layout)
from ml.network import centrality as CE, config as NC, exact as XB  # noqa: E402

# a triangle (10, 20, 30) with a tail 30-40-50-60-70
EDGES = [(10, 20, 1), (10, 30, 1), (20, 30, 1), (30, 40, 1), (40, 50, 1), (50, 60, 1), (60, 70, 1)]
NODES = [(i, f"homepages/x/{i}", f"Author {i}", 3, 2000, 2020) for i in (10, 20, 30, 40, 50, 60, 70)]
MOST_BETWEEN = 40


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    source = tmp_path_factory.mktemp("network-lollipop")
    write_members(source / f"{NC.NAME}.ungraph.txt.gz",
                  [f"Nodes: {len(NODES)} Edges: {len(EDGES)}"], EDGES)
    write_members(source / f"{NC.NAME}.nodes.txt.gz", ["NodeId\tdblpKey\tName"], NODES)
    out = tmp_path_factory.mktemp("centrality-lollipop")
    CE.run(source, out, threads=1, seed=7, betweenness_samples=64, closeness_samples=4)
    payload = XB.run(source, out, out, max_nodes=100, ego_per_band=7, ego_top=3, threads=1, seed=7)
    return payload, out, source


def test_the_fixtures_premise_holds(measured):
    """If the graph's most-between author were not who it is meant to be, every comparison below
    would be measuring the wrong thing and still look fine."""
    payload, out, _ = measured
    top = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["top"]["betweenness"][0]
    assert top["key"] == f"homepages/x/{MOST_BETWEEN}"
    assert payload["graph"]["largest_component_nodes"] == len(NODES)


def test_sampling_is_checked_against_exact_on_the_same_graph(measured):
    """Sixty-four sampled sources out of seven is exhaustive, so the estimator has nowhere to hide.
    This is the comparison that says the full-graph estimate can be trusted."""
    payload, _, _ = measured
    for key in ("same_sample_count", "same_sampling_rate"):
        got = payload["sampling_against_exact"][key]
        assert got["spearman"] is not None and got["spearman"] > 0.9, (key, got)
        assert got["same_most_central_author"] is True, (key, got)


def test_the_two_shortcuts_are_compared_against_the_full_graph(measured):
    """Both comparisons must be reported, over the right number of authors. Whether they agree is a
    finding, not a requirement - so the test checks that the numbers exist and are numbers."""
    payload, _, _ = measured
    shrink, ego = payload["shrinking_the_graph"], payload["ego_networks"]
    assert shrink["authors"] == payload["exact_subgraph"]["nodes"]
    assert -1.0001 <= shrink["spearman"] <= 1.0001
    assert ego["authors"] == len(NODES), "seven per band plus three hubs, deduplicated, is everybody"
    assert -1.0001 <= ego["pooled"]["spearman"] <= 1.0001
    assert ego["degree_range"][0] >= 1


def test_the_ego_comparison_is_not_carried_by_the_degree_spread(measured):
    """A pool spanning every degree correlates well with almost anything, because betweenness rises
    with degree. The within-band figures are the ones that mean something, so they have to be there
    whenever a band has enough authors in it."""
    payload, _, _ = measured
    ego = payload["ego_networks"]
    assert "within_degree_bands" in ego
    assert "pooled" in ego and ego["pooled"]["authors"] == ego["authors"]
    for label, got in ego["within_degree_bands"].items():
        assert got["authors"] >= 10, label
        assert got["spearman"] is None or -1.0001 <= got["spearman"] <= 1.0001


def test_exact_betweenness_is_run_on_a_subgraph_that_fits(measured):
    payload, _, _ = measured
    core = payload["exact_subgraph"]
    assert 2 < core["nodes"] <= 100, core
    assert core["exact_seconds"] >= 0
    assert core["projected_days_for_the_whole_component"] >= 0


def test_a_missing_value_is_dropped_rather_than_poisoning_the_correlation():
    """One NaN turns a correlation into NaN, which reports nothing at all. The pair is dropped and
    the count of dropped pairs is published."""
    got = XB.agreement(np.array([3.0, 2.0, np.nan, 1.0]), np.array([3.0, 2.0, 5.0, 1.0]))
    assert got["authors"] == 3
    assert got["authors_without_a_value"] == 1
    assert got["spearman"] == 1.0


def test_the_report_says_why_the_cheaper_route_was_not_taken(measured):
    payload, out, _ = measured
    text = (out / "EXACT-BETWEENNESS.md").read_text(encoding="utf-8")
    assert "different question" in text
    assert "all pairs" in text, "the reason a subgraph changes the answer has to be stated"
    assert str(payload["exact_subgraph"]["k"]) in text
    assert "Everett and Borgatti" in text, "the ego-betweenness claim needs its source"
    assert "reads better than it deserves to" in text, "the pooled figure must carry its caveat"
