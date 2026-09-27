"""
Centrality, against graphs whose answers are known in advance.

The real run is 3.99 million authors and 24.4 million edges, where nothing can be checked by eye.
So the checking happens here: on a graph with one obvious bridge, whose most-between node is not a
matter of opinion; on a graph with two components, where the measures that are only defined on one
of them must say so rather than say zero; and end to end on the synthetic dump, which is the only
place the numpy-to-DuckDB handover, the file formats and the datasheet all run together.

Two of these tests exist because of specific ways this could be wrong and look right:

  * a sampling algorithm that silently reported the wrong scale would still rank people plausibly,
    so closeness is compared against exact shortest paths;
  * a mistake renumbering sparse dblp ids to 0..n-1 would attach the right scores to the wrong
    authors, so degree is compared between the graph library and the source file.
"""
import json

import networkit as nk
import numpy as np
import pytest

from tests import test_pipeline as tp  # noqa: F401  (sets env, builds the serving db)
from ml import data  # noqa: E402
from ml.network import centrality as CE, config as NC, export as EX  # noqa: E402

# imported plainly, not through importorskip: a native extension that installs but cannot import is
# exactly the failure that must turn CI red rather than skip these tests


@pytest.fixture(scope="module")
def con():
    c, meta = data.connect()
    yield c, meta
    c.close()


@pytest.fixture(scope="module")
def exported(con, tmp_path_factory):
    c, meta = con
    out = tmp_path_factory.mktemp("net")
    payload = EX.export(c, meta, out)
    return payload, out


def clique(G, nodes):
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            G.addEdge(a, b)
    return G


def barbell(extra_component=False):
    """Two cliques of five joined through one node. Node 10 is on every shortest path between the
    halves, so it must win betweenness; it is nobody's most prolific co-author, so it must not win
    degree. If those two ever agree on this graph, the implementation is not measuring what it says."""
    G = nk.Graph(13 if extra_component else 11, False, False)
    clique(G, list(range(5)))
    clique(G, list(range(5, 10)))
    G.addEdge(0, 10)
    G.addEdge(10, 5)
    if extra_component:
        G.addEdge(11, 12)
    return G


def test_the_bridge_wins_betweenness_and_does_not_win_degree():
    G = barbell()
    scores, _, _, summary = CE.measures(G, threads=1, seed=7, betweenness_samples=64,
                                        closeness_samples=8)
    exact = np.asarray(nk.centrality.Betweenness(G, True).run().scores())
    assert int(np.argmax(exact)) == 10, "the fixture's premise: node 10 is the bridge"
    assert int(np.nanargmax(scores["betweenness"])) == 10
    assert int(np.argmax(scores["degree"])) != 10, "the bridge has two co-authors; degree is not this"
    assert scores["degree"][10] == 2
    assert summary["nodes"] == 11 and summary["components"] == 1


def test_a_measure_defined_on_one_component_says_so_instead_of_saying_zero():
    """Zero betweenness and zero closeness mean "measured, and peripheral". An author in another
    component was not measured at all, which is a different statement and must not read as the
    first one."""
    G = barbell(extra_component=True)
    scores, _, back, summary = CE.measures(G, threads=1, seed=7, betweenness_samples=32,
                                           closeness_samples=4)
    assert summary["components"] == 2
    assert summary["largest_component_nodes"] == 11
    for measure in ("closeness", "eigenvector", "betweenness"):
        assert np.isnan(scores[measure][11]) and np.isnan(scores[measure][12]), measure
        assert np.isfinite(scores[measure][back]).all(), measure
    # degree and PageRank are defined everywhere, so they are given everywhere
    assert scores["degree"][11] == 1 and np.isfinite(scores["pagerank"][11])


def test_an_unmeasured_author_is_unranked_rather_than_last():
    ranks = CE._ranks(np.array([5.0, np.nan, 7.0, 1.0]))
    assert ranks.tolist() == [2, 0, 1, 3]


def test_the_closeness_spot_check_measures_the_error_it_promises():
    """On a graph this small the sample is every node, so the approximation has nowhere to hide: the
    ratio against exact shortest paths has to be 1."""
    G = barbell()
    scores, H, _, summary = CE.measures(G, threads=1, seed=7, betweenness_samples=32,
                                        closeness_samples=10)
    check = summary["closeness_spot_check"]
    assert check["sampled_authors"] == H.numberOfNodes()
    assert check["matching_convention"], check
    best = check["by_convention"][check["matching_convention"]]
    assert abs(best["median_ratio"] - 1) < 0.05, check
    assert best["p90_relative_error"] < 0.25, check


def test_more_samples_than_nodes_are_not_requested_of_the_library():
    """The defaults are sized for 3.8 million authors. Handed a small graph they would ask for more
    sampled sources than there are sources, which the library refuses."""
    G = barbell()
    _, _, _, summary = CE.measures(G, threads=1, seed=7,
                                   betweenness_samples=CE.BETWEENNESS_SAMPLES,
                                   closeness_samples=CE.CLOSENESS_SAMPLES)
    used = summary["parameters"]
    assert used["closeness"]["samples"] <= G.numberOfNodes()
    assert used["betweenness"]["samples"] <= G.numberOfNodes()


def test_correlations_are_reported_for_every_pair():
    G = barbell(extra_component=True)
    scores, _, back, _ = CE.measures(G, threads=1, seed=7, betweenness_samples=32, closeness_samples=4)
    out = CE.correlations(scores, back, top=3)
    assert len(out["spearman"]) == 15, "six measures make fifteen pairs"
    assert all(-1.0001 <= v <= 1.0001 for v in out["spearman"].values())
    assert all(0 <= v <= 3 for v in out["top_overlap"].values())


# --------------------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def measured(exported, tmp_path_factory):
    payload, source = exported
    out = tmp_path_factory.mktemp("centrality")
    return CE.run(source, out, threads=1, seed=7, betweenness_samples=64, closeness_samples=8), out, payload


def test_the_published_files_are_measured_not_the_database(measured):
    """The run reads the export's own files. If it quietly fell back to the serving database this
    count could differ from the dataset's."""
    got, _, exported_payload = measured
    assert got["nodes"] == exported_payload["nodes_with_an_edge"]
    assert got["edges"] == exported_payload["edges"]
    assert got["rows"] == got["nodes"], "one row per author in the graph"


def test_a_truncated_dataset_is_an_error_not_a_smaller_graph(exported, tmp_path):
    """The header says how many edges the file has. A reader that stopped early - at the end of the
    first member of a multi-member gzip stream, say - would report a smaller graph and no error."""
    _, source = exported
    plain = CE.decompress(source / f"{NC.NAME}.ungraph.txt.gz", tmp_path / "edges.tsv")
    con = data.plain_connection()
    try:
        with pytest.raises(AssertionError, match="truncated"):
            CE.load_edges(con, plain, expect_edges=10 ** 9)
    finally:
        con.close()


def test_the_header_counts_are_read_back(exported):
    _, source = exported
    nodes, edges = CE.header_counts(source / f"{NC.NAME}.ungraph.txt.gz")
    assert nodes and edges


def test_degree_agrees_between_the_graph_and_the_file(measured):
    got, _, _ = measured
    assert got["degree_cross_check_authors"] > 0, "the cross-check must actually have run"


def test_the_table_is_readable_and_joins_back_to_dblp(measured, con):
    got, out, _ = measured
    c, _ = con
    parquet = out / f"{NC.NAME}.centrality.parquet"
    rows = c.execute(f"SELECT count(*), count(name), max(degree) FROM read_parquet('{parquet.as_posix()}')").fetchone()
    assert rows[0] == got["rows"]
    assert rows[1] == got["rows"], "every author in the graph has a name from the nodes file"
    assert rows[2] == got["max_degree"]
    keys = c.execute(f"""
        SELECT count(*) FROM read_parquet('{parquet.as_posix()}') p
        JOIN s.persons pe ON pe.person_id = p.person_id""").fetchone()[0]
    assert keys == got["rows"], "person_id must still be a dblp author page"


def test_ranks_start_at_one_and_have_no_gaps(measured):
    got, out, _ = measured
    import duckdb
    parquet = out / f"{NC.NAME}.centrality.parquet"
    low, high, distinct = duckdb.execute(f"""
        SELECT min(rank_degree), max(rank_degree), count(DISTINCT rank_degree)
        FROM read_parquet('{parquet.as_posix()}')""").fetchone()
    assert low == 1 and high == got["rows"] and distinct == got["rows"]


def test_the_datasheet_states_what_is_exact_and_what_is_not(measured):
    got, out, _ = measured
    text = (out / "README.md").read_text(encoding="utf-8")
    assert "Betweenness | **no**" in text, "an estimate must not be presented as exact"
    assert "Degree | yes" in text
    assert str(got["parameters"]["betweenness"]["samples"]) in text
    assert "hops" in text, "the datasheet must say the weights are ignored"
    for measure in ("degree", "betweenness", "closeness", "eigenvector"):
        assert got["top"][measure], measure


def test_the_metrics_file_holds_the_parameters_and_the_checks(measured):
    got, out, _ = measured
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved["parameters"]["seed"] == 7
    assert saved["closeness_spot_check"]["exact_sssp_runs"] > 0
    assert set(saved["seconds"]) >= {"degree", "betweenness", "closeness", "eigenvector"}
    assert saved["rules"].get("scope") == "all", "the dataset's own rules travel with the measures"
    assert saved == got


def test_the_full_corpus_export_is_preferred_over_a_reproduction(tmp_path):
    """`network-<fp>` is the full graph; `network-<fp>-no-preprints` exists to reproduce somebody
    else's number. Measuring the second one by accident would be a quiet mistake."""
    for name in ("network-52855f098520", "network-52855f098520-no-preprints"):
        d = tmp_path / name
        d.mkdir()
        (d / f"{NC.NAME}.ungraph.txt.gz").write_bytes(b"")
    assert CE.find_export(tmp_path).name == "network-52855f098520"

    with pytest.raises(FileNotFoundError, match="no exported network"):
        CE.find_export(tmp_path / "empty")
