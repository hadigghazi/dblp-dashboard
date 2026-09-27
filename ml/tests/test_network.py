"""
The network export, against the synthetic serving database.

The point of these tests is the file format and the counting rules: a course, a paper or another
tool will read these files without ever seeing the code, so an edge listed twice, a header that
breaks a parser, or a bin quietly counted as a person are all silent failures.
"""
import gzip

import pytest

from tests import test_pipeline as tp  # noqa: F401  (sets env, builds the serving db)
from ml import data  # noqa: E402
from ml.network import config as NC, export as EX  # noqa: E402


@pytest.fixture(scope="module")
def con():
    c, meta = data.connect()
    yield c, meta
    c.close()


@pytest.fixture
def exported(con, tmp_path):
    c, meta = con
    payload = EX.export(c, meta, tmp_path / "net")
    return payload, tmp_path / "net"


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return fh.read().splitlines()


def edges_of(lines):
    return [tuple(line.split("\t")) for line in lines if not line.startswith("#")]


def test_the_header_is_comments_then_tab_separated_edges(exported):
    payload, out = exported
    lines = read(out / f"{NC.NAME}.ungraph.txt.gz")
    header = [line for line in lines if line.startswith("#")]
    assert len(header) >= 5
    assert any("Nodes:" in line and "Edges:" in line for line in header)
    assert header[-1].endswith("FromNodeId\tToNodeId\tPapersTogether")
    for row in edges_of(lines):
        assert len(row) == 3 and row[0].isdigit() and row[1].isdigit()


def test_every_edge_appears_once_and_in_order(exported):
    _, out = exported
    rows = edges_of(read(out / f"{NC.NAME}.ungraph.txt.gz"))
    pairs = [(int(u), int(v)) for u, v, _ in rows]
    assert all(u < v for u, v in pairs), "an undirected edge is listed once, smaller id first"
    assert len(pairs) == len(set(pairs)), "no duplicate edges"
    assert pairs == sorted(pairs)


def test_the_counts_in_the_header_match_the_file(exported):
    payload, out = exported
    lines = read(out / f"{NC.NAME}.ungraph.txt.gz")
    stated = [line for line in lines if "Edges:" in line][0]
    assert f"Edges: {payload['edges']}" in stated
    assert len(edges_of(lines)) == payload["edges"]


def test_a_bin_is_not_a_node(exported, con):
    payload, out = exported
    c, _ = con
    bins = {r[0] for r in c.execute(
        "SELECT person_id FROM s.persons WHERE page_kind = 'disambiguation'").fetchall()}
    assert bins, "the fixture should contain a disambiguation page"
    ids = {int(u) for u, v, _ in edges_of(read(out / f"{NC.NAME}.ungraph.txt.gz"))}
    ids |= {int(v) for u, v, _ in edges_of(read(out / f"{NC.NAME}.ungraph.txt.gz"))}
    assert not (ids & bins), "a bin is a name, not a person"

    node_ids = {int(line.split("\t")[0]) for line in read(out / f"{NC.NAME}.nodes.txt.gz")
                if not line.startswith("#")}
    assert not (node_ids & bins)


def test_bins_are_members_only_when_asked_for(con):
    """The edge count cannot be compared on this fixture: its block-world co-authors are slots whose
    person_id has no author page, so the join that every real serving database satisfies drops them.
    What can be checked is the thing the flag controls - whether a bin is a member of the graph."""
    c, _ = con
    for with_bins in (False, True):
        c.execute(EX.MEMBER_SQL.format(min_authors=2, max_authors=50,
                                       scope=EX._scope_clause("all"),
                                       bins=EX._bins_clause(with_bins)))
        bins = c.execute("""
            SELECT count(DISTINCT m.person_id) FROM member m
            JOIN s.persons p USING (person_id) WHERE p.page_kind = 'disambiguation'""").fetchone()[0]
        assert (bins > 0) == with_bins, f"with_bins={with_bins} gave {bins} bin members"


def test_the_with_bins_file_is_named_apart(con, tmp_path):
    c, meta = con
    EX.export(c, meta, tmp_path / "b", with_bins=True)
    assert (tmp_path / "b" / f"{NC.NAME}.withbins.txt.gz").exists()
    assert not (tmp_path / "b" / f"{NC.NAME}.ungraph.txt.gz").exists(),         "the two graphs must not overwrite each other"


def test_the_author_cap_is_reported_not_hidden(exported):
    """The cap removes real collaborations, so the number it removes belongs in the datasheet."""
    payload, out = exported
    assert payload["largest_paper_authors"] >= 1
    text = (out / "README.md").read_text(encoding="utf-8")
    assert f"{payload['papers_above_the_author_cap']:,} of them" in text


def test_every_node_in_an_edge_is_in_the_node_file(exported):
    _, out = exported
    rows = edges_of(read(out / f"{NC.NAME}.ungraph.txt.gz"))
    in_edges = {int(u) for u, v, _ in rows} | {int(v) for u, v, _ in rows}
    listed = {int(line.split("\t")[0]) for line in read(out / f"{NC.NAME}.nodes.txt.gz")
              if not line.startswith("#")}
    assert in_edges <= listed, "an edge names a node the node file does not describe"


def test_communities_are_venues_of_author_ids(exported, con):
    payload, out = exported
    c, _ = con
    lines = [line for line in read(out / f"{NC.NAME}.venues.cmty.txt.gz") if line.strip()]
    assert lines and payload["communities"] == len(lines)
    first = lines[0].split("\t")
    assert len(first) >= NC.MIN_COMMUNITY and all(part.isdigit() for part in first)

    names = [line for line in read(out / f"{NC.NAME}.venues.names.txt.gz") if not line.startswith("#")]
    assert len(names) == len(lines), "every community line must be traceable to its venue"
    sid, authors, line_no = names[0].split("\t")
    assert int(line_no) == 1 and int(authors) == len(first)
    assert c.execute("SELECT count(*) FROM s.pubs WHERE sid = ?", [sid]).fetchone()[0] > 0


def test_a_disagreement_with_the_network_job_is_an_error(con, tmp_path):
    """The job counted the edges independently with igraph; if the export disagrees it is wrong."""
    c, meta = con
    with pytest.raises(AssertionError, match="disagree"):
        EX.export(c, meta, tmp_path / "c", expect_edges=999_999_999)


def test_the_datasheet_states_the_rules(exported):
    payload, out = exported
    text = (out / "README.md").read_text(encoding="utf-8")
    assert "com-DBLP" in text
    assert f"{payload['edges']:,}" in text
    assert "disambiguation" in text.lower() and "author cap" not in text.split("## Files")[0]
    assert str(NC.MAX_AUTHORS) in text


def test_scope_changes_the_graph_and_is_recorded(con, tmp_path):
    """The check against the igraph job failed at first because the job counts journal and
    conference papers while the export counts every record type. That is a real choice, so it is a
    flag with both numbers in the datasheet - not a constant somebody has to rediscover."""
    c, meta = con
    everything = EX.export(c, meta, tmp_path / "all", scope="all")
    narrower = EX.export(c, meta, tmp_path / "jc", scope="journal-conference")
    assert narrower["edges"] <= everything["edges"]
    assert narrower["rules"]["scope"] == "journal-conference"
    assert "journal and conference papers only" in narrower["rules"]["record_types"]
    assert "scope = journal-conference" in (tmp_path / "jc" / "README.md").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="scope must be"):
        EX.export(c, meta, tmp_path / "nope", scope="everything-ever")


def test_a_failed_expectation_names_the_likely_cause(con, tmp_path):
    c, meta = con
    with pytest.raises(AssertionError, match="journal-conference"):
        EX.export(c, meta, tmp_path / "x", expect_edges=1)


def test_the_probe_counts_every_definition_it_can_express(con):
    """Three rounds went on guessing which predicate a published number came from. Counting them all
    at once is cheaper than one more guess."""
    c, _ = con
    rows = EX.probe(c)
    named = {r["scope"] for r in rows}
    assert named == set(NC.SCOPES), "every scope must be reported, even one that cannot run here"
    usable = [r for r in rows if "edges" in r]
    assert usable, "the fixture should express at least one definition"
    everything = next(r for r in usable if r["scope"] == "all")
    assert all(r["edges"] <= everything["edges"] for r in usable), "no filter may add edges"


def test_the_probe_says_which_definition_matches_a_number(con):
    c, _ = con
    first = EX.probe(c)
    everything = next(r for r in first if r["scope"] == "all")
    assert everything.get("edges", 0) > 0, first

    again = EX.probe(c, target=everything["edges"])
    repeated = next(r for r in again if r["scope"] == "all")
    assert repeated["edges"] == everything["edges"], (everything, repeated)
    assert repeated.get("matches_target") is True, again
    assert all(r["difference"] == r["edges"] - everything["edges"] for r in again if "edges" in r)
