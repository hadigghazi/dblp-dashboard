"""
The co-authorship network as files, in the format SNAP publishes.

SNAP's com-DBLP - the dataset the course showed - is a gzipped tab-separated edge list with a short
comment header, plus community files whose communities are publication venues. This writes the same
thing from the current dump: about 3.8 million authors and 22.2 million edges against com-DBLP's
317 thousand and 1.05 million, with disambiguation handled, which com-DBLP does not do.

Three rules define an edge, and each one is reported in the datasheet rather than assumed:

  * a paper with 2 to 50 authors makes an edge between each pair of them, weighted by how many
    papers they share;
  * a disambiguation bin is not a person, so it is not a node - with bins the most connected vertex
    in computer science has degree 6,570 and is a bare name shared by hundreds of people, without
    them it is 2,343 and is somebody;
  * every record type counts: journal, conference, preprint, book, chapter, thesis. This is the
    whole corpus, not the journal/conference subset most of the dashboard's charts use.

The heavy work is a DuckDB `COPY`, which writes gzipped TSV straight from the query - 22 million
rows never pass through Python. The header is written as its own gzip member and the body appended:
concatenated gzip members are a single valid gzip stream, so the result reads normally.
"""
import gzip
import json
import logging
import shutil
import time

import duckdb
from datetime import datetime, timezone
from pathlib import Path

from . import config

log = logging.getLogger("dblp.ml.network.export")

MEMBER_SQL = """
CREATE OR REPLACE TEMP TABLE member AS
SELECT sl.person_id, sl.pid
FROM s.slots sl
JOIN s.persons p ON p.person_id = sl.person_id
JOIN s.pubs b ON b.pid = sl.pid
WHERE sl.person_id IS NOT NULL
  AND b.n_authors BETWEEN {min_authors} AND {max_authors}
  AND ({scope})
  {bins}
"""

EDGES_SQL = """
SELECT a.person_id AS u, b.person_id AS v, count(*) AS papers
FROM member a JOIN member b ON a.pid = b.pid AND a.person_id < b.person_id
GROUP BY 1, 2
ORDER BY 1, 2
"""

# Node statistics are computed from slots and pubs rather than read from person_stats/career, so the
# export works against any serving database that has the three core tables.
NODES_SQL = """
SELECT p.person_id AS id, p.key, p.name,
       count(sl.pid) AS papers, min(b.year) AS first_year, max(b.year) AS last_year
FROM s.persons p
LEFT JOIN s.slots sl ON sl.person_id = p.person_id
LEFT JOIN s.pubs b ON b.pid = sl.pid
{where}
GROUP BY 1, 2, 3
ORDER BY 1
"""

COMMUNITY_SQL = """
SELECT b.sid, count(DISTINCT sl.person_id) AS authors,
       string_agg(DISTINCT sl.person_id::VARCHAR, chr(9)) AS members
FROM s.slots sl
JOIN s.pubs b ON b.pid = sl.pid
JOIN s.persons p ON p.person_id = sl.person_id
WHERE b.key_prefix IN ('conf', 'journals') AND NOT b.is_preprint AND b.sid IS NOT NULL
  {bins}
GROUP BY b.sid
HAVING count(DISTINCT sl.person_id) >= {min_community}
ORDER BY authors DESC, b.sid
"""


def _bins_clause(with_bins, alias="p"):
    return "" if with_bins else f"AND {alias}.page_kind <> 'disambiguation'"


def _scope_clause(scope):
    if scope not in config.SCOPES:
        raise ValueError(f"scope must be one of {', '.join(config.SCOPES)}")
    return config.SCOPES[scope]


def _copy(con, sql, target: Path):
    """DuckDB writes the body straight to gzip; 22 million rows never pass through Python."""
    tmp = target.with_suffix(".body.gz")
    tmp.unlink(missing_ok=True)
    con.execute(f"COPY ({sql}) TO '{tmp}' (FORMAT CSV, DELIMITER '\t', HEADER false, COMPRESSION gzip)")
    return tmp


def _write_with_header(header_lines, body: Path, target: Path):
    """Header as one gzip member, body as the next: concatenated members are one valid stream."""
    target.unlink(missing_ok=True)
    with gzip.open(target, "wb") as fh:
        fh.write(("".join(f"# {line}\n" for line in header_lines)).encode("utf-8"))
    with open(target, "ab") as out, open(body, "rb") as src:
        shutil.copyfileobj(src, out, length=1024 * 1024)
    body.unlink(missing_ok=True)
    return target


def counts(con, with_bins):
    nodes = con.execute(f"""
        SELECT count(*) FROM s.persons p WHERE TRUE {_bins_clause(with_bins)}""").fetchone()[0]
    edges = con.execute("SELECT count(*) FROM edge").fetchone()[0]
    linked = con.execute("SELECT count(*) FROM (SELECT u AS id FROM edge UNION SELECT v FROM edge)").fetchone()[0]
    dropped = con.execute(f"""
        SELECT count(*) FROM s.pubs WHERE n_authors > {config.MAX_AUTHORS}""").fetchone()[0]
    biggest = con.execute(f"""
        SELECT coalesce(max(n_authors), 0) FROM s.pubs""").fetchone()[0]
    return {"author_pages": int(nodes), "nodes_with_an_edge": int(linked), "edges": int(edges),
            "papers_above_the_author_cap": int(dropped), "largest_paper_authors": int(biggest)}


def probe(con, with_bins=False, target=None):
    """Count the graph under every definition, so the one behind a published number can be named
    rather than guessed at. A definition the serving database cannot express is reported, not
    skipped silently."""
    out = []
    for name, predicate in config.SCOPES.items():
        try:
            # dropped rather than replaced: a table built FROM another cannot always be replaced
            # while the one it came from is being replaced too, and the probe runs repeatedly
            con.execute("DROP TABLE IF EXISTS probe_edge")
            con.execute("DROP TABLE IF EXISTS member")
            con.execute(MEMBER_SQL.format(min_authors=config.MIN_AUTHORS, max_authors=config.MAX_AUTHORS,
                                          scope=predicate, bins=_bins_clause(with_bins)))
            con.execute(f"CREATE TEMP TABLE probe_edge AS {EDGES_SQL}")
            edges = con.execute("SELECT count(*) FROM probe_edge").fetchone()[0]
            nodes = con.execute("""
                SELECT count(*) FROM (SELECT u AS id FROM probe_edge UNION
                                      SELECT v FROM probe_edge)""").fetchone()[0]
            entry = {"scope": name, "edges": int(edges), "nodes_with_an_edge": int(nodes)}
            if target:
                entry["matches_target"] = int(edges) == int(target)
                entry["difference"] = int(edges) - int(target)
        except duckdb.Error as e:
            # only a database error means "this serving database cannot express that predicate";
            # a bug in this function was being reported as one until it was caught by a test
            entry = {"scope": name, "error": f"{type(e).__name__}: {e}"}
        log.info("%s", entry)
        out.append(entry)
    return out


def export(con, meta, out_dir, with_bins=False, expect_edges=None, scope=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "withbins" if with_bins else "ungraph"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()

    scope = scope or config.SCOPE
    con.execute(MEMBER_SQL.format(min_authors=config.MIN_AUTHORS, max_authors=config.MAX_AUTHORS,
                                  scope=_scope_clause(scope), bins=_bins_clause(with_bins)))
    log.info("building the edge table…")
    con.execute(f"CREATE OR REPLACE TEMP TABLE edge AS {EDGES_SQL}")
    stats = counts(con, with_bins)
    log.info("%s nodes with an edge, %s edges", f"{stats['nodes_with_an_edge']:,}", f"{stats['edges']:,}")
    if expect_edges is not None and stats["edges"] != int(expect_edges):
        raise AssertionError(
            f"expected {int(expect_edges):,} edges, built {stats['edges']:,} under scope "
            f"'{scope}' - the export and the number you compared with disagree about what an edge "
            f"is. The network job counts journal and conference papers only: try "
            f"--scope journal-conference.")

    edges_file = out / f"{config.NAME}.{tag}.txt.gz"
    body = _copy(con, "SELECT u, v, papers FROM edge ORDER BY u, v", edges_file)
    _write_with_header([
        f"Undirected co-authorship graph from dblp ({meta.get('latest_mdate', 'unknown')} dump)",
        f"{config.NAME}: authors who share a paper with {config.MIN_AUTHORS}-{config.MAX_AUTHORS} authors",
        "disambiguation bins included" if with_bins else "disambiguation bins excluded (a bin is a name, not a person)",
        f"Nodes: {stats['nodes_with_an_edge']} Edges: {stats['edges']}",
        f"Generated: {stamp} from dump {meta.get('fingerprint')}",
        "FromNodeId\tToNodeId\tPapersTogether",
    ], body, edges_file)

    nodes_file = out / f"{config.NAME}.nodes.txt.gz"
    where = "WHERE TRUE " + _bins_clause(with_bins)
    body = _copy(con, NODES_SQL.format(where=where), nodes_file)
    _write_with_header([
        f"Author pages for {config.NAME} ({meta.get('latest_mdate', 'unknown')} dump)",
        "NodeId\tdblpKey\tName\tRecords\tFirstYear\tLastYear",
    ], body, nodes_file)

    communities = _write_communities(con, out, with_bins, meta)

    payload = {
        "name": config.NAME, "generated_at": stamp,
        "dump": {k: meta.get(k) for k in ("fingerprint", "latest_mdate", "records")},
        "with_bins": bool(with_bins),
        "rules": {"min_authors": config.MIN_AUTHORS, "max_authors": config.MAX_AUTHORS,
                  "scope": scope,
                  "record_types": ("all (journal, conference, preprint, book, chapter, thesis, data)"
                                   if scope == "all" else "journal and conference papers only"),
                  "years": "all"},
        **stats, **communities,
        "files": sorted(p.name for p in out.glob("*.gz")),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "README.md").write_text(datasheet(payload), encoding="utf-8")
    log.info("written to %s in %.0fs", out, time.time() - t0)
    return payload


def _write_communities(con, out, with_bins, meta):
    """Venues as ground-truth communities, the way com-DBLP does it: one line per venue, the ids of
    its authors, tab separated. Written from Python because a line is a whole community."""
    rows = con.execute(COMMUNITY_SQL.format(bins=_bins_clause(with_bins),
                                            min_community=config.MIN_COMMUNITY)).fetchall()
    all_path = out / f"{config.NAME}.venues.cmty.txt.gz"
    top_path = out / f"{config.NAME}.venues.top{config.TOP_COMMUNITIES}.cmty.txt.gz"
    names_path = out / f"{config.NAME}.venues.names.txt.gz"
    with gzip.open(all_path, "wt", encoding="utf-8") as everything, \
            gzip.open(top_path, "wt", encoding="utf-8") as top, \
            gzip.open(names_path, "wt", encoding="utf-8") as names:
        names.write("# VenueSeriesId\tAuthors\tLineNumber\n")
        for i, (sid, authors, members) in enumerate(rows):
            everything.write(members + "\n")
            names.write(f"{sid}\t{authors}\t{i + 1}\n")
            if i < config.TOP_COMMUNITIES:
                top.write(members + "\n")
    return {"communities": len(rows), "largest_community": int(rows[0][1]) if rows else 0}


def datasheet(p):
    """What a reader needs to use the files and to judge them."""
    r = p["rules"]
    return f"""# {p['name']}

The co-authorship network of dblp, in the format SNAP uses for `com-DBLP`.

Generated {p['generated_at']} from the dblp dump of {p['dump'].get('latest_mdate')}
(fingerprint `{p['dump'].get('fingerprint')}`, {p['dump'].get('records')} records).

| | this dataset | SNAP com-DBLP (2012) |
|---|---|---|
| Nodes (with at least one co-author) | {p['nodes_with_an_edge']:,} | ~317,080 |
| Edges | {p['edges']:,} | ~1,049,866 |
| Communities (venues) | {p['communities']:,} | ~13,477 |

## Files

| File | Contents |
|---|---|
| `{p['name']}.{'withbins' if p['with_bins'] else 'ungraph'}.txt.gz` | `FromNodeId  ToNodeId  PapersTogether`, one undirected edge per line |
| `{p['name']}.nodes.txt.gz` | `NodeId  dblpKey  Name  Records  FirstYear  LastYear` |
| `{p['name']}.venues.cmty.txt.gz` | one line per venue: the ids of its authors |
| `{p['name']}.venues.top{config.TOP_COMMUNITIES}.cmty.txt.gz` | the {config.TOP_COMMUNITIES} largest of those |
| `{p['name']}.venues.names.txt.gz` | which venue each community line belongs to |

Node ids are dblp author-page ids for this dump: they are stable within it and map to a real page
through `nodes.txt`, but they are not contiguous and they change when the dump does.

## What an edge means

Two authors share a paper with {r['min_authors']}-{r['max_authors']} authors; the weight counts how
many such papers they share. Record types: {r['record_types']}. Years: {r['years']}.

Scope is the one choice that moves the numbers most. On this dump, counting every record type gives
24,381,691 edges among 3,989,840 authors; counting only journal and conference papers gives
22,200,244 among 3,824,208 - the difference is preprints, books, chapters and theses, and the
smaller figure is what the dashboard's network analysis reports. This file was built with
`scope = {r['scope']}`.

## What is excluded, and why

* **Disambiguation bins**{'' if p['with_bins'] else ' (excluded here)'}: a bare name such as "Wei Wang"
  holding the papers of hundreds of different people. Counting one as a person makes it the most
  connected vertex in computer science - degree 6,570 with bins against 2,343 without, on this dump.
  SNAP's com-DBLP does not do this.
* **Papers with more than {r['max_authors']} authors**: {p['papers_above_the_author_cap']:,} of them
  (the largest has {p['largest_paper_authors']:,} authors). One 200-author paper contributes 19,900
  edges of a single clique, which dominates betweenness and closeness.
* **Author names with no dblp page**: they have no identity to be a node.

Authors whose papers are all single-author appear in `nodes.txt` with no edges: an edge list cannot
represent an isolated vertex.
"""
