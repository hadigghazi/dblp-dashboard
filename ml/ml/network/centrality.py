"""
Centrality on the exported graph: who is structurally important in computer science, and how
certain the answer is.

This reads the *published dataset files*, not the database. The exercise is to apply centrality to
a SNAP-style edge list, and reading the same files a stranger would download is what proves they
are usable: if these numbers are right, anyone can reproduce them from the download alone.

Four measures, and the honest cost of each on 3.99 million authors and 24.4 million edges:

  * **degree** - exact. How many co-authors somebody has. One pass over the edges.
  * **eigenvector** - exact to a tolerance, by power iteration. Being co-author to important people
    is what makes you important. Its leading eigenvector lives on a single connected component, so
    on a disconnected graph every other component would score ~0 for reasons of arithmetic rather
    than of collaboration; it therefore runs on the largest component only.
  * **betweenness** - estimated. Exact Brandes is O(n·m): 10^14 operations here, which is weeks.
    Riondato-Kornaropoulos sampling runs Brandes from a random sample of sources instead; the
    estimator is unbiased, and the sample count is reported so that the error is a *stated*
    sampling error rather than an unstated one.
  * **closeness** - estimated, Eppstein-Wang, for the same reason, and on the largest component
    because a distance to an unreachable author is not a number.

PageRank and the core number come along nearly free and are worth having. PageRank is the version
of eigenvector centrality that survives disconnection; the core number says whether a high degree
sits inside a dense community or is a hub over a periphery.

Edge weights (papers shared) are deliberately *ignored* by the distance-based measures. A
shortest-path algorithm reads a weight as a length, so feeding it "number of papers together"
would make frequent collaborators far apart - exactly backwards. Degree is therefore a co-author
count, and every path here is a number of hops.

Nothing below trusts a library blindly:

  * the loaded edge count is checked against the count in the dataset's own header, because a
    truncated read of a multi-member gzip stream would otherwise look like a smaller graph;
  * degree is computed twice, once by NetworKit on the loaded graph and once by DuckDB on the
    source file, and the run fails if they disagree - that is the check that the id remapping is
    right;
  * closeness is spot-checked against exact single-source shortest paths from a random sample of
    nodes, and the measured error is reported next to the promised one.

DuckDB does the I/O and the joins, NetworKit does the graph algorithms. Neither is asked to do the
other's job.
"""
import gzip
import json
import logging
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import config as base, data as D
from . import config

log = logging.getLogger("dblp.ml.network.centrality")

# the export writes "# Nodes: 3824208 Edges: 22200244" into the edge file's header
HEADER_COUNTS = re.compile(r"Nodes:\s*(\d+)\s+Edges:\s*(\d+)")

DAMPING = 0.85          # the usual PageRank constant
EIGEN_TOL = 1e-9
PAGERANK_TOL = 1e-9
BETWEENNESS_SAMPLES = 32768
# 1024, not 4096, on measured evidence: at 4096 sampled sources on the real graph the spot check
# against exact shortest paths found a 90th-percentile error of 0.35%, which is far more precision
# than any use of closeness here needs - and it cost 1.9 hours. Quartering it roughly doubles that
# error, to well under 1%, and the spot check reports what it actually was either way.
CLOSENESS_SAMPLES = 1024
CLOSENESS_EPSILON = 0.1
SPOT_CHECK = 200        # exact single-source runs used to measure the closeness error
CHECKPOINT = "scores.npz"              # the expensive results, saved before anything is joined
CHECKPOINT_SUMMARY = "scores-summary.json"
DIAMETER_ERROR = 0.05
CLUSTERING_ERROR = 0.01
TOP = 25                # rows per measure in the summary


# --------------------------------------------------------------------------- reading the dataset

def header_counts(path: Path):
    """The node and edge counts the dataset claims, read with Python's gzip - which, unlike some
    readers, is guaranteed to see past the first member of a multi-member stream."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            found = HEADER_COUNTS.search(line)
            if found:
                return int(found.group(1)), int(found.group(2))
    return None, None


def decompress(src: Path, dst: Path) -> Path:
    """The dataset's body, as a plain file with the comment header removed.

    Two reasons, both of which cost a working day between them. The export writes its header as one
    gzip member and the body as the next, which is a valid stream that not every reader follows to
    the end - and a reader that stops early reports a smaller graph rather than an error. And
    DuckDB's CSV dialect detection looks at those `#` lines before it applies the comment option,
    finds a line with no tab in it, and concludes the file has one column.

    Python's gzip reads every member, and a file with no comments in it cannot be misread as one.
    Only the leading block is dropped: every data line begins with a node id, so a `#` can only
    appear at the top."""
    t0 = time.time()
    dropped = 0
    with gzip.open(src, "rb") as fh, open(dst, "wb") as out:
        line = fh.readline()
        while line.startswith(b"#"):
            dropped += 1
            line = fh.readline()
        out.write(line)
        shutil.copyfileobj(fh, out, length=4 * 1024 * 1024)
    log.info("decompressed %s -> %.2f GB in %.0fs (%s header lines dropped)",
             src.name, dst.stat().st_size / 1e9, time.time() - t0, dropped)
    return dst


def scratch_root():
    """Deliberately not inside DuckDB's temp directory. That one is shared by every ml job and swept
    by the database, and a scratch path shared between two runs of the same dump is what destroyed
    six hours of betweenness: one run's cleanup deleted the other run's inputs."""
    root = base.MODELS_DIR / "network-scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_edges(con, plain: Path, expect_edges=None):
    """The edge list as a DuckDB table, with the claimed edge count enforced."""
    con.execute(f"""
        CREATE OR REPLACE TABLE e AS
        SELECT * FROM read_csv('{plain.as_posix()}', delim='\t', header=false, auto_detect=false,
                               columns={{'u': 'BIGINT', 'v': 'BIGINT', 'w': 'BIGINT'}})""")
    edges = con.execute("SELECT count(*) FROM e").fetchone()[0]
    if expect_edges is not None and edges != expect_edges:
        raise AssertionError(
            f"the dataset's header says {expect_edges:,} edges but {edges:,} were read from "
            f"{plain.name}. The file is truncated or is not the file its header describes.")
    log.info("loaded %s edges", f"{edges:,}")
    return int(edges)


def remap(con, graph_file: Path, expect_nodes=None):
    """dblp author-page ids are sparse; every graph library wants 0..n-1. This builds the mapping
    once, deterministically (by id, so two runs agree), and writes the renumbered edge list for
    NetworKit's C++ reader to swallow whole."""
    con.execute("""
        CREATE OR REPLACE TABLE node AS
        SELECT id, (row_number() OVER (ORDER BY id)) - 1 AS nid
        FROM (SELECT u AS id FROM e UNION SELECT v FROM e)""")
    nodes = con.execute("SELECT count(*) FROM node").fetchone()[0]
    if expect_nodes is not None and nodes != expect_nodes:
        raise AssertionError(
            f"the dataset's header says {expect_nodes:,} nodes with an edge but the edge list "
            f"contains {nodes:,} distinct authors.")
    con.execute(f"""
        COPY (SELECT a.nid AS u, b.nid AS v FROM e
              JOIN node a ON a.id = e.u JOIN node b ON b.id = e.v
              ORDER BY 1, 2)
        TO '{graph_file.as_posix()}' (FORMAT CSV, DELIMITER ' ', HEADER false)""")
    log.info("renumbered %s authors to 0..%s", f"{nodes:,}", f"{nodes - 1:,}")
    return int(nodes)


def load_graph(graph_file: Path, nodes, edges):
    """NetworKit reads the renumbered list itself: 24 million lines never pass through Python."""
    import networkit as nk
    t0 = time.time()
    reader = nk.graphio.EdgeListReader(" ", 0, "#", True, False)
    G = reader.read(str(graph_file))
    log.info("graph loaded: %s nodes, %s edges, %.0fs",
             f"{G.numberOfNodes():,}", f"{G.numberOfEdges():,}", time.time() - t0)
    if G.numberOfNodes() != nodes or G.numberOfEdges() != edges:
        raise AssertionError(
            f"the graph NetworKit built ({G.numberOfNodes():,} nodes, {G.numberOfEdges():,} edges) "
            f"is not the graph that was written ({nodes:,}, {edges:,}).")
    return G


def check_degrees(con, G, sample=1000, seed=None):
    """Degree, twice: once from the loaded graph, once from the source file. They can only differ if
    the renumbering is wrong, which is the one mistake in this pipeline that would silently produce
    plausible centralities for the wrong people."""
    rows = con.execute(f"""
        WITH ends AS (SELECT u AS id FROM e UNION ALL SELECT v FROM e),
             deg AS (SELECT id, count(*) AS degree FROM ends GROUP BY 1)
        SELECT * FROM (SELECT n.nid, d.degree FROM deg d JOIN node n ON n.id = d.id)
        USING SAMPLE reservoir({int(sample)} ROWS) REPEATABLE ({int(seed if seed is not None else base.SEED)})
    """).fetchall()
    bad = [(int(nid), int(expected), G.degree(int(nid)))
           for nid, expected in rows if G.degree(int(nid)) != int(expected)]
    if bad:
        raise AssertionError(
            f"{len(bad)} of {len(rows)} sampled authors have a different degree in the graph than "
            f"in the edge file, e.g. node {bad[0][0]}: file says {bad[0][1]}, graph says "
            f"{bad[0][2]}. The id remapping is wrong.")
    log.info("degree cross-check passed on %s sampled authors", f"{len(rows):,}")
    return len(rows)


# --------------------------------------------------------------------------- the measures

def _timed(timings, name, fn):
    t0 = time.time()
    value = fn()
    timings[name] = round(time.time() - t0, 1)
    log.info("%s: %.0fs", name, timings[name])
    return value


def largest_component(G):
    """The largest connected component, renumbered again to 0..k-1, plus the map back. Returned
    compacted because the distance-based algorithms are happiest on a graph with no holes in it."""
    import networkit as nk
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    sizes = cc.getComponentSizes()
    biggest = max(sizes, key=sizes.get)
    members = cc.getPartition().getMembers(biggest)
    sub = nk.graphtools.subgraphFromNodes(G, members)
    ids = nk.graphtools.getContinuousNodeIds(sub)
    H = nk.graphtools.getCompactedGraph(sub, ids)
    back = np.full(H.numberOfNodes(), -1, dtype=np.int64)
    for old, new in ids.items():
        if new < len(back):
            back[new] = old
    if (back < 0).any():
        raise AssertionError("the component's node mapping has gaps; it cannot be mapped back")
    log.info("largest component: %s of %s authors (%.2f%%), %s edges",
             f"{H.numberOfNodes():,}", f"{G.numberOfNodes():,}",
             100 * H.numberOfNodes() / G.numberOfNodes(), f"{H.numberOfEdges():,}")
    return H, back, {"components": len(sizes),
                     "largest_component_nodes": int(H.numberOfNodes()),
                     "largest_component_edges": int(H.numberOfEdges()),
                     "largest_component_share": round(H.numberOfNodes() / G.numberOfNodes(), 6),
                     "component_size_distribution": _size_histogram(sizes)}


def _size_histogram(sizes):
    """How the rest of the graph is shattered: mostly pairs and triples, which is worth showing."""
    counts = {}
    for size in sizes.values():
        key = str(size) if size <= 10 else ("11-100" if size <= 100 else ("101-1000" if size <= 1000 else ">1000"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _ranks(values):
    """Rank 1 is the most central. Nodes with no value (outside the largest component) get 0, which
    reads as "not ranked" rather than as "last"."""
    scored = np.nan_to_num(values, nan=-np.inf, neginf=-np.inf)
    order = np.argsort(-scored, kind="stable")
    out = np.zeros(len(values), dtype=np.int64)
    out[order] = np.arange(1, len(values) + 1)
    out[~np.isfinite(values)] = 0
    return out


def descriptive(G, H, timings):
    """Facts about the shape of the graph. These are descriptive statistics, not the deliverable, so
    one that fails reports itself and the run continues - the four measures do not get that
    latitude."""
    import networkit as nk
    out = {}
    try:
        out["approx_global_clustering_coefficient"] = _timed(
            timings, "clustering", lambda: round(float(nk.globals.clustering(G, error=CLUSTERING_ERROR)), 5))
        out["clustering_error"] = CLUSTERING_ERROR
    except Exception as e:                       # descriptive only: report, do not abort
        out["clustering_error_message"] = f"{type(e).__name__}: {e}"
    try:
        low, high = _timed(timings, "diameter", lambda: _diameter(H))
        out["largest_component_diameter"] = {"lower": int(low), "upper": int(high),
                                             "method": "estimated range", "error": DIAMETER_ERROR}
    except Exception as e:
        out["diameter_error_message"] = f"{type(e).__name__}: {e}"
    return out


def _diameter(H):
    import networkit as nk
    algo = nk.distance.Diameter(H, nk.distance.DiameterAlgo.ESTIMATED_RANGE, DIAMETER_ERROR)
    algo.run()
    return algo.getDiameter()


def seconds_per_sampled_source(H):
    """How long one sampled source takes, measured rather than assumed.

    Betweenness is the only step here that runs for hours, and it prints nothing while it does, so
    the log needs to say up front roughly how long it will be. One sampled source costs seconds and
    the total is linear in the sample count, so this is the cheapest honest estimate available. It
    runs high: this first source also pays the one-off setup that the other thousands do not."""
    import networkit as nk
    t0 = time.time()
    nk.centrality.EstimateBetweenness(H, 1, True, True).run()
    return time.time() - t0


def measures(G, threads, seed, betweenness_samples, closeness_samples, spot_check=None):
    """Every measure, with its own timing. Whole-graph measures first, then the ones that need
    distances and therefore need the largest component."""
    import networkit as nk
    nk.setNumberOfThreads(int(threads))
    nk.setSeed(int(seed), False)
    log.info("networkit on %s threads, seed %s", nk.getMaxNumberOfThreads(), seed)
    timings, n = {}, G.numberOfNodes()

    degree = _timed(timings, "degree", lambda: np.asarray(
        nk.centrality.DegreeCentrality(G).run().scores(), dtype=np.float64).astype(np.int64))
    core = _timed(timings, "core", lambda: np.asarray(
        nk.centrality.CoreDecomposition(G).run().scores(), dtype=np.float64).astype(np.int32))
    pagerank = _timed(timings, "pagerank", lambda: np.asarray(
        nk.centrality.PageRank(G, DAMPING, PAGERANK_TOL).run().scores(), dtype=np.float64))

    H, back, component = largest_component(G)
    facts = descriptive(G, H, timings)

    # a source sampled twice adds no information, so asking for more sources than the component has
    # is waste at best; on a small graph the library refuses outright
    reachable = H.numberOfNodes()
    betweenness_samples = max(1, min(int(betweenness_samples), reachable))
    closeness_samples = max(1, min(int(closeness_samples), reachable - 1 if reachable > 1 else 1))

    eigen_lcc = _timed(timings, "eigenvector", lambda: np.asarray(
        nk.centrality.EigenvectorCentrality(H, EIGEN_TOL).run().scores(), dtype=np.float64))

    log.info("sampling next: betweenness from %s sources, closeness from %s, on %s threads, seed %s",
             f"{betweenness_samples:,}", f"{closeness_samples:,}", threads, seed)
    per_source = _timed(timings, "betweenness_calibration", lambda: seconds_per_sampled_source(H))
    projected = betweenness_samples * per_source / max(1, int(threads))
    log.info("one sampled source took %.1fs, so betweenness should take in the region of %.0f "
             "minutes - and rather less, since that first source paid the setup as well. It prints "
             "nothing at all until it finishes.", per_source, projected / 60)
    betweenness_lcc = _timed(timings, "betweenness", lambda: np.asarray(
        nk.centrality.EstimateBetweenness(H, int(betweenness_samples), True, True).run().scores(),
        dtype=np.float64))
    log.info("betweenness took %.0f minutes against the %.0f projected (%.2fx)",
             timings["betweenness"] / 60, projected / 60,
             timings["betweenness"] / projected if projected else float("nan"))
    closeness_lcc = _timed(timings, "closeness", lambda: np.asarray(
        nk.centrality.ApproxCloseness(H, int(closeness_samples), CLOSENESS_EPSILON, True).run().scores(),
        dtype=np.float64))

    def spread(values):
        """A largest-component measure, placed back on the whole graph. NaN means "not measured
        here", which is the truth for an author in some other component."""
        full = np.full(n, np.nan, dtype=np.float64)
        full[back] = values
        return full

    scores = {"degree": degree, "core": core, "pagerank": pagerank,
              "eigenvector": spread(eigen_lcc), "betweenness": spread(betweenness_lcc),
              "closeness": spread(closeness_lcc)}
    check = spot_check_closeness(H, closeness_lcc, seed, timings, spot_check)
    return scores, H, back, {
        "nodes": int(n), "edges": int(G.numberOfEdges()),
        "average_degree": round(2 * G.numberOfEdges() / n, 3),
        "max_degree": int(degree.max()), "max_core": int(core.max()),
        "density": 2 * G.numberOfEdges() / (n * (n - 1)),
        **component, **facts,
        "parameters": {
            "betweenness": {"algorithm": "Riondato-Kornaropoulos sampling (NetworKit "
                                         "EstimateBetweenness)", "samples": int(betweenness_samples),
                            "normalized": True, "exact": False,
                            "note": "unbiased estimate of normalised betweenness; the sample count "
                                    "is the whole of the error story"},
            "closeness": {"algorithm": "Eppstein-Wang sampling (NetworKit ApproxCloseness)",
                          "samples": int(closeness_samples), "epsilon": CLOSENESS_EPSILON,
                          "normalized": True, "exact": False},
            "eigenvector": {"algorithm": "power iteration", "tolerance": EIGEN_TOL, "exact": True,
                            "scope": "largest connected component"},
            "pagerank": {"damping": DAMPING, "tolerance": PAGERANK_TOL, "exact": True,
                         "scope": "whole graph"},
            "degree": {"exact": True, "scope": "whole graph", "weighted": False},
            "seed": int(seed), "threads": int(threads),
        },
        "closeness_spot_check": check,
        "betweenness_projection": {"seconds_per_sampled_source": round(per_source, 3),
                                   "projected_seconds": round(projected, 1),
                                   "actual_seconds": timings["betweenness"],
                                   "note": "projected from one timed source before the run; the "
                                           "projection runs high because that source also paid the "
                                           "one-off setup"},
        "seconds": timings}


def spot_check_closeness(H, approx, seed, timings, sample=None):
    """The promised error of a sampling algorithm is a promise. This measures the error instead:
    exact single-source shortest paths from a random sample of authors, against what the
    approximation said about those same authors.

    NetworKit's normalisation convention is not assumed - both candidate conventions are computed
    and the one that matches is named. A median ratio near 1 under one convention means the
    approximation agrees with exact arithmetic; a ratio that is some other constant would mean a
    normalisation difference; ratios scattered around 1 are sampling error, which is what is
    expected and what gets reported."""
    import networkit as nk
    k = H.numberOfNodes()
    size = min(int(sample or SPOT_CHECK), k)
    rng = np.random.default_rng(int(seed))
    sample = rng.choice(k, size=size, replace=False)
    t0 = time.time()
    totals = np.empty(size, dtype=np.float64)
    reached = np.empty(size, dtype=np.float64)
    for i, source in enumerate(sample):
        bfs = nk.distance.BFS(H, int(source), False)
        bfs.run()
        distances = np.asarray(bfs.getDistances(), dtype=np.float64)
        finite = distances[distances < 1e300]
        totals[i] = finite.sum()
        reached[i] = len(finite) - 1        # excluding the source itself
    timings["closeness_spot_check"] = round(time.time() - t0, 1)

    got = approx[sample]
    conventions = {"(reachable-1)/sum(d)": np.divide(reached, totals, out=np.zeros(size), where=totals > 0),
                   "1/sum(d)": np.divide(1.0, totals, out=np.zeros(size), where=totals > 0)}
    best, report = None, {}
    for name, exact in conventions.items():
        usable = (exact > 0) & np.isfinite(got)
        if not usable.any():
            continue
        ratio = got[usable] / exact[usable]
        entry = {"median_ratio": round(float(np.median(ratio)), 6),
                 "p90_relative_error": round(float(np.percentile(np.abs(ratio - 1), 90)), 6),
                 "max_relative_error": round(float(np.abs(ratio - 1).max()), 6)}
        report[name] = entry
        if best is None or abs(entry["median_ratio"] - 1) < abs(report[best]["median_ratio"] - 1):
            best = name
    log.info("closeness spot-check on %s authors: convention %s, median ratio %s, p90 error %s",
             f"{size:,}", best, report.get(best, {}).get("median_ratio"),
             report.get(best, {}).get("p90_relative_error"))
    return {"sampled_authors": int(size), "matching_convention": best, "by_convention": report,
            "exact_sssp_runs": int(size)}


def correlations(scores, back, top=100):
    """What the measures say about each other. Rank correlation on the largest component, where all
    six are defined, plus how much the top lists actually overlap - which is the question a reader
    has and is not answerable from a correlation coefficient.

    Spearman's rho is Pearson's r on the ranks, so each column is ranked once and the fifteen pairs
    are then free; ranking inside the loop meant thirty passes over 3.8 million values."""
    from scipy.stats import rankdata
    names = ["degree", "core", "pagerank", "eigenvector", "betweenness", "closeness"]
    ranked, leaders_ = {}, {}
    for name in names:
        column = scores[name][back].astype(np.float64)
        ranked[name] = rankdata(column)          # ties share the average rank, as Spearman requires
        leaders_[name] = set(np.argsort(-column, kind="stable")[:top].tolist())
    out = {"spearman": {}, "top_overlap": {}, "top": int(top),
           "measured_on": "largest connected component"}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = float(np.corrcoef(ranked[a], ranked[b])[0, 1])
            out["spearman"][f"{a} vs {b}"] = round(rho, 4)
            out["top_overlap"][f"{a} vs {b}"] = len(leaders_[a] & leaders_[b])
    return out


# --------------------------------------------------------------------------- writing it out

COLUMNS = ["degree", "core", "pagerank", "eigenvector", "betweenness", "closeness"]


def save_checkpoint(out: Path, scores, person_id, back, summary, checked):
    """The measures, on disk, before anything is joined to anything.

    Betweenness and closeness take hours; the join, the correlations and the files take minutes. Any
    failure in the cheap part used to destroy the expensive part, which is precisely what happened.
    With this, `--resume` finishes the job from here."""
    target = out / CHECKPOINT
    np.savez(target, person_id=np.asarray(person_id, dtype=np.int64),
             back=np.asarray(back, dtype=np.int64), **{c: scores[c] for c in COLUMNS})
    (out / CHECKPOINT_SUMMARY).write_text(
        json.dumps({"summary": summary, "degree_cross_check_authors": int(checked)}, indent=2),
        encoding="utf-8")
    log.info("checkpointed the measures to %s (%.2f GB) - a failure after this point costs minutes, "
             "not hours", target.name, target.stat().st_size / 1e9)
    return target


def load_checkpoint(out: Path):
    with np.load(out / CHECKPOINT) as data:
        scores = {c: data[c] for c in COLUMNS}
        person_id, back = data["person_id"], data["back"]
    doc = json.loads((out / CHECKPOINT_SUMMARY).read_text(encoding="utf-8"))
    log.info("resuming: %s authors' measures read back from %s", f"{len(person_id):,}", CHECKPOINT)
    return scores, person_id, back, doc["summary"], doc["degree_cross_check_authors"]


def load_node_meta(con, nodes_plain: Path):
    """The author names, loaded before the long measures rather than after them.

    This used to be read at the end, hours after the temporary file was written, and a missing file
    at that point threw away the whole run. Nothing in the expensive phase depends on a file now."""
    con.execute(f"""
        CREATE OR REPLACE TABLE node_meta AS
        SELECT * FROM read_csv('{nodes_plain.as_posix()}', delim='\t', header=false,
                               auto_detect=false, quote='"', escape='"',
                               columns={{'id': 'BIGINT', 'key': 'VARCHAR', 'name': 'VARCHAR',
                                         'records': 'BIGINT', 'first_year': 'INTEGER',
                                         'last_year': 'INTEGER'}})""")
    rows = con.execute("SELECT count(*) FROM node_meta").fetchone()[0]
    log.info("loaded %s author pages from the nodes file", f"{rows:,}")
    return int(rows)


def write(con, scores, person_id, out: Path, name=None):
    """One row per author, joined back to their dblp key and name so the file stands alone.

    The dblp id travels with the scores rather than being looked up through a renumbering table, so
    this step depends on nothing but the database's own `node_meta` and the arrays it is handed."""
    name = name or config.NAME
    frame = {"person_id": np.asarray(person_id, dtype=np.int64)}
    for column in COLUMNS:
        frame[column] = scores[column]
    for column in ["degree", "eigenvector", "betweenness", "closeness"]:
        frame[f"rank_{column}"] = _ranks(scores[column].astype(np.float64))
    frame["in_largest_component"] = np.isfinite(scores["closeness"])
    con.register("score_np", frame)

    # NaN is how numpy says "not measured"; SQL says NULL, and a reader of the parquet expects SQL
    nulled = ",\n               ".join(
        f"CASE WHEN isnan(s.{c}) THEN NULL ELSE s.{c} END AS {c}"
        for c in ["pagerank", "eigenvector", "betweenness", "closeness"])
    con.execute(f"""
        CREATE OR REPLACE TABLE centrality AS
        SELECT s.person_id, nm.key, nm.name, nm.records, nm.first_year, nm.last_year,
               s.degree, s.core,
               {nulled},
               s.rank_degree, s.rank_eigenvector, s.rank_betweenness, s.rank_closeness,
               s.in_largest_component
        FROM score_np s
        LEFT JOIN node_meta nm ON nm.id = s.person_id""")
    rows = con.execute("SELECT count(*) FROM centrality").fetchone()[0]
    missing = con.execute("SELECT count(*) FROM centrality WHERE name IS NULL").fetchone()[0]
    if missing:
        log.warning("%s authors in the graph have no row in the nodes file", f"{missing:,}")

    parquet = out / f"{name}.centrality.parquet"
    table = out / f"{name}.centrality.tsv.gz"
    ordered = "SELECT * FROM centrality ORDER BY degree DESC, person_id"
    con.execute(f"COPY ({ordered}) TO '{parquet.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"COPY ({ordered}) TO '{table.as_posix()}' "
                f"(FORMAT CSV, DELIMITER '\t', HEADER true, COMPRESSION gzip)")
    log.info("wrote %s rows to %s and %s", f"{rows:,}", parquet.name, table.name)
    return {"rows": int(rows), "authors_without_a_name": int(missing),
            "files": [parquet.name, table.name]}


def leaders(con, top=TOP):
    """The top authors by each measure, which is what anybody actually wants to see."""
    out = {}
    for column in ["degree", "betweenness", "closeness", "eigenvector", "pagerank", "core"]:
        out[column] = [
            {"rank": i + 1, "name": r[0], "key": r[1], "records": r[2], "degree": r[3],
             "value": None if r[4] is None else float(r[4])}
            for i, r in enumerate(con.execute(f"""
                SELECT name, key, records, degree, {column} FROM centrality
                WHERE {column} IS NOT NULL
                ORDER BY {column} DESC, person_id LIMIT {int(top)}""").fetchall())]
    return out


def run(export_dir, out_dir, threads, seed, betweenness_samples=BETWEENNESS_SAMPLES,
        closeness_samples=CLOSENESS_SAMPLES, keep_temp=False, spot_check=None, resume=False):
    """The whole job: read the published dataset, measure it, write the results beside it.

    The expensive phase is checkpointed the moment it finishes, so `resume=True` re-does only the
    minutes of joining and file writing that follow it."""
    export_dir, out = Path(export_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()

    stats = _export_stats(export_dir)
    name = stats.get("name", config.NAME)
    edges_gz = _one(export_dir, f"{name}.*graph.txt.gz")
    nodes_gz = export_dir / f"{name}.nodes.txt.gz"
    claimed_nodes, claimed_edges = header_counts(edges_gz)
    log.info("reading %s (header claims %s nodes, %s edges)", edges_gz.name,
             f"{claimed_nodes:,}" if claimed_nodes else "?", f"{claimed_edges:,}" if claimed_edges else "?")

    resuming = resume and (out / CHECKPOINT).exists() and (out / CHECKPOINT_SUMMARY).exists()
    if resume and not resuming:
        log.warning("--resume was asked for but %s is not in %s; measuring from the start",
                    CHECKPOINT, out)
    # its own directory, so no other run can clean it up underneath this one
    scratch = Path(tempfile.mkdtemp(prefix="centrality-", dir=scratch_root()))
    con = D.plain_connection()
    try:
        # the names go into the database first and the file goes away immediately: nothing in the
        # hours that follow should depend on a temporary file still being there at the end
        nodes_plain = decompress(nodes_gz, scratch / "nodes.tsv")
        load_node_meta(con, nodes_plain)
        nodes_plain.unlink(missing_ok=True)

        if resuming:
            scores, person_id, back, summary, checked = load_checkpoint(out)
        else:
            edges_plain = decompress(edges_gz, scratch / "edges.tsv")
            graph_file = scratch / "graph.el"
            edges = load_edges(con, edges_plain, claimed_edges)
            nodes = remap(con, graph_file, claimed_nodes)
            person_id = np.asarray(
                con.execute("SELECT id FROM node ORDER BY nid").fetchnumpy()["id"], dtype=np.int64)
            G = load_graph(graph_file, nodes, edges)
            checked = check_degrees(con, G, seed=seed)
            # the edge table has done its work; holding 24 million rows for the next several hours
            # helps nobody
            for leftover in (edges_plain, graph_file):
                leftover.unlink(missing_ok=True)
            con.execute("DROP TABLE IF EXISTS e")
            con.execute("DROP TABLE IF EXISTS node")
            scores, H, back, summary = measures(G, threads, seed, betweenness_samples,
                                                closeness_samples, spot_check=spot_check)
            save_checkpoint(out, scores, person_id, back, summary, checked)
        written = write(con, scores, person_id, out, name)
        payload = {
            "name": f"{name}.centrality", "generated_at": stamp,
            "dataset": {"directory": export_dir.name, **{k: stats.get(k) for k in ("generated_at", "with_bins")}},
            "dump": stats.get("dump", {}), "rules": stats.get("rules", {}),
            **summary,
            "degree_cross_check_authors": int(checked),
            "resumed_from_checkpoint": bool(resuming),
            "correlations": correlations(scores, back),
            "top": leaders(con),
            **written,
            "total_seconds": round(time.time() - t0, 1),
        }
    finally:
        con.close()
        if not keep_temp:
            shutil.rmtree(scratch, ignore_errors=True)

    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "README.md").write_text(datasheet(payload), encoding="utf-8")
    log.info("centrality written to %s in %.0fs", out, time.time() - t0)
    return payload


def _one(directory: Path, pattern):
    found = sorted(directory.glob(pattern))
    if not found:
        raise FileNotFoundError(f"no file matching {pattern} in {directory}. Run "
                                f"`ml.network.cli export` first.")
    return found[0]


def _export_stats(export_dir: Path):
    path = export_dir / "stats.json"
    if not path.exists():
        log.warning("%s has no stats.json; the dataset's own rules cannot be repeated here", export_dir)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_export(models_dir=None):
    """The graph to measure. The unsuffixed directory is the full-corpus export, which is the one
    worth measuring; a scope-suffixed directory exists to reproduce somebody else's number, so it is
    only used if nothing else is there."""
    models_dir = Path(models_dir or base.MODELS_DIR)
    candidates = [d for d in models_dir.glob("network-*")
                  if d.is_dir() and any(d.glob("*graph.txt.gz"))]
    if not candidates:
        raise FileNotFoundError(
            f"no exported network in {models_dir}. Run `ml.network.cli export` first.")
    plain = [d for d in candidates if re.fullmatch(r"network-[0-9a-f]+", d.name)]
    chosen = max(plain or candidates, key=lambda d: d.stat().st_mtime)
    log.info("measuring %s", chosen.name)
    return chosen


# --------------------------------------------------------------------------- the datasheet

def _table(rows, columns):
    head = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n"
    body = "".join("| " + " | ".join(str(c) for c in row) + " |\n" for row in rows)
    return head + body


def _top_table(entries, value_label):
    return _table([(e["rank"], e["name"], f"{e['degree']:,}",
                    "-" if e["value"] is None else f"{e['value']:.3e}")
                   for e in entries[:10]], ["#", "Author", "Co-authors", value_label])


def datasheet(p):
    check = p.get("closeness_spot_check", {})
    convention = check.get("matching_convention")
    measured = (check.get("by_convention") or {}).get(convention, {})
    diameter = p.get("largest_component_diameter") or {}
    params = p.get("parameters", {})
    return f"""# {p['name']}

Centrality for every author in `{p['dataset'].get('directory')}`, computed from the published
dataset files rather than from a database - the same files anyone can download.

Generated {p['generated_at']} from the dblp dump of {p['dump'].get('latest_mdate')}
(fingerprint `{p['dump'].get('fingerprint')}`), scope `{p.get('rules', {}).get('scope')}`.

## The graph

| | |
|---|---|
| Authors (nodes) | {p['nodes']:,} |
| Co-authorships (edges) | {p['edges']:,} |
| Average degree | {p['average_degree']} |
| Highest degree | {p['max_degree']:,} |
| Largest k-core | {p['max_core']:,} |
| Connected components | {p.get('components', 0):,} |
| Largest component | {p.get('largest_component_nodes', 0):,} authors ({100 * p.get('largest_component_share', 0):.2f}%) |
| Diameter of that component | {diameter.get('lower', '?')}-{diameter.get('upper', '?')} hops (estimated, error {diameter.get('error', '?')}) |
| Global clustering coefficient | {p.get('approx_global_clustering_coefficient', '?')} (approximate, error {p.get('clustering_error', '?')}) |

A graph this clustered with a diameter this small is the textbook small-world shape, which is the
first thing to check and the least surprising thing here.

## What was computed, and how exactly

| Measure | Exact? | Scope | Method |
|---|---|---|---|
| Degree | yes | whole graph | one pass over the edges |
| Core number | yes | whole graph | k-core decomposition |
| PageRank | to {params.get('pagerank', {}).get('tolerance')} | whole graph | power iteration, damping {params.get('pagerank', {}).get('damping')} |
| Eigenvector | to {params.get('eigenvector', {}).get('tolerance')} | largest component | power iteration |
| Betweenness | **no** | largest component | {params.get('betweenness', {}).get('samples', 0):,} sampled sources (Riondato-Kornaropoulos) |
| Closeness | **no** | largest component | {params.get('closeness', {}).get('samples', 0):,} sampled sources (Eppstein-Wang) |

Exact betweenness is O(n·m) - about 10^14 operations on this graph - so it is estimated from
sampled sources. The estimator is unbiased; the sample count above is the entire error story, and
nothing here pretends otherwise.

Distance-based measures ignore the edge weights. A shortest-path algorithm treats a weight as a
length, and "number of papers together" is the opposite of a length: using it would place frequent
collaborators far apart. Every distance here is a number of hops.

Eigenvector centrality and closeness are computed on the largest connected component. The leading
eigenvector of a disconnected graph lives on one component and is zero elsewhere for reasons of
arithmetic rather than of collaboration, and the distance to an unreachable author is not a number.
Authors outside that component have NULL for both, not zero. PageRank, which is defined on a
disconnected graph, is given for everybody.

## How the numbers were checked

* **Edge count against the dataset's own header**: the header claims a count, and the load fails
  unless that many edges were read. A reader that stops at the first member of a multi-member gzip
  stream would otherwise report a smaller graph and no error at all.
* **Degree, twice**: computed by the graph library on the loaded graph and by DuckDB on the source
  file, for {p.get('degree_cross_check_authors', 0):,} randomly sampled authors. They agree. This is
  the check that the renumbering from sparse dblp ids to 0..n-1 is right, which is the one mistake
  here that would produce plausible centralities for the wrong people.
* **Closeness against exact shortest paths**: {check.get('sampled_authors', 0):,} authors chosen at
  random, one exact single-source shortest-path run each, compared with what the approximation said
  about those same authors. Matching convention `{convention}`, median ratio
  {measured.get('median_ratio', '?')}, 90th-percentile relative error
  {measured.get('p90_relative_error', '?')}.

## What the measures say about each other

{_table([(k, v) for k, v in (p.get('correlations', {}).get('spearman') or {}).items()],
        ['Pair', 'Spearman rho'])}
Rank correlation over the largest component. Overlap of the top {p.get('correlations', {}).get('top', 100)}
lists, which is the question a correlation cannot answer:

{_table([(k, v) for k, v in (p.get('correlations', {}).get('top_overlap') or {}).items()],
        ['Pair', 'Authors in both lists'])}

## Most central authors

By degree - the most co-authors:

{_top_table(p['top']['degree'], 'Degree')}
By betweenness - on the most shortest paths between other people:

{_top_table(p['top']['betweenness'], 'Betweenness')}
By closeness - fewest hops to everybody else:

{_top_table(p['top']['closeness'], 'Closeness')}
By eigenvector - co-author to the well-connected:

{_top_table(p['top']['eigenvector'], 'Eigenvector')}
## Files

| File | Contents |
|---|---|
| `{p['name'].split('.')[0]}.centrality.tsv.gz` | one row per author, tab separated, with a header line |
| `{p['name'].split('.')[0]}.centrality.parquet` | the same table, for the dashboard |
| `metrics.json` | every number above, including the parameters and the checks |
| `scores.npz` | the raw measures, saved before any joining; `centrality --resume` rebuilds the files above from it in a minute rather than repeating the hours |

Columns: `person_id`, `key`, `name`, `records`, `first_year`, `last_year`, `degree`, `core`,
`pagerank`, `eigenvector`, `betweenness`, `closeness`, the four `rank_*` columns (1 is most
central, 0 means not ranked because the author is outside the largest component), and
`in_largest_component`.

Run time: {p.get('total_seconds', 0):,.0f}s on {params.get('threads')} threads, seed
{params.get('seed')}. Per-measure timings are in `metrics.json`.
"""
