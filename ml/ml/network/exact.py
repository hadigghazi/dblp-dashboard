"""
Exact betweenness, where exact is possible - and what that says about the two shortcuts.

Exact Brandes betweenness is O(n*m). On the largest component here, 3.77 million authors and 24.1
million edges, that is about 9*10^13 operations: weeks. Nobody disputes that. The question is what to
do instead, and there are two answers that get confused with each other:

  1. **Sample the sources.** Run Brandes from a random sample of sources and scale up. The graph is
     the whole graph, the measure is the real measure, and the only cost is sampling error, which
     shrinks with the sample count. This is what `centrality` does.
  2. **Shrink the graph.** Compute exact betweenness on a subgraph - an ego network, a k-core, one
     venue. Cheap, exact... and a different quantity. Betweenness counts shortest paths between all
     pairs; delete four fifths of the graph and most of those paths no longer exist. A node's
     betweenness inside its ego network is not its betweenness in computer science, and no amount of
     exactness makes it so.

Both are defensible; they answer different questions. This module measures the difference instead of
arguing about it, on the real data, in three steps:

  * take the densest k-core that exact Brandes can still finish, and compute it **exactly**;
  * run the *estimator* on that same subgraph, at the same sample count and at the same sampling
    rate as the full run, and compare it with the exact answer. Same graph, same measure, so this is
    a real validation of the sampling - the only one available at this scale;
  * compare that exact subgraph betweenness against the full-graph estimate for the same authors.
    Same authors, same algorithm, different graph. Whatever disagreement shows up is the price of
    shrinking the graph, and it is the number to put in front of anyone who suggests it.

Ego betweenness gets the same treatment, being the cheapest shortcut of all: exact within one hop,
and compared against the full-graph estimate for the same people. Everett and Borgatti found the two
correlate well on small networks; whether that holds on four million authors is measurable here.

This reads the finished `centrality` output rather than recomputing it, because the step it would
recompute is the one that takes hours.
"""
import json
import logging
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import config as base, data as D
from . import centrality as CE, config

log = logging.getLogger("dblp.ml.network.exact")

MAX_EXACT_NODES = 50_000    # what exact Brandes can finish in minutes rather than days
EGO_RANDOM = 500
EGO_TOP = 50
TOP = 100


def _spearman(a, b):
    """Rank correlation. Both arrays are ranked once, then correlated - Spearman's rho is Pearson's
    r on the ranks."""
    from scipy.stats import rankdata
    if len(a) < 3:
        return None
    return round(float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]), 4)


def agreement(reference, other, top=TOP, label=""):
    """How much two rankings of the same authors agree. A correlation alone hides the thing people
    actually care about, which is whether the same names come out on top, so the overlap of the top
    lists is reported beside it."""
    reference, other = np.asarray(reference, dtype=np.float64), np.asarray(other, dtype=np.float64)
    # one missing value would turn the correlation into NaN and report nothing; drop the pair instead
    both = np.isfinite(reference) & np.isfinite(other)
    dropped = int((~both).sum())
    reference, other = reference[both], other[both]
    order_a = np.argsort(-reference, kind="stable")
    order_b = np.argsort(-other, kind="stable")
    top = min(top, len(reference))
    first, second = set(order_a[:top].tolist()), set(order_b[:top].tolist())
    out = {"authors": int(len(reference)), "spearman": _spearman(reference, other),
           "top": int(top), "top_overlap": len(first & second),
           "top_overlap_share": round(len(first & second) / top, 3) if top else None,
           "same_most_central_author": bool(len(order_a) and order_a[0] == order_b[0])}
    if dropped:
        out["authors_without_a_value"] = dropped
    if label:
        out["comparing"] = label
    return out


def tractable_core(H, max_nodes=MAX_EXACT_NODES):
    """The densest k-core that exact Brandes can still finish.

    k is chosen by measurement rather than guessed: the core sizes are known once the decomposition
    has run, so this takes the largest k whose core fits in the budget. Picking a number out of the
    air would mean either an unnecessarily small subgraph or a run that never ends."""
    import networkit as nk
    scores = np.asarray(nk.centrality.CoreDecomposition(H).run().scores(), dtype=np.int64)
    sizes = {int(k): int((scores >= k).sum()) for k in range(1, int(scores.max()) + 1)}
    fits = [k for k, size in sizes.items() if size <= max_nodes and size > 2]
    if not fits:
        raise RuntimeError(f"no k-core of the component fits in {max_nodes:,} nodes; raise --max-nodes")
    k = min(fits)                       # the smallest such k keeps the largest subgraph that fits
    members = np.flatnonzero(scores >= k).tolist()
    sub = nk.graphtools.subgraphFromNodes(H, members)
    ids = nk.graphtools.getContinuousNodeIds(sub)
    C = nk.graphtools.getCompactedGraph(sub, ids)
    back = np.full(C.numberOfNodes(), -1, dtype=np.int64)
    for old, new in ids.items():
        if new < len(back):
            back[new] = old
    log.info("%s-core: %s authors, %s edges (exact Brandes is affordable here)",
             k, f"{C.numberOfNodes():,}", f"{C.numberOfEdges():,}")
    return C, back, {"k": int(k), "nodes": int(C.numberOfNodes()), "edges": int(C.numberOfEdges()),
                     "core_sizes": {str(kk): vv for kk, vv in sorted(sizes.items()) if vv > 2}}


def ego_betweenness(H, nodes):
    """Each author's betweenness inside their own ego network - the cheapest shortcut there is, and
    exact within that one hop. Real ego networks are sparse (co-authors of one person are mostly not
    co-authors of each other), so this is quick even for the biggest hubs."""
    import networkit as nk
    out = np.zeros(len(nodes), dtype=np.float64)
    t0 = time.time()
    for i, u in enumerate(nodes):
        members = [int(u)] + [int(v) for v in H.iterNeighbors(int(u))]
        if len(members) < 3:
            continue
        sub = nk.graphtools.subgraphFromNodes(H, members)
        ids = nk.graphtools.getContinuousNodeIds(sub)
        ego = nk.graphtools.getCompactedGraph(sub, ids)
        scores = nk.centrality.Betweenness(ego, True).run().scores()
        out[i] = float(scores[ids[int(u)]])
    log.info("ego betweenness for %s authors in %.0fs", f"{len(nodes):,}", time.time() - t0)
    return out


def sampled_betweenness(C, samples, label):
    import networkit as nk
    t0 = time.time()
    scores = np.asarray(nk.centrality.EstimateBetweenness(C, int(samples), True, True).run().scores(),
                        dtype=np.float64)
    seconds = round(time.time() - t0, 1)
    log.info("estimate on the subgraph, %s (%s samples): %.0fs", label, f"{int(samples):,}", seconds)
    return scores, seconds


def full_graph_betweenness(con, parquet: Path, nodes):
    """The finished full-graph estimate, back in graph order. Joined on the dblp author id rather
    than on a row number, so it cannot be silently misaligned by a renumbering."""
    got = con.execute(f"""
        SELECT n.nid, c.betweenness, c.degree
        FROM node n JOIN read_parquet('{parquet.as_posix()}') c ON c.person_id = n.id
        WHERE c.betweenness IS NOT NULL""").fetchnumpy()
    # DuckDB hands back masked arrays for anything it considers nullable; coerce before indexing
    nid = np.asarray(got["nid"], dtype=np.int64)
    values = np.full(nodes, np.nan, dtype=np.float64)
    degrees = np.zeros(nodes, dtype=np.int64)
    values[nid] = np.asarray(got["betweenness"], dtype=np.float64)
    degrees[nid] = np.asarray(got["degree"], dtype=np.int64)
    measured = int(np.isfinite(values).sum())
    log.info("read the full-graph estimate for %s authors", f"{measured:,}")
    if not measured:
        raise RuntimeError(f"{parquet} holds no betweenness values; has the centrality run finished?")
    return values, degrees


def run(export_dir, centrality_dir, out_dir, max_nodes=MAX_EXACT_NODES, ego_random=EGO_RANDOM,
        ego_top=EGO_TOP, threads=None, seed=None, keep_temp=False):
    import networkit as nk
    export_dir, centrality_dir = Path(export_dir), Path(centrality_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = base.SEED if seed is None else int(seed)
    threads = int(threads or base.DUCKDB_THREADS)
    nk.setNumberOfThreads(threads)
    nk.setSeed(seed, False)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()

    metrics = json.loads((centrality_dir / "metrics.json").read_text(encoding="utf-8"))
    full_samples = int(metrics["parameters"]["betweenness"]["samples"])
    name = config.NAME
    parquet = centrality_dir / f"{name}.centrality.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"no {parquet.name} in {centrality_dir}; run `centrality` first")

    found = sorted(export_dir.glob(f"{name}.*graph.txt.gz"))
    if not found:
        raise FileNotFoundError(f"no edge list in {export_dir}; run `export` first")
    edges_gz = found[0]
    claimed_nodes, claimed_edges = CE.header_counts(edges_gz)
    # its own directory, outside the database's shared temp space, for the reason recorded in
    # centrality.scratch_root: a scratch path shared between two runs cost six hours once already
    tmp = Path(tempfile.mkdtemp(prefix="exact-", dir=CE.scratch_root()))
    con = D.plain_connection()
    try:
        plain = CE.decompress(edges_gz, tmp / "edges.tsv")
        edges = CE.load_edges(con, plain, claimed_edges)
        nodes = CE.remap(con, tmp / "graph.el", claimed_nodes)
        G = CE.load_graph(tmp / "graph.el", nodes, edges)
        estimate_by_nid, degree_by_nid = full_graph_betweenness(con, parquet, nodes)
    finally:
        con.close()
        if not keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)

    H, back_H, component = CE.largest_component(G)
    estimate_on_H = estimate_by_nid[back_H]
    degree_on_H = degree_by_nid[back_H]

    C, back_C, core = tractable_core(H, max_nodes)
    log.info("exact Brandes on %s authors and %s edges: this is the step that cannot be done on the "
             "whole graph", f"{core['nodes']:,}", f"{core['edges']:,}")
    t = time.time()
    exact = np.asarray(nk.centrality.Betweenness(C, True).run().scores(), dtype=np.float64)
    exact_seconds = round(time.time() - t, 1)
    log.info("exact betweenness on the subgraph: %.0fs", exact_seconds)
    # what the same work would cost on the whole component, at the same cost per source
    projected_days = exact_seconds / core["nodes"] * component["largest_component_nodes"] \
        * (component["largest_component_edges"] / core["edges"]) / 86400

    same_count, count_seconds = sampled_betweenness(C, full_samples, "same sample count as the full run")
    rate = full_samples / component["largest_component_nodes"]
    same_rate_samples = max(2, round(rate * core["nodes"]))
    same_rate, rate_seconds = sampled_betweenness(C, same_rate_samples, "same sampling rate as the full run")

    ego_nodes = _ego_sample(H, degree_on_H, ego_random, ego_top, seed)
    ego = ego_betweenness(H, ego_nodes)

    payload = {
        "generated_at": stamp,
        "dataset": {"export": export_dir.name, "centrality": centrality_dir.name},
        "dump": metrics.get("dump", {}),
        "graph": {"nodes": int(G.numberOfNodes()), "edges": int(G.numberOfEdges()),
                  **{k: component[k] for k in ("largest_component_nodes", "largest_component_edges")}},
        "exact_subgraph": {**core, "exact_seconds": exact_seconds,
                           "projected_days_for_the_whole_component": round(projected_days, 1)},
        "full_run": {"samples": full_samples, "sampling_rate": round(rate, 6),
                     "seconds": metrics.get("seconds", {}).get("betweenness")},
        # same graph, same measure: this is the validation of the sampling
        "sampling_against_exact": {
            "same_sample_count": {"samples": full_samples, "seconds": count_seconds,
                                  **agreement(exact, same_count, label="exact vs estimate, same graph")},
            "same_sampling_rate": {"samples": int(same_rate_samples), "seconds": rate_seconds,
                                   **agreement(exact, same_rate, label="exact vs estimate, same graph, "
                                                                       "same rate as the full run")},
        },
        # same authors, same measure, different graph: this is the price of shrinking the graph
        "shrinking_the_graph": agreement(
            estimate_on_H[back_C], exact, label="full-graph estimate vs exact on the k-core, same authors"),
        "ego_networks": {
            "authors": int(len(ego_nodes)),
            "random": int(ego_random), "top_by_degree": int(ego_top),
            "degree_range": [int(degree_on_H[ego_nodes].min()), int(degree_on_H[ego_nodes].max())],
            **agreement(estimate_on_H[ego_nodes], ego, top=min(TOP, len(ego_nodes)),
                        label="full-graph estimate vs ego betweenness, same authors"),
        },
        "parameters": {"max_exact_nodes": int(max_nodes), "threads": threads, "seed": seed},
        "total_seconds": round(time.time() - t0, 1),
    }
    (out / "exact-betweenness.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "EXACT-BETWEENNESS.md").write_text(report(payload), encoding="utf-8")
    log.info("written to %s in %.0fs", out, time.time() - t0)
    return payload


def _ego_sample(H, degree, random_n, top_n, seed):
    """Random authors, plus the biggest hubs. Random alone would be almost all small-degree authors,
    and the hubs are the interesting case for a local measure: if ego betweenness tracks the real
    thing anywhere it should be there."""
    rng = np.random.default_rng(seed)
    n = H.numberOfNodes()
    chosen = set(rng.choice(n, size=min(random_n, n), replace=False).tolist())
    chosen.update(np.argsort(-degree, kind="stable")[:top_n].tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def report(p):
    exact, sampling, shrink, ego = (p["exact_subgraph"], p["sampling_against_exact"],
                                    p["shrinking_the_graph"], p["ego_networks"])
    count, rate = sampling["same_sample_count"], sampling["same_sampling_rate"]
    return f"""# Exact betweenness, where exact is possible

Generated {p['generated_at']} from `{p['dataset']['export']}` and `{p['dataset']['centrality']}`
(dump `{p['dump'].get('fingerprint')}`).

Exact Brandes betweenness is O(n*m). On the largest component - {p['graph']['largest_component_nodes']:,}
authors and {p['graph']['largest_component_edges']:,} edges - that is weeks of computation, and the
measured cost here says how many: exact betweenness on the {exact['k']}-core took
{exact['exact_seconds']:,.0f}s for {exact['nodes']:,} authors, which scales to about
**{exact['projected_days_for_the_whole_component']:,.1f} days** for the whole component. So it is not
done exactly. There are two ways not to do it, and they are not equivalent.

## 1. Sampling the sources: same graph, same measure

Brandes from {p['full_run']['samples']:,} random sources instead of all
{p['graph']['largest_component_nodes']:,} of them. The estimator is unbiased and the graph is intact,
so the only cost is sampling error. Measured on the {exact['k']}-core, against the exact answer for
the same graph:

| | Samples | Spearman rho | Top {count['top']} overlap | Same most central author |
|---|---|---|---|---|
| Same sample count as the full run | {count['samples']:,} | {count['spearman']} | {count['top_overlap']}/{count['top']} | {'yes' if count['same_most_central_author'] else 'no'} |
| Same sampling rate as the full run | {rate['samples']:,} | {rate['spearman']} | {rate['top_overlap']}/{rate['top']} | {'yes' if rate['same_most_central_author'] else 'no'} |

The second row is the honest one. The full run samples {100 * p['full_run']['sampling_rate']:.3f}% of
sources, and a fixed sample count applied to a subgraph is a far higher rate than that, which would
flatter the estimator. Both are given.

## 2. Shrinking the graph: same measure, different graph

Exact betweenness on the {exact['k']}-core, against the full-graph estimate **for those same
{shrink['authors']:,} authors**:

| | |
|---|---|
| Spearman rho | {shrink['spearman']} |
| Top {shrink['top']} overlap | {shrink['top_overlap']}/{shrink['top']} ({100 * (shrink['top_overlap_share'] or 0):.0f}%) |
| Same most central author | {'yes' if shrink['same_most_central_author'] else 'no'} |

Both numbers are exact or near-exact computations of betweenness. They disagree because betweenness
counts shortest paths between **all pairs**, and a subgraph has thrown most of those pairs away. An
author who bridges two research communities loses exactly the paths that made them a bridge, if the
subgraph keeps only one of the communities. This is not an error in either computation; it is what
the measure means.

## 3. Ego networks: the cheapest shortcut

Betweenness within one's own ego network, exactly, for {ego['authors']:,} authors
({ego['random']:,} random plus the {ego['top_by_degree']} largest hubs; degrees
{ego['degree_range'][0]:,} to {ego['degree_range'][1]:,}), against the full-graph estimate for the
same people:

| | |
|---|---|
| Spearman rho | {ego['spearman']} |
| Top {ego['top']} overlap | {ego['top_overlap']}/{ego['top']} |

Everett and Borgatti found ego betweenness correlates well with the real thing on small networks.
The number above is whether that holds on four million authors.

## What this means for the deliverable

Sampling keeps the graph and pays in sampling error, which is measurable and shrinks with the sample
count. Restricting the graph keeps exactness and pays in **answering a different question**, which no
amount of computation fixes. The full-graph estimate is therefore the headline result; the exact
subgraph numbers above are the evidence that it can be trusted, and the evidence for why the cheaper
route was not taken.

Total run time {p['total_seconds']:,.0f}s on {p['parameters']['threads']} threads, seed
{p['parameters']['seed']}.
"""
