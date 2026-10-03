"""
The co-authorship network, for the assistant.

`ml.network.cli centrality` measures every author's position in the graph and writes it beside the
models. Nothing was reading it, so three perfectly good questions had no answer and - worse - the
model would reach for `top_authors` and confidently answer a paper-count question instead:

  * who is the most central author in computer science?
  * how central is this particular person?
  * how connected is the field - how many degrees of separation?

Three things this module is careful about, because all three are ways to be confidently wrong:

**Central is not prolific.** Degree, betweenness, closeness and eigenvector rank people by position
in the collaboration graph, not by output. The two disagree: on this dump only 58 of the 100
most-between authors are also in the 100 most-connected. Every answer here says which it is.

**A raw betweenness score means nothing to a reader.** 5.3e-03 is not an answer. What is reportable
is the rank and the percentile, so those are what come back, with the value alongside for anyone who
wants it.

**Two of the four measures are estimates.** Betweenness and closeness are sampled, and the sample
counts and measured error live in the run's own metrics file - so they are read from there and put in
the note rather than quietly dropped.

The measures belong to one dump. If the centrality run is missing, or was computed for a different
snapshot, this refuses and says so instead of answering from stale numbers.
"""
import json
import logging

from . import config
from .tools import as_key, refusal, result, rows_of

log = logging.getLogger("dblp.chat.network")

MEASURES = {
    "betweenness": "how often an author lies on the shortest path between two others - the brokers "
                   "between research communities",
    "degree": "how many distinct co-authors an author has",
    "closeness": "how few hops an author is from everybody else",
    "eigenvector": "how well connected an author's co-authors are, recursively",
    "pagerank": "the same idea as eigenvector, but defined for authors outside the main component too",
    "core": "how deep inside a densely collaborating group an author sits",
}
SAMPLED = ("betweenness", "closeness")
RANKED = ("degree", "betweenness", "closeness", "eigenvector")


def _dir(ctx):
    """The centrality run for exactly this dump. A run for another snapshot is not a near-enough
    answer: the author ids are a different renumbering and the ranks are of a different graph."""
    # the dump's fingerprint lives in the pool's metadata; a guess at an attribute here once made
    # every network tool refuse forever while reading like "not computed yet"
    fingerprint = (ctx.meta or {}).get("fingerprint")
    if not fingerprint:
        return None
    d = config.MODELS_DIR / f"centrality-{fingerprint}"
    return d if (d / "metrics.json").exists() else None


def _metrics(directory):
    return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))


def _table(directory):
    found = sorted(directory.glob("*.centrality.parquet"))
    return found[0] if found else None


def _unavailable():
    return refusal(
        "the co-authorship network has not been measured for this snapshot yet",
        "ask about publication counts instead - those come straight from the dump")


def _note(metrics, measure=None):
    """What the number is, and how sure it is. The sampled measures carry their sample count and the
    error that was actually measured against exact shortest paths, not a promised one."""
    parts = ["Measured on the co-authorship graph of every record type, papers with 2-50 authors, "
             "disambiguation bins excluded."]
    if measure in SAMPLED:
        samples = (metrics.get("parameters", {}).get(measure) or {}).get("samples")
        if samples:
            parts.append(f"{measure.title()} is estimated from {samples:,} sampled sources, not "
                         f"computed exactly - exact betweenness on this graph would take days.")
        if measure == "closeness":
            check = (metrics.get("closeness_spot_check") or {})
            best = (check.get("by_convention") or {}).get(check.get("matching_convention")) or {}
            if best.get("p90_relative_error") is not None:
                parts.append(f"Checked against exact shortest paths from "
                             f"{check.get('sampled_authors', 0)} authors: 90% of values were within "
                             f"{100 * best['p90_relative_error']:.1f}%.")
    if measure in ("betweenness", "closeness", "eigenvector"):
        parts.append("Authors outside the largest connected component are not ranked on this measure.")
    return " ".join(parts)


def network_shape(ctx):
    """The shape of the whole collaboration graph - the 'six degrees of separation' question."""
    directory = _dir(ctx)
    if directory is None:
        return _unavailable()
    m = _metrics(directory)
    diameter = m.get("largest_component_diameter") or {}
    facts = [
        ("Authors with at least one co-author", f"{m['nodes']:,}"),
        ("Co-authorships between them", f"{m['edges']:,}"),
        ("Average co-authors per author", f"{m['average_degree']:,}"),
        ("Most co-authors anyone has", f"{m['max_degree']:,}"),
        ("Separate groups that never connect", f"{m.get('components', 0):,}"),
        ("Share of authors in the single largest group",
         f"{100 * m.get('largest_component_share', 0):.1f}%"),
        ("Degrees of separation across that group",
         f"{diameter.get('upper', '?')} hops at most (estimated)"),
        ("Clustering coefficient",
         f"{m.get('approx_global_clustering_coefficient', '?')} - how often two of your co-authors "
         f"have also written together"),
    ]
    # dicts keyed by column, like every other tool: the assistant panel renders row[column]
    rows = [{"measure": measure, "value": value} for measure, value in facts]
    return result(
        "The shape of the dblp co-authorship network.", ["measure", "value"], rows,
        note=_note(m) + " High clustering with a small diameter is the classic small-world pattern: "
                        "collaboration is intensely local, yet almost everybody is a short chain of "
                        "co-authors from almost everybody else.",
        link={"page": "network"})


def central_authors(ctx, metric=None, limit=10):
    """The most central authors by one measure. Not the most prolific - that is top_authors.

    There is deliberately no default measure. Betweenness used to be it, and "the most connected
    authors" is a degree question - so a call that left the measure out would have ranked brokers
    and presented them as hubs, confidently, with nothing downstream able to tell."""
    if not metric:
        return refusal("centrality has several measures and they rank different people",
                       "call again with metric = degree (most co-authors), betweenness (bridges "
                       "between communities), closeness (fewest hops to everyone) or eigenvector "
                       "(co-author to the well connected)")
    if metric not in MEASURES:
        return refusal(f"'{metric}' is not one of the centrality measures",
                       "choose " + ", ".join(sorted(MEASURES)))
    directory = _dir(ctx)
    table = _table(directory) if directory else None
    if table is None:
        return _unavailable()
    m = _metrics(directory)
    cur = ctx.cursor()
    # `metric` is interpolated, so it is checked against MEASURES above before it gets here. Only the
    # four course measures carry a precomputed rank column; pagerank and core do not.
    ranked = ", rank_{0} AS rank".format(metric) if metric in RANKED else ""
    cols, rows = rows_of(cur, f"""
        SELECT name, key, degree AS co_authors, records AS papers, {metric} AS value{ranked}
        FROM read_parquet('{table.as_posix()}')
        WHERE {metric} IS NOT NULL
        ORDER BY {metric} DESC, person_id
        LIMIT ?""", [int(limit)])
    return result(
        f"Authors ranked by {metric} centrality - {MEASURES[metric]}.", cols, rows,
        note=_note(m, metric) + " This ranks position in the collaboration network, not output: "
                                "the most central author is usually not the most prolific one.",
        link={"page": "network"})


def author_centrality(ctx, key):
    """One author's position in the network, as a rank and a percentile rather than a raw score."""
    directory = _dir(ctx)
    table = _table(directory) if directory else None
    if table is None:
        return _unavailable()
    m = _metrics(directory)
    cur = ctx.cursor()
    key = as_key(cur, key)
    got = cur.execute(f"""
        SELECT name, key, degree, records, in_largest_component,
               rank_degree, rank_betweenness, rank_closeness, rank_eigenvector,
               betweenness, closeness, eigenvector
        FROM read_parquet('{table.as_posix()}') WHERE key = ?""", [key]).fetchone()
    if not got:
        # two different situations, and saying the second when it was the first put a false claim -
        # "he has only single-author papers" - into an answer about a man with 43 co-authors
        page = cur.execute("SELECT name FROM s.persons WHERE key = ?", [key]).fetchone()
        if not page:
            return refusal(f"there is no author page with the key {key!r}",
                           "call resolve_author with the person's name to get the right key - never "
                           "guess one, and do not guess why the lookup failed")
        return refusal(f"{page[0]} has no co-authors in the network",
                       "every paper on their page is single-author, on a paper with more than 50 "
                       "authors, or shared only with unidentified names - so they have no position in "
                       "the co-authorship graph")
    total = int(m["nodes"])
    ranked = int(m.get("largest_component_nodes", total))
    rows = []
    for measure, rank, value, population in (
            ("degree", got[5], got[2], total),
            ("betweenness", got[6], got[9], ranked),
            ("closeness", got[7], got[10], ranked),
            ("eigenvector", got[8], got[11], ranked)):
        if not rank:
            rows.append({"measure": measure, "rank": "not ranked", "value": "-",
                         "meaning": "outside the largest connected component"})
            continue
        rows.append({"measure": measure, "rank": f"{int(rank):,} of {population:,}",
                     "value": f"{value:.4g}" if value is not None else "-",
                     "meaning": f"more central than {100 * (1 - int(rank) / population):.2f}% of authors"})
    return result(
        f"{got[0]} in the co-authorship network: {got[2]:,} co-authors across {got[3]:,} records.",
        ["measure", "rank", "value", "meaning"], rows,
        note=_note(m, "betweenness") + " A rank is the honest form of these numbers: the raw scores "
                                       "are tiny fractions that mean nothing on their own.",
        link={"page": "authors", "key": key})   # the person's own page, as every author tool does
