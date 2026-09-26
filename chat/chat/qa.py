"""
Quality assurance for the tools, without a language model in the loop.

The gold set measures whether the model reaches for the right tool. It cannot tell whether the tool
*told the truth*: "how many people are called Wei Wang" picked exactly the right tool and answered
39 instead of 522, because the tool counted the rows it could show instead of the people that exist.
That class of bug is invisible to an end-to-end chat test and obvious to a cross-check.

So this runs the tools directly and checks three things:

  AGAINST THE DASHBOARD  the same number, computed by the api's own endpoint. The dashboard is the
                         reference implementation: if the assistant and the page disagree, the
                         assistant is wrong.
  INVARIANTS             properties that must hold whatever the dump says - a capped list never
                         becomes a count, a filter never widens a result, an empty answer never
                         reads as zero, the same question twice gives the same number.
  BEHAVIOUR              every tool runs, refuses cleanly on bad input, stays inside its latency
                         budget, and the SQL guard rejects what it must.

Free to run (no model calls) and safe on production data (read-only), so it can go in front of every
release and after every new dump.
"""
import logging
import time

import httpx

from . import config, docs, sqlguard, store, tools as T

log = logging.getLogger("dblp.chat.qa")

SLOW_SECONDS = 4.0
CHECKS = []


def check(name, group="invariant"):
    def register(fn):
        CHECKS.append({"name": name, "group": group, "fn": fn})
        return fn
    return register


class Skip(Exception):
    """The reference is unavailable; not a failure of the thing under test."""


def dash(ctx, path, **params):
    """A dashboard endpoint, the reference for any number the assistant also computes."""
    url = f"{config.DASHBOARD_URL}/api/{path}"
    try:
        r = ctx.http.get(url, params=params, timeout=config.UPSTREAM_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        raise Skip(f"dashboard api unreachable at {url}: {e}") from e
    body = r.json()
    return body.get("data", body)


def call(ctx, tool, **args):
    """`tool`, not `name`: several tools take a `name` argument of their own."""
    out = T.call(ctx, tool, args)
    if out.get("refused"):
        raise AssertionError(f"{tool} refused: {out.get('summary')}")
    return out


def near(a, b, tol=0):
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


# =========================================================== against the dashboard
@check("dataset_facts matches the overview page", "dashboard")
def _facts(ctx):
    f = {r["measure"]: r["value"] for r in call(ctx, "dataset_facts")["rows"]}
    ov = dash(ctx, "overview")["counts"]
    pairs = [("publications", "publications"), ("author pages", "author_pages"),
             ("disambiguation bins", "bins"), ("numbered pages", "numbered"), ("records", "records")]
    bad = [f"{mine}={f.get(mine)} vs page {ov.get(theirs)}"
           for mine, theirs in pairs if int(f.get(mine, -1)) != int(ov.get(theirs, -2))]
    assert not bad, "; ".join(bad)
    return f"{f['publications']:,} publications, {f['author pages']:,} author pages"


@check("most_shared_names matches the identity page", "dashboard")
def _homonyms(ctx):
    mine = call(ctx, "most_shared_names", limit=5)["rows"]
    theirs = dash(ctx, "identity/homonyms", top=5)
    assert [r["base_name"] for r in mine] == [r["base_name"] for r in theirs], "different names"
    assert [r["people"] for r in mine] == [r["distinct_people"] for r in theirs], "different counts"
    return f"top name {mine[0]['base_name']} with {mine[0]['people']:,} people"


@check("namesakes counts the same people the page does", "dashboard")
def _namesakes(ctx):
    """The Wei Wang bug: the tool must report the population, not the rows it can show."""
    top = dash(ctx, "identity/homonyms", top=1)[0]
    out = call(ctx, "namesakes", name=top["base_name"])
    assert out["meta"]["numbered_pages"] == top["distinct_people"], \
        f"namesakes says {out['meta']['numbered_pages']}, the page says {top['distinct_people']}"
    assert str(top["distinct_people"]) in out["summary"].replace(",", ""), \
        "the real count is missing from the summary the model reads"
    assert len(out["rows"]) <= out["meta"]["pages"], "more rows than pages"
    return f"{top['base_name']}: {top['distinct_people']:,} numbered pages, {len(out['rows'])} listed"


@check("top_authors #1 matches the long-tail maximum", "dashboard")
def _top_author(ctx):
    mine = call(ctx, "top_authors", limit=1)["rows"][0]
    tail = dash(ctx, "tails", panel="papers_per_author")["stats"]
    assert near(mine["papers"], tail["max"]), f"leaderboard {mine['papers']} vs tail max {tail['max']}"
    return f"{mine['name']} with {mine['papers']:,} records"


@check("papers per year matches the publishing page", "dashboard")
def _growth(ctx):
    last = ctx.last_full_year()
    mine = {r["year"]: r["value"] for r in call(ctx, "papers_timeseries", metric="papers",
                                                frm=last - 5, to=last)["rows"]}
    theirs = {r["year"]: (r["conference"] or 0) + (r["journal"] or 0)
              for r in dash(ctx, "publishing/growth", **{"from": last - 5, "to": last})}
    bad = [f"{y}: {mine.get(y)} vs {theirs.get(y)}" for y in theirs if mine.get(y) != theirs.get(y)]
    assert not bad, "; ".join(bad)
    return f"{len(theirs)} years agree, {theirs[last]:,} in {last}"


@check("mean authors per year matches the publishing page", "dashboard")
def _teams(ctx):
    last = ctx.last_full_year()
    mine = {r["year"]: r["value"] for r in call(ctx, "papers_timeseries", metric="mean_authors",
                                                frm=last - 3, to=last)["rows"]}
    theirs = {r["year"]: r["mean_authors"] for r in dash(ctx, "publishing/teams",
                                                         **{"from": last - 3, "to": last})}
    bad = [f"{y}: {mine.get(y)} vs {round(theirs[y], 3)}" for y in theirs
           if not near(mine.get(y), theirs[y], 0.002)]
    assert not bad, "; ".join(bad)
    return f"{len(theirs)} years agree, {theirs[last]:.2f} authors in {last}"


@check("an author's profile matches their explorer page", "dashboard")
def _author(ctx):
    top = call(ctx, "top_authors", limit=1)["rows"][0]
    mine = call(ctx, "author_profile", key=top["key"])
    theirs = dash(ctx, "authors/detail", key=top["key"])["stats"]
    assert mine["meta"]["stats"]["papers"] == theirs["papers"], \
        f"papers {mine['meta']['stats']['papers']} vs {theirs['papers']}"
    assert mine["meta"]["coauthors"] == theirs["coauthors"], \
        f"co-authors {mine['meta']['coauthors']} vs {theirs['coauthors']}"
    co = call(ctx, "coauthors", key=top["key"], limit=3)
    assert co["meta"]["coauthors_total"] == theirs["coauthors"], \
        f"coauthors tool total {co['meta']['coauthors_total']} vs page {theirs['coauthors']}"
    return f"{top['name']}: {theirs['papers']:,} papers, {theirs['coauthors']:,} co-authors"


@check("a venue's profile matches its explorer page", "dashboard")
def _venue(ctx):
    sid = call(ctx, "top_venues", limit=1)["rows"][0]["sid"]
    mine = call(ctx, "venue_profile", sid=sid)["meta"]["series"]
    theirs = dash(ctx, "venues/detail", sid=sid)["series"]
    assert mine["papers"] == theirs["papers"], f"papers {mine['papers']} vs {theirs['papers']}"
    assert mine["first_year"] == theirs["first_year"] and mine["last_year"] == theirs["last_year"]
    return f"{sid}: {theirs['papers']:,} papers"


@check("a term's share matches the titles page", "dashboard")
def _terms(ctx):
    last = ctx.last_full_year()
    mine = {r["year"]: r["neural"] for r in call(ctx, "title_terms", terms=["neural"],
                                                 frm=last - 3, to=last)["rows"]}
    theirs = dash(ctx, "titles/terms", terms="neural", **{"from": last - 3, "to": last})
    got = dict(zip(theirs["years"], theirs["series"][0]["values"]))
    # the dashboard's preset pattern for "neural" is \bneural\b; the tool tolerates a plural, so a
    # tiny difference is expected - a large one means a different population
    bad = [f"{y}: {mine.get(y)} vs {got[y]}" for y in got if not near(mine.get(y), got[y], 0.05)]
    assert not bad, "; ".join(bad)
    return f"neural {got[last]}% of {last} titles"


@check("counting papers matches the growth series", "dashboard")
def _counts(ctx):
    last = ctx.last_full_year()
    total = call(ctx, "count_papers", frm=last, to=last)
    by_kind = {r["kind"]: r["papers"] for r in total["rows"]}
    theirs = dash(ctx, "publishing/growth", **{"from": last, "to": last})[0]
    for kind, key in [("journal", "journal"), ("conference", "conference"), ("preprint", "preprint")]:
        assert by_kind.get(kind, 0) == theirs[key], f"{kind}: {by_kind.get(kind)} vs {theirs[key]}"
    return f"{last}: " + ", ".join(f"{k} {v:,}" for k, v in sorted(by_kind.items()))


# =========================================================== invariants
@check("a capped list never becomes a count")
def _caps(ctx):
    """Every tool that shows a shortened list must carry the true total next to it."""
    top = call(ctx, "top_authors", limit=1)["rows"][0]
    name = call(ctx, "most_shared_names", limit=1)["rows"][0]["base_name"]
    cases = [
        ("namesakes", {"name": name, "limit": 1}, "numbered_pages"),
        ("coauthors", {"key": top["key"], "limit": 1}, "coauthors_total"),
        ("author_papers", {"key": top["key"], "limit": 1}, "matching"),
    ]
    out = []
    for tool, args, key in cases:
        small = call(ctx, tool, **args)
        big = call(ctx, tool, **{**args, "limit": 30})
        total, wide = small["meta"][key], big["meta"][key]
        # the decisive one: a total computed by counting the rows on show moves when the cap moves.
        # This is exactly how "how many people are called Wei Wang" answered 39 instead of 522.
        assert total == wide, f"{tool}: total depends on the row limit ({total} at 1, {wide} at 30)"
        assert total >= len(small["rows"]), f"{tool}: total {total} < {len(small['rows'])} rows"
        assert len(small["rows"]) <= args["limit"], f"{tool} ignored its limit"
        assert str(total) in small["summary"].replace(",", ""), f"{tool} hides its total from the model"
        out.append(f"{tool} {len(small['rows'])} of {total:,}")
    return "; ".join(out)


@check("filters only ever narrow")
def _filters(ctx):
    last = ctx.last_full_year()
    everything = call(ctx, "count_papers")["meta"]["total"]
    recent = call(ctx, "count_papers", frm=last - 1)["meta"]["total"]
    journals = call(ctx, "count_papers", frm=last - 1, kind="journal")["meta"]["total"]
    big = call(ctx, "count_papers", frm=last - 1, kind="journal", min_authors=50)["meta"]["total"]
    assert everything > recent > journals >= big, f"{everything} > {recent} > {journals} >= {big} failed"
    return f"all {everything:,} > 2y {recent:,} > journals {journals:,} >= 50+ authors {big:,}"


@check("the kinds of a count add up to its total")
def _kinds_sum(ctx):
    out = call(ctx, "count_papers", frm=ctx.last_full_year() - 1)
    assert sum(r["papers"] for r in out["rows"]) == out["meta"]["total"], "breakdown does not sum"
    return f"{out['meta']['total']:,} across {len(out['rows'])} kinds"


@check("nothing found is never reported as zero")
def _empty(ctx):
    out = T.call(ctx, "resolve_author", {"name": "Zzzqx Nonexistent Person"})
    assert out["rows"] == [], "unexpectedly found someone"
    assert "no author page" in out["summary"].lower(), f"unclear summary: {out['summary']}"
    assert "0" not in out["summary"], "an empty result must not read as a count of zero"
    venue = T.call(ctx, "resolve_venue", {"name": "Zzzqx Nonexistent Venue"})
    assert venue["rows"] == [] and "no venue" in venue["summary"].lower()
    return "author and venue lookups both say 'nothing matches'"


@check("the same question twice gives the same numbers")
def _stable(ctx):
    a = call(ctx, "top_authors", limit=5)["rows"]
    b = call(ctx, "top_authors", limit=5)["rows"]
    assert [r["papers"] for r in a] == [r["papers"] for r in b], "leaderboard moved between calls"
    c = call(ctx, "count_papers", frm=2020, to=2020)["meta"]["total"]
    d = call(ctx, "count_papers", frm=2020, to=2020)["meta"]["total"]
    assert c == d, f"count changed: {c} then {d}"
    return "leaderboard and count are stable"


@check("bins are excluded from anything treated as a person")
def _bins(ctx):
    for row in call(ctx, "top_authors", limit=25)["rows"]:
        assert "disambiguation" not in str(row.get("page_kind", "")), f"a bin is ranked: {row['name']}"
    top = call(ctx, "top_authors", limit=1)["rows"][0]
    for row in call(ctx, "coauthors", key=top["key"], limit=25)["rows"]:
        assert row["page_kind"] != "disambiguation", f"a bin is a co-author: {row['name']}"
    return "leaderboard and co-author lists are people only"


@check("a metric's note describes the population it actually counted")
def _notes(ctx):
    pre = call(ctx, "papers_timeseries", metric="preprints", frm=2020)
    assert "preprints only" in pre["note"].lower(), f"misleading note: {pre['note']}"
    unid = call(ctx, "papers_timeseries", metric="unidentified_author_share", frm=2020)
    assert "preprints are included" in unid["note"].lower(), f"misleading note: {unid['note']}"
    papers = call(ctx, "papers_timeseries", metric="papers", frm=2020)
    assert "preprints" in papers["note"].lower() and "excluded" in papers["note"].lower()
    return "preprint, unidentified and paper metrics each state their own population"


@check("every tool hands the model its counting rules")
def _every_note(ctx):
    top = call(ctx, "top_authors", limit=1)["rows"][0]
    sid = call(ctx, "top_venues", limit=1)["rows"][0]["sid"]
    probes = {
        "dataset_facts": {}, "top_authors": {"limit": 3}, "top_venues": {"limit": 3},
        "most_shared_names": {"limit": 3}, "count_papers": {"frm": 2024},
        "papers_timeseries": {"metric": "papers", "frm": 2020}, "author_profile": {"key": top["key"]},
        "author_papers": {"key": top["key"], "limit": 3}, "coauthors": {"key": top["key"], "limit": 3},
        "venue_profile": {"sid": sid}, "namesakes": {"name": "Wei Wang", "limit": 3},
        "title_terms": {"terms": ["neural"], "frm": 2020}, "rising_words": {"limit": 5},
    }
    missing = [name for name, args in probes.items() if not call(ctx, name, **args).get("note")]
    assert not missing, f"no note on: {', '.join(missing)}"
    return f"{len(probes)} tools carry a note"


# =========================================================== behaviour
@check("every tool runs on representative arguments", "behaviour")
def _smoke(ctx):
    top = call(ctx, "top_authors", limit=1)["rows"][0]
    second = call(ctx, "top_authors", limit=2)["rows"][1]
    sid = call(ctx, "top_venues", limit=1)["rows"][0]["sid"]
    sid_b = call(ctx, "top_venues", limit=2)["rows"][1]["sid"]
    paper = call(ctx, "author_papers", key=top["key"], limit=1)["rows"][0]
    probes = [
        ("dataset_facts", {}), ("docs_lookup", {"question": "what is a disambiguation bin"}),
        ("resolve_author", {"name": top["name"]}), ("author_profile", {"key": top["key"]}),
        ("author_papers", {"key": top["key"], "frm": 2020, "kind": "journal"}),
        ("namesakes", {"name": "Wei Wang"}), ("coauthors", {"key": top["key"], "sid": sid}),
        ("pair_papers", {"key_a": top["key"], "key_b": second["key"]}),
        ("authors_in_both", {"sid_a": sid, "sid_b": sid_b}),
        ("resolve_venue", {"name": "CVPR"}), ("venue_profile", {"sid": sid}),
        ("top_venues", {"metric": "open_access", "kind": "journal", "limit": 5, "min_papers": 500}),
        ("top_authors", {"sid": sid, "limit": 5}), ("most_shared_names", {"limit": 5}),
        ("count_papers", {"frm": 2020, "to": 2024, "kind": "conference", "min_authors": 3}),
        ("papers_timeseries", {"metric": "median_authors", "frm": 2000}),
        ("title_terms", {"terms": ["neural", "quantum"], "frm": 2015}),
        ("rising_words", {"direction": "falling", "limit": 5}),
        ("paper_detail", {"key": paper["key"]}),
        ("run_sql", {"sql": "SELECT count(*) AS n FROM pubs WHERE year = 2024"}),
    ]
    slow = []
    for name, args in probes:
        t = time.time()
        call(ctx, name, **args)
        took = time.time() - t
        if took > SLOW_SECONDS:
            slow.append(f"{name} {took:.1f}s")
    assert not slow, f"slower than {SLOW_SECONDS}s: {', '.join(slow)}"
    return f"{len(probes)} tools ran, none slower than {SLOW_SECONDS}s"


@check("bad input is refused, not guessed at", "behaviour")
def _bad_input(ctx):
    must_refuse = [
        ("papers_timeseries", {"metric": "citations"}),
        ("title_terms", {"terms": ["'; DROP TABLE pubs --"]}),
        ("title_terms", {"terms": []}),
        ("run_sql", {"sql": "DROP TABLE pubs"}),
        ("run_sql", {"sql": "SELECT * FROM read_csv('/etc/passwd')"}),
        ("no_such_tool", {}),
    ]
    for name, args in must_refuse:
        out = T.call(ctx, name, args)
        assert out.get("refused"), f"{name}({args}) was not refused: {out.get('summary')}"
    missing = T.call(ctx, "author_profile", {"key": "homepages/does/not/exist"})
    assert missing["rows"] == [] and "resolve_author" in missing.get("note", "")
    return f"{len(must_refuse)} bad calls refused, a missing key explained"


@check("the SQL guard holds", "behaviour")
def _guard(ctx):
    attacks = [
        "DROP TABLE pubs", "SELECT 1; DELETE FROM pubs", "ATTACH '/tmp/x.db' AS y",
        "COPY pubs TO '/tmp/leak.csv'", "INSTALL httpfs", "SELECT * FROM read_parquet('s3://x/y')",
        "CREATE TABLE evil AS SELECT 1", "PRAGMA database_list", "UPDATE pubs SET year = 0",
        "SELECT 1 /* */; DROP TABLE pubs",
    ]
    for sql in attacks:
        ok, _ = sqlguard.check(sql)
        assert not ok, f"guard allowed: {sql}"
    ok, _ = sqlguard.check("SELECT title FROM pubs WHERE title ILIKE '%drop table%' LIMIT 5")
    assert ok, "guard rejected a legitimate query about the word 'drop'"
    capped = sqlguard.run(ctx.serving_path, "SELECT * FROM pubs", row_cap=5)
    assert capped["ok"] and len(capped["rows"]) == 5 and capped["truncated"]
    return f"{len(attacks)} statements refused, a literal allowed, rows capped"


@check("the documentation answers the questions it claims to", "behaviour")
def _docs(ctx):
    wanted = {
        "what is a disambiguation bin": "Disambiguation bins",
        "does this include preprints": "Preprints",
        "most cited paper": "citation",
        "how fresh is the data": "fresh",
    }
    bad = []
    for question, expect in wanted.items():
        hits = docs.search(question, top=3)
        text = " ".join(h["title"] + " " + h["text"] for h in hits).lower()
        if expect.lower() not in text:
            bad.append(f"{question!r} -> {[h['title'] for h in hits]}")
    assert not bad, "; ".join(bad)
    return f"{len(wanted)} documentation questions land on the right chunk"


@check("the leaderboard store is current", "behaviour")
def _store(ctx):
    meta = ctx.store_meta or {}
    assert meta.get("version") == store.VERSION, \
        f"store built by version {meta.get('version')}, code is {store.VERSION}"
    assert not meta.get("skipped"), f"a leaderboard step was skipped: {meta['skipped']}"
    assert meta.get("fingerprint") == ctx.meta.get("fingerprint"), "store is for a different dump"
    return f"version {meta['version']}, built {meta.get('built_at')}, nothing skipped"


# =========================================================== runner
def run(ctx, only=None):
    results = []
    for entry in CHECKS:
        if only and only not in entry["name"] and only != entry["group"]:
            continue
        t = time.time()
        try:
            detail = entry["fn"](ctx)
            status, message = "pass", detail
        except Skip as e:
            status, message = "skip", str(e)
        except AssertionError as e:
            status, message = "FAIL", str(e)
        except Exception as e:                       # a crash is a failure, with its type named
            log.exception("check %s crashed", entry["name"])
            status, message = "FAIL", f"{type(e).__name__}: {e}"
        results.append({"name": entry["name"], "group": entry["group"], "status": status,
                        "detail": message, "seconds": round(time.time() - t, 2)})
    summary = {
        "checks": len(results),
        "passed": sum(r["status"] == "pass" for r in results),
        "failed": sum(r["status"] == "FAIL" for r in results),
        "skipped": sum(r["status"] == "skip" for r in results),
        "seconds": round(sum(r["seconds"] for r in results), 1),
    }
    return {"summary": summary, "results": results}
