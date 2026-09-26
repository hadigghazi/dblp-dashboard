"""
The tool catalogue: what "retrieval" means for this data.

Each tool is a typed, bounded question the model may ask - never free text over chunks. A handler
returns the same shape every time:

    {"summary": one line the model reads first,
     "columns": [...], "rows": [...],        # capped; the UI renders these as a table
     "note": the counting rules that apply,  # so the answer can state them
     "link": a dashboard route,              # so an answer can point at the real page
     "meta": anything else worth quoting}

Three design rules, learned from the rest of this project:
  * a tool states its own caveats in `note`, because the model will otherwise invent them or omit
    them ("bins excluded", "preprints excluded", "this counts every record type");
  * a tool never returns more rows than a person would read - the model does not need 5,000 rows to
    say who is first, and every row is input tokens and latency;
  * anything a tool cannot do is an explicit refusal in its result, not an empty table: an empty
    table reads as "zero" to a language model.
"""
import logging
import re

from . import config, docs, sqlguard, store

log = logging.getLogger("dblp.chat.tools")

def journal_conf(p=""):
    """The journal/conference predicate, optionally qualified: every column must carry the alias, or
    a join with `slots` (which has its own type and is_preprint) is ambiguous."""
    return f"{p}type IN ('article', 'inproceedings') AND NOT {p}is_preprint"


JOURNAL_CONF = journal_conf()


def kind_expr(p=""):
    """The kind of a publication, optionally qualified with a table alias ('b.')."""
    return (f"CASE WHEN {p}is_preprint THEN 'preprint' WHEN {p}type = 'article' THEN 'journal' "
            f"WHEN {p}type = 'inproceedings' THEN 'conference' ELSE {p}type END")


KIND_SQL = kind_expr()
BINS_NOTE = "Disambiguation bins are excluded: a bin is a name, not a person."
JC_NOTE = "Journal and conference papers only; preprints, theses, books and chapters excluded."


# --------------------------------------------------------------------------- helpers
def rows_of(cur, sql, params=(), cap=None):
    cur.execute(sql, list(params)) if params else cur.execute(sql)
    cols = [d[0] for d in cur.description]
    out = [dict(zip(cols, r)) for r in cur.fetchall()]
    if cap:
        out = out[:cap]
    return cols, out


def one_of(cur, sql, params=()):
    _, rows = rows_of(cur, sql, params)
    return rows[0] if rows else None


def result(summary, columns=None, rows=None, note=None, link=None, **meta):
    out = {"summary": summary}
    if rows is not None:
        out["columns"] = columns or (list(rows[0].keys()) if rows else [])
        out["rows"] = rows[:config.ROWS_TO_MODEL]
        if len(rows) > config.ROWS_TO_MODEL:
            out["rows_omitted"] = len(rows) - config.ROWS_TO_MODEL
    if note:
        out["note"] = note
    if link:
        out["link"] = link
    if meta:
        out["meta"] = meta
    return out


def refusal(why, suggestion=None):
    out = {"summary": f"Not answerable from this data: {why}", "refused": True, "limits": docs.LIMITS}
    if suggestion:
        out["instead"] = suggestion
    return out


def _years(ctx, frm, to):
    last = ctx.last_full_year()
    return (1936 if frm is None else int(frm)), (last if to is None else int(to))


def _kind_clause(kind, params, p=""):
    if not kind:
        return "TRUE"
    if kind == "journal":
        return f"{p}type = 'article' AND NOT {p}is_preprint"
    if kind == "conference":
        return f"{p}type = 'inproceedings' AND NOT {p}is_preprint"
    if kind == "preprint":
        return f"{p}is_preprint"
    params.append(kind)
    return f"{p}type = ?"


def _best(rows, key):
    """The row with the largest value of `key`, ignoring missing ones - a plain max() over a column
    that can be NULL raises when two Nones meet."""
    usable = [r for r in rows if r.get(key) is not None]
    return max(usable, key=lambda r: r[key]) if usable else None


# --------------------------------------------------------------------------- the dataset itself
def dataset_facts(ctx):
    f = store.facts(ctx.cursor())
    rows = [{"measure": k.replace("_", " "), "value": v} for k, v in f.items()]
    return result(
        f"{f['records']:,} records: {f['publications']:,} publications "
        f"({f['journal_conference_papers']:,} journal/conference, {f['preprints']:,} preprints) and "
        f"{f['author_pages']:,} author pages, of which {f['disambiguation_bins']:,} are "
        f"disambiguation bins and {f['numbered_pages']:,} are numbered. "
        f"{f['venue_series']:,} venue series, papers from {f['first_year']} to {f['last_year']}.",
        rows=rows, note=f"Snapshot of the dump, latest record edit {ctx.meta.get('latest_mdate')}; "
                        f"the last complete year is {ctx.last_full_year()}.",
        link={"page": "overview"})


def model_cards(ctx):
    """Live accuracy of the three models and of search - never quote these from memory."""
    out, errors = {}, {}
    for name, url in [("disambiguation", f"{config.ML_URL}/ml/status"),
                      ("links", f"{config.ML_URL}/ml/links/status"),
                      ("venues", f"{config.ML_URL}/ml/venues/status"),
                      ("search", f"{config.SEARCH_URL}/search/status")]:
        try:
            r = ctx.http.get(url, timeout=config.UPSTREAM_TIMEOUT)
            r.raise_for_status()
            out[name] = r.json()
        except Exception as e:
            errors[name] = str(e)
    if not out:
        return refusal(f"the model services are unreachable ({errors})")
    return result("Live model cards, as measured on held-out data.", meta_cards=out, errors=errors or None,
                  note="Quote these numbers, and always name the baseline the model is compared with.")


# --------------------------------------------------------------------------- people
RESOLVE_SELECT = """
        res AS (
            SELECT p.person_id, p.key, p.name, p.base_name, p.page_kind,
                   coalesce(ps.n_pubs, 0) AS papers, c.first_year, c.last_year,
                   list_filter(p.notes, lambda n: n LIKE 'affiliation: %')[1] AS aff
            FROM hits JOIN s.persons p USING (person_id)
            LEFT JOIN s.person_stats ps USING (person_id)
            LEFT JOIN s.career c USING (person_id)
            ORDER BY {order}
            LIMIT ?)
        SELECT r.key, r.name, r.page_kind, r.papers, r.first_year, r.last_year,
               substr(r.aff, 14) AS affiliation,
               (SELECT count(*) FROM s.persons x WHERE x.base_name = r.base_name) AS pages_with_this_name
        FROM res r"""

FUZZY_MIN = 0.86          # jaro-winkler; below this the "closest name" is not a plausible typo
FUZZY_PREFIX = 4          # how much of a query word must survive the typo to be a candidate


def _fuzzy_authors(cur, q, limit):
    """The closest names when nothing contains the query: people mistype names constantly, and
    "no author page matches" is a correct but useless answer.

    Candidates come from a cheap contains-match on the first few letters of each query word (a typo
    usually survives its own prefix: "Schmidthuber" still contains "Schmid"), and only then are they
    scored with Jaro-Winkler, which is too expensive to run over four million names."""
    words = [w for w in re.split(r"[^\w']+", q) if len(w) >= FUZZY_PREFIX]
    if not words:
        return [], []
    likes = [f"%{w[:FUZZY_PREFIX]}%" for w in words[:3]]
    clause = " OR ".join(["name ILIKE ?"] * len(likes))
    return rows_of(cur, f"""
        WITH cand AS (SELECT person_id, name FROM s.person_names WHERE {clause}),
        hits AS (
            SELECT person_id, max(jaro_winkler_similarity(lower(name), lower(?))) AS sim
            FROM cand GROUP BY person_id
            HAVING max(jaro_winkler_similarity(lower(name), lower(?))) >= {FUZZY_MIN}),
        """ + RESOLVE_SELECT.format(order="(SELECT sim FROM hits h WHERE h.person_id = p.person_id) DESC, "
                                          "papers DESC, p.name"),
                   likes + [q, q, int(limit)])


def resolve_author(ctx, name, limit=8):
    cur = ctx.cursor()
    q = (name or "").strip()
    if len(q) < 2:
        return refusal("an author name needs at least two characters")
    cols, rows = rows_of(cur, """
        WITH hits AS (SELECT DISTINCT person_id FROM s.person_names WHERE name ILIKE ?),
        """ + RESOLVE_SELECT.format(order="(lower(p.base_name) = lower(?)) DESC, papers DESC, p.name"),
                         [f"%{q}%", q, int(limit)])
    matched = "contains"
    if not rows:
        matched = "close"
        cols, rows = _fuzzy_authors(cur, q, limit)
    if not rows:
        return result(f"No author page matches “{q}”, and no name in dblp is close to it.", rows=[],
                      note="dblp may spell the name differently, or the person may have no page. "
                           "A surname alone is the best second try.")
    bins = [r for r in rows if r["page_kind"] == "disambiguation"]
    many = max((r["pages_with_this_name"] or 1) for r in rows)
    note = [BINS_NOTE]
    if matched == "close":
        note.insert(0, f"No name contains “{q}”; these are the closest spellings dblp has. Say which "
                       f"one you used, and offer the others if the choice is not obvious.")
    if bins:
        note.append(f"“{bins[0]['name']}” is a disambiguation bin: its papers belong to several people.")
    if many > 1:
        note.append(f"{many} different pages share this base name - if the question is about one person, "
                    f"ask which, or say which page the answer uses.")
    return result(f"{len(rows)} candidate page(s) for “{q}”"
                  + (" by closest spelling, since nothing contains it" if matched == "close" else "")
                  + "; the first is the best match.",
                  cols, rows, note=" ".join(note), matched=matched)


def author_profile(ctx, key):
    cur = ctx.cursor()
    person = one_of(cur, """
        SELECT person_id, key, name, page_kind, names, notes FROM s.persons WHERE key = ?""", [key])
    if not person:
        return result(f"No author page with key {key!r}.", rows=[],
                      note="Call resolve_author first to turn a name into a key.")
    pid = person["person_id"]
    stats = one_of(cur, """
        SELECT count(*) AS papers, min(year) AS first_year, max(year) AS last_year,
               round(avg(n_authors), 2) AS mean_team,
               round(100 * avg(has_orcid::INT), 1) AS pct_with_orcid,
               count(*) FILTER (WHERE n_authors >= 3 AND position = 1) AS first_author,
               count(*) FILTER (WHERE n_authors >= 3 AND position = n_authors) AS last_author
        FROM s.slots WHERE person_id = ?""", [pid])
    degree = one_of(cur, """
        WITH mine AS (SELECT pid FROM s.slots WHERE person_id = ? AND n_authors BETWEEN 2 AND 50)
        SELECT count(DISTINCT sl.person_id) FILTER (WHERE NOT sl.on_bin) AS coauthors,
               count(DISTINCT sl.person_id) FILTER (WHERE sl.on_bin) AS on_bins
        FROM s.slots sl JOIN mine USING (pid)
        WHERE sl.person_id <> ?""", [pid, pid])
    _, venues = rows_of(cur, """
        WITH mine AS (SELECT pid FROM s.slots WHERE person_id = ?)
        SELECT b.sid, mode(b.venue) AS venue, count(*) AS papers
        FROM s.pubs b JOIN mine USING (pid)
        WHERE b.key_prefix IN ('conf', 'journals') AND NOT b.is_preprint
        GROUP BY b.sid ORDER BY papers DESC LIMIT 8""", [pid])
    _, top_co = rows_of(cur, """
        WITH mine AS (SELECT pid FROM s.slots WHERE person_id = ? AND n_authors BETWEEN 2 AND 50)
        SELECT p.key, p.name, count(*) AS papers_together
        FROM s.slots sl JOIN mine USING (pid) JOIN s.persons p USING (person_id)
        WHERE sl.person_id <> ? GROUP BY ALL ORDER BY papers_together DESC, p.name LIMIT 8""", [pid, pid])
    _, recent = rows_of(cur, f"""
        WITH mine AS (SELECT pid FROM s.slots WHERE person_id = ?)
        SELECT b.title, b.year, b.venue, {KIND_SQL} AS kind, b.key
        FROM s.pubs b JOIN mine USING (pid) ORDER BY b.year DESC NULLS LAST, b.key DESC LIMIT 8""", [pid])
    aff = [n[len("affiliation: "):] for n in (person.get("notes") or []) if n.startswith("affiliation: ")]
    summary = (f"{person['name']} ({person['page_kind']} page): {stats['papers']:,} records "
               f"{stats['first_year']}–{stats['last_year']}, {degree['coauthors']:,} identified "
               f"co-authors, mean team {stats['mean_team']}.")
    if degree["on_bins"]:
        summary += (f" A further {degree['on_bins']:,} co-author names sit on disambiguation bins, so "
                    f"they cannot be counted as individual people.")
    return result(summary, rows=recent,
                  note="Paper count covers every record type including preprints; the career table counts "
                       "only journal and conference papers. " + BINS_NOTE,
                  link={"page": "authors", "key": key},
                  stats=stats, coauthors=degree["coauthors"],
                  coauthor_names_on_bins=degree["on_bins"], affiliations=aff,
                  name_variants=person.get("names"), top_venues=venues, top_coauthors=top_co)


def author_papers(ctx, key, frm=None, to=None, kind=None, sid=None, limit=20):
    cur = ctx.cursor()
    pid = one_of(cur, "SELECT person_id FROM s.persons WHERE key = ?", [key])
    if not pid:
        return result(f"No author page with key {key!r}.", rows=[], note="Call resolve_author first.")
    where, params = ["TRUE"], []
    if frm is not None:
        where.append("b.year >= ?")
        params.append(int(frm))
    if to is not None:
        where.append("b.year <= ?")
        params.append(int(to))
    if kind:
        where.append(_kind_clause(kind, params, "b."))
    if sid:
        where.append("b.sid = ?")
        params.append(sid)
    cols, rows = rows_of(cur, f"""
        WITH mine AS (SELECT pid, position FROM s.slots WHERE person_id = ?)
        SELECT b.title, b.year, b.venue, {kind_expr('b.')} AS kind, b.n_authors, m.position, b.key
        FROM s.pubs b JOIN mine m USING (pid)
        WHERE {' AND '.join(where)}
        ORDER BY b.year DESC NULLS LAST, b.key DESC LIMIT ?""",
                        [pid["person_id"]] + params + [int(limit)])
    total = one_of(cur, f"""
        WITH mine AS (SELECT pid FROM s.slots WHERE person_id = ?)
        SELECT count(*) AS n FROM s.pubs b JOIN mine USING (pid) WHERE {' AND '.join(where)}""",
                   [pid["person_id"]] + params)
    return result(f"{total['n']:,} matching records; the {min(int(limit), total['n'])} most recent are listed.",
                  cols, rows, link={"page": "authors", "key": key},
                  note="Quote the total above, not the number of rows. position is the author's slot "
                       "on the paper.",
                  matching=total["n"])


def namesakes(ctx, name, limit=25):
    """The identity picture for a name: how many separate people dblp knows, and whether a bin exists.

    The counts come from their own aggregate, never from the length of the row list: a name like
    "Wei Wang" has hundreds of numbered pages, and counting the rows that fit in the answer would
    report the size of the page rather than the size of the population."""
    cur = ctx.cursor()
    base = (name or "").strip()
    totals = one_of(cur, """
        SELECT count(*) FILTER (WHERE page_kind = 'numbered') AS numbered,
               count(*) FILTER (WHERE page_kind = 'disambiguation') AS bins,
               count(*) FILTER (WHERE page_kind = 'regular') AS regular,
               count(*) AS pages
        FROM s.persons WHERE lower(base_name) = lower(?)""", [base])
    if not totals or totals["pages"] == 0:
        return result(f"dblp has no author page whose base name is exactly “{base}”.", rows=[],
                      note="Try resolve_author for a fuzzy match.")
    cols, rows = rows_of(cur, """
        SELECT p.key, p.name, p.page_kind, coalesce(ps.n_pubs, 0) AS papers,
               substr(list_filter(p.notes, lambda n: n LIKE 'affiliation: %')[1], 14) AS affiliation
        FROM s.persons p LEFT JOIN s.person_stats ps USING (person_id)
        WHERE lower(p.base_name) = lower(?)
        ORDER BY (p.page_kind = 'disambiguation') DESC, papers DESC LIMIT ?""", [base, int(limit)])
    bins = [r for r in rows if r["page_kind"] == "disambiguation"]
    summary = f"“{base}”: {totals['numbered']:,} separate people have a numbered page"
    if bins:
        summary += f", and a disambiguation bin holds {bins[0]['papers']:,} further records that " \
                   f"belong to an unknown number of other people of the same name"
    if totals["regular"]:
        summary += f"; {totals['regular']} plain page(s) carry this name too"
    summary += f". The {min(len(rows), totals['pages'])} largest of {totals['pages']:,} pages are listed."
    return result(summary, cols, rows,
                  note="Quote the count above, not the number of rows: the list is capped. A numbered "
                       "page is one verified person; a bin is a pile of papers by several people that "
                       "nobody has separated yet - the disambiguation model proposes a split.",
                  link={"page": "authors", "q": base},
                  numbered_pages=totals["numbered"], pages=totals["pages"], bins=totals["bins"])


def coauthors(ctx, key, sid=None, min_papers=1, limit=15):
    cur = ctx.cursor()
    pid = one_of(cur, "SELECT person_id, name FROM s.persons WHERE key = ?", [key])
    if not pid:
        return result(f"No author page with key {key!r}.", rows=[], note="Call resolve_author first.")
    # parameters are appended in the order the placeholders appear in the statement
    params = [pid["person_id"], pid["person_id"]]
    extra = ""
    if sid:
        extra = f"""AND sl.person_id IN (
                       SELECT sl2.person_id FROM s.slots sl2 JOIN s.pubs b2 ON b2.pid = sl2.pid
                       WHERE b2.sid = ? AND {journal_conf("b2.")})"""
        params.append(sid)
    params.append(int(min_papers))
    cols, rows = rows_of(cur, f"""
        WITH mine AS (SELECT pid FROM s.slots WHERE person_id = ? AND n_authors BETWEEN 2 AND 50)
        SELECT p.key, p.name, p.page_kind, count(*) AS papers_together, max(sl.year) AS last_year
        FROM s.slots sl JOIN mine USING (pid) JOIN s.persons p USING (person_id)
        WHERE sl.person_id <> ? AND NOT sl.on_bin {extra}
        GROUP BY ALL HAVING count(*) >= ?
        ORDER BY papers_together DESC, p.name LIMIT {int(limit)}""", params)
    # the true degree, so the answer never mistakes the length of a capped list for the total
    total = one_of(cur, """
        WITH mine AS (SELECT pid FROM s.slots WHERE person_id = ? AND n_authors BETWEEN 2 AND 50)
        SELECT count(DISTINCT sl.person_id) FILTER (WHERE NOT sl.on_bin) AS n,
               count(DISTINCT sl.person_id) FILTER (WHERE sl.on_bin) AS on_bins
        FROM s.slots sl JOIN mine USING (pid)
        WHERE sl.person_id <> ?""", [pid["person_id"], pid["person_id"]])
    return result(f"{pid['name']} has {total['n']:,} identified co-authors"
                  + (f"; of those, {len(rows)} also publish in {sid} (listed)" if sid
                     else f"; the {len(rows)} most frequent are listed") + ".",
                  cols, rows,
                  note="Quote the total above, not the number of rows: the list is capped. "
                       "Co-authorship counts papers with 2–50 authors. " + BINS_NOTE
                       + (f" {total['on_bins']:,} further co-author names resolve to a bin and are "
                          f"not counted as people." if total["on_bins"] else ""),
                  link={"page": "authors", "key": key}, coauthors_total=total["n"],
                  coauthor_names_on_bins=total["on_bins"])


def pair_papers(ctx, key_a, key_b, limit=20):
    cur = ctx.cursor()
    ids = rows_of(cur, "SELECT person_id, key, name FROM s.persons WHERE key IN (?, ?)", [key_a, key_b])[1]
    if len(ids) < 2:
        return result("One of the two author keys does not exist.", rows=[], note="Call resolve_author first.")
    a, b = ids[0], ids[1]
    cols, rows = rows_of(cur, f"""
        WITH a AS (SELECT pid FROM s.slots WHERE person_id = ?),
             b AS (SELECT pid FROM s.slots WHERE person_id = ?)
        SELECT p.title, p.year, p.venue, {KIND_SQL} AS kind, p.n_authors, p.key
        FROM s.pubs p JOIN a USING (pid) JOIN b USING (pid)
        ORDER BY p.year DESC NULLS LAST LIMIT ?""", [a["person_id"], b["person_id"], int(limit)])
    if not rows:
        return result(f"{a['name']} and {b['name']} have no paper together in dblp.", rows=[],
                      note="They may still be connected through other people - ask for the link prediction "
                           "or the co-author lists.")
    # the same rule as everywhere: the count is its own query, never the length of a capped list
    total = one_of(cur, """
        WITH a AS (SELECT pid FROM s.slots WHERE person_id = ?),
             b AS (SELECT pid FROM s.slots WHERE person_id = ?)
        SELECT count(*) AS n FROM s.pubs p JOIN a USING (pid) JOIN b USING (pid)""",
                   [a["person_id"], b["person_id"]])
    return result(f"{a['name']} and {b['name']} share {total['n']:,} papers; the "
                  f"{min(len(rows), total['n'])} most recent are listed.",
                  cols, rows, note="Quote the total above, not the number of rows.",
                  papers_together=total["n"])


# Counting each side first and joining is both simpler and faster than an INTERSECT plus correlated
# subqueries - and DuckDB will not ORDER BY an alias whose expression contains a subquery anyway.
IN_BOTH_CTE = """
    WITH ca AS (
        SELECT sl.person_id, count(*) AS papers_a
        FROM s.slots sl JOIN s.pubs b ON b.pid = sl.pid
        WHERE b.sid = ? AND b.type IN ('article', 'inproceedings') AND NOT b.is_preprint
          AND sl.person_id IS NOT NULL AND NOT sl.on_bin
        GROUP BY 1),
    cb AS (
        SELECT sl.person_id, count(*) AS papers_b
        FROM s.slots sl JOIN s.pubs b ON b.pid = sl.pid
        WHERE b.sid = ? AND b.type IN ('article', 'inproceedings') AND NOT b.is_preprint
          AND sl.person_id IS NOT NULL AND NOT sl.on_bin
        GROUP BY 1)
"""


def authors_in_both(ctx, sid_a, sid_b, limit=15):
    cur = ctx.cursor()
    cols, rows = rows_of(cur, IN_BOTH_CTE + """
        SELECT p.key, p.name, ca.papers_a, cb.papers_b, ca.papers_a + cb.papers_b AS papers_total
        FROM ca JOIN cb USING (person_id) JOIN s.persons p USING (person_id)
        ORDER BY papers_total DESC, p.name LIMIT ?""", [sid_a, sid_b, int(limit)])
    total = one_of(cur, IN_BOTH_CTE + """
        SELECT count(*) AS n FROM ca JOIN cb USING (person_id)""", [sid_a, sid_b])
    return result(f"{total['n']:,} people have published in both {sid_a} and {sid_b}; the most active are listed.",
                  cols, rows, note="Quote the total above, not the number of rows. " + BINS_NOTE + " " + JC_NOTE,
                  people_in_both=total["n"])


# --------------------------------------------------------------------------- venues
def resolve_venue(ctx, name, kind=None, limit=8):
    cur = ctx.cursor()
    q = (name or "").strip()
    where, params = ["(usual_name ILIKE ? OR sid ILIKE ?)"], [f"%{q}%", f"%{q}%"]
    if kind:
        where.append("kind = ?")
        params.append(kind)
    cols, rows = rows_of(cur, f"""
        SELECT sid, kind, usual_name AS name, papers, first_year, last_year,
               round(100 * oa_share, 1) AS pct_oa, round(100 * doi_share, 1) AS pct_doi, name_variants
        FROM s.series WHERE {' AND '.join(where)}
        ORDER BY (lower(usual_name) = lower(?)) DESC, papers DESC LIMIT ?""", params + [q, int(limit)])
    matched = "contains"
    if not rows:
        # only ~15,000 series, so every one can be scored: no prefix trick needed here
        matched = "close"
        where, params = ["TRUE"], [q, q]
        if kind:
            where.append("kind = ?")
            params.append(kind)
        cols, rows = rows_of(cur, f"""
            SELECT sid, kind, usual_name AS name, papers, first_year, last_year,
                   round(100 * oa_share, 1) AS pct_oa, round(100 * doi_share, 1) AS pct_doi,
                   name_variants,
                   round(greatest(jaro_winkler_similarity(lower(usual_name), lower(?)),
                                  jaro_winkler_similarity(lower(sid), lower(?))), 3) AS closeness
            FROM s.series WHERE {' AND '.join(where)}
            QUALIFY closeness >= {FUZZY_MIN}
            ORDER BY closeness DESC, papers DESC LIMIT ?""", params + [int(limit)])
    if not rows:
        return result(f"No venue series matches “{q}”, and none is close to it.", rows=[],
                      note="Venues are identified by series key like conf/cvpr or journals/tit; the name "
                           "string in a record may differ from the usual name. An acronym often works "
                           "better than a full title.")
    return result(f"{len(rows)} venue series match “{q}”"
                  + (" by closest spelling, since nothing contains it" if matched == "close" else "")
                  + "; the first is the best match.", cols, rows,
                  note=("No venue name contains that text; these are the closest. Say which one you "
                        "used. " if matched == "close" else "")
                       + "A series covers a conference or journal across all its years.",
                  matched=matched)


def venue_profile(ctx, sid):
    cur = ctx.cursor()
    head = one_of(cur, """
        SELECT sid, kind, usual_name AS name, papers, first_year, last_year, active_years, name_variants,
               round(100 * doi_share, 1) AS pct_doi, round(100 * oa_share, 1) AS pct_oa
        FROM s.series WHERE sid = ?""", [sid])
    if not head:
        return result(f"No venue series {sid!r}.", rows=[], note="Call resolve_venue first.")
    _, yearly = rows_of(cur, f"""
        SELECT year, count(*) AS papers, round(avg(n_authors), 2) AS mean_authors,
               round(100 * avg(has_oa::INT), 1) AS pct_oa,
               round(100 * avg((n_unidentified > 0)::INT), 1) AS pct_with_unidentified_author
        FROM s.pubs WHERE sid = ? AND {JOURNAL_CONF} AND year IS NOT NULL
        GROUP BY year ORDER BY year DESC LIMIT 10""", [sid])
    _, top = rows_of(cur, f"""
        WITH ps AS (SELECT pid FROM s.pubs WHERE sid = ? AND {JOURNAL_CONF})
        SELECT p.key, p.name, count(*) AS papers FROM s.slots sl JOIN ps USING (pid)
        JOIN s.persons p USING (person_id) WHERE NOT sl.on_bin
        GROUP BY ALL ORDER BY papers DESC, p.name LIMIT 10""", [sid])
    _, names = rows_of(cur, """
        SELECT venue AS name, count(*) AS papers, min(year) AS first_year, max(year) AS last_year
        FROM s.pubs WHERE sid = ? GROUP BY venue ORDER BY papers DESC LIMIT 6""", [sid])
    return result(f"{head['name']} ({head['kind']}, {sid}): {head['papers']:,} papers "
                  f"{head['first_year']}–{head['last_year']}, active in {head['active_years']} years, "
                  f"{head['pct_doi']}% with a DOI, {head['pct_oa']}% flagged open access.",
                  rows=yearly, note=JC_NOTE + " " + BINS_NOTE,
                  link={"page": "venues", "sid": sid},
                  series=head, top_authors=top, name_variants=names)


def top_venues(ctx, metric="papers", kind=None, frm=None, to=None, limit=10, min_papers=20):
    cur = ctx.cursor()
    if metric == "papers" and frm is None and to is None:
        where, params = ["TRUE"], []
        if kind:
            where.append("kind = ?")
            params.append(kind)
        cols, rows = rows_of(cur, f"""
            SELECT rank, sid, kind, name, papers, first_year, last_year, pct_oa, pct_doi
            FROM c.top_venue WHERE {' AND '.join(where)} ORDER BY rank LIMIT ?""", params + [int(limit)])
        return result(f"Largest venue series by papers{' (' + kind + 's)' if kind else ''}, all years.",
                      cols, rows, note=JC_NOTE + " Precomputed leaderboard for this dump.",
                      link={"page": "venues"})
    frm, to = _years(ctx, frm, to)
    order = {"papers": "papers DESC", "open_access": "pct_oa DESC, papers DESC",
             "authors_per_paper": "mean_authors DESC, papers DESC"}.get(metric, "papers DESC")
    where, params = [JOURNAL_CONF, "year BETWEEN ? AND ?", "key_prefix IN ('conf', 'journals')"], [frm, to]
    if kind:
        where.append("key_prefix = ?")
        params.append("journals" if kind == "journal" else "conf")
    cols, rows = rows_of(cur, f"""
        SELECT p.sid, mode(p.venue) AS name,
               CASE WHEN p.key_prefix = 'journals' THEN 'journal' ELSE 'conference' END AS kind,
               count(*) AS papers, round(avg(p.n_authors), 2) AS mean_authors,
               round(100 * avg(p.has_oa::INT), 1) AS pct_oa
        FROM s.pubs p WHERE {' AND '.join(where)}
        GROUP BY p.sid, kind HAVING count(*) >= ?
        ORDER BY {order} LIMIT ?""", params + [int(min_papers), int(limit)])
    return result(f"Top venue series by {metric.replace('_', ' ')}, {frm}–{to}.", cols, rows,
                  note=JC_NOTE + f" Series with fewer than {int(min_papers)} papers in the window "
                                 f"are excluded, so a small venue can be missing rather than absent.",
                  link={"page": "venues"})


# --------------------------------------------------------------------------- leaderboards
def top_authors(ctx, metric="papers", sid=None, frm=None, to=None, limit=10):
    cur = ctx.cursor()
    if sid is None and frm is None and to is None:
        table = "c.top_author_coauthors" if metric == "coauthors" else "c.top_author_papers"
        if not store.has_table(cur, table.split(".")[1]):
            return refusal("the co-author leaderboard is not available for this dump "
                           "(the api's optional per-author co-author table failed to build)",
                           "ask for papers instead, or for one author's co-author count")
        cols, rows = rows_of(cur, f"SELECT * FROM {table} ORDER BY rank LIMIT ?", [int(limit)])
        what = "distinct co-authors" if metric == "coauthors" else "records in dblp"
        return result(f"Authors ranked by {what}, all years.", cols, rows,
                      note=BINS_NOTE + " Papers here count every record type, preprints included. "
                           "Precomputed leaderboard for this dump.",
                      link={"page": "authors"})
    frm, to = _years(ctx, frm, to)
    where, params = ["sl.person_id IS NOT NULL", "NOT sl.on_bin", "sl.year BETWEEN ? AND ?"], [frm, to]
    if sid:
        where.append("b.sid = ?")
        params.append(sid)
    cols, rows = rows_of(cur, f"""
        SELECT p.key, p.name, count(*) AS papers, min(sl.year) AS first_year, max(sl.year) AS last_year
        FROM s.slots sl JOIN s.pubs b ON b.pid = sl.pid JOIN s.persons p USING (person_id)
        WHERE {' AND '.join(where)} AND b.type IN ('article', 'inproceedings') AND NOT b.is_preprint
        GROUP BY ALL ORDER BY papers DESC, p.name LIMIT ?""", params + [int(limit)])
    scope = f"in {sid} " if sid else ""
    return result(f"Most prolific authors {scope}{frm}–{to}.", cols, rows,
                  note=BINS_NOTE + " " + JC_NOTE, link={"page": "authors"})


def most_shared_names(ctx, limit=10):
    cur = ctx.cursor()
    cols, rows = rows_of(cur, "SELECT rank, base_name, people FROM c.top_name ORDER BY rank LIMIT ?",
                         [int(limit)])
    return result("Names shared by the most separate people (counting numbered pages only).", cols, rows,
                  note="Each numbered page is one verified person, so this is a lower bound: unseparated "
                       "namesakes still sit in the bin for that name.",
                  link={"page": "identity"})


# --------------------------------------------------------------------------- counting and trends
def count_papers(ctx, frm=None, to=None, kind=None, sid=None, author_key=None, min_authors=None,
                 max_authors=None, open_access=None, has_doi=None, with_unidentified_author=None,
                 title_contains=None):
    cur = ctx.cursor()
    where, params = ["TRUE"], []
    if frm is not None:
        where.append("b.year >= ?")
        params.append(int(frm))
    if to is not None:
        where.append("b.year <= ?")
        params.append(int(to))
    if kind:
        where.append(_kind_clause(kind, params, "b."))
    if sid:
        where.append("b.sid = ?")
        params.append(sid)
    if min_authors is not None:
        where.append("b.n_authors >= ?")
        params.append(int(min_authors))
    if max_authors is not None:
        where.append("b.n_authors <= ?")
        params.append(int(max_authors))
    if open_access is not None:
        where.append("b.has_oa" if open_access else "NOT b.has_oa")
    if has_doi is not None:
        where.append("b.has_doi" if has_doi else "NOT b.has_doi")
    if with_unidentified_author is not None:
        where.append("b.n_unidentified > 0" if with_unidentified_author else "b.n_unidentified = 0")
    if title_contains:
        where.append("b.title ILIKE ?")
        params.append(f"%{title_contains}%")
    join = ""
    if author_key:
        pid = one_of(cur, "SELECT person_id FROM s.persons WHERE key = ?", [author_key])
        if not pid:
            return result(f"No author page with key {author_key!r}.", rows=[], note="Call resolve_author first.")
        join = "JOIN (SELECT DISTINCT pid FROM s.slots WHERE person_id = ?) mine USING (pid)"
        params = [pid["person_id"]] + params
    cols, rows = rows_of(cur, f"""
        SELECT {kind_expr('b.')} AS kind, count(*) AS papers
        FROM s.pubs b {join} WHERE {' AND '.join(where)} GROUP BY ALL ORDER BY papers DESC""", params)
    total = sum(r["papers"] for r in rows)
    return result(f"{total:,} records match.", cols, rows,
                  note="Breakdown by kind; 'preprint' is CoRR/informal, so it overlaps no other kind. "
                       "No filter means every year and every record type except author pages and "
                       "proceedings volumes.",
                  total=total)


# (expression, population, what the population actually is). The third element exists because a note
# that says "preprints excluded" above a preprint count is a lie the model would faithfully repeat.
METRICS = {
    "papers": ("count(*)", JOURNAL_CONF, JC_NOTE),
    "mean_authors": ("round(avg(n_authors), 3)", JOURNAL_CONF + " AND n_authors > 0", JC_NOTE + " Papers with no author listed are excluded."),
    "median_authors": ("median(n_authors)", JOURNAL_CONF + " AND n_authors > 0", JC_NOTE + " Papers with no author listed are excluded."),
    "single_author_share": ("round(100 * avg((n_authors = 1)::INT), 2)", JOURNAL_CONF + " AND n_authors > 0", JC_NOTE),
    "share_10plus_authors": ("round(100 * avg((n_authors >= 10)::INT), 2)", JOURNAL_CONF + " AND n_authors > 0", JC_NOTE),
    "open_access_share": ("round(100 * avg(has_oa::INT), 2)", JOURNAL_CONF, JC_NOTE + " The flag is dblp's own open-access marker."),
    "doi_share": ("round(100 * avg(has_doi::INT), 2)", JOURNAL_CONF, JC_NOTE),
    "orcid_share": ("round(100 * avg((n_orcids > 0)::INT), 2)", JOURNAL_CONF, JC_NOTE + " A paper counts if ANY author carries an ORCID."),
    "unidentified_author_share": ("round(100 * avg((n_unidentified > 0)::INT), 2)",
                                  "type IN ('article', 'inproceedings')",
                                  "Journal, conference AND preprint records - preprints are included here, "
                                  "unlike the other metrics. A paper counts if any of its author slots "
                                  "lands on a disambiguation bin."),
    "preprints": ("count(*)", "is_preprint", "Preprints ONLY (CoRR/arXiv and records marked informal); "
                                             "this metric counts the population the others exclude."),
}


def papers_timeseries(ctx, metric="papers", frm=None, to=None, sid=None, kind=None):
    cur = ctx.cursor()
    if metric not in METRICS:
        return refusal(f"unknown metric {metric!r}", f"one of: {', '.join(METRICS)}")
    expr, base, population = METRICS[metric]
    frm, to = _years(ctx, frm, to)
    where, params = [base, "year BETWEEN ? AND ?"], [frm, to]
    if sid:
        where.append("sid = ?")
        params.append(sid)
    if kind in ("journal", "conference"):
        where.append("type = ?")
        params.append("article" if kind == "journal" else "inproceedings")
    cols, rows = rows_of(cur, f"""
        SELECT year, {expr} AS value, count(*) AS papers FROM s.pubs
        WHERE {' AND '.join(where)} GROUP BY year ORDER BY year""", params)
    if not rows:
        return result("No papers in that window.", rows=[])
    first, last = rows[0], rows[-1]
    # the total over the window, so "is X bigger than Y" never becomes arithmetic in the model
    papers_total = sum(r["papers"] or 0 for r in rows)
    counted = metric in ("papers", "preprints")
    return result(f"{metric.replace('_', ' ')} per year, {frm}–{to}: "
                  f"{first['value']} in {first['year']} → {last['value']} in {last['year']}"
                  + (f"; {papers_total:,} in total over the window" if counted else "")
                  + f" ({len(rows)} years).",
                  cols, rows[-40:], note=population + f" The last complete year is {ctx.last_full_year()}.",
                  link={"page": "publishing"}, series_length=len(rows),
                  first=first, last=last, peak=_best(rows, "value"),
                  window_total=papers_total if counted else None)


def title_terms(ctx, terms, frm=None, to=None):
    cur = ctx.cursor()
    clean = [t.strip().lower() for t in (terms or []) if t and t.strip()][:6]
    if not clean:
        return refusal("give at least one term")
    for t in clean:
        if not re.fullmatch(r"[a-z0-9][a-z0-9 \-+.#]{0,38}", t):
            return refusal(f"unsupported term {t!r}", "letters, digits, spaces, - + . # only")
    frm, to = _years(ctx, frm, to)
    frm = max(frm, 1970)
    cols = ", ".join(f"round(100 * avg(regexp_matches(lower(title), ?)::INT), 3) AS t{i}"
                     for i in range(len(clean)))
    patterns = [r"\b" + re.escape(t) + r"s?\b" for t in clean]
    _, data = rows_of(cur, f"""
        SELECT year, count(*) AS titles, {cols} FROM s.pubs
        WHERE {JOURNAL_CONF} AND title IS NOT NULL AND year BETWEEN ? AND ?
        GROUP BY year ORDER BY year""", patterns + [frm, to])
    rows = [{"year": r["year"], **{clean[i]: r[f"t{i}"] for i in range(len(clean))}} for r in data]
    peaks = {t: _best(rows, t) for t in clean}
    return result("Share of journal and conference titles containing each term, per year (%). "
                  + "; ".join(f"{t} peaks at {peaks[t][t]}% in {peaks[t]['year']}"
                              for t in clean if peaks[t]),
                  ["year"] + clean, rows[-40:],
                  note="Whole-word match, plural tolerated. A term's share reflects title wording only - "
                       "dblp has no abstracts or keywords.",
                  link={"page": "titles"})


def rising_words(ctx, old_from=None, new_from=None, direction="rising", limit=12, min_titles=1500):
    cur = ctx.cursor()
    last = ctx.last_full_year()
    new_from = int(new_from) if new_from else last - 4
    old_from = int(old_from) if old_from else new_from - 10
    if direction not in ("rising", "falling"):
        return refusal("direction must be 'rising' or 'falling'")
    side = "new_n" if direction == "rising" else "old_n"
    order = "DESC" if direction == "rising" else "ASC"
    cols, rows = rows_of(cur, f"""
        WITH p AS (
            SELECT word,
                   coalesce(sum(titles) FILTER (WHERE year BETWEEN ? AND ?), 0) AS old_n,
                   coalesce(sum(titles) FILTER (WHERE year BETWEEN ? AND ?), 0) AS new_n
            FROM s.word_year GROUP BY word),
        t AS (SELECT count(*) FILTER (WHERE year BETWEEN ? AND ?) AS old_t,
                     count(*) FILTER (WHERE year BETWEEN ? AND ?) AS new_t
              FROM s.pubs WHERE {JOURNAL_CONF} AND title IS NOT NULL)
        SELECT word, old_n AS titles_before, new_n AS titles_after,
               round(((new_n + 5.0) / new_t) / ((old_n + 5.0) / old_t), 2) AS change_x
        FROM p, t WHERE {side} >= ?
        ORDER BY change_x {order} LIMIT ?""",
                        [old_from, old_from + 4, new_from, new_from + 4,
                         old_from, old_from + 4, new_from, new_from + 4, int(min_titles), int(limit)])
    return result(f"Title words {direction} fastest between {old_from}–{old_from + 4} and "
                  f"{new_from}–{new_from + 4}. change_x is the ratio of the word's share of titles.",
                  cols, rows,
                  note="Shares are smoothed by +5 titles so a word going from 1 to 10 cannot top the list; "
                       f"a word needs {int(min_titles):,} titles in the later window.",
                  link={"page": "titles"})


# --------------------------------------------------------------------------- papers
def search_papers(ctx, q, kind=None, frm=None, to=None, top=8):
    """Hybrid search (BM25 + embeddings) through the search service, with an exact-word fallback."""
    try:
        params = {"q": q, "top": int(top)}
        if kind:
            params["kind"] = kind
        if frm is not None:
            params["from"] = int(frm)
        if to is not None:
            params["to"] = int(to)
        r = ctx.http.get(f"{config.SEARCH_URL}/search/papers", params=params,
                         timeout=config.UPSTREAM_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        rows = [{"title": x["title"], "year": x["year"], "venue": x["venue"], "kind": x["kind"],
                 "matched_by": "+".join(x.get("sources") or []), "key": x["key"]}
                for x in payload.get("results", [])]
        dense = payload.get("dense_available")
        return result(f"{len(rows)} papers for “{q}”, ranked by meaning and words together."
                      if dense else f"{len(rows)} papers for “{q}”, word match only "
                                    f"(the embedding index is unavailable).",
                      rows=rows, note="Relevance ranking, not an exhaustive list: a paper missing here is "
                                      "not evidence it does not exist. Titles are the only text dblp has.",
                      link={"page": "papers", "q": q}, dense_available=dense)
    except Exception as e:
        log.warning("search service unavailable: %s", e)
    cur = ctx.cursor()
    words = [w for w in (q or "").split() if w][:8]
    if not words:
        return refusal("give something to search for")
    where = ["title ILIKE ?"] * len(words)
    params = [f"%{w}%" for w in words]
    if frm is not None:
        where.append("year >= ?")
        params.append(int(frm))
    if to is not None:
        where.append("year <= ?")
        params.append(int(to))
    cols, rows = rows_of(cur, f"""
        SELECT title, year, venue, {KIND_SQL} AS kind, key FROM s.pubs
        WHERE {' AND '.join(where)} ORDER BY year DESC NULLS LAST LIMIT ?""", params + [int(top)])
    return result(f"{len(rows)} papers whose title contains every word of “{q}” "
                  f"(the semantic index is unavailable, so this is an exact-word match).",
                  cols, rows, note="An exact-word match misses paraphrases.", link={"page": "papers", "q": q})


def paper_detail(ctx, key=None, title=None):
    cur = ctx.cursor()
    if not key and not title:
        return refusal("give a record key or a title")
    if not key:
        hit = one_of(cur, """
            SELECT key FROM s.pubs WHERE title ILIKE ?
            ORDER BY length(title), year DESC NULLS LAST LIMIT 1""", [f"%{title.strip()}%"])
        if not hit:
            return result(f"No record whose title contains “{title}”.", rows=[],
                          note="Try search_papers, which ranks by meaning as well as words.")
        key = hit["key"]
    head = one_of(cur, f"""
        SELECT key, title, year, venue, sid, {KIND_SQL} AS kind, n_authors, n_orcids, n_unidentified,
               has_twin, has_oa, has_doi, pages, publisher, school
        FROM s.pubs WHERE key = ?""", [key])
    if not head:
        return result(f"No record with key {key!r}.", rows=[])
    # The author strings live in `src`, which is a view over the parquet - reachable here, but the
    # registry alone still answers the question if that mount is ever missing.
    note = ("page_kind says how firmly each author is identified: 'numbered' is a verified person, "
            "'disambiguation' means dblp does not know which person it is, 'unresolved' means the "
            "name has no page at all.")
    try:
        _, authors = rows_of(cur, """
            WITH a AS (SELECT unnest(authors) AS name, unnest(range(1, n_authors + 1)) AS position
                       FROM s.src WHERE key = ?)
            SELECT a.position, a.name, p.key AS author_key,
                   coalesce(p.page_kind, 'unresolved') AS page_kind
            FROM a LEFT JOIN s.person_names pn ON pn.name = a.name
            LEFT JOIN s.persons p ON p.person_id = pn.person_id ORDER BY a.position""", [key])
    except Exception as e:
        log.warning("author strings unavailable (%s); falling back to the registry", e)
        _, authors = rows_of(cur, """
            SELECT sl.position, coalesce(p.name, '(unresolved name)') AS name, p.key AS author_key,
                   coalesce(p.page_kind, 'unresolved') AS page_kind
            FROM s.slots sl LEFT JOIN s.persons p USING (person_id)
            WHERE sl.pid = (SELECT pid FROM s.pubs WHERE key = ?) ORDER BY sl.position""", [key])
        note += (" The names as printed on the paper were not available, so these come from the author "
                 "registry: a name dblp could not resolve shows as '(unresolved name)'.")
    twins = []
    if head["has_twin"]:
        _, twins = rows_of(cur, """
            SELECT key, title, year, venue FROM s.pubs
            WHERE title_norm = (SELECT title_norm FROM s.pubs WHERE key = ?) AND key <> ?
            ORDER BY year LIMIT 5""", [key, key])
    return result(f"“{head['title']}” ({head['year']}, {head['venue'] or 'no venue'}, {head['kind']}), "
                  f"{head['n_authors']} authors, "
                  f"{head['n_unidentified']} of them unidentified.",
                  rows=authors, note=note,
                  link={"page": "papers", "key": key}, paper=head, twins=twins)


# --------------------------------------------------------------------------- the models
def predict_venue(ctx, title, author_names=None, top=5):
    try:
        params = {"title": title, "top": int(top)}
        if author_names:
            params["authors"] = ", ".join(author_names[:20])
        r = ctx.http.get(f"{config.ML_URL}/ml/venues", params=params, timeout=config.UPSTREAM_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        return refusal(f"the venue model is unreachable ({e})")
    rows = [{"venue": s.get("name"), "sid": s.get("sid"), "score": s.get("score"),
             "came_true_rate": s.get("calibrated"), "why": s.get("why")}
            for s in (payload.get("suggestions") or [])]
    return result(f"Venue suggestions for “{title}”.", rows=rows,
                  note="A ranking, not a prediction of acceptance. Quote the model's measured accuracy "
                       "(call model_cards) next to this, and prefer the calibrated rate over the raw score.",
                  link={"page": "where-to-publish"}, raw=payload.get("metrics"))


def predict_coauthors(ctx, author_key, top=5):
    try:
        r = ctx.http.get(f"{config.ML_URL}/ml/links", params={"key": author_key, "top": int(top)},
                         timeout=config.UPSTREAM_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        return refusal(f"the link model is unreachable ({e})")
    rows = [{"name": s.get("name"), "key": s.get("key"), "score": s.get("score"),
             "shared_coauthors": s.get("common_coauthors"),
             "via": ", ".join(w.get("name", "") for w in (s.get("via") or [])[:3])}
            for s in (payload.get("suggestions") or [])]
    return result(f"Predicted next co-authors for {author_key}.", rows=rows,
                  note="Only distance-2 candidates are rankable, and about a quarter of real new "
                       "co-authors come from distance 2 - say so. Report the measured came-true rate for "
                       "the score band, not the score.",
                  link={"page": "collaborators"}, calibration=payload.get("calibration"))


def docs_lookup(ctx, question, top=3):
    hits = docs.search(question, top=top)
    if not hits:
        return result("Nothing in the documentation matches that.", rows=[])
    return result("Definitions and methodology from the project's own documentation.",
                  rows=[{"topic": h["title"], "text": h["text"]} for h in hits],
                  note="Quote these definitions rather than guessing how the data is counted.")


def run_sql(ctx, sql, reason=None):
    out = sqlguard.run(ctx.serving_path, sql)
    if not out["ok"]:
        return {"summary": f"SQL refused or failed: {out['error']}", "refused": True, "sql": sql,
                "hint": "Tables (no prefix): pubs, persons, person_names, slots, career, person_stats, "
                        "series, word_year, person_degree, src. Prefer a typed tool when one fits."}
    return result(f"{out['row_count']} row(s){' (truncated)' if out['truncated'] else ''}.",
                  out["columns"], out["rows"],
                  note="Ad-hoc SQL: state the definitions you used, and remember that a page can be a "
                       "disambiguation bin and a record can be a preprint unless you filtered them out.",
                  sql=out["sql"], reason=reason)


# --------------------------------------------------------------------------- the catalogue
def _fn(name, description, properties, required=(), heavy=False, remote=False):
    return {"name": name, "description": description, "heavy": heavy, "remote": remote,
            "schema": {"type": "function", "function": {
                "name": name, "description": description,
                "parameters": {"type": "object", "properties": properties, "required": list(required),
                               "additionalProperties": False}}}}


YEAR = {"type": "integer", "description": "year, inclusive"}
KIND = {"type": "string", "enum": ["journal", "conference", "preprint"]}

SPECS = [
    _fn("dataset_facts", "Size and shape of the dblp snapshot: records, publications, preprints, author "
        "pages, bins, venue series, year range, freshness. Use for 'how big is dblp' and as a sanity "
        "check before any share or percentage.", {}),
    _fn("docs_lookup", "Definitions, counting rules, methodology and model descriptions from the "
        "project's own documentation. Use for 'what is a disambiguation bin', 'does this include "
        "preprints', 'how fresh is the data', and whenever a question needs a definition to be answered "
        "honestly.", {"question": {"type": "string"}, "top": {"type": "integer"}}, ["question"]),
    _fn("model_cards", "Live measured accuracy of the three ML models and of paper search, with their "
        "baselines. Call this before quoting any accuracy number.", {}, remote=True),

    _fn("resolve_author", "Turn an author name into dblp author-page keys, tolerating misspellings "
        "(it falls back to the closest names and says so). ALWAYS call this before "
        "any author tool. It also reports each candidate's record count - enough to answer a plain "
        "'how many papers does X have' outright - plus how many different people share the name and "
        "whether a disambiguation bin exists, which decides whether the question is even well posed. "
        "For anything more than a count, follow it with author_profile.",
        {"name": {"type": "string"}, "limit": {"type": "integer"}}, ["name"]),
    _fn("author_profile", "Everything about one author page: record count, active years, distinct "
        "co-authors, mean team size, top venues, top co-authors, recent papers, affiliations.",
        {"key": {"type": "string", "description": "author page key from resolve_author"}}, ["key"]),
    _fn("author_papers", "One author's papers, newest first, with optional year, kind and venue filters. "
        "Returns the matching total as well as the listed rows.",
        {"key": {"type": "string"}, "from": YEAR, "to": YEAR, "kind": KIND,
         "sid": {"type": "string", "description": "venue series key"}, "limit": {"type": "integer"}}, ["key"]),
    _fn("namesakes", "How many separate people share a name, and whether an unassigned disambiguation "
        "bin exists for it. Use for 'how many people are called X' and for identity questions. The "
        "counts in the summary are the real totals; the row list is capped, so never count the rows.",
        {"name": {"type": "string", "description": "base name without a number, e.g. 'Wei Wang'"},
         "limit": {"type": "integer"}}, ["name"]),
    _fn("coauthors", "One author's co-authors, most frequent first. With `sid`, only co-authors who also "
        "publish in that venue series - use this for two-step questions.",
        {"key": {"type": "string"}, "sid": {"type": "string"}, "min_papers": {"type": "integer"},
         "limit": {"type": "integer"}}, ["key"]),
    _fn("pair_papers", "The papers two authors wrote together, if any.",
        {"key_a": {"type": "string"}, "key_b": {"type": "string"}, "limit": {"type": "integer"}},
        ["key_a", "key_b"]),
    _fn("authors_in_both", "People who have published in both of two venue series, most active first.",
        {"sid_a": {"type": "string"}, "sid_b": {"type": "string"}, "limit": {"type": "integer"}},
        ["sid_a", "sid_b"], heavy=True),

    _fn("resolve_venue", "Turn a conference or journal name into dblp series keys (conf/cvpr, "
        "journals/tit), tolerating misspellings. ALWAYS call this before any venue tool. It returns only enough to pick the "
        "right series - for anything ABOUT a venue ('tell me about X', 'what is X like', how it has "
        "changed, who publishes there) call venue_profile with the key it gives you.",
        {"name": {"type": "string"}, "kind": {"type": "string", "enum": ["journal", "conference"]},
         "limit": {"type": "integer"}}, ["name"]),
    _fn("venue_profile", "Everything about one venue series: size, span, DOI and open-access share, "
        "the last ten years, its most frequent authors, and the name variants it has used. This is "
        "the answer to any open question about a venue; resolve_venue only identifies it.",
        {"sid": {"type": "string"}}, ["sid"], heavy=True),
    _fn("top_venues", "Largest or most open venue series, optionally within a period or restricted to "
        "journals or conferences.",
        {"metric": {"type": "string", "enum": ["papers", "open_access", "authors_per_paper"]},
         "kind": {"type": "string", "enum": ["journal", "conference"]}, "from": YEAR, "to": YEAR,
         "limit": {"type": "integer"},
         "min_papers": {"type": "integer", "description": "minimum papers in the window (default 20)"}}),

    _fn("top_authors", "Authors ranked by records or by distinct co-authors. Global and all-time is a "
        "precomputed leaderboard and instant; with a venue (`sid`) or a year range it is computed live.",
        {"metric": {"type": "string", "enum": ["papers", "coauthors"]},
         "sid": {"type": "string", "description": "venue series key, e.g. conf/nips"},
         "from": YEAR, "to": YEAR, "limit": {"type": "integer"}}, heavy=True),
    _fn("most_shared_names", "The names shared by the most separate people.", {"limit": {"type": "integer"}}),

    _fn("count_papers", "How many records match a combination of filters, broken down by kind. Use this "
        "for every 'how many' question rather than counting rows yourself.",
        {"from": YEAR, "to": YEAR, "kind": KIND, "sid": {"type": "string"},
         "author_key": {"type": "string"}, "min_authors": {"type": "integer"},
         "max_authors": {"type": "integer"}, "open_access": {"type": "boolean"},
         "has_doi": {"type": "boolean"}, "with_unidentified_author": {"type": "boolean"},
         "title_contains": {"type": "string"}}, heavy=True),
    _fn("papers_timeseries", "A metric per year: papers, mean/median authors, single-author share, "
        "10+-author share, open-access, DOI, ORCID or unidentified-author share, or preprint counts. "
        "Optionally for one venue series or one kind.",
        {"metric": {"type": "string", "enum": list(METRICS)}, "from": YEAR, "to": YEAR,
         "sid": {"type": "string"}, "kind": {"type": "string", "enum": ["journal", "conference"]}},
        ["metric"], heavy=True),
    _fn("title_terms", "Share of titles per year containing each of up to six terms - the way to answer "
        "'is X rising', 'when did Y take off', 'compare X and Y'.",
        {"terms": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
         "from": YEAR, "to": YEAR}, ["terms"], heavy=True),
    _fn("rising_words", "Title words whose share grew or shrank most between two five-year windows.",
        {"old_from": YEAR, "new_from": YEAR,
         "direction": {"type": "string", "enum": ["rising", "falling"]}, "limit": {"type": "integer"},
         "min_titles": {"type": "integer"}}),

    _fn("search_papers", "Find papers by topic or by a remembered title, ranked by meaning and words "
        "together (hybrid search). Use for 'papers about ...' and 'what is the paper that ...'.",
        {"q": {"type": "string"}, "kind": KIND, "from": YEAR, "to": YEAR, "top": {"type": "integer"}},
        ["q"], remote=True),
    _fn("paper_detail", "One record in full: venue, year, authors with how firmly each is identified, "
        "flags, and any preprint/published twin. Give a key, or a title to look up.",
        {"key": {"type": "string"}, "title": {"type": "string"}}),

    _fn("predict_venue", "Where a paper with this title (and optionally these authors) would be "
        "published, from the venue-recommendation model.",
        {"title": {"type": "string"}, "author_names": {"type": "array", "items": {"type": "string"}},
         "top": {"type": "integer"}}, ["title"], remote=True),
    _fn("predict_coauthors", "Who an author is likely to publish with next, from the link-prediction "
        "model, with the shared co-authors that explain each suggestion.",
        {"author_key": {"type": "string"}, "top": {"type": "integer"}}, ["author_key"], remote=True),

    _fn("run_sql", "Last resort: one read-only SELECT over the serving tables (pubs, persons, "
        "person_names, slots, career, person_stats, series, word_year, person_degree, src) when no typed "
        "tool fits. Results are row-capped and the SQL is shown to the user, so keep it simple and "
        "aggregate rather than listing.",
        {"sql": {"type": "string"}, "reason": {"type": "string",
         "description": "why no typed tool fits"}}, ["sql"], heavy=True),
]

HANDLERS = {
    "dataset_facts": dataset_facts, "docs_lookup": docs_lookup, "model_cards": model_cards,
    "resolve_author": resolve_author, "author_profile": author_profile, "author_papers": author_papers,
    "namesakes": namesakes, "coauthors": coauthors, "pair_papers": pair_papers,
    "authors_in_both": authors_in_both, "resolve_venue": resolve_venue, "venue_profile": venue_profile,
    "top_venues": top_venues, "top_authors": top_authors, "most_shared_names": most_shared_names,
    "count_papers": count_papers, "papers_timeseries": papers_timeseries, "title_terms": title_terms,
    "rising_words": rising_words, "search_papers": search_papers, "paper_detail": paper_detail,
    "predict_venue": predict_venue, "predict_coauthors": predict_coauthors, "run_sql": run_sql,
}

# `from`/`to` are reserved words in Python, so the schema's names are mapped on the way in.
RENAME = {"from": "frm", "to": "to"}


def schemas():
    return [s["schema"] for s in SPECS]


def spec(name):
    return next((s for s in SPECS if s["name"] == name), None)


def call(ctx, name, args):
    handler = HANDLERS.get(name)
    if handler is None:
        return {"summary": f"No tool named {name!r}.", "refused": True,
                "available": sorted(HANDLERS)}
    kwargs = {RENAME.get(k, k): v for k, v in (args or {}).items() if v is not None}
    try:
        return handler(ctx, **kwargs)
    except TypeError as e:
        return {"summary": f"{name}: wrong arguments ({e})", "refused": True,
                "schema": spec(name)["schema"]["function"]["parameters"]}
    except Exception as e:
        log.exception("tool %s failed", name)
        return {"summary": f"{name} failed: {type(e).__name__}: {e}", "refused": True}
