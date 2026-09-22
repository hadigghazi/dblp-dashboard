"""
Live queries. Each function takes a DuckDB cursor on the serving database plus request parameters and
returns JSON-ready data. Predicates follow the analysis scripts named in each docstring, so the
defaults reproduce the report's numbers; the parameters go beyond what the report fixed.
"""
import math
import re
from decimal import Decimal

import numpy as np
from scipy import stats as sstats

JOURNAL_CONF = "type IN ('article', 'inproceedings') AND NOT is_preprint"
KIND_SQL = ("CASE WHEN is_preprint THEN 'preprint' WHEN type = 'article' THEN 'journal' "
            "WHEN type = 'inproceedings' THEN 'conference' ELSE type END")

# eda_03_venues.py
PUBLISHERS = {
    "10.1109": "IEEE", "10.1007": "Springer", "10.1016": "Elsevier", "10.1145": "ACM", "10.3390": "MDPI",
    "10.1002": "Wiley", "10.1080": "Taylor & Francis", "10.4230": "Dagstuhl LIPIcs", "10.1142": "World Scientific",
    "10.1093": "Oxford UP", "10.1017": "Cambridge UP", "10.1137": "SIAM", "10.1038": "Nature", "10.1371": "PLOS",
    "10.1155": "Hindawi", "10.3233": "IOS Press", "10.1049": "IET", "10.18653": "ACL Anthology", "10.1177": "SAGE",
    "10.1098": "Royal Society", "10.1088": "IOP", "10.1117": "SPIE", "10.21437": "ISCA", "10.5220": "SciTePress",
    "10.1515": "De Gruyter", "10.1609": "AAAI", "10.24963": "IJCAI", "10.14778": "VLDB", "10.1162": "MIT Press",
    "10.1613": "JAIR",
}

# eda_01_titles.py (the report's terms keep their exact patterns; anything else is matched as a whole word)
PRESET_TERMS = {
    "neural": r"\bneural\b", "deep": r"\bdeep\b", "transformer": r"\btransformers?\b",
    "llm": r"\bllms?\b|large language model", "graph": r"\bgraphs?\b", "federated": r"\bfederated\b",
    "blockchain": r"blockchain", "quantum": r"\bquantum\b", "iot": r"\biot\b|internet of things",
    "cloud": r"\bcloud\b", "diffusion": r"\bdiffusion\b", "explainable": r"explainab", "privacy": r"\bprivacy\b",
}

# charts_eda_a.py: words the report's chart skipped as noise / duplicates of another entry
RISING_SKIP = {"llms", "explainability", "ris", "irs", "mec", "noma", "siamese", "twins", "lora"}


class BadRequest(ValueError):
    pass


def _clean(v):
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def rows(cur, sql, params=()):
    if params:
        cur.execute(sql, list(params))
    else:
        cur.execute(sql)
    cols = [d[0] for d in cur.description]
    return [{c: _clean(v) for c, v in zip(cols, r)} for r in cur.fetchall()]


def one(cur, sql, params=()):
    r = rows(cur, sql, params)
    return r[0] if r else None


def last_full_year(meta):
    try:
        return int(meta.get("last_full_year", 2025))
    except (TypeError, ValueError):
        return 2025


def _years(frm, to, lo, hi):
    frm = lo if frm is None else frm
    to = hi if to is None else to
    if frm > to:
        raise BadRequest("'from' must not be after 'to'")
    return frm, to


# =========================================================================== overview
def overview(cur, meta):
    c = one(cur, f"""
        SELECT count(*) AS publications,
               count(*) FILTER (WHERE {JOURNAL_CONF}) AS journal_conference,
               count(*) FILTER (WHERE is_preprint) AS preprints,
               round(100.0 * avg((n_unidentified > 0)::INT) FILTER (WHERE type IN ('article', 'inproceedings')), 1)
                   AS pct_unidentified,
               round(100.0 * avg((n_unidentified > 0)::INT)
                     FILTER (WHERE type IN ('article', 'inproceedings') AND year BETWEEN 2020 AND 2029), 1)
                   AS pct_unidentified_2020s
        FROM pubs""")
    p = one(cur, """
        SELECT count(*) AS author_pages,
               count(*) FILTER (WHERE page_kind = 'disambiguation') AS bins,
               count(*) FILTER (WHERE page_kind = 'numbered') AS numbered,
               round(100.0 * avg(has_affiliation::INT) FILTER (WHERE page_kind = 'numbered'), 1) AS aff_numbered,
               round(100.0 * avg(has_affiliation::INT) FILTER (WHERE page_kind = 'regular'), 1) AS aff_regular
        FROM persons""")
    hom = one(cur, """
        SELECT base_name, count(*) AS people FROM persons WHERE page_kind = 'numbered'
        GROUP BY base_name ORDER BY people DESC, base_name LIMIT 1""") or {"base_name": "-", "people": 0}
    twins = one(cur, """
        WITH pre AS (SELECT DISTINCT title_norm FROM pubs WHERE is_preprint AND length(title_norm) >= 30),
             pub AS (SELECT DISTINCT title_norm FROM pubs WHERE NOT is_preprint AND length(title_norm) >= 30)
        SELECT round(100.0 * count(pub.title_norm) / nullif(count(*), 0), 1) AS pct
        FROM pre LEFT JOIN pub ON pre.title_norm = pub.title_norm""")
    words = one(cur, f"""
        SELECT round(avg(len(string_split(title, ' '))), 1) AS mean_words FROM pubs
        WHERE {JOURNAL_CONF} AND year BETWEEN ? AND ?""", [last_full_year(meta) - 5, last_full_year(meta)])
    rate = growth_rates(cur, meta, 1990, 2023, 2005)
    all_rate = next((r for r in rate["rows"] if r["kind"] == "all" and r["period"] == "1990-2023"), None)
    records = int(meta.get("records", 0))
    return {
        "counts": {**c, **p, "records": records},
        "kpis": [
            {"n": f"{records / 1e6:.2f}M", "l": f"records: {c['publications'] / 1e6:.2f}M publications, "
                                                f"{p['author_pages'] / 1e6:.2f}M author pages"},
            {"n": f"{words['mean_words']} words", "l": "of text per recent paper: titles are the only text; "
                                                       "no abstracts, citations or affiliations on papers"},
            {"n": f"{hom['people']:,}", "l": f"different people named “{hom['base_name']}”"},
            {"n": f"{c['pct_unidentified']}%", "l": f"of papers have an author dblp hasn’t identified; "
                                                    f"{c['pct_unidentified_2020s']}% in the 2020s"},
            {"n": f"{p['aff_numbered']:.0f}% vs {p['aff_regular']}%",
             "l": "of numbered vs. regular author pages carry an affiliation"},
            {"n": f"{all_rate['rate']}%/yr" if all_rate else "-",
             "l": f"growth in papers 1990–2023, doubling every {all_rate['doubling']} years" if all_rate else ""},
            {"n": f"{twins['pct']}%", "l": "of preprint titles also exist as a separate, unlinked published paper"},
            {"n": f"{p['bins']:,}", "l": "unassigned “disambiguation” pages holding many people’s papers"},
        ],
    }


# =========================================================================== publishing
def growth(cur, meta, frm=None, to=None):
    """profile_dblp.py 04_growth / make_charts.py chart_growth."""
    frm, to = _years(frm, to, 1970, last_full_year(meta))
    return rows(cur, """
        SELECT year,
               count(*) FILTER (WHERE type = 'inproceedings' AND NOT is_preprint) AS conference,
               count(*) FILTER (WHERE type = 'article' AND NOT is_preprint) AS journal,
               count(*) FILTER (WHERE is_preprint AND type IN ('article', 'inproceedings')) AS preprint,
               count(*) FILTER (WHERE type = 'phdthesis') AS phd_thesis,
               count(*) FILTER (WHERE type IN ('book', 'incollection')) AS book_or_chapter
        FROM pubs WHERE year BETWEEN ? AND ? GROUP BY year ORDER BY year""", [frm, to])


def teams(cur, meta, frm=None, to=None):
    """make_charts.py chart_team_size."""
    frm, to = _years(frm, to, 1970, last_full_year(meta))
    return rows(cur, f"""
        SELECT year, avg(n_authors) AS mean_authors, avg((n_authors = 1)::INT) AS single_author_share,
               avg((n_authors >= 10)::INT) AS share_10plus, median(n_authors) AS median_authors
        FROM pubs WHERE {JOURNAL_CONF} AND n_authors > 0 AND year BETWEEN ? AND ?
        GROUP BY year ORDER BY year""", [frm, to])


def metadata_trends(cur, meta, frm=None, to=None):
    """make_charts.py chart_metadata_trends (journal, conference and preprint records)."""
    frm, to = _years(frm, to, 2000, last_full_year(meta))
    return rows(cur, """
        SELECT year, avg((n_orcids > 0)::INT) AS with_orcid,
               avg((n_unidentified > 0)::INT) AS with_unidentified_author,
               avg(has_twin::INT) AS with_title_twin, avg(has_oa::INT) AS open_access
        FROM pubs WHERE type IN ('article', 'inproceedings') AND year BETWEEN ? AND ?
        GROUP BY year ORDER BY year""", [frm, to])


def growth_rates(cur, meta, frm=1990, to=2023, split=2005):
    """eda_07_statistics.py: OLS on log yearly counts, t-based 95% interval."""
    if not (frm < split < to):
        raise BadRequest("need from < split < to")
    yearly = cur.execute(f"""
        SELECT year, {KIND_SQL} AS kind, count(*) AS n FROM pubs
        WHERE type IN ('article', 'inproceedings') AND year BETWEEN ? AND ? GROUP BY ALL""", [frm, to]).fetchall()
    out = []
    for kind in ["all", "journal", "conference", "preprint"]:
        for lo, hi in [(frm, to), (frm, split), (split + 1, to)]:
            counts = {}
            for year, k, n in yearly:
                if lo <= year <= hi and (kind == "all" or k == kind):
                    counts[year] = counts.get(year, 0) + n
            years = np.array(sorted(y for y, n in counts.items() if n > 0))
            if len(years) < 4:
                continue
            res = sstats.linregress(years, np.log([counts[y] for y in years]))
            half = sstats.t.ppf(0.975, len(years) - 2) * res.stderr
            if not (math.isfinite(res.slope) and math.isfinite(half) and math.isfinite(res.rvalue)):
                continue  # flat or degenerate series: no growth rate to report
            out.append({
                "kind": kind, "period": f"{lo}-{hi}",
                "rate": round(100 * (math.exp(res.slope) - 1), 1),
                "lo": round(100 * (math.exp(res.slope - half) - 1), 1),
                "hi": round(100 * (math.exp(res.slope + half) - 1), 1),
                "doubling": round(math.log(2) / res.slope, 1) if res.slope > 0 else None,
                "r_squared": round(res.rvalue ** 2, 3),
            })
    return {"from": frm, "to": to, "split": split, "rows": out}


# =========================================================================== identity
def homonyms(cur, top=15):
    """make_charts.py chart_homonyms."""
    return rows(cur, """
        SELECT base_name, count(*) AS distinct_people FROM persons WHERE page_kind = 'numbered'
        GROUP BY base_name ORDER BY distinct_people DESC, base_name LIMIT ?""", [top])


def affiliation(cur):
    """make_charts.py chart_curation."""
    data = rows(cur, """
        SELECT page_kind, count(*) AS pages, avg(has_affiliation::INT) AS affiliation,
               avg(has_orcid_link::INT) AS orcid, avg(has_wikidata_link::INT) AS wikidata,
               avg((n_names >= 2)::INT) AS name_variants
        FROM persons GROUP BY page_kind ORDER BY page_kind""")
    return {r["page_kind"]: r for r in data}


def newcomers(cur, meta, frm=None, to=None):
    """eda_04_careers.py 'New authors per year'."""
    frm, to = _years(frm, to, 1970, last_full_year(meta))
    return rows(cur, f"""
        WITH active AS (SELECT year, count(DISTINCT person_id) AS active_authors FROM slots
                        WHERE person_id IS NOT NULL AND NOT on_bin AND {JOURNAL_CONF} GROUP BY year),
             newc AS (SELECT first_year AS year, count(*) AS new_authors FROM career GROUP BY first_year)
        SELECT a.year, a.active_authors, n.new_authors,
               round(100.0 * n.new_authors / a.active_authors, 1) AS pct_newcomers
        FROM active a JOIN newc n USING (year) WHERE a.year BETWEEN ? AND ? ORDER BY a.year""", [frm, to])


def cohorts(cur, meta, frm=1975, to=2005, step=5):
    """eda_04_careers.py 'Cohorts' (horizons past the data end are left empty)."""
    end = last_full_year(meta)
    return rows(cur, f"""
        SELECT first_year AS cohort, count(*) AS authors,
               round(100 * avg((papers = 1)::INT), 1) AS pct_single_paper,
               round(100 * avg(CASE WHEN first_year + 5 <= {end} THEN (last_year >= first_year + 5)::INT END), 1) AS pct_5y,
               round(100 * avg(CASE WHEN first_year + 10 <= {end} THEN (last_year >= first_year + 10)::INT END), 1) AS pct_10y,
               round(100 * avg(CASE WHEN first_year + 20 <= {end} THEN (last_year >= first_year + 20)::INT END), 1) AS pct_20y,
               round(100 * avg(any_orcid::INT), 1) AS pct_with_orcid
        FROM career WHERE (first_year - ?) % ? = 0 AND first_year BETWEEN ? AND ?
        GROUP BY first_year ORDER BY first_year""", [frm, step, frm, to])


def positions(cur, min_authors=3):
    """eda_04_careers.py 'Where authors appear'."""
    return rows(cur, f"""
        SELECT CASE WHEN c.papers <= 2 THEN '1-2' WHEN c.papers <= 10 THEN '3-10' WHEN c.papers <= 50 THEN '11-50'
                    WHEN c.papers <= 200 THEN '51-200' ELSE '201+' END AS author_total_papers,
               count(DISTINCT s.person_id) AS authors, count(*) AS author_slots,
               round(100 * avg((s.position = 1)::INT), 1) AS pct_first,
               round(100 * avg((s.position > 1 AND s.position < s.n_authors)::INT), 1) AS pct_middle,
               round(100 * avg((s.position = s.n_authors)::INT), 1) AS pct_last
        FROM slots s JOIN career c USING (person_id)
        WHERE s.n_authors >= ? AND NOT s.on_bin AND s.type IN ('article', 'inproceedings') AND NOT s.is_preprint
        GROUP BY ALL ORDER BY min(c.papers)""", [min_authors])


def unidentified_by_position(cur, since=2015, min_authors=3):
    """eda_04_careers.py 'Identification by author position'."""
    return rows(cur, f"""
        SELECT CASE WHEN position = 1 THEN 'first' WHEN position = n_authors THEN 'last' ELSE 'middle' END
                   AS author_position,
               count(*) AS slots,
               round(100 * avg(on_bin::INT), 2) AS pct_unidentified,
               round(100 * avg(has_orcid::INT), 1) AS pct_with_orcid
        FROM slots
        WHERE person_id IS NOT NULL AND n_authors >= ? AND year >= ? AND {JOURNAL_CONF}
        GROUP BY ALL ORDER BY min(CASE WHEN position = 1 THEN 0 WHEN position = n_authors THEN 2 ELSE 1 END)""",
                [min_authors, since])


def alphabetical(cur, since=2000, min_papers=2000, n_most=10, n_least=5):
    """eda_04_careers.py 'Series ... alphabetically' (2-10 authors, distinct surnames)."""
    base = f"""
        SELECT mode(venue) AS usual_name, sid, count(*) AS papers,
               round(100 * avg(alphabetical::INT), 1) AS pct_alphabetical,
               round(100 * avg(1.0 / factorial(n_authors)), 1) AS pct_by_chance
        FROM pubs
        WHERE {JOURNAL_CONF} AND n_authors BETWEEN 2 AND 10 AND distinct_surnames AND year >= ?
        GROUP BY sid HAVING count(*) >= ?"""
    most = rows(cur, base + " ORDER BY pct_alphabetical DESC LIMIT ?", [since, min_papers, n_most])
    least = rows(cur, base + " ORDER BY pct_alphabetical ASC LIMIT ?", [since, min_papers, n_least])
    by_decade = rows(cur, f"""
        SELECT year - year % 10 AS decade, count(*) AS papers,
               round(100 * avg(alphabetical::INT), 1) AS pct_alphabetical,
               round(100 * avg(1.0 / factorial(n_authors)), 1) AS pct_by_chance
        FROM pubs WHERE {JOURNAL_CONF} AND n_authors BETWEEN 2 AND 10 AND distinct_surnames AND year >= 1970
        GROUP BY ALL ORDER BY decade""")
    return {"most": most, "least": least, "by_decade": by_decade}


# =========================================================================== titles
def _term_pattern(term):
    t = term.strip().lower()
    if t in PRESET_TERMS:
        return PRESET_TERMS[t]
    if not re.fullmatch(r"[a-z0-9][a-z0-9 \-+.#]{0,38}[a-z0-9+#]?", t):
        raise BadRequest(f"unsupported term: {term!r} (letters, digits, spaces, - + . #)")
    return r"\b" + re.escape(t) + r"\b"


def terms(cur, meta, term_list, frm=None, to=None):
    """eda_01_titles.py 'Topic terms' (share of journal/conference titles, in %)."""
    terms_clean = []
    for t in term_list:
        t = t.strip().lower()
        if t and t not in terms_clean:
            terms_clean.append(t)
    if not terms_clean:
        raise BadRequest("give at least one term")
    if len(terms_clean) > 8:
        raise BadRequest("at most 8 terms")
    frm, to = _years(frm, to, 2000, last_full_year(meta))
    patterns = [_term_pattern(t) for t in terms_clean]
    cols = ", ".join(f"round(100 * avg(regexp_matches(lower(title), ?)::INT), 3) AS t{i}"
                     for i in range(len(patterns)))
    data = rows(cur, f"""
        SELECT year, count(*) AS titles, {cols}
        FROM pubs WHERE {JOURNAL_CONF} AND title IS NOT NULL AND year BETWEEN ? AND ?
        GROUP BY year ORDER BY year""", patterns + [frm, to])
    series = [{"term": t, "pattern": p, "values": [r[f"t{i}"] for r in data]}
              for i, (t, p) in enumerate(zip(terms_clean, patterns))]
    return {"years": [r["year"] for r in data], "titles": [r["titles"] for r in data], "series": series}


def title_style(cur, meta):
    """eda_01_titles.py 'Title style by decade'."""
    return rows(cur, f"""
        SELECT year - year % 10 AS decade, count(*) AS titles,
               round(100 * avg(contains(title, '?')::INT), 1) AS question,
               round(100 * avg(contains(title, ':')::INT), 1) AS colon,
               round(100 * avg(regexp_matches(title, '^[A-Za-z0-9-]*[A-Z][A-Za-z0-9-]*[A-Z][A-Za-z0-9-]*:')::INT), 1)
                   AS name_colon,
               round(100 * avg(regexp_matches(lower(title), '^towards? ')::INT), 1) AS towards,
               round(avg(len(string_split(title, ' '))), 1) AS mean_words
        FROM pubs WHERE {JOURNAL_CONF} AND title IS NOT NULL AND year BETWEEN 1970 AND ?
        GROUP BY decade ORDER BY decade""", [last_full_year(meta)])


def words(cur, old=(2011, 2015), new=(2021, 2025), direction="rising", min_titles=1500, limit=15, skip_noise=True):
    """eda_01_titles.py GROWTH: +5 smoothing, share of titles in each window."""
    if direction not in ("rising", "falling"):
        raise BadRequest("direction must be rising or falling")
    side = "new_n" if direction == "rising" else "old_n"
    order = "DESC" if direction == "rising" else "ASC"
    skip = list(RISING_SKIP) if skip_noise else [""]
    return rows(cur, f"""
        WITH p AS (
            SELECT word,
                   coalesce(sum(titles) FILTER (WHERE year BETWEEN ? AND ?), 0) AS old_n,
                   coalesce(sum(titles) FILTER (WHERE year BETWEEN ? AND ?), 0) AS new_n
            FROM word_year WHERE year BETWEEN ? AND ? GROUP BY word),
        t AS (
            SELECT count(*) FILTER (WHERE year BETWEEN ? AND ?) AS old_t,
                   count(*) FILTER (WHERE year BETWEEN ? AND ?) AS new_t
            FROM pubs WHERE {JOURNAL_CONF} AND title IS NOT NULL)
        SELECT word, old_n AS titles_old, new_n AS titles_new,
               round(100.0 * old_n / old_t, 3) AS pct_old, round(100.0 * new_n / new_t, 3) AS pct_new,
               round(((new_n + 5) / new_t) / ((old_n + 5) / old_t), 2) AS change_x
        FROM p, t
        WHERE {side} >= ? AND word NOT IN (SELECT unnest(?::VARCHAR[]))
        ORDER BY change_x {order} LIMIT ?""",
                [old[0], old[1], new[0], new[1], min(old[0], new[0]), max(old[1], new[1]),
                 old[0], old[1], new[0], new[1], min_titles, skip, limit])


# =========================================================================== venues
def lifespans(cur, meta):
    """eda_03_venues.py 'Series lifespans'."""
    recent = last_full_year(meta) - 1
    return rows(cur, """
        SELECT kind, first_year - first_year % 10 AS started, count(*) AS series,
               round(100.0 * avg((last_year >= ?)::INT), 1) AS pct_still_active,
               median(active_years) AS median_active_years, median(papers) AS median_papers
        FROM series WHERE first_year >= 1970 GROUP BY ALL ORDER BY kind, started""", [recent])


def concentration(cur, meta, frm=1980, to=None, top=10, step=5):
    """eda_03_venues.py 'Concentration over time'."""
    frm, to = _years(frm, to, 1980, last_full_year(meta))
    return rows(cur, f"""
        WITH y AS (
            SELECT CASE WHEN key_prefix = 'journals' THEN 'journal' ELSE 'conference' END AS kind, year, sid,
                   count(*) AS n
            FROM pubs WHERE {JOURNAL_CONF} AND key_prefix IN ('conf', 'journals') AND year BETWEEN ? AND ?
            GROUP BY ALL),
        r AS (SELECT *, row_number() OVER (PARTITION BY kind, year ORDER BY n DESC) AS rk,
                     sum(n) OVER (PARTITION BY kind, year) AS total,
                     count(*) OVER (PARTITION BY kind, year) AS active FROM y)
        SELECT year,
               max(active) FILTER (WHERE kind = 'conference') AS conf_series,
               round(100.0 * sum(n) FILTER (WHERE kind = 'conference' AND rk <= ?)
                     / max(total) FILTER (WHERE kind = 'conference'), 1) AS conf_top_pct,
               max(active) FILTER (WHERE kind = 'journal') AS journal_series,
               round(100.0 * sum(n) FILTER (WHERE kind = 'journal' AND rk <= ?)
                     / max(total) FILTER (WHERE kind = 'journal'), 1) AS journal_top_pct
        FROM r WHERE (year - ?) % ? = 0 GROUP BY year ORDER BY year""", [frm, to, top, top, frm, step])


def publishers(cur, periods, top=8):
    """eda_03_venues.py 'Publishers by DOI prefix' (charts_eda_b.py drops 'other DOI prefix', keeps 8)."""
    if not 1 <= len(periods) <= 4:
        raise BadRequest("give 1-4 periods")
    filters = ", ".join(f"count(*) FILTER (WHERE year BETWEEN {int(a)} AND {int(b)}) AS p{i}"
                        for i, (a, b) in enumerate(periods))
    shares = ", ".join(f"round(100.0 * p{i} / sum(p{i}) OVER (), 1) AS pct_{i}" for i in range(len(periods)))
    pairs = ", ".join(f"('{k}', '{v}')" for k, v in PUBLISHERS.items())
    data = rows(cur, f"""
        WITH names(doi_prefix, publisher) AS (VALUES {pairs}),
        p AS (
            SELECT s.year, CASE WHEN s.doi_prefix = '' THEN '(no DOI)'
                                ELSE coalesce(n.publisher, 'other DOI prefix') END AS publisher
            FROM pubs s LEFT JOIN names n USING (doi_prefix)
            WHERE s.type IN ('article', 'inproceedings') AND NOT s.is_preprint AND s.key_prefix IN ('conf', 'journals')),
        per AS (SELECT publisher, count(*) AS total, {filters} FROM p GROUP BY publisher)
        SELECT publisher, total, {shares} FROM per ORDER BY total DESC""")
    data = [r for r in data if r["publisher"] != "other DOI prefix"][:top]
    return {"periods": [f"{a}–{b}" for a, b in periods], "rows": data}


def doi_gaps(cur, min_papers=2000, max_doi_pct=1.0, limit=12):
    """eda_03_venues.py 'Largest series ... lowest DOI coverage' (charts_eda_b.py keeps pct_doi < 1)."""
    return rows(cur, """
        SELECT kind, usual_name, sid, papers, round(100 * doi_share, 1) AS pct_doi,
               round(100 * oa_share, 1) AS pct_oa, last_year
        FROM series WHERE papers >= ? AND round(100 * doi_share, 1) < ?
        ORDER BY doi_share, papers DESC LIMIT ?""", [min_papers, max_doi_pct, limit])


def venue_search(cur, q, kind=None, limit=25):
    q = (q or "").strip()
    params, where = [], ["TRUE"]
    if q:
        where.append("(usual_name ILIKE ? OR sid ILIKE ?)")
        params += [f"%{q}%", f"%{q}%"]
    if kind:
        where.append("kind = ?")
        params.append(kind)
    return rows(cur, f"""
        SELECT sid, kind, usual_name, papers, first_year, last_year, name_variants,
               round(100 * doi_share, 1) AS pct_doi, round(100 * oa_share, 1) AS pct_oa
        FROM series WHERE {' AND '.join(where)}
        ORDER BY (lower(usual_name) = lower(?)) DESC, papers DESC LIMIT ?""", params + [q, limit])


def venue_detail(cur, sid):
    head = one(cur, "SELECT * FROM series WHERE sid = ?", [sid])
    if not head:
        return None
    base = f"sid = ? AND {JOURNAL_CONF}"
    yearly = rows(cur, f"""
        SELECT year, count(*) AS papers, avg(n_authors) AS mean_authors,
               avg((doi_prefix <> '')::INT) AS doi_share, avg(has_oa::INT) AS oa_share,
               avg((n_orcids > 0)::INT) AS orcid_share, avg((n_unidentified > 0)::INT) AS unidentified_share,
               avg(has_twin::INT) AS twin_share
        FROM pubs WHERE {base} AND year IS NOT NULL GROUP BY year ORDER BY year""", [sid])
    names = rows(cur, f"""
        SELECT venue AS name, count(*) AS papers, min(year) AS first_year, max(year) AS last_year
        FROM pubs WHERE {base} GROUP BY venue ORDER BY papers DESC LIMIT 20""", [sid])
    authors = rows(cur, """
        WITH ps AS (SELECT pid FROM pubs WHERE sid = ? AND type IN ('article', 'inproceedings') AND NOT is_preprint)
        SELECT p.key, p.name, p.page_kind, count(*) AS papers
        FROM slots s JOIN ps USING (pid) JOIN persons p USING (person_id)
        WHERE NOT s.on_bin GROUP BY ALL ORDER BY papers DESC, p.name LIMIT 15""", [sid])
    recent = rows(cur, f"""
        SELECT key, title, year, n_authors FROM pubs WHERE {base}
        ORDER BY year DESC NULLS LAST, key DESC LIMIT 12""", [sid])
    return {"series": head, "yearly": yearly, "names": names, "top_authors": authors, "recent": recent}


# The six dimensions a venue is profiled on. Each is turned into a percentile rank among all series
# with enough papers, so the axes share one scale and "the median venue" is a regular polygon.
PROFILE_AXES = [
    ("papers_per_year", "Papers per active year", "n"),
    ("growth", "Recent growth", "growth"),
    ("oa_share", "Open access", "share"),
    ("doi_share", "Carry a DOI", "share"),
    ("mean_authors", "Authors per paper", "n"),
    ("unidentified_share", "Papers with an unidentified author", "share"),
]


def venue_profiles(cur, meta, min_papers=50):
    """eda_03_venues.py metrics per series, plus their percentile ranks. One table for every series
    that qualifies; `venue_profile` picks a row. Growth is (recent - earlier) / (recent + earlier) over
    the last three complete years vs. the three before, so it is bounded and symmetric."""
    last = last_full_year(meta)
    keys = [k for k, _, _ in PROFILE_AXES]
    ranks = ", ".join(f"round(100 * percent_rank() OVER (ORDER BY {k}), 1) AS rank_{k}" for k in keys)
    data = rows(cur, f"""
        WITH m AS (
            SELECT sid, avg(n_authors) AS mean_authors, avg((n_unidentified > 0)::INT) AS unidentified_share,
                   count(*) FILTER (WHERE year BETWEEN ? AND ?) AS recent,
                   count(*) FILTER (WHERE year BETWEEN ? AND ?) AS earlier
            FROM pubs WHERE {JOURNAL_CONF} AND key_prefix IN ('conf', 'journals') GROUP BY sid),
        v AS (
            SELECT s.sid, s.kind, s.usual_name, s.papers,
                   s.papers::DOUBLE / greatest(s.active_years, 1) AS papers_per_year,
                   CASE WHEN m.recent + m.earlier = 0 THEN 0.0
                        ELSE (m.recent - m.earlier)::DOUBLE / (m.recent + m.earlier) END AS growth,
                   s.oa_share, s.doi_share, m.mean_authors, m.unidentified_share
            FROM series s JOIN m USING (sid) WHERE s.papers >= ?)
        SELECT *, {ranks} FROM v""", [last - 2, last, last - 5, last - 3, min_papers])
    medians = {k: float(np.median([r[k] for r in data])) for k in keys} if data else {}
    return {"rows": {r["sid"]: r for r in data}, "medians": medians, "series_count": len(data),
            "min_papers": min_papers, "windows": {"recent": f"{last - 2}–{last}", "earlier": f"{last - 5}–{last - 3}"}}


def venue_profile(profiles, sid):
    r = profiles["rows"].get(sid)
    if not r:
        return None
    return {
        "sid": sid, "name": r["usual_name"], "kind": r["kind"], "papers": r["papers"],
        "axes": [{"key": k, "label": label, "unit": unit, "value": r[k], "rank": r[f"rank_{k}"],
                  "median": profiles["medians"][k]} for k, label, unit in PROFILE_AXES],
        "series_count": profiles["series_count"], "min_papers": profiles["min_papers"], "windows": profiles["windows"],
    }


def venue_treemap(cur, frm, to, publishers_top=10, series_top=12):
    """Papers of a period by publisher (from the DOI prefix), then by series inside each publisher -
    the hierarchy the publishers bar chart flattens."""
    pairs = ", ".join(f"('{k}', '{v}')" for k, v in PUBLISHERS.items())
    data = rows(cur, f"""
        WITH names(doi_prefix, publisher) AS (VALUES {pairs}),
        p AS (
            SELECT s.sid, CASE WHEN s.doi_prefix = '' THEN '(no DOI)'
                               ELSE coalesce(n.publisher, 'other DOI prefix') END AS publisher
            FROM pubs s LEFT JOIN names n USING (doi_prefix)
            WHERE {JOURNAL_CONF} AND s.key_prefix IN ('conf', 'journals') AND s.year BETWEEN ? AND ?),
        per AS (SELECT publisher, sid, count(*) AS papers FROM p GROUP BY ALL),
        ranked AS (SELECT *, row_number() OVER (PARTITION BY publisher ORDER BY papers DESC) AS rk,
                          sum(papers) OVER (PARTITION BY publisher) AS pub_total FROM per)
        SELECT r.publisher, r.pub_total, r.rk, r.sid, r.papers, se.usual_name AS name, se.kind
        FROM ranked r LEFT JOIN series se USING (sid)
        WHERE r.rk <= ?
        ORDER BY r.pub_total DESC, r.rk""", [frm, to, series_top])
    by_pub = {}
    for r in data:
        pub = by_pub.setdefault(r["publisher"], {"name": r["publisher"], "value": r["pub_total"], "children": [], "shown": 0})
        if r["rk"] <= series_top:
            pub["children"].append({"name": r["name"] or r["sid"], "sid": r["sid"], "kind": r["kind"], "value": r["papers"]})
            pub["shown"] += r["papers"]
    out = []
    for pub in sorted(by_pub.values(), key=lambda x: -x["value"])[:publishers_top]:
        rest = pub["value"] - pub["shown"]
        if rest > 0:
            pub["children"].append({"name": "other series", "sid": None, "kind": None, "value": rest})
        out.append({k: v for k, v in pub.items() if k != "shown"})
    return {"period": f"{frm}–{to}", "children": out}


def venue_scatter(cur, meta, min_papers=300):
    """Every active series large enough to plot: size against open-access share, by kind."""
    last = last_full_year(meta)
    return rows(cur, """
        SELECT sid, usual_name AS name, kind, papers, round(100 * oa_share, 1) AS pct_oa,
               round(100 * doi_share, 1) AS pct_doi, last_year
        FROM series WHERE papers >= ? AND last_year >= ? ORDER BY papers DESC""", [min_papers, last - 2])


# =========================================================================== quality
COVERAGE_TYPES = ["article", "inproceedings", "www", "phdthesis", "incollection", "proceedings", "data", "book",
                  "mastersthesis"]
COVERAGE_FIELDS = [
    ("title", "title IS NOT NULL"), ("authors", "n_authors > 0"), ("author ORCID", "n_orcids > 0"),
    ("editors", "n_editors > 0"), ("year", "year IS NOT NULL"), ("journal", "journal IS NOT NULL"),
    ("booktitle", "booktitle IS NOT NULL"), ("volume", "volume IS NOT NULL"), ("pages", "pages IS NOT NULL"),
    ("publisher", "publisher IS NOT NULL"), ("school", "school IS NOT NULL"),
    ("crossref", "crossref IS NOT NULL"), ("isbn", "len(isbn) > 0"), ("link (ee)", "len(ee) > 0"),
    ("open access flag", "has_oa"), ("notes", "len(notes) > 0"), ("citations", "n_cites > 0"),
]


def coverage(cur):
    """make_charts.py chart_field_coverage (all 12.9M records, straight from the parquet)."""
    cols = ", ".join(f"avg(({cond})::INT) AS f{i}" for i, (_, cond) in enumerate(COVERAGE_FIELDS))
    by_type = {r["type"]: r for r in rows(cur, f"SELECT type, count(*) AS records, {cols} FROM src GROUP BY type")}
    types = [t for t in COVERAGE_TYPES if t in by_type]
    return {
        "types": types,
        "records": [by_type[t]["records"] for t in types],
        "fields": [f for f, _ in COVERAGE_FIELDS],
        "matrix": [[by_type[t][f"f{i}"] for t in types] for i in range(len(COVERAGE_FIELDS))],
    }


def page_formats(cur):
    """make_charts.py chart_page_formats."""
    return rows(cur, r"""
        SELECT CASE WHEN pages IS NULL THEN 'No pages'
                    WHEN regexp_matches(pages, '^[0-9]+-[0-9]+$') THEN 'Start-end  (482-494)'
                    WHEN regexp_matches(pages, '^[0-9]+$') THEN 'Single number  (604)'
                    WHEN regexp_matches(pages, '^[0-9]+:[0-9]+-[0-9]+:[0-9]+$') THEN 'Article:page  (41:1-41:5)'
                    WHEN regexp_matches(pages, '^[A-Za-z]') THEN 'Starts with a letter  (xiv)'
                    ELSE 'Other  (186-)' END AS page_format,
               count(*) AS papers
        FROM pubs WHERE type IN ('article', 'inproceedings')
        GROUP BY ALL ORDER BY papers DESC""")


def theses(cur, limit=12):
    """eda_05_other_types.py 'Last part of the school string'."""
    by_country = rows(cur, """
        SELECT trim(regexp_extract(school, '([^,]+)$', 1)) AS country, count(*) AS theses,
               round(100.0 * count(*) / sum(count(*)) OVER (), 1) AS pct
        FROM pubs WHERE type = 'phdthesis' AND school IS NOT NULL
        GROUP BY ALL ORDER BY theses DESC LIMIT ?""", [limit])
    by_year = rows(cur, """
        SELECT year, count(*) AS theses, round(100 * avg(has_oa::INT), 1) AS pct_oa,
               round(100 * avg((n_orcids > 0)::INT), 1) AS pct_orcid
        FROM pubs WHERE type = 'phdthesis' AND year BETWEEN 1990 AND 2026 GROUP BY year ORDER BY year""")
    hosts = rows(cur, """
        SELECT link_host, count(*) AS theses FROM pubs WHERE type = 'phdthesis'
        GROUP BY ALL ORDER BY theses DESC LIMIT 8""")
    return {"countries": by_country, "by_year": by_year, "hosts": hosts}


def research_data(cur, meta, frm=2015, to=None):
    """eda_05_other_types.py 'Research data records by year'."""
    frm, to = _years(frm, to, 2015, last_full_year(meta))
    by_year = rows(cur, """
        SELECT year, count(*) AS records,
               count(*) FILTER (WHERE publtype = 'version') AS versions,
               count(*) FILTER (WHERE publtype = 'concept') AS concepts,
               count(*) FILTER (WHERE publtype IS NULL) AS plain,
               round(100 * avg((n_orcids > 0)::INT), 1) AS pct_orcid
        FROM pubs WHERE type = 'data' AND year BETWEEN ? AND ? GROUP BY year ORDER BY year""", [frm, to])
    hosts = rows(cur, """
        SELECT link_host, coalesce(publisher, '(none)') AS publisher, count(*) AS records
        FROM pubs WHERE type = 'data' GROUP BY ALL ORDER BY records DESC LIMIT 8""")
    return {"by_year": by_year, "hosts": hosts}


# =========================================================================== tails
TAIL_PANELS = {
    "papers_per_author": ("Papers per author", "papers per author", """
        SELECT ps.n_pubs AS v, count(*) AS n FROM person_stats ps JOIN persons p USING (person_id)
        WHERE p.page_kind <> 'disambiguation' AND ps.n_pubs > 0 GROUP BY 1 ORDER BY 1"""),
    "coauthors_per_author": ("Co-authors per author", "co-authors per author", """
        SELECT d.n_coauthors AS v, count(*) AS n FROM person_degree d JOIN persons p USING (person_id)
        WHERE p.page_kind <> 'disambiguation' AND d.n_coauthors > 0 GROUP BY 1 ORDER BY 1"""),
    "papers_per_series": ("Papers per venue series", "papers per venue series", """
        SELECT papers AS v, count(*) AS n FROM series GROUP BY 1 ORDER BY 1"""),
}


def tails(cur, panel, fits):
    """make_charts.py ccdf + eda_07_statistics.py inequality summary, on all data."""
    if panel not in TAIL_PANELS:
        raise BadRequest(f"panel must be one of {', '.join(TAIL_PANELS)}")
    label, fit_name, sql = TAIL_PANELS[panel]
    hist = cur.execute(sql).fetchall()
    if not hist:
        return {"panel": panel, "label": label, "points": [], "stats": None}
    x = np.array([h[0] for h in hist], dtype=float)
    c = np.array([h[1] for h in hist], dtype=float)
    n = c.sum()
    share_at_least = c[::-1].cumsum()[::-1] / n
    total = (x * c).sum()
    # Gini with grouped values: ranks r+1..r+c for each group
    ranks_before = np.concatenate([[0], c.cumsum()[:-1]])
    rank_sum = x * (c * ranks_before + c * (c + 1) / 2)
    gini = 2 * rank_sum.sum() / (n * total) - (n + 1) / n
    k = max(1, int(n // 100))
    top_sum, need = 0.0, k
    for xi, ci in zip(x[::-1], c[::-1]):
        take = min(ci, need)
        top_sum += take * xi
        need -= take
        if need <= 0:
            break
    median = float(x[np.searchsorted(c.cumsum() / n, 0.5)])
    fit = fits.get(fit_name, {}) if fits else {}
    xmin = fit.get("xmin")
    live_alpha = None
    if isinstance(xmin, (int, float)) and xmin >= 1:
        tail = x >= xmin
        n_tail = c[tail].sum()
        if n_tail > 0:
            # discrete MLE approximation (Clauset, Shalizi & Newman 2009, eq. 3.7) at the job's xmin
            live_alpha = float(1 + n_tail / (c[tail] * np.log(x[tail] / (xmin - 0.5))).sum())
    return {
        "panel": panel, "label": label,
        "points": [{"x": float(a), "n": int(b), "share": float(s)} for a, b, s in zip(x, c, share_at_least)],
        "stats": {
            "n": int(n), "mean": round(float(total / n), 2), "median": median, "max": float(x[-1]),
            "gini": round(float(gini), 3), "top1_share": round(float(100 * top_sum / total), 1),
            "alpha_live": round(live_alpha, 2) if live_alpha else None,
        },
        "fit": fit or None,
    }


def joint_density(cur, bins_per_decade=6):
    """
    Papers per author against distinct co-authors per author, as a 2-D histogram in log10 bins: the
    joint distribution behind the two tails panels. Millions of authors make a scatter unreadable;
    a density is the right form. Bins (disambiguation pages) are excluded as everywhere; authors with
    no co-author or no paper cannot sit on a log axis and are reported as the share left out.
    """
    b = int(bins_per_decade)
    cells = rows(cur, f"""
        SELECT floor(log10(ps.n_pubs) * {b})::INTEGER AS bx, floor(log10(d.n_coauthors) * {b})::INTEGER AS by,
               count(*) AS n
        FROM person_stats ps JOIN person_degree d USING (person_id) JOIN persons p USING (person_id)
        WHERE p.page_kind <> 'disambiguation' AND ps.n_pubs > 0 AND d.n_coauthors > 0
        GROUP BY ALL""")
    total_all = one(cur, """
        SELECT count(*) AS n FROM person_stats ps JOIN persons p USING (person_id)
        WHERE p.page_kind <> 'disambiguation' AND ps.n_pubs > 0""")["n"]
    plotted = sum(c["n"] for c in cells)
    nx = (max(c["bx"] for c in cells) + 1) if cells else 0
    ny = (max(c["by"] for c in cells) + 1) if cells else 0
    grid = [[0] * nx for _ in range(ny)]
    for c in cells:
        grid[c["by"]][c["bx"]] = c["n"]
    return {
        "bins_per_decade": b, "nx": nx, "ny": ny, "grid": grid,
        "authors_plotted": int(plotted), "authors_total": int(total_all),
        "left_out_share": round(1 - plotted / total_all, 4) if total_all else None,
    }


def team_boxes(cur, meta, frm=1970, to=None, step=5):
    """Authors per paper as a distribution per period, not just its mean: the whiskers are the 5th
    and 95th percentiles, the box the quartiles."""
    frm, to = _years(frm, to, 1970, last_full_year(meta))
    return rows(cur, f"""
        SELECT (year - ?) // ? * ? + ? AS period_start,
               least((year - ?) // ? * ? + ? + ? - 1, ?) AS period_end,
               count(*) AS papers,
               quantile_cont(n_authors, 0.05) AS p5, quantile_cont(n_authors, 0.25) AS q1,
               quantile_cont(n_authors, 0.5) AS median, quantile_cont(n_authors, 0.75) AS q3,
               quantile_cont(n_authors, 0.95) AS p95, avg(n_authors) AS mean, max(n_authors) AS max
        FROM pubs WHERE {JOURNAL_CONF} AND n_authors > 0 AND year BETWEEN ? AND ?
        GROUP BY 1, 2 ORDER BY 1""", [frm, step, step, frm, frm, step, step, frm, step, to, frm, to])


# =========================================================================== explore: authors
def author_search(cur, q, limit=30):
    q = (q or "").strip()
    if len(q) < 2:
        raise BadRequest("type at least 2 characters")
    # one namesake count over the whole result set, not a scan of 4.2M pages per row
    return rows(cur, """
        WITH hits AS (SELECT DISTINCT person_id FROM person_names WHERE name ILIKE ?),
        res AS (
            SELECT p.key, p.name, p.base_name, p.page_kind, p.n_names,
                   coalesce(ps.n_pubs, 0) AS papers, c.first_year, c.last_year,
                   list_filter(p.notes, lambda n: n LIKE 'affiliation: %')[1] AS affiliation_note
            FROM hits h JOIN persons p USING (person_id)
            LEFT JOIN person_stats ps USING (person_id)
            LEFT JOIN career c USING (person_id)
            ORDER BY (lower(p.base_name) = lower(?)) DESC, papers DESC, p.name
            LIMIT ?),
        ns AS (SELECT base_name, count(*) AS namesakes FROM persons
               WHERE base_name IN (SELECT base_name FROM res) GROUP BY base_name)
        SELECT res.key, res.name, res.page_kind, res.n_names, res.papers, res.first_year, res.last_year,
               substr(res.affiliation_note, 14) AS affiliation, ns.namesakes
        FROM res LEFT JOIN ns USING (base_name)
        ORDER BY (lower(res.base_name) = lower(?)) DESC, res.papers DESC, res.name""",
                [f"%{q}%", q, limit, q])


def author_detail(cur, key):
    person = one(cur, """
        SELECT person_id, key, name, names, page_kind, publtype, urls, notes, base_name,
               has_affiliation, has_orcid_link, has_wikidata_link
        FROM persons WHERE key = ?""", [key])
    if not person:
        return None
    pid = person.pop("person_id")
    person["affiliations"] = [n[len("affiliation: "):] for n in (person.get("notes") or [])
                              if n.startswith("affiliation: ")]
    stats = one(cur, """
        SELECT count(*) AS papers, min(year) AS first_year, max(year) AS last_year,
               round(avg(n_authors), 2) AS mean_team,
               round(100 * avg(has_orcid::INT), 1) AS pct_with_orcid,
               count(*) FILTER (WHERE n_authors >= 3 AND position = 1) AS first_author,
               count(*) FILTER (WHERE n_authors >= 3 AND position = n_authors) AS last_author,
               count(*) FILTER (WHERE n_authors >= 3) AS on_3plus
        FROM slots WHERE person_id = ?""", [pid])
    yearly = rows(cur, f"""
        SELECT year, {KIND_SQL} AS kind, count(*) AS papers
        FROM slots WHERE person_id = ? AND year IS NOT NULL GROUP BY ALL ORDER BY year""", [pid])
    coauthors = rows(cur, """
        WITH mine AS (SELECT pid FROM slots WHERE person_id = ? AND n_authors BETWEEN 2 AND 50)
        SELECT p.key, p.name, p.page_kind, count(*) AS papers
        FROM slots s JOIN mine USING (pid) JOIN persons p USING (person_id)
        WHERE s.person_id <> ?
        GROUP BY ALL ORDER BY papers DESC, p.name LIMIT 15""", [pid, pid])
    n_coauthors = one(cur, """
        WITH mine AS (SELECT pid FROM slots WHERE person_id = ? AND n_authors BETWEEN 2 AND 50)
        SELECT count(DISTINCT s.person_id) AS n FROM slots s JOIN mine USING (pid)
        WHERE s.person_id <> ?""", [pid, pid])
    venues = rows(cur, """
        WITH mine AS (SELECT pid FROM slots WHERE person_id = ?)
        SELECT b.sid, mode(b.venue) AS name, count(*) AS papers
        FROM pubs b JOIN mine USING (pid)
        WHERE b.key_prefix IN ('conf', 'journals') AND NOT b.is_preprint
        GROUP BY b.sid ORDER BY papers DESC LIMIT 10""", [pid])
    papers = rows(cur, f"""
        WITH mine AS (SELECT pid, position FROM slots WHERE person_id = ?)
        SELECT b.key, b.title, b.year, b.venue, {KIND_SQL} AS kind,
               b.n_authors, m.position
        FROM pubs b JOIN mine m USING (pid)
        ORDER BY b.year DESC NULLS LAST, b.key DESC LIMIT 30""", [pid])
    namesakes = rows(cur, """
        SELECT p.key, p.name, p.page_kind, coalesce(ps.n_pubs, 0) AS papers
        FROM persons p LEFT JOIN person_stats ps USING (person_id)
        WHERE p.base_name = ? AND p.key <> ?
        ORDER BY papers DESC LIMIT 12""", [person["base_name"], key])
    total_namesakes = one(cur, "SELECT count(*) AS n FROM persons WHERE base_name = ?", [person["base_name"]])
    return {"person": person, "stats": {**stats, "coauthors": n_coauthors["n"]}, "yearly": yearly,
            "coauthors": coauthors, "venues": venues, "papers": papers,
            "namesakes": namesakes, "namesake_count": total_namesakes["n"]}


# =========================================================================== explore: papers
def paper_search(cur, q, kind=None, frm=None, to=None, limit=30):
    words_ = [w for w in re.split(r"\s+", (q or "").strip()) if w]
    if not words_ or sum(len(w) for w in words_) < 3:
        raise BadRequest("type at least 3 characters")
    if len(words_) > 8:
        raise BadRequest("at most 8 words")
    where = ["title ILIKE ?"] * len(words_)
    params = [f"%{w}%" for w in words_]
    if kind:
        where.append(f"{KIND_SQL} = ?")
        params.append(kind)
    if frm is not None:
        where.append("year >= ?")
        params.append(frm)
    if to is not None:
        where.append("year <= ?")
        params.append(to)
    return rows(cur, f"""
        SELECT key, title, year, venue, {KIND_SQL} AS kind, n_authors, n_unidentified, has_twin, has_oa
        FROM pubs WHERE {' AND '.join(where)}
        ORDER BY year DESC NULLS LAST, key LIMIT ?""", params + [limit])


def author_ego(cur, key, limit=30):
    """The author's strongest co-authors and which of those also work with each other: the ego
    network, whose groups a force layout makes visible. Same graph as the network job - papers with
    2-50 authors, disambiguation bins excluded. Clustering is computed among the co-authors shown."""
    person = one(cur, "SELECT person_id, key, name, page_kind FROM persons WHERE key = ?", [key])
    if not person:
        return None
    pid = person.pop("person_id")
    mine = "SELECT pid FROM slots WHERE person_id = ? AND n_authors BETWEEN 2 AND 50"
    coauthors = rows(cur, f"""
        WITH mine AS ({mine})
        SELECT p.person_id, p.key, p.name, p.page_kind, count(*) AS papers, max(s.year) AS last_year
        FROM slots s JOIN mine USING (pid) JOIN persons p USING (person_id)
        WHERE s.person_id <> ? AND NOT s.on_bin
        GROUP BY ALL ORDER BY papers DESC, p.name LIMIT ?""", [pid, pid, limit])
    degree = one(cur, f"""
        WITH mine AS ({mine})
        SELECT count(DISTINCT s.person_id) AS n FROM slots s JOIN mine USING (pid)
        WHERE s.person_id <> ? AND NOT s.on_bin""", [pid, pid])["n"]
    ids = [c["person_id"] for c in coauthors]
    edges = rows(cur, """
        WITH members AS (
            SELECT person_id, pid FROM slots
            WHERE person_id IN (SELECT unnest(?::INTEGER[])) AND n_authors BETWEEN 2 AND 50)
        SELECT a.person_id AS a, b.person_id AS b, count(*) AS papers
        FROM members a JOIN members b ON a.pid = b.pid AND a.person_id < b.person_id
        GROUP BY ALL""", [ids]) if ids else []
    k = len(coauthors)
    clustering = round(2 * len(edges) / (k * (k - 1)), 3) if k >= 2 else None
    return {"author": person, "coauthors": coauthors, "edges": edges,
            "degree": int(degree), "shown": k, "clustering": clustering}


def paper_detail(cur, key):
    head = one(cur, f"""
        SELECT pid, key, sid, venue, {KIND_SQL} AS kind, is_preprint, has_twin, n_unidentified, title_norm
        FROM pubs WHERE key = ?""", [key])
    if not head:
        return None
    record = one(cur, "SELECT * FROM src WHERE key = ?", [key]) or {}
    orcids = record.get("author_orcids") or []
    authors = [{"position": i, "name": name, "orcid": orcids[i - 1] if i - 1 < len(orcids) else None}
               for i, name in enumerate(record.get("authors") or [], start=1)]
    if authors:
        resolved = {r["name"]: r for r in rows(cur, """
            SELECT pn.name, p.key, p.page_kind FROM person_names pn JOIN persons p USING (person_id)
            WHERE pn.name IN (SELECT unnest(?::VARCHAR[]))""", [[a["name"] for a in authors]])}
        for a in authors:
            r = resolved.get(a["name"])
            a["key"] = r["key"] if r else None
            a["page_kind"] = r["page_kind"] if r else "unresolved"
    twins = []
    if head["has_twin"]:
        twins = rows(cur, f"""
            SELECT key, title, year, venue, {KIND_SQL} AS kind FROM pubs
            WHERE title_norm = ? AND key <> ? ORDER BY year LIMIT 10""", [head["title_norm"], key])
    head.pop("pid")
    head.pop("title_norm")
    return {"paper": head, "record": record, "authors": authors, "twins": twins}

