"""
Check that the live queries reproduce the report: compare them with the CSVs the analysis wrote
(~/dblp/charts, mounted at CHARTS_DIR).

    docker compose -f docker-compose.prod.yml exec api python -m app.validate

Network, OpenAlex and power-law results are not compared: the API reads those jobs' own output files.
"""
import csv
import sys

from . import config, queries as Q
from .serving import serving


def num(v):
    s = str(v).replace(",", "").replace("%", "").strip()
    if s in ("", "-", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return s


def load(name):
    path = config.CHARTS_DIR / f"{name}.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Report:
    def __init__(self):
        self.lines = []
        self.failed = 0

    def compare(self, name, pairs, tol):
        """pairs: (label, expected, actual). tol: absolute tolerance."""
        worst, worst_label, missing, n = 0.0, "", 0, 0
        for label, exp, act in pairs:
            exp, act = num(exp), num(act)
            if exp is None:
                continue
            n += 1
            if act is None:
                missing += 1
                continue
            if isinstance(exp, str) or isinstance(act, str):
                d = 0.0 if str(exp) == str(act) else float("inf")
            else:
                d = abs(exp - act)
            if d > worst:
                worst, worst_label = d, label
        ok = missing == 0 and worst <= tol
        self.failed += 0 if ok else 1
        mark = "OK  " if ok else "DIFF"
        detail = f"max diff {worst:g} at {worst_label}" if worst else "identical"
        if missing:
            detail += f"; {missing} missing"
        self.lines.append(f"{mark} {name:<26} {n:>5} values  {detail}")

    def skip(self, name, why):
        self.lines.append(f"--   {name:<26} {why}")


def main():
    serving.ensure()
    cur = serving.cursor()
    meta = serving.meta
    r = Report()

    def check(name, fn):
        rows = load(name)
        if rows is None:
            r.skip(name, "CSV not found")
            return
        try:
            fn(rows)
        except Exception as e:  # keep going: one broken check should not hide the others
            r.failed += 1
            r.lines.append(f"ERR  {name:<26} {type(e).__name__}: {e}")

    def by(rows, key):
        return {str(x[key]): x for x in rows}

    def growth(rows):
        live = by(Q.growth(cur, meta, 1970, 2025), "year")
        r.compare("01_growth_by_kind", [(f"{x['year']}/{k}", x[k], live.get(x["year"], {}).get(k))
                                        for x in rows for k in ("conference", "journal", "preprint")], 0)
    check("01_growth_by_kind", growth)

    def teams(rows):
        live = by(Q.teams(cur, meta, 1970, 2025), "year")
        r.compare("02_team_size", [(f"{x['year']}/{k}", x[k], live.get(x["year"], {}).get(k))
                                   for x in rows for k in ("mean_authors", "single_author_share")], 1e-6)
    check("02_team_size", teams)

    def metadata(rows):
        live = by(Q.metadata_trends(cur, meta, 2000, 2025), "year")
        cols = ("with_orcid", "with_unidentified_author", "with_title_twin", "open_access")
        r.compare("03_metadata_trends", [(f"{x['year']}/{k}", x[k], live.get(x["year"], {}).get(k))
                                         for x in rows for k in cols], 1e-6)
    check("03_metadata_trends", metadata)

    def tails(rows):
        for panel, key in [("Papers per author", "papers_per_author"), ("Co-authors per author", "coauthors_per_author")]:
            live = {int(p["x"]): p["n"] for p in Q.tails(cur, key, None)["points"]}
            r.compare(f"04_heavy_tails/{key}", [(x["value"], x["count"], live.get(int(float(x["value"]))))
                                                for x in rows if x["panel"] == panel], 0)
        r.skip("04_heavy_tails/venues", "report used venue names; the dashboard uses venue series (as the stats job did)")
    check("04_heavy_tails", tails)

    def homonyms(rows):
        live = by(Q.homonyms(cur, 100), "base_name")
        r.compare("05_top_homonyms", [(x["base_name"], x["distinct_people"],
                                       live.get(x["base_name"], {}).get("distinct_people")) for x in rows], 0)
    check("05_top_homonyms", homonyms)

    def affiliation(rows):
        live = Q.affiliation(cur)
        cols = {"Has an affiliation": "affiliation", "Links to ORCID": "orcid", "Links to Wikidata": "wikidata"}
        r.compare("06_affiliation_effect", [(f"{x['page_kind']}/{c}", x[c], live.get(x["page_kind"], {}).get(k))
                                            for x in rows for c, k in cols.items()], 1e-9)
    check("06_affiliation_effect", affiliation)

    def pages(rows):
        live = by(Q.page_formats(cur), "page_format")
        r.compare("07_page_formats", [(x["page_format"], x["papers"],
                                       live.get(x["page_format"], {}).get("papers")) for x in rows], 0)
    check("07_page_formats", pages)

    def cov(rows):
        live = Q.coverage(cur)
        idx = {f: i for i, f in enumerate(live["fields"])}
        pairs = []
        for x in rows:
            for t in live["types"]:
                if t in x and x["field"] in idx:
                    pairs.append((f"{x['field']}/{t}", x[t], live["matrix"][idx[x["field"]]][live["types"].index(t)]))
        r.compare("08_field_coverage", pairs, 6e-5)
    check("08_field_coverage", cov)

    def waves(rows):
        cols = [c for c in rows[0] if c != "year"]
        live = Q.terms(cur, meta, cols, 2000, 2025)
        at = {y: i for i, y in enumerate(live["years"])}
        vals = {s["term"]: s["values"] for s in live["series"]}
        r.compare("09_topic_waves", [(f"{x['year']}/{c}", x[c], vals[c][at[int(x["year"])]] if int(x["year"]) in at else None)
                                     for x in rows for c in cols], 0.006)
    check("09_topic_waves", waves)

    def rising(rows):
        live = {w["word"]: w["change_x"] for d in ("rising", "falling")
                for w in Q.words(cur, (2011, 2015), (2021, 2025), d, 1500, 100, False)}
        r.compare("10_rising_falling_words", [(x["word"], x["change_x"], live.get(x["word"])) for x in rows], 0.006)
    check("10_rising_falling_words", rising)

    def style(rows):
        live = by(Q.title_style(cur, meta), "decade")
        r.compare("11_title_style", [(f"{x['decade']}/{k}", x[k], live.get(x["decade"], {}).get(k))
                                     for x in rows for k in ("colon", "name_colon", "question")], 0.051)
    check("11_title_style", style)

    for name in ("12_network_growth", "13_distances", "14_communities"):
        r.skip(name, "served from the network job's output file")

    def pubs(rows):
        live = Q.publishers(cur, ((2001, 2005), (2011, 2015), (2021, 2025)), 30)
        m = {x["publisher"]: x for x in live["rows"]}
        cols = {"pct_2001_05": "pct_0", "pct_2011_15": "pct_1", "pct_2021_25": "pct_2"}
        r.compare("15_publishers", [(f"{x['publisher']}/{c}", x[c], m.get(x["publisher"], {}).get(k))
                                    for x in rows for c, k in cols.items()], 0.051)
    check("15_publishers", pubs)

    def life(rows):
        live = {(x["kind"], str(x["started"])): x for x in Q.lifespans(cur, meta)}
        r.compare("16_series_lifespans", [(f"{x['kind']}/{x['started']}/{k}", x[k],
                                           live.get((x["kind"], x["started"]), {}).get(k))
                                          for x in rows for k in ("pct_still_active", "median_active_years")], 0.051)
    check("16_series_lifespans", life)

    def conc(rows):
        live = by(Q.concentration(cur, meta, 1980, 2025, 10, 5), "year")
        cols = {"conf_top10_pct": "conf_top_pct", "journal_top10_pct": "journal_top_pct"}
        r.compare("17_concentration", [(f"{x['year']}/{c}", x[c], live.get(x["year"], {}).get(k))
                                       for x in rows for c, k in cols.items()], 0.051)
    check("17_concentration", conc)

    def gaps(rows):
        live = by(Q.doi_gaps(cur, 2000, 1.0, 50), "usual_name")
        r.compare("18_doi_gaps", [(f"{x['usual_name']}/{k}", x[k], live.get(x["usual_name"], {}).get(k))
                                  for x in rows for k in ("papers", "pct_doi", "pct_oa")], 0.051)
    check("18_doi_gaps", gaps)

    def newc(rows):
        live = by(Q.newcomers(cur, meta, 1970, 2025), "year")
        r.compare("19_newcomers", [(f"{x['year']}/{k}", x[k], live.get(x["year"], {}).get(k))
                                   for x in rows for k in ("active_authors", "new_authors", "pct_newcomers")], 0.051)
    check("19_newcomers", newc)

    def coh(rows):
        live = by(Q.cohorts(cur, meta, 1975, 2020, 5), "cohort")
        cols = {"pct_single_paper": "pct_single_paper", "pct_publishing_5y_later": "pct_5y",
                "pct_10y_later": "pct_10y", "pct_20y_later": "pct_20y"}
        r.compare("20_cohort_survival", [(f"{x['cohort']}/{c}", x[c], live.get(x["cohort"], {}).get(k))
                                         for x in rows for c, k in cols.items()], 0.051)
    check("20_cohort_survival", coh)

    def pos(rows):
        live = by(Q.positions(cur, 3), "author_total_papers")
        r.compare("21_author_position", [(f"{x['author_total_papers']}/{k}", x[k],
                                          live.get(x["author_total_papers"], {}).get(k))
                                         for x in rows for k in ("pct_first", "pct_middle", "pct_last")], 0.051)
    check("21_author_position", pos)

    def alpha(rows):
        live = Q.alphabetical(cur, 2000, 2000, 50, 50)
        m = {x["usual_name"]: x for x in live["most"] + live["least"]}
        r.compare("22_alphabetical_order", [(f"{x['usual_name']}/{k}", x[k], m.get(x["usual_name"], {}).get(k))
                                            for x in rows for k in ("papers", "pct_alphabetical", "pct_by_chance")],
                  0.051)
    check("22_alphabetical_order", alpha)

    def th(rows):
        live = by(Q.theses(cur, 50)["countries"], "country")
        r.compare("23_thesis_countries", [(f"{x['country']}/{k}", x[k], live.get(x["country"], {}).get(k))
                                          for x in rows for k in ("theses", "pct")], 0.051)
    check("23_thesis_countries", th)

    def rd(rows):
        live = by(Q.research_data(cur, meta, 2015, 2025)["by_year"], "year")
        r.compare("24_research_data", [(f"{x['year']}/{k}", x[k], live.get(x["year"], {}).get(k))
                                       for x in rows for k in ("records", "versions", "plain", "pct_orcid")], 0.051)
    check("24_research_data", rd)

    for name in ("25_openalex_coverage", "26_openalex_fields"):
        r.skip(name, "served from the OpenAlex job's output file")

    def rates(rows):
        live = {(x["kind"], x["period"]): x for x in Q.growth_rates(cur, meta, 1990, 2023, 2005)["rows"]}
        r.compare("27_growth_rates", [(f"{x['kind']}/{x['period']}/{c}", x[c], live.get((x["kind"], x["period"]), {}).get(k))
                                      for x in rows for c, k in (("growth_per_year", "rate"), ("doubling_years", "doubling"))],
                  0.051)
    check("27_growth_rates", rates)

    print(f"serving database built {meta.get('built_at')} from {meta.get('parquet')} "
          f"(latest record edit {meta.get('latest_mdate')})\n")
    print("\n".join(r.lines))
    print(f"\n{r.failed} check(s) differ" if r.failed else "\nall checks match the report")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
