#!/usr/bin/env python3
"""Convert the 27 chart CSVs into one clean data.js for the dashboard."""
import csv
import json
import re
from pathlib import Path

SRC = Path(r"C:\Users\User\Downloads\dblp_charts\charts")
OUT = Path(r"C:/Users/User/dblp-dashboard/src/data.json")


def num(v):
    if v is None or v == "":
        return None
    s = str(v).replace(",", "").replace("%", "").strip()
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return v


def load(name, cast_all=True):
    rows = list(csv.DictReader(open(SRC / f"{name}.csv", encoding="utf-8")))
    if cast_all:
        rows = [{k: num(v) for k, v in r.items()} for r in rows]
    return rows


data = {}

data["growth_by_kind"] = load("01_growth_by_kind")
data["team_size"] = load("02_team_size")
data["metadata_trends"] = load("03_metadata_trends")

tails = load("04_heavy_tails")
by_panel = {}
for r in tails:
    by_panel.setdefault(r["panel"], []).append({"x": r["value"], "n": r["count"], "share": r["share_at_least"]})
data["heavy_tails"] = by_panel

data["top_homonyms"] = load("05_top_homonyms")

aff = load("06_affiliation_effect")
data["affiliation_effect"] = {r["page_kind"]: {"affiliation": r["Has an affiliation"],
                                                "orcid": r["Links to ORCID"],
                                                "wikidata": r["Links to Wikidata"]} for r in aff}

data["page_formats"] = load("07_page_formats")

cov_rows = list(csv.DictReader(open(SRC / "08_field_coverage.csv", encoding="utf-8")))
cov_types = [c for c in cov_rows[0].keys() if c != "field"]
data["field_coverage"] = {"types": cov_types,
                           "fields": [r["field"] for r in cov_rows],
                           "matrix": [[num(r[t]) for t in cov_types] for r in cov_rows]}

data["topic_waves"] = load("09_topic_waves")
data["rising_falling_words"] = load("10_rising_falling_words")
data["title_style"] = load("11_title_style")
data["network_growth"] = load("12_network_growth")
data["distances"] = load("13_distances")

comm_rows = list(csv.DictReader(open(SRC / "14_communities.csv", encoding="utf-8")))
communities = []
for r in comm_rows:
    if not r.get("authors"):
        continue
    venues = [{"name": m.group(1).strip(), "papers": num(m.group(2))}
              for m in re.finditer(r"([^,()]+)\s*\((\d[\d,]*)\)", r["top_venues"])]
    communities.append({"authors": num(r["authors"]), "venues": venues})
data["communities"] = communities

data["publishers"] = load("15_publishers")
data["series_lifespans"] = load("16_series_lifespans")
data["concentration"] = load("17_concentration")
data["doi_gaps"] = load("18_doi_gaps")
data["newcomers"] = load("19_newcomers")
data["cohort_survival"] = load("20_cohort_survival")
data["author_position"] = load("21_author_position")
data["alphabetical_order"] = load("22_alphabetical_order")
data["thesis_countries"] = load("23_thesis_countries")
data["research_data"] = load("24_research_data")

oa_rows = list(csv.DictReader(open(SRC / "25_openalex_coverage.csv", encoding="utf-8")))
data["openalex_coverage"] = [{"kind": r["kind"], "found": num(r["Found in OpenAlex"]),
                               "abstract": num(r["Has an abstract"]), "institution": num(r["Has an institution"])}
                              for r in oa_rows]
data["openalex_fields"] = load("26_openalex_fields")

gr_rows = list(csv.DictReader(open(SRC / "27_growth_rates.csv", encoding="utf-8")))
growth_rates = []
for r in gr_rows:
    lo, hi = [float(x.replace("%", "")) for x in r["ci_95"].split(" to ")]
    growth_rates.append({"kind": r["kind"], "period": r["period"],
                          "rate": float(r["growth_per_year"].replace("%", "")),
                          "lo": lo, "hi": hi, "doubling": num(r["doubling_years"])})
data["growth_rates"] = growth_rates

# ---- headline KPIs (from the report) --------------------------------------
data["kpis"] = [
    {"n": "12.93M", "l": "records: 8.67M publications, 4.19M author pages, 65K proceedings volumes"},
    {"n": "11 words", "l": "of text per paper \u2014 titles are the only text; no abstracts, citations or affiliations"},
    {"n": "522", "l": "different people named \u201cWei Wang\u201d alone"},
    {"n": "1 in 6", "l": "papers has an author dblp hasn\u2019t identified (17.1%); 1 in 4 in the 2020s"},
    {"n": "100% vs 1.7%", "l": "of numbered vs. regular author pages carry an affiliation"},
    {"n": "9.5%/yr", "l": "growth in papers since 1990 \u2014 doubling every 7.6 years"},
    {"n": "44.7%", "l": "of preprint titles also exist as a separate, unlinked published paper"},
    {"n": "5.6 steps", "l": "average co-authorship distance; 94.5% of authors in one component"},
    {"n": "51%", "l": "of dblp papers are primarily CS by OpenAlex\u2019s classification"},
]

OUT.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB), {len(data)} datasets")
for k, v in data.items():
    n = len(v) if isinstance(v, list) else (len(v) if isinstance(v, dict) else "?")
    print(f"  {k:<22} {n}")
