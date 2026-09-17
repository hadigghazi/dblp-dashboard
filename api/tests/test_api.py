import os

import duckdb
import pytest


def get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, (path, r.status_code, r.text[:500])
    return r.json()["data"]


def src(data_dir, sql):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW src AS SELECT * FROM read_parquet('{(data_dir / 'parquet' / 'dblp.parquet').as_posix()}')")
    return con.execute(sql).fetchall()


# ------------------------------------------------------------------ status
def test_status_reports_ready_and_jobs(client):
    body = client.get("/api/status").json()
    assert body["status"]["state"] == "ready"
    assert body["meta"]["last_full_year"] == "2025"
    assert body["meta"]["latest_mdate"] == "2026-08-31"
    assert body["jobs"]["network"]["file"] == "02_network.txt"


def test_optional_step_built(serving):
    assert serving.meta["optional_failed"] == ""


# ------------------------------------------------------------------ publishing
def test_growth_matches_direct_counts(client, data_dir):
    rows = {r["year"]: r for r in get(client, "/api/publishing/growth")}
    assert min(rows) == 1970 and max(rows) == 2025          # partial year excluded by default
    expected = dict(src(data_dir, """
        SELECT year, count(*) FROM src WHERE type = 'inproceedings'
          AND NOT (coalesce(journal, '') = 'CoRR' OR coalesce(publtype, '') LIKE 'informal%') GROUP BY year"""))
    for y, r in rows.items():
        assert r["conference"] == expected.get(y, 0)
    corr = dict(src(data_dir, "SELECT year, count(*) FROM src WHERE journal = 'CoRR' GROUP BY year"))
    informal = dict(src(data_dir, "SELECT year, count(*) FROM src WHERE publtype = 'informal' GROUP BY year"))
    assert rows[2015]["preprint"] == corr.get(2015, 0) + informal.get(2015, 0)


def test_growth_filters(client):
    rows = get(client, "/api/publishing/growth", **{"from": 2000, "to": 2005})
    assert [r["year"] for r in rows] == list(range(2000, 2006))
    assert client.get("/api/publishing/growth", params={"from": 2010, "to": 2000}).status_code == 400


def test_teams_and_metadata(client):
    teams = get(client, "/api/publishing/teams")
    assert all(r["mean_authors"] >= 1 for r in teams)
    meta = get(client, "/api/publishing/metadata")
    assert meta[0]["year"] == 2000
    assert any(r["with_unidentified_author"] > 0 for r in meta)
    assert any(r["with_title_twin"] > 0 for r in meta)


def test_growth_rates(client):
    body = get(client, "/api/publishing/growth-rates")
    kinds = {(r["kind"], r["period"]) for r in body["rows"]}
    assert ("all", "1990-2005") in kinds and ("preprint", "2006-2023") in kinds
    assert all(r["lo"] <= r["rate"] <= r["hi"] for r in body["rows"])
    assert client.get("/api/publishing/growth-rates", params={"split": 1980}).status_code == 400


# ------------------------------------------------------------------ identity
def test_homonyms_and_affiliation(client):
    hom = get(client, "/api/identity/homonyms", top=3)
    assert hom[0] == {"base_name": "Wei Wang", "distinct_people": 30}
    aff = get(client, "/api/identity/affiliation")
    assert aff["numbered"]["affiliation"] == 1.0
    assert aff["regular"]["affiliation"] < 0.1
    assert aff["disambiguation"]["pages"] == 3


def test_careers(client):
    assert get(client, "/api/identity/newcomers")[0]["year"] >= 1970
    coh = get(client, "/api/identity/cohorts", **{"from": 1970, "to": 2020, "step": 1})
    assert coh and all(1970 <= r["cohort"] <= 2020 for r in coh)
    assert all(r["pct_20y"] is None for r in coh if r["cohort"] > 2005)   # horizon past the data end
    pos = get(client, "/api/identity/positions")
    assert all(abs(r["pct_first"] + r["pct_middle"] + r["pct_last"] - 100) < 0.5 for r in pos)
    unid = get(client, "/api/identity/unidentified-by-position")
    assert [r["author_position"] for r in unid] == ["first", "middle", "last"]
    alpha = get(client, "/api/identity/alphabetical", min_papers=50)
    assert alpha["most"] and alpha["by_decade"]


# ------------------------------------------------------------------ job outputs
def test_network_reads_job_output(client):
    net = get(client, "/api/network")
    assert net["available"]
    assert net["distance_summary"] == {"mean": 5.63, "median": 6, "p90": 7}
    assert net["communities"]["communities"] == 2202
    assert net["structure"]["max k-core"] == 49
    assert net["largest_communities"][0]["authors"] == 552642
    assert net["largest_communities"][0]["venues"][0] == {"name": "IEEE Access", "papers": 44197}
    assert [g["up_to"] for g in net["growth"]][-1] == 2025
    assert len(net["distances"]) >= 10


def test_openalex_reads_job_output(client):
    oa = get(client, "/api/enrichment/openalex")
    assert oa["match"]["preprint"]["found"] == 80.4
    assert oa["fields"][0] == {"field": "Computer Science", "works": 995, "pct": 50.8}
    assert any(a["kind"] == "journal" and a["period"] == "2013-2025" for a in oa["adds"])


# ------------------------------------------------------------------ titles
def test_terms_presets_and_custom(client):
    body = get(client, "/api/titles/terms", terms="llm,blockchain,robust systems")
    assert [s["term"] for s in body["series"]] == ["llm", "blockchain", "robust systems"]
    llm = dict(zip(body["years"], body["series"][0]["values"]))
    assert llm[2015] == 0 and max(llm.values()) > 0            # llm only appears from 2022 in the fixture
    assert client.get("/api/titles/terms", params={"terms": "a'); DROP TABLE pubs; --"}).status_code == 400
    assert client.get("/api/titles/terms", params={"terms": ""}).status_code == 400


def test_title_style_and_words(client):
    style = get(client, "/api/titles/style")
    assert style[-1]["decade"] == 2020 and style[-1]["colon"] > 0
    rising = get(client, "/api/titles/words", min_titles=5)
    assert rising and rising[0]["change_x"] >= rising[-1]["change_x"]
    falling = get(client, "/api/titles/words", min_titles=5, direction="falling")
    assert falling[0]["change_x"] <= falling[-1]["change_x"]
    assert client.get("/api/titles/words", params={"old": "2011"}).status_code == 400


# ------------------------------------------------------------------ venues
def test_venue_aggregates(client):
    life = get(client, "/api/venues/lifespans")
    assert {r["kind"] for r in life} == {"conference", "journal"}
    conc = get(client, "/api/venues/concentration")
    assert conc[0]["year"] == 1980 and all(r["year"] % 5 == 0 for r in conc)
    pubs = get(client, "/api/venues/publishers")
    names = [r["publisher"] for r in pubs["rows"]]
    assert "IEEE" in names and "(no DOI)" in names and "other DOI prefix" not in names
    gaps = get(client, "/api/venues/doi-gaps", min_papers=10)
    assert {g["usual_name"] for g in gaps} >= {"NeurIPS", "ICLR", "J. Mach. Learn. Res."}


def test_venue_search_and_detail(client):
    hits = get(client, "/api/venues/search", q="neur")
    assert hits[0]["sid"] == "conf/nips"
    detail = get(client, "/api/venues/detail", sid="conf/nips")
    assert {n["name"] for n in detail["names"]} == {"NeurIPS", "NIPS"}   # one series, two name strings
    assert detail["yearly"] and detail["top_authors"]
    assert client.get("/api/venues/detail", params={"sid": "conf/nope"}).status_code == 404


# ------------------------------------------------------------------ quality
def test_quality(client):
    cov = get(client, "/api/quality/coverage")
    assert cov["types"][:3] == ["article", "inproceedings", "www"]
    school = cov["matrix"][cov["fields"].index("school")]
    assert school[cov["types"].index("phdthesis")] == 1.0 and school[cov["types"].index("article")] == 0.0
    formats = {r["page_format"] for r in get(client, "/api/quality/page-formats")}
    assert formats == {"No pages", "Start-end  (482-494)", "Single number  (604)", "Article:page  (41:1-41:5)",
                       "Starts with a letter  (xiv)", "Other  (186-)"}
    th = get(client, "/api/quality/theses")
    assert sum(r["theses"] for r in th["countries"]) == 150
    rd = get(client, "/api/quality/research-data")
    assert rd["by_year"][0]["year"] == 2015


# ------------------------------------------------------------------ tails
@pytest.mark.parametrize("panel", ["papers_per_author", "coauthors_per_author", "papers_per_series"])
def test_tails(client, panel):
    body = get(client, "/api/tails", panel=panel)
    pts = body["points"]
    assert pts[0]["share"] == pytest.approx(1.0)
    assert all(a["share"] >= b["share"] for a, b in zip(pts, pts[1:]))
    s = body["stats"]
    assert s["n"] == sum(p["n"] for p in pts) and 0 <= s["gini"] < 1
    assert body["fit"]["xmin"] > 0                               # from 07_statistics.txt


def test_tails_rejects_unknown_panel(client):
    assert client.get("/api/tails", params={"panel": "nope"}).status_code == 400


# ------------------------------------------------------------------ explore
def test_author_search_and_detail(client):
    hits = get(client, "/api/authors/search", q="Wei Wang")
    assert hits[0]["name"] in ("Wei Wang",) or hits[0]["name"].startswith("Wei Wang")
    numbered = next(h for h in hits if h["page_kind"] == "numbered")
    assert numbered["namesakes"] == 31                            # 30 numbered + the bin
    assert numbered["affiliation"].startswith("Institute")
    d = get(client, "/api/authors/detail", key=numbered["key"])
    assert d["person"]["page_kind"] == "numbered"
    assert d["namesake_count"] == 31 and len(d["namesakes"]) <= 12
    assert d["stats"]["papers"] == sum(y["papers"] for y in d["yearly"])
    assert client.get("/api/authors/search", params={"q": "W"}).status_code == 400
    assert client.get("/api/authors/detail", params={"key": "homepages/none"}).status_code == 404


def test_bin_detail(client):
    d = get(client, "/api/authors/detail", key="homepages/bin/WeiWang")
    assert d["person"]["page_kind"] == "disambiguation"
    assert d["stats"]["papers"] > 0


def test_variant_name_resolves(client, data_dir):
    key, variant = src(data_dir, """
        SELECT key, authors[2] FROM src WHERE type = 'www' AND n_authors = 2 LIMIT 1""")[0]
    hits = get(client, "/api/authors/search", q=variant)
    assert key in {h["key"] for h in hits}


def test_paper_search_and_detail(client):
    hits = get(client, "/api/papers/search", q="blockchain", **{"from": 2015})
    assert hits and all("blockchain" in h["title"].lower() and h["year"] >= 2015 for h in hits)
    d = get(client, "/api/papers/detail", key=hits[0]["key"])
    assert d["record"]["key"] == hits[0]["key"]
    kinds = {a["page_kind"] for a in d["authors"]}
    assert kinds <= {"regular", "numbered", "disambiguation", "unresolved"}
    withdrawn = get(client, "/api/papers/detail", key="journals/aada/W0")
    assert withdrawn["authors"] == [] and withdrawn["record"]["publtype"] == "withdrawn"
    assert client.get("/api/papers/search", params={"q": "ab"}).status_code == 400


def test_twins_are_linked(client, serving):
    cur = serving.cursor()
    key = cur.execute("SELECT key FROM pubs WHERE has_twin AND NOT is_preprint LIMIT 1").fetchone()[0]
    d = get(client, "/api/papers/detail", key=key)
    assert d["paper"]["has_twin"]
    assert d["twins"] and all(t["key"] != key for t in d["twins"])
    assert any(t["kind"] == "preprint" for t in d["twins"])


def test_overview(client):
    body = get(client, "/api/overview")
    assert len(body["kpis"]) == 8
    assert body["counts"]["bins"] == 3
    assert any("Wei Wang" in k["l"] for k in body["kpis"])


# ------------------------------------------------------------------ lifecycle
def test_rebuilds_when_parquet_changes(serving, client, data_dir):
    before = serving.generation
    p = data_dir / "parquet" / "dblp.parquet"
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 10))
    serving.ensure()
    assert serving.generation == before + 1
    assert client.get("/api/publishing/growth").status_code == 200


def test_validate_skips_missing_csvs(serving, capsys):
    from app import validate
    assert validate.main() == 0
    assert "CSV not found" in capsys.readouterr().out
