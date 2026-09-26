"""
A tiny synthetic serving database with *known* answers.

Unlike the ML fixtures, nothing here is random: every count the tests assert is written down below,
so a test failure means the SQL changed meaning, not that a seed moved. The tables carry the names
and columns the api's serving database exposes, and the derived tables (person_stats, career,
series, word_year, person_degree) are built with the api's own definitions, so tools run unchanged.

The world:
  Ada Alpha    10 records (the top author), conf/aaa + journals/bbb, 2 papers with Ben
  Ben Beta      6, conf/aaa + journals/bbb          -> so Ada and Ben are the two "in both venues"
  Cleo Gamma    8, conf/ccc plus the co-authored bin records
  Dan Delta     6, conf/ccc
  Eve Epsilon   2, journals/bbb  (one of them a preprint)
  Fay Zeta      2, conf/aaa
  Sam Same      a disambiguation bin with 4 records, plus Sam Same 0001 (3) and Sam Same 0002 (2)
Titles carry "cloud" in the early years and "graph" in the later ones, so a term trend exists.
"""
from pathlib import Path

import duckdb

VENUES = {"conf/aaa": ("AAA Conference", "conf", "inproceedings"),
          "journals/bbb": ("BBB Journal", "journals", "article"),
          "conf/ccc": ("CCC Symposium", "conf", "inproceedings")}

PEOPLE = [
    # (key, name, base_name, page_kind, affiliation)
    ("homepages/a/Ada", "Ada Alpha", "Ada Alpha", "regular", "University of Alpha"),
    ("homepages/b/Ben", "Ben Beta", "Ben Beta", "regular", None),
    ("homepages/c/Cleo", "Cleo Gamma", "Cleo Gamma", "regular", None),
    ("homepages/d/Dan", "Dan Delta", "Dan Delta", "regular", None),
    ("homepages/e/Eve", "Eve Epsilon", "Eve Epsilon", "regular", None),
    ("homepages/f/Fay", "Fay Zeta", "Fay Zeta", "regular", None),
    ("homepages/s/bin", "Sam Same", "Sam Same", "disambiguation", None),
    ("homepages/s/1", "Sam Same 0001", "Sam Same", "numbered", "Institute One"),
    ("homepages/s/2", "Sam Same 0002", "Sam Same", "numbered", "Institute Two"),
]

# (sid, year, authors by name, preprint?, open access?, doi?, title)
PAPERS = [
    ("conf/aaa", 2010, ["Ada Alpha", "Ben Beta"], False, False, True, "Cloud systems for analysis"),
    ("conf/aaa", 2011, ["Ada Alpha", "Ben Beta"], False, False, True, "Cloud scheduling revisited"),
    ("conf/aaa", 2012, ["Ada Alpha", "Fay Zeta"], False, False, False, "Cloud storage layers"),
    ("journals/bbb", 2013, ["Ada Alpha"], False, True, True, "Efficient cloud indexing"),
    ("journals/bbb", 2015, ["Ada Alpha", "Cleo Gamma"], False, True, True, "Robust data pipelines"),
    ("conf/aaa", 2018, ["Ada Alpha", "Ben Beta", "Dan Delta"], False, False, True, "Graph learning at scale"),
    ("journals/bbb", 2020, ["Ada Alpha"], False, True, True, "Graph neural networks for traffic"),
    ("journals/bbb", 2022, ["Ada Alpha", "Eve Epsilon"], False, True, True, "Graph embeddings for retrieval"),
    ("conf/aaa", 2024, ["Ada Alpha"], False, False, True, "Graph transformers in practice"),
    ("conf/aaa", 2024, ["Ada Alpha", "Ben Beta"], True, False, False, "Graph pretraining preprint"),
    ("journals/bbb", 2019, ["Ben Beta"], False, True, True, "Distributed graph queries"),
    ("conf/aaa", 2021, ["Ben Beta", "Fay Zeta"], False, False, True, "Graph partitioning methods"),
    ("conf/ccc", 2016, ["Cleo Gamma", "Dan Delta"], False, False, False, "Secure protocols for sensors"),
    ("conf/ccc", 2017, ["Cleo Gamma", "Dan Delta"], False, False, False, "Secure key exchange"),
    ("conf/ccc", 2023, ["Cleo Gamma"], False, True, True, "Graph models of trust"),
    ("journals/bbb", 2023, ["Eve Epsilon"], True, False, False, "Adaptive graph sampling preprint"),
    ("conf/ccc", 2025, ["Dan Delta"], False, True, True, "Graph compression for logs"),
    # the bin's records, and its two numbered namesakes
    ("conf/aaa", 2019, ["Sam Same", "Cleo Gamma"], False, False, True, "Graph kernels for chemistry"),
    ("conf/aaa", 2020, ["Sam Same", "Cleo Gamma"], False, False, True, "Graph kernels revisited"),
    ("conf/ccc", 2021, ["Sam Same", "Dan Delta"], False, False, False, "Secure graph release"),
    ("conf/ccc", 2022, ["Sam Same"], False, False, False, "Secure aggregation notes"),
    ("conf/aaa", 2017, ["Sam Same 0001", "Cleo Gamma"], False, False, True, "Graph kernels early work"),
    ("conf/aaa", 2018, ["Sam Same 0001", "Cleo Gamma"], False, False, True, "Graph kernels follow-up"),
    ("journals/bbb", 2019, ["Sam Same 0001"], False, True, True, "Graph kernels journal version"),
    ("conf/ccc", 2018, ["Sam Same 0002", "Dan Delta"], False, False, False, "Secure channels for graphs"),
    ("conf/ccc", 2020, ["Sam Same 0002"], False, False, False, "Secure release of graphs"),
]

EXPECTED = {           # what the tests assert against
    "top_author": ("Ada Alpha", 10),
    "papers_total": len(PAPERS),
    "ada_and_ben_together": 4,          # 2010, 2011, 2018, and the 2024 preprint
    "ada_coauthors": 5,                 # Ben, Fay, Cleo, Dan, Eve
    "in_both_aaa_and_bbb": {"Ada Alpha", "Ben Beta"},
    "sam_numbered": 2,
    "sam_bin_records": 4,
}

STOP_WORDS = ("the and for with from into over under via using based toward towards its their this that "
              "these those are can not than versus vs our your who how what when why which where").split()


def make(out_dir, fingerprint="testfp000001"):
    path = Path(out_dir) / f"serve-{fingerprint}.duckdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))

    con.execute("""CREATE TABLE persons (person_id INTEGER, key VARCHAR, publtype VARCHAR, names VARCHAR[],
                   name VARCHAR, n_names INTEGER, urls VARCHAR[], notes VARCHAR[], page_kind VARCHAR,
                   base_name VARCHAR, has_affiliation BOOLEAN, has_orcid_link BOOLEAN,
                   has_wikidata_link BOOLEAN)""")
    con.execute("CREATE TABLE person_names (person_id INTEGER, name VARCHAR)")
    con.execute("""CREATE TABLE pubs (pid INTEGER, key VARCHAR, type VARCHAR, publtype VARCHAR, year INTEGER,
                   title VARCHAR, n_authors INTEGER, n_orcids INTEGER, has_oa BOOLEAN, pages VARCHAR,
                   journal VARCHAR, booktitle VARCHAR, school VARCHAR, publisher VARCHAR, mdate VARCHAR,
                   is_preprint BOOLEAN, title_norm VARCHAR, key_prefix VARCHAR, sid VARCHAR, venue VARCHAR,
                   doi_prefix VARCHAR, has_doi BOOLEAN, link_host VARCHAR, is_withdrawn BOOLEAN,
                   alphabetical BOOLEAN, distinct_surnames BOOLEAN, n_unidentified INTEGER,
                   has_twin BOOLEAN)""")
    con.execute("""CREATE TABLE slots (pid INTEGER, year INTEGER, n_authors SMALLINT, position SMALLINT,
                   type VARCHAR, is_preprint BOOLEAN, has_orcid BOOLEAN, person_id INTEGER,
                   on_bin BOOLEAN)""")
    con.execute("CREATE TABLE src (key VARCHAR, type VARCHAR, title VARCHAR, authors VARCHAR[], "
                "author_orcids VARCHAR[], n_authors INTEGER, year INTEGER, ee VARCHAR[])")

    persons, names = [], []
    by_name = {}
    for i, (key, name, base, kind, aff) in enumerate(PEOPLE, start=1):
        notes = [f"affiliation: {aff}"] if aff else []
        persons.append((i, key, "disambiguation" if kind == "disambiguation" else None, [name], name, 1,
                        [], notes, kind, base, bool(aff), False, False))
        names.append((i, name))
        by_name[name] = i
    con.executemany("INSERT INTO persons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", persons)
    con.executemany("INSERT INTO person_names VALUES (?, ?)", names)

    pubs, slots, src = [], [], []
    for pid, (sid, year, authors, preprint, oa, doi, title) in enumerate(PAPERS, start=1):
        venue_name, prefix, rec_type = VENUES[sid]
        key = f"{sid}/p{pid}"
        n = len(authors)
        on_bin_count = sum(1 for a in authors if by_name.get(a) and PEOPLE[by_name[a] - 1][3] == "disambiguation")
        pubs.append((pid, key, rec_type, "informal" if preprint else None, year, title, n, 0, oa, "1-10",
                     "CoRR" if preprint else (venue_name if prefix == "journals" else None),
                     None if prefix == "journals" else venue_name, None, None, "2026-08-15",
                     preprint, "".join(ch for ch in title.lower() if ch.isalnum()), prefix, sid,
                     "CoRR" if preprint else venue_name, "10.1109" if doi else "", doi, "doi.org",
                     False, True, True, on_bin_count, False))
        src.append((key, rec_type, title, authors, [None] * n, n, year,
                    [f"https://doi.org/10.1109/{pid}"] if doi else []))
        for position, author in enumerate(authors, start=1):
            person_id = by_name.get(author)
            on_bin = bool(person_id and PEOPLE[person_id - 1][3] == "disambiguation")
            slots.append((pid, year, n, position, rec_type, preprint, False, person_id, on_bin))
    con.executemany("INSERT INTO pubs VALUES (" + ", ".join(["?"] * 28) + ")", pubs)
    con.executemany("INSERT INTO slots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", slots)
    con.executemany("INSERT INTO src VALUES (?, ?, ?, ?, ?, ?, ?, ?)", src)

    # derived tables, with the api's definitions
    con.execute("""CREATE TABLE person_stats AS
                   SELECT person_id, count(*) AS n_pubs FROM slots WHERE person_id IS NOT NULL
                   GROUP BY person_id""")
    con.execute("""CREATE TABLE career AS
                   SELECT person_id, min(year) AS first_year, max(year) AS last_year, count(*) AS papers,
                          bool_or(has_orcid) AS any_orcid
                   FROM slots WHERE person_id IS NOT NULL AND NOT on_bin
                     AND type IN ('article', 'inproceedings') AND NOT is_preprint
                   GROUP BY person_id""")
    con.execute("""CREATE TABLE series AS
                   SELECT sid, CASE WHEN key_prefix = 'journals' THEN 'journal' ELSE 'conference' END AS kind,
                          mode(venue) AS usual_name, count(*) AS papers, min(year) AS first_year,
                          max(year) AS last_year, count(DISTINCT year) AS active_years,
                          count(DISTINCT venue) AS name_variants,
                          avg((doi_prefix <> '')::INT) AS doi_share, avg(has_oa::INT) AS oa_share
                   FROM pubs WHERE type IN ('article', 'inproceedings') AND NOT is_preprint
                     AND key_prefix IN ('conf', 'journals')
                   GROUP BY ALL""")
    stop = "['" + "','".join(STOP_WORDS) + "']"
    con.execute(f"""CREATE TABLE word_year AS
                   WITH t AS (SELECT year, list_distinct(regexp_split_to_array(lower(title), '[^a-z0-9]+')) AS words
                              FROM pubs WHERE type IN ('article', 'inproceedings') AND NOT is_preprint
                                AND title IS NOT NULL AND year >= 1970)
                   SELECT year, word, count(*)::INTEGER AS titles
                   FROM (SELECT year, unnest(words) AS word FROM t)
                   WHERE length(word) >= 3 AND NOT regexp_matches(word, '^[0-9]+$')
                     AND word NOT IN (SELECT unnest({stop}::VARCHAR[]))
                   GROUP BY ALL""")
    con.execute("""CREATE TABLE person_degree AS
                   WITH small AS (SELECT person_id, pid FROM slots
                                  WHERE person_id IS NOT NULL AND n_authors BETWEEN 2 AND 50)
                   SELECT a.person_id, count(DISTINCT b.person_id) AS n_coauthors
                   FROM small a JOIN small b ON a.pid = b.pid AND a.person_id <> b.person_id
                   GROUP BY a.person_id""")
    con.execute("CREATE TABLE _meta (k VARCHAR, v VARCHAR)")
    con.executemany("INSERT INTO _meta VALUES (?, ?)", [
        ("fingerprint", fingerprint), ("records", str(len(PAPERS) + len(PEOPLE))),
        ("latest_mdate", "2026-09-01"), ("parquet", "synthetic"), ("last_full_year", "2025"),
        ("built_at", "2026-09-01T00:00:00+00:00"),
    ])
    con.execute("CHECKPOINT")
    con.close()
    return path


if __name__ == "__main__":
    import sys
    print(make(sys.argv[1] if len(sys.argv) > 1 else "cache"))
