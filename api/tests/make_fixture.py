"""
Write a small synthetic dblp dump with the exact schema of parse_dblp.py's parquet, covering the cases
the queries care about: disambiguation bins, numbered namesakes, name variants, unresolved names,
withdrawn papers without authors, preprint/published twins, CoRR, proceedings, theses, data records,
every page format. Usage: python -m tests.make_fixture <out_dir>
"""
import json
import random
import shutil
import sys
from pathlib import Path

import duckdb

SCHEMA = {
    "key": "VARCHAR", "type": "VARCHAR", "mdate": "VARCHAR", "publtype": "VARCHAR", "title": "VARCHAR",
    "authors": "VARCHAR[]", "author_orcids": "VARCHAR[]", "n_authors": "INTEGER", "n_orcids": "INTEGER",
    "editors": "VARCHAR[]", "n_editors": "INTEGER", "year": "INTEGER", "journal": "VARCHAR",
    "booktitle": "VARCHAR", "volume": "VARCHAR", "number": "VARCHAR", "pages": "VARCHAR", "crossref": "VARCHAR",
    "publisher": "VARCHAR", "school": "VARCHAR", "series": "VARCHAR", "isbn": "VARCHAR[]", "ee": "VARCHAR[]",
    "has_oa": "BOOLEAN", "url": "VARCHAR", "urls": "VARCHAR[]", "notes": "VARCHAR[]", "n_cites": "INTEGER",
}

FIRST = ["Anna", "Ben", "Chen", "Dana", "Eli", "Fatima", "Goran", "Hana", "Ivan", "Jun", "Kofi", "Lena",
         "Mehdi", "Nora", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tariq", "Uma", "Viktor", "Wen", "Yara"]
LAST = ["Abe", "Bauer", "Costa", "Diaz", "Eriksen", "Fischer", "Garcia", "Haddad", "Ito", "Jensen", "Khan",
        "Lopez", "Moreau", "Novak", "Okafor", "Petrov", "Quist", "Rossi", "Silva", "Tanaka", "Ueda", "Vogel"]
WORDS = ["learning", "efficient", "networks", "analysis", "robust", "systems", "detection", "model", "data",
         "optimization", "distributed", "secure", "adaptive", "framework", "towards", "scalable", "method"]
TERMS_BY_ERA = [(1970, ["neural", "cloud"]), (2008, ["cloud", "deep", "iot"]), (2015, ["deep", "blockchain", "neural"]),
                (2018, ["transformer", "deep", "quantum"]), (2022, ["llm", "large language model", "transformer"])]
JOURNALS = [("tit", "IEEE Trans. Inf. Theory", "10.1109"), ("access", "IEEE Access", "10.1109"),
            ("tcs", "Theor. Comput. Sci.", "10.1016"), ("jmlr", "J. Mach. Learn. Res.", None),
            ("sensors", "Sensors", "10.3390")]
CONFS = [("nips", "NeurIPS", None), ("icse", "ICSE", "10.1145"), ("iclr", "ICLR", None),
         ("icassp", "ICASSP", "10.1109"), ("stoc", "STOC", "10.1145")]
COUNTRIES = ["Germany", "USA", "France", "UK", "Brazil"]


def row(**kw):
    base = {k: None for k in SCHEMA}
    for k in ("authors", "author_orcids", "editors", "isbn", "ee", "urls", "notes"):
        base[k] = []
    base.update(n_authors=0, n_orcids=0, n_editors=0, has_oa=False, n_cites=0)
    base.update(kw)
    base["n_authors"] = len(base["authors"])
    base["author_orcids"] = kw.get("author_orcids") or [None] * len(base["authors"])
    base["n_orcids"] = sum(o is not None for o in base["author_orcids"])
    base["n_editors"] = len(base["editors"])
    if base["urls"] and not base["url"]:
        base["url"] = base["urls"][0]
    return base


def make(out: Path, seed=7):
    rnd = random.Random(seed)
    recs = []

    # ---- author pages
    people = []  # (key, [names])
    for i in range(300):
        name = f"{rnd.choice(FIRST)} {rnd.choice(LAST)}"
        if any(name in n for _, n in people):
            name += f" {rnd.choice(LAST)}"
        names = [name] + ([name.split()[0][0] + ". " + name.split()[-1]] if i % 15 == 0 else [])
        key = f"homepages/{i % 100:02d}/{i}"
        urls = [f"https://orcid.org/0000-0000-0000-{i:04d}"] if i % 25 == 0 else []
        notes = ["affiliation: University of Somewhere"] if i % 50 == 0 else []
        recs.append(row(key=key, type="www", mdate="2024-01-0" + str(1 + i % 9), title="Home Page",
                        authors=names, urls=urls, notes=notes))
        people.append((key, names))
    for base, n in [("Wei Wang", 30), ("Yang Liu", 20), ("Wei Zhang", 12)]:
        for j in range(1, n + 1):
            key = f"homepages/{base[:1]}{j}/{len(base) * 37}-{j}"
            nm = f"{base} {j:04d}"
            urls = [f"https://orcid.org/0000-0001-0000-{j:04d}"] if j % 2 else []
            recs.append(row(key=key, type="www", mdate="2023-05-05", title="Home Page", authors=[nm], urls=urls,
                            notes=[f"affiliation: Institute {j}"]))
            people.append((key, [nm]))
        recs.append(row(key=f"homepages/bin/{base.replace(' ', '')}", type="www", publtype="disambiguation",
                        mdate="2017-06-27", title="Home Page", authors=[base]))
    # www records that are not author pages (excluded by key LIKE 'homepages/%')
    recs.append(row(key="persons/Ley2003", type="www", mdate="2023-12-19", authors=["Michael Ley"],
                    title="ACM SIGMOD Contribution Award 2003 Acceptance Speech", year=2003))

    bins = ["Wei Wang", "Yang Liu", "Wei Zhang"]

    def pick_authors(k):
        out = []
        for _ in range(k):
            r = rnd.random()
            if r < 0.08:
                out.append(rnd.choice(bins))                      # lands on a disambiguation bin
            elif r < 0.11:
                out.append(f"Unknown Person{rnd.randint(1, 99)}")  # resolves to no author page
            else:
                _, names = rnd.choice(people)
                out.append(rnd.choice(names))                    # sometimes a name variant
        return list(dict.fromkeys(out))

    def title_for(year):
        terms = [t for y, ts in TERMS_BY_ERA if year >= y for t in ts]
        words = rnd.sample(WORDS, 4)
        if terms and rnd.random() < 0.5:
            words.insert(rnd.randint(0, 3), rnd.choice(terms))
        t = " ".join(words).capitalize()
        if rnd.random() < 0.2:
            t = "ACRONYM: " + t
        if rnd.random() < 0.05:
            t += "?"
        return t + "."

    page_formats = [lambda: f"{rnd.randint(1, 900)}-{rnd.randint(901, 999)}", lambda: str(rnd.randint(1, 999)),
                    lambda: f"{rnd.randint(1, 50)}:1-{rnd.randint(1, 50)}:9", lambda: "xiv", lambda: "186-",
                    lambda: None]

    published_titles = []
    n = 0
    for year in range(1970, 2027):
        per_year = 20 + (year - 1970) * 4
        for _ in range(per_year):
            n += 1
            is_conf = rnd.random() < 0.5
            sid, name, prefix = rnd.choice(CONFS if is_conf else JOURNALS)
            k = max(1, min(60, int(rnd.expovariate(1 / (1.5 + (year - 1970) / 20))) + 1))
            authors = pick_authors(k)
            orcids = [f"0000-0002-{rnd.randint(1000, 9999)}-000X" if year > 2012 and rnd.random() < 0.4 else None
                      for _ in authors]
            ee = []
            if prefix and rnd.random() < 0.9:
                ee.append(f"https://doi.org/{prefix}/x{n}")
            elif rnd.random() < 0.3:
                ee.append(f"https://openreview.net/forum?id={n}")
            title = title_for(year)
            if rnd.random() < 0.02:
                title = "Editorial."
            common = dict(mdate=f"{min(year + 1, 2026)}-0{1 + n % 8}-1{n % 9}", title=title, authors=authors,
                          author_orcids=orcids, year=year, ee=ee, has_oa=rnd.random() < (0.1 + (year - 1970) / 120),
                          pages=rnd.choice(page_formats)())
            if is_conf:
                recs.append(row(key=f"conf/{sid}/{sid.upper()}{year}-{n}", type="inproceedings", booktitle=name,
                                crossref=f"conf/{sid}/{year}", **common))
            else:
                recs.append(row(key=f"journals/{sid}/{sid}{year}-{n}", type="article", journal=name,
                                volume=str(year - 1960), **common))
            if len(title) >= 30 and year >= 2012:
                published_titles.append((title, year))
    # a venue whose name string changed once (series id stays the same)
    for y in (2019, 2020):
        recs.append(row(key=f"conf/nips/NIPS{y}-old", type="inproceedings", booktitle="NIPS", year=y,
                        title=title_for(y), authors=pick_authors(3), mdate="2020-01-01"))

    # preprints: CoRR (journal = 'CoRR') and informal; some share a title with a published paper
    for i in range(900):
        year = rnd.randint(2008, 2026)
        if published_titles and rnd.random() < 0.35:
            title, y0 = rnd.choice(published_titles)
            year = max(2008, y0 - 1)
        else:
            title = title_for(year)
        recs.append(row(key=f"journals/corr/abs-{year}-{i:05d}", type="article", journal="CoRR", year=year,
                        title=title, authors=pick_authors(rnd.randint(1, 8)), mdate="2025-02-02",
                        ee=[f"https://arxiv.org/abs/{year % 100:02d}01.{i:05d}"], has_oa=True))
    for i in range(40):
        recs.append(row(key=f"journals/tr/TR{i}", type="article", publtype="informal", journal="Tech. Rep.",
                        year=1975 + i, title=title_for(1975 + i), authors=pick_authors(2), mdate="2019-01-01"))

    # withdrawn: no authors
    for i in range(5):
        recs.append(row(key=f"journals/aada/W{i}", type="article", publtype="withdrawn", journal="Adv. Data Sci.",
                        year=2022, title="Social-Interactive Sports Monitor for Children.", mdate="2022-12-15",
                        ee=["https://doi.org/10.1142/S2424922X2142003"]))

    # proceedings volumes, theses, books, chapters, data, a master's thesis
    for sid, name, _ in CONFS:
        for y in range(2000, 2026, 5):
            recs.append(row(key=f"conf/{sid}/{y}", type="proceedings", title=f"Proceedings of {name} {y}",
                            editors=[rnd.choice(FIRST) + " Editor"], year=y, publisher="Springer",
                            series="LNCS", isbn=["978-3-16-148410-0"], mdate="2021-01-01"))
    for i in range(150):
        y = rnd.randint(1990, 2025)
        _, names = rnd.choice(people)
        recs.append(row(key=f"phd/{i}", type="phdthesis", title=title_for(y), authors=[names[0]], year=y,
                        school=f"Some University, {rnd.choice(COUNTRIES)}", mdate="2024-08-22",
                        ee=[rnd.choice(["https://d-nb.info/", "https://hal.science/", "https://ethos.bl.uk/"]) + str(i)]))
    for i in range(30):
        y = rnd.randint(1980, 2025)
        recs.append(row(key=f"books/b/{i}", type="book", title=title_for(y), authors=pick_authors(2), year=y,
                        publisher=rnd.choice(["Springer", "MIT Press"]), isbn=["978-0-262-00000-0"], mdate="2020-01-01"))
        recs.append(row(key=f"reference/r/{i}", type="incollection", title=title_for(y), authors=pick_authors(1),
                        year=y, booktitle="Encyclopedia of Things", publtype="encyclopedia" if i % 2 else None,
                        mdate="2020-01-01"))
    for i in range(120):
        y = rnd.randint(2015, 2026)
        recs.append(row(key=f"data/10/{i}", type="data", publtype=rnd.choice(["version", "concept", None]),
                        title=f"Dataset {i}", authors=pick_authors(2), year=y, publisher="Zenodo",
                        ee=[f"https://doi.org/10.5281/zenodo.{i}"], mdate="2026-08-31",
                        author_orcids=None))
    recs.append(row(key="phd/de/Schmidhuber09", type="mastersthesis", year=1987, mdate="2024-08-22",
                    title="Evolutionary principles in self-referential learning", authors=[people[0][1][0]],
                    school="Technical University of Munich, Germany"))

    out.mkdir(parents=True, exist_ok=True)
    (out / "parquet").mkdir(exist_ok=True)
    jsonl = out / "fixture.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    cols = "{" + ", ".join(f"'{k}': '{v}'" for k, v in SCHEMA.items()) + "}"
    con = duckdb.connect()
    con.execute(f"""
        COPY (SELECT * FROM read_json('{jsonl.as_posix()}', format = 'newline_delimited', columns = {cols}))
        TO '{(out / 'parquet' / 'dblp.parquet').as_posix()}' (FORMAT parquet)""")
    jsonl.unlink()
    eda = out / "eda_out"
    eda.mkdir(exist_ok=True)
    for f in (Path(__file__).parent / "data").glob("*.txt"):
        shutil.copy(f, eda / f.name)
    (out / "charts").mkdir(exist_ok=True)
    return len(recs)


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "fixture")
    print(f"wrote {make(target)} records to {target}")
