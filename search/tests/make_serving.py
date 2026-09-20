"""
A synthetic serving database with topic-distinctive title vocabularies, so search tests can check
that a query about a topic ranks that topic's papers highly - not just that the code runs. Carries
only the `s.pubs` columns search actually reads; `s._meta` for the dump fingerprint.

Includes a few preprints (excluded from the index by kind) and pre-2010 papers (excluded by year),
so tests can exercise the exact-word fallback for record kinds and years the hybrid index skips.
"""
import random
from pathlib import Path

import duckdb

TOPICS = {
    "conf/aaa": ("conf", "AAA", "graph clustering community spectral partition vertex".split()),
    "conf/bbb": ("conf", "BBB", "reinforcement policy reward agent bandit exploration".split()),
    "journals/ccc": ("journals", "CCC", "protein genome sequence clinical patient diagnosis".split()),
}
FILLER = "efficient robust scalable novel adaptive distributed secure private".split()
# Preprints and pre-2010 papers deliberately share no vocabulary with any indexed topic, so a query
# built from one of their titles cannot be confused with an indexed (BM25/dense-reachable) paper -
# the only way to find it is the exact-word fallback, which is exactly what that population tests.
OFFSCOPE_VOCAB = "quantum blockchain metaverse holographic photonic".split()
YEARS = range(2010, 2026)
PAPERS_PER_TOPIC_PER_YEAR = 4
EXTRA_PER_TOPIC = 6   # preprints, and separately pre-2010 papers


def _title(rnd, vocab, k=3):
    words = rnd.sample(vocab, min(k, len(vocab))) + rnd.sample(FILLER, 2)
    rnd.shuffle(words)
    return " ".join(words).capitalize() + "."


def _offscope_title(rnd):
    """No filler words: zero token overlap with any indexed title, so a query built from this
    title cannot pick up a stray hybrid match - only the exact-word fallback can find it."""
    words = rnd.sample(OFFSCOPE_VOCAB, 3)
    rnd.shuffle(words)
    return " ".join(words).capitalize() + "."


def make(cache_dir: Path, fingerprint="testfp0001", seed=13):
    rnd = random.Random(seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"serve-{fingerprint}.duckdb"
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))

    pubs, pid = [], 0
    for sid, (prefix, venue, vocab) in TOPICS.items():
        kind_type = "article" if prefix == "journals" else "inproceedings"
        for year in YEARS:
            for _ in range(PAPERS_PER_TOPIC_PER_YEAR):
                pid += 1
                pubs.append(dict(pid=pid, key=f"{sid}/p{pid}", type=kind_type, year=year, title=_title(rnd, vocab),
                                 n_authors=rnd.randint(1, 5), venue=venue, key_prefix=prefix, sid=sid,
                                 is_preprint=False, n_unidentified=0, has_twin=False, has_oa=rnd.random() < 0.3))
        for _ in range(EXTRA_PER_TOPIC):
            pid += 1
            pubs.append(dict(pid=pid, key=f"corr/p{pid}", type="article", year=2024,
                             title=_offscope_title(rnd), n_authors=2, venue="CoRR", key_prefix="corr",
                             sid=None, is_preprint=True, n_unidentified=0, has_twin=False, has_oa=True))
        for _ in range(EXTRA_PER_TOPIC):
            pid += 1
            pubs.append(dict(pid=pid, key=f"{sid}/old{pid}", type=kind_type, year=rnd.randint(1995, 2009),
                             title=_offscope_title(rnd), n_authors=2, venue=venue, key_prefix=prefix,
                             sid=sid, is_preprint=False, n_unidentified=0, has_twin=False, has_oa=False))

    con.execute("""CREATE TABLE pubs (pid INTEGER, key VARCHAR, type VARCHAR, year SMALLINT, title VARCHAR,
                                      n_authors INTEGER, venue VARCHAR, key_prefix VARCHAR, sid VARCHAR,
                                      is_preprint BOOLEAN, n_unidentified INTEGER, has_twin BOOLEAN, has_oa BOOLEAN)""")
    con.executemany("INSERT INTO pubs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(p["pid"], p["key"], p["type"], p["year"], p["title"], p["n_authors"], p["venue"],
                      p["key_prefix"], p["sid"], p["is_preprint"], p["n_unidentified"], p["has_twin"], p["has_oa"])
                     for p in pubs])
    con.execute("CREATE TABLE _meta (k VARCHAR, v VARCHAR)")
    con.executemany("INSERT INTO _meta VALUES (?, ?)", [
        ("fingerprint", fingerprint), ("records", str(len(pubs))), ("latest_mdate", "2026-09-01"),
        ("parquet", "synthetic"), ("last_full_year", "2025"),
    ])
    con.execute("CHECKPOINT")
    con.close()
    return path, {"papers": len(pubs), "topics": list(TOPICS)}


if __name__ == "__main__":
    import sys
    p, stats = make(Path(sys.argv[1] if len(sys.argv) > 1 else "cache"))
    print(f"wrote {p}: {stats}")
