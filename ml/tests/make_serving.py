"""
A synthetic serving database with *planted* signal, so the tests can check that the model learns
something rather than merely runs.

Each person has a small set of regular collaborators, one or two preferred venues and a career
window, which is the structure real disambiguation relies on. The tables carry the same names and
columns the api's serving database exposes, so ml/ code runs against this unchanged.
"""
import random
from pathlib import Path

import duckdb

BLOCKS = 40            # name blocks ("Name 00" ... ), enough for the hash split to fill every bucket
PEOPLE_PER_BLOCK = 5
PAPERS_PER_PERSON = (5, 9)
VENUES = [("conf/aaa", "AAA", "conf"), ("conf/bbb", "BBB", "conf"), ("journals/ccc", "CCC", "journals"),
          ("journals/ddd", "DDD", "journals"), ("conf/eee", "EEE", "conf")]
WORDS = "learning graph neural robust efficient system network analysis model data adaptive secure".split()

# The link world: communities whose members keep collaborating, where new collaborations mostly
# close triangles (a co-author of a co-author) between people who are active now - the structure
# link prediction relies on. People join over the years, so at any snapshot there are newcomers
# with few links and veterans with many.
LINK_COMMUNITIES = 5
LINK_PEOPLE = 40
LINK_YEARS = range(2006, 2026)
LINK_PAPERS_PER_YEAR = 12

# Each venue has a vocabulary its titles draw from, next to the shared WORDS: the content signal
# venue recommendation relies on (authors' loyalty to their community's venues is the other one).
VENUE_WORDS = {
    "conf/aaa": "reinforcement agents planning policy reward bandit".split(),
    "conf/bbb": "graph vertex clustering spectral community embedding".split(),
    "journals/ccc": "protein genome sequence cell clinical patients".split(),
    "journals/ddd": "wireless antenna channel spectrum radio latency".split(),
    "conf/eee": "compiler kernel scheduling memory runtime concurrency".split(),
}


def _link_world(rnd, person_id, pid, persons, slots, pubs, src):
    people, name_of = {}, {}   # person_id -> (community, start, end) / name
    person_id = max(person_id, 4_000_000)   # real dblp ids run into the millions: arithmetic on them must not overflow INT32
    for c in range(LINK_COMMUNITIES):
        for i in range(LINK_PEOPLE):
            person_id += 1
            name = f"Link Person {c:02d}-{i:02d}"
            persons.append((person_id, f"homepages/link/{c}/{i}", name, name, "regular"))
            name_of[person_id] = name
            start = rnd.randint(LINK_YEARS[0], LINK_YEARS[-1] - 3)
            people[person_id] = (c, start, start + rnd.randint(8, 14))
    venues = {c: rnd.sample(VENUES, 2) for c in range(LINK_COMMUNITIES)}
    collab = {p: set() for p in people}

    def active(c, year):
        return [p for p, (cc, s, e) in people.items() if cc == c and s <= year <= e]

    for year in LINK_YEARS:
        for c in range(LINK_COMMUNITIES):
            members = active(c, year)
            if len(members) < 3:
                continue
            for _ in range(LINK_PAPERS_PER_YEAR):
                first = rnd.choice(members)
                team = [first]
                for _ in range(rnd.randint(1, 3)):
                    r = rnd.random()
                    pool = []
                    if r < 0.5:
                        pool = [p for p in collab[first] if p in members and p not in team]
                    elif r < 0.85:
                        pool = [q for p in collab[first] for q in collab[p]
                                if q in members and q not in team and q not in collab[first] and q != first]
                    if not pool:
                        pool = [p for p in members if p not in team]
                    if pool:
                        team.append(rnd.choice(pool))
                if rnd.random() < 0.03:
                    other = active((c + 1) % LINK_COMMUNITIES, year)
                    if other:
                        team.append(rnd.choice(other))
                pid += 1
                sid, venue, prefix = rnd.choice(venues[c])
                names = [name_of[p] for p in team]
                title_words = rnd.sample(VENUE_WORDS[sid], 3) + rnd.sample(WORDS, 2)
                rnd.shuffle(title_words)
                pubs.append((pid, f"{sid}/l{pid}", year, sid, venue, prefix, len(team),
                             " ".join(title_words).capitalize() + ".", False))
                src.append((f"{sid}/l{pid}", names, [None] * len(names)))
                for pos, p in enumerate(team, start=1):
                    slots.append((p, pid, pos))
                for p in team:
                    collab[p].update(q for q in team if q != p)
    return person_id, pid, len(people)


def make(cache_dir: Path, fingerprint="testfp0001", seed=11):
    rnd = random.Random(seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"serve-{fingerprint}.duckdb"
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))

    persons, slots, pubs, src = [], [], [], []
    person_id = pid = 0
    collaborator_pool = list(range(100000, 100600))   # ids that are co-authors but not in any block

    for b in range(BLOCKS):
        base = f"Name {b:02d}"
        for k in range(1, PEOPLE_PER_BLOCK + 1):
            person_id += 1
            me = person_id
            persons.append((me, f"homepages/{b:02d}/{k}", f"{base} {k:04d}", base, "numbered"))
            collaborators = rnd.sample(collaborator_pool, 3)
            venues = rnd.sample(VENUES, 2)
            start = rnd.randint(1995, 2012)
            orcid = f"0000-0002-{me:04d}-0000" if me % 3 == 0 else None
            for _ in range(rnd.randint(*PAPERS_PER_PERSON)):
                pid += 1
                sid, venue, prefix = rnd.choice(venues)
                year = start + rnd.randint(0, 8)
                # co-authors: usually the person's regulars, sometimes strangers
                others = [c for c in collaborators if rnd.random() < 0.75] or [rnd.choice(collaborators)]
                others += rnd.sample(collaborator_pool, rnd.randint(0, 1))
                names = [f"{base} {k:04d}"] + [f"Collab {c}" for c in others]
                orcids = [orcid if rnd.random() < 0.7 else None] + [None] * len(others)
                title = " ".join(rnd.sample(WORDS, 5)).capitalize() + "."
                pubs.append((pid, f"{sid}/p{pid}", year, sid, venue, prefix, len(names), title, False))
                src.append((f"{sid}/p{pid}", names, orcids))
                slots.append((me, pid, 1))
                for i, c in enumerate(others, start=2):
                    slots.append((c, pid, i))

    # two disambiguation bins, each holding papers of two "hidden" people, for the predict path
    for b, base in enumerate(["Bin One", "Bin Two"]):
        person_id += 1
        bin_id = person_id
        persons.append((bin_id, f"homepages/bin/{b}", base, base, "disambiguation"))
        # the same block also has numbered pages the bin's papers could belong to
        for k in (1, 2):
            person_id += 1
            numbered = person_id
            persons.append((numbered, f"homepages/bin{b}/{k}", f"{base} {k:04d}", base, "numbered"))
            collaborators = rnd.sample(collaborator_pool, 3)
            sid, venue, prefix = VENUES[k]
            for _ in range(6):
                pid += 1
                names = [f"{base} {k:04d}"] + [f"Collab {c}" for c in collaborators]
                pubs.append((pid, f"{sid}/n{pid}", 2015 + rnd.randint(0, 5), sid, venue, prefix, len(names),
                             " ".join(rnd.sample(WORDS, 5)).capitalize() + ".", False))
                src.append((f"{sid}/n{pid}", names, [None] * len(names)))
                slots.append((numbered, pid, 1))
                for i, c in enumerate(collaborators, start=2):
                    slots.append((c, pid, i))
            # unassigned papers sitting on the bin, sharing this person's collaborators
            for _ in range(5):
                pid += 1
                names = [base] + [f"Collab {c}" for c in collaborators]
                pubs.append((pid, f"{sid}/b{pid}", 2018 + rnd.randint(0, 4), sid, venue, prefix, len(names),
                             " ".join(rnd.sample(WORDS, 5)).capitalize() + ".", False))
                src.append((f"{sid}/b{pid}", names, [None] * len(names)))
                slots.append((bin_id, pid, 1))
                for i, c in enumerate(collaborators, start=2):
                    slots.append((c, pid, i))

    person_id, pid, link_people = _link_world(rnd, person_id, pid, persons, slots, pubs, src)

    con.execute("CREATE TABLE persons (person_id INTEGER, key VARCHAR, name VARCHAR, base_name VARCHAR, page_kind VARCHAR)")
    con.execute("CREATE TABLE slots (person_id INTEGER, pid INTEGER, position SMALLINT)")
    con.execute("""CREATE TABLE pubs (pid INTEGER, key VARCHAR, year SMALLINT, sid VARCHAR, venue VARCHAR,
                                      key_prefix VARCHAR, n_authors INTEGER, title VARCHAR,
                                      is_preprint BOOLEAN, type VARCHAR)""")
    con.execute("CREATE TABLE src (key VARCHAR, authors VARCHAR[], author_orcids VARCHAR[])")
    con.execute("CREATE TABLE _meta (k VARCHAR, v VARCHAR)")
    con.executemany("INSERT INTO persons VALUES (?, ?, ?, ?, ?)", persons)
    con.executemany("INSERT INTO slots VALUES (?, ?, ?)", slots)
    # the record type follows the key prefix, as it does in the real dump
    pubs = [row + ("article" if row[5] == "journals" else "inproceedings",) for row in pubs]
    con.executemany("INSERT INTO pubs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", pubs)
    con.executemany("INSERT INTO src VALUES (?, ?, ?)", src)
    con.executemany("INSERT INTO _meta VALUES (?, ?)", [
        ("fingerprint", fingerprint), ("records", str(len(pubs))), ("latest_mdate", "2026-09-01"),
        ("parquet", "synthetic"), ("last_full_year", "2025"),
    ])
    con.execute("CHECKPOINT")
    con.close()
    return path, {"people": len(persons), "papers": len(pubs), "slots": len(slots), "link_people": link_people}


if __name__ == "__main__":
    import sys
    p, stats = make(Path(sys.argv[1] if len(sys.argv) > 1 else "cache"))
    print(f"wrote {p}: {stats}")
