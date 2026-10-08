"""
Questions about dblp's records: Dewey against a RAGScholar-style pipeline, scored without a judge.

DBLP-QA asks what papers say. Much of what people ask a bibliography is about its records instead:
where and when a paper appeared, who wrote it, how many papers someone has, how many a venue
published in a year, who publishes most there, how often two people wrote together, how many people
share a name. A pipeline that retrieves five abstracts and reads them can answer the first two when
retrieval finds the paper; it has nothing to count, rank or tell people apart with. The question
types follow DBLP-QuAD (Banerjee et al., 2023), the question-answering benchmark over the dblp
knowledge graph; the questions are drawn from today's dump.

`build` draws them, ten per type, and computes every reference answer by SQL over the raw parquet -
not through Dewey's tools or the serving tables they read - then freezes them in questions.json before
any system answers. `score` needs no judge: the reference number must appear in the answer, or the
reference names (the year and venue, every author's surname, the top author's name) must.

Arms (`run`):
  dewey   the assistant as deployed;
  rag     RAGScholar's pipeline re-implemented for these questions: the search the abstract tool uses
          (dblp's title search and OpenAlex, BM25 over title and abstract), with the top five records
          given as context with the fields RAGScholar's index holds - dblp key, DOI, title, authors,
          year and abstract - answered by Mistral-7B with the paper's sampling, or by gpt-4.1-mini;
  closed  the question alone, to gpt-4.1-mini: what a model knows without retrieval.
"""
import json
import logging
import math
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from . import agent, budget, config, content, dblpqa as DQ, dblpqa_rag as RAG

log = logging.getLogger("dblp.chat.dblpqa_structured")

TYPES = ("venue_year", "authors", "author_count", "venue_year_count", "venue_top_author", "pair_count",
         "namesakes")
PER_TYPE = 10
# what a question may be about: big enough to be worth asking, small enough to be checkable
RANGES = {"paper_years": (2005, 2024), "paper_authors": (2, 4), "title_chars": 40,
          "author_records": (20, 400), "pair_author_records": (30, 300), "pair_records": 2,
          "venue_years": (2010, 2024), "venue_year_records": (100, 3000), "venue_records": (1000, 30000),
          "namesakes": (2, 100)}
ARMS = ("dewey", "rag", "closed", "sql")
# the "sql" arm: the same model, loop and limits as Dewey, with only the read-only SQL tool and the schema -
# what Dewey's typed tools and counting rules add over plain access to the same database
# not src: on the serving database it is a view over the raw parquet, which the guarded SQL may not read
SQL_TABLES = ("pubs", "persons", "person_names", "slots", "career", "person_stats", "series", "word_year",
              "person_degree")
SQL_SYSTEM = """You answer questions about the dblp computer-science bibliography by querying its database
with run_sql, your only tool (its description calls it a last resort; here it is how you answer). DuckDB,
one read-only SELECT per call, tables named without a prefix; each query is stopped after {timeout:.0f}
seconds. You see at most {rows} rows of any result, so count and rank in SQL (count(*), ORDER BY ... LIMIT)
rather than by reading rows. Answer only from what your queries return; if the data cannot answer the
question, say so. Two to four sentences; lead with the answer.

Tables and columns:
{schema}

What the data means:
- pubs: one row per dblp record except author pages and proceedings volumes - journal articles,
  conference papers, preprints, books, chapters, theses (type; is_preprint marks CoRR and other informal
  records). title is written as dblp writes it, usually ending with a period; title_norm is the title
  lower-cased with only letters and digits. venue is that record's journal or booktitle string.
- Venues: sid is the series a record belongs to (conf/icml, journals/tit), taken from its key. One
  series' records can carry several venue strings (volumes, workshops, renamings), so identify a venue by
  its sid and count by sid. series has one row per conference or journal series; usual_name is its most
  common venue string, and series.papers counts only its journal and conference papers, preprints excluded.
- People: persons are dblp author pages. page_kind 'numbered' (such as 'Wei Wang 0003') is one distinct
  person; 'regular' is an unnumbered page; 'disambiguation' is a bin holding the papers of different
  people who share a name, not a person. base_name is the name without its number. A page can carry
  several names (persons.names); person_names has one row per name, to find a page from a name exactly as
  written.
- slots links each record (pid) to its author pages (person_id), one row per author position. It holds
  no names: join persons for them. person_id is NULL for a name that has no page, and on_bin marks a slot
  whose page is a bin.
- person_stats.n_pubs counts every record of a page, preprints included; career.papers counts only its
  journal and conference papers, preprints excluded; person_degree counts co-authors on papers with 2 to
  50 authors; word_year counts title words of journal and conference papers."""
RECORD_SYSTEM = ("You answer questions about computer-science papers using the search results you are given: "
                 "each has its dblp key, DOI, title, authors, year and abstract. Answer in one to three sentences.")
NUMBERED = re.compile(r" \d{4}$")
# citation markers ("[2]") are not numbers the answer states, and a journal is not named by an answer
# that places it in a conference ("Conference on Computer Vision and Pattern Recognition" is not the
# journal "Pattern Recognit.")
CITATION = re.compile(r"\[\d+\]")
CONFERENCE = re.compile(r"\b(conference|proceedings|workshop|symposium)\b", re.I)


def study_dir():
    return config.MODELS_DIR / "dblpqa-structured"


# --------------------------------------------------------------------------- normalising and scoring

def fold(text):
    """Lower case, accents removed: 'Jürgen' and 'Jurgen' are the same name in an answer."""
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def compact(text):
    return re.sub(r"[^a-z0-9]", "", fold(text))


def base_name(name):
    return NUMBERED.sub("", name or "")


SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def surname(name):
    """The last word of a name that is not a suffix such as 'Jr.'."""
    words = [w for w in fold(base_name(name)).replace(".", " ").split() if w not in SUFFIXES]
    return words[-1] if words else ""


def numbers(text):
    """Every whole number written in an answer, thousands separators allowed."""
    return {int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text or "") if x.replace(",", "").isdigit()}


def words(text):
    return re.findall(r"[a-z0-9]+", fold(text))


def abbreviation_in(venue, answer):
    """dblp writes venues abbreviated ("Int. J. Medical Informatics", "Comput. Electr. Eng."): the venue
    is named when each of its words begins a word of the answer, in order - so "International Journal
    of Medical Informatics" names it, and so does the abbreviation itself."""
    said, at = words(answer), 0
    for token in words(venue):
        while at < len(said) and not said[at].startswith(token):
            at += 1
        if at == len(said):
            return False
        at += 1
    return bool(words(venue))


def score(question, answer):
    """True when the answer states the reference: its number, or its names."""
    answer = CITATION.sub(" ", answer or "")
    ref, kind = question["ref"], question["type"]
    if "number" in ref:
        return ref["number"] in numbers(answer)
    low = fold(answer)
    if kind == "venue_year":
        venue, series = ref["venues"][0], ref["venues"][-1]
        # the series' short name ("iwqos") counts as a whole word only, and only if it is not a
        # two-letter code that would be found inside any other word
        named = abbreviation_in(venue, answer) or (len(series) >= 3 and series.lower() in words(answer))
        if named and (question.get("entity") or {}).get("key", "").startswith("journals/") and CONFERENCE.search(answer):
            named = False
        return str(ref["year"]) in (answer or "") and named
    if kind == "authors":
        return all(surname(n) in low for n in ref["names"])
    if kind == "venue_top_author":
        # initials ("K.") may be left out or written differently; the names may not
        return all(w in words(answer) for w in words(base_name(ref["names"][0])) if len(w) > 1)
    raise ValueError(f"unknown question type {kind}")


# --------------------------------------------------------------------------- drawing the questions

def _sample(con, sql, params, n, seed, by):
    """n rows of `sql`, in an order fixed by the seed (a hash of column `by`), so a rebuild draws the
    same questions."""
    return con.execute(f"SELECT * FROM ({sql}) ORDER BY hash(CAST({by} AS VARCHAR) || '{int(seed)}') LIMIT {int(n)}",
                       params).fetchall()


def build(parquet=None, seed=7, out=print):
    """Draw the questions and compute their answers from the raw parquet; write questions.json."""
    import duckdb
    parquet = Path(parquet or RAG.PARQUET)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY}'")
    con.execute(f"SET threads = {config.DUCKDB_THREADS}")
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{config.TMP_DIR}'")
    src = f"read_parquet('{parquet.as_posix()}')"
    t0 = time.time()
    con.execute(f"""CREATE TEMP TABLE r AS
        SELECT key, type, title, year, authors, journal, booktitle,
               split_part(key, '/', 1) || '/' || split_part(key, '/', 2) AS series,
               coalesce(journal, booktitle) AS venue,
               regexp_replace(lower(title), '[^a-z0-9]', '', 'g') AS title_norm
        FROM {src} WHERE type NOT IN ('www', 'proceedings')""")
    con.execute(f"""CREATE TEMP TABLE pages AS
        SELECT key AS page, authors AS names, coalesce(publtype, '') = 'disambiguation' AS bin
        FROM {src} WHERE type = 'www' AND key LIKE 'homepages/%'""")
    # every name a page carries, and whether it belongs to exactly one person page
    con.execute("""CREATE TEMP TABLE page_names AS
        SELECT unnest(names) AS name, page, bin FROM pages""")
    con.execute("""CREATE TEMP TABLE name_pages AS
        SELECT name, count(*) AS pages, bool_or(bin) AS on_bin FROM page_names GROUP BY name""")
    con.execute("""CREATE TEMP TABLE name_counts AS
        SELECT name, count(DISTINCT key) AS records FROM (SELECT key, unnest(authors) AS name FROM r)
        GROUP BY name""")
    # people a question can name without ambiguity: one page, not a bin, that page has one name
    con.execute("""CREATE TEMP TABLE people AS
        SELECT pn.name, nc.records FROM page_names pn
        JOIN pages p ON p.page = pn.page AND len(p.names) = 1 AND NOT p.bin
        JOIN name_pages np ON np.name = pn.name AND np.pages = 1
        JOIN name_counts nc ON nc.name = pn.name""")
    out(f"tables in {time.time() - t0:.0f}s")
    questions = []

    def add(kind, question, ref, entity):
        questions.append({"id": f"s{len(questions) + 1}", "type": kind, "question": question, "ref": ref,
                          "entity": entity})

    # 1-2: a paper's venue and year, and its authors - answerable by any system that finds the record
    papers = _sample(con, """
        SELECT key, title, year, venue, series, authors FROM r
        WHERE type IN ('article', 'inproceedings') AND coalesce(journal, '') <> 'CoRR'
          AND year BETWEEN ? AND ? AND len(authors) BETWEEN ? AND ? AND length(title) >= ?
          AND venue IS NOT NULL
          AND title_norm IN (SELECT title_norm FROM r GROUP BY title_norm HAVING count(*) = 1)""",
                     [*RANGES["paper_years"], *RANGES["paper_authors"], RANGES["title_chars"]],
                     2 * PER_TYPE, seed, "key")
    for i, (key, title, year, venue, series, authors) in enumerate(papers):
        title = title.rstrip(".")
        if i < PER_TYPE:
            add("venue_year", f"Where and when was the paper '{title}' published?",
                {"year": year, "venues": [venue, series.split("/")[1]]}, {"key": key})
        else:
            add("authors", f"Who are the authors of the paper '{title}'?",
                {"names": [base_name(a) for a in authors]}, {"key": key})

    # 3: how many publications one person has
    for name, records in _sample(con, "SELECT name, records FROM people WHERE records BETWEEN ? AND ?",
                                 list(RANGES["author_records"]), PER_TYPE, seed, "name"):
        add("author_count", f"How many publications does dblp list for {name}?", {"number": records},
            {"name": name})

    # 4-5: venues, by series (conf/icml, journals/tit), named by their usual venue string
    con.execute("""CREATE TEMP TABLE series AS
        SELECT series, mode(venue) AS name, count(*) AS records FROM r
        WHERE (series LIKE 'conf/%' OR series LIKE 'journals/%') AND series <> 'journals/corr'
        GROUP BY series""")
    con.execute("""CREATE TEMP TABLE unique_series AS
        SELECT * FROM series WHERE name IN (SELECT name FROM series GROUP BY name HAVING count(*) = 1)""")
    used = set()
    for series, name, year, records in _sample(con, """
        SELECT s.series, s.name, r.year, count(*) AS records FROM r JOIN unique_series s USING (series)
        WHERE r.year BETWEEN ? AND ? GROUP BY s.series, s.name, r.year HAVING count(*) BETWEEN ? AND ?""",
                                               [*RANGES["venue_years"], *RANGES["venue_year_records"]],
                                               20 * PER_TYPE, seed, "series || CAST(year AS VARCHAR)"):
        if series in used or len(used) == PER_TYPE:
            continue                            # one year per venue
        used.add(series)
        add("venue_year_count", f"How many papers does dblp list for {name} in {year}?", {"number": records},
            {"series": series, "year": year})
    tops = []
    for series, name in _sample(con, "SELECT series, name FROM unique_series WHERE records BETWEEN ? AND ?",
                                list(RANGES["venue_records"]), 4 * PER_TYPE, seed + 1, "series"):
        ranked = con.execute("""
            SELECT a.name, count(DISTINCT a.key) AS n, any_value(np.on_bin) AS on_bin, any_value(np.pages) AS pages
            FROM (SELECT key, unnest(authors) AS name FROM r WHERE series = ?) a
            LEFT JOIN name_pages np USING (name) GROUP BY a.name ORDER BY n DESC LIMIT 2""", [series]).fetchall()
        if len(ranked) == 2 and ranked[0][1] > ranked[1][1] and not ranked[0][2] and ranked[0][3] == 1:
            tops.append((series, name, ranked[0][0], ranked[0][1]))
        if len(tops) == PER_TYPE:
            break
    for series, name, top, n in tops:
        add("venue_top_author", f"Who has published the most papers at {name}?", {"names": [top]},
            {"series": series, "papers": n})

    # 6: how many papers two people wrote together
    pairs = []
    for a, _ in _sample(con, "SELECT name, records FROM people WHERE records BETWEEN ? AND ?",
                        list(RANGES["pair_author_records"]), 4 * PER_TYPE, seed + 2, "name"):
        found = con.execute("""
            SELECT b, count(DISTINCT key) AS n
            FROM (SELECT key, unnest(authors) AS b FROM r WHERE list_contains(authors, ?))
            WHERE b <> ? AND b IN (SELECT name FROM people) GROUP BY b HAVING count(DISTINCT key) >= ?
            ORDER BY hash(b || ?) LIMIT 1""", [a, a, RANGES["pair_records"], str(seed)]).fetchall()
        if found:
            pairs.append((a, found[0][0], found[0][1]))
        if len(pairs) == PER_TYPE:
            break
    for a, b, n in pairs:
        add("pair_count", f"How many papers have {a} and {b} written together?", {"number": n}, {"a": a, "b": b})

    # 7: how many people share a name - numbered pages only, so "how many people" has one answer
    for base, numbered in _sample(con, """
        SELECT base, count(*) FILTER (WHERE numbered) AS numbered FROM (
            SELECT regexp_replace(names[1], ' [0-9]{4}$', '') AS base,
                   regexp_matches(names[1], ' [0-9]{4}$') AS numbered, bin
            FROM pages WHERE NOT bin)
        GROUP BY base HAVING count(*) FILTER (WHERE numbered) BETWEEN ? AND ?
                         AND count(*) FILTER (WHERE NOT numbered) = 0""", list(RANGES["namesakes"]),
                                  PER_TYPE, seed + 3, "base"):
        add("namesakes", f"How many different people named {base} does dblp distinguish?", {"number": numbered},
            {"base": base})

    con.close()
    counts = {t: sum(1 for q in questions if q["type"] == t) for t in TYPES}
    payload = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed,
               "parquet": {"path": str(parquet), "bytes": parquet.stat().st_size,
                           "mtime": datetime.fromtimestamp(parquet.stat().st_mtime, timezone.utc).isoformat()},
               "per_type": counts, "seconds": round(time.time() - t0, 1), "questions": questions}
    study_dir().mkdir(parents=True, exist_ok=True)
    (study_dir() / "questions.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out(f"{len(questions)} questions ({counts}) in {payload['seconds']}s -> {study_dir() / 'questions.json'}")
    return payload


def load():
    path = study_dir() / "questions.json"
    if not path.exists():
        raise SystemExit("no questions yet: run `dblpqa structured-build` first")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- the arms

def _authors(cur, keys):
    rows = cur.execute("SELECT key, authors FROM s.src WHERE key IN (SELECT unnest(?::VARCHAR[]))",
                       [list(keys)]).fetchall() if keys else []
    return {k: [base_name(a) for a in (authors or [])] for k, authors in rows}


def record_context(ctx, question):
    """RAGScholar's retrieval for one question: the five best records with the fields its index holds."""
    t0 = time.time()
    cands, status, degraded, _cached = content.pool_for(ctx, question, t0 + config.CONTENT_DEADLINE)
    keys = content.rank(question, cands)[:config.CONTENT_TOP]
    cur = ctx.cursor()
    info = content._records(cur, keys)
    authors = _authors(cur, keys)
    blocks = []
    for i, key in enumerate(keys, 1):
        c, rec = cands[key], info.get(key, {})
        blocks.append(f"[{i}] Key: {key}\nDOI: {c.get('doi') or '-'}\nTitle: {rec.get('title') or c.get('title')}\n"
                      f"Authors: {', '.join(authors.get(key) or []) or '-'}\nYear: {rec.get('year') or '-'}\n"
                      f"Abstract: {c.get('abstract') or '(no abstract available)'}")
    return "\n\n".join(blocks) or "(no records were retrieved)", keys, status


def sql_system(ctx):
    """The SQL arm's instructions, with every column of the tables it may query."""
    rows = ctx.cursor().execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_catalog = 's' "
        "AND table_name IN (SELECT unnest(?::VARCHAR[])) ORDER BY table_name, ordinal_position",
        [list(SQL_TABLES)]).fetchall()
    cols = {}
    for table, column, kind in rows:
        cols.setdefault(table, []).append(f"{column} {kind}")
    schema = "\n".join(f"- {t}({', '.join(cols[t])})" for t in SQL_TABLES if t in cols)
    return SQL_SYSTEM.format(schema=schema, rows=config.ROWS_TO_MODEL, timeout=config.SQL_TIMEOUT)


def run(ctx, client, arm, model=None, sampling=None, limit=None, out=print):
    data = load()
    questions = data["questions"][:limit] if limit else data["questions"]
    model = model or ("dewey" if arm == "dewey" else "gpt-4.1-mini")
    sampling = sampling or ("paper" if model.startswith(DQ.OLLAMA_PREFIX) else "ours")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = re.sub(r"[^a-z0-9.-]+", "-", model.lower())
    out_dir = study_dir() / "runs" / f"{stamp}-{arm}-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meter = DQ.Meter()
    ledger = budget.Ledger(path=study_dir() / "dewey-ledger.json")
    records, started = [], time.time()
    # the SQL arm gets the per-tool time Dewey's typed tools get, not run_sql's shorter default
    sql_timeout = config.SQL_TIMEOUT
    if arm == "sql":
        config.SQL_TIMEOUT = config.TOOL_TIMEOUT
    system = sql_system(ctx) if arm == "sql" else None
    dump = (getattr(ctx, "meta", None) or {}).get("fingerprint")
    with open(out_dir / "answers.jsonl", "w", encoding="utf-8") as fh:
        for i, q in enumerate(questions, 1):
            t0, extra = time.time(), {}
            if arm == "dewey":
                payloads = []
                done = agent.answer(ctx, client, q["question"], ledger=ledger, collect=payloads, channel="cli")
                text = done.get("answer") or ""
                extra = {"tools": done.get("tools") or [], "writer": done.get("model"), "error": done.get("error"),
                         "cost_usd": done.get("cost_usd"), "dump": dump}
            elif arm == "rag":
                context, keys, status = record_context(ctx, q["question"])
                messages = [{"role": "system", "content": RECORD_SYSTEM},
                            {"role": "user", "content": f"Search results:\n{context}\n\nQuestion: {q['question']}"}]
                text = DQ.answer(client, meter, model, messages, sampling)
                extra = {"retrieved": keys, "sources": status,
                         "paper_found": q["entity"].get("key") in keys if q["entity"].get("key") else None}
            elif arm == "sql":
                payloads = []
                done = agent.answer(ctx, client, q["question"], ledger=ledger, collect=payloads, channel="cli",
                                    tools=["run_sql"], system=system)
                text = done.get("answer") or ""
                extra = {"tools": done.get("tools") or [], "writer": done.get("model"), "error": done.get("error"),
                         "cost_usd": done.get("cost_usd"), "dump": dump,
                         "sql": [(p.get("arguments") or {}).get("sql") for p in payloads if p.get("name") == "run_sql"],
                         "sql_errors": [(p.get("result") or {}).get("summary") for p in payloads
                                        if p.get("name") == "run_sql" and (p.get("result") or {}).get("refused")],
                         "off_arm_tools": [t for t in (done.get("tools") or []) if t != "run_sql"]}
            elif arm == "closed":
                text = DQ.answer(client, meter, model, DQ.messages_for("closed-book", q["question"]), sampling)
            else:
                raise ValueError(f"unknown arm {arm!r}")
            rec = {"id": q["id"], "type": q["type"], "question": q["question"], "ref": q["ref"], "arm": arm,
                   "model": model, "answer": text, "correct": score(q, text), "seconds": round(time.time() - t0, 2),
                   **extra}
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(questions):
                out(f"  {arm} {model}: {i}/{len(questions)} answered, {sum(r['correct'] for r in records)} correct, "
                    f"{time.time() - started:.0f}s")
    config.SQL_TIMEOUT = sql_timeout
    summary = summarize(records)
    payload = {"arm": arm, "model": model, "sampling": sampling, "run_at": stamp, "questions": len(records),
               "built_at": data["built_at"], "results": summary, "cost_usd": round(meter.cost() + sum(
                   r.get("cost_usd") or 0 for r in records), 4), "seconds": round(time.time() - started, 1)}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(arm, model, summary, out)
    return payload


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 3), round(centre + half, 3)


def summarize(records):
    by_type = {}
    for t in TYPES:
        part = [r for r in records if r["type"] == t]
        if part:
            k = sum(r["correct"] for r in part)
            by_type[t] = {"questions": len(part), "correct": k, "accuracy": round(k / len(part), 3)}
    k, n = sum(r["correct"] for r in records), len(records)
    low, high = wilson(k, n)
    return {"questions": n, "correct": k, "accuracy": round(k / n, 3) if n else None, "low": low, "high": high,
            "by_type": by_type}


def print_summary(arm, model, summary, out=print):
    out(f"{arm} ({model}): {summary['correct']}/{summary['questions']} correct "
        f"({summary['accuracy']:.0%}, 95% CI {summary['low']:.0%}-{summary['high']:.0%})")
    for t, part in summary["by_type"].items():
        out(f"  {t:18s} {part['correct']:>2}/{part['questions']}")


def compare(out=print):
    """Every arm's latest run side by side, and Dewey against each other arm question by question (an
    exact sign test on the questions where exactly one of the two is right). Every answer is scored
    again with the current scorer, so a correction to it reaches runs answered before it."""
    qs = {q["id"]: q for q in load()["questions"]}
    runs = {}
    for summary in sorted((study_dir() / "runs").glob("*/summary.json")):
        got = json.loads(summary.read_text(encoding="utf-8"))
        recs = [json.loads(x) for x in (summary.parent / "answers.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        for r in recs:
            if r["id"] in qs and "answer" in r:      # a record without its answer keeps its verdict
                r["correct"] = score(qs[r["id"]], r["answer"] or "")
        runs[(got["arm"], got["model"])] = {r["id"]: r for r in recs}
    table = {f"{arm} ({model})": summarize(list(recs.values())) for (arm, model), recs in runs.items()}
    for label, s in table.items():
        out(f"{label:32s} {s['correct']:>3}/{s['questions']}  " + "  ".join(
            f"{t[:12]} {p['correct']}/{p['questions']}" for t, p in s["by_type"].items()))
    dewey = runs.get(("dewey", "dewey"))
    paired = {}
    if dewey:
        for (arm, model), recs in runs.items():
            if arm == "dewey":
                continue
            shared = sorted(set(dewey) & set(recs))
            only_dewey = sum(1 for q in shared if dewey[q]["correct"] and not recs[q]["correct"])
            only_other = sum(1 for q in shared if recs[q]["correct"] and not dewey[q]["correct"])
            m = only_dewey + only_other
            p = min(1.0, 2 * sum(math.comb(m, j) for j in range(0, min(only_dewey, only_other) + 1)) / 2 ** m) if m else 1.0
            paired[f"{arm} ({model})"] = {"questions": len(shared), "only_dewey": only_dewey, "only_other": only_other,
                                          "sign_test_p": round(p, 6)}
            out(f"Dewey vs {arm} ({model}): right only for Dewey on {only_dewey}, only for the other on {only_other} "
                f"of {len(shared)} (exact sign test p = {p:.2g})")
    result = {"arms": table, "dewey_vs": paired}
    (study_dir() / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
