"""
Dewey's own abstract index: the abstracts of dblp's papers, searchable with BM25.

RAGScholar answered DBLP-QA from a Lucene index of 4.6 million Semantic Scholar abstracts. Dewey's
abstract search (content.py) found abstracts live instead - dblp's title search and OpenAlex's two
searches, about 70 candidates a question - and the study measured what that costs: the paper a
question was written from is among them for about 60% of DBLP-QA's questions. This is the closed
index rebuilt from open data:

  1. fetch - OpenAlex's quarterly snapshot (CC0: 476M works, 707 GB of parquet), read where it lies
     over HTTPS: only the DOI and abstract columns (about 30% of the bytes), keeping the works whose
     DOI a dblp record links to (paperids: 7.4M DOIs, and 0.4M arXiv ids as 10.48550/arxiv.<id>).
     One part file per batch of snapshot files, so a stopped fetch resumes where it stopped.
  2. build - each abstract rebuilt from OpenAlex's inverted index and filed under its dblp record,
     with dblp's title; BM25 (Tantivy: k1 = 1.2, b = 0.75, idf over the whole index) over the
     study's own tokens (dblpqa_rag.tokens) of title + abstract, so it ranks as the study's BM25
     does, with whole-corpus statistics as RAGScholar's Lucene index had.
  3. search - the question's tokens OR-ed, best first.

The index is named by the snapshot's date and kept across dblp dumps (a fetch takes hours, a dump
changes monthly): a key dblp no longer has is dropped where results are shown, and a paper newer
than the snapshot is still found by the live searches.

  python -m chat.cli abstract-index fetch      stream the snapshot (hours; resumable)
  python -m chat.cli abstract-index build      the searchable index from what was fetched
  python -m chat.cli abstract-index status     what is fetched, built and opened
  python -m chat.cli abstract-index search Q   the top ten for a query
"""
import json
import logging
import re
import shutil
import threading
import time
from datetime import datetime, timezone

from . import config, dblpqa_rag as RAG

try:
    import tantivy
except ImportError:             # optional: without it the abstract search is live only, as before
    tantivy = None

log = logging.getLogger("dblp.chat.abstractindex")

VERSION = "1"
SNAPSHOT = "https://openalex.s3.amazonaws.com/data/parquet/"
MIN_WORDS = 20                  # shorter "abstracts" are mostly placeholders ("No abstract available.")
SAFE_URL = re.compile(r"^[A-Za-z0-9:/._=-]+$")
META = "dewey-index.json"

# dblp's DOIs, and its arXiv ids as the DOIs arXiv registers for them (paperids)
WANTED = """
    CREATE OR REPLACE TEMP TABLE wanted AS
    SELECT DISTINCT CASE WHEN kind = 'doi' THEN id ELSE '10.48550/arxiv.' || id END AS doi FROM x.ids"""
IDS = "SELECT key, CASE WHEN kind = 'doi' THEN id ELSE '10.48550/arxiv.' || id END AS doi FROM x.ids"


def parts_dir(date):
    return config.MODELS_DIR / f"openalex-{date}" / "parts"


def index_dir(date):
    return config.MODELS_DIR / f"chat-abstracts-{date}"


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- 1. fetch

def snapshot(http):
    """(release date, [(https url, records, bytes)]) for the works of OpenAlex's latest snapshot."""
    r = http.get(SNAPSHOT + "manifest.json", timeout=120)
    r.raise_for_status()
    manifest = r.json()
    works = next(e for e in manifest["entities"] if e["entity"] == "works")
    return manifest["date"], [(f["url"].replace("s3://openalex/", "https://openalex.s3.amazonaws.com/"),
                               f["meta"].get("record_count", 0), f["meta"].get("content_length", 0))
                              for f in works["files"]]


def _remote(con, threads):
    """httpfs for reading the snapshot in place; the build side of the DOI match pinned to dblp's DOIs."""
    ext = config.MODELS_DIR / "duckdb-extensions"
    ext.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET extension_directory = '{ext}'")
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("SET http_retries = 8")
    con.execute("SET http_retry_wait_ms = 1000")
    con.execute(f"SET threads = {int(threads)}")


def _pin_build_side(con):
    # a batch of snapshot files can hold fewer rows than dblp has DOIs, and DuckDB would then hash
    # the snapshot's side - abstracts and all - instead of the DOIs: keep the query's own order
    for setting in ("join_order,build_side_probe_side", "join_order"):
        try:
            con.execute(f"SET disabled_optimizers = '{setting}'")
            return
        except Exception:
            continue


def _batch_sql(urls, target=None):
    listing = ", ".join(f"'{u}'" for u in urls)
    select = f"""
        SELECT w.doi, w.id AS openalex, w.abstract_inverted_index AS inverted
        FROM (SELECT lower(replace(doi, 'https://doi.org/', '')) AS doi, id, abstract_inverted_index
              FROM read_parquet([{listing}], union_by_name = true, hive_partitioning = false)
              WHERE doi IS NOT NULL AND abstract_inverted_index IS NOT NULL) w
        WHERE w.doi IN (SELECT doi FROM wanted)"""
    return select if target is None else f"COPY ({select}) TO '{target}' (FORMAT parquet, COMPRESSION zstd)"


def fetch(con, http=None, files=None, date=None, threads=16, batch=20, limit=None, out=print):
    """Stream the snapshot's DOI and abstract columns, keeping the works dblp links to; returns the
    fetch's summary. `con` has the paper ids attached (x). Batches already written are skipped."""
    if files is None:
        date, files = snapshot(http)
    bad = [u for u, _, _ in files if not SAFE_URL.match(u)]
    if bad:
        raise ValueError(f"unexpected characters in a snapshot url: {bad[0]}")
    folder = parts_dir(date)
    folder.mkdir(parents=True, exist_ok=True)
    if any(u.startswith("http") for u, _, _ in files):
        _remote(con, threads)
    _pin_build_side(con)
    con.execute(WANTED)
    wanted = con.execute("SELECT count(*) FROM wanted").fetchone()[0]
    batches = [files[i:i + batch] for i in range(0, len(files), batch)]
    todo = batches[:limit] if limit else batches
    total_bytes = sum(b for _, _, b in files)
    out(f"OpenAlex snapshot {date}: {len(files)} works files, {total_bytes / 1e9:.0f} GB, "
        f"{sum(r for _, r, _ in files) / 1e6:.0f}M works; keeping those among dblp's {wanted:,} DOIs, "
        f"{len(batches)} batches of {batch}")
    if todo:
        # the join's type is the check that the DOIs, not the abstracts, are what gets hashed
        plan = con.execute("EXPLAIN " + _batch_sql([u for u, _, _ in todo[0]])).fetchall()
        joins = [ln.strip(" │┃") for _, text in plan for ln in text.splitlines() if "JOIN" in ln or "Join Type" in ln]
        out("  plan: " + (" / ".join(j for j in joins if j) or "(no join line found)"))
    summary_path = folder.parent / "fetch.json"
    t0 = time.time()
    done_bytes = 0
    for i, group in enumerate(todo):
        part = folder / f"part-{i:04d}.parquet"
        size = sum(b for _, _, b in group)
        if part.exists():
            continue
        tmp = folder / f"part-{i:04d}.tmp"
        t = time.time()
        try:
            con.execute(_batch_sql([u for u, _, _ in group], tmp))
        except Exception as e:
            tmp.unlink(missing_ok=True)
            out(f"  batch {i + 1}/{len(batches)} failed ({type(e).__name__}: {str(e)[:200]}); a later fetch retries it")
            continue
        tmp.replace(part)
        done_bytes += size
        kept = con.execute(f"SELECT count(*) FROM read_parquet('{part}')").fetchone()[0]
        rate = done_bytes / max(1e-9, time.time() - t0)
        left = sum(b for g in todo[i + 1:] for _, _, b in g)
        out(f"  batch {i + 1}/{len(batches)}: {len(group)} files, {kept:,} kept, {time.time() - t:.0f}s; "
            f"{rate / 1e6:.0f} MB/s of snapshot, about {left / max(rate, 1) / 3600:.1f} h left")
    parts = sorted(folder.glob("part-*.parquet"))
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{folder}/part-*.parquet')").fetchone()[0] if parts else 0
    summary = {"snapshot": date, "files": len(files), "batch": batch, "batches": len(batches),
               "batches_done": len(parts), "complete": len(parts) == len(batches), "works_kept": rows,
               "dblp_dois": wanted, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _save(summary_path, summary)
    out(f"{len(parts)}/{len(batches)} batches fetched, {rows:,} works with an abstract kept"
        + ("" if summary["complete"] else " - run fetch again to finish"))
    return summary


def latest_fetch():
    """The newest snapshot fetched (complete or not): its fetch.json, or {}."""
    found = sorted(config.MODELS_DIR.glob("openalex-*/fetch.json"))
    return _load(found[-1]) if found else {}


# --------------------------------------------------------------------------- 2. build

def clean(text):
    """An abstract rebuilt from OpenAlex's inverted index, without markup or a leading "Abstract"."""
    text = " ".join(re.sub(r"<[^>]+>", " ", text or "").split())
    return re.sub(r"^(abstract|summary)\s*[:.\-]?\s+", "", text, flags=re.I)


def _schema():
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("key", stored=True, tokenizer_name="raw", index_option="basic")
    sb.add_text_field("text", stored=False, tokenizer_name="default", index_option="freq")
    for name in ("title", "abstract", "doi"):
        sb.add_bytes_field(name, stored=True, indexed=False)
    return sb.build()


def build(con, date=None, partial=False, heap=1_000_000_000, threads=4, out=print):
    """The searchable index from the fetched parts; returns its metadata. `con` has the serving
    database (s) and the paper ids (x) attached."""
    if tantivy is None:
        raise RuntimeError("tantivy is not installed in this image")
    fetched = _load(parts_dir(date).parent / "fetch.json") if date else latest_fetch()
    if not fetched:
        raise RuntimeError("nothing fetched yet: run `abstract-index fetch` first")
    date = fetched["snapshot"]
    if not fetched.get("complete") and not partial:
        raise RuntimeError(f"the fetch of {date} is incomplete ({fetched['batches_done']}/{fetched['batches']} "
                           f"batches): run fetch again, or build with --partial")
    folder = parts_dir(date)
    parts = sorted(folder.glob("part-*.parquet"))
    target = index_dir(date)
    building = target.with_name(target.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir(parents=True)
    t0 = time.time()
    # the few records two fetched DOIs point to (a record with two DOIs): only these need a memory of
    # what was indexed, so millions of keys are never held
    multi = {k for (k,) in con.execute(f"""
        SELECT i.key FROM (SELECT DISTINCT doi FROM read_parquet('{folder}/part-*.parquet')) p
        JOIN ({IDS}) i ON i.doi = p.doi GROUP BY i.key HAVING count(*) > 1""").fetchall()}
    index = tantivy.Index(_schema(), path=str(building), reuse=False)
    writer = index.writer(heap_size=heap, num_threads=threads)
    seen = set()
    docs = short = unreadable = 0
    for n, part in enumerate(parts, 1):
        # one part at a time: it is the small side of both joins, so DuckDB never hashes the abstracts
        rows = con.execute(f"""
            SELECT i.key, p.doi, p.inverted, pb.title
            FROM read_parquet('{part}') p
            JOIN ({IDS}) i ON i.doi = p.doi
            JOIN s.pubs pb ON pb.key = i.key""").fetchall()
        for key, doi, inverted, title in rows:
            if key in multi:
                if key in seen:
                    continue                    # a record with two DOIs: the first one's abstract
                seen.add(key)
            try:
                text = clean(RAG._inverted(json.loads(inverted)))
            except (ValueError, TypeError, AttributeError):
                unreadable += 1
                continue
            if len(text.split()) < MIN_WORDS:
                short += 1
                continue
            doc = tantivy.Document()
            doc.add_text("key", key)
            doc.add_text("text", " ".join(RAG.tokens(f"{title or ''}. {text}")))
            doc.add_bytes("title", (title or "").encode("utf-8"))
            doc.add_bytes("abstract", text.encode("utf-8"))
            doc.add_bytes("doi", (doi or "").encode("utf-8"))
            writer.add_document(doc)
            docs += 1
        if n % 20 == 0 or n == len(parts):
            out(f"  {n}/{len(parts)} parts, {docs:,} abstracts indexed, {time.time() - t0:.0f}s")
    writer.commit()
    writer.wait_merging_threads()
    meta = {"version": VERSION, "snapshot": date, "complete_fetch": bool(fetched.get("complete")),
            "documents": docs, "short_dropped": short, "unreadable": unreadable,
            "works_fetched": fetched.get("works_kept"), "dblp_dois": fetched.get("dblp_dois"),
            "dump": con.execute("SELECT v FROM s._meta WHERE k = 'fingerprint'").fetchone()[0],
            "bm25": {"k1": 1.2, "b": 0.75, "idf": "whole index", "tokens": "dblpqa_rag.tokens of title + abstract"},
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "build_seconds": round(time.time() - t0, 1)}
    _save(building / META, meta)
    shutil.rmtree(target, ignore_errors=True)
    building.rename(target)
    forget()
    out(f"built {target.name}: {docs:,} abstracts ({short:,} too short, {unreadable:,} unreadable) "
        f"in {time.time() - t0:.0f}s")
    return meta


# --------------------------------------------------------------------------- 3. search

_lock = threading.Lock()
_open = {"index": None, "meta": None, "stamp": None, "checked": 0.0}


def _newest():
    found = sorted(p for p in config.MODELS_DIR.glob("chat-abstracts-*")
                   if p.is_dir() and not p.name.endswith(".building") and (p / META).exists())
    return found[-1] if found else None


def opened(recheck=60):
    """(index, meta) of the newest built index, or (None, None). Looked for again every `recheck`
    seconds, so a build finished while the service runs is picked up without a restart."""
    if tantivy is None:
        return None, None
    with _lock:
        now = time.time()
        if now - _open["checked"] < recheck:
            return _open["index"], _open["meta"]
        _open["checked"] = now
        path = _newest()
        stamp = (str(path), (path / META).stat().st_mtime) if path else None
        if stamp != _open["stamp"]:
            index = meta = None
            if path:
                try:
                    meta = _load(path / META)
                    if meta.get("version") == VERSION:
                        index = tantivy.Index.open(str(path))
                    else:
                        log.info("abstract index %s is version %s (now %s): build it again",
                                 path.name, meta.get("version"), VERSION)
                        meta = None
                except Exception as e:
                    log.warning("could not open the abstract index %s: %s", path.name, e)
                    index = meta = None
            _open.update(index=index, meta=meta, stamp=stamp)
        return _open["index"], _open["meta"]


def forget():
    """Look for the index again on the next call (after a build in this process)."""
    with _lock:
        _open.update(index=None, meta=None, stamp=None, checked=0.0)


def available():
    return opened()[0] is not None


def _hit(doc, score=None):
    get = lambda name: (doc.get_first(name) or b"").decode("utf-8")
    hit = {"key": doc.get_first("key"), "title": get("title"), "abstract": get("abstract"), "doi": get("doi") or None}
    if score is not None:
        hit["score"] = round(float(score), 4)
    return hit


def search(query, limit=50):
    """[{key, title, abstract, doi, score}] best first: BM25 over the whole index for the query's
    tokens, OR-ed. Empty without an index."""
    index, _meta = opened()
    terms = RAG.tokens(query)
    if index is None or not terms:
        return []
    schema = index.schema
    q = tantivy.Query.boolean_query([(tantivy.Occur.Should, tantivy.Query.term_query(schema, "text", t, index_option="freq"))
                                     for t in terms])
    searcher = index.searcher()
    # count=False: no total, so Tantivy may skip the documents that cannot reach the top
    return [_hit(searcher.doc(address), score) for score, address in searcher.search(q, limit, count=False).hits]


def lookup(keys):
    """{key: {key, title, abstract, doi}} for the keys the index holds."""
    index, _meta = opened()
    keys = [k for k in dict.fromkeys(keys or []) if isinstance(k, str) and k]
    if index is None or not keys:
        return {}
    q = tantivy.Query.term_set_query(index.schema, "key", keys)
    searcher = index.searcher()
    found = {}
    for _score, address in searcher.search(q, len(keys) * 2, count=False).hits:
        hit = _hit(searcher.doc(address))
        found.setdefault(hit["key"], hit)
    return found


def info():
    """What the status page and the evaluation report: the index's metadata, or None."""
    index, meta = opened()
    if index is None:
        return None
    return dict(meta, searchable=index.searcher().num_docs)
