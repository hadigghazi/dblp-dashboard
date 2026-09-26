# dblp Explorer

Interactive dashboard over the complete [dblp](https://dblp.org) dump, queried live: publishing trends,
author identity, the co-authorship network, titles, venues, data quality and an OpenAlex check, plus
explorers for any author, venue series or publication, four ML features and an assistant that
answers questions in plain language by querying the dump.

**https://dblp.hadighazi.com**

```
browser ──► Caddy (TLS) ──► web (nginx + React/ECharts) ──/api/──────► api    (FastAPI + DuckDB) ──► ~/dblp (read-only)
                                                        ├─/api/ml/────► mlapi  (3 sklearn models)
                                                        ├─/api/search/► searchapi (BM25 + embeddings)
                                                        └─/api/chat/──► chatapi (tool-calling assistant)
```

## How the data flows

The **api** reads `~/dblp/parquet/dblp.parquet`, the file the analysis's `dblp.duckdb` view points at.
It never opens `dblp.duckdb` itself: DuckDB locks its file, so an API holding it open would stop the
analysis scripts (and later ML jobs) from writing to it.

On startup the api builds its own **serving tables** into `cache/` (publications, author pages, one row
per author slot, careers, venue series, title words), using the same SQL predicates as the analysis
scripts. That takes a few minutes on the full dump and happens once per dump: a watcher rebuilds
automatically when a new `dblp.parquet` appears. After that **every request runs its own SQL** with the
filters you choose, and results are cached until the next rebuild.

Three results are too expensive to compute per request, so those pages show the **latest output of
the analysis job** (labelled with its date in the UI) and update when you re-run it:

| Page | Job output read from `~/dblp/eda_out/` |
|---|---|
| Co-author network | `02_network.txt` (igraph over 26M co-author pairs) |
| OpenAlex check | `06_openalex.txt` (external API sample) |
| Long tails: power-law fits | `07_statistics.txt` (the distributions themselves are live) |

### Check the live numbers against the report

```bash
cd ~/dblp-dashboard
docker compose -f docker-compose.prod.yml exec api python -m app.validate
```

compares every live query with the CSVs in `~/dblp/charts/` and prints the largest difference per chart.

## Author disambiguation (ML)

dblp holds 33,659 disambiguation bins - bare names like "Wei Wang" holding papers by many different
people - and 17.5% of papers (26.8% in the 2020s) have an author sitting on one. The 147,223
**numbered** pages (`Wei Wang 0001`) are editor-verified: each is one real person. They are the
training labels.

`ml/` learns a pairwise model - given two papers carrying the same name, P(same person) - from
co-author overlap, venue, year, title and ORCID, then clusters a bin's papers with it and names the
numbered page each cluster resembles.

```bash
docker compose run --rm ml python -m ml.cli dataset    # build the labelled pair dataset, print its shape
docker compose run --rm ml python -m ml.cli train      # train, tune, evaluate, save model + metrics
docker compose run --rm ml python -m ml.cli predict --key homepages/35/7092
```

On the VM, swap in `-f docker-compose.prod.yml`. The job reads the api's serving tables read-only, so
it never blocks the dashboard or the analysis scripts. Artifacts land in `models/disambiguation-<dump
fingerprint>/`, so a model is always traceable to the dump it was trained on.

**How it is evaluated.** Splits are by *whole name block*, so no person and no block appears in both
training and test. Three numbers are reported, each against the co-author-overlap heuristic a
hand-written rule would use:

| Measure | Question |
|---|---|
| pairwise ROC-AUC / F1 | are two papers by the same person? |
| B-cubed F1, ARI | is the resulting clustering of a block right? |
| assignment top-1 | hold out a paper: does it land on the right person? |

`metrics.json` next to the model carries all of it, plus permutation feature importances, and the
`predict` output embeds the test metrics - a suggestion should never be shown without its accuracy.

## Co-author link prediction (ML)

Who will an author publish with next? `ml/ml/links/` ranks an author's **distance-2 neighbours** -
co-authors of co-authors who are not co-authors yet - by the probability of a joint paper within two
years (the framing of Liben-Nowell & Kleinberg). Labels are real and temporal: the graph as it stood
at the end of one year, and who actually collaborated in the two years after.

| Snapshot | Graph seen | Labels | Used for |
|---|---|---|---|
| T = last complete year − 4 (2021) | papers ≤ 2021 | new pairs 2022–2023 | training (anchors from hash buckets 0–7) |
| T = last complete year − 2 (2023) | papers ≤ 2023 | new pairs 2024–2025 | every reported number (buckets 8–9) |

Disambiguation bins are not nodes: a bin mixes hundreds of people and would be the best-connected
node in the graph. Features are the classic neighbourhood heuristics (common neighbours, Jaccard,
Adamic–Adar, resource allocation, preferential attachment) plus what they ignore: when the bridge
between the two people was last active, how active each of them is now, career age, shared venues.
The heuristics are also the baselines, scored on the same candidate sets.

```bash
docker compose run --rm ml python -m ml.links.cli graph     # the graph store for this dump (once)
docker compose run --rm ml python -m ml.links.cli train     # both snapshots, train, evaluate, save
docker compose run --rm ml python -m ml.links.cli predict --key homepages/s/JurgenSchmidhuber
```

The **graph store** (`models/links-graph-<dump fingerprint>.duckdb`: one row per author, co-author
and year, plus per-year degree, activity and venues) is built once per dump; training snapshots and
live suggestions both read it, so nothing is computed differently at serving time. Artifacts land in
`models/links-<fingerprint>/`.

**How it is evaluated.** Pooled ROC-AUC / average precision over every candidate pair, and per author
the ranking a user meets: MRR, Hits@10, Precision@5, Recall@10, over authors who did gain a new
distance-2 co-author. The job also reports **where new co-authors came from** at the snapshot
(distance 2 / farther / newcomers with no paper yet): the distance-2 share is the ceiling for any
local method, and the site says so. A calibration table (how often pairs in each score range became
co-authors) is stored with the model and shown next to every suggestion.

## Venue recommendation (ML)

Where would a paper with this title, by these authors, be published? `ml/ml/venues/` ranks the
journal and conference series for a title, and the same index answers "which dblp titles are
closest" - the search feature as a by-product of a supervised problem with real labels: every
paper's actual series.

Two stages, because a flat classifier over thousands of series and a million title tokens does not
fit a 4-thread VM. Three cheap scorers propose candidates and are also the baselines: **Naive
Bayes** and a **TF-IDF centroid** over title words and bigrams (content), and the **authors'
history** (where they published before this year). A gradient-boosting **ranker** orders the union
from those scores, the series' popularity and recency, and how many authors have history there.

| Papers of | Statistics from | Used for |
|---|---|---|
| last complete year − 2 (2023) | papers ≤ 2022 | training the ranker |
| last complete year (2025) | papers ≤ 2024 | every reported number |
| — | everything in the dump | serving (`stats.parquet` next to the model) |

No paper contributes to the statistics it is scored by. The class set at a snapshot is the series
with ≥ 100 papers and a paper in the last three years; the share of test papers whose venue is in
it is reported as the ceiling, as is the share whose venue the candidate stage found at all.

```bash
docker compose run --rm ml python -m ml.venues.cli store     # the venue store for this dump (once)
docker compose run --rm ml python -m ml.venues.cli train     # statistics, ranker, evaluation, serving stats
docker compose run --rm ml python -m ml.venues.cli predict --title "Graph neural networks for traffic forecasting"
docker compose run --rm ml python -m ml.venues.cli predict --key conf/nips/VaswaniSPUJGKP17
```

The **venue store** (`models/venues-store-<fingerprint>.duckdb`: eligible papers, the inverted
index of title tokens, each author's papers per series and year) is built once per dump. The
serving statistics are written as parquet into `models/venues-<fingerprint>/` by `train`, so a
retrain never contends with the server for a file lock.

**How it is evaluated.** Accuracy@1/3/5/10 and MRR over *all* sampled papers of the test year (a
venue outside the class set or outside the candidates is a miss), for the model and for each scorer
alone plus "most popular venue"; the same split by whether any author had a history. A calibration
table (how often a candidate at each score was the real venue) is shown next to every suggestion.

## Hybrid paper search (ML)

The plain paper search (`/api/papers/search`, still in `api/` as an always-available exact-word
fallback) only finds a title that contains every word you typed. `search/` upgrades the dashboard's
own search box: BM25 over title words and bigrams, fused by reciprocal rank with cosine similarity
over [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) embeddings, so a query
that paraphrases a title - different words, same meaning - can still rank it first.

```bash
docker compose run --rm search python -m search.cli store                 # the sparse index (~1 min)
docker compose run -d --name search-embed search python -m search.cli build-index   # embeddings: hours, resumable
docker compose run --rm search python -m search.cli evaluate              # self-retrieval check, feeds /api/search/status
docker compose run --rm search python -m search.cli search --q "graph neural networks for traffic forecasting"
```

**Scope.** The index covers the same population as venue recommendation: journal and conference
papers from 2010 on (~5.4M). Preprints, theses, books and older papers are not embedded; a query
still finds them through an exact-word match run inside the same request, so upgrading to hybrid
ranking never finds *fewer* papers, only ranks the indexed ones better.

**Building the embeddings is the slow part** - every eligible title, encoded on CPU, checkpointed
every few batches into `models/search-progress-<fingerprint>.duckdb`. Interrupting and re-running
the same command picks up where it left off; `python -m search.cli status` reports how much of the
index exists. The model itself is baked into the `search` image at build time (see
`search/Dockerfile`), so the VM never needs to reach Hugging Face.

**How it is evaluated.** There are no external relevance judgments, so `evaluate` measures something
concrete instead: for a sample of indexed papers, one distinctive title word is swapped for a
synonym (a real user rarely types a title's exact words), and the job checks whether the paper still
comes back. BM25 can partially recover through the words that were *not* swapped; only the
embeddings can recover through the swapped word itself - this is the specific case exact-word search
cannot handle by construction, and `/api/search/status` reports accuracy@1/5/10 and MRR for BM25
alone, embeddings alone, and the fused ranking, so the gain is a measured number, not a claim.

Served at `/api/search/papers` (a query, with `kind`/`from`/`to` filters) and `/api/search/status`.
Each result carries which signal(s) found it (`bm25`, `dense`, or `exact_word`); the Papers page
tags a result found by meaning alone.

## Ask dblp (the assistant)

`chat/` answers natural-language questions - "which author has the most papers?", "papers about
learning from few demonstrations", "does this include preprints?" - by **calling typed tools over the
dump**, not by retrieving text chunks. That choice is the whole design: a superlative, a count or a
trend lives in an `ORDER BY` over millions of rows, and no amount of embedding similarity will
produce it. Chunk retrieval is used for exactly one thing here, the questions about the data's own
definitions, where a few dozen documentation paragraphs are ranked by idf-weighted overlap.

Three retrieval paths, one router:

| Question | Path |
|---|---|
| counts, rankings, trends, entity facts, two-step joins | 20 typed SQL tools over the api's serving tables |
| "papers about ..." | the hybrid search service (BM25 + embeddings) |
| "what is a bin?", "how accurate is the model?" | the documentation corpus, and the live model cards |

```bash
docker compose run --rm chat python -m chat.cli tools                      # the catalogue
docker compose run --rm chat python -m chat.cli ask "who has the most papers?"
docker compose run --rm chat python -m chat.cli evaluate                   # the gold set
```

**How a question is answered.** One call to the model picks tools (several at once where they are
independent); they run in parallel against a read-only DuckDB cursor; the results go back; the answer
streams as server-sent events. Two model calls in the common case, three when a question needs a
second round. Rounds, tool calls and wall-clock are all capped, and a question that runs out of time
is answered with what is in hand rather than abandoned.

**Latency.** The tools are the fast part (5-80 ms; the serving tables are already built and cached),
so the budget goes to the model. The superlatives that would otherwise sort four million rows -
authors by papers, authors by co-authors, venues by size, the most-shared names - are precomputed once
per dump into `chat-store-<fingerprint>.duckdb`, and identical questions are answered from a disk
cache keyed on the dump, so a repeated demo costs nothing.

**Grounding, and what it will not do.** Every tool returns its own counting rules ("bins excluded",
"preprints excluded", "this counts every record type") and the system prompt requires them to be
carried into the answer. Nothing may be stated that did not come from a tool in that turn. Questions
that need citations, abstracts, affiliations, impact factors, awards or author demographics are
refused with a sentence about why, because dblp has none of those.

**The escape hatch is real but caged.** When no typed tool fits, the model may write one `SELECT`:
checked against a keyword denylist on the comment- and string-stripped statement, run on a read-only
connection with external access disabled, row-capped, interrupted after five seconds, and shown in the
UI next to the answer.

**How it is evaluated.** `chat/chat/goldset.py` holds ~37 questions across every class the taxonomy
covers, each naming the tools that must be involved; six of them are out of scope and must be refused.
`evaluate` reports tool-choice accuracy, refusal accuracy, the share of answers that were grounded in
at least one tool call, median latency and total cost. A chatbot without this file is a demo.

**Access and spend.** `CHAT_TOKEN` gates the page (unset in dev means open); a per-minute rate limit,
a daily request cap and a daily dollar ceiling all fail closed, and the ledger survives a restart.
`OPENAI_API_KEY` and `OPENAI_BASE_URL` select the provider - any OpenAI-compatible endpoint works,
and without a key the service starts and says it is not configured instead of failing.

## Run locally (Docker only)

```bash
docker compose --profile fixture run --rm fixture   # once: a small synthetic dump in api/fixture
docker compose up -d --build                        # -> http://localhost:8090
docker compose --profile dev up dev                 # hot-reload UI -> http://localhost:5173
```

Set `DBLP_DIR` to a real dblp folder (with `parquet/`, `eda_out/`, `charts/`) to run on the full dump.
API docs: `/api/docs`.

### Tests

```bash
docker build --target test -t dblp-api:test    api    && docker run --rm dblp-api:test
docker build --target test -t dblp-ml:test     ml     && docker run --rm dblp-ml:test
docker build --target test -t dblp-search:test search && docker run --rm dblp-search:test
docker build --target test -t dblp-chat:test   chat   && docker run --rm dblp-chat:test
```

The suite builds a synthetic dump with the exact 28-column schema of `parse_dblp.py` (bins, numbered
namesakes, name variants, unresolved names, withdrawn papers, preprint twins, every page format) and
exercises every endpoint, input validation, and rebuild-on-new-dump.

## CI/CD

`.github/workflows/deploy.yml`:

1. **test-api**, **test-ml**, **test-search** and **test-chat** run the suites above.
2. **build** builds both images; on `main` pushes `ghcr.io/hadigghazi/dblp-dashboard` and
   `ghcr.io/hadigghazi/dblp-dashboard-api` (tags `latest` and the commit SHA).
3. **deploy** (`main` only) SSHes into the VM, writes `~/dblp-dashboard/.env` on first run, pulls both
   images, restarts the stack and checks `http://127.0.0.1:8081/` and `/api/health`.

Secrets: `VM_HOST`, `VM_USER`, `VM_SSH_KEY`.

### On the VM

- `web` listens on `127.0.0.1:8081`. The Caddy proxy in `~/proxy` owns ports 80/443, serves
  `dblp.hadighazi.com` with a Let's Encrypt certificate it renews itself, and redirects plain HTTP on the
  bare IP to that address. Its config is kept in the automl-studio repo (`deploy/proxy/Caddyfile`).
- DNS: Cloudflare A record `dblp` → `34.89.182.10`, **DNS only**. The orange proxy would cut requests
  at 100 s, and the certificate lives on the VM.
- `api` has no published port; only `web` reaches it.
- `~/dblp-dashboard/.env`: `DBLP_DIR` (the analysis folder), `DATA_UID`/`DATA_GID` (its owner; the api
  runs as that user), optional `DUCKDB_MEMORY` (default 10GB) and `DUCKDB_THREADS` (default 4).
- The serving database lives in `~/dblp-dashboard/cache/` (a few GB).

## Layout

```
api/app/serving.py     serving-table build, rebuild watcher, connections
api/app/queries.py     every live query (each names the analysis script it mirrors)
api/app/snapshots.py   readers for the analysis jobs' output files
api/app/main.py        HTTP routes, result cache, concurrency limit
api/app/validate.py    live-vs-report comparison
api/tests/             synthetic dump + test suite
ml/ml/data.py          attaches the api's serving tables read-only
ml/ml/features.py      pair features (SQL) + matrix assembly
ml/ml/model.py         pairwise model, threshold tuning, artifacts
ml/ml/evaluate.py      held-out-by-block metrics + baselines
ml/ml/predict.py       split a disambiguation bin, suggest a person
ml/ml/server.py        HTTP front for both models (/ml/...)
ml/ml/links/graph.py   graph store, snapshots, distance-2 candidates + heuristics (SQL)
ml/ml/links/evaluate.py pooled + per-author ranking metrics, baselines, calibration
ml/ml/links/predict.py suggestions for one author, with the shared co-authors behind each
ml/ml/venues/store.py  venue store (papers, title-token index, author history), statistics as of a year
ml/ml/venues/candidates.py  Naive Bayes + centroid + history scoring, candidate pairs, related papers
ml/ml/venues/evaluate.py    accuracy@k / MRR over all test papers, per-scorer baselines, calibration
ml/ml/venues/predict.py     suggestions for a title or an existing paper, with the evidence
ml/tests/              synthetic serving db with planted signal (name blocks, bins, a link world with venue vocabularies)
search/search/store.py     sparse index (paper population, title-token inverted index) - self-contained
search/search/bm25.py      Okapi BM25 over the token index
search/search/embed.py     the embedding model, imported lazily (tests never need torch)
search/search/vectors.py   resumable memory-mapped embedding index, build + block-wise dense search
search/search/fuse.py      reciprocal rank fusion of the two rankings
search/search/evaluate.py  synonym-substitution self-retrieval check (BM25 vs. dense vs. fused)
search/search/search.py    orchestration + exact-word fallback for out-of-index records
search/tests/           synthetic serving db with per-topic title vocabularies + a fake encoder
chat/chat/tools.py      the tool catalogue: one typed, bounded question per tool
chat/chat/agent.py      the loop: pick tools, run them in parallel, stream the answer
chat/chat/sqlguard.py   the guarded ad-hoc SELECT (denylist, read-only, row cap, timeout)
chat/chat/docs.py       the documentation corpus for definitional questions
chat/chat/store.py      precomputed leaderboards, so a superlative is a lookup
chat/chat/goldset.py    the gold set: required tools per question, and what must be refused
chat/chat/budget.py     rate limits and the daily spend ledger
chat/tests/             a fixture whose answers are written down + a scripted model client
src/api.js             fetch hooks (loading, warming-up, errors)
src/pages.jsx          the eight findings pages + overview
src/explore.jsx        author, venue and paper explorers
src/charts.jsx         ECharts option builders
src/components.jsx     cards, filters, tables, badges
```
