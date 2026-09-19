# dblp Explorer

Interactive dashboard over the complete [dblp](https://dblp.org) dump, queried live: publishing trends,
author identity, the co-authorship network, titles, venues, data quality and an OpenAlex check, plus
explorers for any author, venue series or publication.

**https://dblp.hadighazi.com**

```
browser ──► Caddy (TLS) ──► web (nginx + React/ECharts) ──/api/──► api (FastAPI + DuckDB) ──► ~/dblp (read-only)
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
docker build --target test -t dblp-api:test api && docker run --rm dblp-api:test
docker build --target test -t dblp-ml:test  ml  && docker run --rm dblp-ml:test
```

The suite builds a synthetic dump with the exact 28-column schema of `parse_dblp.py` (bins, numbered
namesakes, name variants, unresolved names, withdrawn papers, preprint twins, every page format) and
exercises every endpoint, input validation, and rebuild-on-new-dump.

## CI/CD

`.github/workflows/deploy.yml`:

1. **test-api** runs the suite above.
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
src/api.js             fetch hooks (loading, warming-up, errors)
src/pages.jsx          the eight findings pages + overview
src/explore.jsx        author, venue and paper explorers
src/charts.jsx         ECharts option builders
src/components.jsx     cards, filters, tables, badges
```
