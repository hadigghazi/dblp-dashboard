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
src/api.js             fetch hooks (loading, warming-up, errors)
src/pages.jsx          the eight findings pages + overview
src/explore.jsx        author, venue and paper explorers
src/charts.jsx         ECharts option builders
src/components.jsx     cards, filters, tables, badges
```
