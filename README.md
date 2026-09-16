# dblp Explorer

Interactive dashboard over an exploratory analysis of the complete [dblp](https://dblp.org) dump
(release 2026-09-01, 12.9M records): publishing trends, author identity, the co-authorship network,
titles, venues, data quality and an OpenAlex check. Nine pages, every chart filterable.

Vite + React 19 + Apache ECharts. Data is the set of pre-aggregated CSVs behind the analysis's
charts, bundled as `src/data.json` (no backend needed).

## Run locally (Docker, nothing installed on the host)

```bash
docker compose up -d --build            # production build       -> http://localhost:8090
docker compose --profile dev up dev     # hot-reload dev server  -> http://localhost:5173
docker compose down
```

## Refresh the data

`src/data.json` is generated from the analysis's chart CSVs:

```bash
python prep_data.py     # reads the 27 CSVs, writes src/data.json
```

## CI/CD

`.github/workflows/deploy.yml` runs on every push:

1. **build** — builds the Docker image (fails the run on any compile error). On `main` it also
   pushes the image to `ghcr.io/hadigghazi/dblp-dashboard` tagged `latest` and with the commit SHA.
2. **deploy** (`main` only) — SSHes into the VM, logs into GHCR with the run's own token, pulls
   the image and restarts the container via `docker-compose.prod.yml`, then curls the site.
   Skips (without failing) until the `VM_HOST` secret exists.

### Repository secrets

| Secret       | Value                                                     |
|--------------|-----------------------------------------------------------|
| `VM_HOST`    | the VM's external IP (a static one, or deploys break on restart) |
| `VM_USER`    | the Linux user on the VM (e.g. `hadi_devancy`)            |
| `VM_SSH_KEY` | private half of a dedicated deploy keypair (ed25519)      |

### One-time VM setup

```bash
# Docker (official repo), then let your user run it without sudo
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER" && newgrp docker

# Authorise the deploy key GitHub Actions will use
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "<contents of the deploy public key>" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
```

Open port 80 on the GCP firewall and give the instance a static IP (from Cloud Shell):

```bash
gcloud compute firewall-rules create allow-http-80 --network default --allow tcp:80 --target-tags http-server
gcloud compute instances add-tags dblp-vm --zone europe-west3-c --tags http-server
gcloud compute addresses create dblp-vm-ip --region europe-west3
gcloud compute instances delete-access-config dblp-vm --zone europe-west3-c --access-config-name "external-nat"
gcloud compute instances add-access-config dblp-vm --zone europe-west3-c --address "$(gcloud compute addresses describe dblp-vm-ip --region europe-west3 --format='get(address)')"
```

After that, every push to `main` deploys.

## Layout

```
src/data.json        the 27 datasets
src/charts.jsx       ECharts option builders + <Chart> wrapper + theme handling
src/components.jsx   KPI tiles, cards, filter chips, sliders, sortable table, heatmap
src/pages.jsx        the 9 pages
src/App.jsx          sidebar, hash routing, light/dark/system toggle
Dockerfile           node:22-alpine build -> nginx:1.27-alpine serve
docker-compose.yml   local: web (:8090) + optional dev profile (:5173)
docker-compose.prod.yml   VM: pulls the GHCR image, serves on :80
```
