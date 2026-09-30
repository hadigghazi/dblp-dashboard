"""
Publishing the dataset for download from the website.

The files are served by nginx straight off disk - a 500 MB download should be resumable and should
never pass through Python - so publishing is just a matter of putting the right files in the one
folder the web container can see, with a manifest the page reads to list them.

Four things this is careful about:

**Only what belongs in the dataset.** The web container mounts `models/downloads` and nothing else
under `models`, which also holds the assistant's logs and the model files. This copies in the graph,
the node file, the venue communities, the centrality table and the two datasheets, and nothing more.

**The table and the graph must be the same graph.** A centrality table measured on a different export
- another dump, another scope - would list plausible numbers for the wrong network. Publishing
checks the dump fingerprint and the node and edge counts agree before anything is copied.

**Nobody downloads a half-written file.** Each publish writes a new versioned folder, and a symlink is
switched to it in one atomic rename; a reader sees the old version or the new one, never a mixture.
Copies rather than hard links, because a later run that rewrites its output in place would otherwise
change a published file under somebody halfway through downloading it.

**Anyone can check what they got.** Every file's SHA-256 is computed while it is copied and published
beside it, in the manifest and in a standard `SHA256SUMS` file.
"""
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .. import config as base
from . import config

log = logging.getLogger("dblp.ml.network.publish")

SLUG = "dblp-coauthor"
CHECKSUMS = "SHA256SUMS.txt"   # the standard name, plus an extension so a browser shows it
CHUNK = 4 * 1024 * 1024


def files(name=None):
    """(published name, source: 'network' or 'centrality', source file, kind, what it is)."""
    name = name or config.NAME
    top = config.TOP_COMMUNITIES
    return [
        (f"{name}.ungraph.txt.gz", "network", f"{name}.ungraph.txt.gz", "data",
         "The graph: one co-authorship per line - FromNodeId, ToNodeId, PapersTogether"),
        (f"{name}.nodes.txt.gz", "network", f"{name}.nodes.txt.gz", "data",
         "Who each node is: dblp key, name, number of records, first and last year"),
        (f"{name}.centrality.tsv.gz", "centrality", f"{name}.centrality.tsv.gz", "data",
         "Degree, betweenness, closeness, eigenvector, PageRank and core number for every author, "
         "with their ranks"),
        (f"{name}.venues.cmty.txt.gz", "network", f"{name}.venues.cmty.txt.gz", "data",
         "Venues as ground-truth communities, as in SNAP's com-DBLP: one per line, its authors' ids"),
        (f"{name}.venues.top{top}.cmty.txt.gz", "network", f"{name}.venues.top{top}.cmty.txt.gz", "data",
         f"The {top:,} largest of those communities"),
        (f"{name}.venues.names.txt.gz", "network", f"{name}.venues.names.txt.gz", "data",
         "Which venue each community line is"),
        # the datasheets are published as readable pages, rendered from their Markdown
        ("datasheet-network.html", "network", "README.md", "doc",
         "About the graph: how it was built, and what was left out and why"),
        ("datasheet-centrality.html", "centrality", "README.md", "doc",
         "About the centrality table: how each measure was computed and checked"),
    ]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_same_graph(stats, metrics):
    """The centrality table has to describe this export, not some other one."""
    problems = []
    fp_graph = (stats.get("dump") or {}).get("fingerprint")
    fp_table = (metrics.get("dump") or {}).get("fingerprint")
    if fp_graph != fp_table:
        problems.append(f"the graph is from dump {fp_graph} but the centrality table from {fp_table}")
    if stats.get("nodes_with_an_edge") != metrics.get("nodes"):
        problems.append(f"the graph has {stats.get('nodes_with_an_edge')} authors with an edge but the "
                        f"table measured {metrics.get('nodes')}")
    if stats.get("edges") != metrics.get("edges"):
        problems.append(f"the graph has {stats.get('edges')} edges but the table measured "
                        f"{metrics.get('edges')}")
    if problems:
        raise ValueError("refusing to publish a centrality table for a different graph: "
                         + "; ".join(problems) + ". Re-run `centrality` on this export first.")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{ --ground:#f4f5f7; --surface:#fff; --ink:#12151a; --ink-2:#4c5361; --muted:#868d99;
  --rule:#e2e5ea; --rule-2:#edeff3; --accent-ink:#1c5cab; --code:#f6f7f9; }}
@media (prefers-color-scheme: dark) {{ :root {{ --ground:#0f1115; --surface:#171a20; --ink:#eef0f3;
  --ink-2:#b7bec9; --muted:#7c8492; --rule:#282e38; --rule-2:#20242c; --accent-ink:#8ab8f5; --code:#1b1f26; }} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font:15px/1.65 "IBM Plex Sans", system-ui, sans-serif; }}
main {{ max-width:860px; margin:0 auto; padding:28px 16px 64px; }}
.back {{ font-size:13px; color:var(--accent-ink); text-decoration:none; }}
.back:hover {{ text-decoration:underline; }}
article {{ background:var(--surface); border:1px solid var(--rule); border-radius:12px; padding:8px 28px 24px; margin-top:14px; }}
h1 {{ font-size:26px; line-height:1.25; margin:22px 0 6px; }}
h2 {{ font-size:18px; margin:30px 0 8px; padding-top:14px; border-top:1px solid var(--rule-2); }}
p, li {{ color:var(--ink-2); }}
strong {{ color:var(--ink); font-weight:600; }}
a {{ color:var(--accent-ink); }}
code, pre {{ font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:.88em; }}
code {{ background:var(--code); padding:1px 5px; border-radius:4px; }}
pre {{ background:var(--code); border:1px solid var(--rule-2); border-radius:8px; padding:12px 14px; overflow-x:auto; line-height:1.5; }}
pre code {{ background:none; padding:0; }}
.tablewrap {{ overflow-x:auto; margin:10px 0 14px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th, td {{ text-align:left; padding:6px 12px 6px 0; border-bottom:1px solid var(--rule-2); vertical-align:top; }}
th {{ color:var(--muted); font-weight:500; font-size:12.5px; }}
td {{ color:var(--ink-2); font-variant-numeric:tabular-nums; }}
@media (max-width:600px) {{ article {{ padding:4px 16px 18px; }} }}
</style>
</head>
<body>
<main>
<a class="back" href="/#network">&larr; The co-authorship network on dblp Explorer</a>
<article>
{body}
</article>
</main>
</body>
</html>
"""


def render_page(markdown_text):
    """A datasheet as a page a person can read, rather than Markdown source shown as plain text.
    Tables are wrapped so a wide one scrolls on a phone instead of stretching the page."""
    import markdown
    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    body = body.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")
    first = next((line[2:].strip() for line in markdown_text.splitlines() if line.startswith("# ")),
                 "dblp co-authorship network")
    return PAGE.format(title=f"{first} - datasheet", body=body)


def _write_and_hash(text: str, dst: Path):
    raw = text.encode("utf-8")
    dst.write_bytes(raw)
    os.chmod(dst, 0o644)
    return len(raw), hashlib.sha256(raw).hexdigest()


def _copy_and_hash(src: Path, dst: Path):
    digest, size = hashlib.sha256(), 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while True:
            block = fin.read(CHUNK)
            if not block:
                break
            digest.update(block)
            fout.write(block)
            size += len(block)
    os.chmod(dst, 0o644)            # nginx runs as its own user and only needs to read
    return size, digest.hexdigest()


def summary(stats, metrics):
    diameter = metrics.get("largest_component_diameter") or {}
    return {
        "authors_with_a_coauthor": stats.get("nodes_with_an_edge"),
        "author_pages": stats.get("author_pages"),
        "coauthorships": stats.get("edges"),
        "venue_communities": stats.get("communities"),
        "max_coauthors": stats.get("max_degree", metrics.get("max_degree")),
        "average_coauthors": metrics.get("average_degree"),
        "components": metrics.get("components"),
        "largest_component_share": metrics.get("largest_component_share"),
        "clustering_coefficient": metrics.get("approx_global_clustering_coefficient"),
        "diameter": diameter.get("upper"),
    }


def publish(export_dir, centrality_dir, root=None):
    """Put the dataset where the website serves it, and return the manifest the page lists."""
    export_dir, centrality_dir = Path(export_dir), Path(centrality_dir)
    root = Path(root or base.MODELS_DIR / "downloads")
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK):
        # Docker creates a missing bind-mount source as root; the deploy creates this folder first to
        # prevent that, but a manual `up` before any deploy could still have
        raise PermissionError(f"{root} is not writable by this user - it was probably created by Docker "
                              f"as root. On the VM: sudo chown $(id -u):$(id -g) ~/dblp-dashboard/models/downloads")
    os.chmod(root, 0o755)

    stats = _json(export_dir / "stats.json")
    metrics = _json(centrality_dir / "metrics.json")
    check_same_graph(stats, metrics)
    name = stats.get("name", config.NAME)
    sources = {"network": export_dir, "centrality": centrality_dir}
    missing = [str(sources[where] / src) for _, where, src, _, _ in files(name)
               if not (sources[where] / src).exists()]
    if missing:
        raise FileNotFoundError("cannot publish, these files are missing: " + ", ".join(missing))

    stamp = datetime.now(timezone.utc)
    # microseconds, so two publishes in the same second cannot collide
    version = f"{SLUG}-{stamp.strftime('%Y%m%dT%H%M%S.%fZ')}"
    for leftover in root.glob(f".{SLUG}-*.partial"):   # from a publish that died halfway
        shutil.rmtree(leftover, ignore_errors=True)
    staging = root / f".{version}.partial"
    staging.mkdir()
    try:
        published = []
        for out_name, where, src, kind, about in files(name):
            if out_name.endswith(".html"):
                page = render_page((sources[where] / src).read_text(encoding="utf-8"))
                size, sha = _write_and_hash(page, staging / out_name)
            else:
                size, sha = _copy_and_hash(sources[where] / src, staging / out_name)
            published.append({"name": out_name, "kind": kind, "bytes": size, "sha256": sha,
                              "description": about})
            log.info("published %s (%.1f MB)", out_name, size / 1e6)
        (staging / CHECKSUMS).write_text(
            "".join(f"{f['sha256']}  {f['name']}\n" for f in published), encoding="utf-8")
        manifest = {
            "name": SLUG,
            "title": "The dblp co-authorship network",
            "format": "SNAP-style gzipped text, as in com-DBLP",
            "license": "CC0 1.0, as dblp's own data - free to reuse, attribution appreciated",
            "source": "https://dblp.org",
            "dump": stats.get("dump", {}),
            "scope": (stats.get("rules") or {}).get("scope"),
            "graph_generated_at": stats.get("generated_at"),
            "centrality_generated_at": metrics.get("generated_at"),
            "published_at": stamp.isoformat(timespec="seconds"),
            "summary": summary(stats, metrics),
            "total_bytes": sum(f["bytes"] for f in published if f["kind"] == "data"),
            "checksums": CHECKSUMS,
            "files": published,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for extra in (CHECKSUMS, "manifest.json"):
            os.chmod(staging / extra, 0o644)
        os.chmod(staging, 0o755)

        final = root / version
        os.rename(staging, final)
        # the switch: a new link, renamed over the old one, is a single atomic step. Relative, so it
        # resolves the same inside the web container, where this folder is mounted somewhere else.
        link, tmp_link = root / SLUG, root / f".{SLUG}.link"
        if tmp_link.is_symlink() or tmp_link.exists():
            tmp_link.unlink()
        os.symlink(version, tmp_link)
        os.replace(tmp_link, link)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # older versions go; a download already in progress keeps reading from its open file
    for old in root.glob(f"{SLUG}-*"):
        if old.is_dir() and old.name != version:
            shutil.rmtree(old, ignore_errors=True)
    log.info("published %s: %d files, %.0f MB, served at /downloads/%s/", version, len(published),
             manifest["total_bytes"] / 1e6, SLUG)
    return manifest
