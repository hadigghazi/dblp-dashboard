"""
Results of the analysis jobs that cannot run per request (the igraph network over 26M edges, the
OpenAlex sample, the power-law fits). Read from the jobs' latest text output on every request, so
re-running a job updates the dashboard; each payload says which file it came from and when.
"""
import re
from datetime import datetime, timezone

from . import config


def _num(v):
    if v is None:
        return None
    s = str(v).replace(",", "").replace("%", "").strip()
    try:
        f = float(s)
    except ValueError:
        return v
    return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f


def _source(path):
    return {"file": path.name, "modified": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            .isoformat(timespec="seconds")}


def read_sections(filename):
    """Parse every '=== title' table of a job output file: {title: [row dicts]} plus loose text lines."""
    path = config.EDA_OUT / filename
    if not path.exists():
        return None, None, None
    lines = path.read_text(encoding="utf-8").splitlines()
    sections, notes = {}, {}
    for i, line in enumerate(lines):
        if not line.startswith("=== "):
            continue
        title = line[4:].strip()
        rows, extra = [], []
        # a table is: header line, dashes line, rows until a blank line
        if i + 2 < len(lines) and set(lines[i + 2].replace(" ", "")) == {"-"}:
            header = re.split(r"\s{2,}", lines[i + 1].strip())
            for row in lines[i + 3:]:
                if not row.strip() or row.startswith("=== "):
                    break
                rows.append(dict(zip(header, re.split(r"\s{2,}", row.strip()))))
        # free text right after the table (e.g. "mean distance 5.63, median 6, ...")
        for row in lines[i + 1 + (len(rows) + 2 if rows else 0):]:
            if row.startswith("=== "):
                break
            if row.strip() and not row.startswith("["):
                extra.append(row.strip())
        sections[title] = rows
        notes[title] = extra
    return sections, notes, _source(path)


def _find(sections, start):
    for title, rows in sections.items():
        if title.startswith(start):
            return rows
    return []


def _find_note(notes, start):
    for title, lines in notes.items():
        if title.startswith(start):
            return lines
    return []


def network():
    sections, notes, source = read_sections("02_network.txt")
    if sections is None:
        return {"available": False, "reason": "02_network.txt not found; run eda_02_network.py"}
    overview = {r["graph"]: {k: _num(v) for k, v in r.items() if k != "graph"}
                for r in _find(sections, "Network overview")}
    structure = {r["measure"]: _num(r["value"]) for r in _find(sections, "Structure of the giant component")}
    communities = {r["measure"]: _num(r["value"]) for r in _find(sections, "Communities in the giant component")}
    dist_rows = _find(sections, "Distances between authors")
    dist_note = " ".join(_find_note(notes, "Distances between authors"))
    m = re.search(r"mean distance ([\d.]+), median (\d+), 90th percentile (\d+)", dist_note)
    top = []
    for r in _find(sections, "What the 10 largest communities"):
        venues = [{"name": v.group(1).strip(), "papers": int(v.group(2))}
                  for v in re.finditer(r"([^,()]+?)\s*\((\d+)\)", r.get("top_venues", ""))]
        top.append({"authors": _num(r["authors"]), "venues": venues})
    return {
        "available": True,
        "source": source,
        "overview": overview,
        "structure": structure,
        "communities": communities,
        "edge_weights": [{k: _num(v) for k, v in r.items()} for r in _find(sections, "Edge weights")],
        "components": [{k: _num(v) for k, v in r.items()} for r in _find(sections, "Component sizes")],
        "most_connected": [{k: _num(v) for k, v in r.items()} for r in _find(sections, "Most-connected authors")],
        "distances": [{"hops": _num(r["hops"]), "pct_of_pairs": _num(r["pct_of_pairs"]),
                       "cumulative_pct": _num(r.get("cumulative_pct"))} for r in dist_rows],
        "distance_summary": ({"mean": float(m.group(1)), "median": int(m.group(2)), "p90": int(m.group(3))}
                             if m else None),
        "largest_communities": top,
        "growth": [{k: _num(v) for k, v in r.items()} for r in _find(sections, "Network up to each year")],
    }


def openalex():
    sections, _, source = read_sections("06_openalex.txt")
    if sections is None:
        return {"available": False, "reason": "06_openalex.txt not found; run eda_06_openalex.py"}
    match = {r["kind"]: {k: _num(v) for k, v in r.items() if k != "kind"}
             for r in _find(sections, "Match rate and agreement")}
    add = [{k: _num(v) for k, v in r.items()} for r in _find(sections, "What OpenAlex would add")]
    return {
        "available": True,
        "source": source,
        "match": match,
        "adds": add,
        "open_access": [{k: _num(v) for k, v in r.items()} for r in _find(sections, "Open access: dblp flag")],
        "fields": [{"field": r["value"], "works": _num(r["works"]), "pct": _num(r["pct"])}
                   for r in _find(sections, "OpenAlex research field")],
        "countries": [{k: _num(v) for k, v in r.items()} for r in _find(sections, "Countries")],
    }


def power_law_fits():
    sections, _, source = read_sections("07_statistics.txt")
    if sections is None:
        return {"available": False}
    fits = {}
    for r in _find(sections, "Power-law fits"):
        alpha = re.match(r"([\d.]+)\s*±\s*([\d.]+)", r.get("alpha", ""))
        fits[r["distribution"]] = {
            "xmin": _num(r.get("xmin")),
            "alpha": float(alpha.group(1)) if alpha else None,
            "alpha_se": float(alpha.group(2)) if alpha else None,
            "share_in_tail": r.get("share_in_tail"),
            "fit_sample": _num(r.get("fit_sample")),
        }
    for r in _find(sections, "Is the tail a power law"):
        fits.setdefault(r["distribution"], {})["comparisons"] = {
            k.replace("vs_", ""): v for k, v in r.items() if k.startswith("vs_")}
    return {"available": True, "source": source, "fits": fits}
