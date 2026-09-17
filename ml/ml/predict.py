"""
Splitting a disambiguation bin.

Given a bin (a bare name like "Wei Wang" holding papers by many people), cluster its papers and, for
each cluster, name the numbered page it most resembles. Three rules keep the output honest:

  * the clustering cut and the "show a name" cut are different numbers (see evaluate.py);
  * a name is only shown if it also beats the runner-up by a margin, so near-ties stay silent;
  * clusters that point at the same person are merged, because the clustering errs on the side of
    splitting one person into several groups.

The output always carries the model's measured accuracy, so a caller can present a suggestion as a
suggestion.
"""
import json
import logging

import numpy as np

from . import data, evaluate as E, features as F, model as M

log = logging.getLogger("dblp.ml.predict")

MAX_BIN_PAPERS = 300
KNOWN_PAPERS_PER_PERSON = 8


def _suggestion(ranked, assign_threshold):
    """The best candidate, but only when it is both confident enough and clearly ahead."""
    if not ranked:
        return None, None
    best_id, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - runner_up
    if best_score < assign_threshold or margin < E.MIN_MARGIN:
        return None, {"person_id": best_id, "score": round(best_score, 3), "margin": round(margin, 3)}
    return {"person_id": best_id, "score": round(best_score, 3), "margin": round(margin, 3)}, None


def merge_by_suggestion(clusters):
    """
    Fold together clusters whose accepted suggestion is the same person. The clustering produces
    about 30% more groups than there are people (measured), so one person routinely appears as
    several groups; if they all point at the same page, that page is the answer.
    """
    merged, by_person = [], {}
    for c in clusters:
        person = (c.get("suggested_person") or {}).get("key")
        if person is None:
            merged.append(c)
            continue
        if person in by_person:
            into = by_person[person]
            into["papers"].extend(c["papers"])
            into["size"] += c["size"]
            into["merged_from"] = into.get("merged_from", 1) + 1
            best = max(into["suggested_person"]["score"], c["suggested_person"]["score"])
            into["suggested_person"]["score"] = best
        else:
            by_person[person] = c
            merged.append(c)
    for c in merged:
        c["papers"].sort(key=lambda p: -(p["year"] or 0))
    return merged


def split_bin(con, key, model=None, thresholds=None, features=None, model_dir=None,
              max_papers=MAX_BIN_PAPERS):
    person = data.person_by_key(con, key)
    if person is None:
        return {"error": f"no author page {key}"}
    if person["page_kind"] != "disambiguation":
        return {"error": f"{key} is a {person['page_kind']} page, not a disambiguation bin"}
    if model is None:
        model, thresholds, features, model_dir = M.load()
    cluster_t = thresholds["cluster"]
    assign_t = thresholds.get("assign", cluster_t)

    known = data.numbered_in_block(con, person["base_name"])

    data.build_instances(con, [person["person_id"]], labelled=False, cap_per_person=max_papers)
    con.execute("CREATE OR REPLACE TEMP TABLE inst_bin AS SELECT * FROM inst")
    if known:
        data.build_instances(con, [k["person_id"] for k in known], labelled=False,
                             cap_per_person=KNOWN_PAPERS_PER_PERSON)
        con.execute("CREATE OR REPLACE TEMP TABLE inst_known AS SELECT * FROM inst")
    else:
        con.execute("CREATE OR REPLACE TEMP TABLE inst_known AS SELECT * FROM inst_bin WHERE false")
    con.execute("CREATE OR REPLACE TEMP TABLE inst AS "
                "SELECT * FROM inst_bin UNION ALL SELECT * FROM inst_known")

    n_bin = con.execute("SELECT count(*) FROM inst_bin").fetchone()[0]
    if n_bin < 2:
        return {"bin": person, "papers": n_bin, "clusters": [], "note": "too few papers on this bin to split"}

    F.build_pairs(con, sampled=False)
    X, _, info = F.matrix(con, feature_names=features)
    if not len(info):
        return {"bin": person, "papers": n_bin, "clusters": [], "note": "no comparable pairs"}
    p = model.predict_proba(X)[:, 1]

    bin_pids = [int(r[0]) for r in con.execute("SELECT pid FROM inst_bin ORDER BY pid").fetchall()]
    owner = dict(con.execute("SELECT pid, person_id FROM inst_known").fetchall())
    idx = {pid: i for i, pid in enumerate(bin_pids)}
    prob = np.zeros((len(bin_pids), len(bin_pids)), dtype=np.float32)
    to_known = {}   # bin pid -> {person_id: [probabilities]}
    for a, b, pr in zip(info["pid_a"], info["pid_b"], p):
        a, b = int(a), int(b)
        if a in idx and b in idx:
            prob[idx[a], idx[b]] = prob[idx[b], idx[a]] = pr
        else:
            bin_pid, known_pid = (a, b) if a in idx else (b, a)
            if bin_pid in idx and known_pid in owner:
                to_known.setdefault(bin_pid, {}).setdefault(owner[known_pid], []).append(float(pr))

    labels = E.cluster(prob, cluster_t)
    meta = con.execute(
        "SELECT pid, key, title, year, venue FROM s.pubs WHERE pid IN (SELECT pid FROM inst_bin)").fetchall()
    papers = {int(r[0]): {"key": r[1], "title": r[2], "year": r[3], "venue": r[4]} for r in meta}
    known_by_id = {k["person_id"]: k for k in known}

    clusters = []
    for label in sorted(set(labels.tolist())):
        members = [bin_pids[i] for i in np.nonzero(labels == label)[0]]
        scores = {}
        for pid in members:
            for person_id, probs in to_known.get(pid, {}).items():
                scores.setdefault(person_id, []).append(float(np.mean(probs)))
        ranked = sorted(((pid, float(np.mean(v))) for pid, v in scores.items()), key=lambda t: -t[1])
        accepted, rejected = _suggestion(ranked, assign_t)

        def name_of(entry):
            if entry is None:
                return None
            who = known_by_id.get(entry["person_id"], {})
            return {"key": who.get("key"), "name": who.get("name"),
                    "score": entry["score"], "margin": entry["margin"]}

        clusters.append({
            "size": len(members),
            "papers": [papers[pid] for pid in members],
            "suggested_person": name_of(accepted),
            "best_candidate_below_threshold": name_of(rejected),
            "looks_new": accepted is None and rejected is None,
        })

    clusters = merge_by_suggestion(clusters)
    clusters.sort(key=lambda c: -c["size"])

    named = [c for c in clusters if c["suggested_person"]]
    out = {
        "bin": {"key": person["key"], "name": person["name"]},
        "papers": n_bin,
        "numbered_people_in_block": len(known),
        "clusters": clusters,
        "summary": {
            "clusters": len(clusters),
            "matched_to_a_numbered_page": len(named),
            "uncertain": sum(1 for c in clusters if c["best_candidate_below_threshold"]),
            "look_new": sum(1 for c in clusters if c["looks_new"]),
            "papers_matched": sum(c["size"] for c in named),
        },
        "thresholds": {"cluster": cluster_t, "assign": assign_t, "min_margin": E.MIN_MARGIN},
    }
    if model_dir is not None:
        metrics = model_dir / M.METRICS_FILE
        if metrics.exists():
            saved = json.loads(metrics.read_text(encoding="utf-8"))
            out["model"] = {"trained_at": saved.get("trained_at"), "dump": saved.get("dump"),
                            "test_metrics": saved.get("metrics", {}).get("test")}
    return out
