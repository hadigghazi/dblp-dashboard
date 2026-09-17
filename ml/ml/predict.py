"""
Splitting a disambiguation bin.

Given a bin (a bare name like "Wei Wang" holding papers by many people), cluster its papers and, for
each cluster, name the numbered page it most resembles. The output always carries the model's measured
accuracy so a caller can present a suggestion as a suggestion.
"""
import json
import logging

import numpy as np

from . import data, evaluate, features as F, model as M

log = logging.getLogger("dblp.ml.predict")

MAX_BIN_PAPERS = 300
KNOWN_PAPERS_PER_PERSON = 8


def split_bin(con, key, model=None, threshold=None, model_dir=None, max_papers=MAX_BIN_PAPERS):
    person = data.person_by_key(con, key)
    if person is None:
        return {"error": f"no author page {key}"}
    if person["page_kind"] != "disambiguation":
        return {"error": f"{key} is a {person['page_kind']} page, not a disambiguation bin"}
    if model is None:
        model, threshold, model_dir = M.load()

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
        return {"bin": person, "papers": n_bin, "clusters": [],
                "note": "too few papers on this bin to split"}

    F.build_pairs(con, sampled=False)
    X, _, info = F.matrix(con)
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

    labels = evaluate._cluster(prob, threshold)
    meta = con.execute("""
        SELECT pid, key, title, year, venue FROM s.pubs WHERE pid IN (SELECT pid FROM inst_bin)
    """).fetchall()
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
        suggestion = None
        if ranked and ranked[0][1] >= threshold:
            best = known_by_id.get(ranked[0][0], {})
            suggestion = {"key": best.get("key"), "name": best.get("name"),
                          "score": round(ranked[0][1], 3)}
        clusters.append({
            "size": len(members),
            "papers": [papers[pid] for pid in sorted(members, key=lambda x: -(papers[x]["year"] or 0))],
            "suggested_person": suggestion,
            "looks_new": suggestion is None,
            "runner_up": ({"key": known_by_id.get(ranked[1][0], {}).get("key"),
                           "score": round(ranked[1][1], 3)} if len(ranked) > 1 else None),
        })
    clusters.sort(key=lambda c: -c["size"])

    out = {
        "bin": {"key": person["key"], "name": person["name"]},
        "papers": n_bin,
        "numbered_people_in_block": len(known),
        "clusters": clusters,
        "threshold": threshold,
    }
    if model_dir is not None:
        metrics = model_dir / M.METRICS_FILE
        if metrics.exists():
            saved = json.loads(metrics.read_text(encoding="utf-8"))
            out["model"] = {"trained_at": saved.get("trained_at"), "dump": saved.get("dump"),
                            "test_metrics": saved.get("metrics", {}).get("test")}
    return out
