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

# Below this, the best candidate is not merely unproven but implausible: with hundreds of numbered
# pages in a block, something always scores highest, so "no candidate" has to mean "nothing close".
NEW_PERSON_CEILING = 0.15


def _suggestion(ranked, max_links, assign_threshold):
    """
    Three outcomes, not two:
      accepted  - confident, clearly ahead of the runner-up, an outlier against the whole candidate
                  pool, and backed by at least one strong pairwise link: name it.
      uncertain - plausible but unproven: show it as a candidate, not an answer.
      new       - nothing comes close: this person most likely has no page yet.
    ranked is [(person_id, mean score)] best first; max_links maps person_id to the strongest single
    pairwise probability between the cluster and that person's papers.
    """
    if not ranked:
        return None, None
    best_id, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    decision = {"score": best_score, "margin": best_score - runner_up,
                "z": E.outlier_z(best_score, [sc for _, sc in ranked[1:]]),
                "max_link": max_links.get(best_id, 0.0)}
    entry = {"person_id": best_id, "score": round(best_score, 3), "margin": round(decision["margin"], 3),
             "z": None if decision["z"] == float("inf") else round(decision["z"], 1),
             "max_link": round(decision["max_link"], 3), "candidates": len(ranked)}
    if best_score < NEW_PERSON_CEILING:
        return None, None
    if not E.accept(decision, assign_threshold):
        return None, entry
    return entry, None


STRONG_MERGE = 0.6   # every member of a merged group must match the person at least this well


def merge_by_suggestion(clusters):
    """
    Fold together clusters whose accepted suggestion is the same person - but only when each of
    them matches that person strongly on its own. Two independent strong matches to one page are
    good evidence the groups are one person; a strong match plus a marginal one is not, and merging
    them let a barely-accepted group borrow the credibility of a confident one (the merged group
    used to report the best member's score). Weaker groups stay separate, still labelled with the
    page they point at, so a reader sees several groups naming one person and can judge.
    """
    merged, by_person = [], {}
    for c in clusters:
        sug = c.get("suggested_person")
        person = (sug or {}).get("key")
        if person is None or sug["score"] < STRONG_MERGE:
            merged.append(c)
            continue
        if person in by_person:
            into = by_person[person]
            into["papers"].extend(c["papers"])
            into["size"] += c["size"]
            into["merged_from"] = into.get("merged_from", 1) + 1
            prev_weakest = into["suggested_person"].get("weakest_member_score", into["suggested_person"]["score"])
            into["suggested_person"]["weakest_member_score"] = min(prev_weakest, sug["score"])
            into["suggested_person"]["score"] = max(into["suggested_person"]["score"], sug["score"])
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
    # the cut and linkage tuned on bin-like blocks, not on the labelled ones: a bin is mostly
    # people with a single paper, where average linkage chains unrelated work together
    cluster_t = thresholds.get("cluster_bin", thresholds["cluster"])
    linkage = thresholds.get("linkage_bin", "average")
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

    F.build_pairs(con, sampled=False, touching=[person["person_id"]])
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

    labels = E.cluster(prob, cluster_t, linkage)
    meta = con.execute(
        "SELECT pid, key, title, year, venue FROM s.pubs WHERE pid IN (SELECT pid FROM inst_bin)").fetchall()
    papers = {int(r[0]): {"key": r[1], "title": r[2], "year": r[3], "venue": r[4]} for r in meta}
    known_by_id = {k["person_id"]: k for k in known}

    clusters = []
    for label in sorted(set(labels.tolist())):
        members = [bin_pids[i] for i in np.nonzero(labels == label)[0]]
        scores, strongest = {}, {}
        for pid in members:
            for person_id, probs in to_known.get(pid, {}).items():
                scores.setdefault(person_id, []).append(float(np.mean(probs)))
                strongest[person_id] = max(strongest.get(person_id, 0.0), max(probs))
        ranked = sorted(((pid, float(np.mean(v))) for pid, v in scores.items()), key=lambda t: -t[1])
        accepted, rejected = _suggestion(ranked, strongest, assign_t)

        def name_of(entry):
            if entry is None:
                return None
            who = known_by_id.get(entry["person_id"], {})
            return {"key": who.get("key"), "name": who.get("name"), "score": entry["score"],
                    "margin": entry["margin"], "z": entry["z"], "max_link": entry["max_link"],
                    "candidates": entry["candidates"]}

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
    groups_per_person = {}
    for c in named:
        groups_per_person[c["suggested_person"]["key"]] = groups_per_person.get(c["suggested_person"]["key"], 0) + 1
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
            # a page named by several separate groups: either over-splitting of one person, or
            # the page itself mixes people; worth a human look either way
            "pages_named_by_several_groups": sorted(
                [{"key": k, "groups": n} for k, n in groups_per_person.items() if n > 1],
                key=lambda r: -r["groups"]),
        },
        "thresholds": {"cluster": cluster_t, "linkage": linkage, "assign": assign_t,
                       "min_margin": E.MIN_MARGIN, "min_z": E.ASSIGN_Z, "min_max_link": E.MIN_MAX_LINK,
                       "new_person_ceiling": NEW_PERSON_CEILING},
    }
    if model_dir is not None:
        metrics = model_dir / M.METRICS_FILE
        if metrics.exists():
            saved = json.loads(metrics.read_text(encoding="utf-8"))
            out["model"] = {"trained_at": saved.get("trained_at"), "dump": saved.get("dump"),
                            "test_metrics": saved.get("metrics", {}).get("test")}
    return out
