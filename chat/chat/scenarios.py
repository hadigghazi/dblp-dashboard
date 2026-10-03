"""
Scenario suites: what one particular person is likely to ask, run end to end and printed in full.

The gold set checks the assistant across the whole catalogue. A scenario suite is narrower and more
demanding: one subject - here the course instructor, whose dblp page is "Hussein Hazimeh 0002" and
who shares the name with another researcher - asked about in every way a person tries out a chatbot
on themselves. Plain questions, the affiliation instead of the number, a misspelling, Arabic,
follow-ups that point back ("the second one"), questions dblp cannot answer.

Two things decide whether a scenario passes, beyond what the gold set already checks:

  * **The right person.** If any lookup in the answer used the *other* Hussein Hazimeh's page, the
    answer is about the wrong person however fluent it reads - the failure that matters most when the
    person asking is the subject. Pages are found at run time by name, so nothing here hard-codes a
    key that a new dump could change.
  * **It actually worked.** Tools in `succeed` must have returned a result, not a refusal.

Every answer is printed in full, with the calls behind it, because the point of a suite like this is
for a person to read what the subject will read.
"""
import json
import time
from datetime import datetime, timezone

from . import config, evaluate as E

AUTHOR_KEY_ARGS = ("key", "key_a", "key_b", "author_key")

SUITES = {
    "instructor": {
        "subject": "Hussein Hazimeh 0002",
        "why": "the course instructor; another researcher has a dblp page under the same name",
        # scenarios that name the affiliation assume it is on the subject's page; checked before running
        "affiliation_hint": "Arab Open University",
        "cases": [
            # ---- finding him
            dict(q="How many papers does Hussein Hazimeh have?", all_of=["resolve_author"],
                 note="ambiguous on purpose: a good answer says two people share the name"),
            dict(q="How many papers does Hussein Hazimeh 0002 have?",
                 any_of=["resolve_author", "author_profile", "count_papers"], subject=True),
            dict(q="How many papers does Dr. Hussein Hazimeh from Arab Open University have?",
                 all_of=["resolve_author"], subject=True),
            dict(q="Tell me about Hussein Hazimeh from Lebanon", any_of=["author_profile"],
                 succeed=["author_profile"], subject=True),
            dict(q="Husein Hazime publications", all_of=["resolve_author"],
                 note="misspelled: the resolver falls back to the closest names"),
            dict(q="Who is Hazimeh in dblp?", all_of=["resolve_author"], note="surname only"),
            dict(q="كم عدد الأوراق المنشورة لحسين حزيمة؟", all_of=["resolve_author"],
                 note="Arabic: dblp names are in Latin script, so the name has to be transliterated"),
            dict(q="How many papers does my instructor Hussein Hazimeh have?", all_of=["resolve_author"],
                 note="'my instructor' cannot pick a page: it should say which one it used, or ask"),

            # ---- about him
            dict(q="What are Hussein Hazimeh 0002's most recent papers?", any_of=["author_papers"],
                 succeed=["author_papers"], subject=True),
            dict(q="Who does Hussein Hazimeh from Arab Open University work with most?",
                 any_of=["coauthors", "author_profile"], subject=True),
            dict(q="Which venues does Hussein Hazimeh 0002 publish in?", any_of=["author_profile"],
                 succeed=["author_profile"], subject=True),
            dict(q="When did Hussein Hazimeh 0002 start publishing, and how active has he been since?",
                 any_of=["author_profile", "author_papers", "count_papers"], subject=True),
            dict(q="How many journal papers has Hussein Hazimeh 0002 published since 2020?",
                 any_of=["count_papers", "author_papers"], subject=True),
            dict(q="How central is Hussein Hazimeh 0002 in the co-authorship network?",
                 any_of=["author_centrality"], succeed=["author_centrality"], subject=True),
            dict(q="Who is Hussein Hazimeh 0002 likely to collaborate with next?",
                 any_of=["predict_coauthors"], subject=True),
            dict(q="Show me Hussein Hazimeh 0002's papers about deep learning",
                 any_of=["author_papers", "search_papers"], subject=True,
                 note="no tool filters one author's papers by topic: it should say how it chose"),
            dict(q="Has Hussein Hazimeh 0002 ever published in IEEE Access?",
                 any_of=["author_papers", "author_profile", "count_papers"], subject=True),

            # ---- follow-ups that point back
            dict(turns=["How many papers does Hussein Hazimeh have?",
                        "The second one - give me some details about him"],
                 all_of=["author_profile"], succeed=["author_profile"], subject=True,
                 note="'second' is the order of the first answer, which lists the larger page first"),
            dict(turns=["How many papers does Hussein Hazimeh have?",
                        "The one at Arab Open University - who are his co-authors?"],
                 any_of=["coauthors", "author_profile"], subject=True),
            dict(turns=["Tell me about Hussein Hazimeh 0002", "What about his co-authors?"],
                 any_of=["coauthors", "author_profile"], subject=True),
            dict(turns=["Tell me about Hussein Hazimeh 0002", "Only his journal papers since 2020, please"],
                 any_of=["author_papers", "count_papers"], subject=True),
            dict(turns=["Who are Hussein Hazimeh 0002's co-authors?",
                        "Has he written anything with the first one since 2022?"],
                 any_of=["pair_papers", "author_papers"], subject=True),
            dict(turns=["How many papers does Hussein Hazimeh have?", "Compare the two of them"],
                 any_of=["author_profile", "resolve_author"],
                 note="both pages, side by side; no single subject"),

            # ---- things dblp cannot answer about him
            dict(q="What is Hussein Hazimeh 0002's h-index?", refuses=True),
            dict(q="How many citations does Hussein Hazimeh 0002 have?", refuses=True),
            dict(q="Which courses does Hussein Hazimeh teach?", refuses=True),
            dict(q="What is Hussein Hazimeh's email address?", refuses=True),
        ],
    },
}


def pages(ctx, subject_name):
    """Every page sharing the subject's base name: the subject's own, and the ones that would make an
    answer about the wrong person."""
    cur = ctx.cursor()
    row = cur.execute("SELECT base_name FROM s.persons WHERE name = ?", [subject_name]).fetchone()
    if not row:
        return None, []
    rows = cur.execute("""
        SELECT p.key, p.name, p.page_kind, coalesce(ps.n_pubs, 0) AS papers,
               substr(list_filter(p.notes, lambda n: n LIKE 'affiliation: %')[1], 14) AS affiliation
        FROM s.persons p LEFT JOIN s.person_stats ps USING (person_id)
        WHERE p.base_name = ? ORDER BY papers DESC, p.name""", [row[0]]).fetchall()
    found = [dict(zip(["key", "name", "page_kind", "papers", "affiliation"], r)) for r in rows]
    subject = next((p for p in found if p["name"] == subject_name), None)
    return subject, [p for p in found if p is not subject]


def right_person(calls, answer, subject, others):
    """Whether the answer is about the subject and nobody it could be confused with.

    When a page-specific tool was called, the keys it was given decide: the subject's must be among
    them and no namesake's may be. A co-author's key is fine - "has he written with the first one"
    needs it. When none was called - a plain count, which the name lookup answers on its own - the
    answer has to quote the subject's count."""
    # a page's exact name is accepted as its key, so it counts as that page; a failed call looked
    # nobody up, so it does not count at all
    alias = {subject["name"].lower(): subject["key"]}
    alias.update({p["name"].lower(): p["key"] for p in others})
    other_keys = {p["key"] for p in others}
    used = set()
    for c in calls:
        if c.get("refused"):
            continue
        for a in AUTHOR_KEY_ARGS:
            value = (c.get("arguments") or {}).get(a)
            if value:
                used.add(alias.get(str(value).strip().lower(), value))
    wrong = sorted(used & other_keys)
    if wrong:
        return False, f"used the wrong person's page: {', '.join(wrong)}"
    if used:
        if subject["key"] not in used:
            return False, "never looked up the subject's own page"
        return True, ""
    count = int(subject["papers"])
    if str(count) in (answer or "") or f"{count:,}" in (answer or ""):
        return True, ""
    return False, f"no page-specific lookup, and the answer does not quote the subject's {count} records"


def run(ctx, client, suite="instructor", limit=None, out=print):
    spec = SUITES[suite]
    subject, others = pages(ctx, spec["subject"])
    if subject is None:
        raise SystemExit(f"No dblp page named {spec['subject']!r} in this snapshot.")
    out(f"Subject: {subject['name']}  ({subject['key']}, {subject['papers']} records, "
        f"{subject['affiliation'] or 'no affiliation on the page'})")
    for p in others:
        out(f"  same name: {p['name']}  ({p['key']}, {p['page_kind']}, {p['papers']} records, "
            f"{p['affiliation'] or 'no affiliation'})")
    hint = spec.get("affiliation_hint")
    premise_ok = not hint or hint.lower() in (subject["affiliation"] or "").lower()
    if not premise_ok:
        out(f"\n!! The scenarios that say '{hint}' assume it is on the subject's page, and it is not:"
            f" read those answers knowing the question itself points elsewhere.")
    out("")

    cases = spec["cases"][:limit] if limit else spec["cases"]
    results, started = [], time.time()
    stopped = None
    for i, case in enumerate(cases, 1):
        try:
            r = E.run_case(ctx, client, case)
            if r.get("error") and r.get("error_kind") in E.TERMINAL:
                stopped = f"stopped after {i - 1} of {len(cases)}: {r['error']}"
                out(f"\n!! {stopped}")
                break
        except Exception as e:                    # one broken scenario must not lose the rest
            question = " -> ".join(case.get("turns") or [case.get("q", "")])
            r = {"question": question, "passed": False, "reason": f"crashed: {type(e).__name__}: {e}",
                 "calls": [], "answer": "", "seconds": 0, "cost_usd": 0,
                 "grounding": {"ok": True, "ungrounded": []}}
        if case.get("subject") and not case.get("refuses"):
            ok, why = right_person(r.get("calls", []), r.get("answer"), subject, others)
            if not ok:
                r["passed"] = False
                r["reason"] = "; ".join(x for x in (r["reason"], why) if x)
        if not r["grounding"]["ok"]:
            r["passed"] = False
            r["reason"] = "; ".join(x for x in (r["reason"], "a number in the answer has no source: "
                                                + ", ".join(map(str, r["grounding"]["ungrounded"]))) if x)
        r["note"] = case.get("note")
        results.append(r)
        _print(out, i, r)

    passed = sum(r["passed"] for r in results)
    summary = {"suite": suite, "subject": subject, "same_name": others, "stopped": stopped,
               "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "scenarios": len(results), "passed": passed,
               "cost_usd": round(sum(r["cost_usd"] or 0 for r in results), 4),
               "median_seconds": sorted(r["seconds"] for r in results)[len(results) // 2] if results else None,
               "minutes": round((time.time() - started) / 60, 1),
               "failed": [{"question": r["question"], "reason": r["reason"]}
                          for r in results if not r["passed"]]}
    out(f"\n{passed}/{len(results)} passed · ${summary['cost_usd']} · "
        f"median {summary['median_seconds']}s per scenario")
    return {"summary": summary, "results": results}


def _print(out, i, r):
    out(f"{'PASS' if r['passed'] else 'FAIL'}  {i}. {r['question']}")
    for c in r.get("calls", []):
        args = ", ".join(f"{k}={v!r}" for k, v in (c.get("arguments") or {}).items())
        out(f"      {'x ' if c.get('refused') else '  '}{c['tool']}({args})"
            + (f"   <- refused: {c.get('summary')}" if c.get("refused") else ""))
    if not r["grounding"]["ok"]:
        # the rows the numbers should have come from, so a misread is visible without rerunning
        for c in r.get("calls", []):
            for row in c.get("rows") or []:
                out(f"          {c['tool']} row: {row}")
    if r.get("note"):
        out(f"        note: {r['note']}")
    if not r["passed"]:
        out(f"        why it failed: {r['reason']}")
    answer = (r.get("answer") or r.get("error") or "(no answer)").strip()
    out("        " + answer.replace("\n", "\n        "))
    out(f"        {r['seconds']}s · ${r['cost_usd']}\n")


def save(payload, fingerprint):
    d = config.MODELS_DIR / "chat-eval"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"scenarios-{payload['summary']['suite']}-{fingerprint}.json"
    path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return path
