"""
Running the gold set.

Reports three numbers, each one a way this assistant can be wrong:

  tool choice   - did the required tools get called? (a wrong tool is a wrong answer with a
                  confident tone)
  refusals      - did the out-of-scope questions get refused rather than answered?
  grounding     - did every answer come after at least one tool call? An answer with no tool behind
                  it is, by construction, unsourced.

It also records latency and cost per question, because both are part of whether this is usable.
Results are written next to the models so /chat/status can show them.
"""
import json
import logging
import re
import time
from datetime import datetime, timezone

from . import agent, budget, config, goldset, grounding

log = logging.getLogger("dblp.chat.evaluate")


def _refused(text):
    low = (text or "").lower()
    return any(marker in low for marker in goldset.REFUSAL_MARKERS)


KEY_PATTERN = re.compile(r"\b(?:homepages|journals|conf)/[\w./-]+")


def shown_keys(answer, question):
    """dblp keys written into an answer the user did not ask for by key. People never need one, and an
    answer that ends "[author page: homepages/165/0820-2]" reads like a database dump."""
    asked = set(KEY_PATTERN.findall(question or ""))
    return sorted({k.rstrip(".,;)]") for k in KEY_PATTERN.findall(answer or "")} - asked)


def arguments_ok(case, calls):
    """Whether every call to a pinned tool used the pinned arguments.

    `case["args"]` maps a tool to {parameter: expected}, where expected is a value or a list of
    acceptable values. A tool that was not called is not constrained here - whether it should have
    been is the tool-choice check's job. This exists because the right tool with the wrong argument
    is a wrong answer that looks exactly like a right one."""
    wrong = []
    for tool, pinned in (case.get("args") or {}).items():
        for name, args in calls:
            if name != tool:
                continue
            for param, expected in pinned.items():
                allowed = expected if isinstance(expected, list) else [expected]
                if (args or {}).get(param) not in allowed:
                    wrong.append(f"{tool}({param}={(args or {}).get(param)!r}), expected {expected!r}")
    return not wrong, wrong


def run_case(ctx, client, case):
    tools_called, calls, payloads, t0 = [], [], [], time.time()
    succeeded, log = set(), []

    def emit(event):
        if event.get("type") == "tool":
            tools_called.append(event["name"])
            calls.append((event["name"], event.get("arguments") or {}))
            log.append({"tool": event["name"], "arguments": event.get("arguments") or {},
                        "refused": bool(event.get("refused")), "summary": event.get("summary"),
                        "rows": (event.get("rows") or [])[:6]})
            if not event.get("refused"):
                succeeded.add(event["name"])

    # A case may be a conversation. Only the last turn is measured; the ones before it exist to give
    # the last one something to refer to ("and his co-authors?"), which is where a chatbot that looks
    # fine on single questions usually breaks.
    turns = case.get("turns") or [case["q"]]
    history = []
    for earlier in turns[:-1]:
        prior = agent.answer(ctx, client, earlier, history=history, ledger=budget.ledger, channel="cli")
        # exactly what the panel sends back: the answer with the pages behind it
        history += [{"role": "user", "content": earlier},
                    {"role": "assistant", "content": prior.get("memory") or prior.get("answer", "")}]
    out = agent.answer(ctx, client, turns[-1], history=history, emit=emit, ledger=budget.ledger,
                       collect=payloads, channel="cli")
    answer_text = out.get("answer", "")
    want_all = set(case.get("all_of", []))
    want_any = set(case.get("any_of", []))
    called = set(tools_called)
    ok_tools = want_all.issubset(called) and (not want_any or bool(want_any & called))
    if case.get("refuses"):
        passed = _refused(answer_text)
        reason = "" if passed else "answered instead of refusing"
    else:
        ok_args, wrong_args = arguments_ok(case, calls)
        # called is not the same as worked: a guessed key calls the right tool and gets nothing back
        failed = sorted(set(case.get("succeed", [])) - succeeded)
        # several tools can each answer some questions - author_profile already lists the top
        # co-authors - so a case can ask that one of them worked rather than a particular one
        either = case.get("succeed_any")
        if either and not set(either) & succeeded:
            failed.append("one of " + "/".join(either))
        passed = ok_tools and ok_args and not failed
        if not ok_tools:
            reason = f"missing tools: all_of={sorted(want_all - called)} any_of={sorted(want_any)}"
        elif not ok_args:
            reason = "right tool, wrong arguments: " + "; ".join(wrong_args)
        elif failed:
            reason = f"called but never succeeded: {failed}"
        else:
            reason = ""
        keys = shown_keys(answer_text, " ".join(turns))
        if keys:
            passed = False
            reason = "; ".join(x for x in (reason, "shows dblp keys to the user: " + ", ".join(keys)) if x)
    lint = grounding.check(answer_text, payloads, question=" ".join(turns))
    return {
        "question": " -> ".join(turns), "passed": bool(passed), "reason": reason, "grounding": lint,
        "tools": tools_called, "calls": log,
        "grounded": bool(tools_called) or bool(case.get("refuses")),
        "refused": _refused(answer_text), "answer": answer_text,
        "seconds": round(time.time() - t0, 2), "cost_usd": out.get("cost_usd", 0.0),
        "rounds": out.get("rounds"), "error": out.get("error"), "error_kind": out.get("error_kind"),
    }


TERMINAL = ("credit", "key")


def run_cases(ctx, client, cases, each=None):
    """Run cases in order, stopping at the first provider error waiting will not fix. Past that point
    every case fails the same way, and scoring them would report wrong answers that were never
    answers. Returns (results, why it stopped or None)."""
    results = []
    for case in cases:
        r = run_case(ctx, client, case)
        if r.get("error") and r.get("error_kind") in TERMINAL:
            return results, f"stopped after {len(results)} of {len(cases)}: {r['error']}"
        results.append(r)
        if each:
            each(case, r)
    return results, None


def run(ctx, client, cases=None, limit=None, only=None):
    cases = (cases or goldset.CASES)[:limit] if limit else (cases or goldset.CASES)
    if only:
        # a diagnostic subset: the cases whose question (or any turn) contains one of the fragments
        parts = [x.strip().lower() for x in only.split(",") if x.strip()]
        cases = [c for c in cases if any(f in " ".join(c.get("turns") or [c.get("q", "")]).lower() for f in parts)]
    results, stopped = run_cases(ctx, client, cases)
    cases = cases[:len(results)]
    scoped = [r for c, r in zip(cases, results) if not c.get("refuses")]
    out_of_scope = [r for c, r in zip(cases, results) if c.get("refuses")]
    n = len(results) or 1
    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": len(results),
        # a provider error is not a wrong answer; it is counted on its own, and a terminal one stops
        # the run before it can be scored as one
        "stopped": stopped,
        "provider_errors": sum(1 for r in results if r.get("error")),
        "tool_choice_accuracy": round(sum(r["passed"] for r in scoped) / (len(scoped) or 1), 3),
        "refusal_accuracy": round(sum(r["passed"] for r in out_of_scope) / (len(out_of_scope) or 1), 3),
        "grounded_share": round(sum(r["grounded"] for r in results) / n, 3),
        # every figure in the answer traced back to something a tool returned
        "numbers_grounded_share": round(sum(r["grounding"]["ok"] for r in results) / n, 3),
        "median_seconds": sorted(r["seconds"] for r in results)[len(results) // 2] if results else None,
        "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in results), 4),
        "models": {"router": config.MODEL_FAST, "answers": config.MODEL_FAST,
                   "answers_when_tools_run_out": config.MODEL_DEEP},
        "failures": [{"question": r["question"], "reason": r["reason"], "tools": r["tools"]}
                     for r in results if not r["passed"]],
        "numbers_without_a_source": [{"question": r["question"], "numbers": r["grounding"]["ungrounded"],
                                      "answer": r["answer"]}
                                     for r in results if not r["grounding"]["ok"]],
    }
    return {"summary": summary, "results": results}


def save(payload, fingerprint):
    d = config.MODELS_DIR / "chat-eval"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{fingerprint}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
