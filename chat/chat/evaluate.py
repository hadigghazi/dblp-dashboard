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
import time
from datetime import datetime, timezone

from . import agent, budget, config, goldset, grounding

log = logging.getLogger("dblp.chat.evaluate")


def _refused(text):
    low = (text or "").lower()
    return any(marker in low for marker in goldset.REFUSAL_MARKERS)


def run_case(ctx, client, case):
    tools_called, tool_events, t0 = [], [], time.time()

    def emit(event):
        if event.get("type") == "tool":
            tools_called.append(event["name"])
            tool_events.append(event)

    out = agent.answer(ctx, client, case["q"], emit=emit, ledger=budget.ledger)
    answer_text = out.get("answer", "")
    want_all = set(case.get("all_of", []))
    want_any = set(case.get("any_of", []))
    called = set(tools_called)
    ok_tools = want_all.issubset(called) and (not want_any or bool(want_any & called))
    if case.get("refuses"):
        passed = _refused(answer_text)
        reason = "" if passed else "answered instead of refusing"
    else:
        passed = ok_tools
        reason = "" if passed else f"missing tools: all_of={sorted(want_all - called)} any_of={sorted(want_any)}"
    lint = grounding.check(answer_text, tool_events)
    return {
        "question": case["q"], "passed": bool(passed), "reason": reason, "grounding": lint,
        "tools": tools_called, "grounded": bool(tools_called) or bool(case.get("refuses")),
        "refused": _refused(answer_text), "answer": answer_text,
        "seconds": round(time.time() - t0, 2), "cost_usd": out.get("cost_usd", 0.0),
        "rounds": out.get("rounds"), "error": out.get("error"),
    }


def run(ctx, client, cases=None, limit=None):
    cases = (cases or goldset.CASES)[:limit] if limit else (cases or goldset.CASES)
    results = [run_case(ctx, client, case) for case in cases]
    scoped = [r for c, r in zip(cases, results) if not c.get("refuses")]
    out_of_scope = [r for c, r in zip(cases, results) if c.get("refuses")]
    n = len(results) or 1
    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": len(results),
        "tool_choice_accuracy": round(sum(r["passed"] for r in scoped) / (len(scoped) or 1), 3),
        "refusal_accuracy": round(sum(r["passed"] for r in out_of_scope) / (len(out_of_scope) or 1), 3),
        "grounded_share": round(sum(r["grounded"] for r in results) / n, 3),
        # every figure in the answer traced back to something a tool returned
        "numbers_grounded_share": round(sum(r["grounding"]["ok"] for r in results) / n, 3),
        "median_seconds": sorted(r["seconds"] for r in results)[len(results) // 2] if results else None,
        "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in results), 4),
        "models": {"router": config.MODEL_FAST, "answers": config.MODEL_DEEP},
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
