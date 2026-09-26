"""
The loop, with a scripted model: no network, no key, deterministic.

What is worth testing here is not the prose - it is that the loop calls what it was told to call,
survives a tool that fails, respects its caps, and accounts for what it spent.
"""
from chat import agent, config, tools as T
from chat.llm import FakeClient


def tool_call(name, arguments, call_id="c1"):
    return {"id": call_id, "name": name, "arguments": arguments}


def collect(ctx, client, question, ledger=None, history=None):
    events = []
    out = agent.answer(ctx, client, question, history=history, emit=events.append, ledger=ledger)
    return events, out


def test_system_prompt_carries_the_dump_and_the_limits(ctx):
    prompt = agent.system_prompt(ctx)
    assert "2026-09-01" in prompt and "2025" in prompt
    assert "no citation" in prompt.lower()
    assert "resolve_author" in prompt


def test_one_tool_then_an_answer(ctx, ledger):
    client = FakeClient(script=[{"tool_calls": [tool_call("top_authors", {"limit": 3})]}],
                        answer="Ada Alpha has the most records, 10.")
    events, out = collect(ctx, client, "who has the most papers?", ledger=ledger)
    kinds = [e["type"] for e in events]
    assert kinds.count("tool_start") == 1 and kinds.count("tool") == 1
    tool_event = next(e for e in events if e["type"] == "tool")
    assert tool_event["name"] == "top_authors"
    assert tool_event["rows"][0]["name"] == "Ada Alpha"
    assert tool_event["ms"] >= 0
    assert "".join(e["text"] for e in events if e["type"] == "token").strip() == \
           "Ada Alpha has the most records, 10."
    assert out["tools"] == ["top_authors"]
    assert out["usage"]["input_tokens"] > 0
    assert ledger.snapshot()["requests"] == 1


def test_parallel_tool_calls_in_one_round(ctx, ledger):
    client = FakeClient(script=[{"tool_calls": [
        tool_call("resolve_author", {"name": "Ada"}, "a"),
        tool_call("resolve_venue", {"name": "AAA"}, "b"),
    ]}])
    events, out = collect(ctx, client, "does Ada publish at AAA?", ledger=ledger)
    assert out["tools"] == ["resolve_author", "resolve_venue"]
    assert sum(1 for e in events if e["type"] == "tool") == 2


def test_two_rounds_escalate_to_the_deep_model(ctx, ledger):
    client = FakeClient(script=[
        {"tool_calls": [tool_call("resolve_author", {"name": "Ada"}, "a")]},
        {"tool_calls": [tool_call("author_profile", {"key": "homepages/a/Ada"}, "b")]},
    ], answer="Ada Alpha, 10 records.")
    _, out = collect(ctx, client, "what does Ada publish?", ledger=ledger)
    assert out["rounds"] == 2
    assert out["model"] == config.MODEL_DEEP
    assert [c["kind"] for c in client.calls][-1] == "stream"


def test_a_failing_tool_does_not_kill_the_turn(ctx, ledger):
    client = FakeClient(script=[{"tool_calls": [tool_call("run_sql", {"sql": "DROP TABLE pubs"})]}],
                        answer="I could not run that query.")
    events, out = collect(ctx, client, "drop everything", ledger=ledger)
    tool_event = next(e for e in events if e["type"] == "tool")
    assert tool_event["refused"] is True
    assert out["answer"]


def test_an_unknown_tool_is_reported_not_raised(ctx, ledger):
    client = FakeClient(script=[{"tool_calls": [tool_call("h_index", {"author": "x"})]}],
                        answer="dblp has no citation data.")
    events, out = collect(ctx, client, "h-index of Ada?", ledger=ledger)
    assert next(e for e in events if e["type"] == "tool")["refused"] is True
    assert "citation" in out["answer"]


def test_the_tool_call_cap_is_enforced(ctx, ledger, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_CALLS", 2)
    client = FakeClient(script=[{"tool_calls": [
        tool_call("dataset_facts", {}, f"c{i}") for i in range(5)]}])
    events, out = collect(ctx, client, "facts please", ledger=ledger)
    assert sum(1 for e in events if e["type"] == "tool") == 2
    assert len(out["tools"]) == 2


def test_an_answer_without_tools_is_streamed_as_is(ctx, ledger):
    client = FakeClient(script=[{"content": "Which Wei Wang do you mean?"}])
    events, out = collect(ctx, client, "papers by Wei Wang", ledger=ledger)
    assert not any(e["type"] == "tool" for e in events)
    assert "Wei Wang" in "".join(e["text"] for e in events if e["type"] == "token")
    assert out["tools"] == []


def test_history_is_passed_and_trimmed(ctx, ledger, monkeypatch):
    monkeypatch.setattr(config, "MAX_HISTORY_TURNS", 2)
    history = [{"role": "user", "content": f"turn {i}"} for i in range(6)]
    client = FakeClient()
    collect(ctx, client, "and now?", ledger=ledger, history=history)
    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert [m["content"] for m in messages[1:-1]] == ["turn 4", "turn 5"]


def test_tool_results_handed_back_are_bounded(ctx):
    payload = {"summary": "x", "rows": [{"i": i, "pad": "y" * 500} for i in range(50)]}
    trimmed = agent._trim(payload)
    assert len(trimmed) <= agent.MAX_RESULT_CHARS + 32


def test_every_gold_case_names_tools_that_exist():
    from chat import goldset
    known = set(T.HANDLERS)
    for case in goldset.CASES:
        for name in list(case.get("any_of", [])) + list(case.get("all_of", [])):
            assert name in known, f"{case['q']} names an unknown tool {name}"
        assert case.get("refuses") or case.get("any_of") or case.get("all_of")
