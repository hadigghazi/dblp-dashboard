"""
The question log: the only record of which tools the catalogue is missing.

It must never break an answer that already worked, so the tests cover the failure paths as much as
the happy one.
"""
import json

from chat import usage
from chat.llm import FakeClient


def events(tools=(), refused=(), done=True, error=None):
    out = [{"type": "status", "text": "looking it up"}]
    for name in tools:
        out.append({"type": "tool", "name": name, "arguments": {}, "ms": 12,
                    "summary": "x", "refused": name in refused})
    if error:
        out.append({"type": "error", "message": error})
    if done:
        out.append({"type": "done", "answer": "An answer.", "rounds": 2, "tools": list(tools),
                    "seconds": 1.5, "cost_usd": 0.002, "model": "gpt-4.1-mini"})
    return out


def test_an_entry_describes_the_question_and_what_it_took():
    entry = usage.from_events("who has the most papers?", events(["top_authors", "author_profile"]), "fp1")
    assert entry["tools"] == ["top_authors", "author_profile"]
    assert entry["rounds"] == 2 and entry["seconds"] == 1.5 and entry["cost_usd"] == 0.002
    assert entry["answered"] and not entry["used_sql"] and not entry["no_tool"]
    assert entry["dump"] == "fp1" and entry["at"]


def test_the_catalogue_gaps_are_flagged():
    assert usage.from_events("q", events(["run_sql"]))["used_sql"]
    assert usage.from_events("q", events([]))["no_tool"]
    assert usage.from_events("q", events(["count_papers"], refused=["count_papers"]))["refused_tools"] \
        == ["count_papers"]
    assert usage.from_events("q", events([], done=False, error="boom"))["error"] == "boom"


def test_nothing_about_the_person_is_recorded():
    entry = usage.from_events("who has the most papers?", events(["top_authors"]))
    assert not {"ip", "token", "user", "session", "address"} & set(entry)


def test_records_and_reads_back(tmp_path):
    path = tmp_path / "2026-09.jsonl"
    for q, tools in [("a", ["top_authors"]), ("b", ["run_sql"]), ("c", [])]:
        usage.record(usage.from_events(q, events(tools)), path=path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["question"] == "a"

    report = usage.summarize(days=30, directory=tmp_path)
    assert report["questions"] == 3
    assert report["tools"] == {"top_authors": 1, "run_sql": 1}
    assert report["fell_back_to_sql"] == ["b"]
    assert report["answered_without_a_tool"] == ["c"]
    assert report["median_seconds"] == 1.5


def test_a_broken_log_file_is_skipped_not_fatal(tmp_path):
    (tmp_path / "2026-09.jsonl").write_text("{not json\n" + json.dumps(
        usage.from_events("ok", events(["top_authors"]))) + "\n", encoding="utf-8")
    assert usage.summarize(days=30, directory=tmp_path)["questions"] == 1


def test_an_unwritable_path_does_not_raise(tmp_path):
    assert usage.record({"at": "now"}, path=tmp_path / "missing" / "deep" / "x.jsonl") is None


def test_the_server_logs_every_question(client, dump, monkeypatch):
    monkeypatch.setattr(usage.config, "MODELS_DIR", dump["models"])
    before = len(usage.read(days=1))
    client.post("/chat/ask", json={"question": "how big is dblp, for the log?"})
    after = usage.read(days=1)
    assert len(after) == before + 1
    assert after[-1]["question"] == "how big is dblp, for the log?"
    assert after[-1]["tools"] == ["dataset_facts"]
