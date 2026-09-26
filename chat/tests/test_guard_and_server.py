"""The SQL guard, the spend ledger, and the HTTP surface."""
import json

import pytest
from fastapi.testclient import TestClient

from chat import budget, config, data, server, sqlguard


# ------------------------------------------------------------------ the guard
@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "select count(*) from pubs",
    "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
    "SELECT title FROM pubs WHERE title ILIKE '%update%'",          # a keyword inside a literal
    "SELECT 1 -- ; DROP TABLE pubs",                                 # hidden in a comment
])
def test_allowed(sql):
    assert sqlguard.check(sql)[0] is True, sql


@pytest.mark.parametrize("sql", [
    "DROP TABLE pubs", "INSERT INTO pubs VALUES (1)", "UPDATE pubs SET year = 1",
    "SELECT 1; DROP TABLE pubs", "ATTACH 'x.db' AS y", "COPY pubs TO '/tmp/x.csv'",
    "SELECT * FROM read_csv('/etc/passwd')", "SELECT * FROM read_parquet('s3://x/y')",
    "INSTALL httpfs", "LOAD httpfs", "PRAGMA database_list", "SET memory_limit='99GB'",
    "CREATE TABLE t AS SELECT 1", "", "CALL pragma_version()",
])
def test_denied(sql):
    ok, message = sqlguard.check(sql)
    assert ok is False and message, sql


def test_run_caps_rows_and_reports_the_statement(loaded):
    out = sqlguard.run(loaded["serving"], "SELECT * FROM pubs", row_cap=3)
    assert out["ok"] and len(out["rows"]) == 3 and out["truncated"] is True
    assert out["sql"] == "SELECT * FROM pubs"


def test_run_reports_a_broken_query_without_raising(loaded):
    out = sqlguard.run(loaded["serving"], "SELECT nope FROM pubs")
    assert out["ok"] is False and "nope" in out["error"]


def test_run_cannot_write(loaded):
    out = sqlguard.run(loaded["serving"], "CREATE TABLE evil AS SELECT 1")
    assert out["ok"] is False


# ------------------------------------------------------------------ the ledger
def test_budget_blocks_when_the_day_is_spent(dump, monkeypatch):
    ledger = budget.Ledger(path=dump["models"] / "budget-block.json")
    monkeypatch.setattr(config, "BUDGET_USD_PER_DAY", 0.0001)
    ledger.record(config.MODEL_FAST, {"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    ok, why = ledger.check()
    assert ok is False and "budget" in why


def test_budget_blocks_a_burst(dump, monkeypatch):
    ledger = budget.Ledger(path=dump["models"] / "budget-burst.json")
    monkeypatch.setattr(config, "RATE_PER_MINUTE", 2)
    assert ledger.check()[0] and ledger.check()[0]
    ok, why = ledger.check()
    assert ok is False and "minute" in why


def test_budget_survives_a_corrupt_ledger(dump):
    path = dump["models"] / "budget-corrupt.json"
    path.write_text("{not json", encoding="utf-8")
    assert budget.Ledger(path=path).check()[0] is True


# ------------------------------------------------------------------ the server
def events_of(response):
    return [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")]


def test_health_and_status(client):
    assert client.get("/chat/health").json()["ok"] is True
    status = client.get("/chat/status").json()
    assert status["ready"] is True
    assert len(status["tools"]) >= 20
    assert status["examples"] and status["cannot_answer"]
    assert status["dump"]["latest_mdate"] == "2026-09-01"
    assert status["budget"]["usd_left"] >= 0


def test_ask_streams_tool_events_then_tokens_then_done(client):
    r = client.post("/chat/ask", json={"question": "how big is dblp?"})
    assert r.status_code == 200
    kinds = [e["type"] for e in events_of(r)]
    assert kinds[0] == "status"
    assert "tool" in kinds and "token" in kinds and kinds[-1] == "done"
    done = events_of(r)[-1]
    assert done["tools"] == ["dataset_facts"]
    assert done["dump"]["fingerprint"] == data.pool.fingerprint()


def test_the_same_question_twice_is_served_from_cache(client):
    first = events_of(client.post("/chat/ask", json={"question": "how big is dblp, exactly?"}))[-1]
    assert not first.get("cached")
    second = events_of(client.post("/chat/ask", json={"question": "how big is dblp, exactly?"}))[-1]
    assert second["cached"] is True and second["cost_usd"] == 0.0


def test_refresh_skips_the_cache(client):
    client.post("/chat/ask", json={"question": "cache me"})
    again = events_of(client.post("/chat/ask", json={"question": "cache me", "refresh": True}))[-1]
    assert not again.get("cached")


def test_a_long_question_is_refused(client):
    r = client.post("/chat/ask", json={"question": "x" * (config.MAX_QUESTION_CHARS + 1)})
    assert r.status_code == 422


def test_a_token_is_required_when_one_is_set(loaded, monkeypatch):
    monkeypatch.setattr(config, "TOKEN", "secret")
    server.state.client = FakeClient()
    c = TestClient(server.app)
    assert c.get("/chat/status").status_code == 401
    assert c.post("/chat/ask", json={"question": "hello there"}).status_code == 401
    assert c.get("/chat/status", headers={"X-Chat-Token": "secret"}).status_code == 200
    assert c.get("/chat/health").status_code == 200        # health stays open for the deploy check


def test_no_provider_key_is_a_clear_503(loaded, monkeypatch):
    monkeypatch.setattr(config, "TOKEN", "")
    from chat.llm import Client
    server.state.client = Client(api_key="")
    c = TestClient(server.app)
    r = c.post("/chat/ask", json={"question": "who has the most papers?"})
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]


def test_budget_exhaustion_is_an_event_not_a_crash(client, monkeypatch):
    monkeypatch.setattr(config, "RATE_PER_DAY", 0)
    events = events_of(client.post("/chat/ask", json={"question": "a brand new question here"}))
    assert events[0]["type"] == "error" and "limit" in events[0]["message"]


# ------------------------------------------------------------------ the store
def test_a_store_built_by_older_code_is_rebuilt(loaded, dump):
    """The live store was first built with a step that silently failed; without a version stamp the
    fix could not reach it."""
    from chat import data, store
    con, meta = data.connect(loaded["serving"])
    try:
        path = dump["models"] / "store-version-test.duckdb"
        store.build(con, meta, path)
        con.execute(f"ATTACH '{path}' AS c (READ_ONLY)")
        assert dict(con.execute("SELECT k, v FROM c._meta").fetchall())["version"] == store.VERSION
        con.execute("DETACH c")
        # pretend it came from older code, then attach: it must be rebuilt, not served as it is
        stale = duckdb_set_version(path, "1")
        assert stale == "1"
        out = store.attach(con, meta, path)
        assert out["version"] == store.VERSION
        assert store.has_table(con, "facts")
    finally:
        con.close()


def duckdb_set_version(path, version):
    import duckdb
    con = duckdb.connect(str(path))
    try:
        con.execute("UPDATE _meta SET v = ? WHERE k = 'version'", [version])
        con.execute("CHECKPOINT")
        return con.execute("SELECT v FROM _meta WHERE k = 'version'").fetchone()[0]
    finally:
        con.close()


def test_status_serves_the_gold_set_report_when_there_is_one(client, dump):
    import json as _json
    assert client.get("/chat/status").json()["evaluation"] is None
    d = dump["models"] / "chat-eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{data.pool.fingerprint()}.json").write_text(
        _json.dumps({"summary": {"cases": 38, "tool_choice_accuracy": 0.97, "refusal_accuracy": 1.0}}),
        encoding="utf-8")
    assert client.get("/chat/status").json()["evaluation"]["cases"] == 38
