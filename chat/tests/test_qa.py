"""
The QA harness, run against the fixture.

The harness exists to catch tool bugs on the real dump, but it is code too: these tests prove its
checks actually fail when the thing they watch is broken, which is the only way to trust a green
run. The dashboard cross-checks are skipped here (no api service in the test image) and exercised
on the VM.
"""
import pytest

from chat import qa, tools as T


def test_every_check_is_registered_once():
    names = [c["name"] for c in qa.CHECKS]
    assert len(names) == len(set(names))
    assert {c["group"] for c in qa.CHECKS} == {"dashboard", "invariant", "behaviour"}
    assert len([c for c in qa.CHECKS if c["group"] == "dashboard"]) >= 8


def test_the_harness_runs_and_reports(ctx):
    report = qa.run(ctx)
    assert report["summary"]["checks"] == len(qa.CHECKS)
    # without an api service every dashboard check must SKIP, never fail or silently pass
    dashboard = [r for r in report["results"] if r["group"] == "dashboard"]
    assert dashboard and all(r["status"] == "skip" for r in dashboard), \
        [r for r in dashboard if r["status"] != "skip"]


def test_invariant_and_behaviour_checks_pass_on_the_fixture(ctx):
    report = qa.run(ctx)
    failed = [f"{r['name']}: {r['detail']}" for r in report["results"]
              if r["group"] != "dashboard" and r["status"] == "FAIL"]
    assert not failed, failed


def test_the_cap_check_catches_a_tool_that_counts_its_rows(ctx, monkeypatch):
    """Re-introduce the Wei Wang bug and prove the harness sees it."""
    real = T.HANDLERS["namesakes"]

    def counts_its_rows(_ctx, name, limit=25):
        """The original bug, exactly: the count comes from the rows that fit in the answer."""
        out = real(_ctx, name, limit=limit)
        shown = sum(1 for r in out["rows"] if r["page_kind"] == "numbered")
        out["meta"]["numbered_pages"] = shown
        out["summary"] = f"“{name}”: {shown} separate people have a numbered page."
        return out
    monkeypatch.setitem(T.HANDLERS, "namesakes", counts_its_rows)
    report = qa.run(ctx, only="a capped list never becomes a count")
    assert report["summary"]["failed"] == 1, report["results"]


def test_the_filter_check_catches_a_filter_that_does_nothing(ctx, monkeypatch):
    real = T.HANDLERS["count_papers"]

    def ignores_filters(_ctx, **kwargs):
        return real(_ctx)                       # every filter dropped on the floor
    monkeypatch.setitem(T.HANDLERS, "count_papers", ignores_filters)
    report = qa.run(ctx, only="filters only ever narrow")
    assert report["summary"]["failed"] == 1, report["results"]


def test_the_note_check_catches_a_note_that_lies(ctx, monkeypatch):
    real = T.HANDLERS["papers_timeseries"]

    def wrong_note(_ctx, **kwargs):
        out = real(_ctx, **kwargs)
        out["note"] = "Journal and conference papers only; preprints excluded."
        return out
    monkeypatch.setitem(T.HANDLERS, "papers_timeseries", wrong_note)
    report = qa.run(ctx, only="a metric's note describes the population it actually counted")
    assert report["summary"]["failed"] == 1, report["results"]


@pytest.mark.parametrize("group", ["invariant", "behaviour"])
def test_groups_can_be_run_alone(ctx, group):
    report = qa.run(ctx, only=group)
    assert report["summary"]["checks"] == len([c for c in qa.CHECKS if c["group"] == group])
