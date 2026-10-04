"""
The second judge, without a model: the agreement statistics, which answer sets it re-judges, and the
contrasts and self-preference check it reports.
"""
import json

import pytest

from chat import dblpqa as DQ, dblpqa_crossjudge as CJ

from tests.test_dblpqa import CSV, Scripted
from tests.test_dblpqa_audit import rec, write_run

A, B, C = "A is the first thing.", "B is the second thing.", "C is the third thing."


def test_weighted_kappa():
    assert CJ.weighted_kappa([0, 1, 2], [0, 1, 2]) == 1.0
    near = CJ.weighted_kappa([0, 1, 2, 2], [0, 1, 2, 1])
    far = CJ.weighted_kappa([0, 1, 2, 2], [0, 1, 2, 0])
    assert near > far, "a 2-versus-0 disagreement costs more than a 2-versus-1"
    assert CJ.weighted_kappa([2, None], [2, 1]) == 1.0 and CJ.weighted_kappa([], []) is None


def test_a_local_judge_is_routed_to_ollama_retried_and_not_billed(monkeypatch):
    seen = []

    class Local:
        def __init__(self, api_key=None, base_url=None, timeout=None):
            self.base_url = base_url

        def complete(self, messages, model, tools=None, temperature=None, extra=None):
            seen.append((model, temperature, extra))
            if len(seen) == 1:
                raise RuntimeError("500 from a busy CPU")
            return {"content": '```json\n{"score": 1, "reason": "partly"}\n```', "usage": {"input_tokens": 9}}

    import chat.llm as L
    monkeypatch.setattr(L, "Client", Local)
    monkeypatch.setattr(DQ.time, "sleep", lambda _s: None)
    meter = DQ.Meter()
    got = DQ.judge(object(), meter, "ollama:qwen2.5:14b", "q", "gold", "answer")
    assert got["score"] == 1 and len(seen) == 2
    assert seen[-1] == ("qwen2.5:14b", 0, {"max_tokens": DQ.LOCAL_JUDGE_TOKENS})
    assert not meter.usage, "a local judge costs nothing"


@pytest.fixture
def study(tmp_path):
    runs = tmp_path / "runs"
    ours = {"ours": {"temperature": 0}}
    # the first judge's scores are what the runs recorded; the second judge (Scripted) gives 2 to the
    # gold answer and 0 to anything else
    write_run(runs, "20260101T000000Z-closed-book", {"condition": "closed-book", "models": ["m1"], "sampling": ours},
              [rec("qa1", 2, A), rec("qa2", 1, "wrong"), rec("qa3", 0, "wrong")])
    write_run(runs, "20260102T000000Z-oracle", {"condition": "oracle", "models": ["m1"], "sampling": ours},
              [rec("qa1", 2, A), rec("qa2", 2, B), rec("qa3", 2, C)])
    write_run(runs, "20260103T000000Z-rag-bm25", {"condition": "rag-bm25", "models": ["m1"], "sampling": ours,
                                                   "pool_sha256": "P1"},
              [rec("qa1", 2, A, source_in_context=True), rec("qa2", 2, B, source_in_context=False),
               rec("qa3", 0, "nope", source_in_context=False)])
    write_run(runs, "20260104T000000Z-rag-bm25-gated", {"condition": "rag-bm25-gated", "models": ["m1"],
                                                         "sampling": ours, "pool_sha256": "P1"},
              [rec("qa1", 2, A, source_in_context=True)])
    return tmp_path


def test_the_second_judge_reports_agreement_contrasts_and_self_preference(study):
    client = Scripted()
    lines = []
    report = CJ.run(client, DQ.parse(CSV), judge_model="judge2", out=lines.append, cache_dir=study)
    assert report["controls"]["passed"]
    assert [(s["condition"], s["questions"]) for s in report["sets"]] == \
        [("closed-book", 3), ("rag-bm25", 3), ("oracle", 3)], "plain RAG only by default; the gated run is left out"
    assert client.calls == 6 + 3, "controls, then the three answers no control had graded already"

    a = report["agreement"]
    assert a["answers"] == 9 and a["exact"] == 8 and a["confusion"]["1->0"] == 1

    def contrast(name, group="all"):
        return next(c for c in report["contrasts"] if c["contrast"] == name and c["questions"] == group)
    plain = contrast("rag-bm25 - closed-book")
    assert plain["first"]["delta"]["mean"] == 0.333 and plain["second"]["delta"]["mean"] == 0.667
    assert contrast("oracle - closed-book")["second"]["delta"]["mean"] == 1.333
    missed = contrast("rag-bm25 - closed-book", "source missed")
    assert missed["first"]["questions"] == 2 and missed["second"]["delta"]["mean"] == 1.0
    assert report["first_minus_second"]["m1"]["mean"] == 0.111
    assert (study / "crossjudge.json").exists() and any("self-preference" in line for line in lines)

    again = Scripted()
    CJ.run(again, DQ.parse(CSV), judge_model="judge2", out=lambda *_: None, cache_dir=study)
    assert again.calls == 0, "verdicts are cached, so an interrupted run resumes"


def test_a_second_judge_that_fails_its_controls_judges_nothing(study):
    client = Scripted(honest=False)
    report = CJ.run(client, DQ.parse(CSV), judge_model="judge2", out=lambda *_: None, cache_dir=study)
    assert not report["controls"]["passed"] and report["stopped"] and "sets" not in report
    assert client.calls == 6


def test_an_unreadable_verdict_is_left_out_and_counted(study):
    class Mumbles(Scripted):
        def complete(self, messages, model, tools=None, temperature=None, extra=None):
            if "Answer to grade: nope" in messages[-1]["content"]:
                return {"content": "hmm, hard to say", "usage": {}}
            return super().complete(messages, model, tools, temperature, extra)

    report = CJ.run(Mumbles(), DQ.parse(CSV), judge_model="judge2", out=lambda *_: None, cache_dir=study)
    assert report["agreement"]["unreadable"] == 1 and report["agreement"]["answers"] == 9
    saved = json.loads((study / "crossjudge.json").read_text(encoding="utf-8"))
    assert saved["scores"]["rag-bm25|m1|ours"]["qa3"] == [0, None]
