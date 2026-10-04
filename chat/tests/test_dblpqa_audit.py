"""
The DBLP-QA audit, without a model: which runs it reads, what it sends the labellers, and how it turns
their labels into the numbers that go into the write-up.
"""
import json

import pytest

from chat import dblpqa as DQ, dblpqa_audit as AU

from tests.test_dblpqa import CSV


def test_labels_are_read_strictly():
    allowed = AU.ANSWER_LABELS
    assert AU.parse_label('{"label": "misled", "reason": "x"}', allowed)["label"] == "misled"
    assert AU.parse_label('Verdict: {"label": "Valid_Other_Paper"}', allowed)["label"] == "valid_other_paper"
    assert AU.parse_label("I would say own_knowledge here.", allowed)["label"] == "own_knowledge"
    with pytest.raises(ValueError):
        AU.parse_label("either misled or own_knowledge", allowed)
    with pytest.raises(ValueError):
        AU.parse_label('{"label": "great"}', allowed)


def test_kappa():
    assert AU.cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]) == 1.0
    assert AU.cohen_kappa(["a", "b"], ["b", "a"]) == -1.0, "always disagreeing is worse than chance"
    assert AU.cohen_kappa(["a", None], ["a", "b"]) == 1.0, "an unreadable label is left out"
    assert AU.cohen_kappa([], []) is None


def write_run(runs, name, summary, records):
    d = runs / name
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (d / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def rec(qid, score, answer="x", **kw):
    return dict({"model": "m1", "id": qid, "score": score, "answer": answer}, **kw)


class Labeller:
    """gpt-4.1 and gpt-4.1-mini disagree on one question; answers are labelled by their text."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, model, tools=None, temperature=None, extra=None):
        self.calls += 1
        system, user = messages[0]["content"], messages[1]["content"]
        if system == AU.QUESTION_SYSTEM:
            question = user.split("Question: ")[1].split("\n")[0]
            label = {"What is A?": "general", "What is B?": "underspecified",
                     "What is C?": "underspecified" if model == "gpt-4.1" else "identifiable"}[question]
        else:
            assert "[1] Nine\nNine abstract." in user, "the labeller sees the abstracts the system saw"
            label = {"Nine says so.": "valid_other_paper", "Nine again.": "misled"}[user.split("Answer: ")[1].split("\n")[0]]
        return {"content": json.dumps({"label": label, "reason": "r"}),
                "usage": {"input_tokens": 100, "output_tokens": 5}}


@pytest.fixture
def study(tmp_path):
    runs = tmp_path / "runs"
    ours = {"ours": {"temperature": 0}}
    write_run(runs, "20260101T000000Z-closed-book", {"condition": "closed-book", "models": ["m1"], "sampling": ours},
              [rec("qa1", 2, "cb one"), rec("qa2", 1, "cb two"), rec("qa3", 2, "cb three")])
    # an older pool, which the audit must not read
    write_run(runs, "20260101T120000Z-rag-bm25", {"condition": "rag-bm25", "models": ["m1"], "sampling": ours,
                                                   "pool_sha256": "P0"},
              [rec("qa1", 0, source_in_context=False, retrieved=["k9"])])
    write_run(runs, "20260102T000000Z-rag-bm25", {"condition": "rag-bm25", "models": ["m1"], "sampling": ours,
                                                   "pool_sha256": "P1"},
              [rec("qa1", 2, source_in_context=True, retrieved=["k1"]),
               rec("qa2", 2, source_in_context=False, retrieved=["k9"]),
               rec("qa3", 0, "Nine says so.", source_in_context=False, retrieved=["k9"])])
    write_run(runs, "20260103T000000Z-rag-bm25-gated", {"condition": "rag-bm25-gated", "models": ["m1"],
                                                         "sampling": ours, "pool_sha256": "P1"},
              [rec("qa1", 2, source_in_context=True, retrieved=["k1"], kept=["k1"]),
               rec("qa2", 1, "Nine again.", source_in_context=False, retrieved=["k9"], kept=["k9"]),
               rec("qa3", 1, source_in_context=False, retrieved=["k9"], kept=[])])
    (tmp_path / "pools.json").write_text(json.dumps({q: {"candidates": {"k9": {"title": "Nine", "abstract": "Nine abstract."}}}
                                                     for q in ("qa1", "qa2", "qa3")}), encoding="utf-8")
    (tmp_path / "abstracts.json").write_text(json.dumps({q: {"title": "T", "abstract": "Abs.", "source": "x"}
                                                         for q in ("qa1", "qa2", "qa3")}), encoding="utf-8")
    return tmp_path


def test_the_audit_explains_lost_points_on_the_latest_pool(study):
    client = Labeller()
    lines = []
    report = AU.run(client, DQ.parse(CSV), out=lines.append, cache_dir=study)

    assert report["pool_sha256"] == "P1", "the latest rag run's pool is the default"
    assert report["question_agreement"] == {"kappa": report["question_agreement"]["kappa"], "same": 2, "of": 3}
    assert client.calls == 3 * 2 + 1 * 2 + 1 * 2, "qa3's gate kept nothing, so that answer needs no labeller"

    plain, gated = sorted(report["missed_questions"], key=lambda r: r["condition"])
    assert plain["condition"] == "rag-bm25" and plain["missed"] == 2 and plain["lost_points"] == 1
    assert plain["labels"] == {"valid_other_paper": 1, "misled": 0, "own_knowledge": 0}
    assert (plain["closed_book"], plain["graded"], plain["crediting_valid"]) == (1.5, 1.0, 2.0)
    assert gated["labels"] == {"valid_other_paper": 0, "misled": 1, "own_knowledge": 1}
    assert gated["graded"] == gated["crediting_valid"] == 1.0, "nothing valid to credit"
    assert plain["by_question_label"]["underspecified"] == 1

    under = report["by_question_label"]["underspecified"]
    assert under["questions"] == 2 and under["source_retrieved"] == 0
    assert under["models"]["m1"] == {"closed_book": 1.5, "plain_rag": 1.0}
    assert report["by_question_label"]["general"]["source_retrieved"] == 1
    assert (study / "audit.json").exists() and any("credited" in line for line in lines)

    again = Labeller()
    AU.run(again, DQ.parse(CSV), out=lambda *_: None, cache_dir=study)
    assert again.calls == 0, "labels are cached"


def test_an_unknown_pool_is_refused(study):
    with pytest.raises(SystemExit, match="no rag runs"):
        AU.run(Labeller(), DQ.parse(CSV), out=lambda *_: None, cache_dir=study, pool="nope")


class Grader:
    """A multi-reference judge: the reference and a few other answers are correct; `lenient` passes
    everything, which the controls must catch."""
    VALID = {("What is B?", "cb two"), ("What is C?", "Nine says so.")}

    def __init__(self, lenient=False):
        self.calls, self.lenient = 0, lenient

    def complete(self, messages, model, tools=None, temperature=None, extra=None):
        self.calls += 1
        assert messages[0]["content"] == AU.MULTI_SYSTEM
        user = messages[1]["content"]
        assert "[1] Nine\nNine abstract." in user, "every answer is graded with the same retrieved abstracts"
        question = user.split("Question: ")[1].split("\n")[0]
        reference = user.split("written from): ")[1].split("\n")[0]
        answer = user.split("Answer to grade: ")[1].split("\n")[0]
        score = 2 if self.lenient or answer == reference or (question, answer) in self.VALID else 0
        return {"content": json.dumps({"score": score, "reason": "r"}),
                "usage": {"input_tokens": 400, "output_tokens": 10}}


def test_the_regrade_grades_every_answer_set_alike_after_its_controls(study):
    client = Grader()
    lines = []
    report = AU.regrade(client, DQ.parse(CSV), out=lines.append, cache_dir=study)
    assert report["missed"] == ["qa2", "qa3"], "the plain run on the latest pool says which were missed"
    assert report["controls"]["passed"] and report["controls"]["another question's RAG answer scored 0"] == 2

    closed, plain, gated = report["sets"]
    assert closed["condition"] == "closed-book" and "vs_closed_book" not in closed
    assert (closed["gold_graded"], closed["regraded"]["mean"], closed["raised"], closed["lowered"]) == (1.5, 1.0, 1, 1), \
        "closed-book answers are credited the same way - and can lose credit too"
    assert plain["condition"] == "rag-bm25" and plain["regraded"]["mean"] == 1.0
    assert plain["vs_closed_book"]["delta"]["mean"] == 0.0
    assert (plain["vs_closed_book"]["better"], plain["vs_closed_book"]["worse"]) == (1, 1)
    assert gated["regraded"]["mean"] == 0.0 and gated["lowered"] == 2 and gated["vs_closed_book"]["worse"] == 1
    assert client.calls == 6 + 5, "six control grades; one answer was already graded as a control"
    assert (study / "regrade.json").exists() and any("re-graded" in line for line in lines)

    again = Grader()
    AU.regrade(again, DQ.parse(CSV), out=lambda *_: None, cache_dir=study)
    assert again.calls == 0, "grades are cached"


def test_a_lenient_regrade_judge_is_stopped_by_its_controls(study):
    client = Grader(lenient=True)
    report = AU.regrade(client, DQ.parse(CSV), out=lambda *_: None, cache_dir=study)
    assert not report["controls"]["passed"] and report["stopped"]
    assert report["sets"] == [] and client.calls == 6, "no answer was graded by a judge that passes everything"
