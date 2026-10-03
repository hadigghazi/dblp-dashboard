"""
The DBLP-QA experiment harness, without a model.

Every number this produces may end up in a paper, so the parts that turn answers into numbers are
tested on their own: the metric, the interval, reading the judge's reply, and the controls that decide
whether a judge is trusted at all.
"""
import json

import pytest

from chat import dblpqa as DQ

CSV = """id,question,answer,dblp_key,semantic_scholar_id
qa1,What is A?,A is the first thing.,conf/x/A1,1
qa2,What is B?,B is the second thing.,conf/x/B2,2
qa3,What is C?,C is the third thing.,conf/x/C3,3
"""


def test_the_dataset_is_read_and_its_columns_checked():
    rows = DQ.parse(CSV)
    assert len(rows) == 3 and rows[0]["question"] == "What is A?"
    with pytest.raises(ValueError, match="columns"):
        DQ.parse("q,a\n1,2\n")


def test_rouge_l_matches_the_reference_definition():
    assert DQ.rouge_l("the cat sat", "the cat sat") == 1.0
    assert DQ.rouge_l("", "anything") == 0.0
    assert DQ.rouge_l("dog", "cat") == 0.0
    # LCS "the cat" of 2: precision 2/3, recall 2/2 -> F1 0.8
    assert abs(DQ.rouge_l("the cat sat", "The CAT!") - 0.8) < 1e-9


def test_the_interval_contains_the_mean_and_is_reproducible():
    values = [2, 2, 1, 0, 2, 1, 2, 2, 0, 1]
    ci = DQ.bootstrap_ci(values)
    assert ci["low"] <= ci["mean"] <= ci["high"] and ci["n"] == 10
    assert DQ.bootstrap_ci(values) == ci, "seeded, so a re-run reports the same interval"
    assert DQ.bootstrap_ci([2] * 5) == {"mean": 2.0, "low": 2.0, "high": 2.0, "n": 5}


def test_the_judges_reply_is_read_even_with_prose_around_it():
    assert DQ.parse_judgement('{"score": 2, "reason": "same"}')["score"] == 2
    assert DQ.parse_judgement('Sure! {"score": 1, "reason": "partial"} hope it helps')["score"] == 1
    assert DQ.parse_judgement('score: 0')["score"] == 0
    with pytest.raises(ValueError):
        DQ.parse_judgement("I think it is good")
    with pytest.raises(ValueError):
        DQ.parse_judgement('{"score": 5}')


def test_the_swap_control_never_pairs_a_question_with_its_own_answer():
    for n in (2, 3, 10, 50):
        order = DQ.derangement(n)
        assert sorted(order) == list(range(n)) and all(i != j for i, j in enumerate(order))


class Scripted:
    """Answers with a fixed string; judges by comparing the answer to the ground truth, optionally
    badly, so the controls can be seen to pass and to fail."""

    def __init__(self, honest=True):
        self.honest = honest
        self.calls = 0

    def complete(self, messages, model, tools=None, temperature=None):
        self.calls += 1
        user = messages[-1]["content"]
        if messages[0]["content"] == DQ.JUDGE_SYSTEM:
            truth = user.split("Ground truth: ")[1].split("\n")[0]
            candidate = user.split("Answer to grade: ")[1]
            score = (2 if candidate == truth else 0) if self.honest else 2
            return {"content": json.dumps({"score": score, "reason": "x"}),
                    "usage": {"input_tokens": 100, "output_tokens": 10}}
        return {"content": "I do not know.", "usage": {"input_tokens": 20, "output_tokens": 5}}


def test_a_closed_book_run_writes_scores_intervals_and_cost(tmp_path):
    rows = DQ.parse(CSV)
    got = DQ.run_closed_book(Scripted(), ["m1"], "judge", rows, "sha", out_dir=tmp_path, out=lambda *_: None)
    assert got["judge_controls"]["passed"]
    assert got["results"]["m1"]["judge_score"]["mean"] == 0.0, "'I do not know' is not the answer"
    assert got["cost_usd"] >= 0 and got["usage"]["judge"]["calls"] == 3 * 2 + 3
    lines = (tmp_path / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["condition"] == "closed-book"


def test_a_judge_that_fails_its_controls_stops_the_run_before_any_answer_is_paid_for(tmp_path):
    """A judge that gives everything 2 would report a perfect model. The control catches that."""
    client = Scripted(honest=False)
    got = DQ.run_closed_book(client, ["m1"], "judge", DQ.parse(CSV), "sha", out_dir=tmp_path,
                             out=lambda *_: None)
    assert got is None
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "failed its controls" in saved["stopped"]
    assert client.calls == 6, "only the control judgements were made - no answers were generated"
