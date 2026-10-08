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

    def complete(self, messages, model, tools=None, temperature=None, extra=None):
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
    got = DQ.run_closed_book(Scripted(), ["m1"], "judge", rows, "sha", out_dir=tmp_path, out=lambda *_: None,
                             reuse_controls=False)
    assert got["judge_controls"]["passed"]
    assert got["results"]["m1"]["judge_score"]["mean"] == 0.0, "'I do not know' is not the answer"
    assert got["cost_usd"] >= 0 and got["usage"]["judge"]["calls"] == 3 * 2 + 3
    lines = (tmp_path / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["condition"] == "closed-book"


def test_a_judge_that_fails_its_controls_stops_the_run_before_any_answer_is_paid_for(tmp_path):
    """A judge that gives everything 2 would report a perfect model. The control catches that."""
    client = Scripted(honest=False)
    got = DQ.run_closed_book(client, ["m1"], "judge", DQ.parse(CSV), "sha", out_dir=tmp_path,
                             out=lambda *_: None, reuse_controls=False)
    assert got is None
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "failed its controls" in saved["stopped"]
    assert client.calls == 6, "only the control judgements were made - no answers were generated"


def test_open_models_are_routed_to_the_local_server_and_cost_nothing(monkeypatch):
    seen = {}

    class Local:
        def __init__(self, api_key=None, base_url=None, timeout=None):
            seen["base_url"] = base_url

        def complete(self, messages, model, tools=None, temperature=None, extra=None):
            seen.update(model=model, temperature=temperature, extra=extra)
            return {"content": "local answer", "usage": {"input_tokens": 9, "output_tokens": 9}}

    import chat.llm as L
    monkeypatch.setattr(L, "Client", Local)
    meter = DQ.Meter()
    got = DQ.answer_closed_book(object(), meter, "ollama:mistral:v0.1", "What is A?", "paper")
    assert got == "local answer"
    assert seen["model"] == "mistral:v0.1" and seen["base_url"].endswith("/v1")
    assert seen["temperature"] == 0.7 and seen["extra"] == {"top_p": 0.9, "max_tokens": 512}
    assert meter.cost() == 0 and not meter.usage, "a local model is not billed"


def test_a_judge_that_passed_its_controls_is_reused(tmp_path):
    run = tmp_path / "20261003T000000Z-closed-book"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "judge": "gpt-4.1", "dataset_sha256": "abc", "questions": 50,
        "judge_controls": {"passed": True, "gold_scored_2": 1.0}}), encoding="utf-8")
    assert DQ.reusable_controls("gpt-4.1", "abc", tmp_path)["reused_from"] == run.name
    assert DQ.reusable_controls("gpt-4.1", "other-dataset", tmp_path) is None
    assert DQ.reusable_controls("gpt-4.1-mini", "abc", tmp_path) is None


def test_the_report_puts_models_side_by_side(tmp_path):
    rows = DQ.parse(CSV)
    DQ.run_closed_book(Scripted(), ["m1", "m2"], "judge", rows, "sha", out_dir=tmp_path / "r",
                       out=lambda *_: None, reuse_controls=False)
    lines = []
    got = DQ.report(tmp_path / "r", out=lines.append)
    assert set(got["qa1"]) >= {"m1", "m2", "question"}
    assert any("below 2" in line for line in lines)


def test_a_local_model_that_fails_once_is_retried(monkeypatch):
    attempts = []

    class Flaky:
        def __init__(self, api_key=None, base_url=None, timeout=None):
            pass

        def complete(self, messages, model, tools=None, temperature=None, extra=None):
            attempts.append(model)
            if len(attempts) == 1:
                raise RuntimeError("500 mid-generation")
            return {"content": "fine", "usage": {}}

    import chat.llm as L
    monkeypatch.setattr(L, "Client", Flaky)
    monkeypatch.setattr(DQ.time, "sleep", lambda _s: None)
    assert DQ.answer_closed_book(object(), DQ.Meter(), "ollama:m", "q", "paper") == "fine"
    assert len(attempts) == 2


def test_a_hosted_model_failure_is_not_retried_here(monkeypatch):
    """The hosted client already retries rate limits itself; anything else is reported at once."""
    class Broken:
        def complete(self, *a, **k):
            raise RuntimeError("no")

    with pytest.raises(RuntimeError, match="gpt-x failed"):
        DQ.answer_closed_book(Broken(), DQ.Meter(), "gpt-x", "q", "ours")


def test_the_oracle_prompt_carries_the_abstract_and_closed_book_does_not():
    with_it = DQ.messages_for("oracle", "What is A?", "A is a thing studied here.")
    assert "A is a thing studied here." in with_it[-1]["content"] and "What is A?" in with_it[-1]["content"]
    without = DQ.messages_for("closed-book", "What is A?")
    assert without[-1]["content"] == "What is A?"


def test_abstracts_fall_back_when_semantic_scholar_withholds_one(tmp_path):
    """The paper notes Semantic Scholar now withholds some abstracts; arXiv fills the gap here."""
    import httpx as H
    long_text = " ".join(["word"] * 30)

    def handler(request):
        if "semanticscholar" in request.url.host:
            return H.Response(200, json=[
                {"title": "A", "abstract": long_text, "externalIds": {}},
                {"title": "B", "abstract": None, "externalIds": {"ArXiv": "2401.00001"}},
                {"title": "C", "abstract": None, "externalIds": {}}])
        if "arxiv" in request.url.host:
            return H.Response(200, text=f"<feed><entry><summary> {long_text} </summary></entry></feed>")
        return H.Response(404)

    http = H.Client(transport=H.MockTransport(handler))
    got = DQ.fetch_abstracts(DQ.parse(CSV), cache_dir=tmp_path, http=http, out=lambda *_: None)
    assert got["qa1"]["source"] == "semantic-scholar"
    assert got["qa2"]["source"] == "arxiv"
    assert got["qa3"]["abstract"] is None, "nothing anywhere: left empty, not invented"
    again = DQ.fetch_abstracts(DQ.parse(CSV), cache_dir=tmp_path, http=None, out=lambda *_: None)
    assert again == got, "cached: the second call makes no requests"


def test_an_oracle_run_leaves_out_questions_with_no_abstract_and_pairs_with_closed_book(tmp_path):
    rows = DQ.parse(CSV)
    runs = tmp_path / "runs"
    DQ.run_condition(Scripted(), ["m1"], "judge", rows, "sha", "closed-book",
                     out_dir=runs / "20260101T000000Z-closed-book", out=lambda *_: None, reuse_controls=False)
    contexts = {"qa1": {"abstract": "A is the first thing.", "source": "x"},
                "qa2": {"abstract": "B is the second thing.", "source": "x"}, "qa3": {"abstract": None}}
    got = DQ.run_condition(Scripted(), ["m1"], "judge", rows, "sha", "oracle", contexts=contexts,
                           out_dir=runs / "20260102T000000Z-oracle", out=lambda *_: None, reuse_controls=False)
    assert got["questions"] == 2 and got["questions_in_dataset"] == 3
    paired = got["results"]["m1"]["vs_closed_book"]
    assert paired["questions"] == 2 and paired["against_run"].endswith("closed-book")


def test_answers_survive_the_credit_running_out_and_are_scored_later(tmp_path, monkeypatch):
    """The Mistral oracle run lost ten free answers when judging hit a credit error. Now generation
    carries on, everything is saved, and rescore finishes the job."""
    from chat.llm import LLMError
    rows = DQ.parse(CSV)

    class Broke(Scripted):
        def complete(self, messages, model, tools=None, temperature=None, extra=None):
            if messages[0]["content"] == DQ.JUDGE_SYSTEM:
                raise LLMError("out of credit", "credit")
            return super().complete(messages, model, tools, temperature, extra)

    runs = tmp_path / "runs"
    # the judge passed its controls earlier; the credit runs out while scoring the answers
    monkeypatch.setattr(DQ, "reusable_controls", lambda *a, **k: {"passed": True, "reused_from": "earlier"})
    got = DQ.run_condition(Broke(), ["m1"], "judge", rows, "sha", "closed-book",
                           out_dir=runs / "20260101T000000Z-closed-book", out=lambda *_: None)
    assert got["stopped"] and "out of credit" in got["stopped"]
    saved = [json.loads(x) for x in (runs / "20260101T000000Z-closed-book" / "answers.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 3 and all(r["score"] is None for r in saved), "every answer kept, none scored"

    monkeypatch.setattr(DQ.config, "MODELS_DIR", tmp_path / "models-unused")
    monkeypatch.setattr(DQ, "load_dataset", lambda: (rows, "sha"))
    done = DQ.rescore(Scripted(), runs / "20260101T000000Z-closed-book", "judge", out=lambda *_: None)
    assert done["results"]["m1"]["judged"] == 3 and done["stopped"] is None


def test_a_closed_book_run_from_before_the_sampling_field_still_pairs(tmp_path):
    old = tmp_path / "20250101T000000Z-closed-book"
    old.mkdir()
    (old / "summary.json").write_text(json.dumps({"models": ["m1"]}), encoding="utf-8")
    (old / "answers.jsonl").write_text(json.dumps({"model": "m1", "id": "qa1", "score": 1}) + "\n",
                                        encoding="utf-8")
    got = DQ.paired_delta("m1", "ours", [{"model": "m1", "id": "qa1", "score": 2}], tmp_path)
    assert got and got["questions"] == 1 and got["better"] == 1


def test_a_reference_cut_to_its_first_sentence_is_the_boundary_control():
    assert DQ.first_sentence("Compressive Sensing is a technique. It needs few measurements.") ==         "Compressive Sensing is a technique."
    assert DQ.first_sentence("One sentence only.") is None

