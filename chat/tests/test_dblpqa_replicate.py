"""
The original paper's experiments, re-run: its context strategies, its RQ1 measure and the comparison
with its Table 3 - without the network or a model.

The strategies decide what a model is given, so each is checked to give exactly the paper's context:
one abstract alone, the top k concatenated, or the model's own single-abstract answers combined. The
grid must run every variant once and skip what it already has; the RQ1 judge must pass its controls
before it is believed; and none of this may leak into the study's own analyses of plain RAG.
"""
import json

import pytest

from chat import dblpqa as DQ, dblpqa_audit as AU, dblpqa_rag as RAG, dblpqa_replicate as RP

from tests.test_dblpqa import CSV, Scripted
from tests.test_dblpqa_rag import SMALL_POOLS

quiet = lambda *_: None
TRUTHS = ("A is the first thing.", "B is the second thing.", "C is the third thing.")
RANKINGS = {"bm25": {"qa1": ["conf/x/A1", "conf/x/Z9"], "qa2": ["conf/x/Z9"], "qa3": []}}


class GridReader(Scripted):
    """Right exactly when what it is given states the answer - an abstract, or an earlier answer."""

    def complete(self, messages, model, tools=None, temperature=None, extra=None):
        if messages[0]["content"] in (DQ.SINGLE_SYSTEM, DQ.RAG_SYSTEM, DQ.CA_SYSTEM):
            self.calls += 1
            given = messages[-1]["content"].split("Question:")[0]
            said = next((t for t in TRUTHS if t in given), "I do not know.")
            return {"content": said, "usage": {"input_tokens": 50, "output_tokens": 5}}
        return super().complete(messages, model, tools, temperature, extra)


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(DQ, "study_dir", lambda: tmp_path)
    report = {"pool": {"sha256": "P1"},
              "rankers": {"bm25": {"recall@1": 1 / 3, "recall@3": 1 / 3, "recall@5": 1 / 3}}}
    monkeypatch.setattr(RAG, "prepare", lambda rows, **kw: (SMALL_POOLS, RANKINGS, report))
    return tmp_path


def runs_of(path):
    return sorted(d.name.split("-", 1)[1] for d in (path / "runs").iterdir() if d.is_dir())


# --------------------------------------------------------------------------- the strategies

def test_each_strategy_has_its_own_name_and_prompt():
    assert RAG.condition_name("bm25") == "rag-bm25", "the study's main condition keeps its name"
    assert RAG.condition_name("bm25", strategy="single", k=2) == "rag-bm25-a2"
    assert RAG.condition_name("bm25", strategy="cd", k=3) == "rag-bm25-cd3"
    assert RAG.condition_name("bm25", strategy="ca", k=5) == "rag-bm25-ca5"
    assert RAG.condition_name("bm25", "gated") == "rag-bm25-gated"
    assert DQ.rag_strategy("rag-bm25") == ("cd", 5) and DQ.rag_strategy("rag-bm25-gated") == ("cd", 5)
    assert DQ.rag_strategy("rag-openalex-semantic") == ("cd", 5)
    assert DQ.rag_strategy("rag-bm25-a4") == ("single", 4) and DQ.rag_strategy("rag-bm25-ca3") == ("ca", 3)

    single = DQ.messages_for("rag-bm25-a1", "q?", "[1] T\nabstract")
    assert single[0]["content"] == DQ.SINGLE_SYSTEM and single[1]["content"].startswith("Abstract:\n[1] T")
    ca = DQ.messages_for("rag-bm25-ca3", "q?", "[1] an answer")
    assert ca[0]["content"] == DQ.CA_SYSTEM and ca[1]["content"].startswith("Answers:\n[1] an answer")
    assert DQ.messages_for("rag-bm25", "q?", "x")[1]["content"].startswith("Abstracts:\nx"), "unchanged"
    assert DQ.messages_for("rag-bm25-permissive", "q?", "x")[0]["content"] == DQ.PERMISSIVE_SYSTEM


def test_a_single_document_context_is_the_jth_paper_alone():
    rows = DQ.parse(CSV)
    first = RAG.single_contexts(rows, SMALL_POOLS, RANKINGS, "bm25", 1)
    second = RAG.single_contexts(rows, SMALL_POOLS, RANKINGS, "bm25", 2)
    assert first["qa1"]["abstract"] == "[1] On A\nA is the first thing." and first["qa1"]["source_in_context"]
    assert second["qa1"]["abstract"] == "[1] Z\nZ is unrelated." and not second["qa1"]["source_in_context"]
    assert second["qa1"]["source_rank"] == 1 and second["qa1"]["retrieved"] == ["conf/x/Z9"]
    assert second["qa2"]["abstract"] == "(no paper was retrieved at this rank)"


def test_a_short_window_model_gets_evenly_shortened_abstracts():
    long = " ".join(f"w{i}" for i in range(900))
    contexts = {"qa1": {"abstract": "\n\n".join(f"[{i}] T{i}\n{long}" for i in range(1, 6))}}
    cut, info = RP.fitted("ollama:tinyllama:1.1b-chat", contexts, 5)
    blocks = cut["qa1"]["abstract"].split("\n\n")
    assert len(blocks) == 5 and blocks[0].startswith("[1] T1\n"), "the numbered title line is kept"
    assert all(len(b.split("\n", 1)[1].split()) <= info["words_per_abstract"] + 1 for b in blocks)
    assert blocks[0].endswith("[...]") and info["window_tokens"] == 2048
    same, none = RP.fitted("ollama:mistral:v0.1", contexts, 5)
    assert same is contexts and none is None, "a model with room is given the abstracts whole"


# --------------------------------------------------------------------------- the grid

def test_the_grid_runs_every_variant_once_and_combines_the_models_own_answers(study):
    rows = DQ.parse(CSV)
    client = GridReader()
    RP.run_grid(client, rows, "sha", ["m1"], sampling="ours", judge_model="judge", out=quiet)
    assert runs_of(study) == sorted(["closed-book", "rag-bm25-a1", "rag-bm25-a2", "rag-bm25-a3", "rag-bm25-a4",
                                     "rag-bm25-a5", "rag-bm25-cd3", "rag-bm25", "rag-bm25-ca3", "rag-bm25-ca5"])

    ca3 = next(d for d in (study / "runs").iterdir() if d.name.endswith("-rag-bm25-ca3"))
    summary = json.loads((ca3 / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["concatenated_answers_from"]) == 3 and summary["answer_prompt"] == DQ.CA_SYSTEM
    answers = {json.loads(x)["id"]: json.loads(x) for x in (ca3 / "answers.jsonl").read_text(encoding="utf-8").splitlines()}
    assert answers["qa1"]["answer"] == "A is the first thing.", "built from its A1 answer, which had the source"
    assert answers["qa1"]["source_in_context"] and summary["pool_sha256"] == "P1"

    calls = client.calls
    RP.run_grid(client, rows, "sha", ["m1"], sampling="ours", judge_model="judge", out=quiet)
    assert client.calls == calls and len(runs_of(study)) == 10, "a variant with a complete run is not re-run"

    result = RP.report(rows, ["m1"], sampling="ours", out=quiet)
    means = result["models"]["m1"]["means"]
    assert means["A1"] == pytest.approx(2 / 3, abs=1e-3) and means["A2"] == 0 and means["no context"] == 0
    assert means["Top-5-CD"] == pytest.approx(2 / 3, abs=1e-3) and means["Top-3-CA"] == pytest.approx(2 / 3, abs=1e-3)
    claims = {c["claim"]: c for c in result["models"]["m1"]["findings"]}
    assert claims["A1 is the best single-document variant"]["holds"] is True
    assert claims["Top-5-CD against Top-3-CD (the paper: depends on the model)"]["compare"]
    assert "paper" not in result["models"]["m1"], "only the paper's own models are set beside its table"
    assert (study / "replication.json").exists()


def test_concatenated_answers_refuse_to_run_without_the_single_runs(study):
    rows = DQ.parse(CSV)
    with pytest.raises(SystemExit, match="needs its A1 answers"):
        RP.contexts_for(("ca", 3), rows, SMALL_POOLS, RANKINGS, "bm25", "m1", "ours", "P1", study / "runs")


def test_the_grid_stays_out_of_the_studys_own_analyses(study):
    rows = DQ.parse(CSV)
    RP.run_grid(GridReader(), rows, "sha", ["m1"], sampling="ours", judge_model="judge",
                variants=[("single", 1), ("cd", 5)], out=quiet)
    chosen = AU.rag_runs(AU.load_runs(study / "runs"), "P1")
    assert [c[0] for c in chosen] == ["rag-bm25"], "the audit, re-grade and second judge see plain RAG only"


# --------------------------------------------------------------------------- RQ1

class BearingJudge(Scripted):
    """An abstract answers a question when it contains the ground-truth answer."""

    def complete(self, messages, model, tools=None, temperature=None, extra=None):
        if messages[0]["content"] == RP.BEARING_SYSTEM:
            self.calls += 1
            user = messages[1]["content"]
            gold = user.split("Ground-truth answer: ")[1].split("\n")[0]
            abstract = user.split("Abstract:\n")[1]
            return {"content": json.dumps({"answers": gold in abstract}),
                    "usage": {"input_tokens": 200, "output_tokens": 5}}
        return super().complete(messages, model, tools, temperature, extra)


def test_the_rq1_measure_finds_the_first_abstract_that_answers(study, monkeypatch):
    rows = DQ.parse(CSV)
    monkeypatch.setattr(DQ, "fetch_abstracts", lambda rows, **kw: {r["id"]: {"abstract": r["answer"]} for r in rows})
    rankings = {"bm25": {"qa1": ["conf/x/Z9", "conf/x/A1"], "qa2": ["conf/x/Z9"], "qa3": []}}
    monkeypatch.setattr(RAG, "prepare", lambda rows, **kw: (SMALL_POOLS, rankings, {
        "pool": {"sha256": "P1"}, "rankers": {"bm25": {"recall@1": 0.0, "recall@3": 1 / 3, "recall@5": 1 / 3}}}))
    judge = BearingJudge()
    got = RP.run_bearing(judge, rows, rankers=["bm25"], judge_model="judge", out=quiet)
    assert got["controls"]["passed"] and got["controls"]["questions"] == 3
    bm25 = got["rankers"]["bm25"]
    assert bm25["first_answering_rank"] == {"qa1": 2, "qa2": None, "qa3": None}
    assert bm25["answer_bearing"]["recall@1"] == 0 and bm25["answer_bearing"]["recall@3"] == pytest.approx(0.333, abs=1e-3)
    assert bm25["answer_bearing"]["mrr@3"] == pytest.approx(0.167, abs=1e-3)

    calls = judge.calls
    RP.run_bearing(judge, rows, rankers=["bm25"], judge_model="judge", out=quiet)
    assert judge.calls == calls, "every verdict is cached"


def test_a_lenient_rq1_judge_is_stopped_by_its_controls(study, monkeypatch):
    rows = DQ.parse(CSV)
    monkeypatch.setattr(DQ, "fetch_abstracts", lambda rows, **kw: {r["id"]: {"abstract": r["answer"]} for r in rows})

    class YesMan(Scripted):
        def complete(self, messages, model, tools=None, temperature=None, extra=None):
            return {"content": '{"answers": true}', "usage": {"input_tokens": 1, "output_tokens": 1}}

    assert RP.run_bearing(YesMan(), rows, rankers=["bm25"], judge_model="judge", out=quiet) is None


def test_metrics_and_parsing():
    got = RP.bearing_metrics([1, 2, None])
    assert got["recall@1"] == pytest.approx(0.333, abs=1e-3) and got["recall@3"] == pytest.approx(0.667, abs=1e-3)
    assert got["mrr@3"] == pytest.approx(0.5, abs=1e-3)
    assert RP.parse_bearing('{"answers": true}') is True and RP.parse_bearing('so: "answers": false') is False
    with pytest.raises(ValueError):
        RP.parse_bearing("maybe")
    assert RP.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert RP.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert RP.spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.949, abs=1e-3)
    assert RP.spearman([1, 1, 1], [1, 2, 3]) is None
    assert len(RP.PAPER_TABLE3["manual"]["Mistral-7B"]) == len(RP.VARIANTS)
