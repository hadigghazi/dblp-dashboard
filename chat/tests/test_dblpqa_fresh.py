"""
DBLP-QA-Fresh, without the network or a model: which papers it samples, which generated questions it
keeps, and that a build resumes from its cache without paying again.
"""
import json

import httpx
import pytest

from chat import config, dblpqa as DQ, dblpqa_fresh as FR

LONG = lambda word: " ".join([word] * 90)
ABSTRACTS = {"10.1/new1": LONG("alpha"), "10.1/new2": LONG("beta"), "10.1/new3": "far too short"}


def make_parquet(path):
    import duckdb
    con = duckdb.connect()
    con.execute("CREATE TABLE t (key VARCHAR, type VARCHAR, title VARCHAR, year VARCHAR, ee VARCHAR[], "
                "journal VARCHAR, publtype VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?)", [
        ["journals/x/New1", "article", "New one.", "2025", ["https://doi.org/10.1/new1"], "X", None],
        ["conf/y/New2", "inproceedings", "New two", "2026", ["https://doi.org/10.1/new2"], None, None],
        ["conf/y/New3", "inproceedings", "New three", "2025", ["https://doi.org/10.1/new3"], None, None],
        ["journals/corr/abs-2501-00001", "article", "A preprint", "2025", ["https://doi.org/10.48550/x"], "CoRR", "informal"],
        ["journals/x/Old", "article", "Old", "2020", ["https://doi.org/10.1/old"], "X", None],
        ["conf/y/NoDoi", "inproceedings", "No DOI", "2025", ["https://example.org/p"], None, None],
        ["homepages/z/a", "www", "A home page", "2025", ["https://doi.org/10.1/www"], None, None]])
    con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()
    return path


def test_only_recent_published_papers_with_a_doi_are_sampled(tmp_path):
    got = FR.sample_papers(10, make_parquet(tmp_path / "dblp.parquet"))
    assert sorted(k for k, *_ in got) == ["conf/y/New2", "conf/y/New3", "journals/x/New1"]
    assert FR.sample_papers(10, tmp_path / "dblp.parquet") == got, "a fixed seed: the same sample every time"


def test_a_generated_pair_must_be_a_standalone_question():
    assert FR.parse_pair('{"question": "What is alpha?", "answer": "Alpha is a thing."}') == \
        ("What is alpha?", "Alpha is a thing.")
    for bad in ('{"question": "What does this paper propose?", "answer": "A new thing."}',
                '{"question": "Alpha is great", "answer": "Yes it is."}', "no json at all"):
        with pytest.raises(ValueError):
            FR.parse_pair(bad)


class Writer:
    """Writes a good question for the 'alpha' abstract and a paper-bound one for 'beta'; answers from
    the abstract; judges by exact match."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, model, tools=None, temperature=None, extra=None):
        self.calls += 1
        system, user = messages[0]["content"], messages[-1]["content"]
        if system == FR.GEN_SYSTEM:
            pair = ({"question": "What is alpha?", "answer": "Alpha is good."} if "alpha" in user else
                    {"question": "What does this paper propose?", "answer": "Beta."})
            return {"content": json.dumps(pair), "usage": {"input_tokens": 300, "output_tokens": 30}}
        if system == DQ.CONTEXT_SYSTEM:
            return {"content": "Alpha is good.", "usage": {"input_tokens": 200, "output_tokens": 5}}
        truth = user.split("Ground truth: ")[1].split("\n")[0]
        candidate = user.split("Answer to grade: ")[1]
        return {"content": json.dumps({"score": 2 if candidate == truth else 0, "reason": "r"}),
                "usage": {"input_tokens": 100, "output_tokens": 5}}


def s2(requests):
    def handler(request):
        requests.append(request.url)
        ids = json.loads(request.content)["ids"]
        dois = [i.split("DOI:")[1] for i in ids]
        if request.url.params["fields"] == "abstract":
            return httpx.Response(200, json=[{"abstract": ABSTRACTS.get(d)} for d in dois])
        return httpx.Response(200, json=[{"corpusId": 1000 + i, "externalIds": {"DBLP": f"dblp/{d}"}}
                                         for i, d in enumerate(dois)])
    return handler


def test_the_build_keeps_checked_questions_and_resumes_from_its_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(DQ, "DATASET", "dblpqa")          # restored afterwards: build switches to fresh
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    parquet = make_parquet(tmp_path / "dblp.parquet")
    requests = []
    http = httpx.Client(transport=httpx.MockTransport(s2(requests)))
    lines = []
    rows = FR.build(Writer(), target=5, out=lines.append, parquet=parquet, http=http)

    assert [r["question"] for r in rows] == ["What is alpha?"], \
        "beta's question names 'this paper'; New3's abstract is too short"
    assert rows[0]["id"] == "fq1" and rows[0]["dblp_key"] == "journals/x/New1" and rows[0]["semantic_scholar_id"]
    folder = tmp_path / "models" / "dblpqa-fresh"
    assert DQ.parse((folder / "fresh.csv").read_text(encoding="utf-8")) == rows
    oracle = json.loads((folder / "abstracts.json").read_text(encoding="utf-8"))
    assert oracle["fq1"]["abstract"].startswith("alpha alpha") and oracle["fq1"]["title"] == "New one"
    assert any("fewer than 5" in line for line in lines)

    again, before = Writer(), len(requests)
    assert FR.build(again, target=5, out=lambda *_: None, parquet=parquet, http=http) == rows
    assert again.calls == 0 and len(requests) == before, "resumed from the cache: nothing paid twice"
    assert DQ.load_dataset()[0] == rows, "the harness reads the built set once switched to it"
