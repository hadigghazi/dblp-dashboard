"""
Questions about dblp's records, without the network: a tiny parquet in dblp's shape, references that
can be counted by hand, and the scoring that needs no judge.

The references are the ground truth of the comparison with RAGScholar's pipeline, so each kind is
checked against a count made by hand, and the people a question may name are checked to be
unambiguous: one page, not a disambiguation bin, one name.
"""
import json

import pytest

from chat import config, dblpqa_structured as ST

# (key, type, title, year, authors, journal, booktitle, publtype)
RECORDS = [
    ("conf/xx/A1", "inproceedings", "Alpha systems for the first time.", 2020, ["Ann Able", "Cat Cole"], None, "XX", None),
    ("conf/xx/A2", "inproceedings", "Alpha systems for the second time.", 2020, ["Ann Able", "Cat Cole"], None, "XX", None),
    ("conf/xx/A3", "inproceedings", "Alpha systems for the third time.", 2020, ["Ann Able", "Dan Dorn"], None, "XX", None),
    ("conf/xx/A4", "inproceedings", "Alpha systems once more alone.", 2021, ["Ann Able"], None, "XX", None),
    ("conf/xx/E1", "inproceedings", "Echo chambers in alpha systems.", 2021, ["Eve Echo", "Dan Dorn"], None, "XX", None),
    ("journals/yy/B1", "article", "Bold claims about beta structures.", 2019, ["Bob Bold 0001", "Cat Cole"], "YY J.", None, None),
    ("journals/yy/B2", "article", "Bolder claims about beta structures.", 2019, ["Bob Bold 0002"], "YY J.", None, None),
    ("journals/corr/abs-1", "article", "A preprint by Ann Able.", 2020, ["Ann Able"], "CoRR", None, "informal"),
    ("conf/xx/2020", "proceedings", "Proceedings of XX 2020", 2020, [], None, "XX", None),
    ("homepages/a/Able", "www", "Home Page", None, ["Ann Able"], None, None, None),
    ("homepages/c/Cole", "www", "Home Page", None, ["Cat Cole"], None, None, None),
    ("homepages/d/Dorn", "www", "Home Page", None, ["Dan Dorn"], None, None, None),
    ("homepages/b/1", "www", "Home Page", None, ["Bob Bold 0001"], None, None, None),
    ("homepages/b/2", "www", "Home Page", None, ["Bob Bold 0002"], None, None, None),
    ("homepages/e/Echo", "www", "Home Page", None, ["Eve Echo"], None, None, "disambiguation"),
    ("homepages/f/Fox", "www", "Home Page", None, ["Fay Fox", "F. Fox"], None, None, None),
]


@pytest.fixture
def small(tmp_path, monkeypatch):
    import duckdb
    path = tmp_path / "dblp.parquet"
    con = duckdb.connect()
    con.execute("CREATE TABLE t (key VARCHAR, type VARCHAR, title VARCHAR, year INTEGER, authors VARCHAR[], "
                "journal VARCHAR, booktitle VARCHAR, publtype VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [list(r) for r in RECORDS])
    con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()
    monkeypatch.setattr(ST, "study_dir", lambda: tmp_path / "structured")
    monkeypatch.setattr(config, "TMP_DIR", tmp_path / "tmp")
    monkeypatch.setattr(ST, "PER_TYPE", 2)
    monkeypatch.setattr(ST, "RANGES", {"paper_years": (2018, 2022), "paper_authors": (1, 4), "title_chars": 10,
                                       "author_records": (1, 100), "pair_author_records": (1, 100),
                                       "pair_records": 2, "venue_years": (2018, 2022),
                                       "venue_year_records": (1, 100), "venue_records": (1, 100),
                                       "namesakes": (2, 10)})
    return path


def by_type(payload, kind):
    return [q for q in payload["questions"] if q["type"] == kind]


def test_every_reference_is_counted_from_the_records(small):
    got = ST.build(parquet=small, out=lambda *_: None)
    counts = {"Ann Able": 5, "Cat Cole": 3, "Dan Dorn": 2, "Bob Bold 0001": 1, "Bob Bold 0002": 1}
    for q in by_type(got, "author_count"):
        assert q["ref"]["number"] == counts[q["entity"]["name"]], q
    assert not {q["entity"]["name"] for q in by_type(got, "author_count")} & {"Eve Echo", "Fay Fox", "F. Fox"}, \
        "a bin, or a page with two names, is never the subject of a count"

    for q in by_type(got, "pair_count"):
        assert {q["entity"]["a"], q["entity"]["b"]} == {"Ann Able", "Cat Cole"} and q["ref"]["number"] == 2

    [names] = by_type(got, "namesakes")
    assert names["entity"]["base"] == "Bob Bold" and names["ref"]["number"] == 2

    expected = {("conf/xx", 2020): 3, ("conf/xx", 2021): 2, ("journals/yy", 2019): 2}
    venue_years = by_type(got, "venue_year_count")
    assert len({q["entity"]["series"] for q in venue_years}) == len(venue_years) == 2, "one year per venue"
    for q in venue_years:
        assert q["ref"]["number"] == expected[(q["entity"]["series"], q["entity"]["year"])]

    [top] = by_type(got, "venue_top_author")
    assert top["question"] == "Who has published the most papers at XX?" and top["ref"]["names"] == ["Ann Able"]
    # journals/yy has no clear winner (three names with one paper each), so it is not asked

    papers = by_type(got, "venue_year") + by_type(got, "authors")
    assert len(papers) == 4 and not any(q["entity"]["key"].startswith("journals/corr") for q in papers)
    for q in by_type(got, "venue_year"):
        record = next(r for r in RECORDS if r[0] == q["entity"]["key"])
        assert q["ref"]["year"] == record[3] and record[2].rstrip(".") in q["question"]
    for q in by_type(got, "authors"):
        record = next(r for r in RECORDS if r[0] == q["entity"]["key"])
        assert q["ref"]["names"] == [ST.base_name(a) for a in record[4]]
    assert (small.parent / "structured" / "questions.json").exists()


def test_scoring_needs_the_reference_and_nothing_else():
    venue = {"type": "venue_year", "ref": {"year": 2020, "venues": ["SIGMOD Conference", "sigmod"]}}
    assert ST.score(venue, "It appeared at SIGMOD Conference 2020.")
    assert ST.score(venue, "Published at SIGMOD in 2020."), "the series name counts as the venue"
    assert not ST.score(venue, "It appeared at SIGMOD in 2019.")
    journal = {"type": "venue_year", "ref": {"year": 2005, "venues": ["Int. J. Medical Informatics", "ijmi"]}}
    assert ST.score(journal, "It was published in the International Journal of Medical Informatics in 2005.")
    assert ST.score(journal, "Int. J. Medical Informatics, 2005.")
    assert not ST.score(journal, "It was published in Medical Informatics Europe in 2005.")
    short = {"type": "venue_year", "ref": {"year": 2023, "venues": ["Pattern Recognit.", "pr"]}}
    assert ST.score(short, "Pattern Recognition, 2023.")
    assert not ST.score(short, "In the proceedings of CVPR 2023."), "a two-letter code is not found inside words"
    # a journal is not named by an answer that places the paper at a conference of a similar name
    in_journal = dict(short, entity={"key": "journals/pr/X23"})
    assert ST.score(in_journal, "It appeared in Pattern Recognition in 2023.")
    assert not ST.score(in_journal, "It appeared at the IEEE Conference on Computer Vision and Pattern Recognition 2023.")
    # a citation marker is not a number the answer states
    pair = {"type": "pair_count", "ref": {"number": 2}}
    assert not ST.score(pair, "I could not find their joint papers [2].")
    assert ST.score(pair, "They wrote 2 papers together [1].")
    authors = {"type": "authors", "ref": {"names": ["Jürgen Schmidhuber", "Sepp Hochreiter", "John Smith Jr."]}}
    assert ST.score(authors, "By Jurgen Schmidhuber, Sepp Hochreiter and John Smith.")
    assert not ST.score(authors, "By Jürgen Schmidhuber and John Smith.")
    count = {"type": "author_count", "ref": {"number": 1234}}
    assert ST.score(count, "dblp lists 1,234 publications.") and not ST.score(count, "about 1,200")
    top = {"type": "venue_top_author", "ref": {"names": ["Wei Wang 0003"]}}
    assert ST.score(top, "Wei Wang (0003) has the most.") and not ST.score(top, "Wei Zhang has the most.")
    initials = {"type": "venue_top_author", "ref": {"names": ["Sajal K. Das 0001"]}}
    assert ST.score(initials, "Sajal Das has published the most.") and not ST.score(initials, "Sajal Kumar has.")


class Knows:
    """Answers the count questions right and everything else wrong."""

    def complete(self, messages, model, tools=None, temperature=None, extra=None):
        q = messages[-1]["content"]
        text = "It is 5." if "Ann Able" in q and "publications" in q else "I am not sure."
        return {"content": text, "usage": {"input_tokens": 10, "output_tokens": 3}}


def test_an_arm_is_scored_per_type_and_compared_with_dewey(small):
    ST.build(parquet=small, out=lambda *_: None)
    got = ST.run(None, Knows(), "closed", model="gpt-4.1-mini", out=lambda *_: None)
    total = got["results"]["questions"]
    assert got["results"]["correct"] == (1 if any(q["entity"].get("name") == "Ann Able"
                                                  for q in ST.load()["questions"] if q["type"] == "author_count") else 0)
    assert set(got["results"]["by_type"]) <= set(ST.TYPES) and total == len(ST.load()["questions"])
    # a Dewey run that got everything right: the comparison counts what only one of them got
    runs = small.parent / "structured" / "runs" / "20990101T000000Z-dewey-dewey"
    runs.mkdir(parents=True)
    recs = [{"id": q["id"], "type": q["type"], "correct": True} for q in ST.load()["questions"]]
    (runs / "answers.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    (runs / "summary.json").write_text(json.dumps({"arm": "dewey", "model": "dewey"}), encoding="utf-8")
    result = ST.compare(out=lambda *_: None)
    vs = result["dewey_vs"]["closed (gpt-4.1-mini)"]
    assert vs["only_dewey"] == total - got["results"]["correct"] and vs["only_other"] == 0
    assert vs["sign_test_p"] < 0.05


def test_the_wilson_interval_brackets_the_rate():
    low, high = ST.wilson(7, 10)
    assert low < 0.7 < high and 0 <= low and high <= 1


def test_the_sql_arm_is_given_every_column_of_the_tables_it_may_query(ctx):
    text = ST.sql_system(ctx)
    for table in ("pubs", "persons", "slots"):
        assert f"- {table}(" in text, table
    assert "person_id" in text and "page_kind" in text and "run_sql" in text
    assert "- src(" not in text, "the raw records are a view the guarded SQL cannot read"


def test_an_arm_that_offers_only_sql_refuses_any_other_tool(ctx):
    from chat import agent
    from chat.llm import FakeClient
    client = FakeClient(script=[{"tool_calls": [{"id": "c1", "name": "resolve_author", "arguments": {"name": "Ada Alpha"}}]}],
                        answer="I could not look that up.")
    payloads = []
    agent.answer(ctx, client, "Who is Ada Alpha?", collect=payloads, channel="cli", tools=["run_sql"], system="SQL only.")
    got = [p for p in payloads if p.get("name") == "resolve_author"]
    assert got and (got[0].get("result") or {}).get("refused"), payloads

