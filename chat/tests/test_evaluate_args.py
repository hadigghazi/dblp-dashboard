"""
The argument check in the gold-set evaluation.

The right tool with the wrong argument is a wrong answer that a tool-choice check passes. That is the
whole reason `arguments_ok` exists, so it is tested on its own rather than only through a live run.
"""
from chat.evaluate import arguments_ok

CASE = {"args": {"central_authors": {"metric": "degree"}}}


def test_the_pinned_argument_passes():
    assert arguments_ok(CASE, [("central_authors", {"metric": "degree"})]) == (True, [])


def test_the_wrong_argument_fails_and_says_what_it_was():
    ok, wrong = arguments_ok(CASE, [("central_authors", {"metric": "betweenness"})])
    assert not ok
    assert "betweenness" in wrong[0] and "degree" in wrong[0]


def test_a_missing_argument_is_wrong_not_assumed():
    ok, _ = arguments_ok(CASE, [("central_authors", {})])
    assert not ok


def test_a_tool_that_was_not_called_is_not_constrained():
    """Whether it should have been called is the tool-choice check's business."""
    assert arguments_ok(CASE, [("top_authors", {"metric": "coauthors"})]) == (True, [])


def test_any_of_several_values_is_accepted():
    case = {"args": {"central_authors": {"metric": ["degree", "pagerank"]}}}
    assert arguments_ok(case, [("central_authors", {"metric": "pagerank"})])[0]


def test_every_call_to_a_pinned_tool_is_checked():
    """Two calls, one wrong: a model that tries the right measure and then the wrong one has still
    produced an answer from the wrong one."""
    ok, wrong = arguments_ok(CASE, [("central_authors", {"metric": "degree"}),
                                    ("central_authors", {"metric": "closeness"})])
    assert not ok and len(wrong) == 1


def test_keys_shown_to_the_user_are_caught():
    from chat.evaluate import shown_keys
    assert shown_keys("He has 25 papers. [author page: homepages/165/0820-2]", "how many papers?") \
        == ["homepages/165/0820-2"]
    assert shown_keys("Published in journals/access in 2025.", "q") == ["journals/access"]
    assert shown_keys("Here is conf/nips/VaswaniSPUJGKP17.", "Show me conf/nips/VaswaniSPUJGKP17") == []
    assert shown_keys("He has 25 papers.", "q") == []
