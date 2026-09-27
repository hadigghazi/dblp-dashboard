"""
The API-backed encoder, against a stub transport: no key, no network, no cost.

What is worth testing is not that it can parse a response. It is the three things that would be
silently wrong in production: vectors must come back in the order they were sent, a truncated vector
must be re-normalised (everything downstream treats a dot product as a cosine), and a rate-limited
request must be retried rather than lost.
"""
import time

import httpx
import numpy as np
import pytest

# test_pipeline sets the environment before any search module reads it; pytest collects this file
# first alphabetically, so without the import `config` would bind the production paths
from tests import test_pipeline as tp  # noqa: F401
from search import config, embed as E, openai_encoder as OE  # noqa: E402


def transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5)


def reply(vectors, tokens=10, shuffle=False):
    data = [{"index": i, "embedding": list(map(float, v))} for i, v in enumerate(vectors)]
    if shuffle:
        data = list(reversed(data))          # the API does not promise an order
    return httpx.Response(200, json={"data": data, "usage": {"total_tokens": tokens}})


def encoder(handler, dim=3):
    enc = OE.OpenAIEncoder("openai:text-embedding-3-large", dim, api_key="test-key",
                           throttle=OE.Throttle(tokens_per_minute=10 ** 9))
    enc._http = transport(handler)
    return enc


def test_the_factory_picks_the_backend_from_the_model_name(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "test-key")
    assert isinstance(E.make_encoder("openai:text-embedding-3-large", 256), OE.OpenAIEncoder)
    assert OE.is_openai("openai:text-embedding-3-small")
    assert not OE.is_openai("BAAI/bge-small-en-v1.5")
    assert OE.model_of("openai:text-embedding-3-large") == "text-embedding-3-large"


def test_vectors_come_back_in_the_order_they_were_sent():
    enc = encoder(lambda request: reply([[1, 0, 0], [0, 1, 0], [0, 0, 1]], shuffle=True))
    out = enc.encode_docs(["first", "second", "third"])
    assert np.allclose(out, np.eye(3)), "the index field, not the arrival order, decides"


def test_a_truncated_vector_is_renormalised():
    """`dimensions` returns a prefix of the full vector, which is not unit length any more."""
    enc = encoder(lambda request: reply([[3.0, 4.0, 0.0]]))
    out = enc.encode_docs(["one"])
    assert np.isclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[0], [0.6, 0.8, 0.0])


def test_the_requested_width_is_passed_on():
    seen = {}

    def handler(request):
        seen.update(__import__("json").loads(request.content))
        return reply([[1.0, 0.0, 0.0]])

    encoder(handler, dim=512).encode_docs(["one"])
    assert seen["dimensions"] == 512
    assert seen["model"] == "text-embedding-3-large"


def test_a_rate_limited_request_is_retried(monkeypatch):
    monkeypatch.setattr(config, "API_RETRIES", 3)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        return reply([[1.0, 0.0, 0.0]])

    out = encoder(handler).encode_docs(["one"])
    assert calls["n"] == 2 and out.shape == (1, 3)


def test_giving_up_is_an_error_not_an_empty_vector(monkeypatch):
    monkeypatch.setattr(config, "API_RETRIES", 2)
    enc = encoder(lambda request: httpx.Response(500, headers={"retry-after": "0"}, json={}))
    with pytest.raises(RuntimeError, match="failed after"):
        enc.encode_docs(["one"])


def test_a_large_batch_is_split_into_requests():
    seen = []

    def handler(request):
        inputs = __import__("json").loads(request.content)["input"]
        seen.append(len(inputs))
        return reply([[1.0, 0.0, 0.0]] * len(inputs))

    enc = OE.OpenAIEncoder("openai:text-embedding-3-large", 3, api_key="k", request_size=2,
                           concurrency=2, throttle=OE.Throttle(tokens_per_minute=10 ** 9))
    enc._http = transport(handler)
    out = enc.encode_docs(["a", "b", "c", "d", "e"])
    assert out.shape == (5, 3)
    assert sorted(seen) == [1, 2, 2]


def test_an_empty_title_is_not_sent_as_an_empty_string():
    """The API rejects one, and dblp has titles that normalise to nothing."""
    seen = {}

    def handler(request):
        seen.update(__import__("json").loads(request.content))
        return reply([[1.0, 0.0, 0.0]])

    encoder(handler).encode_docs([""])
    assert seen["input"] == [" "]


def test_cost_is_counted_from_what_the_api_reported():
    enc = encoder(lambda request: reply([[1.0, 0.0, 0.0]], tokens=1_000_000))
    enc.encode_docs(["one"])
    assert enc.tokens == 1_000_000
    assert enc.cost_usd() == OE.PRICES["text-embedding-3-large"]


def test_the_budget_makes_a_request_wait_rather_than_fail():
    """The pilot died with "Limit 1000000, Used 995362" - eight workers retrying into a budget they
    had already spent. Waiting for room is the only thing that fixes a rate limit."""
    throttle = OE.Throttle(tokens_per_minute=600)      # 10 a second, 100 of burst
    throttle.take(100)                                  # empties the burst
    started = time.monotonic()
    throttle.take(20)                                   # must wait about two seconds for a refill
    assert 1.0 < time.monotonic() - started < 5.0


def test_a_429_pauses_every_worker():
    throttle = OE.Throttle(tokens_per_minute=10 ** 9)
    throttle.pause(0.5)
    started = time.monotonic()
    throttle.take(1)
    assert time.monotonic() - started >= 0.4


def test_tokens_are_estimated_from_the_text():
    assert OE.estimate_tokens([]) == 8
    one = OE.estimate_tokens(["a title of some length"])
    assert one > OE.estimate_tokens(["short"])
    assert OE.estimate_tokens(["x" * 350]) >= 100


def test_a_rate_limited_request_waits_then_succeeds(monkeypatch):
    monkeypatch.setattr(config, "API_RETRIES", 3)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "tpm"})
        return reply([[1.0, 0.0, 0.0]])

    enc = OE.OpenAIEncoder("openai:text-embedding-3-large", 3, api_key="k",
                           throttle=OE.Throttle(tokens_per_minute=10 ** 9))
    enc._http = transport(handler)
    assert enc.encode_docs(["one"]).shape == (1, 3)
    assert calls["n"] == 2


def test_an_empty_account_stops_at_once_rather_than_retrying():
    """OpenAI returns "no credits remaining" as a 429, the same status as "too fast". The embedding
    run spent two and a half minutes backing off against it before giving up."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, json={"error": {
            "type": "insufficient_quota",
            "message": "You have no credits remaining. Add credits to continue using the API."}})

    with pytest.raises(OE.Stop, match="refused permanently"):
        encoder(handler).encode_docs(["one"])
    assert calls["n"] == 1, "a dead end must not be retried"


def test_a_rejected_key_stops_at_once():
    def handler(request):
        return httpx.Response(401, json={"error": {"code": "invalid_api_key", "message": "bad key"}})

    with pytest.raises(OE.Stop):
        encoder(handler).encode_docs(["one"])


def test_a_plain_rate_limit_is_still_retried(monkeypatch):
    monkeypatch.setattr(config, "API_RETRIES", 3)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"},
                                  json={"error": {"type": "rate_limit_exceeded", "message": "slow"}})
        return reply([[1.0, 0.0, 0.0]])

    assert encoder(handler).encode_docs(["one"]).shape == (1, 3)
    assert calls["n"] == 2


def test_an_error_body_that_is_not_an_object_is_handled(monkeypatch):
    """Gateways and proxies do not all return OpenAI's error shape."""
    monkeypatch.setattr(config, "API_RETRIES", 2)
    for body in ({"error": "slow down"}, {"error": None}, {}, {"nope": 1}):
        enc = encoder(lambda request, b=body: httpx.Response(429, headers={"retry-after": "0"}, json=b))
        with pytest.raises(RuntimeError, match="failed after"):
            enc.encode_docs(["one"])       # retried as a rate limit, not mistaken for terminal
