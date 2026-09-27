"""
The API-backed encoder, against a stub transport: no key, no network, no cost.

What is worth testing is not that it can parse a response. It is the three things that would be
silently wrong in production: vectors must come back in the order they were sent, a truncated vector
must be re-normalised (everything downstream treats a dot product as a cosine), and a rate-limited
request must be retried rather than lost.
"""
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
    enc = OE.OpenAIEncoder("openai:text-embedding-3-large", dim, api_key="test-key")
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
                           concurrency=2)
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
