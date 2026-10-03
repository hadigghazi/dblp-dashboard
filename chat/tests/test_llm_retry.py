"""
A rate limit is a wait, not a failure: the last gold-set run lost every question from about the
thirty-ninth onward to a per-minute limit while the account had credit. Running out of credit is not
retried - waiting does not fix it.
"""
import json

import httpx
import pytest

from chat import config
from chat.llm import Client, LLMError, wait_for

OK = {"choices": [{"message": {"content": "fine", "tool_calls": []}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 3, "completion_tokens": 1}, "model": "m"}
LIMITED = json.dumps({"error": {"code": "rate_limit_exceeded", "message": "Rate limit reached"}})
NO_CREDIT = json.dumps({"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}})


def client_with(responses, monkeypatch):
    monkeypatch.setattr(config, "RATE_WAIT_MAX", 0.01)
    sent = []

    def handler(request):
        sent.append(request)
        return responses[min(len(sent), len(responses)) - 1]

    c = Client(api_key="k", base_url="http://provider.test")
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c, sent


def test_a_rate_limit_is_waited_out(monkeypatch):
    c, sent = client_with([httpx.Response(429, text=LIMITED, headers={"retry-after-ms": "5"}),
                           httpx.Response(200, json=OK)], monkeypatch)
    assert c.complete([{"role": "user", "content": "hi"}], model="m")["content"] == "fine"
    assert len(sent) == 2


def test_running_out_of_credit_is_not_retried(monkeypatch):
    c, sent = client_with([httpx.Response(429, text=NO_CREDIT), httpx.Response(200, json=OK)], monkeypatch)
    with pytest.raises(LLMError) as e:
        c.complete([{"role": "user", "content": "hi"}], model="m")
    assert e.value.kind == "credit" and len(sent) == 1


def test_a_limit_that_never_lifts_gives_up_with_a_readable_message(monkeypatch):
    monkeypatch.setattr(config, "RATE_RETRIES", 2)
    c, sent = client_with([httpx.Response(429, text=LIMITED)], monkeypatch)
    with pytest.raises(LLMError) as e:
        c.complete([{"role": "user", "content": "hi"}], model="m")
    assert e.value.kind == "busy" and len(sent) == 3
    assert "{" not in str(e.value)


def test_the_streamed_answer_is_retried_too(monkeypatch):
    stream = ('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
              'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\ndata: [DONE]\n\n')
    c, sent = client_with([httpx.Response(429, text=LIMITED), httpx.Response(200, text=stream)], monkeypatch)
    got = list(c.stream([{"role": "user", "content": "hi"}], model="m"))
    assert ("token", "ok") in got and len(sent) == 2


def test_the_providers_reset_hints_are_read():
    assert wait_for(httpx.Response(429, headers={"retry-after-ms": "250"}), 0) == 0.25
    assert wait_for(httpx.Response(429, headers={"x-ratelimit-reset-tokens": "1.5s"}), 0) == 1.5
    capped = wait_for(httpx.Response(429, headers={"x-ratelimit-reset-requests": "6m0s"}), 0)
    assert capped == config.RATE_WAIT_MAX, "a visitor is never kept waiting minutes"
