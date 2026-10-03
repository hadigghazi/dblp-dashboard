"""
The language-model client.

Deliberately thin: one POST to an OpenAI-compatible /chat/completions endpoint, tool calling and
streaming. No vendor SDK, so the provider is a base URL - OpenAI today, a local vLLM or any
compatible gateway tomorrow - and there is nothing to keep up to date but the model name.

Two shapes are used. `complete` is for the tool rounds: short outputs, and the loop needs the whole
tool-call list before it can act. `stream` is for the final answer, so the reader sees words within
a second even when the answer takes four.
"""
import json
import logging
import re
import time

import httpx

from . import config

log = logging.getLogger("dblp.chat.llm")


class LLMError(RuntimeError):
    """A provider failure, with a message a visitor can read. `kind` says whether waiting will help:
    "credit" and "key" will not, "busy" and "down" may. `raw` keeps the provider's own words for logs."""

    def __init__(self, message, kind="down", raw=""):
        super().__init__(message)
        self.kind = kind
        self.raw = raw

    @property
    def terminal(self):
        return self.kind in ("credit", "key")


def _duration(text):
    """'6m0s', '1.5s', '20ms' -> seconds; None when there is nothing to read."""
    parts = re.findall(r"([\d.]+)(ms|s|m|h)", text or "")
    if not parts:
        return None
    return sum(float(n) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[u] for n, u in parts)


def wait_for(response, attempt):
    """How long to wait before retrying a rate-limited request: the provider's own hint when it gives
    one, otherwise 1, 2, 4, 8 s - capped either way, so a visitor is never kept waiting long."""
    headers = response.headers
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            if headers.get(name):
                return min(config.RATE_WAIT_MAX, max(0.2, float(headers[name]) * scale))
        except ValueError:
            pass
    hinted = [d for d in (_duration(headers.get("x-ratelimit-reset-tokens")),
                          _duration(headers.get("x-ratelimit-reset-requests"))) if d is not None]
    if hinted:
        return min(config.RATE_WAIT_MAX, max(0.2, max(hinted)))
    return min(config.RATE_WAIT_MAX, float(2 ** attempt))


def provider_error(status, text):
    """Turn a provider's error response into an LLMError a person can read."""
    low = (text or "").lower()
    if "insufficient_quota" in low or "billing" in low or "exceeded your current quota" in low:
        return LLMError("Dewey has used up its AI credit for now, so it can't answer questions until the "
                        "site's owner tops it up. Everything else on the site still works.", "credit", text)
    if status in (401, 403) or "invalid_api_key" in low:
        return LLMError("Dewey isn't set up correctly right now (the AI provider rejected its key). "
                        "Everything else on the site still works.", "key", text)
    if status == 429:
        return LLMError("Dewey is getting a lot of questions at once. Please try again in a minute.",
                        "busy", text)
    return LLMError(f"Dewey couldn't reach its AI provider just now (error {status}). Please try again "
                    f"in a moment.", "down", text)


def _usage(payload):
    u = payload.get("usage") or {}
    return {"input_tokens": int(u.get("prompt_tokens") or 0),
            "output_tokens": int(u.get("completion_tokens") or 0)}


class Client:
    def __init__(self, api_key=None, base_url=None, timeout=None):
        self.api_key = api_key if api_key is not None else config.API_KEY
        self.base_url = (base_url or config.BASE_URL).rstrip("/")
        self.timeout = timeout or config.LLM_TIMEOUT
        self._http = httpx.Client(timeout=self.timeout)

    def configured(self):
        return bool(self.api_key)

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _body(self, messages, model, tools=None, temperature=None, stream=False):
        body = {"model": model, "messages": messages,
                "temperature": config.TEMPERATURE if temperature is None else temperature}
        if tools:
            body["tools"] = tools
            body["parallel_tool_calls"] = True
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    def complete(self, messages, model, tools=None, temperature=None):
        """{content, tool_calls: [{id, name, arguments}], usage, model, finish_reason}."""
        if not self.configured():
            raise LLMError("no API key configured (set OPENAI_API_KEY)")
        body = self._body(messages, model, tools, temperature)
        for attempt in range(config.RATE_RETRIES + 1):
            try:
                r = self._http.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                                    json=body)
            except httpx.HTTPError as e:
                raise LLMError("Dewey couldn't reach its AI provider just now. Please try again in a "
                               "moment.", "down", str(e)) from e
            if r.status_code < 400:
                break
            err = provider_error(r.status_code, r.text[:600])
            if err.kind != "busy" or attempt == config.RATE_RETRIES:
                raise err
            wait = wait_for(r, attempt)
            log.info("rate-limited; retrying in %.1fs (attempt %d)", wait, attempt + 1)
            time.sleep(wait)
        payload = r.json()
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__unparsed__": fn.get("arguments")}
            calls.append({"id": call.get("id"), "name": fn.get("name"), "arguments": args})
        return {"content": message.get("content") or "", "tool_calls": calls,
                "usage": _usage(payload), "model": payload.get("model", model),
                "finish_reason": choice.get("finish_reason")}

    def stream(self, messages, model, temperature=None):
        """Yields ('token', text) then ('usage', {...}). Tools are not offered: this is the answer."""
        if not self.configured():
            raise LLMError("no API key configured (set OPENAI_API_KEY)")
        body = self._body(messages, model, tools=None, temperature=temperature, stream=True)
        for attempt in range(config.RATE_RETRIES + 1):
            wait = None
            try:
                with self._http.stream("POST", f"{self.base_url}/chat/completions",
                                       headers=self._headers(), json=body) as r:
                    if r.status_code >= 400:
                        err = provider_error(r.status_code, r.read()[:600].decode("utf-8", "replace"))
                        if err.kind != "busy" or attempt == config.RATE_RETRIES:
                            raise err
                        wait = wait_for(r, attempt)
                    else:
                        usage = {"input_tokens": 0, "output_tokens": 0}
                        for line in r.iter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            if chunk.get("usage"):
                                usage = _usage(chunk)
                            for choice in chunk.get("choices") or []:
                                piece = (choice.get("delta") or {}).get("content")
                                if piece:
                                    yield "token", piece
                        yield "usage", usage
                        return
            except httpx.HTTPError as e:
                raise LLMError("Dewey lost its connection to the AI provider mid-answer. Please try "
                               "again.", "down", str(e)) from e
            log.info("rate-limited; retrying the answer in %.1fs (attempt %d)", wait, attempt + 1)
            time.sleep(wait)

    def close(self):
        self._http.close()


class FakeClient:
    """A scripted client for the tests: no network, deterministic tool calls.

    `script` is a list of dicts shaped like `complete`'s return value; `answer` is the text the
    final streamed call produces."""

    def __init__(self, script=None, answer="Answer."):
        self.script = list(script or [])
        self.answer = answer
        self.calls = []

    def configured(self):
        return True

    def complete(self, messages, model, tools=None, temperature=None):
        self.calls.append({"kind": "complete", "model": model, "messages": messages})
        if self.script:
            step = self.script.pop(0)
            return {"content": step.get("content", ""), "tool_calls": step.get("tool_calls", []),
                    "usage": step.get("usage", {"input_tokens": 10, "output_tokens": 5}),
                    "model": model, "finish_reason": "tool_calls" if step.get("tool_calls") else "stop"}
        return {"content": self.answer, "tool_calls": [], "usage": {"input_tokens": 10, "output_tokens": 5},
                "model": model, "finish_reason": "stop"}

    def stream(self, messages, model, temperature=None):
        self.calls.append({"kind": "stream", "model": model, "messages": messages})
        for word in self.answer.split(" "):
            yield "token", word + " "
        yield "usage", {"input_tokens": 12, "output_tokens": 8}

    def close(self):
        pass
