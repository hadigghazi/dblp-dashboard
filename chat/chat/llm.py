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

import httpx

from . import config

log = logging.getLogger("dblp.chat.llm")


class LLMError(RuntimeError):
    pass


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
        try:
            r = self._http.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                                json=self._body(messages, model, tools, temperature))
        except httpx.HTTPError as e:
            raise LLMError(f"could not reach the model provider: {e}") from e
        if r.status_code >= 400:
            raise LLMError(f"model provider returned {r.status_code}: {r.text[:400]}")
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
        try:
            with self._http.stream("POST", f"{self.base_url}/chat/completions", headers=self._headers(),
                                   json=body) as r:
                if r.status_code >= 400:
                    raise LLMError(f"model provider returned {r.status_code}: {r.read()[:400]!r}")
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
        except httpx.HTTPError as e:
            raise LLMError(f"the model provider dropped the connection: {e}") from e

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
