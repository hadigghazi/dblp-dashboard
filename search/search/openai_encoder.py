"""
An OpenAI-backed encoder, behind the same two methods as the local one.

Why it exists: measured on 177 paraphrase queries, the right paper is somewhere in the two thousand
nearest vectors only half the time, and sits at a median rank of 183 when it is there. The ordering
can be repaired with a reranker; the missing half cannot, and needs better vectors than
bge-small-en-v1.5 produces from a ten-word title.

Two things differ from the local path and both matter. Requests cost money, so the encoder counts
its tokens and can price them. And an API is latency-bound rather than CPU-bound: the build loop
hands over one batch at a time, so the concurrency lives in here, splitting a batch into requests
that fly together. With `dimensions` the vectors come back truncated (these models are trained so
that a prefix of the vector is still a good vector), which breaks unit norm - so they are
re-normalised here, because everything downstream treats a dot product as a cosine.
"""
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from . import config

log = logging.getLogger("dblp.search.openai")

PREFIX = "openai:"
# per 1M tokens, for the build's own report; override if your contract differs
PRICES = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}


def is_openai(model_name):
    return (model_name or "").startswith(PREFIX)


def model_of(model_name):
    return (model_name or "").split(PREFIX, 1)[-1]


class Throttle:
    """A token budget shared by every worker, refilled at the rate the API allows.

    A 429 pauses all of them, not just the request that received it: when the limit is on tokens
    per minute, the other workers are equally over it and retrying independently just keeps the
    budget pinned at zero."""

    def __init__(self, tokens_per_minute=None):
        rate = float(tokens_per_minute or config.API_TOKENS_PER_MINUTE)
        self.rate = rate / 60.0                 # tokens per second
        self.capacity = rate / 6.0              # ten seconds of burst
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.paused_until = 0.0
        self.lock = threading.Lock()

    def take(self, need):
        """Block until `need` tokens of budget exist, and spend them."""
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                wait = self.paused_until - now
                if wait <= 0:
                    if self.tokens >= need:
                        self.tokens -= need
                        return
                    wait = (need - self.tokens) / self.rate
            time.sleep(min(max(wait, 0.05), 10) + random.random() * 0.1)

    def pause(self, seconds):
        with self.lock:
            self.paused_until = max(self.paused_until, time.monotonic() + seconds)
            self.tokens = 0.0


def estimate_tokens(texts):
    """Close enough to pace by: English runs about four characters to the token, and a title is
    short enough that the per-input overhead matters."""
    return int(sum(len(t or "") for t in texts) / 3.5) + len(texts) + 8


class OpenAIEncoder:
    """encode_docs(list[str]) -> float32 (N, DIM) L2-normalised; encode_query(str) -> float32 (DIM,)."""

    def __init__(self, model_name=None, dim=None, api_key=None, base_url=None,
                 request_size=None, concurrency=None, throttle=None):
        self.model = model_of(model_name or config.MODEL_NAME)
        self.dim = int(dim or config.EMBED_DIM)
        self.api_key = api_key if api_key is not None else config.API_KEY
        self.base_url = (base_url or config.API_BASE_URL).rstrip("/")
        self.request_size = int(request_size or config.API_REQUEST_SIZE)
        self.concurrency = int(concurrency or config.API_CONCURRENCY)
        self._http = httpx.Client(timeout=config.API_TIMEOUT)
        self._lock = threading.Lock()
        self.throttle = throttle if throttle is not None else Throttle()
        self.tokens = 0
        self.waited = 0.0

    # ---- one request, with the retries an API always eventually needs ----
    def _embed(self, texts):
        if not self.api_key:
            raise RuntimeError("no API key for the embedding model (set OPENAI_API_KEY)")
        body = {"model": self.model, "input": list(texts)}
        if self.dim:
            body["dimensions"] = self.dim
        need = estimate_tokens(texts)
        last = None
        for attempt in range(config.API_RETRIES):
            t0 = time.monotonic()
            self.throttle.take(need)            # wait for room before sending, not after failing
            with self._lock:
                self.waited += time.monotonic() - t0
            try:
                r = self._http.post(f"{self.base_url}/embeddings", json=body,
                                    headers={"Authorization": f"Bearer {self.api_key}"})
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = float(r.headers.get("retry-after") or 0) or min(60, 2 ** attempt)
                    # everyone stops, because everyone is over the same budget
                    self.throttle.pause(wait)
                    log.warning("embedding request %s; pausing every worker %.1fs", r.status_code, wait)
                    last = RuntimeError(f"{r.status_code}: {r.text[:200]}")
                    continue
                r.raise_for_status()
                payload = r.json()
            except httpx.HTTPError as e:
                last = e
                self.throttle.pause(min(30, 2 ** attempt))
                continue
            rows = sorted(payload["data"], key=lambda d: d["index"])
            with self._lock:
                self.tokens += int((payload.get("usage") or {}).get("total_tokens") or 0)
            return np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        raise RuntimeError(f"embedding request failed after {config.API_RETRIES} attempts: {last}")

    @staticmethod
    def _normalise(vectors):
        """`dimensions` returns a truncated vector, which is no longer unit length - and every
        comparison downstream is a dot product standing in for a cosine."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def encode_docs(self, texts):
        texts = [t if (t or "").strip() else " " for t in texts]     # the API rejects an empty string
        chunks = [texts[i:i + self.request_size] for i in range(0, len(texts), self.request_size)]
        if len(chunks) == 1:
            return self._normalise(self._embed(chunks[0]))
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(chunks))) as pool:
            parts = list(pool.map(self._embed, chunks))
        return self._normalise(np.vstack(parts))

    def encode_query(self, text):
        # no instruction prefix: that is a bge convention, and these models are not trained with one
        return self._normalise(self._embed([text]))[0]

    def cost_usd(self):
        return round(self.tokens / 1e6 * PRICES.get(self.model, 0.0), 4)

    def close(self):
        self._http.close()
