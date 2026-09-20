"""A deterministic bag-of-hashed-words encoder: same interface as search.embed.Encoder, no model
or network. Titles sharing vocabulary get similar (cosine-close) vectors, enough to exercise the
ranking machinery - top-k selection, RRF fusion, the resumable build - without a real model. It is
not semantically aware (it cannot recognise a synonym it was never told about), so tests do not
assert it beats BM25 on the synonym-substitution evaluation; that property needs the real model and
is checked on the VM, not in CI.
"""
import hashlib
import re

import numpy as np

from search import config


class FakeEncoder:
    def __init__(self, dim=None):
        self.dim = dim or config.EMBED_DIM

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.split(r"[^a-z0-9]+", text.lower()):
            if not w:
                continue
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode_docs(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_query(self, text):
        return self._vec(text.replace(config.QUERY_PREFIX, ""))
