"""
The embedding model. Imported lazily (inside a function, not at module load) so nothing outside
`build_index`/`server` startup ever needs torch or sentence-transformers - tests inject a fake
encoder with the same interface and never touch either.

The model is baked into the Docker image at build time (see search/Dockerfile), so a container on
a VM with flaky internet never needs to reach Hugging Face at runtime.
"""
import logging
import threading

from . import config

log = logging.getLogger("dblp.search.embed")

_lock = threading.Lock()
_model = None


def _load_model(model_name):
    global _model
    with _lock:
        if _model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(config.ENCODE_THREADS)
            log.info("loading %s (cpu, %d threads)", model_name, config.ENCODE_THREADS)
            _model = SentenceTransformer(model_name, device="cpu")
        return _model


class Encoder:
    """encode_docs(list[str]) -> float32 (N, DIM) L2-normalised; encode_query(str) -> float32 (DIM,)."""

    def __init__(self, model_name=None, batch_size=None):
        self.model_name = model_name or config.MODEL_NAME
        self.batch_size = batch_size or config.ENCODE_BATCH

    def encode_docs(self, texts):
        model = _load_model(self.model_name)
        return model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True).astype("float32")

    def encode_query(self, text):
        model = _load_model(self.model_name)
        v = model.encode([config.QUERY_PREFIX + text], normalize_embeddings=True, convert_to_numpy=True)
        return v[0].astype("float32")
