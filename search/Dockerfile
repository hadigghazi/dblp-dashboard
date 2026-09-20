# The embedding model (embed.py) is only ever imported inside a function, never at module load, so
# neither `store`/`bm25`/`vectors`/`fuse`/`search`/`server` nor the test suite (which injects a fake
# encoder) needs torch. Only the `runtime` stage below carries it - keeps the test image, and CI, fast.
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install --retries 10 --timeout 180 --prefer-binary -r requirements.txt
COPY search ./search

# `docker build --target test search/` then `docker run --rm <image>` runs the suite - no model,
# no network.
FROM base AS test
COPY requirements-dev.txt .
RUN pip install --retries 10 --timeout 180 -r requirements-dev.txt
COPY tests ./tests
CMD ["pytest", "-q", "-p", "no:cacheprovider", "tests"]

# The runtime image carries the actual model. torch installs from its CPU-only index first, so
# sentence-transformers (which depends on torch) never pulls the multi-gigabyte CUDA build; the
# model weights are then baked into the image, so the VM - whose internet has been flaky before -
# never needs to reach Hugging Face.
FROM base AS runtime
ARG SEARCH_MODEL=BAAI/bge-small-en-v1.5
ENV HF_HOME=/opt/hf-cache SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache SEARCH_MODEL=${SEARCH_MODEL}
RUN pip install --retries 10 --timeout 300 --prefer-binary torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --retries 10 --timeout 180 --prefer-binary "sentence-transformers>=3.0,<4"
RUN python -c "from sentence_transformers import SentenceTransformer; import os; SentenceTransformer(os.environ['SEARCH_MODEL'])"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN useradd --create-home --uid 10001 app && mkdir -p /models && chown app /models
USER app
# a batch job, not a server by default: run it with `docker compose run --rm search python -m search.cli <command>`
ENTRYPOINT []
CMD ["python", "-m", "search.cli", "--help"]
