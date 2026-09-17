FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv
COPY requirements.txt .
# scipy/scikit-learn are large wheels; retry hard, because a truncated download surfaces as a
# confusing "packages do not match the hashes" error rather than a network error
RUN pip install --retries 10 --timeout 180 --prefer-binary -r requirements.txt
COPY ml ./ml

# `docker build --target test ml/` then `docker run --rm <image>` runs the suite on a synthetic dump
FROM base AS test
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY tests ./tests
CMD ["pytest", "-q", "-p", "no:cacheprovider", "tests"]

FROM base AS runtime
RUN useradd --create-home --uid 10001 app && mkdir -p /models && chown app /models
USER app
# a batch job, not a server: run it with `docker compose run --rm ml python -m ml.cli <command>`
ENTRYPOINT []
CMD ["python", "-m", "ml.cli", "--help"]
