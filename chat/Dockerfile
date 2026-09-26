# No model weights, no torch: this service talks to an OpenAI-compatible endpoint over HTTP, so the
# image is small and the test stage needs no network and no API key (the suite injects a fake client).
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install --retries 10 --timeout 180 --prefer-binary -r requirements.txt
COPY chat ./chat

# `docker build --target test -t dblp-chat:test chat` then `docker run --rm dblp-chat:test`
FROM base AS test
COPY requirements-dev.txt .
RUN pip install --retries 10 --timeout 180 -r requirements-dev.txt
COPY tests ./tests
CMD ["pytest", "-q", "-p", "no:cacheprovider", "tests"]

FROM base AS runtime
RUN useradd --create-home --uid 10001 app && mkdir -p /models && chown app /models
USER app
EXPOSE 8004
CMD ["uvicorn", "chat.server:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]
