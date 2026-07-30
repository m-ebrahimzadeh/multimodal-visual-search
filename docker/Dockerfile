# Multimodal visual search -- API + Gradio UI in one process.
#
# Targets Hugging Face Spaces (Docker SDK), which means:
#   * listen on 7860
#   * run as uid 1000; anything written at runtime must be writable by it
#   * no GPU on the free tier, so this image is deliberately CPU-only
#
# torch resolves from the PyTorch CPU index on Linux (see [tool.uv.sources] in
# pyproject.toml). The default PyPI Linux wheel bundles CUDA at ~2.5 GB, which
# is unusable weight in a CPU-only image.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# libgomp: OpenMP runtime required by both faiss and torch.
# libjpeg/libpng: Pillow decodes corpus images and uploads.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgomp1 \
        libjpeg62-turbo \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Spaces runs containers as uid 1000. Creating the user here means HOME and the
# caches below are owned correctly rather than falling back to a read-only /.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies first, as their own layer: source changes then rebuild in
# seconds instead of re-resolving and re-downloading torch every time.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra onnx --no-install-project

COPY src/ ./src/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN uv sync --frozen --no-dev --extra onnx \
    && chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R app:app /app

USER app

# Model and dataset caches live under the app user's HOME. Without HF_HOME the
# hub would try to write to a directory this uid cannot create.
ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    VSEARCH_DATA_DIR=/app/data \
    VSEARCH_ARTIFACTS_DIR=/app/artifacts \
    VSEARCH_RESULTS_DIR=/app/results \
    PATH="/app/.venv/bin:$PATH"

RUN mkdir -p "$HF_HOME" /app/artifacts /app/data /app/results

EXPOSE 7860

# The app reports "degraded" rather than exiting when no index is present, so
# a healthcheck on /health distinguishes "up but unconfigured" from "down".
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
