FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_DIR=/opt/fastembed \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Зависимости отдельным слоем: пересобираются только при изменении lock-файла
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# BM25-модель запекается в образ, чтобы worker и api не качали её при старте
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25', cache_dir='/opt/fastembed')"

COPY app ./app

RUN useradd --create-home appuser \
    && mkdir -p /data/uploads \
    && chown -R appuser /data /opt/fastembed
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
