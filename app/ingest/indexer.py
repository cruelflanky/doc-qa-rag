"""Индексация PDF: разбор → фрагменты → эмбеддинги → запись в Qdrant → ready=true.

Модуль не знает про Celery: прогресс сообщается через колбэк, поэтому его легко тестировать.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path

from app.config import Settings
from app.embeddings import Embedder
from app.ingest.chunker import Chunk, chunk_document
from app.ingest.pdf import load_pdf
from app.schemas import IndexResult
from app.store.models import DocMeta
from app.store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


def _write_chunks(
    meta: DocMeta,
    chunks: list[Chunk],
    *,
    store: QdrantStore,
    embedder: Embedder,
    batch_size: int,
    on_progress: ProgressCallback,
) -> None:
    batches = [list(batch) for batch in batched(chunks, batch_size)]
    vectors = []
    for done, batch in enumerate(batches, start=1):
        texts = [chunk.text for chunk in batch]
        vectors.append((embedder.embed_dense(texts), embedder.embed_sparse(texts)))
        on_progress("embedding", done, len(batches))
    for done, (batch, (dense, sparse)) in enumerate(zip(batches, vectors, strict=True), start=1):
        store.upsert_chunks(meta, batch, dense, sparse, total_chunks=len(chunks))
        on_progress("upserting", done, len(batches))


def index_pdf(
    path: Path,
    doc_id: str,
    title: str,
    *,
    store: QdrantStore,
    embedder: Embedder,
    settings: Settings,
    on_progress: ProgressCallback,
) -> IndexResult:
    # повторный запуск задачи (ретрай Celery) начинает с чистого листа
    store.delete_document(doc_id)

    on_progress("parsing", 0, 1)
    parsed = load_pdf(path, max_pages=settings.max_pages)
    on_progress("chunking", 0, 1)
    chunks = chunk_document(parsed, settings.chunk_size, settings.chunk_overlap)
    meta = DocMeta(
        doc_id=doc_id, title=title, pages=parsed.page_count, uploaded_at=datetime.now(UTC)
    )
    logger.info("Документ %s: %d страниц, %d фрагментов", doc_id, parsed.page_count, len(chunks))

    try:
        _write_chunks(
            meta,
            chunks,
            store=store,
            embedder=embedder,
            batch_size=settings.embed_batch_size,
            on_progress=on_progress,
        )
        on_progress("finalizing", 0, 1)
        store.mark_ready(doc_id)
    except Exception:
        logger.exception("Индексация %s прервана, удаляем записанные фрагменты", doc_id)
        store.delete_document(doc_id)
        raise
    return IndexResult(doc_id=doc_id, pages=parsed.page_count, chunks=len(chunks))
