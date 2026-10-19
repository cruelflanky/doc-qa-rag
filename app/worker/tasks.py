"""Celery-задача индексации и чтение её состояния."""

import logging
from functools import lru_cache
from pathlib import Path

from celery import Task
from celery.result import AsyncResult

from app.config import Settings, get_settings
from app.embeddings import Embedder, build_embedder
from app.ingest.indexer import index_pdf
from app.rag.pipeline import build_store
from app.retry import is_transient
from app.schemas import IndexResult, TaskProgress, TaskStatusResponse
from app.store.qdrant_store import QdrantStore
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

MAX_TASK_RETRIES = 3
RETRY_BASE_DELAY_S = 5
INTERNAL_ERROR_MESSAGE = "Внутренняя ошибка индексации, подробности в логах worker"


@lru_cache
def _resources() -> tuple[QdrantStore, Embedder, Settings]:
    """Клиенты создаются один раз на процесс worker (модель BM25 грузится не на каждую задачу)."""
    settings = get_settings()
    store = build_store(settings)
    store.ensure_collection()
    return store, build_embedder(settings), settings


@celery_app.task(bind=True, name="index_document", max_retries=MAX_TASK_RETRIES)
def index_document(self: Task, doc_id: str, path: str, title: str) -> dict:
    def report(stage: str, current: int, total: int) -> None:
        self.update_state(
            state="PROGRESS", meta={"stage": stage, "current": current, "total": total}
        )

    try:
        store, embedder, settings = _resources()  # Qdrant может быть недоступен и здесь
        result = index_pdf(
            Path(path),
            doc_id,
            title,
            store=store,
            embedder=embedder,
            settings=settings,
            on_progress=report,
        )
    except Exception as exc:
        if is_transient(exc):
            delay = RETRY_BASE_DELAY_S * 2**self.request.retries
            logger.warning("Индексация %s: временная ошибка, повтор через %d с", doc_id, delay)
            raise self.retry(exc=exc, countdown=delay) from exc
        raise
    return result.model_dump()


def _error_message(error: object) -> str:
    # Celery восстанавливает исключение из backend; сверяем по имени класса
    if type(error).__name__ == "DocumentError":
        return str(error)
    return INTERNAL_ERROR_MESSAGE


def task_status(task_id: str, doc_id: str) -> TaskStatusResponse:
    result = AsyncResult(task_id, app=celery_app)
    state = result.state
    response = TaskStatusResponse(task_id=task_id, doc_id=doc_id, state=state)
    if state == "PROGRESS" and isinstance(result.info, dict):
        return response.model_copy(update={"progress": TaskProgress(**result.info)})
    if state == "SUCCESS":
        return response.model_copy(update={"result": IndexResult(**result.result)})
    if state == "FAILURE":
        return response.model_copy(update={"error": _error_message(result.result)})
    return response
