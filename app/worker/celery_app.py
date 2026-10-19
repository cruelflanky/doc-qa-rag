"""Celery: брокер и хранилище результатов — Redis."""

from celery import Celery

from app.config import get_settings

RESULT_TTL_SECONDS = 7 * 24 * 3600

_settings = get_settings()

celery_app = Celery(
    "doc_qa",
    broker=_settings.redis_url,
    backend=_settings.redis_url,
    include=["app.worker.tasks"],
)
celery_app.conf.update(
    task_track_started=True,  # видно состояние STARTED
    task_acks_late=True,  # подтверждаем после выполнения: задача не потеряется при падении
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # индексация тяжёлая — не берём задачи впрок
    result_extended=True,
    result_expires=RESULT_TTL_SECONDS,
)
