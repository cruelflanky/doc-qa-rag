"""Связь task_id → doc_id в Redis.

Celery отвечает PENDING на любой неизвестный id, поэтому без реестра нельзя отличить
«задача ещё в очереди» от «такой задачи нет».
"""

from redis import Redis

TASK_TTL_SECONDS = 7 * 24 * 3600


def _key(task_id: str) -> str:
    return f"task:{task_id}"


def register_task(redis: Redis, task_id: str, doc_id: str) -> None:
    redis.set(_key(task_id), doc_id, ex=TASK_TTL_SECONDS)


def lookup_doc_id(redis: Redis, task_id: str) -> str | None:
    value = redis.get(_key(task_id))
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else value
