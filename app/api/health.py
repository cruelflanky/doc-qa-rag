"""Проверка доступности Redis и Qdrant. OpenRouter — внешняя зависимость, здесь не проверяется."""

import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from redis import Redis

from app.api.deps import get_redis, get_store
from app.store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


def _probe(name: str, check: Callable[[], object]) -> str:
    try:
        check()
    except Exception:  # noqa: BLE001 — любая ошибка означает «недоступен»
        logger.warning("Health: %s недоступен", name, exc_info=True)
        return "unavailable"
    return "ok"


@router.get("/health")
def health(
    redis: Annotated[Redis, Depends(get_redis)],
    store: Annotated[QdrantStore, Depends(get_store)],
) -> JSONResponse:
    components = {
        "redis": _probe("redis", redis.ping),
        "qdrant": _probe("qdrant", store.client.get_collections),
    }
    healthy = all(value == "ok" for value in components.values())
    return JSONResponse(
        {"status": "ok" if healthy else "degraded", **components},
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
    )
