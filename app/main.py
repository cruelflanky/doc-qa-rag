"""Точка входа FastAPI."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api import documents, health, tasks
from app.api.deps import get_store
from app.errors import DocumentNotFoundError
from app.retry import is_transient

logger = logging.getLogger(__name__)

UPSTREAM_UNAVAILABLE = "Внешний сервис временно недоступен, повторите запрос позже"
INTERNAL_ERROR = "Внутренняя ошибка сервиса"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    store = app.dependency_overrides.get(get_store, get_store)()
    store.ensure_collection()
    yield


async def _not_found(_: Request, exc: DocumentNotFoundError) -> JSONResponse:
    return JSONResponse(
        {"detail": "Документ не найден или ещё индексируется"},
        status_code=status.HTTP_404_NOT_FOUND,
    )


async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    if is_transient(exc):
        logger.warning("Внешний сервис недоступен после ретраев: %s %s", request.url.path, exc)
        return JSONResponse(
            {"detail": UPSTREAM_UNAVAILABLE}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    logger.exception("Необработанная ошибка: %s", request.url.path)
    return JSONResponse(
        {"detail": INTERNAL_ERROR}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
    )


def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = FastAPI(
        title="doc-qa-rag",
        description="Ответы на вопросы по PDF-документу со ссылками на страницы и разделы",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(documents.router)
    app.include_router(tasks.router)
    app.include_router(health.router)
    app.add_exception_handler(DocumentNotFoundError, _not_found)
    app.add_exception_handler(Exception, _unhandled)
    return app


app = create_app()
