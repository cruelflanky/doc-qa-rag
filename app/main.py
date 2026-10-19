"""Точка входа FastAPI."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from app.api import documents, health, tasks
from app.api.deps import get_settings_dep, get_store
from app.errors import DocumentNotFoundError
from app.retry import is_transient

logger = logging.getLogger(__name__)

UPSTREAM_UNAVAILABLE = "Внешний сервис временно недоступен, повторите запрос позже"
INTERNAL_ERROR = "Внутренняя ошибка сервиса"
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # заголовки частей и поле title сверх самого файла


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


async def _limit_upload_size(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Отклоняет слишком большую загрузку по Content-Length, не принимая тело.

    Без этого Starlette сначала целиком запишет multipart-тело во временный файл.
    """
    if request.method == "POST" and request.url.path == "/documents":
        settings = request.app.dependency_overrides.get(get_settings_dep, get_settings_dep)()
        limit = settings.max_upload_mb * documents.BYTES_IN_MB
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit + MULTIPART_OVERHEAD_BYTES:
            return JSONResponse(
                {"detail": f"Файл больше {settings.max_upload_mb} МБ"},
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            )
    return await call_next(request)


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
    app.middleware("http")(_limit_upload_size)
    app.include_router(documents.router)
    app.include_router(tasks.router)
    app.include_router(health.router)
    app.add_exception_handler(DocumentNotFoundError, _not_found)
    app.add_exception_handler(Exception, _unhandled)
    return app


app = create_app()
