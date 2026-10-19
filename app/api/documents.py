"""Маршруты документов: загрузка, список, вопрос по документу."""

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from redis import Redis

from app.api.deps import get_pipeline, get_redis, get_settings_dep, get_store
from app.config import Settings
from app.ingest.pdf import PDF_MAGIC
from app.rag.pipeline import QAPipeline
from app.schemas import AnswerResponse, AskRequest, DocumentInfo, UploadResponse
from app.store.qdrant_store import QdrantStore
from app.task_registry import register_task
from app.worker.tasks import index_document

router = APIRouter(prefix="/documents", tags=["documents"])

READ_CHUNK_BYTES = 1024 * 1024
BYTES_IN_MB = 1024 * 1024


async def _save_upload(file: UploadFile, target: Path, max_bytes: int) -> None:
    """Копирует загруженный файл, проверяя сигнатуру PDF и лимит размера.

    Starlette к этому моменту уже разобрал multipart-тело (во временный файл), поэтому
    слишком большой запрос отсекается раньше — по Content-Length в main.py.
    """
    written = 0
    with target.open("wb") as out:
        while chunk := await file.read(READ_CHUNK_BYTES):
            if written == 0 and not chunk.startswith(PDF_MAGIC):
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Ожидается PDF-файл")
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    f"Файл больше {max_bytes // BYTES_IN_MB} МБ",
                )
            out.write(chunk)
    if written == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Пустой файл")


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=UploadResponse)
async def upload_document(
    file: UploadFile,
    settings: Annotated[Settings, Depends(get_settings_dep)],
    redis: Annotated[Redis, Depends(get_redis)],
    title: Annotated[str | None, Form()] = None,
) -> UploadResponse:
    """Принимает PDF и ставит фоновую задачу индексации."""
    doc_id = str(uuid.uuid4())
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{doc_id}.pdf"
    try:
        await _save_upload(file, target, settings.max_upload_mb * BYTES_IN_MB)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise

    doc_title = (title or "").strip() or Path(file.filename or "document").stem
    try:
        task = index_document.delay(doc_id, str(target), doc_title)
        register_task(redis, task.id, doc_id)
    except Exception:
        # без task_id клиент не узнает о задаче; worker, если задача всё же ушла,
        # завершится ошибкой «нет файла», и точки в Qdrant не появятся
        target.unlink(missing_ok=True)
        raise
    return UploadResponse(doc_id=doc_id, task_id=task.id, status_url=f"/tasks/{task.id}")


@router.get("", response_model=list[DocumentInfo])
def list_documents(store: Annotated[QdrantStore, Depends(get_store)]) -> list[DocumentInfo]:
    """Полностью проиндексированные документы (идущие в индексации видны через /tasks)."""
    return [DocumentInfo(**vars(doc)) for doc in store.list_documents()]


@router.post("/{doc_id}/ask", response_model=AnswerResponse, response_model_exclude_none=True)
def ask(
    doc_id: str,
    body: AskRequest,
    pipeline: Annotated[QAPipeline, Depends(get_pipeline)],
) -> AnswerResponse:
    """Ответ строго по документу; debug=true добавляет диагностику поиска."""
    return pipeline.ask(doc_id, body.question, debug=body.debug)
