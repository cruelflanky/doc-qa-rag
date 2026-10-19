"""Состояние фоновой задачи индексации."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from redis import Redis

from app.api.deps import get_redis
from app.schemas import TaskStatusResponse
from app.task_registry import lookup_doc_id
from app.worker.tasks import task_status

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskStatusResponse, response_model_exclude_none=True)
def get_task(task_id: str, redis: Annotated[Redis, Depends(get_redis)]) -> TaskStatusResponse:
    doc_id = lookup_doc_id(redis, task_id)
    if doc_id is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Задача не найдена")
    return task_status(task_id, doc_id)
