from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import Retry

from app.errors import DocumentError
from app.schemas import IndexResult
from app.worker import tasks
from app.worker.celery_app import celery_app


@pytest.fixture(autouse=True)
def eager(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    celery_app.conf.task_always_eager = True
    with patch.object(tasks, "_resources", return_value=(MagicMock(), MagicMock(), MagicMock())):
        yield
    celery_app.conf.task_always_eager = False


def run_task():
    return tasks.index_document.apply(args=("doc-1", "/data/uploads/doc-1.pdf", "Договор"))


def test_task_success_returns_index_result():
    result = IndexResult(doc_id="doc-1", pages=3, chunks=12)
    with patch.object(tasks, "index_pdf", return_value=result):
        outcome = run_task()

    assert outcome.successful()
    assert outcome.result == {"doc_id": "doc-1", "pages": 3, "chunks": 12}


def test_task_reports_progress_meta():
    def fake_index(*args, on_progress, **kwargs):
        on_progress("embedding", 2, 5)
        return IndexResult(doc_id="doc-1", pages=1, chunks=1)

    with (
        patch.object(tasks, "index_pdf", side_effect=fake_index),
        patch.object(tasks.index_document, "update_state") as update_state,
    ):
        run_task()

    update_state.assert_called_with(
        state="PROGRESS", meta={"stage": "embedding", "current": 2, "total": 5}
    )


def test_task_retries_on_transient_error():
    error = ConnectionError("Qdrant недоступен")
    with (
        patch.object(tasks, "index_pdf", side_effect=error),
        patch.object(tasks.index_document, "retry", side_effect=Retry()) as retry,
    ):
        run_task()

    assert retry.call_args.kwargs["exc"] is error


def test_task_does_not_retry_document_error():
    with (
        patch.object(tasks, "index_pdf", side_effect=DocumentError("скан")),
        patch.object(tasks.index_document, "retry") as retry,
    ):
        outcome = run_task()

    assert outcome.failed()
    retry.assert_not_called()


def fake_async_result(state, info=None):
    return SimpleNamespace(state=state, info=info, result=info)


def status(state, info=None):
    with patch.object(tasks, "AsyncResult", return_value=fake_async_result(state, info)):
        return tasks.task_status("task-1", "doc-1")


def test_task_status_progress_shape():
    response = status("PROGRESS", {"stage": "embedding", "current": 2, "total": 5})

    assert response.state == "PROGRESS"
    assert response.progress.stage == "embedding"
    assert response.doc_id == "doc-1"


def test_task_status_success_has_result():
    response = status("SUCCESS", {"doc_id": "doc-1", "pages": 3, "chunks": 12})

    assert response.result.chunks == 12
    assert response.error is None


def test_task_status_maps_document_error_message():
    response = status("FAILURE", DocumentError("В документе нет текстового слоя"))

    assert response.error == "В документе нет текстового слоя"


def test_task_status_hides_internal_error_details():
    response = status("FAILURE", RuntimeError("secret stack detail"))

    assert "secret" not in response.error
    assert response.error == tasks.INTERNAL_ERROR_MESSAGE


def test_task_retries_when_worker_resources_unavailable():
    error = ConnectionError("Qdrant недоступен при старте worker")
    with (
        patch.object(tasks, "_resources", side_effect=error),
        patch.object(tasks.index_document, "retry", side_effect=Retry()) as retry,
    ):
        run_task()

    assert retry.call_args.kwargs["exc"] is error
