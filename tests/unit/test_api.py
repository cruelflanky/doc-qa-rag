from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import fakeredis
import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.config import Settings
from app.errors import DocumentNotFoundError
from app.main import create_app
from app.schemas import AnswerResponse, Citation, DebugInfo, TaskStatusResponse
from app.store.models import StoredDocument

PDF_BYTES = b"%PDF-1.4\n%fake but enough for upload checks\n"


class FakePipeline:
    def __init__(self, error=None):
        self.error = error

    def ask(self, doc_id, question, debug=False):
        if self.error:
            raise self.error
        citation = Citation(
            text="Неустойка 0,1%",
            page=12,
            page_to=12,
            section="4.2 Штрафные санкции",
            chunk_index=41,
        )
        response = AnswerResponse(answer="0,1% в день", found=True, citations=[citation])
        if debug:
            info = DebugInfo(
                dense_hits=30,
                sparse_hits=12,
                fused_candidates=30,
                fusion="rrf",
                reranked=[],
                after_threshold=1,
                used_in_answer=[41],
                timings_ms={},
            )
            response = response.model_copy(update={"debug": info})
        return response


@pytest.fixture
def ctx(tmp_path):
    settings = Settings(
        _env_file=None, openrouter_api_key="k", upload_dir=str(tmp_path), max_upload_mb=1
    )
    redis = fakeredis.FakeRedis(decode_responses=True)
    store = SimpleNamespace(
        client=MagicMock(),
        list_documents=lambda: [
            StoredDocument("doc-1", "Договор", 80, 300, datetime(2026, 10, 19, tzinfo=UTC))
        ],
    )
    state = SimpleNamespace(settings=settings, redis=redis, store=store, pipeline=FakePipeline())
    app = create_app()
    app.dependency_overrides[deps.get_settings_dep] = lambda: state.settings
    app.dependency_overrides[deps.get_redis] = lambda: state.redis
    app.dependency_overrides[deps.get_store] = lambda: state.store
    app.dependency_overrides[deps.get_pipeline] = lambda: state.pipeline
    state.client = TestClient(app, raise_server_exceptions=False)
    return state


def upload(client, content=PDF_BYTES, name="contract.pdf"):
    with patch("app.api.documents.index_document") as task:
        task.delay.return_value = SimpleNamespace(id="task-1")
        response = client.post("/documents", files={"file": (name, content, "application/pdf")})
    return response, task


def test_upload_pdf_returns_202_and_enqueues(ctx):
    response, task = upload(ctx.client)

    assert response.status_code == 202
    body = response.json()
    assert body["task_id"] == "task-1"
    assert body["status_url"] == "/tasks/task-1"
    doc_id, path, title = task.delay.call_args.args
    assert doc_id == body["doc_id"]
    assert title == "contract"
    assert Path(path).read_bytes() == PDF_BYTES
    assert ctx.redis.get("task:task-1") == doc_id


def test_upload_rejects_non_pdf_with_pdf_extension(ctx):
    response, task = upload(ctx.client, content=b"PK\x03\x04 zip archive")

    assert response.status_code == 400
    assert response.json()["detail"] == "Ожидается PDF-файл"
    task.delay.assert_not_called()


def test_upload_rejects_empty_file(ctx):
    response, _ = upload(ctx.client, content=b"")

    assert response.status_code == 400
    assert response.json()["detail"] == "Пустой файл"


def test_upload_rejects_too_large_file(ctx, tmp_path):
    response, task = upload(ctx.client, content=PDF_BYTES + b"0" * (1024 * 1024 + 1))

    assert response.status_code == 413
    task.delay.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_task_status_unknown_returns_404(ctx):
    response = ctx.client.get("/tasks/unknown")

    assert response.status_code == 404
    assert response.json()["detail"] == "Задача не найдена"


def test_task_status_progress_shape(ctx):
    ctx.redis.set("task:task-1", "doc-1")
    status = TaskStatusResponse(
        task_id="task-1",
        doc_id="doc-1",
        state="PROGRESS",
        progress={"stage": "embedding", "current": 2, "total": 5},
    )
    with patch("app.api.tasks.task_status", return_value=status) as task_status:
        response = ctx.client.get("/tasks/task-1")

    task_status.assert_called_once_with("task-1", "doc-1")
    assert response.json()["progress"] == {"stage": "embedding", "current": 2, "total": 5}


def test_list_documents(ctx):
    response = ctx.client.get("/documents")

    assert response.status_code == 200
    assert response.json()[0]["doc_id"] == "doc-1"
    assert response.json()[0]["chunks"] == 300


def test_ask_returns_tz_shape_without_debug_key(ctx):
    response = ctx.client.post("/documents/doc-1/ask", json={"question": "Какой штраф?"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "found", "citations"}
    assert body["citations"][0]["page"] == 12


def test_ask_debug_includes_debug_block(ctx):
    response = ctx.client.post(
        "/documents/doc-1/ask", json={"question": "Какой штраф?", "debug": True}
    )

    assert response.json()["debug"]["dense_hits"] == 30


def test_ask_unknown_document_404(ctx):
    ctx.pipeline = FakePipeline(error=DocumentNotFoundError("doc-x"))

    response = ctx.client.post("/documents/doc-x/ask", json={"question": "Какой штраф?"})

    assert response.status_code == 404
    assert response.json()["detail"] == "Документ не найден или ещё индексируется"


def test_ask_blank_question_422(ctx):
    response = ctx.client.post("/documents/doc-1/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_upstream_unavailable_returns_503_without_internals(ctx):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/rerank")
    error = httpx.HTTPStatusError(
        "upstream secret", request=request, response=httpx.Response(503, request=request)
    )
    ctx.pipeline = FakePipeline(error=error)

    response = ctx.client.post("/documents/doc-1/ask", json={"question": "Какой штраф?"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Внешний сервис временно недоступен, повторите запрос позже"
    }


def test_unexpected_error_returns_500_without_internals(ctx):
    ctx.pipeline = FakePipeline(error=RuntimeError("secret detail"))

    response = ctx.client.post("/documents/doc-1/ask", json={"question": "Какой штраф?"})

    assert response.status_code == 500
    assert "secret" not in response.text


def test_health_reports_components(ctx):
    response = ctx.client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "redis": "ok", "qdrant": "ok"}


def test_health_503_when_qdrant_down(ctx):
    ctx.store.client.get_collections.side_effect = ConnectionError("down")

    response = ctx.client.get("/health")

    assert response.status_code == 503
    assert response.json()["qdrant"] == "unavailable"
