import uuid

import pytest

from app.config import Settings
from app.errors import DocumentError
from app.ingest.indexer import index_pdf
from tests.fakes import FakeEmbedder
from tests.fixtures.make_pdf import make_pdf

LINE = "Поставщик обязан передать товар Покупателю в согласованный срок."


class RecordingStore:
    def __init__(self):
        self.calls: list[str] = []
        self.upserted = 0

    def delete_document(self, doc_id):
        self.calls.append("delete_document")

    def upsert_chunks(self, meta, chunks, dense, sparse, *, total_chunks):
        self.calls.append("upsert_chunks")
        self.upserted += len(chunks)
        self.total_chunks = total_chunks

    def mark_ready(self, doc_id):
        self.calls.append("mark_ready")


class FailingEmbedder(FakeEmbedder):
    def embed_dense(self, texts):
        raise ConnectionError("OpenRouter недоступен")


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return Settings(_env_file=None, chunk_size=200, chunk_overlap=20, embed_batch_size=4)


@pytest.fixture
def pdf(tmp_path):
    page = "1. ПРЕДМЕТ ДОГОВОРА\n" + "\n".join([LINE] * 20)
    return make_pdf(tmp_path / "doc.pdf", [page, page])


def run(pdf, settings, store, embedder=None, progress=None):
    return index_pdf(
        pdf,
        str(uuid.uuid4()),
        "Договор",
        store=store,
        embedder=embedder or FakeEmbedder(),
        settings=settings,
        on_progress=lambda *event: progress.append(event) if progress is not None else None,
    )


def test_index_pdf_happy_path_marks_ready_last(pdf, settings):
    store = RecordingStore()

    result = run(pdf, settings, store)

    assert store.calls[0] == "delete_document"
    assert store.calls[-1] == "mark_ready"
    assert result.chunks == store.upserted == store.total_chunks
    assert result.pages == 2
    assert store.calls.count("upsert_chunks") > 1  # пишем батчами


def test_index_pdf_reports_progress_stages_in_order(pdf, settings):
    events = []

    run(pdf, settings, RecordingStore(), progress=events)

    stages = list(dict.fromkeys(stage for stage, _, _ in events))
    assert stages == ["parsing", "chunking", "embedding", "upserting", "finalizing"]


def test_index_pdf_cleans_up_on_embedding_failure(pdf, settings):
    store = RecordingStore()

    with pytest.raises(ConnectionError):
        run(pdf, settings, store, embedder=FailingEmbedder())

    assert store.calls[-1] == "delete_document"
    assert "mark_ready" not in store.calls


def test_index_pdf_propagates_document_error_without_writes(tmp_path, settings):
    store = RecordingStore()
    scan = make_pdf(tmp_path / "scan.pdf", ["", ""])

    with pytest.raises(DocumentError):
        run(scan, settings, store)

    assert "upsert_chunks" not in store.calls
