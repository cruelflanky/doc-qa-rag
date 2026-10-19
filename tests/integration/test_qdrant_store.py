import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from langchain_qdrant import FastEmbedSparse
from qdrant_client import QdrantClient, models

from app.store.models import DocMeta, SearchParams
from app.store.qdrant_store import QdrantStore
from tests.fakes import fake_dense, make_chunks

pytestmark = pytest.mark.integration

DIM = 8
FILLER = [
    f"Пункт {i}. Стороны исполняют обязательства добросовестно, раздел {i}." for i in range(20)
]


@pytest.fixture(scope="module")
def sparse_model():
    return FastEmbedSparse(model_name="Qdrant/bm25", language="russian")


@pytest.fixture
def store():
    client = QdrantClient(url="http://localhost:6333")
    collection = f"test_{uuid.uuid4().hex[:8]}"
    store = QdrantStore(client, collection, dense_dim=DIM, retry_attempts=1)
    store.ensure_collection()
    yield store
    client.delete_collection(collection)


def to_qdrant(vector) -> models.SparseVector:
    return models.SparseVector(indices=vector.indices, values=vector.values)


def index(store, sparse_model, texts, *, ready=True, title="Договор") -> str:
    doc_id = str(uuid.uuid4())
    meta = DocMeta(doc_id=doc_id, title=title, pages=len(texts), uploaded_at=datetime.now(UTC))
    chunks = make_chunks(texts)
    store.upsert_chunks(
        meta,
        chunks,
        dense=[fake_dense(t, DIM) for t in texts],
        sparse=[to_qdrant(v) for v in sparse_model.embed_documents(texts)],
        total_chunks=len(chunks),
    )
    if ready:
        store.mark_ready(doc_id)
    return doc_id


def search(store, sparse_model, doc_id, query, **overrides):
    params = SearchParams(
        **(dict(dense_limit=10, sparse_limit=10, fusion="rrf", limit=10) | overrides)
    )
    return store.hybrid_search(
        doc_id, fake_dense(query, DIM), to_qdrant(sparse_model.embed_query(query)), params
    )


def test_ensure_collection_is_idempotent_and_creates_indexes(store):
    store.ensure_collection()

    schema = store.client.get_collection(store.collection).payload_schema
    assert {"doc_id", "ready", "chunk_index"} <= set(schema)


def test_hybrid_search_finds_exact_term_via_bm25(store, sparse_model):
    texts = [*FILLER, "Реквизиты Поставщика: ИНН 7701234567, КПП 770101001."]
    doc_id = index(store, sparse_model, texts)

    results = search(store, sparse_model, doc_id, "ИНН 7701234567")

    assert "7701234567" in " ".join(c.text for c in results[:3])


def test_bm25_russian_stemming(sparse_model):
    first = sparse_model.embed_query("поставки")
    second = sparse_model.embed_query("поставка")

    assert set(first.indices) & set(second.indices)


def test_search_is_isolated_by_doc_id(store, sparse_model):
    doc_a = index(store, sparse_model, ["Неустойка за просрочку поставки 0,1% в день."])
    index(store, sparse_model, ["Неустойка за просрочку поставки 0,3% в день."])

    results = search(store, sparse_model, doc_a, "неустойка за просрочку поставки")

    assert [c.text for c in results] == ["Неустойка за просрочку поставки 0,1% в день."]


def test_not_ready_document_is_invisible(store, sparse_model):
    doc_id = index(store, sparse_model, ["Текст договора о поставке."], ready=False)

    assert store.document_ready(doc_id) is False
    assert search(store, sparse_model, doc_id, "договор поставки") == []
    assert all(d.doc_id != doc_id for d in store.list_documents())

    store.mark_ready(doc_id)

    assert store.document_ready(doc_id) is True
    assert len(search(store, sparse_model, doc_id, "договор поставки")) == 1


def test_delete_document_removes_all_points(store, sparse_model):
    doc_id = index(store, sparse_model, FILLER[:5])

    store.delete_document(doc_id)

    assert store.document_ready(doc_id) is False
    assert store.client.count(store.collection, exact=True).count == 0


def test_hybrid_search_is_single_query_call(store, sparse_model):
    doc_id = index(store, sparse_model, FILLER[:5])

    with (
        patch.object(store.client, "query_points", wraps=store.client.query_points) as single,
        patch.object(store.client, "query_batch_points") as batch,
    ):
        search(store, sparse_model, doc_id, "обязательства сторон")

    assert single.call_count == 1
    assert batch.call_count == 0
    assert len(single.call_args.kwargs["prefetch"]) == 2


def test_debug_trace_reports_branch_ids(store, sparse_model):
    doc_id = index(store, sparse_model, FILLER[:5])
    query = "обязательства добросовестно"
    dense, sparse = fake_dense(query, DIM), to_qdrant(sparse_model.embed_query(query))
    params = SearchParams(dense_limit=10, sparse_limit=10, fusion="rrf", limit=10)

    trace = store.hybrid_search_debug(doc_id, dense, sparse, params)

    assert trace.dense_ids
    assert trace.sparse_ids
    plain = store.hybrid_search(doc_id, dense, sparse, params)
    # при равных RRF-оценках порядок не определён, поэтому сравниваем набор и оценки
    assert {c.point_id for c in trace.candidates} == {c.point_id for c in plain}
    assert [c.fused_score for c in trace.candidates] == [c.fused_score for c in plain]
    assert [c.fused_rank for c in plain] == list(range(1, len(plain) + 1))


def test_list_documents_returns_meta_from_payload(store, sparse_model):
    doc_id = index(store, sparse_model, FILLER[:3], title="Договор поставки")

    docs = store.list_documents()

    assert len(docs) == 1
    assert docs[0].doc_id == doc_id
    assert docs[0].title == "Договор поставки"
    assert docs[0].pages == 3
    assert docs[0].chunks == 3


def test_dbsf_fusion_is_supported(store, sparse_model):
    doc_id = index(store, sparse_model, FILLER[:5])

    assert search(store, sparse_model, doc_id, "обязательства", fusion="dbsf")
