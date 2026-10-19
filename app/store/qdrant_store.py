"""Работа с Qdrant: одна коллекция на все документы, в каждой точке dense- и BM25-вектор.

Документы разделяются фильтром по doc_id. Флаг ready=true выставляется только после полной
индексации, поэтому частично записанный документ не виден ни поиску, ни списку.
"""

import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.ingest.chunker import Chunk
from app.retry import with_retry
from app.store.models import Candidate, DocMeta, SearchParams, SearchTrace, StoredDocument

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
SCROLL_PAGE = 256
HTTP_CONFLICT = 409

FUSIONS = {"rrf": models.Fusion.RRF, "dbsf": models.Fusion.DBSF}
PAYLOAD_INDEXES = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "ready": models.PayloadSchemaType.BOOL,
    "chunk_index": models.PayloadSchemaType.INTEGER,
}

T = TypeVar("T")


def _match(key: str, value: str | bool | int) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _doc_filter(doc_id: str, *, only_ready: bool) -> models.Filter:
    conditions = [_match("doc_id", doc_id)]
    if only_ready:
        conditions.append(_match("ready", True))
    return models.Filter(must=conditions)


def _point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.UUID(doc_id), str(chunk_index)))


def _to_candidate(point: models.ScoredPoint, rank: int) -> Candidate:
    payload = point.payload or {}
    return Candidate(
        point_id=str(point.id),
        chunk_index=payload["chunk_index"],
        text=payload["text"],
        page_from=payload["page_from"],
        page_to=payload["page_to"],
        section=payload["section"],
        fused_score=point.score,
        fused_rank=rank,
    )


class QdrantStore:
    def __init__(
        self, client: QdrantClient, collection: str, dense_dim: int, retry_attempts: int
    ) -> None:
        self.client = client
        self.collection = collection
        self.dense_dim = dense_dim
        self.retry_attempts = retry_attempts

    def _call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return with_retry(self.retry_attempts)(func)(*args, **kwargs)

    def ensure_collection(self) -> None:
        if not self._call(self.client.collection_exists, self.collection):
            try:
                self._call(
                    self.client.create_collection,
                    self.collection,
                    vectors_config={
                        DENSE_VECTOR: models.VectorParams(
                            size=self.dense_dim, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
                    },
                )
            except UnexpectedResponse as exc:  # api и worker могут создавать одновременно
                if exc.status_code != HTTP_CONFLICT:
                    raise
        for field, schema in PAYLOAD_INDEXES.items():
            self._call(self.client.create_payload_index, self.collection, field, schema)

    def upsert_chunks(
        self,
        meta: DocMeta,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[models.SparseVector],
        *,
        total_chunks: int,
    ) -> None:
        points = [
            models.PointStruct(
                id=_point_id(meta.doc_id, chunk.chunk_index),
                vector={DENSE_VECTOR: dense_vec, SPARSE_VECTOR: sparse_vec},
                payload={
                    "doc_id": meta.doc_id,
                    "doc_title": meta.title,
                    "doc_pages": meta.pages,
                    "doc_chunks": total_chunks,
                    "uploaded_at": meta.uploaded_at.isoformat(),
                    "chunk_index": chunk.chunk_index,
                    "page_from": chunk.page_from,
                    "page_to": chunk.page_to,
                    "section": chunk.section,
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "ready": False,
                },
            )
            for chunk, dense_vec, sparse_vec in zip(chunks, dense, sparse, strict=True)
        ]
        self._call(self.client.upsert, self.collection, points=points, wait=True)

    def mark_ready(self, doc_id: str) -> None:
        self._call(
            self.client.set_payload,
            self.collection,
            payload={"ready": True},
            points=_doc_filter(doc_id, only_ready=False),
            wait=True,
        )

    def delete_document(self, doc_id: str) -> None:
        self._call(
            self.client.delete,
            self.collection,
            points_selector=models.FilterSelector(filter=_doc_filter(doc_id, only_ready=False)),
            wait=True,
        )

    def document_ready(self, doc_id: str) -> bool:
        result = self._call(
            self.client.count,
            self.collection,
            count_filter=_doc_filter(doc_id, only_ready=True),
            exact=True,
        )
        return result.count > 0

    def list_documents(self) -> list[StoredDocument]:
        first_chunks = models.Filter(must=[_match("chunk_index", 0), _match("ready", True)])
        documents: list[StoredDocument] = []
        offset = None
        while True:
            points, offset = self._call(
                self.client.scroll,
                self.collection,
                scroll_filter=first_chunks,
                limit=SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            documents.extend(
                StoredDocument(
                    doc_id=p.payload["doc_id"],
                    title=p.payload["doc_title"],
                    pages=p.payload["doc_pages"],
                    chunks=p.payload["doc_chunks"],
                    uploaded_at=datetime.fromisoformat(p.payload["uploaded_at"]),
                )
                for p in points
            )
            if offset is None:
                return sorted(documents, key=lambda d: d.uploaded_at, reverse=True)

    def _fused_request(
        self,
        doc_id: str,
        dense: list[float],
        sparse: models.SparseVector,
        params: SearchParams,
    ) -> models.QueryRequest:
        doc_filter = _doc_filter(doc_id, only_ready=True)
        return models.QueryRequest(
            prefetch=[
                models.Prefetch(
                    query=dense, using=DENSE_VECTOR, limit=params.dense_limit, filter=doc_filter
                ),
                models.Prefetch(
                    query=sparse, using=SPARSE_VECTOR, limit=params.sparse_limit, filter=doc_filter
                ),
            ],
            query=models.FusionQuery(fusion=FUSIONS[params.fusion]),
            limit=params.limit,
            with_payload=True,
        )

    def hybrid_search(
        self, doc_id: str, dense: list[float], sparse: models.SparseVector, params: SearchParams
    ) -> list[Candidate]:
        """Один запрос к Qdrant: обе ветви в prefetch, объединение — FusionQuery на сервере."""
        request = self._fused_request(doc_id, dense, sparse, params)
        response = self._call(
            self.client.query_points,
            self.collection,
            prefetch=request.prefetch,
            query=request.query,
            limit=request.limit,
            with_payload=True,
        )
        return [_to_candidate(p, rank) for rank, p in enumerate(response.points, start=1)]

    def hybrid_search_debug(
        self, doc_id: str, dense: list[float], sparse: models.SparseVector, params: SearchParams
    ) -> SearchTrace:
        """Debug: в одном batch-вызове основной запрос плюс каждая ветвь отдельно для подсчёта."""
        doc_filter = _doc_filter(doc_id, only_ready=True)
        requests = [
            self._fused_request(doc_id, dense, sparse, params),
            models.QueryRequest(
                query=dense, using=DENSE_VECTOR, limit=params.dense_limit, filter=doc_filter
            ),
            models.QueryRequest(
                query=sparse, using=SPARSE_VECTOR, limit=params.sparse_limit, filter=doc_filter
            ),
        ]
        fused, dense_only, sparse_only = self._call(
            self.client.query_batch_points, self.collection, requests=requests
        )
        return SearchTrace(
            candidates=[_to_candidate(p, rank) for rank, p in enumerate(fused.points, start=1)],
            dense_ids=[str(p.id) for p in dense_only.points],
            sparse_ids=[str(p.id) for p in sparse_only.points],
        )
