"""Эмбеддинги через LangChain: dense — OpenAIEmbeddings (OpenRouter), sparse — BM25 (fastembed)."""

from typing import Protocol

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from qdrant_client import models

from app.config import Settings
from app.retry import with_retry


class Embedder(Protocol):
    def embed_dense(self, texts: list[str]) -> list[list[float]]: ...

    def embed_sparse(self, texts: list[str]) -> list[models.SparseVector]: ...

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]: ...


def _to_qdrant(vector: SparseVector) -> models.SparseVector:
    return models.SparseVector(indices=list(vector.indices), values=list(vector.values))


class LangChainEmbedder:
    def __init__(
        self,
        dense: Embeddings,
        sparse: SparseEmbeddings,
        batch_size: int,
        retry_attempts: int,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.batch_size = batch_size
        self._retry = with_retry(retry_attempts)

    def _batches(self, texts: list[str]) -> list[list[str]]:
        return [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        embed = self._retry(self.dense.embed_documents)
        return [vector for batch in self._batches(texts) for vector in embed(batch)]

    def embed_sparse(self, texts: list[str]) -> list[models.SparseVector]:
        return [_to_qdrant(v) for v in self.sparse.embed_documents(texts)]

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        dense = self._retry(self.dense.embed_query)(text)
        return dense, _to_qdrant(self.sparse.embed_query(text))


def build_embedder(settings: Settings) -> LangChainEmbedder:
    dense = OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        # OpenRouter ждёт строки; без этого LangChain отправляет токены tiktoken
        check_embedding_ctx_length=False,
        max_retries=0,  # ретраи делаем сами, чтобы они не перемножались
        timeout=settings.http_timeout_s,
    )
    sparse = FastEmbedSparse(
        model_name=settings.sparse_model,
        language=settings.sparse_language,
        cache_dir=settings.fastembed_cache_dir,
    )
    return LangChainEmbedder(
        dense, sparse, batch_size=settings.embed_batch_size, retry_attempts=settings.retry_attempts
    )
