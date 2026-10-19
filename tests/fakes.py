"""Детерминированные подмены внешних зависимостей для тестов."""

import hashlib
import math

from qdrant_client import models

from app.ingest.chunker import Chunk


def fake_dense(text: str, dim: int) -> list[float]:
    """Псевдослучайный, но детерминированный единичный вектор по тексту."""
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 127.5 for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def make_chunks(texts: list[str], section: str = "1. ПРЕДМЕТ ДОГОВОРА") -> list[Chunk]:
    chunks = []
    position = 0
    for index, text in enumerate(texts):
        chunks.append(
            Chunk(
                chunk_index=index,
                text=text,
                char_start=position,
                char_end=position + len(text),
                page_from=index + 1,
                page_to=index + 1,
                section=section,
            )
        )
        position += len(text) + 1
    return chunks


class FakeEmbedder:
    """Реализация протокола Embedder без сети: dense — хеш, sparse — мешок слов."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [fake_dense(text, self.dim) for text in texts]

    def embed_sparse(self, texts: list[str]) -> list[models.SparseVector]:
        return [_bag_of_words(text) for text in texts]

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        return fake_dense(text, self.dim), _bag_of_words(text)


def _bag_of_words(text: str) -> models.SparseVector:
    counts: dict[int, float] = {}
    for word in text.lower().split():
        index = int(hashlib.md5(word.encode()).hexdigest()[:6], 16)
        counts[index] = counts.get(index, 0.0) + 1.0
    return models.SparseVector(indices=list(counts), values=list(counts.values()))
