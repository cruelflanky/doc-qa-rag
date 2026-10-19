"""Структуры данных хранилища фрагментов."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DocMeta:
    """Сведения о документе; денормализуются в payload каждого фрагмента."""

    doc_id: str
    title: str
    pages: int
    uploaded_at: datetime


@dataclass(frozen=True)
class Candidate:
    """Фрагмент-кандидат из гибридного поиска."""

    point_id: str
    chunk_index: int
    text: str
    page_from: int
    page_to: int
    section: str
    fused_score: float
    fused_rank: int  # позиция после объединения ветвей, с 1


@dataclass(frozen=True)
class SearchTrace:
    """Результат поиска в debug-режиме: итог и что дала каждая ветвь отдельно."""

    candidates: list[Candidate]
    dense_ids: list[str]
    sparse_ids: list[str]


@dataclass(frozen=True)
class StoredDocument:
    doc_id: str
    title: str
    pages: int
    chunks: int
    uploaded_at: datetime
