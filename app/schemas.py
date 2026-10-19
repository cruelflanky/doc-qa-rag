"""Pydantic-схемы API и структурированного ответа LLM."""

from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import get_settings


class Citation(BaseModel):
    text: str
    page: int = Field(description="Страница начала фрагмента")
    page_to: int = Field(description="Страница конца фрагмента")
    section: str
    chunk_index: int


class RerankedItem(BaseModel):
    chunk_index: int
    rerank_score: float
    fused_rank: int
    in_dense: bool
    in_sparse: bool
    page: int
    page_to: int
    section: str


class DebugInfo(BaseModel):
    dense_hits: int = Field(description="Сколько фрагментов дала dense-ветвь")
    sparse_hits: int = Field(description="Сколько фрагментов дала BM25-ветвь")
    fused_candidates: int = Field(description="Кандидатов после объединения в Qdrant")
    fusion: str
    reranked: list[RerankedItem] = Field(description="Top-N после реранкера")
    after_threshold: int = Field(description="Осталось после порога RERANK_MIN_SCORE")
    used_in_answer: list[int] = Field(
        description="chunk_index фрагментов, на которые сослался ответ"
    )
    timings_ms: dict[str, int]


class AnswerResponse(BaseModel):
    answer: str
    found: bool
    citations: list[Citation]
    debug: DebugInfo | None = None

    @model_validator(mode="after")
    def _citations_match_found(self) -> Self:
        if self.found and not self.citations:
            raise ValueError("found=true требует хотя бы одной цитаты")
        if not self.found and self.citations:
            raise ValueError("found=false несовместим с цитатами")
        return self


class AskRequest(BaseModel):
    question: str
    debug: bool = False

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Вопрос не может быть пустым")
        limit = get_settings().max_question_chars
        if len(question) > limit:
            raise ValueError(f"Вопрос длиннее {limit} символов")
        return question


class LLMAnswer(BaseModel):
    """Структурированный ответ модели. Цитаты модель не пишет — только номера фрагментов."""

    found: bool = Field(description="true, если во фрагментах есть сведения для ответа")
    answer: str = Field(
        description="Ответ на русском только по фрагментам; пусто, если found=false"
    )
    used_fragments: list[int] = Field(
        description="Номера фрагментов [N], на которые опирается каждое утверждение ответа"
    )


class UploadResponse(BaseModel):
    doc_id: str
    task_id: str
    status_url: str


class TaskProgress(BaseModel):
    stage: str
    current: int
    total: int


class IndexResult(BaseModel):
    doc_id: str
    pages: int
    chunks: int


# Основные состояния Celery плюс PROGRESS; редкие (REVOKED, REJECTED, …) отдаются как есть
TaskState = str


class TaskStatusResponse(BaseModel):
    task_id: str
    doc_id: str
    state: TaskState
    progress: TaskProgress | None = None
    result: IndexResult | None = None
    error: str | None = None


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    pages: int
    chunks: int
    uploaded_at: datetime
