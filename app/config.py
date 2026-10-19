"""Настройки сервиса. Все значения берутся из переменных окружения (или файла .env)."""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.store.models import Fusion


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenRouter: LLM, эмбеддинги и реранкер через один ключ
    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-4.1-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = Field(default=1536, gt=0)
    rerank_model: str = "cohere/rerank-v3.5"

    # BM25 (локально, fastembed)
    sparse_model: str = "Qdrant/bm25"
    sparse_language: str = "russian"
    fastembed_cache_dir: str | None = None

    # Поиск и реранк
    dense_limit: int = Field(default=50, gt=0)
    sparse_limit: int = Field(default=50, gt=0)
    fusion: Fusion = "rrf"
    rerank_candidates: int = Field(default=100, gt=0)
    rerank_top_n: int = Field(default=5, gt=0)
    rerank_min_score: float = Field(default=0.1, ge=0.0, le=1.0)

    # Разбиение и индексация
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    embed_batch_size: int = Field(default=64, gt=0)

    # Подключения
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    redis_url: str = "redis://localhost:6379/0"
    upload_dir: str = "/data/uploads"

    # Ограничения и устойчивость
    max_upload_mb: int = Field(default=50, gt=0)
    max_pages: int = Field(default=100, gt=0)
    max_question_chars: int = Field(default=2000, gt=0)
    retry_attempts: int = Field(default=4, gt=0)
    http_timeout_s: float = Field(default=60.0, gt=0)

    @field_validator("fastembed_cache_dir", mode="before")
    @classmethod
    def _empty_as_default(cls, value: str | None) -> str | None:
        return value or None

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")
        if self.rerank_top_n > self.rerank_candidates:
            raise ValueError("rerank_top_n не может превышать rerank_candidates")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
