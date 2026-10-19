"""Зависимости FastAPI: один экземпляр клиентов на процесс; в тестах подменяются."""

from functools import lru_cache

from redis import Redis

from app.config import Settings, get_settings
from app.rag.pipeline import QAPipeline, build_pipeline, build_store
from app.store.qdrant_store import QdrantStore


def get_settings_dep() -> Settings:
    return get_settings()


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


@lru_cache
def get_store() -> QdrantStore:
    return build_store(get_settings())


@lru_cache
def get_pipeline() -> QAPipeline:
    return build_pipeline(get_settings(), store=get_store())
