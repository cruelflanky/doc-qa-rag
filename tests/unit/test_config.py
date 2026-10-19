import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.fixture
def minimal_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_settings_defaults_load_from_env(minimal_env):
    settings = Settings(_env_file=None)

    assert settings.openrouter_api_key.get_secret_value() == "test-key"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.embedding_model == "openai/text-embedding-3-small"
    assert settings.embedding_dim == 1536
    assert settings.sparse_model == "Qdrant/bm25"
    assert settings.sparse_language == "russian"
    assert settings.rerank_model == "cohere/rerank-v3.5"
    assert settings.fusion == "rrf"
    assert settings.qdrant_collection == "documents"
    assert settings.max_pages == 100


def test_overlap_must_be_less_than_size(minimal_env):
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=500, chunk_overlap=500)


def test_top_n_cannot_exceed_candidates(minimal_env):
    with pytest.raises(ValidationError, match="rerank_top_n"):
        Settings(_env_file=None, rerank_candidates=5, rerank_top_n=6)


def test_fusion_rejects_unknown_value(minimal_env):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, fusion="sum")


def test_min_score_must_be_within_unit_interval(minimal_env):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, rerank_min_score=1.5)


def test_reranker_sees_whole_union_of_branches_by_default():
    # RRF вытесняет кандидатов, найденных одной ветвью, — реранкер должен видеть объединение целиком
    settings = Settings(_env_file=None)

    assert (settings.dense_limit, settings.sparse_limit, settings.rerank_candidates) == (
        50,
        50,
        100,
    )
