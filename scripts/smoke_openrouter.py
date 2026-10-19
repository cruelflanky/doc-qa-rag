"""Живая проверка OpenRouter: эмбеддинги, реранк и chat.

Запуск: uv run python scripts/smoke_openrouter.py  (ключ берётся из .env)
"""

import sys

import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import get_settings


def check_embeddings() -> str:
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        check_embedding_ctx_length=False,
    )
    vector = embeddings.embed_query("Штраф за просрочку поставки")
    assert len(vector) == settings.embedding_dim, f"размерность {len(vector)}"
    return f"dim={len(vector)}"


def check_rerank() -> str:
    settings = get_settings()
    response = httpx.post(
        f"{settings.openrouter_base_url}/rerank",
        headers={"Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"},
        json={
            "model": settings.rerank_model,
            "query": "Какой штраф за просрочку поставки?",
            "documents": [
                "Покупатель оплачивает товар в течение 10 дней.",
                "За просрочку поставки Поставщик уплачивает неустойку 0,1% в день.",
            ],
            "top_n": 2,
        },
        timeout=60,
    )
    response.raise_for_status()
    results = response.json()["results"]
    best = max(results, key=lambda r: r["relevance_score"])
    assert best["index"] == 1, f"неожиданный порядок: {results}"
    return f"best_index=1 score={best['relevance_score']:.3f}"


def check_chat() -> str:
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=0,
    )
    reply = llm.invoke("Ответь одним словом: столица Франции?")
    return f"model={settings.llm_model} reply={reply.content!r}"


def main() -> int:
    failed = False
    for name, check in [("embeddings", check_embeddings), ("rerank", check_rerank),
                        ("chat", check_chat)]:
        try:
            print(f"OK   {name}: {check()}")
        except Exception as exc:  # noqa: BLE001 — smoke-скрипт показывает любую ошибку
            failed = True
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
