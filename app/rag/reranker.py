"""Реранкер через OpenRouter POST /rerank (модели Cohere) в виде LangChain-компрессора.

Cross-encoder смотрит на пару «вопрос + фрагмент» целиком, поэтому точнее оценивает
релевантность, чем сравнение готовых векторов. Готовой интеграции OpenRouter-rerank в LangChain
нет, поэтому класс свой: наследник BaseDocumentCompressor.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import SecretStr

from app.config import Settings
from app.retry import with_retry


@dataclass(frozen=True)
class RerankResult:
    index: int  # позиция во входном списке текстов
    score: float


class OpenRouterReranker(BaseDocumentCompressor):
    api_key: SecretStr
    base_url: str
    model: str
    top_n: int
    timeout_s: float
    retry_attempts: int

    def _post(self, payload: dict) -> dict:
        response = httpx.post(
            f"{self.base_url}/rerank",
            headers={"Authorization": f"Bearer {self.api_key.get_secret_value()}"},
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def rerank(self, query: str, texts: list[str]) -> list[RerankResult]:
        if not texts:
            return []
        payload = {"model": self.model, "query": query, "documents": texts, "top_n": self.top_n}
        data = with_retry(self.retry_attempts)(self._post)(payload)
        results = [RerankResult(r["index"], r["relevance_score"]) for r in data["results"]]
        return sorted(results, key=lambda r: r.score, reverse=True)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> list[Document]:
        ranked = self.rerank(query, [doc.page_content for doc in documents])
        return [
            Document(
                page_content=documents[r.index].page_content,
                metadata={**documents[r.index].metadata, "rerank_score": r.score},
            )
            for r in ranked
        ]


def build_reranker(settings: Settings) -> OpenRouterReranker:
    return OpenRouterReranker(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.rerank_model,
        top_n=settings.rerank_top_n,
        timeout_s=settings.http_timeout_s,
        retry_attempts=settings.retry_attempts,
    )
