"""Ответ на вопрос: гибридный поиск → реранк → LLM → постобработка.

Каждый запрос независим: истории диалога и состояния между запросами нет.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from app.config import Settings
from app.embeddings import Embedder, build_embedder
from app.errors import DocumentNotFoundError
from app.rag.answer import build_messages, finalize, not_found_response
from app.rag.reranker import build_reranker
from app.retry import with_retry
from app.schemas import AnswerResponse, DebugInfo, LLMAnswer, RerankedItem
from app.store.models import Candidate, SearchTrace
from app.store.qdrant_store import QdrantStore


@dataclass(frozen=True)
class RankedFragment:
    candidate: Candidate
    score: float


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, int] = {}

    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        yield
        self.ms[name] = round((time.perf_counter() - start) * 1000)


class QAPipeline:
    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        reranker: BaseDocumentCompressor,
        llm: Runnable,
        settings: Settings,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.llm = llm
        self.settings = settings

    def _search(self, doc_id: str, question: str, debug: bool, timer: _Timer) -> SearchTrace:
        with timer.measure("embed"):
            dense, sparse = self.embedder.embed_query(question)
        params = {
            "dense_limit": self.settings.dense_limit,
            "sparse_limit": self.settings.sparse_limit,
            "fusion": self.settings.fusion,
            "limit": self.settings.rerank_candidates,
        }
        with timer.measure("search"):
            if debug:
                return self.store.hybrid_search_debug(doc_id, dense, sparse, **params)
            candidates = self.store.hybrid_search(doc_id, dense, sparse, **params)
            return SearchTrace(candidates=candidates, dense_ids=[], sparse_ids=[])

    def _rerank(self, question: str, candidates: list[Candidate]) -> list[RankedFragment]:
        documents = [
            Document(page_content=c.text, metadata={"candidate_idx": i})
            for i, c in enumerate(candidates)
        ]
        reranked = self.reranker.compress_documents(documents, question)
        return [
            RankedFragment(candidates[d.metadata["candidate_idx"]], d.metadata["rerank_score"])
            for d in reranked[: self.settings.rerank_top_n]
        ]

    def ask(self, doc_id: str, question: str, debug: bool = False) -> AnswerResponse:
        if not self.store.document_ready(doc_id):
            raise DocumentNotFoundError(doc_id)
        timer = _Timer()
        trace = self._search(doc_id, question, debug, timer)

        ranked: list[RankedFragment] = []
        if trace.candidates:
            with timer.measure("rerank"):
                ranked = self._rerank(question, trace.candidates)
        relevant = [r for r in ranked if r.score >= self.settings.rerank_min_score]

        if relevant:
            fragments = [r.candidate for r in relevant]
            with timer.measure("llm"):
                llm_answer: LLMAnswer = with_retry(self.settings.retry_attempts)(self.llm.invoke)(
                    build_messages(question, fragments)
                )
            response = finalize(llm_answer, fragments)
        else:
            response = not_found_response()  # нечего показать модели — не вызываем её вовсе

        if not debug:
            return response
        info = self._debug_info(trace, ranked, relevant, response, timer)
        return response.model_copy(update={"debug": info})

    def _debug_info(
        self,
        trace: SearchTrace,
        ranked: list[RankedFragment],
        relevant: list[RankedFragment],
        response: AnswerResponse,
        timer: _Timer,
    ) -> DebugInfo:
        dense_ids, sparse_ids = set(trace.dense_ids), set(trace.sparse_ids)
        return DebugInfo(
            dense_hits=len(trace.dense_ids),
            sparse_hits=len(trace.sparse_ids),
            fused_candidates=len(trace.candidates),
            fusion=self.settings.fusion,
            reranked=[
                RerankedItem(
                    chunk_index=r.candidate.chunk_index,
                    rerank_score=r.score,
                    fused_rank=r.candidate.fused_rank,
                    in_dense=r.candidate.point_id in dense_ids,
                    in_sparse=r.candidate.point_id in sparse_ids,
                    page=r.candidate.page_from,
                    page_to=r.candidate.page_to,
                    section=r.candidate.section,
                )
                for r in ranked
            ],
            after_threshold=len(relevant),
            used_in_answer=[c.chunk_index for c in response.citations],
            timings_ms={
                name: timer.ms.get(name, 0) for name in ("embed", "search", "rerank", "llm")
            },
        )


def build_llm(settings: Settings) -> Runnable:
    chat = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=0,
        max_retries=0,  # ретраи — наши, единые для всех внешних вызовов
        timeout=settings.http_timeout_s,
    )
    return chat.with_structured_output(LLMAnswer)


def build_store(settings: Settings) -> QdrantStore:
    client = QdrantClient(url=settings.qdrant_url, timeout=int(settings.http_timeout_s))
    return QdrantStore(
        client,
        settings.qdrant_collection,
        dense_dim=settings.embedding_dim,
        retry_attempts=settings.retry_attempts,
    )


def build_pipeline(settings: Settings, store: QdrantStore | None = None) -> QAPipeline:
    return QAPipeline(
        store=store or build_store(settings),
        embedder=build_embedder(settings),
        reranker=build_reranker(settings),
        llm=build_llm(settings),
        settings=settings,
    )
