import pytest
from langchain_core.documents import Document

from app import retry as retry_module
from app.config import Settings
from app.errors import DocumentNotFoundError
from app.rag.pipeline import QAPipeline
from app.schemas import LLMAnswer
from app.store.models import Candidate, SearchTrace
from tests.fakes import FakeEmbedder

CANDIDATES = [
    Candidate(f"p{i}", i, f"Фрагмент {i}", i + 1, i + 1, f"{i}. РАЗДЕЛ", 1.0 / (i + 1), i + 1)
    for i in range(6)
]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(retry_module, "WAIT", retry_module.wait_none())


class FakeStore:
    def __init__(self, ready=True, candidates=CANDIDATES):
        self.ready = ready
        self.candidates = candidates
        self.calls: list[str] = []

    def document_ready(self, doc_id):
        return self.ready

    def hybrid_search(self, doc_id, dense, sparse, **params):
        self.calls.append("hybrid_search")
        self.params = params
        return self.candidates

    def hybrid_search_debug(self, doc_id, dense, sparse, **params):
        self.calls.append("hybrid_search_debug")
        return SearchTrace(
            candidates=self.candidates,
            dense_ids=["p0", "p1", "p2", "p3"],
            sparse_ids=["p0", "p5"],
        )


class FakeReranker:
    """Даёт фрагменту с chunk_index i score из словаря, top_n лучших."""

    def __init__(self, scores, top_n=3):
        self.scores = scores
        self.top_n = top_n

    def compress_documents(self, documents, query, callbacks=None):
        scored = [
            Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": s})
            for d in documents
            if (s := self.scores.get(int(d.page_content.split()[-1]))) is not None
        ]
        scored.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
        return scored[: self.top_n]


class FakeLLM:
    def __init__(self, answers):
        self.answers = list(answers)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return Settings(
        _env_file=None, rerank_candidates=6, rerank_top_n=3, rerank_min_score=0.3, retry_attempts=3
    )


def make_pipeline(settings, store=None, scores=None, answers=None):
    store = store or FakeStore()
    llm = FakeLLM(answers or [LLMAnswer(found=True, answer="Ответ", used_fragments=[1, 2])])
    reranker = FakeReranker(scores or {4: 0.9, 1: 0.7, 0: 0.5, 2: 0.1}, top_n=3)
    return QAPipeline(store, FakeEmbedder(), reranker, llm, settings), store, llm


def test_ask_unknown_or_not_ready_document_raises_not_found(settings):
    pipeline, _, _ = make_pipeline(settings, store=FakeStore(ready=False))

    with pytest.raises(DocumentNotFoundError):
        pipeline.ask("doc", "вопрос")


def test_ask_below_threshold_skips_llm(settings):
    pipeline, _, llm = make_pipeline(settings, scores={0: 0.2, 1: 0.1})

    response = pipeline.ask("doc", "вопрос")

    assert response.found is False
    assert llm.messages == []


def test_ask_no_candidates_returns_not_found(settings):
    pipeline, _, llm = make_pipeline(settings, store=FakeStore(candidates=[]))

    response = pipeline.ask("doc", "вопрос")

    assert response.found is False
    assert llm.messages == []


def test_ask_passes_only_top_n_reranked_fragments_to_llm(settings):
    pipeline, store, llm = make_pipeline(settings)

    response = pipeline.ask("doc", "вопрос")

    prompt = llm.messages[0][-1].content
    assert "[1] (стр. 5–5, раздел «4. РАЗДЕЛ»)" in prompt
    assert "[2] (стр. 2–2" in prompt
    assert "[3] (стр. 1–1" in prompt
    assert "[4]" not in prompt
    assert [c.chunk_index for c in response.citations] == [4, 1]
    assert store.params["limit"] == 6


def test_ask_without_debug_has_no_debug_and_uses_single_search(settings):
    pipeline, store, _ = make_pipeline(settings)

    response = pipeline.ask("doc", "вопрос")

    assert response.debug is None
    assert store.calls == ["hybrid_search"]


def test_ask_debug_reports_branch_counts_and_used_fragments(settings):
    pipeline, store, _ = make_pipeline(settings)

    response = pipeline.ask("doc", "вопрос", debug=True)

    debug = response.debug
    assert store.calls == ["hybrid_search_debug"]
    assert (debug.dense_hits, debug.sparse_hits, debug.fused_candidates) == (4, 2, 6)
    assert debug.fusion == "rrf"
    assert [item.chunk_index for item in debug.reranked] == [4, 1, 0]
    assert debug.reranked[0].in_dense is False and debug.reranked[0].in_sparse is False
    assert debug.reranked[2].in_dense is True and debug.reranked[2].in_sparse is True
    assert debug.after_threshold == 3
    assert debug.used_in_answer == [4, 1]
    assert set(debug.timings_ms) == {"embed", "search", "rerank", "llm"}


def test_ask_retries_llm_transient_error(settings):
    answers = [ConnectionError("timeout"), LLMAnswer(found=True, answer="Ок", used_fragments=[1])]
    pipeline, _, llm = make_pipeline(settings, answers=answers)

    response = pipeline.ask("doc", "вопрос")

    assert response.found is True
    assert len(llm.messages) == 2
