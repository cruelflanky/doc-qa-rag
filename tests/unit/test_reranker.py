import json

import httpx
import pytest
import respx
from langchain_core.documents import Document
from pydantic import SecretStr
from tenacity import wait_none

from app import retry as retry_module
from app.rag.reranker import OpenRouterReranker

BASE = "https://openrouter.test/api/v1"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(retry_module, "WAIT", wait_none())


@pytest.fixture
def reranker():
    return OpenRouterReranker(
        api_key=SecretStr("secret"),
        base_url=BASE,
        model="cohere/rerank-v3.5",
        top_n=2,
        timeout_s=5,
        retry_attempts=3,
    )


def ok(results):
    return httpx.Response(200, json={"id": "x", "model": "m", "results": results})


@respx.mock
def test_rerank_sends_expected_payload(reranker):
    route = respx.post(f"{BASE}/rerank").mock(return_value=ok([]))

    reranker.rerank("вопрос", ["a", "b", "c"])

    request = route.calls.last.request
    assert json.loads(request.content) == {
        "model": "cohere/rerank-v3.5",
        "query": "вопрос",
        "documents": ["a", "b", "c"],
        "top_n": 2,
    }
    assert request.headers["Authorization"] == "Bearer secret"


@respx.mock
def test_rerank_sorts_by_relevance_score(reranker):
    respx.post(f"{BASE}/rerank").mock(
        return_value=ok(
            [{"index": 0, "relevance_score": 0.2}, {"index": 2, "relevance_score": 0.9}]
        )
    )

    results = reranker.rerank("q", ["a", "b", "c"])

    assert [(r.index, r.score) for r in results] == [(2, 0.9), (0, 0.2)]


@respx.mock
def test_rerank_retries_on_503_then_succeeds(reranker):
    route = respx.post(f"{BASE}/rerank").mock(
        side_effect=[httpx.Response(503), ok([{"index": 0, "relevance_score": 0.5}])]
    )

    results = reranker.rerank("q", ["a"])

    assert route.call_count == 2
    assert results[0].index == 0


@respx.mock
def test_rerank_raises_on_400_without_retry(reranker):
    route = respx.post(f"{BASE}/rerank").mock(return_value=httpx.Response(400))

    with pytest.raises(httpx.HTTPStatusError):
        reranker.rerank("q", ["a"])
    assert route.call_count == 1


def test_rerank_empty_input_makes_no_request(reranker):
    assert reranker.rerank("q", []) == []


@respx.mock
def test_compress_documents_sets_rerank_score(reranker):
    respx.post(f"{BASE}/rerank").mock(
        return_value=ok(
            [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.1}]
        )
    )
    docs = [
        Document(page_content="a", metadata={"k": 0}),
        Document(page_content="b", metadata={"k": 1}),
    ]

    result = reranker.compress_documents(docs, "q")

    assert [d.page_content for d in result] == ["b", "a"]
    assert result[0].metadata == {"k": 1, "rerank_score": 0.8}
    assert "rerank_score" not in docs[1].metadata  # исходные документы не мутируются
