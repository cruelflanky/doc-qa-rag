from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from qdrant_client import models

from app.config import Settings
from app.embeddings import LangChainEmbedder, build_embedder


class RecordingDense(Embeddings):
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t))] for t in texts]

    def embed_query(self, text):
        return [float(len(text))]


class SimpleSparse(SparseEmbeddings):
    def embed_documents(self, texts):
        return [SparseVector(indices=[len(t)], values=[1.0]) for t in texts]

    def embed_query(self, text):
        return SparseVector(indices=[len(text)], values=[1.0])


def make_embedder(batch_size=64):
    dense = RecordingDense()
    return LangChainEmbedder(dense, SimpleSparse(), batch_size=batch_size, retry_attempts=1), dense


def test_embed_dense_batches_and_keeps_order():
    embedder, dense = make_embedder(batch_size=64)
    texts = ["x" * i for i in range(1, 131)]

    vectors = embedder.embed_dense(texts)

    assert [len(call) for call in dense.calls] == [64, 64, 2]
    assert vectors == [[float(i)] for i in range(1, 131)]


def test_sparse_converted_to_qdrant_type():
    embedder, _ = make_embedder()

    vectors = embedder.embed_sparse(["abc", "de"])

    assert all(isinstance(v, models.SparseVector) for v in vectors)
    assert vectors[0].indices == [3]


def test_embed_query_returns_both_vectors():
    embedder, _ = make_embedder()

    dense, sparse = embedder.embed_query("вопрос")

    assert dense == [6.0]
    assert isinstance(sparse, models.SparseVector)


def test_build_embedder_disables_ctx_length_check(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    settings = Settings(_env_file=None)

    embedder = build_embedder(settings)

    assert embedder.dense.check_embedding_ctx_length is False
    assert embedder.dense.model == "openai/text-embedding-3-small"
    assert embedder.dense.max_retries == 0
