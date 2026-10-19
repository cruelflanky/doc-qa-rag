import fakeredis

from app.task_registry import TASK_TTL_SECONDS, lookup_doc_id, register_task


def test_registry_roundtrip_and_unknown_returns_none():
    redis = fakeredis.FakeRedis(decode_responses=True)

    register_task(redis, "task-1", "doc-1")

    assert lookup_doc_id(redis, "task-1") == "doc-1"
    assert lookup_doc_id(redis, "missing") is None
    assert 0 < redis.ttl("task:task-1") <= TASK_TTL_SECONDS
