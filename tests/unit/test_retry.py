import httpx
import openai
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app import retry as retry_module
from app.retry import is_transient, with_retry


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(retry_module, "WAIT", retry_module.wait_none())


def status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test")
    return httpx.HTTPStatusError(
        "err", request=request, response=httpx.Response(code, request=request)
    )


def counting(errors):
    calls = {"n": 0}

    def func():
        calls["n"] += 1
        if calls["n"] <= len(errors):
            raise errors[calls["n"] - 1]
        return "ok"

    return func, calls


def test_retries_transient_then_succeeds():
    func, calls = counting([httpx.ConnectError("down"), httpx.ConnectError("down")])

    assert with_retry(attempts=4)(func)() == "ok"
    assert calls["n"] == 3


def test_gives_up_after_attempts_and_reraises():
    func, calls = counting([status_error(503)] * 10)

    with pytest.raises(httpx.HTTPStatusError):
        with_retry(attempts=3)(func)()
    assert calls["n"] == 3


def test_does_not_retry_client_errors():
    func, calls = counting([status_error(400)])

    with pytest.raises(httpx.HTTPStatusError):
        with_retry(attempts=5)(func)()
    assert calls["n"] == 1


def _openai_request() -> httpx.Request:
    return httpx.Request("POST", "https://openrouter.test")


def _unexpected(code: int) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=code, reason_phrase="x", content=b"", headers=httpx.Headers()
    )


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ConnectError("x"), True),
        (httpx.ReadTimeout("x"), True),
        (status_error(429), True),
        (status_error(500), True),
        (status_error(503), True),
        (status_error(400), False),
        (status_error(404), False),
        (openai.APIConnectionError(request=_openai_request()), True),
        (openai.APITimeoutError(request=_openai_request()), True),
        (ResponseHandlingException(Exception("x")), True),
        (_unexpected(502), True),
        (_unexpected(429), True),
        (_unexpected(400), False),
        (ConnectionError("x"), True),
        (TimeoutError("x"), True),
        (ValueError("x"), False),
    ],
)
def test_is_transient_table(exc, expected):
    assert is_transient(exc) is expected


def test_openai_rate_limit_and_server_errors_are_transient():
    response = httpx.Response(429, request=_openai_request())
    assert is_transient(openai.RateLimitError("x", response=response, body=None))
    response = httpx.Response(500, request=_openai_request())
    assert is_transient(openai.InternalServerError("x", response=response, body=None))
    response = httpx.Response(400, request=_openai_request())
    assert not is_transient(openai.BadRequestError("x", response=response, body=None))
