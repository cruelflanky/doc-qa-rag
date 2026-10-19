"""Повторные попытки при временной недоступности внешних сервисов (OpenRouter, Qdrant, Redis).

Повторяем только то, что может пройти само: сетевые сбои, таймауты, 429 и 5xx.
Ошибки клиента (4xx) повторять бессмысленно — они пробрасываются сразу.
"""

import functools
import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import httpx
import kombu.exceptions
import openai
import redis.exceptions
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_none,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

TOO_MANY_REQUESTS = 429
SERVER_ERROR = 500
WAIT = wait_random_exponential(multiplier=0.5, max=10)

__all__ = ["is_transient", "with_retry", "wait_none"]


def _is_retryable_status(status: int) -> bool:
    return status == TOO_MANY_REQUESTS or status >= SERVER_ERROR


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return _is_retryable_status(exc.response.status_code)
    if isinstance(exc, openai.APIStatusError):
        return _is_retryable_status(exc.status_code)
    if isinstance(exc, UnexpectedResponse):
        return _is_retryable_status(exc.status_code)
    return isinstance(
        exc,
        httpx.TransportError
        | openai.APIConnectionError
        | ResponseHandlingException
        | ConnectionError
        | TimeoutError
        | redis.exceptions.ConnectionError
        | redis.exceptions.TimeoutError
        | kombu.exceptions.OperationalError,
    )


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception()
    logger.warning(
        "Временная ошибка %s (попытка %d), повторяем: %s",
        type(exc).__name__,
        retry_state.attempt_number,
        exc,
    )


def with_retry(attempts: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Декоратор: повторяет вызов при временных ошибках, затем пробрасывает исходную."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            retrying = Retrying(
                retry=retry_if_exception(is_transient),
                wait=WAIT,
                stop=stop_after_attempt(attempts),
                before_sleep=_log_retry,
                reraise=True,
            )
            return retrying(func, *args, **kwargs)

        return wrapper

    return decorator
