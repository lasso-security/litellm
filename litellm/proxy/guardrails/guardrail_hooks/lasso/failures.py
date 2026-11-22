import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Literal, TypeVar

import httpx

import litellm

UnreachableFallback = Literal["fail_closed", "fail_open"]

_T = TypeVar("_T")

UNAVAILABLE_STATUSES: Final = frozenset({408, 429, 500, 502, 503, 504})
MAX_RETRY_WAIT_SECONDS: Final = 5.0
_BASE_BACKOFF_SECONDS: Final = 0.25
_MAX_BACKOFF_SECONDS: Final = 2.0


@dataclass(frozen=True, slots=True)
class LassoFailure:
    message: str
    unavailable: bool
    status_code: int | None = None
    retry_after_seconds: float | None = None


def classify_failure(error: Exception) -> LassoFailure:
    """Lasso is unavailable when it timed out, could not be reached, answered 408, 429 or 5xx, or its edge throttled the call."""
    match error:
        case httpx.TimeoutException() | litellm.Timeout():
            return LassoFailure("Lasso API timeout", unavailable=True)
        case httpx.HTTPStatusError(response=response):
            return LassoFailure(
                _status_message(response.status_code),
                unavailable=response.status_code in UNAVAILABLE_STATUSES or _is_edge_throttle(response),
                status_code=response.status_code,
                retry_after_seconds=_retry_after_seconds(next(iter(response.headers.get_list("retry-after")), None)),
            )
        case httpx.TransportError():
            return LassoFailure(f"Failed to verify request safety with Lasso API: {error}", unavailable=True)
        case _:
            return LassoFailure(f"Failed to verify request safety with Lasso API: {error}", unavailable=False)


def retry_wait_seconds(failure: LassoFailure, attempt: int) -> float | None:
    """How long to wait before retrying, or None when the failure is not worth retrying.

    A Retry-After longer than MAX_RETRY_WAIT_SECONDS is not waited out, so a rate limit can't hold the caller's
    request for longer than that.
    """
    if not failure.unavailable:
        return None
    if failure.retry_after_seconds is not None:
        return failure.retry_after_seconds if failure.retry_after_seconds <= MAX_RETRY_WAIT_SECONDS else None
    return random.uniform(0, min(_BASE_BACKOFF_SECONDS * (1 << attempt), _MAX_BACKOFF_SECONDS))


async def call_with_retries(
    call: Callable[[], Awaitable[_T]],
    retries_left: int,
    sleep: Callable[[float], Awaitable[None]],
    attempt: int = 0,
) -> _T:
    try:
        return await call()
    except Exception as error:
        wait: Final = retry_wait_seconds(classify_failure(error), attempt) if retries_left > 0 else None
        if wait is None:
            raise
        await sleep(wait)
    return await call_with_retries(call, retries_left - 1, sleep, attempt + 1)


def _is_edge_throttle(response: httpx.Response) -> bool:
    """Lasso's edge firewall throttles bursts with a 403 page; Lasso itself answers a refused key or scope as JSON."""
    return response.status_code == 403 and not response.headers.get("content-type", "").startswith("application/json")


def _status_message(status_code: int) -> str:
    if status_code == 429:
        return "Lasso API rate limit exceeded"
    return f"API error: {status_code}"


def _retry_after_seconds(value: str | None) -> float | None:
    try:
        seconds: Final = float(value) if value is not None else None
    except ValueError:
        return None
    return seconds if seconds is not None and seconds >= 0 else None
