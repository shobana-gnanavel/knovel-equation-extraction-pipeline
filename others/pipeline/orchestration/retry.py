"""Configurable retry for stage execution (feature 015, FR-024/25).

Wraps a stage call with tenacity-style exponential backoff up to a configurable attempt limit, classifying
failures as recoverable (retried) vs non-recoverable/critical (raised immediately). Determinism: no jitter by
default, so test runs are reproducible. Imports only ``pipeline`` + stdlib + tenacity (already required).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from pipeline.models import RetryPolicy
from pipeline.run_logging import _pipeline_logger

__all__ = ["RetryOutcome", "execute_with_retry", "is_recoverable"]

T = TypeVar("T")


class RetryOutcome:
    """Result of a retried call: the number of retries performed and the final exception (if any)."""

    def __init__(self) -> None:
        self.retries = 0
        self.attempts = 0


def is_recoverable(exc: Exception, policy: RetryPolicy) -> bool:
    """Classify an exception as recoverable per the policy.

    Critical errors are never recoverable. If ``recoverable_errors`` is set, only those types are recoverable;
    otherwise any non-critical exception is recoverable.
    """
    name = type(exc).__name__
    if name in policy.critical_errors:
        return False
    if policy.recoverable_errors:
        return name in policy.recoverable_errors
    return True


def _delay_for(attempt: int, policy: RetryPolicy) -> float:
    """Exponential backoff delay (seconds) for a given 1-based attempt, capped at ``max_delay_s``."""
    delay = policy.base_delay_s * (policy.backoff_factor ** (attempt - 1))
    return min(delay, policy.max_delay_s)


def execute_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    stage: str = "",
    sleep: Callable[[float], None] = time.sleep,
    outcome: RetryOutcome | None = None,
) -> T:
    """Execute ``fn`` with retry per ``policy``; re-raises the last error after the limit or on a critical error.

    ``outcome`` (if provided) records the retry count. ``sleep`` is injectable so tests run without real delays.
    """
    outcome = outcome or RetryOutcome()
    last_exc: Exception | None = None
    for attempt in range(1, max(1, policy.max_attempts) + 1):
        outcome.attempts = attempt
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            recoverable = is_recoverable(exc, policy)
            if not recoverable or attempt >= policy.max_attempts:
                raise
            outcome.retries += 1
            delay = _delay_for(attempt, policy)
            _pipeline_logger().info(
                "stage_retry",
                stage=stage,
                attempt=attempt,
                next_delay_s=delay,
                error=str(exc),
            )
            if delay > 0:
                sleep(delay)
    # Unreachable: the loop either returns or raises; included for type-completeness.
    assert last_exc is not None  # noqa: S101
    raise last_exc
