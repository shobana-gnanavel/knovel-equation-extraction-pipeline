"""Document/batch-level parallel execution for the orchestrator (feature 015, FR-026..028).

Runs independent documents concurrently with a configurable worker limit, preserving serial-equivalence (each
document yields an independent ``PipelineContext`` and results are returned in input order). Under memory pressure
the worker count is reduced (down to serial) rather than risking an out-of-memory crash. Page-level parallelism
stays inside the stages (the existing per-page process pool); this module owns the document axis. Imports only
``pipeline`` + stdlib.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from pipeline.orchestration.metrics import sample_rss_mb
from pipeline.run_logging import _pipeline_logger

__all__ = ["resolve_workers", "map_documents"]

T = TypeVar("T")
R = TypeVar("R")


def resolve_workers(
    requested: int,
    *,
    memory_limit_mb: int = 0,
    item_count: int = 0,
    sampler: Callable[[], float | None] = sample_rss_mb,
) -> int:
    """Resolve the effective worker count, shrinking under memory pressure (FR-028).

    Returns 1 (serial) when only one item is present, when fewer than 2 workers are requested, or when the
    current RSS already exceeds ``memory_limit_mb`` (a logged graceful degradation rather than an OOM risk).
    """
    workers = max(1, requested)
    if item_count <= 1 or workers <= 1:
        return 1
    if memory_limit_mb > 0:
        rss = sampler()
        if rss is not None and rss > memory_limit_mb:
            _pipeline_logger().info(
                "parallel_backoff",
                reason="memory_pressure",
                rss_mb=round(rss, 1),
                limit_mb=memory_limit_mb,
                workers_from=workers,
                workers_to=1,
            )
            return 1
    return workers


def map_documents(
    documents: Sequence[T],
    worker_fn: Callable[[T], R],
    *,
    workers: int,
    memory_limit_mb: int = 0,
) -> list[R]:
    """Apply ``worker_fn`` to each document, concurrently up to ``workers``, in input order.

    Serial-equivalence: results match a serial ``[worker_fn(d) for d in documents]`` because documents are
    independent. Falls back to serial execution for a single item or under memory pressure.
    """
    effective = resolve_workers(workers, memory_limit_mb=memory_limit_mb, item_count=len(documents))
    if effective <= 1:
        return [worker_fn(doc) for doc in documents]
    with ThreadPoolExecutor(max_workers=effective) as executor:
        return list(executor.map(worker_fn, documents))
