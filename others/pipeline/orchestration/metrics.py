"""Per-stage and per-run metrics collection for the orchestrator (feature 015, FR-029, SC-006).

Timing and counts use only the stdlib. CPU/RSS sampling prefers ``psutil`` when available and the metrics backend
is not forced to ``stdlib``, degrading to ``resource.getrusage`` / ``os.times`` (and finally ``None``) otherwise —
so the default install collects metrics without a new required dependency.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator

from pipeline import config as _config
from pipeline.models import RunMetrics, StageMetrics

__all__ = ["MetricsCollector", "sample_rss_mb", "sample_cpu_seconds"]


def _psutil_enabled() -> bool:
    return _config.KNOVEL_ORCH_METRICS_BACKEND.lower() != "stdlib"


def sample_rss_mb() -> float | None:
    """Resident-set size in MB; psutil if available, else stdlib resource, else None."""
    if _psutil_enabled():
        try:  # pragma: no cover - exercised only when psutil is installed
            import psutil

            return float(psutil.Process().memory_info().rss) / (1024 * 1024)
        except Exception:
            pass
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is KB on Linux, bytes on macOS — normalize conservatively to MB.
        divisor = 1024 * 1024 if rss > 10_000_000 else 1024
        return float(rss) / divisor
    except Exception:  # pragma: no cover - resource missing (e.g. Windows)
        return None


def sample_cpu_seconds() -> float | None:
    """Process CPU seconds (user+system); psutil if available, else os.times, else None."""
    if _psutil_enabled():
        try:  # pragma: no cover - exercised only when psutil is installed
            import psutil

            times = psutil.Process().cpu_times()
            return float(times.user + times.system)
        except Exception:
            pass
    try:
        times = os.times()
        return float(times.user + times.system)
    except Exception:  # pragma: no cover
        return None


class MetricsCollector:
    """Accumulates per-stage and per-run metrics for a single orchestration run."""

    def __init__(self, pipeline_run_id: str) -> None:
        self._run = RunMetrics(pipeline_run_id=pipeline_run_id)
        self._run_start = time.perf_counter()
        self._peak_rss_mb: float | None = None
        self._lock = threading.Lock()  # document-level parallelism may mutate from worker threads

    @contextmanager
    def stage_timer(self, stage: str, *, provider: str | None = None) -> Iterator[StageMetrics]:
        """Time a stage and append a ``StageMetrics`` record."""
        cpu_before = sample_cpu_seconds()
        start = time.perf_counter()
        metric = StageMetrics(stage=stage, provider=provider)
        try:
            yield metric
        finally:
            metric.duration_s = time.perf_counter() - start
            cpu_after = sample_cpu_seconds()
            if cpu_before is not None and cpu_after is not None:
                metric.cpu_seconds = max(0.0, cpu_after - cpu_before)
            metric.peak_rss_mb = sample_rss_mb()
            with self._lock:
                self._observe_rss(metric.peak_rss_mb)
                self._run.stages.append(metric)

    def _observe_rss(self, rss: float | None) -> None:
        if rss is None:
            return
        if self._peak_rss_mb is None or rss > self._peak_rss_mb:
            self._peak_rss_mb = rss

    def record_document(self, status: str) -> None:
        with self._lock:
            self._run.documents_total += 1
            if status in ("completed", "partial"):
                self._run.documents_succeeded += 1
            elif status == "failed":
                self._run.documents_failed += 1
            elif status == "skipped":
                self._run.documents_skipped += 1

    def add_retries(self, count: int) -> None:
        with self._lock:
            self._run.retry_total += count

    def add_resume_reused(self, count: int) -> None:
        with self._lock:
            self._run.resume_reused += count

    def finalize(self) -> RunMetrics:
        self._run.total_duration_s = time.perf_counter() - self._run_start
        self._run.peak_rss_mb = (
            self._peak_rss_mb if self._peak_rss_mb is not None else sample_rss_mb()
        )
        return self._run
