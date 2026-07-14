"""CLI progress reporting for long-running orchestration runs (feature 015, FR-030).

Uses ``rich`` when attached to a TTY; degrades to structlog/plain logging in non-TTY/CI environments so batch
runs stay observable without requiring an interactive terminal.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator, Protocol

from pipeline.run_logging import _pipeline_logger

__all__ = ["ProgressReporter", "progress_for"]


class ProgressReporter(Protocol):
    """Minimal progress surface the runner depends on."""

    def advance(self, document: str) -> None: ...


class _LoggingProgress:
    """Plain/structlog progress used in non-TTY/CI runs."""

    def __init__(self, total: int) -> None:
        self._total = total
        self._done = 0

    def advance(self, document: str) -> None:
        self._done += 1
        _pipeline_logger().info(
            "orchestration_progress", document=document, done=self._done, total=self._total
        )


class _RichProgress:
    """Rich progress bar used when a TTY is present."""

    def __init__(self, progress: object, task_id: object) -> None:
        self._progress = progress
        self._task_id = task_id

    def advance(self, document: str) -> None:
        self._progress.advance(self._task_id)  # type: ignore[attr-defined]


def _is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:  # pragma: no cover - unusual stdout
        return False


@contextmanager
def progress_for(
    total: int, *, description: str = "Processing documents"
) -> Iterator[ProgressReporter]:
    """Yield a progress reporter for ``total`` documents, Rich if interactive else logging."""
    progress = None
    if _is_tty() and total > 0:
        try:  # pragma: no cover - exercised only in an interactive terminal
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeRemainingColumn,
            )

            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            )
        except Exception:  # Rich unavailable — fall back to logging below.
            progress = None

    # NB: the ``yield`` must live outside the ``except`` above. If it were inside,
    # an exception thrown back into this generator at the yield (a failing/aborted
    # run body) would be swallowed and the generator would yield twice, raising
    # "generator didn't stop after throw()".
    if progress is not None:
        with progress:  # pragma: no cover - exercised only in an interactive terminal
            task_id = progress.add_task(description, total=total)
            yield _RichProgress(progress, task_id)
        return

    yield _LoggingProgress(total)
