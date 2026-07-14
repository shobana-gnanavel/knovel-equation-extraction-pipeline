"""Structured logging setup for pipeline execution."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from equation_extraction_pipeline.domain.models import StageFailure

try:  # pragma: no cover - optional dependency handling
    import structlog
except Exception:  # pragma: no cover - optional dependency handling

    class _BoundLogger:
        def __init__(self, **context):
            self._context = context

        def bind(self, **context):
            return _BoundLogger(**{**self._context, **context})

        def _emit(self, level: str, event: str, **kwargs):
            payload = {
                "event": event,
                "level": level,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **self._context,
                **kwargs,
            }
            print(json.dumps(payload, ensure_ascii=False), file=sys.stdout)

        def info(self, event: str, **kwargs):
            self._emit("info", event, **kwargs)

        def error(self, event: str, **kwargs):
            self._emit("error", event, **kwargs)

    class _StructlogFallback:
        @staticmethod
        def get_logger(name: str | None = None):
            return _BoundLogger(logger=name) if name else _BoundLogger()

        @staticmethod
        def configure(*args, **kwargs):
            return None

        class stdlib:
            @staticmethod
            def LoggerFactory():
                return logging.getLogger

    structlog = _StructlogFallback()  # type: ignore[assignment]  # optional-dep fallback

try:  # pragma: no cover - optional dependency handling
    from structlog.stdlib import LoggerFactory, add_log_level
except Exception:  # pragma: no cover - optional dependency handling

    def add_log_level(logger, method_name, event_dict):  # type: ignore[misc]  # fallback stub
        event_dict.setdefault("level", method_name)
        return event_dict

    def LoggerFactory():  # type: ignore[no-redef]  # optional-dep fallback stub
        return logging.getLogger


try:  # pragma: no cover - optional dependency handling
    from structlog.processors import JSONRenderer
except Exception:  # pragma: no cover - optional dependency handling

    def JSONRenderer():  # type: ignore[no-redef]  # optional-dep fallback stub
        def renderer(_, __, event_dict):
            return json.dumps(event_dict, ensure_ascii=False)

        return renderer


try:  # pragma: no cover - optional dependency handling
    from structlog.processors import TimeStamper
except Exception:  # pragma: no cover - optional dependency handling

    def TimeStamper(fmt="iso", utc=True):  # type: ignore[no-redef]  # optional-dep fallback stub
        def stamper(_, __, event_dict):
            event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
            return event_dict

        return stamper


__all__ = [
    "configure_logging",
    "log_stage_failure",
    "get_failures",
    "mark_gold_candidate",
]


def configure_logging(pipeline_run_id: str, level: int = logging.INFO) -> None:
    structlog.configure(
        processors=[
            add_log_level,
            TimeStamper(fmt="iso", utc=True),
            JSONRenderer(),
        ],
        logger_factory=LoggerFactory(),
        wrapper_class=getattr(structlog, "BoundLogger", object),  # type: ignore[arg-type]
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=level, stream=sys.stdout, force=True)
    root_logger = structlog.get_logger().bind(pipeline_run_id=pipeline_run_id)
    setattr(structlog, "_pipeline_logger", root_logger)


def _pipeline_logger():
    logger = getattr(structlog, "_pipeline_logger", None)
    if logger is not None:
        return logger
    return structlog.get_logger()


def log_stage_failure(failure: StageFailure, output_dir: Path) -> None:
    run_dir = output_dir / failure.pipeline_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    failures_path = run_dir / "failures.jsonl"
    payload = {
        "pipeline_run_id": failure.pipeline_run_id,
        "book_id": failure.book_id,
        "table_id": failure.table_id,
        "page_no": failure.page_no,
        "stage": failure.stage,
        "error_type": failure.error_type,
        "error_msg": failure.error_msg,
        "retry_count": failure.retry_count,
        "is_gold_candidate": failure.is_gold_candidate,
        "timestamp": failure.timestamp.isoformat(),
    }
    with failures_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _pipeline_logger().error("stage_failure", **payload)


def get_failures(pipeline_run_id: str, output_dir: Path) -> list[StageFailure]:
    failures_path = output_dir / pipeline_run_id / "failures.jsonl"
    if not failures_path.exists():
        return []

    failures: list[StageFailure] = []
    with failures_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            failure = StageFailure(
                pipeline_run_id=payload["pipeline_run_id"],
                book_id=payload["book_id"],
                table_id=payload.get("table_id"),
                page_no=payload.get("page_no"),
                stage=payload["stage"],
                error_type=payload["error_type"],
                error_msg=payload["error_msg"],
                retry_count=int(payload.get("retry_count", 0)),
                is_gold_candidate=bool(payload.get("is_gold_candidate", False)),
                timestamp=datetime.fromisoformat(payload["timestamp"]),
            )
            if failure.retry_count < 3:
                failures.append(failure)
    return failures


def mark_gold_candidate(failure: StageFailure, output_dir: Path) -> None:
    failure.is_gold_candidate = True
    run_dir = output_dir / failure.pipeline_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = run_dir / "gold_candidates.txt"
    with candidates_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{failure.table_id or ''}\n")
