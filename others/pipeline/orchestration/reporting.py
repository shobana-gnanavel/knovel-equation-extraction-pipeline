"""Report-family builders and JSON/CSV exporters for the orchestrator (feature 015, FR-031/32).

Reports are pure projections of the per-document ``PipelineContext`` accumulators + the run ``RunMetrics``, built
after the run so a report failure never affects extraction output. JSON uses ``orjson`` with sorted keys
(deterministic); CSV uses the stdlib ``csv`` module (house style). Row ordering is stable (SC-007).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from pipeline.models import (
    ConfigurationSnapshot,
    ErrorReport,
    ExecutionReport,
    PerformanceReport,
    PipelineConfig,
    PipelineContext,
    PipelineSummary,
    ProcessingStatistics,
    ProviderSummary,
    RunContext,
    RunMetrics,
)

__all__ = ["build_reports", "write_reports", "write_config_snapshot"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_reports(
    run_context: RunContext,
    contexts: list[PipelineContext],
    metrics: RunMetrics,
    *,
    finished_at: str | None = None,
) -> ExecutionReport:
    """Project the per-document contexts + run metrics into the full report family."""
    summary = _build_pipeline_summary(run_context, contexts)
    provider_summary = _build_provider_summary(contexts)
    performance = _build_performance_report(
        metrics, len(contexts), benchmark_mode=run_context.config.benchmark_mode
    )
    error_report = _build_error_report(contexts)
    statistics = _build_processing_statistics(contexts)

    failed = summary.documents_failed
    succeeded = summary.documents_succeeded
    if failed and succeeded:
        exit_status = "partial"
    elif failed and not succeeded:
        exit_status = "failed"
    else:
        exit_status = "success"

    return ExecutionReport(
        pipeline_run_id=run_context.pipeline_run_id,
        config_hash=run_context.config.config_hash,
        plan_name=run_context.plan.name,
        plan_order=run_context.plan.stage_names(),
        started_at=run_context.started_at.isoformat() if run_context.started_at else "",
        finished_at=finished_at or _now_iso(),
        exit_status=exit_status,
        documents=[ctx.to_dict() for ctx in contexts],
        pipeline_summary=summary,
        provider_summary=provider_summary,
        performance_report=performance,
        error_report=error_report,
        processing_statistics=statistics,
        metrics=metrics,
    )


def _build_pipeline_summary(
    run_context: RunContext, contexts: list[PipelineContext]
) -> PipelineSummary:
    summary = PipelineSummary(documents_total=len(contexts))
    for ctx in contexts:
        if ctx.status in ("completed", "partial"):
            summary.documents_succeeded += 1
        elif ctx.status == "failed":
            summary.documents_failed += 1
        elif ctx.status == "skipped":
            summary.documents_skipped += 1
        for result in ctx.stage_results.values():
            summary.stages_total += 1
            if result.status in ("completed", "reused"):
                summary.stages_succeeded += 1
            elif result.status == "failed":
                summary.stages_failed += 1
            elif result.status == "skipped":
                summary.stages_skipped += 1
    if summary.documents_total:
        summary.document_success_rate = summary.documents_succeeded / summary.documents_total
    if summary.stages_total:
        summary.stage_success_rate = summary.stages_succeeded / summary.stages_total
    return summary


def _build_provider_summary(contexts: list[PipelineContext]) -> ProviderSummary:
    usage: dict[str, int] = {}
    for ctx in contexts:
        for result in ctx.stage_results.values():
            if result.provider:
                usage[result.provider] = usage.get(result.provider, 0) + 1
    return ProviderSummary(usage_counts=dict(sorted(usage.items())))


def _build_performance_report(
    metrics: RunMetrics, doc_count: int, *, benchmark_mode: bool = False
) -> PerformanceReport:
    throughput = 0.0
    if metrics.total_duration_s > 0:
        throughput = doc_count / (metrics.total_duration_s / 60.0)
    provider_durations: dict[str, float] = {}
    if benchmark_mode:
        for stage in metrics.stages:
            if stage.provider:
                provider_durations[stage.provider] = (
                    provider_durations.get(stage.provider, 0.0) + stage.duration_s
                )
    return PerformanceReport(
        total_duration_s=metrics.total_duration_s,
        peak_rss_mb=metrics.peak_rss_mb,
        throughput_docs_per_min=throughput,
        stages=list(metrics.stages),
        provider_durations=dict(sorted(provider_durations.items())),
        benchmark_mode=benchmark_mode,
    )


def _build_error_report(contexts: list[PipelineContext]) -> ErrorReport:
    failures: list[dict[str, Any]] = []
    for ctx in contexts:
        for result in sorted(ctx.stage_results.values(), key=lambda r: r.stage):
            if result.status == "failed" and result.failure is not None:
                failures.append(
                    {
                        "book_id": result.failure.book_id,
                        "stage": result.stage,
                        "page_no": result.failure.page_no,
                        "table_id": result.failure.table_id,
                        "error_type": result.failure.error_type,
                        "error_msg": result.failure.error_msg,
                        "retry_count": result.retry_count,
                        "action_taken": result.status,
                    }
                )
    failures.sort(key=lambda f: (str(f["book_id"]), str(f["stage"]), str(f.get("page_no"))))
    return ErrorReport(failures=failures, failure_count=len(failures))


def _build_processing_statistics(contexts: list[PipelineContext]) -> ProcessingStatistics:
    stats = ProcessingStatistics(documents_processed=len(contexts))
    for ctx in contexts:
        stats.pages_processed += ctx.book_record.page_count
        for result in ctx.stage_results.values():
            stats.stages_run += 1
            if result.status == "reused":
                stats.stages_reused += 1
            elif result.status == "completed":
                stats.stages_computed += 1
    return stats


# --------------------------------------------------------------------------------------------------------------
# Export


def _resolve_report_dir(config: PipelineConfig, run_id: str) -> Path:
    if config.report_dir is not None:
        return config.report_dir
    return config.output_dir / run_id / "reports"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_config_snapshot(
    snapshot: ConfigurationSnapshot, *, output_dir: Path, run_id: str
) -> Path:
    """Write the ConfigurationSnapshot to ``<output_dir>/<run_id>/config_snapshot.json`` (FR-016)."""
    path = output_dir / run_id / "config_snapshot.json"
    _write_json(path, snapshot.to_dict())
    return path


def write_reports(
    report: ExecutionReport,
    config: PipelineConfig,
    *,
    run_id: str,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """Write the report family (JSON and/or CSV) and the machine-readable metrics artifact (FR-031/32)."""
    formats = formats or config.report_formats or ["json"]
    report_dir = _resolve_report_dir(config, run_id)
    written: dict[str, Path] = {}

    json_artifacts = {
        "execution_report": report.to_dict(),
        "pipeline_summary": report.pipeline_summary.to_dict(),
        "provider_summary": report.provider_summary.to_dict(),
        "performance_report": report.performance_report.to_dict(),
        "error_report": report.error_report.to_dict(),
        "processing_statistics": report.processing_statistics.to_dict(),
        "performance_metrics": report.metrics.to_dict(),
    }
    csv_artifacts = {
        "pipeline_summary": [report.pipeline_summary.to_dict()],
        "provider_summary": [
            {"provider": k, "usage_count": v}
            for k, v in report.provider_summary.usage_counts.items()
        ],
        "performance_report": [s.to_dict() for s in report.performance_report.stages],
        "error_report": report.error_report.failures,
        "processing_statistics": [report.processing_statistics.to_dict()],
    }

    if "json" in formats:
        for name, payload in json_artifacts.items():
            path = report_dir / f"{name}.json"
            _write_json(path, payload)
            written[f"{name}.json"] = path
    if "csv" in formats:
        for name, rows in csv_artifacts.items():
            path = report_dir / f"{name}.csv"
            _write_csv(path, rows)
            written[f"{name}.csv"] = path
    return written
