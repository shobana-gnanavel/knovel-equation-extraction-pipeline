"""The orchestration run controller (feature 015, FR-001/04/33/34/35).

``process_documents`` drives the resolved ``ExecutionPlan`` over a document set. It supports two seams:

* a generic, plan-driven path over an injectable ``stage_executors`` map — the unit-testable core that iterates
  stages in dependency order, contains per-stage failures, honors skip/critical-abort/error actions, and records a
  ``StageResult`` per stage; and
* the production path, which delegates each document to the proven ``process_book`` engine (preserving its exact
  behavior and sidecars) and synthesizes per-stage results from the resolved plan + ``failures.jsonl``.

This is the one orchestration module permitted to bind stage execution (Principle II wiring layer); it lazy-imports
``process_book`` to avoid an import cycle with ``pipeline.orchestrator``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pipeline.models import (
    BookRecord,
    CheckpointLedger,
    ExecutionReport,
    PipelineConfig,
    PipelineContext,
    RunContext,
    StageFailure,
    StageResult,
)
from pipeline.orchestration.checkpoint import (
    invalidate_from,
    is_stage_reusable,
    load_ledger,
    record_stage,
    save_ledger,
)
from pipeline.orchestration.config_loader import (
    build_run_context,
    build_snapshot,
    load_pipeline_config,
)
from pipeline.orchestration.metrics import MetricsCollector
from pipeline.orchestration.parallel import map_documents
from pipeline.orchestration.progress import progress_for
from pipeline.orchestration.registry import resolve_provider_id
from pipeline.orchestration.reporting import (
    build_reports,
    write_config_snapshot,
    write_reports,
)
from pipeline.orchestration.retry import RetryOutcome, execute_with_retry
from pipeline.run_logging import _pipeline_logger, configure_logging, get_failures

__all__ = ["process_documents", "run_pipeline", "run_plan_for_document"]

# A stage executor receives the shared per-document state dict and performs the stage's work (or raises).
StageExecutor = Callable[[dict], object]
# A document executor turns one document into a PipelineContext.
DocumentExecutor = Callable[[Path, RunContext, MetricsCollector], PipelineContext]


def _error_action(config: PipelineConfig, stage: str, *, critical: bool) -> str:
    explicit = config.error_actions.get(stage)
    if explicit:
        return explicit
    return "abort" if critical else "continue"


def run_plan_for_document(
    doc: Path,
    run_context: RunContext,
    stage_executors: Mapping[str, StageExecutor],
    metrics: MetricsCollector,
    *,
    ledger: CheckpointLedger | None = None,
) -> PipelineContext:
    """Generic plan-driven execution of one document over an injected stage-executor map.

    Iterates the resolved plan in order; a stage with no executor is ``skipped``; a stage that raises is contained
    (recorded ``failed`` with a ``StageFailure``), and a critical stage with an ``abort`` action stops the document
    (FR-035). When a ``ledger`` is supplied and resume is enabled, stages already recorded complete are reused
    rather than re-executed (FR-002). Returns the accumulated ``PipelineContext``.
    """
    book_record = BookRecord(
        book_id=doc.stem,
        pdf_path=doc,
        sha256="",
        page_count=0,
        pipeline_run_id=run_context.pipeline_run_id,
        created_at=datetime.now(timezone.utc),
        source_filename=doc.name,
    )
    pctx = PipelineContext(book_record=book_record)
    state: dict = {"doc": doc, "run_context": run_context, "book_record": book_record}
    logger = _pipeline_logger()
    aborted = False
    config_hash = run_context.config.config_hash
    fingerprint = doc.name
    resume = run_context.config.resume and ledger is not None
    entry = ledger.entries.get(fingerprint) if ledger is not None else None

    for spec in run_context.plan.stages:
        if aborted:
            pctx.stage_results[spec.name] = StageResult(stage=spec.name, status="skipped")
            continue
        executor = stage_executors.get(spec.name)
        if executor is None:
            # An intentionally not-run stage (no executor / optional). Record it so a document that has
            # traversed the whole plan counts as complete for incremental processing (FR-003).
            pctx.stage_results[spec.name] = StageResult(stage=spec.name, status="skipped")
            if ledger is not None:
                record_stage(
                    ledger,
                    book_id=book_record.book_id,
                    fingerprint=fingerprint,
                    config_hash=config_hash,
                    stage=spec.name,
                )
                entry = ledger.entries.get(fingerprint)
            continue
        if (
            resume
            and entry is not None
            and is_stage_reusable(entry, spec.name, config_hash=config_hash)
        ):
            pctx.stage_results[spec.name] = StageResult(stage=spec.name, status="reused")
            metrics.add_resume_reused(1)
            continue
        provider = resolve_provider_id(spec.name, run_context.config.providers)
        logger.info("stage_start", stage=spec.name, document=doc.name, provider=provider)
        with metrics.stage_timer(spec.name, provider=provider) as metric:
            outcome = RetryOutcome()
            stage_executor = executor
            stage_state = state
            try:
                execute_with_retry(
                    lambda: stage_executor(stage_state),
                    run_context.config.retry,
                    stage=spec.name,
                    outcome=outcome,
                )
                metrics.add_retries(outcome.retries)
                pctx.stage_results[spec.name] = StageResult(
                    stage=spec.name,
                    status="completed",
                    provider=provider,
                    duration_s=metric.duration_s,
                    retry_count=outcome.retries,
                )
                if ledger is not None:
                    record_stage(
                        ledger,
                        book_id=book_record.book_id,
                        fingerprint=fingerprint,
                        config_hash=config_hash,
                        stage=spec.name,
                    )
                    entry = ledger.entries.get(fingerprint)
            except Exception as exc:  # contained — never aborts the batch
                metric.failures = 1
                metrics.add_retries(outcome.retries)
                failure = StageFailure(
                    pipeline_run_id=run_context.pipeline_run_id,
                    book_id=book_record.book_id,
                    table_id=None,
                    page_no=None,
                    stage=spec.name,
                    error_type=type(exc).__name__,
                    error_msg=str(exc),
                    retry_count=outcome.retries,
                    is_gold_candidate=False,
                    timestamp=datetime.now(timezone.utc),
                )
                pctx.stage_results[spec.name] = StageResult(
                    stage=spec.name,
                    status="failed",
                    provider=provider,
                    duration_s=metric.duration_s,
                    retry_count=outcome.retries,
                    failure=failure,
                )
                action = _error_action(run_context.config, spec.name, critical=spec.critical)
                logger.error(
                    "stage_failed",
                    stage=spec.name,
                    document=doc.name,
                    action=action,
                    error=str(exc),
                )
                if action == "abort":
                    aborted = True

    pctx.status = _document_status(pctx)
    return pctx


def _document_status(pctx: PipelineContext) -> str:
    statuses = [r.status for r in pctx.stage_results.values()]
    if not statuses:
        return "skipped"
    if any(s == "failed" for s in statuses) and any(s in ("completed", "reused") for s in statuses):
        return "partial"
    if all(s == "failed" for s in statuses):
        return "failed"
    return "completed"


def _process_book_document_executor(
    doc: Path, run_context: RunContext, metrics: MetricsCollector
) -> PipelineContext:
    """Production path: delegate the whole document to ``process_book`` and synthesize per-stage results.

    ``process_book`` already executes stages 002-014 in order with per-stage sidecar caching and fault isolation;
    this adapter records the run-level orchestration view (PipelineContext + metrics) and derives each plan
    stage's status from ``failures.jsonl`` so failures surface in the Error Report.
    """
    from pipeline.orchestrator import process_book  # lazy import to avoid an import cycle

    logger = _pipeline_logger()
    logger.info(
        "document_start",
        document=doc.name,
        plan=run_context.plan.name,
        config_hash=run_context.config.config_hash,
    )
    with metrics.stage_timer("__document__"):
        book_record = process_book(doc, output_dir=run_context.config.output_dir)

    pctx = PipelineContext(book_record=book_record)
    failures = get_failures(book_record.pipeline_run_id, run_context.config.output_dir)
    failed_stages = {f.stage for f in failures}
    failure_by_stage = {f.stage: f for f in failures}
    for spec in run_context.plan.stages:
        if spec.name in failed_stages:
            pctx.stage_results[spec.name] = StageResult(
                stage=spec.name,
                status="failed",
                failure=failure_by_stage.get(spec.name),
            )
        else:
            pctx.stage_results[spec.name] = StageResult(stage=spec.name, status="completed")
    pctx.status = _document_status(pctx)
    return pctx


def process_documents(
    run_context: RunContext,
    *,
    stage_executors: Mapping[str, StageExecutor] | None = None,
    document_executor: DocumentExecutor | None = None,
    metrics: MetricsCollector | None = None,
    ledger: CheckpointLedger | None = None,
) -> list[PipelineContext]:
    """Drive the plan over ``run_context.documents`` and return a ``PipelineContext`` per document.

    By default each document is processed by the production ``process_book`` adapter. Tests/extensions may inject
    a ``stage_executors`` map (generic plan-driven path) or a custom ``document_executor``.
    """
    metrics = metrics or MetricsCollector(run_context.pipeline_run_id)

    def _resolve_executor() -> DocumentExecutor:
        if document_executor is not None:
            return document_executor
        if stage_executors is not None:
            return lambda doc, ctx, mc: run_plan_for_document(
                doc, ctx, stage_executors, mc, ledger=ledger
            )
        return _process_book_document_executor

    executor = _resolve_executor()
    parallelism = run_context.config.parallelism
    contexts: list[PipelineContext] = []
    with progress_for(len(run_context.documents)) as progress:

        def _process_one(doc: Path) -> PipelineContext:
            pctx = executor(doc, run_context, metrics)
            metrics.record_document(pctx.status)
            progress.advance(doc.name)
            return pctx

        if parallelism.doc_workers > 1 and len(run_context.documents) > 1:
            contexts = map_documents(
                run_context.documents,
                _process_one,
                workers=parallelism.doc_workers,
                memory_limit_mb=parallelism.memory_limit_mb,
            )
        else:
            contexts = [_process_one(doc) for doc in run_context.documents]
    return contexts


def run_pipeline(
    documents: Sequence[Path] | Path | str,
    config: PipelineConfig | None = None,
    *,
    resume: bool = False,
    incremental: bool = False,
    restart_from: str | None = None,
    stage_executors: Mapping[str, StageExecutor] | None = None,
    document_executor: DocumentExecutor | None = None,
) -> ExecutionReport:
    """Top-level orchestration entrypoint (feature 015, FR-001/31/32).

    Discovers the document set, builds the effective config + plan + run context, drives the plan with metrics +
    progress, writes the ConfigurationSnapshot + report family, and returns the ExecutionReport. A contained
    per-document/per-stage failure never raises; only config-validation or a critical abort surfaces as an error.
    """
    from pipeline.orchestration.documents import discover_documents

    config = config or load_pipeline_config()
    if resume:
        config.resume = True
    if incremental:
        config.incremental = True
    if restart_from is not None:
        config.restart_from = restart_from

    resolved_docs = discover_documents(documents)
    run_context = build_run_context(resolved_docs, config)

    configure_logging(run_context.pipeline_run_id)
    logger = _pipeline_logger()
    logger.info(
        "pipeline_run_start",
        run_id=run_context.pipeline_run_id,
        documents=len(resolved_docs),
        plan=run_context.plan.name,
        plan_order=run_context.plan.stage_names(),
        config_hash=config.config_hash,
    )

    ledger: CheckpointLedger | None = None
    if config.checkpoint_enabled:
        ledger = load_ledger(config.output_dir, run_context.pipeline_run_id)
        run_context.ledger = ledger
        if config.restart_from:
            invalidate_from(ledger, config.restart_from, run_context.plan)
        if config.incremental:
            enabled = set(run_context.plan.stage_names())

            def _is_complete(doc: Path) -> bool:
                entry = ledger.entries.get(doc.name) if ledger is not None else None
                return entry is not None and enabled.issubset(set(entry.completed_stages))

            kept = [doc for doc in run_context.documents if not _is_complete(doc)]
            skipped = len(run_context.documents) - len(kept)
            if skipped:
                logger_inc = _pipeline_logger()
                logger_inc.info("incremental_skip", skipped=skipped, remaining=len(kept))
            run_context.documents = kept

    metrics = MetricsCollector(run_context.pipeline_run_id)
    contexts = process_documents(
        run_context,
        stage_executors=stage_executors,
        document_executor=document_executor,
        metrics=metrics,
        ledger=ledger,
    )
    run_metrics = metrics.finalize()

    if ledger is not None:
        save_ledger(ledger, config.output_dir)

    report = build_reports(run_context, contexts, run_metrics)

    snapshot = build_snapshot(run_context)
    write_config_snapshot(
        snapshot, output_dir=config.output_dir, run_id=run_context.pipeline_run_id
    )
    write_reports(report, config, run_id=run_context.pipeline_run_id)

    logger.info(
        "pipeline_run_complete",
        run_id=run_context.pipeline_run_id,
        exit_status=report.exit_status,
        documents_succeeded=report.pipeline_summary.documents_succeeded,
        documents_failed=report.pipeline_summary.documents_failed,
    )
    return report
