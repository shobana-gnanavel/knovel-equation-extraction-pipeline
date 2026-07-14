"""Pipeline orchestration: plan-driven execution, configuration, checkpoints, retry, parallelism, reporting.

This subpackage formalizes the orchestration core that wires pipeline stages 002-014 together. It performs no
extraction itself; it executes existing stages in dependency order from a declared execution plan, loads and
validates layered configuration, persists run-level checkpoints, applies retry policies, runs document/batch
parallelism, collects metrics, and produces the run-level report family.

Stage callables are bound only in ``runner``/``pipeline.orchestrator`` (the Principle II cross-stage wiring point);
every other module here imports only ``pipeline`` and the stdlib.
"""

from __future__ import annotations

from pipeline.orchestration.config_loader import build_run_context, load_pipeline_config
from pipeline.orchestration.documents import discover_documents
from pipeline.orchestration.plan import build_execution_plan
from pipeline.orchestration.reporting import build_reports
from pipeline.orchestration.runner import process_documents, run_pipeline

__all__ = [
    "load_pipeline_config",
    "build_run_context",
    "build_execution_plan",
    "discover_documents",
    "process_documents",
    "run_pipeline",
    "build_reports",
]
