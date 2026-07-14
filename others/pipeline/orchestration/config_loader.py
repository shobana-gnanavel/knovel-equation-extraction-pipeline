"""Layered configuration loading + validation for the orchestrator (feature 015, FR-013..017).

Builds an effective ``PipelineConfig`` by merging, in precedence order, **defaults -> config file (YAML/JSON, with
``extends`` inheritance) -> environment -> CLI overrides**, validates it fail-fast, records per-key layer
provenance for the ConfigurationSnapshot, and computes a deterministic ``config_hash``. Imports only ``pipeline``
+ stdlib (+ optional pyyaml/pydantic, import-guarded).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pipeline import config as _config
from pipeline.models import (
    ConfigurationSnapshot,
    ExecutionPlan,
    ParallelismConfig,
    PipelineConfig,
    RetryPolicy,
    RunContext,
)
from pipeline.orchestration.plan import STAGE_BY_NAME, build_execution_plan, resolve_enabled_stages

__all__ = [
    "compute_config_hash",
    "load_pipeline_config",
    "build_run_context",
    "build_snapshot",
    "validate_config",
    "ConfigValidationError",
]


class ConfigValidationError(ValueError):
    """Raised when the effective configuration fails validation (fail-fast, FR-015)."""


_VALID_ERROR_ACTIONS = {"retry", "skip", "continue", "abort"}


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion for CLI/file string overrides."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _disabled_from_env() -> set[str]:
    raw = _config.KNOVEL_ORCH_DISABLED_STAGES
    return {name.strip() for name in raw.split(",") if name.strip()}


def _report_formats_from_env() -> list[str]:
    raw = _config.KNOVEL_ORCH_REPORT_FORMATS
    return [fmt.strip() for fmt in raw.split(",") if fmt.strip()]


def compute_config_hash(config: PipelineConfig) -> str:
    """Deterministic hash of the effective config, excluding volatile fields (FR-036, SC-007)."""
    payload = config.to_dict()
    payload.pop("config_hash", None)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# --------------------------------------------------------------------------------------------------------------
# Layer construction


def _defaults_layer() -> dict[str, Any]:
    """Built-in defaults (lowest precedence): the env-derived KNOVEL_ORCH_* base."""
    return {
        "plan_name": _config.KNOVEL_ORCH_PLAN,
        "disabled_stages": sorted(_disabled_from_env()),
        "providers": {},
        "retry": {
            "max_attempts": _config.KNOVEL_ORCH_RETRY_MAX_ATTEMPTS,
            "backoff_factor": _config.KNOVEL_ORCH_RETRY_BACKOFF,
            "base_delay_s": _config.KNOVEL_ORCH_RETRY_BASE_DELAY_S,
            "max_delay_s": _config.KNOVEL_ORCH_RETRY_MAX_DELAY_S,
        },
        "parallelism": {
            "doc_workers": _config.KNOVEL_ORCH_DOC_WORKERS,
            "batch_workers": _config.KNOVEL_ORCH_BATCH_WORKERS,
            "memory_limit_mb": _config.KNOVEL_ORCH_MEMORY_LIMIT_MB,
        },
        "report_formats": _report_formats_from_env(),
        "checkpoint_enabled": _config.KNOVEL_ORCH_CHECKPOINT_ENABLED,
        "resume": _config.KNOVEL_ORCH_RESUME,
        "incremental": _config.KNOVEL_ORCH_INCREMENTAL,
        "benchmark_mode": _config.KNOVEL_ORCH_BENCHMARK_MODE,
        "validation_mode": _config.KNOVEL_ORCH_VALIDATION_MODE,
    }


def _load_file_layer(config_file: Path | None, _seen: set[str] | None = None) -> dict[str, Any]:
    """Load a YAML/JSON config file, resolving ``extends``/``base`` inheritance first (deep-merged)."""
    if config_file is None:
        return {}
    config_file = config_file.expanduser()
    if not config_file.exists():
        raise ConfigValidationError(f"config file not found: {config_file}")
    _seen = _seen or set()
    if str(config_file) in _seen:
        raise ConfigValidationError(f"cyclic config inheritance via {config_file}")
    _seen.add(str(config_file))

    text = config_file.read_text(encoding="utf-8")
    if config_file.suffix.lower() in (".yaml", ".yml"):
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text) if text.strip() else {}
    if not isinstance(data, dict):
        raise ConfigValidationError(f"config file must be a mapping: {config_file}")

    bases = data.pop("extends", None) or data.pop("base", None)
    merged: dict[str, Any] = {}
    if bases:
        base_list = bases if isinstance(bases, list) else [bases]
        for base in base_list:
            base_path = (config_file.parent / str(base)).resolve()
            merged = _deep_merge(merged, _load_file_layer(base_path, _seen))
    return _deep_merge(merged, data)


def _env_layer() -> dict[str, Any]:
    """Only the KNOVEL_ORCH_* values explicitly present in the environment (precedence above file)."""
    layer: dict[str, Any] = {}
    if "KNOVEL_ORCH_PLAN" in os.environ:
        layer["plan_name"] = _config.KNOVEL_ORCH_PLAN
    if "KNOVEL_ORCH_DISABLED_STAGES" in os.environ:
        layer["disabled_stages"] = sorted(_disabled_from_env())
    parallel: dict[str, Any] = {}
    if "KNOVEL_ORCH_DOC_WORKERS" in os.environ:
        parallel["doc_workers"] = _config.KNOVEL_ORCH_DOC_WORKERS
    if "KNOVEL_ORCH_BATCH_WORKERS" in os.environ:
        parallel["batch_workers"] = _config.KNOVEL_ORCH_BATCH_WORKERS
    if "KNOVEL_ORCH_MEMORY_LIMIT_MB" in os.environ:
        parallel["memory_limit_mb"] = _config.KNOVEL_ORCH_MEMORY_LIMIT_MB
    if parallel:
        layer["parallelism"] = parallel
    if "KNOVEL_ORCH_RESUME" in os.environ:
        layer["resume"] = _config.KNOVEL_ORCH_RESUME
    if "KNOVEL_ORCH_INCREMENTAL" in os.environ:
        layer["incremental"] = _config.KNOVEL_ORCH_INCREMENTAL
    if "KNOVEL_ORCH_BENCHMARK_MODE" in os.environ:
        layer["benchmark_mode"] = _config.KNOVEL_ORCH_BENCHMARK_MODE
    return layer


def _cli_layer(cli_overrides: dict[str, str] | None) -> dict[str, Any]:
    """Dotted-key CLI overrides (highest precedence): ``parallelism.doc_workers=8`` -> nested dict."""
    layer: dict[str, Any] = {}
    for key, raw in (cli_overrides or {}).items():
        parts = key.split(".")
        cursor = layer
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce(raw)
    return layer


def _normalize_aliases(layer: dict[str, Any]) -> dict[str, Any]:
    """Map human-friendly config-file keys to PipelineConfig field names (e.g. ``plan`` -> ``plan_name``)."""
    if "plan" in layer and "plan_name" not in layer:
        layer = dict(layer)
        layer["plan_name"] = layer.pop("plan")
    return layer


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _record_provenance(provenance: dict[str, str], layer: dict[str, Any], name: str) -> None:
    for key in layer:
        provenance[key] = name


# --------------------------------------------------------------------------------------------------------------
# Validation


def validate_config(config: PipelineConfig) -> None:
    """Fail-fast validation; raises ``ConfigValidationError`` before any stage runs (FR-015, SC-005)."""
    for name in list(config.enabled_stages) + list(config.disabled_stages):
        if name not in STAGE_BY_NAME:
            raise ConfigValidationError(f"unknown stage '{name}'")
    if config.parallelism.doc_workers < 1 or config.parallelism.batch_workers < 1:
        raise ConfigValidationError("worker count must be >= 1")
    if config.retry.max_attempts < 1:
        raise ConfigValidationError("invalid retry policy: max_attempts must be >= 1")
    if config.retry.backoff_factor < 1.0:
        raise ConfigValidationError("invalid retry policy: backoff_factor must be >= 1.0")
    if config.retry.base_delay_s < 0 or config.retry.max_delay_s < 0:
        raise ConfigValidationError("invalid retry policy: delays must be >= 0")
    for stage, timeout in config.timeouts.items():
        if timeout <= 0:
            raise ConfigValidationError(f"timeout must be > 0 for '{stage}'")
    for stage, action in config.error_actions.items():
        if action not in _VALID_ERROR_ACTIONS:
            raise ConfigValidationError(f"invalid error action '{action}' for '{stage}'")
    if config.restart_from is not None and config.restart_from not in config.enabled_stages:
        raise ConfigValidationError(f"restart-from stage '{config.restart_from}' not enabled")
    # Dependency-conflict detection (FR-008): building the plan raises on a disabled required dependency.
    try:
        build_execution_plan(config)
    except ValueError as exc:
        raise ConfigValidationError(str(exc)) from exc
    if _config.KNOVEL_ORCH_CONFIG_VALIDATOR.lower() == "pydantic":
        _validate_with_pydantic(config)


def _validate_with_pydantic(config: PipelineConfig) -> None:
    """Optional alternative validator; import-guarded, degrades to the pure-Python path when absent."""
    try:
        from pydantic import TypeAdapter
    except Exception:
        return
    TypeAdapter(dict).validate_python(config.to_dict())


# --------------------------------------------------------------------------------------------------------------
# Public API


def load_pipeline_config(
    *,
    config_file: Path | None = None,
    cli_overrides: dict[str, str] | None = None,
    validate: bool = True,
) -> PipelineConfig:
    """Build the effective ``PipelineConfig`` by merging defaults -> file -> env -> CLI, then validate."""
    if config_file is None and _config.KNOVEL_ORCH_CONFIG:
        config_file = Path(_config.KNOVEL_ORCH_CONFIG)

    provenance: dict[str, str] = {}
    merged = _defaults_layer()
    _record_provenance(provenance, merged, "default")

    file_layer = _normalize_aliases(_load_file_layer(config_file))
    if file_layer:
        merged = _deep_merge(merged, file_layer)
        _record_provenance(provenance, file_layer, "file")

    env_layer = _env_layer()
    if env_layer:
        merged = _deep_merge(merged, env_layer)
        _record_provenance(provenance, env_layer, "env")

    cli_layer = _normalize_aliases(_cli_layer(cli_overrides))
    if cli_layer:
        merged = _deep_merge(merged, cli_layer)
        _record_provenance(provenance, cli_layer, "cli")

    for key in ("unknown_keys",):  # placeholder for future strict-key checks
        merged.pop(key, None)

    config = PipelineConfig(
        plan_name=str(merged.get("plan_name", "full")),
        disabled_stages=sorted(set(merged.get("disabled_stages", []))),
        providers=dict(merged.get("providers", {})),
        retry=RetryPolicy.from_dict(merged.get("retry", {})),
        parallelism=ParallelismConfig.from_dict(merged.get("parallelism", {})),
        timeouts={str(k): int(v) for k, v in merged.get("timeouts", {}).items()},
        output_dir=Path(str(merged.get("output_dir", _config.KNOVEL_OUTPUT_DIR))),
        report_dir=(
            Path(str(merged["report_dir"]))
            if merged.get("report_dir")
            else (Path(_config.KNOVEL_ORCH_REPORT_DIR) if _config.KNOVEL_ORCH_REPORT_DIR else None)
        ),
        report_formats=list(merged.get("report_formats", ["json", "csv"])),
        checkpoint_enabled=bool(merged.get("checkpoint_enabled", True)),
        resume=bool(merged.get("resume", False)),
        incremental=bool(merged.get("incremental", False)),
        restart_from=merged.get("restart_from"),
        validation_mode=bool(merged.get("validation_mode", False)),
        benchmark_mode=bool(merged.get("benchmark_mode", False)),
        error_actions=dict(merged.get("error_actions", {})),
    )

    # Resolve enabled stages: an explicit allow-list (file/CLI) wins; otherwise resolve from the config flags.
    explicit_enabled = merged.get("enabled_stages")
    config.enabled_stages = resolve_enabled_stages(
        disabled=set(config.disabled_stages),
        explicit_enabled=set(explicit_enabled) if explicit_enabled else None,
    )

    if validate:
        validate_config(config)

    config.config_hash = compute_config_hash(config)
    config._provenance = provenance  # type: ignore[attr-defined]  # carried to the snapshot
    return config


def build_run_context(
    documents: list[Path],
    config: PipelineConfig | None = None,
    *,
    plan: ExecutionPlan | None = None,
) -> RunContext:
    """Assemble a ``RunContext`` (the spec's Execution Context) for a run."""
    config = config or load_pipeline_config()
    plan = plan or build_execution_plan(config)
    return RunContext(
        pipeline_run_id=str(uuid4()),
        config=config,
        plan=plan,
        documents=list(documents),
        started_at=datetime.now(timezone.utc),
    )


def build_snapshot(run_context: RunContext) -> ConfigurationSnapshot:
    """Build the ConfigurationSnapshot with per-key layer provenance (FR-016)."""
    provenance = getattr(run_context.config, "_provenance", {}) or {}
    return ConfigurationSnapshot(
        config=run_context.config,
        provenance=dict(provenance),
        plan_order=run_context.plan.stage_names(),
        config_hash=run_context.config.config_hash,
    )
