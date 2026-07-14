"""Execution plan: the declarative stage DAG for pipeline stages 002-014 and its resolver.

This module is data-only — it declares the stages, their dependencies, and optional/critical flags, and builds a
topologically ordered ``ExecutionPlan`` from a ``PipelineConfig``. It imports **no** stage package; the runner binds
the actual stage callables (Principle II). The default registry reproduces the order of the existing
``process_book`` sequence so the default plan is behavior-preserving.
"""

from __future__ import annotations

from pipeline import config as _config
from pipeline.models import ExecutionPlan, PipelineConfig, StageSpec

__all__ = [
    "STAGE_SPECS",
    "STAGE_BY_NAME",
    "default_critical_stages",
    "resolve_enabled_stages",
    "build_execution_plan",
]

# Canonical stage chain (002 -> 014). `enabled_flag` is the pipeline.config attribute that gates the stage
# ("" => always on). `depends_on` is the immediate predecessor in the canonical chain; a stage enabled while a
# required predecessor is disabled is a configuration error (FR-008). Validation (014) runs BEFORE serialization
# (013) so serialization can project the ValidationContext (constitution / orchestrator order).
STAGE_SPECS: list[StageSpec] = [
    StageSpec(name="ingestion", feature_id="002", enabled_flag="", depends_on=[], optional=False),
    StageSpec(
        name="classification",
        feature_id="003",
        enabled_flag="",
        depends_on=["ingestion"],
        optional=False,
    ),
    StageSpec(
        name="preprocessing",
        feature_id="004",
        enabled_flag="KNOVEL_PREPROCESS_ENABLED",
        depends_on=["classification"],
        optional=True,
    ),
    StageSpec(
        name="layout",
        feature_id="005",
        enabled_flag="KNOVEL_LAYOUT_ENABLED",
        depends_on=["classification"],
        optional=True,
    ),
    StageSpec(
        name="reading_order",
        feature_id="006",
        enabled_flag="KNOVEL_READING_ORDER_ENABLED",
        depends_on=["layout"],
        optional=True,
    ),
    StageSpec(
        name="text_extraction",
        feature_id="007",
        enabled_flag="KNOVEL_TEXT_ENABLED",
        depends_on=["reading_order"],
        optional=True,
    ),
    StageSpec(
        name="equation_extraction",
        feature_id="008",
        enabled_flag="KNOVEL_EQUATION_ENABLED",
        depends_on=["text_extraction"],
        optional=True,
    ),
    StageSpec(
        name="table_extraction",
        feature_id="009",
        enabled_flag="KNOVEL_TABLE_ENABLED",
        depends_on=["equation_extraction"],
        optional=True,
    ),
    StageSpec(
        name="visual_extraction",
        feature_id="010",
        enabled_flag="KNOVEL_VISUAL_ENABLED",
        depends_on=["table_extraction"],
        optional=True,
    ),
    StageSpec(
        name="metadata_extraction",
        feature_id="011",
        enabled_flag="KNOVEL_METADATA_ENABLED",
        depends_on=["visual_extraction"],
        optional=True,
    ),
    StageSpec(
        name="relationship_builder",
        feature_id="012",
        enabled_flag="KNOVEL_RELATIONSHIP_ENABLED",
        depends_on=["metadata_extraction"],
        optional=True,
    ),
    StageSpec(
        name="validation",
        feature_id="014",
        enabled_flag="KNOVEL_VALIDATION_ENABLED",
        depends_on=["relationship_builder"],
        optional=True,
    ),
    StageSpec(
        name="serialization",
        feature_id="013",
        enabled_flag="KNOVEL_EXPORT_ENABLED",
        depends_on=["validation"],
        optional=True,
    ),
]

STAGE_BY_NAME: dict[str, StageSpec] = {spec.name: spec for spec in STAGE_SPECS}
_STAGE_INDEX: dict[str, int] = {spec.name: i for i, spec in enumerate(STAGE_SPECS)}


def default_critical_stages() -> set[str]:
    """Stages whose failure aborts the document, from KNOVEL_ORCH_CRITICAL_STAGES."""
    raw = _config.KNOVEL_ORCH_CRITICAL_STAGES
    return {name.strip() for name in raw.split(",") if name.strip()}


def _flag_enabled(spec: StageSpec) -> bool:
    """Whether a stage is on per its pipeline.config enabled flag ("" => always on)."""
    if not spec.enabled_flag:
        return True
    return bool(getattr(_config, spec.enabled_flag, True))


def resolve_enabled_stages(
    *,
    disabled: set[str] | None = None,
    explicit_enabled: set[str] | None = None,
) -> list[str]:
    """Resolve which canonical stages run.

    A stage runs when its config flag is on (or it is always-on), it is not explicitly disabled, and — if an
    explicit enabled allow-list is given — it is in that list. Order follows the canonical chain.
    """
    disabled = disabled or set()
    enabled: list[str] = []
    for spec in STAGE_SPECS:
        if spec.name in disabled:
            continue
        if explicit_enabled is not None and spec.name not in explicit_enabled:
            continue
        if explicit_enabled is None and not _flag_enabled(spec):
            continue
        enabled.append(spec.name)
    return enabled


def build_execution_plan(config: PipelineConfig) -> ExecutionPlan:
    """Resolve enabled/optional stages, validate dependency conflicts, and topologically sort.

    Raises ``ValueError`` if an enabled stage requires a disabled/absent stage (FR-008) or a dependency cycle is
    detected (should never occur for the fixed chain). An all-disabled plan is valid and yields empty stages.
    """
    enabled = [name for name in config.enabled_stages if name in STAGE_BY_NAME]
    enabled_set = set(enabled)
    critical = default_critical_stages()

    # Dependency-conflict validation (FR-008): every dependency of an enabled stage must also be enabled.
    for name in enabled:
        for dep in STAGE_BY_NAME[name].depends_on:
            if dep not in enabled_set:
                raise ValueError(f"stage '{name}' requires disabled stage '{dep}'")

    # Kahn topological sort over the enabled subgraph; ties broken by canonical declared index for determinism.
    indegree: dict[str, int] = {name: 0 for name in enabled}
    adjacency: dict[str, list[str]] = {name: [] for name in enabled}
    for name in enabled:
        for dep in STAGE_BY_NAME[name].depends_on:
            adjacency[dep].append(name)
            indegree[name] += 1

    ready = sorted((n for n in enabled if indegree[n] == 0), key=lambda n: _STAGE_INDEX[n])
    ordered: list[str] = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for successor in adjacency[name]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
        ready.sort(key=lambda n: _STAGE_INDEX[n])

    if len(ordered) != len(enabled):
        raise ValueError("dependency cycle detected in execution plan")

    stages: list[StageSpec] = []
    for name in ordered:
        spec = STAGE_BY_NAME[name]
        stages.append(
            StageSpec(
                name=spec.name,
                feature_id=spec.feature_id,
                enabled_flag=spec.enabled_flag,
                depends_on=list(spec.depends_on),
                optional=spec.optional,
                critical=spec.name in critical,
                provider_keys=list(spec.provider_keys),
                timeout_key=spec.timeout_key,
            )
        )

    skipped = [
        {"name": spec.name, "reason": "disabled"}
        for spec in STAGE_SPECS
        if spec.name not in enabled_set
    ]
    return ExecutionPlan(name=config.plan_name, stages=stages, skipped=skipped)
