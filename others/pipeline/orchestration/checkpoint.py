"""Run-level checkpoint ledger over the existing per-stage sidecars (feature 015, FR-021..023).

Per-stage idempotent caching already exists in each stage's ``<pdf>.<stage>.json`` sidecar. This module adds a
run-level ledger that records, per document, which stages completed (with sidecar checksums) so a run can resume
from the first incomplete stage, restart from a chosen stage, and detect corrupted checkpoints and recompute them.
Imports only ``pipeline`` + stdlib.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.models import CheckpointEntry, CheckpointLedger, ExecutionPlan
from pipeline.run_logging import _pipeline_logger

__all__ = [
    "ledger_path",
    "load_ledger",
    "save_ledger",
    "checksum_file",
    "is_stage_reusable",
    "record_stage",
    "invalidate_from",
    "dependents_of",
]


def ledger_path(output_dir: Path) -> Path:
    # Stable, run-id-independent location so a later --resume run finds the prior ledger. Entries are gated by
    # (fingerprint, config_hash) so a different configuration never wrongly reuses a stage.
    return output_dir / "checkpoints" / "ledger.json"


def checksum_file(path: Path) -> str:
    """SHA-256 of a sidecar file; empty string if absent/unreadable (treated as corrupt)."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def load_ledger(output_dir: Path, run_id: str) -> CheckpointLedger:
    """Load the persisted ledger, tolerating a missing or corrupted file (returns a fresh ledger)."""
    path = ledger_path(output_dir)
    if not path.exists():
        return CheckpointLedger(pipeline_run_id=run_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CheckpointLedger.from_dict(data)
    except Exception:
        _pipeline_logger().error("checkpoint_ledger_corrupt", run_id=run_id, path=str(path))
        return CheckpointLedger(pipeline_run_id=run_id)


def save_ledger(ledger: CheckpointLedger, output_dir: Path) -> Path:
    path = ledger_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
    return path


def is_stage_reusable(
    entry: CheckpointEntry, stage: str, *, config_hash: str, sidecar: Path | None = None
) -> bool:
    """A stage is reusable iff it is recorded complete, the config hash matches, and (if a sidecar path is
    given) the sidecar still exists with a matching checksum (corruption detection, FR-023)."""
    if entry.config_hash != config_hash:
        return False
    if stage not in entry.completed_stages:
        return False
    if sidecar is not None:
        recorded = entry.sidecar_checksums.get(stage, "")
        current = checksum_file(sidecar)
        if not current or current != recorded:
            _pipeline_logger().error(
                "checkpoint_sidecar_corrupt", stage=stage, sidecar=str(sidecar)
            )
            return False
    return True


def record_stage(
    ledger: CheckpointLedger,
    *,
    book_id: str,
    fingerprint: str,
    config_hash: str,
    stage: str,
    sidecar: Path | None = None,
) -> CheckpointEntry:
    """Record a completed stage for a document in the ledger."""
    entry = ledger.entries.get(fingerprint)
    if entry is None or entry.config_hash != config_hash:
        entry = CheckpointEntry(book_id=book_id, fingerprint=fingerprint, config_hash=config_hash)
        ledger.entries[fingerprint] = entry
    if stage not in entry.completed_stages:
        entry.completed_stages.append(stage)
    if sidecar is not None:
        entry.sidecar_checksums[stage] = checksum_file(sidecar)
    return entry


def dependents_of(stage: str, plan: ExecutionPlan) -> set[str]:
    """All stages that (transitively) depend on ``stage`` within the plan."""
    result: set[str] = set()
    frontier = {stage}
    changed = True
    while changed:
        changed = False
        for spec in plan.stages:
            if spec.name in result:
                continue
            if frontier.intersection(spec.depends_on):
                result.add(spec.name)
                frontier.add(spec.name)
                changed = True
    result.discard(stage)
    return result


def invalidate_from(ledger: CheckpointLedger, stage: str, plan: ExecutionPlan) -> None:
    """Drop a stage and all its dependents from every ledger entry (restart-from-stage, FR-022)."""
    to_drop = {stage} | dependents_of(stage, plan)
    for entry in ledger.entries.values():
        entry.completed_stages = [s for s in entry.completed_stages if s not in to_drop]
        for dropped in to_drop:
            entry.sidecar_checksums.pop(dropped, None)
        entry.status = "partial"
