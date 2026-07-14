"""Document discovery and selection for the orchestrator (feature 015, FR-001/02/03).

Resolves a run's document set from a single PDF, an explicit list, or a directory, with deterministic ordering.
Incremental and resume filtering (US2) build on ``discover_documents`` via the checkpoint ledger. Imports only
``pipeline`` + stdlib.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

__all__ = ["discover_documents", "filter_incremental"]


def discover_documents(source: Path | str | Iterable[Path | str]) -> list[Path]:
    """Resolve the document set for a run.

    - A single ``.pdf`` file path -> ``[that file]``.
    - A directory -> all ``*.pdf`` under it (recursive), sorted deterministically.
    - An iterable of paths -> those paths (files kept, directories expanded), sorted, de-duplicated.

    Returns an empty list for an empty directory or empty iterable (clean zero-document completion, FR edge case).
    """
    if isinstance(source, (str, Path)):
        return _resolve_single(Path(source))

    seen: dict[str, Path] = {}
    for item in source:
        for resolved in _resolve_single(Path(item)):
            seen[str(resolved)] = resolved
    return sorted(seen.values(), key=lambda p: str(p))


def _resolve_single(path: Path) -> list[Path]:
    path = path.expanduser()
    if path.is_dir():
        return sorted(
            (p for p in path.rglob("*.pdf") if p.is_file()),
            key=lambda p: str(p),
        )
    if path.suffix.lower() == ".pdf" and path.is_file():
        return [path]
    if path.suffix.lower() == ".pdf":
        # Non-existent but pdf-suffixed path: keep it so downstream validation reports it explicitly.
        return [path]
    return []


def filter_incremental(
    documents: list[Path],
    *,
    processed_fingerprints: set[str],
    fingerprint_of: Callable[[Path], str],
) -> list[Path]:
    """Keep only documents whose fingerprint is not already recorded as processed (FR-003).

    ``fingerprint_of`` maps a path to its content fingerprint (the ingestion sha256). Documents whose fingerprint
    is in ``processed_fingerprints`` are skipped as unchanged.
    """
    selected: list[Path] = []
    for doc in documents:
        try:
            fingerprint = fingerprint_of(doc)
        except Exception:
            selected.append(doc)
            continue
        if fingerprint not in processed_fingerprints:
            selected.append(doc)
    return selected
