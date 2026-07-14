"""Duplicate resolver — merged module.

Sections
--------
1. Fingerprinting                 (from ingestion/fingerprint.py)
   Streaming SHA-256 binary fingerprint computation for ingested PDFs.

2. Duplicate index                (from ingestion/duplicates.py)
   Persisted ingestion index backed by a JSON file on disk.  The on-disk
   index is the single source of truth: it is updated atomically after each
   document is ingested, so within-batch duplicates are caught the same way
   as cross-run ones.  An in-memory set is layered on top purely as a
   fast-path and never changes the outcome.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from equation_extraction_pipeline.domain.models import (
    DocumentIdentity,
    DuplicateRelationship,
)

__all__ = [
    # fingerprint
    "compute_fingerprint",
    "FINGERPRINT_ALGORITHM",
    # duplicate index
    "DuplicateIndex",
    "INDEX_VERSION",
]


# ---------------------------------------------------------------------------
# Section 1 — Fingerprinting  (from ingestion/fingerprint.py)
# ---------------------------------------------------------------------------
# Streaming SHA-256 binary fingerprint for ingested PDFs.

FINGERPRINT_ALGORITHM = "sha256"

# 1 MiB chunks: bounds memory so extremely large PDFs do not exhaust RAM (SC-006).
_CHUNK_SIZE = 1024 * 1024


def compute_fingerprint(pdf_path: Path, *, chunk_size: int = _CHUNK_SIZE) -> str:
    """Return the full SHA-256 hex digest of the file's raw bytes, read in chunks."""
    hasher = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Section 2 — Duplicate index  (from ingestion/duplicates.py)
# ---------------------------------------------------------------------------
# Duplicate detection backed by a persisted ingestion index.

INDEX_VERSION = "1.0.0"


def _empty_index() -> dict:
    return {"index_version": INDEX_VERSION, "by_fingerprint": {}, "by_logical_id": {}}


class DuplicateIndex:
    """Load/query/update the persisted ingestion index."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data = self._load()
        self._seen_fingerprints: set[str] = set()
        self._seen_logical_ids: set[str] = set()

    def _load(self) -> dict:
        if not self.path.exists():
            return _empty_index()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return _empty_index()
        data.setdefault("index_version", INDEX_VERSION)
        data.setdefault("by_fingerprint", {})
        data.setdefault("by_logical_id", {})
        return data

    def find_duplicate(self, identity: DocumentIdentity) -> DuplicateRelationship | None:
        """Return a DuplicateRelationship if this identity is already known, else None.

        Binary (matching fingerprint) takes precedence over logical (matching id).
        """
        by_fp = self._data["by_fingerprint"]
        by_lid = self._data["by_logical_id"]

        if identity.fingerprint in by_fp:
            entry = by_fp[identity.fingerprint]
            detected = (
                "batch" if identity.fingerprint in self._seen_fingerprints else "index"
            )
            return DuplicateRelationship(
                duplicate_type="binary",
                original_logical_id=entry.get("logical_id", ""),
                original_fingerprint=identity.fingerprint,
                original_manifest_ref=entry.get("manifest_ref", ""),
                detected_in=detected,
            )

        if identity.logical_id in by_lid:
            entry = by_lid[identity.logical_id]
            detected = (
                "batch" if identity.logical_id in self._seen_logical_ids else "index"
            )
            return DuplicateRelationship(
                duplicate_type="logical",
                original_logical_id=identity.logical_id,
                original_fingerprint=entry.get("fingerprint", ""),
                original_manifest_ref=entry.get("manifest_ref", ""),
                detected_in=detected,
            )

        return None

    def register(
        self,
        identity: DocumentIdentity,
        manifest_ref: str,
        source_filename: str,
    ) -> None:
        """Record a newly-ingested document and persist the index atomically."""
        self._data["by_fingerprint"][identity.fingerprint] = {
            "logical_id": identity.logical_id,
            "manifest_ref": manifest_ref,
            "source_filename": source_filename,
        }
        self._data["by_logical_id"][identity.logical_id] = {
            "fingerprint": identity.fingerprint,
            "manifest_ref": manifest_ref,
        }
        self._seen_fingerprints.add(identity.fingerprint)
        self._seen_logical_ids.add(identity.logical_id)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp_path, self.path)
