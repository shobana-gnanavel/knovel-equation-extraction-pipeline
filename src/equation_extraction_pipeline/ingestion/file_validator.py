"""Validate input PDFs before identity assignment.

Rejects empty, corrupt, encrypted/password-protected, and over-size files as
recoverable failures (structured result, no exception past the batch loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ValidationResult", "validate_pdf"]

# failure_reason values surfaced to the manifest.
_EMPTY = "empty"
_CORRUPT = "corrupt"
_ENCRYPTED = "encrypted"
_TOO_LARGE = "too_large"

_BYTES_PER_MB = 1024 * 1024


@dataclass
class ValidationResult:
    """Outcome of validating a single PDF (stage-internal, not a shared model)."""

    ok: bool
    failure_reason: str | None = None
    page_count: int | None = None


def validate_pdf(pdf_path: Path, *, max_file_mb: int = 0) -> ValidationResult:
    """Validate a PDF. ``max_file_mb`` of 0 (default) means no size limit."""
    if not pdf_path.exists() or not pdf_path.is_file():
        return ValidationResult(ok=False, failure_reason=_EMPTY)

    size = pdf_path.stat().st_size
    if size == 0:
        return ValidationResult(ok=False, failure_reason=_EMPTY)
    if max_file_mb > 0 and size > max_file_mb * _BYTES_PER_MB:
        return ValidationResult(ok=False, failure_reason=_TOO_LARGE)

    # Lazy import to avoid a circular dependency: pdf_loader imports validate_pdf
    # from this module, and this function needs open_document from pdf_loader.
    from equation_extraction_pipeline.ingestion.pdf_loader import open_document  # noqa: PLC0415

    try:
        with open_document(str(pdf_path)) as document:
            page_count = len(document)
    except Exception as exc:  # backend raises on corrupt/encrypted documents
        message = str(exc).lower()
        reason = _ENCRYPTED if ("password" in message or "encrypt" in message) else _CORRUPT
        return ValidationResult(ok=False, failure_reason=reason)

    if page_count <= 0:
        return ValidationResult(ok=False, failure_reason=_CORRUPT)

    return ValidationResult(ok=True, page_count=page_count)
