"""JSON report writer for the equation-extraction pipeline.

Provides a standalone ``write_json_report(results, output_dir)`` function that
serialises a pre-assembled results dict to ``<output_dir>/document.json``.

The ``_build_document_json`` helper is retained so callers that hold raw
pipeline objects (ClassificationResult, ExtractedEquation, RenderedPage) can
assemble the canonical dict before handing it to ``write_json_report``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from equation_extraction_pipeline.domain.models import (
    ClassificationResult,
    ExtractedEquation,
    RenderedPage,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

__all__ = ["SCHEMA_VERSION", "write_json_report", "build_document_json"]


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def build_document_json(
    pdf_path: Path,
    classification: ClassificationResult,
    pages: list[RenderedPage],
    equations: list[ExtractedEquation],
    equation_numbers: dict[str, str] | None = None,
    completeness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the final document.json structure from pipeline objects.

    Parameters
    ----------
    pdf_path:
        Path to the source PDF (used to derive ``document_id`` and
        ``source_filename``).
    classification:
        Classification result for the document.
    pages:
        List of rendered page objects; used to attach DPI / quality metadata
        to each equation.
    equations:
        All extracted equations for the document.
    equation_numbers:
        Optional mapping of ``equation_id → human-readable number string``
        (produced by ``numbering.build_equation_numbers``).

    Returns
    -------
    dict
        Canonical document dict ready to be serialised as JSON.
    """
    page_summaries = [p.to_dict() for p in pages]
    eq_nums = equation_numbers or {}

    eq_dicts = []
    for eq in equations:
        d = eq.to_dict()
        # Attach the rendering info from the corresponding page
        rp = next((p for p in pages if p.page_number == eq.region.page_number), None)
        if rp:
            d["rendering"] = {"dpi": rp.dpi, "quality_score": rp.quality_score}
        # Attach equation number from numbering.build_equation_numbers
        d["equation_number"] = eq_nums.get(eq.region.equation_id) or eq.region.label
        eq_dicts.append(d)

    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": pdf_path.stem,
        "document": {
            "book_id": pdf_path.stem,
            "source_filename": pdf_path.name,
            "page_count": classification.page_count,
            "classification": classification.to_dict(),
            "pages": page_summaries,
            "equations": eq_dicts,
            "summary": {
                "total_equations": len(equations),
                "success": sum(1 for e in equations if e.status() == "SUCCESS"),
                "uncertain": sum(1 for e in equations if e.status() == "UNCERTAIN"),
                "rejected": sum(1 for e in equations if e.status() == "REJECTED"),
            },
            "completeness": completeness,
        },
    }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_json_report(results: dict[str, Any], output_dir: Path | str) -> Path:
    """Write *results* as ``document.json`` inside *output_dir*.

    Parameters
    ----------
    results:
        The canonical document dict (as returned by :func:`build_document_json`
        or assembled by the caller).
    output_dir:
        Directory where ``document.json`` will be written.  Created if it does
        not exist.

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "document.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    doc_id = results.get("document_id", "<unknown>")
    eq_count = results.get("document", {}).get("summary", {}).get("total_equations", "?")
    logger.info("json_report_written path=%s document_id=%s equations=%s", out_path, doc_id, eq_count)
    return out_path
