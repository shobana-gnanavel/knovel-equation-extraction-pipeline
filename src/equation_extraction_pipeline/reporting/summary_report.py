"""Summary / quality reporting for the equation-extraction pipeline.

This module consolidates four quality sub-systems:

* **Evaluation** (recall / precision / F1) — :func:`evaluate_equations`
* **Quality signals** — :func:`table_quality_signals`, :func:`page_quality_signals`
* **Coverage validation** — :class:`EquationCoverageValidator`,
  :func:`validate_equation_coverage`
* **Content filtering** — :func:`filter_tables`

Public API mirrors the original source modules so existing call-sites need
only update their import path.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from equation_extraction_pipeline.domain.models import PageMeta, RawTable, TableRecord

logger = structlog.get_logger(__name__)

# ===========================================================================
# Section 1 — Evaluation  (source: quality/equation_eval.py)
# ===========================================================================

"""Ground-truth equation-extraction evaluation (recall / precision / F1).

Compares the equation numbers the pipeline actually extracted for a document
against a human-labeled ground-truth set, and reports recall, precision, F1,
plus the concrete *missed* and *spurious* equation numbers so a regression is
a number and a diff, not a guess.

Recall     = |detected ∩ expected| / |expected|      → "did we find the real equations?"
Precision  = |detected ∩ expected| / |detected|      → "is what we found actually equations?"

Number strings are normalised (whitespace stripped, OCR ``l``→``1`` after a
digit-dot, surrounding brackets removed) before comparison so ``"(12.2.1)"``
and ``"12.2.1"`` match.
"""

__all__ = [
    # Evaluation
    "EquationEvalResult",
    "normalize_number",
    "evaluate_equations",
    # Quality signals
    "table_quality_signals",
    "page_quality_signals",
    # Coverage validation
    "CoverageResult",
    "EquationCoverageValidator",
    "validate_equation_coverage",
    # Content filtering
    "filter_tables",
]

_OCR_DIGIT_L = re.compile(r"(?<=\d\.)l(?=[(\s]|$)")


def normalize_number(raw: str) -> str:
    """Normalize an equation-number string for set comparison (brackets, whitespace, OCR l→1)."""
    if not raw:
        return ""
    text = _OCR_DIGIT_L.sub("1", raw.strip())
    text = text.strip("()[]{} \t")
    return text.replace(" ", "").lower()


@dataclass
class EquationEvalResult:
    """Recall/precision/F1 plus the concrete disagreements for one document."""

    expected_count: int
    detected_count: int
    true_positives: int
    recall: float
    precision: float
    f1: float
    missed: list[str] = field(default_factory=list)   # in ground truth, not detected
    spurious: list[str] = field(default_factory=list)  # detected, not in ground truth

    def as_dict(self) -> dict:
        return {
            "expected_count": self.expected_count,
            "detected_count": self.detected_count,
            "true_positives": self.true_positives,
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "missed": self.missed,
            "spurious": self.spurious,
        }


def evaluate_equations(
    detected_numbers: list[str],
    expected_numbers: list[str],
) -> EquationEvalResult:
    """Compare detected vs expected equation numbers; return recall/precision/F1 and the diffs.

    Both inputs are lists of raw number strings (possibly with brackets / OCR noise).  Empty and
    duplicate entries are dropped after normalisation, so the comparison is set-based.
    """
    detected = {n for n in (normalize_number(x) for x in detected_numbers) if n}
    expected = {n for n in (normalize_number(x) for x in expected_numbers) if n}

    tp = len(detected & expected)
    recall = tp / len(expected) if expected else 0.0
    precision = tp / len(detected) if detected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return EquationEvalResult(
        expected_count=len(expected),
        detected_count=len(detected),
        true_positives=tp,
        recall=recall,
        precision=precision,
        f1=f1,
        missed=sorted(expected - detected),
        spurious=sorted(detected - expected),
    )


# ===========================================================================
# Section 2 — Quality signals  (source: quality/signals.py)
# ===========================================================================

"""Quality signal definitions and helper utilities."""


def table_quality_signals(table: RawTable) -> dict[str, object]:
    return {
        "source_extractor": table.source_extractor,
        "cell_count": len(table.cells),
        "caption_present": bool(table.caption.strip()),
        "footnote_count": len(table.footnotes),
        "confidence": table.confidence,
        "parsing_accuracy": table.parsing_accuracy,
    }


def page_quality_signals(page_meta: PageMeta) -> dict[str, object]:
    return {
        "page_type": page_meta.page_type,
        "word_count": page_meta.word_count,
        "has_real_fonts": page_meta.has_real_fonts,
        "image_coverage": page_meta.image_coverage,
        "render_similarity": page_meta.render_similarity,
        "classification_confidence": page_meta.classification_confidence,
    }


# ===========================================================================
# Section 3 — Coverage validation  (source: quality/coverage_validator.py)
# ===========================================================================

"""Coverage validation: measures whether the extraction pipeline found all equations.

Uses pdfminer.six to scan the PDF text layer for standalone equation labels
(e.g. "Eq. 12.3.4", "(2-26)") and compares them against the extracted
equation set.

Fast, deterministic, and dependency-light — no LLM calls, no ML models, no
Docling.
"""

# Matches standalone equation-number labels in the PDF text layer.
# Kept in sync with equation_extraction/detection.py — supports dot-separated
# (12.2.1) and dash-separated (2-26) equation numbering schemes.
_EQUATION_LABEL = re.compile(
    r"""
    ^\s*
    (?:
        \(\s*(?:\d{1,3}(?:[.\-]\d{1,3}){0,3}|[A-Z][.\-]\d{1,3}|[ivxlIVXL]{1,5})\s*\)
        | \[\s*(?:\d{1,3}(?:[.\-]\d{1,3}){0,3}|[A-Z][.\-]\d{1,3})\s*\]
        | Eq(?:uation)?[.:]?\s*\(?\s*\d{1,3}(?:[.\-]\d{1,3}){0,3}\s*\)?
    )
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Requires at least one separator segment so bare list markers like (1) or (i)
# are not counted as equation labels.
_LABEL_NUMBER = re.compile(
    r"\d{1,3}(?:[.\-]\d{1,3}){1,3}|\d{1,3}[.\-]\d{1,3}|[A-Z][.\-]\d{1,3}",
    re.IGNORECASE,
)


@dataclass
class CoverageResult:
    """Result of a single coverage validation run."""

    labeled_count: int           # standalone equation labels found in PDF text layer
    extracted_count: int         # equations in the extraction output
    coverage_score: float        # min(extracted / labeled, 1.0); 0.0 when labeled_count == 0
    coverage_verdict: str        # "complete" | "partial" | "incomplete" | "unknown"
    missing_labels: list[str] = field(default_factory=list)  # in PDF but not extracted

    def summary(self) -> str:
        pct = round(self.coverage_score * 100, 1)
        icon = {"complete": "✓", "partial": "⚠", "incomplete": "✗"}.get(
            self.coverage_verdict, "?"
        )
        lines = [
            "━━━ COVERAGE VALIDATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  PDF labeled equations : {self.labeled_count}",
            f"  Extracted equations   : {self.extracted_count}",
            f"  Coverage              : {pct}%  →  {icon} {self.coverage_verdict.upper()}",
        ]
        if self.missing_labels:
            shown = ", ".join(self.missing_labels[:20])
            if len(self.missing_labels) > 20:
                shown += f" … (+{len(self.missing_labels) - 20} more)"
            lines.append(f"  Missing labels        : {shown}")
        else:
            lines.append("  Missing labels        : none detected")
        lines.append(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        return "\n".join(lines)


class EquationCoverageValidator:
    """
    Validates extraction coverage by scanning the PDF text layer for equation labels
    and comparing them against the extracted equation set.

    Uses pdfminer.six only — no ML models, no LLM calls, no Docling dependency.
    Fast and deterministic for any PDF with a readable text layer.
    """

    def _scan_labels(self, pdf_path: Path) -> tuple[int, list[str]]:
        """Return (count, labels) from a pdfminer.six scan of the PDF text layer."""
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextBoxHorizontal

            seen: dict[str, int] = {}
            for page_idx, page_layout in enumerate(extract_pages(str(pdf_path))):
                page_no = page_idx + 1
                for element in page_layout:
                    if not isinstance(element, LTTextBoxHorizontal):
                        continue
                    for line in element.get_text().splitlines():
                        s = line.strip()
                        if _EQUATION_LABEL.match(s):
                            m = _LABEL_NUMBER.search(s)
                            if m and m.group(0) not in seen:
                                seen[m.group(0)] = page_no

            logger.info(
                "coverage_validator.scan_done",
                pdf=str(pdf_path),
                labeled_count=len(seen),
            )
            return len(seen), list(seen.keys())

        except ImportError:
            logger.warning("coverage_validator.pdfminer_unavailable", pdf=str(pdf_path))
            return 0, []
        except Exception:
            logger.exception("coverage_validator.scan_failed", pdf=str(pdf_path))
            return 0, []

    def validate(
        self,
        equations: list[dict[str, Any]],
        pdf_path: str | Path,
    ) -> CoverageResult:
        """Compare extracted equations against labeled equations found in the PDF.

        Args:
            equations: List of equation dicts from the extraction output.
            pdf_path: Path to the source PDF.

        Returns:
            CoverageResult with coverage score, verdict, and missing label list.
        """
        labeled_count, labels = self._scan_labels(Path(pdf_path))
        extracted_count = len(equations)

        extracted_numbers: set[str] = {
            str(eq.get("equation_number", "")).strip()
            for eq in equations
            if eq.get("equation_number")
        }
        missing = [lbl for lbl in labels if lbl not in extracted_numbers]

        if labeled_count == 0:
            return CoverageResult(
                labeled_count=0,
                extracted_count=extracted_count,
                coverage_score=0.0,
                coverage_verdict="unknown",
                missing_labels=[],
            )

        coverage_score = round(min(1.0, extracted_count / labeled_count), 3)
        pct = coverage_score * 100
        if pct >= 95.0:
            verdict = "complete"
        elif pct >= 70.0:
            verdict = "partial"
        else:
            verdict = "incomplete"

        return CoverageResult(
            labeled_count=labeled_count,
            extracted_count=extracted_count,
            coverage_score=coverage_score,
            coverage_verdict=verdict,
            missing_labels=missing,
        )


def validate_equation_coverage(
    json_path: str | Path,
    pdf_path: str | Path | None = None,
) -> CoverageResult:
    """Load an equation_extraction.json and run coverage validation.

    Auto-locates the source PDF from json_path when pdf_path is not provided
    (looks for ``data/input/<doc_id>.pdf`` relative to the output directory).
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)
    equations: list[dict[str, Any]] = data.get("equations", [])

    if pdf_path is None:
        doc_id = json_path.parent.name
        candidate = json_path.parent.parent.parent / "input" / f"{doc_id}.pdf"
        if candidate.exists():
            pdf_path = candidate
            logger.info("coverage_validator.pdf_autolocated", pdf=str(pdf_path))
        else:
            logger.warning("coverage_validator.pdf_not_found", tried=str(candidate))
            return CoverageResult(
                labeled_count=0,
                extracted_count=len(equations),
                coverage_score=0.0,
                coverage_verdict="unknown",
                missing_labels=[],
            )

    return EquationCoverageValidator().validate(equations, pdf_path)


# ===========================================================================
# Section 4 — Content filtering  (source: quality/content_filter.py)
# ===========================================================================

"""Content filtering rules for low-quality or irrelevant output."""


def filter_tables(tables: list[TableRecord]) -> list[TableRecord]:
    """Filter out tables that contain no useful content.

    A table is considered useful if it has at least one of: columns, rows,
    caption, or footnotes.
    """
    return [
        table for table in tables
        if table.columns or table.rows or table.caption or table.footnotes
    ]
