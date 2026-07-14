"""Equation-extraction validation (feature 008, FR-021/FR-022).

Cross-equation and reference checks that detect and record (never silently fix) findings: duplicate
equations, reading-order inconsistency versus feature 006, missing equation numbers where one was
expected, and invalid parent references. Per-equation findings that need recognition context
(invalid bbox/LaTeX/MathML, broken multi-line, low-confidence, unsupported category) are set by the
extractor/representations/confidence; this module adds the cross-equation flags and tallies every
flag into counts for the run summary. Findings are flagged and retained — nothing is dropped.
"""

from __future__ import annotations

from collections import Counter

from pipeline.models import Equation, ReadingOrderContext

__all__ = ["validate", "VALIDATION_FLAGS", "expected_numbered_regions"]

VALIDATION_FLAGS: list[str] = [
    "missing_equation",
    "duplicate",
    "invalid_bbox",
    "missing_number",
    "broken_multiline",
    "invalid_latex",
    "invalid_mathml",
    "order_inconsistent",
    "low_confidence_classification",
    "low_confidence_recognition",
    "invalid_parent",
    "unsupported_category",
]


def _flag(equation: Equation, name: str) -> None:
    if name not in equation.validation_flags:
        equation.validation_flags.append(name)


def expected_numbered_regions(reading_order: ReadingOrderContext | None) -> set[str]:
    """Region ids that are targets of a non-orphan ``equation_number`` association (FR-014)."""
    expected: set[str] = set()
    if reading_order is None:
        return expected
    for assoc in (reading_order.associations or []):
        if assoc.association_type == "equation_number" and not assoc.orphan:
            expected.update(assoc.target_region_ids)
    return expected


def _equation_key(equation: Equation) -> str:
    return (equation.latex or equation.structured_form or equation.plain_text or "").strip()


def _mark_duplicates(equations: list[Equation]) -> None:
    seen: dict[str, Equation] = {}
    for eq in equations:
        key = _equation_key(eq)
        if not key:
            continue
        if key in seen:
            _flag(seen[key], "duplicate")
            _flag(eq, "duplicate")
        else:
            seen[key] = eq


def _mark_order(equations: list[Equation], reading_order: ReadingOrderContext | None) -> None:
    if reading_order is None or not reading_order.document_sequence:
        return
    position = {region_id: i for i, region_id in enumerate(reading_order.document_sequence)}
    last = -1
    for eq in equations:
        index = position.get(eq.region_id)
        if index is None:
            continue
        if index < last:
            _flag(eq, "order_inconsistent")
        else:
            last = index


def _mark_missing_numbers(equations: list[Equation], expected: set[str]) -> None:
    for eq in equations:
        if eq.region_id in expected and not eq.equation_number:
            _flag(eq, "missing_number")


def _mark_invalid_parents(equations: list[Equation], valid_region_ids: set[str] | None) -> None:
    if valid_region_ids is None:
        return
    for eq in equations:
        parent = eq.structural_parent_id
        if parent and parent not in valid_region_ids:
            _flag(eq, "invalid_parent")


def validate(
    equations: list[Equation],
    *,
    reading_order: ReadingOrderContext | None = None,
    valid_region_ids: set[str] | None = None,
) -> dict[str, int]:
    """Run cross-equation validation, set flags, and return per-flag counts (FR-021)."""
    _mark_duplicates(equations)
    _mark_order(equations, reading_order)
    _mark_missing_numbers(equations, expected_numbered_regions(reading_order))
    _mark_invalid_parents(equations, valid_region_ids)

    counts: Counter[str] = Counter()
    for eq in equations:
        for flag in eq.validation_flags:
            counts[flag] += 1
    return dict(counts)
