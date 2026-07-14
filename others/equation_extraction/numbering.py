"""Equation numbering carry-through (feature 008, FR-014).

Carries the equation number from the feature-006 ``equation_number`` association (source = the
``equation_number`` region whose text is recorded in its layout ``attributes["value"]``, target = the
equation region). Numbered equations copy the number string; unnumbered equations carry ``None`` and
are not flagged. Numbers are never invented. Pure functions.
"""

from __future__ import annotations

from pipeline.models import LayoutRegion, ReadingOrderContext

__all__ = ["build_equation_numbers"]


def build_equation_numbers(
    reading_order: ReadingOrderContext | None,
    region_by_id: dict[str, LayoutRegion],
) -> dict[str, str]:
    """Map equation region_id → number string from the 006 ``equation_number`` associations (FR-014)."""
    numbers: dict[str, str] = {}
    if reading_order is None:
        return numbers
    for assoc in reading_order.associations:
        if assoc.association_type != "equation_number" or assoc.orphan:
            continue
        source = region_by_id.get(assoc.source_region_id)
        value = ""
        if source is not None and isinstance(source.attributes, dict):
            value = str(source.attributes.get("value", "")).strip()
        if not value:
            continue
        for target_id in assoc.target_region_ids:
            numbers[target_id] = value
    return numbers
