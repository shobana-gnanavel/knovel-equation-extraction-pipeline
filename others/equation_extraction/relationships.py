"""Equation relationship carry-through (feature 008, FR-015..FR-018).

Carries the upstream references that place each equation in the document: structural parent (from the
feature-006 ``ReadingOrderEntry``), caption association, page, and reading-order position. Also links
multi-page equation parts via a forward-compatible ``equation_continuation`` association when feature
006 emits one (otherwise ``continuation_ref`` stays ``None``). The stage invents no new relationship
semantics — it only resolves and carries upstream references. Pure functions.
"""

from __future__ import annotations

from pipeline.models import ReadingOrderContext

__all__ = ["build_caption_refs", "build_continuation_refs", "CAPTION_ASSOCIATION_TYPES"]

CAPTION_ASSOCIATION_TYPES: frozenset[str] = frozenset(
    {"figure_caption", "table_caption", "equation_caption"}
)


def build_caption_refs(reading_order: ReadingOrderContext | None) -> dict[str, str]:
    """Map equation region_id → caption region_id from the 006 caption associations (FR-017)."""
    refs: dict[str, str] = {}
    if reading_order is None:
        return refs
    for assoc in reading_order.associations:
        if assoc.association_type not in CAPTION_ASSOCIATION_TYPES or assoc.orphan:
            continue
        for target_id in assoc.target_region_ids:
            refs.setdefault(target_id, assoc.source_region_id)
    return refs


def build_continuation_refs(reading_order: ReadingOrderContext | None) -> dict[str, str]:
    """Map a continued equation region_id → its continuation partner (FR-018).

    Consumes a forward-compatible ``equation_continuation`` association (source = the continued part,
    target = the next part) if feature 006 emits one. Absent that signal the map is empty and
    ``continuation_ref`` stays ``None`` (multi-line equations are still handled as a single equation).
    """
    refs: dict[str, str] = {}
    if reading_order is None:
        return refs
    for assoc in reading_order.associations:
        if assoc.association_type != "equation_continuation" or assoc.orphan:
            continue
        for target_id in assoc.target_region_ids:
            refs.setdefault(assoc.source_region_id, target_id)
    return refs
