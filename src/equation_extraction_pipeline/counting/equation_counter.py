"""Equation numbering carry-through.

Carries the equation number from the detected label (stored in ``EquationRegion.label``)
into a simple mapping that the orchestrator can use when writing document.json.
Pure functions.
"""

from __future__ import annotations

from equation_extraction_pipeline.domain.models import EquationRegion

__all__ = ["build_equation_numbers"]


def build_equation_numbers(regions: list[EquationRegion]) -> dict[str, str]:
    """Map equation_id → number string from the region label.

    Regions with no label (``label is None`` or empty) are omitted; the
    orchestrator leaves ``equation_number`` as ``None`` for those entries.
    """
    return {r.equation_id: r.label for r in regions if r.label}
