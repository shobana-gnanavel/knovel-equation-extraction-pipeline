"""Scoring logic for extraction quality and confidence."""

from __future__ import annotations

from pipeline.models import PageMeta, RawTable, TableQualityScore

__all__ = ["score_table"]


def score_table(table: RawTable, page_meta: PageMeta) -> TableQualityScore:
    cell_count = max(len(table.cells), 1)
    text_cells = sum(1 for cell in table.cells if cell.text.strip())
    structure_score = max(0.0, min(100.0, 60.0 + min(cell_count, 20) * 2.0))
    text_score = max(0.0, min(100.0, (text_cells / cell_count) * 100.0))
    completeness_base = (
        50.0 + (20.0 if table.caption else 0.0) + min(len(table.footnotes) * 5.0, 20.0)
    )
    completeness_score = max(
        0.0, min(100.0, completeness_base + (10.0 if page_meta.has_real_fonts else 0.0))
    )
    return TableQualityScore(
        structure_score=structure_score,
        text_score=text_score,
        completeness_score=completeness_score,
    )
