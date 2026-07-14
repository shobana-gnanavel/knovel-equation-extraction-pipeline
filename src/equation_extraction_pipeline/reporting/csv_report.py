"""CSV report writer for the equation-extraction pipeline.

Provides a standalone ``write_csv_report(results, output_dir)`` function that
writes a flat CSV of all extracted equations to
``<output_dir>/equations.csv``.

Uses only the Python standard-library ``csv`` module — no pandas dependency.

The *results* argument is the same canonical document dict produced by
:func:`equation_extraction_pipeline.reporting.json_report.build_document_json`
or written to ``document.json``.

CSV columns
-----------
equation_id, page_number, equation_number, latex, confidence, category
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["write_csv_report"]

# Ordered column names written to the CSV header.
_COLUMNS: list[str] = [
    "equation_id",
    "page_number",
    "equation_number",
    "latex",
    "confidence",
    "category",
]


def write_csv_report(results: dict[str, Any], output_dir: Path | str) -> Path:
    """Write a flat CSV of all extracted equations from *results*.

    Parameters
    ----------
    results:
        The canonical document dict (``document.json`` structure) produced by
        the pipeline.  Expected shape::

            {
                "document": {
                    "equations": [
                        {
                            "equation_id": str,
                            "page_number": int,
                            "equation_number": str | None,
                            "ocr": {"latex": str, "confidence": float},
                            "category": str | None,
                            ...
                        },
                        ...
                    ]
                }
            }

    output_dir:
        Directory where ``equations.csv`` will be written.  Created if it does
        not exist.

    Returns
    -------
    Path
        Absolute path to the written CSV file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "equations.csv"

    equations: list[dict[str, Any]] = results.get("document", {}).get("equations", [])

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for eq in equations:
            ocr: dict[str, Any] = eq.get("ocr") or {}
            row: dict[str, Any] = {
                "equation_id": eq.get("equation_id", ""),
                "page_number": eq.get("page_number", ""),
                "equation_number": eq.get("equation_number", ""),
                "latex": ocr.get("latex", ""),
                "confidence": ocr.get("confidence", ""),
                "category": eq.get("category", ""),
            }
            writer.writerow(row)

    logger.info(
        "csv_report_written path=%s rows=%d",
        out_path,
        len(equations),
    )
    return out_path
