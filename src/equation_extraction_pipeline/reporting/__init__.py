"""Reporting sub-package for the equation-extraction pipeline.

Re-exports the primary public symbols so callers can import directly from
``equation_extraction_pipeline.reporting`` without knowing which internal
module each symbol lives in.

Usage examples::

    from equation_extraction_pipeline.reporting import write_json_report
    from equation_extraction_pipeline.reporting import write_csv_report
    from equation_extraction_pipeline.reporting import evaluate_equations, EquationEvalResult
"""

from equation_extraction_pipeline.reporting.csv_report import write_csv_report
from equation_extraction_pipeline.reporting.json_report import (
    SCHEMA_VERSION,
    build_document_json,
    write_json_report,
)
from equation_extraction_pipeline.reporting.summary_report import (
    CoverageResult,
    EquationCoverageValidator,
    EquationEvalResult,
    evaluate_equations,
    filter_tables,
    normalize_number,
    page_quality_signals,
    table_quality_signals,
    validate_equation_coverage,
)

__all__ = [
    # json_report
    "write_json_report",
    "build_document_json",
    "SCHEMA_VERSION",
    # csv_report
    "write_csv_report",
    # summary_report — evaluation
    "evaluate_equations",
    "EquationEvalResult",
    "normalize_number",
    # summary_report — quality signals
    "table_quality_signals",
    "page_quality_signals",
    # summary_report — coverage validation
    "CoverageResult",
    "EquationCoverageValidator",
    "validate_equation_coverage",
    # summary_report — content filtering
    "filter_tables",
]
