"""Ingestion package for the equation-extraction pipeline.

Re-exports the key public symbols so callers can import directly from
``equation_extraction_pipeline.ingestion`` without knowing which sub-module
a symbol lives in.
"""

from equation_extraction_pipeline.ingestion.file_validator import validate_pdf
from equation_extraction_pipeline.ingestion.pdf_loader import (
    classify_pdf,
    ingest_batch,
    ingest_document,
    load_pdf,
)

__all__ = [
    "load_pdf",
    "classify_pdf",
    "ingest_document",
    "ingest_batch",
    "validate_pdf",
]
