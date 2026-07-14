"""Common utilities and exceptions for the equation-extraction pipeline."""

from equation_extraction_pipeline.common.exceptions import (
    ConfigurationError,
    DetectionError,
    ExtractionError,
    PipelineError,
    RecognitionError,
    ValidationError,
)
from equation_extraction_pipeline.common.utils import (
    MULTI_EQ_NOTES,
    Representations,
    assemble,
    is_valid_latex,
    is_valid_mathml,
    split_stacked_crop,
    stage_guard,
)

__all__ = [
    # exceptions
    "PipelineError",
    "ExtractionError",
    "RecognitionError",
    "DetectionError",
    "ConfigurationError",
    "ValidationError",
    # utils — crop splitting
    "split_stacked_crop",
    "MULTI_EQ_NOTES",
    # utils — representations
    "Representations",
    "assemble",
    "is_valid_latex",
    "is_valid_mathml",
    # utils — stage guard
    "stage_guard",
]
