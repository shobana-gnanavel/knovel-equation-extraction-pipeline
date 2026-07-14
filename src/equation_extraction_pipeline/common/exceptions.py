"""Custom exception hierarchy for the equation-extraction pipeline.

All pipeline exceptions derive from :class:`PipelineError` so callers can
catch the entire family with a single ``except PipelineError`` clause while
still being able to distinguish sub-types when needed.
"""

from __future__ import annotations

__all__ = [
    "PipelineError",
    "ExtractionError",
    "RecognitionError",
    "DetectionError",
    "ConfigurationError",
    "ValidationError",
]


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class ExtractionError(PipelineError):
    """Raised when an extraction stage fails for a specific equation or page."""


class RecognitionError(ExtractionError):
    """Raised when VLM/OCR recognition fails and cannot be recovered."""


class DetectionError(PipelineError):
    """Raised when equation detection produces an unusable result."""


class ConfigurationError(PipelineError):
    """Raised when the pipeline is misconfigured or required settings are missing."""


class ValidationError(PipelineError):
    """Raised when a validation constraint is violated at the pipeline level."""
