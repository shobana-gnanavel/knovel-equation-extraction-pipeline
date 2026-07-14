"""Confidence normalization and low-confidence handling (feature 008, FR-008/FR-019).

Pure functions: clamp classification and recognition scores into ``[0,1]``, aggregate a page/document
recognition confidence, and flag equations below the configured thresholds. Sub-threshold equations
are flagged and retained — never dropped. Deterministic for a given input + config.
"""

from __future__ import annotations

from pipeline.models import Equation

__all__ = ["clamp", "page_confidence", "flag_low_confidence"]


def clamp(value: float) -> float:
    """Clamp a confidence into ``[0,1]`` and round for stable serialization."""
    return round(max(0.0, min(1.0, float(value))), 4)


def page_confidence(equations: list[Equation]) -> float:
    """Mean of equation recognition confidences (0.0 for a page with no equations)."""
    if not equations:
        return 0.0
    return round(sum(eq.recognition_confidence for eq in equations) / len(equations), 4)


def flag_low_confidence(
    equations: list[Equation],
    *,
    classification_threshold: float,
    recognition_threshold: float,
) -> tuple[int, int]:
    """Flag sub-threshold classification/recognition and retain. Returns ``(class_count, recog_count)``."""
    class_count = 0
    recog_count = 0
    for eq in equations:
        if eq.classification_confidence < classification_threshold:
            class_count += 1
            if "low_confidence_classification" not in eq.validation_flags:
                eq.validation_flags.append("low_confidence_classification")
        if eq.recognition_confidence < recognition_threshold:
            recog_count += 1
            if "low_confidence_recognition" not in eq.validation_flags:
                eq.validation_flags.append("low_confidence_recognition")
    return class_count, recog_count
