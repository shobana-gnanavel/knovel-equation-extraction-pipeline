"""Configuration for the confidence estimation module."""

from __future__ import annotations

from dataclasses import dataclass, field

from .aggregation import AggregationStrategy


@dataclass
class ConfidenceConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        "recognition": 0.35,
        "layout":      0.15,
        "syntax":      0.30,
        "ocr_quality": 0.20,
    })
    strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_AVERAGE

    # Layout thresholds
    border_strip_width_px: int = 3
    border_darkness_threshold: int = 240
    min_recommended_dpi: float = 150.0
    critical_min_dpi: float = 72.0
    min_padding_px: float = 8.0
    whitespace_band_min_height_px: int = 8
    max_plausible_aspect_ratio: float = 20.0
    min_plausible_aspect_ratio: float = 1.5

    # Recognition thresholds
    uncertain_token_threshold: float = 0.4
    min_token_prob_normalizer: float = 0.3

    # Syntax
    max_nesting_depth: int = 12

    # OCR quality
    high_text_ratio_threshold: float = 0.5

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Confidence weights must sum to 1.0, got {total:.4f}")
