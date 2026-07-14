"""Confidence aggregation strategies."""

from __future__ import annotations

import math
from enum import Enum


class AggregationStrategy(str, Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    GEOMETRIC_MEAN = "geometric_mean"
    MIN = "min"
    HARMONIC_MEAN = "harmonic_mean"


def redistribute_weights(
    weights: dict[str, float],
    unavailable: set[str],
) -> dict[str, float]:
    active = {k: v for k, v in weights.items() if k not in unavailable}
    total = sum(active.values())
    if total == 0.0:
        return {}
    return {k: v / total for k, v in active.items()}


def aggregate(
    scores: dict[str, float],
    weights: dict[str, float],
    strategy: AggregationStrategy,
) -> float:
    if not scores:
        return 0.0

    if strategy == AggregationStrategy.WEIGHTED_AVERAGE:
        return sum(scores[k] * weights[k] for k in scores)

    if strategy == AggregationStrategy.GEOMETRIC_MEAN:
        if any(scores[k] == 0.0 for k in scores):
            return 0.0
        log_sum = sum(weights[k] * math.log(scores[k]) for k in scores)
        return math.exp(log_sum)

    if strategy == AggregationStrategy.MIN:
        return min(scores.values())

    if strategy == AggregationStrategy.HARMONIC_MEAN:
        denom = sum(weights[k] / scores[k] for k in scores if scores[k] > 0.0)
        if denom == 0.0:
            return 0.0
        return 1.0 / denom

    raise ValueError(f"Unknown aggregation strategy: {strategy}")
