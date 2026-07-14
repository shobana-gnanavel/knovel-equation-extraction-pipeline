"""ConfidenceEstimator — orchestrates the four sub-estimators and aggregates scores."""

from __future__ import annotations

import logging
from typing import Any

from .aggregation import aggregate, redistribute_weights
from .config import ConfidenceConfig
from .layout import estimate_layout
from .models import ComponentDetails, ConfidenceResult, Issue
from .ocr_quality import estimate_ocr_quality
from .recognition import estimate_recognition
from .syntax import estimate_syntax

log = logging.getLogger(__name__)


class ConfidenceEstimator:
    """Stateless orchestrator — safe to share across threads and equation candidates."""

    def __init__(self, config: ConfidenceConfig | None = None) -> None:
        self._config = config or ConfidenceConfig()

    def estimate(
        self,
        *,
        latex: str,
        crop_image: Any | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        image_metadata: dict | None = None,
        token_logprobs: list[float] | None = None,
    ) -> ConfidenceResult:
        cfg = self._config
        all_issues: list[Issue] = []
        scores: dict[str, float] = {}
        unavailable: set[str] = set()
        rec_det = lay_det = syn_det = ocq_det = None

        # Recognition
        try:
            rec_score, rec_det, rec_issues = estimate_recognition(
                latex=latex, token_logprobs=token_logprobs, config=cfg,
            )
            scores["recognition"] = rec_score
            all_issues.extend(rec_issues)
        except Exception as exc:
            log.error("confidence_recognition_failed: %s", exc, exc_info=True)
            unavailable.add("recognition")
            all_issues.append(Issue("COMPONENT_RUNTIME_ERROR", "error",
                                    f"Recognition estimator raised: {exc}", "recognition"))

        # Layout
        try:
            lay_score, lay_det, lay_issues = estimate_layout(
                crop_image=crop_image, bbox=bbox,
                image_metadata=image_metadata, config=cfg,
            )
            if lay_score is None:
                unavailable.add("layout")
            else:
                scores["layout"] = lay_score
            all_issues.extend(lay_issues)
        except Exception as exc:
            log.error("confidence_layout_failed: %s", exc, exc_info=True)
            unavailable.add("layout")
            all_issues.append(Issue("COMPONENT_RUNTIME_ERROR", "error",
                                    f"Layout estimator raised: {exc}", "layout"))

        # Syntax
        try:
            syn_score, syn_det, syn_issues = estimate_syntax(latex=latex, config=cfg)
            scores["syntax"] = syn_score
            all_issues.extend(syn_issues)
        except Exception as exc:
            log.error("confidence_syntax_failed: %s", exc, exc_info=True)
            unavailable.add("syntax")
            all_issues.append(Issue("COMPONENT_RUNTIME_ERROR", "error",
                                    f"Syntax estimator raised: {exc}", "syntax"))

        # OCR quality
        try:
            ocq_score, ocq_det, ocq_issues = estimate_ocr_quality(latex=latex, config=cfg)
            scores["ocr_quality"] = ocq_score
            all_issues.extend(ocq_issues)
        except Exception as exc:
            log.error("confidence_ocr_quality_failed: %s", exc, exc_info=True)
            unavailable.add("ocr_quality")
            all_issues.append(Issue("COMPONENT_RUNTIME_ERROR", "error",
                                    f"OCR quality estimator raised: {exc}", "ocr_quality"))

        # Aggregation
        weights = redistribute_weights(dict(cfg.weights), unavailable=unavailable)
        overall = aggregate(scores, weights, cfg.strategy) if scores else 0.0

        return ConfidenceResult(
            overall_confidence=round(max(0.0, min(1.0, overall)), 4),
            recognition=round(scores["recognition"], 4) if "recognition" in scores else None,
            layout=round(scores["layout"], 4) if "layout" in scores else None,
            syntax=round(scores["syntax"], 4) if "syntax" in scores else None,
            ocr_quality=round(scores["ocr_quality"], 4) if "ocr_quality" in scores else None,
            issues=all_issues,
            details=ComponentDetails(
                recognition=rec_det,
                layout=lay_det,
                syntax=syn_det,
                ocr_quality=ocq_det,
            ),
            aggregation_strategy=cfg.strategy.value,
            weights_used=weights,
            components_available={
                "recognition": "recognition" not in unavailable,
                "layout":      "layout"      not in unavailable,
                "syntax":      "syntax"      not in unavailable,
                "ocr_quality": "ocr_quality" not in unavailable,
            },
        )
