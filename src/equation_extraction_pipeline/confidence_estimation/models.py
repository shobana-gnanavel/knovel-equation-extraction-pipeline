"""Output models for the confidence estimation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SCHEMA_VERSION = "1.0"
Severity = Literal["error", "warning", "info"]


@dataclass
class Issue:
    code: str
    severity: Severity
    message: str
    component: str


@dataclass
class RecognitionDetails:
    method: Literal["logprob", "proxy"] = "proxy"
    geometric_mean_prob: float | None = None
    min_token_prob: float | None = None
    uncertain_token_ratio: float | None = None
    proxy_penalties: list[str] = field(default_factory=list)


@dataclass
class LayoutDetails:
    touches_left_border: bool | None = None
    touches_right_border: bool | None = None
    touches_top_border: bool | None = None
    touches_bottom_border: bool | None = None
    estimated_dpi: float | None = None
    aspect_ratio: float | None = None
    multiple_equations_suspected: bool | None = None
    min_margin_px: float | None = None


@dataclass
class SyntaxDetails:
    balanced_braces: bool | None = None
    balanced_brackets: bool | None = None
    balanced_parens: bool | None = None
    left_right_matched: bool | None = None
    environments_closed: bool | None = None
    frac_has_two_args: bool | None = None
    scripts_have_args: bool | None = None
    commands_are_known: bool | None = None
    depth_within_limit: bool | None = None
    unknown_commands: list[str] = field(default_factory=list)


@dataclass
class OCRQualityDetails:
    triggered_checks: list[str] = field(default_factory=list)
    text_command_ratio: float | None = None
    has_math_commands: bool | None = None


@dataclass
class ComponentDetails:
    recognition: RecognitionDetails | None = None
    layout: LayoutDetails | None = None
    syntax: SyntaxDetails | None = None
    ocr_quality: OCRQualityDetails | None = None


@dataclass
class ConfidenceResult:
    schema_version: str = SCHEMA_VERSION
    overall_confidence: float = 0.0
    recognition: float | None = None
    layout: float | None = None
    syntax: float | None = None
    ocr_quality: float | None = None
    issues: list[Issue] = field(default_factory=list)
    details: ComponentDetails = field(default_factory=ComponentDetails)
    aggregation_strategy: str = "weighted_average"
    weights_used: dict[str, float] = field(default_factory=dict)
    components_available: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _issue(i: Issue) -> dict:
            return {"code": i.code, "severity": i.severity, "message": i.message, "component": i.component}

        def _rec(d: RecognitionDetails | None) -> dict | None:
            if d is None:
                return None
            return {
                "method": d.method,
                "geometric_mean_prob": d.geometric_mean_prob,
                "min_token_prob": d.min_token_prob,
                "uncertain_token_ratio": d.uncertain_token_ratio,
                "proxy_penalties": list(d.proxy_penalties),
            }

        def _lay(d: LayoutDetails | None) -> dict | None:
            if d is None:
                return None
            return {
                "touches_left_border": d.touches_left_border,
                "touches_right_border": d.touches_right_border,
                "touches_top_border": d.touches_top_border,
                "touches_bottom_border": d.touches_bottom_border,
                "estimated_dpi": d.estimated_dpi,
                "aspect_ratio": d.aspect_ratio,
                "multiple_equations": d.multiple_equations_suspected,
                "multiple_equations_suspected": d.multiple_equations_suspected,
                "min_margin_px": d.min_margin_px,
            }

        def _syn(d: SyntaxDetails | None) -> dict | None:
            if d is None:
                return None
            return {
                "balanced_braces": d.balanced_braces,
                "balanced_brackets": d.balanced_brackets,
                "balanced_parens": d.balanced_parens,
                "left_right_matched": d.left_right_matched,
                "environments_closed": d.environments_closed,
                "frac_has_two_args": d.frac_has_two_args,
                "scripts_have_args": d.scripts_have_args,
                "commands_are_known": d.commands_are_known,
                "depth_within_limit": d.depth_within_limit,
                "unknown_commands": list(d.unknown_commands),
            }

        def _ocq(d: OCRQualityDetails | None) -> dict | None:
            if d is None:
                return None
            return {
                "triggered_checks": list(d.triggered_checks),
                "text_command_ratio": d.text_command_ratio,
                "has_math_commands": d.has_math_commands,
            }

        return {
            "schema_version": self.schema_version,
            "overall_confidence": self.overall_confidence,
            "recognition": self.recognition,
            "layout": self.layout,
            "syntax": self.syntax,
            "ocr_quality": self.ocr_quality,
            "issues": [_issue(i) for i in self.issues],
            "details": {
                "recognition": _rec(self.details.recognition),
                "layout": _lay(self.details.layout),
                "syntax": _syn(self.details.syntax),
                "ocr_quality": _ocq(self.details.ocr_quality),
            },
            "aggregation_strategy": self.aggregation_strategy,
            "weights_used": dict(self.weights_used),
            "components_available": dict(self.components_available),
        }
