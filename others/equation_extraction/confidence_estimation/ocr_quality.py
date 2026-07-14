"""OCR quality confidence estimator — semantic plausibility checks on LaTeX output."""

from __future__ import annotations

import re

from .config import ConfidenceConfig
from .models import Issue, OCRQualityDetails

_MATH_CMD_RE    = re.compile(r"\\(?!text)[a-zA-Z]+")
_TEXT_BLOCK_RE  = re.compile(r"\\text\{[^}]*\}")
_TOKEN_RE       = re.compile(r"\\[a-zA-Z]+|\S")
_REPEATED_OP_RE = re.compile(r"(?<![\\a-zA-Z])[+\-*/]{2,}")
_REPEATED_CMD_RE = re.compile(r"(\\[a-zA-Z]+)(\s*\1){2,}")
_UNICODE_MATH_RE = re.compile(r"[×÷∑∏∫√≤≥≠≈∞∂∇]")
_DANGLING_OP_RE  = re.compile(r"(?:\\times|\\cdot|\\div|\\pm|[+\-*/])\s*$")
_STARTS_EQ_RE   = re.compile(r"^\s*=")
_INCOMPLETE_RE  = re.compile(r"(\\cdots|\\ldots)\s*$")
_ISOLATED_NUM_RE = re.compile(r"^\s*[\d.,]+\s*$")

_SEVERITY_WEIGHT: dict[str, float] = {"error": 0.40, "warning": 0.15, "info": 0.05}


def estimate_ocr_quality(
    *,
    latex: str,
    config: ConfidenceConfig,
) -> tuple[float, OCRQualityDetails, list[Issue]]:
    issues: list[Issue] = []
    details = OCRQualityDetails()

    if not latex or not latex.strip():
        issues.append(Issue("EMPTY_OUTPUT", "error", "LaTeX output is empty.", "ocr_quality"))
        details.triggered_checks = ["EMPTY_OUTPUT"]
        return 0.0, details, issues

    text_blocks = _TEXT_BLOCK_RE.findall(latex)
    all_tokens  = _TOKEN_RE.findall(latex)
    math_cmds   = _MATH_CMD_RE.findall(latex)
    text_ratio  = len(text_blocks) / len(all_tokens) if all_tokens else 0.0

    details.text_command_ratio = round(text_ratio, 3)
    details.has_math_commands  = len(math_cmds) > 0

    triggered: list[str] = []
    total_weight = 0.0

    def _fire(code: str, severity: str, message: str) -> None:
        triggered.append(code)
        total_weight_ref.append(_SEVERITY_WEIGHT[severity])
        issues.append(Issue(code, severity, message, "ocr_quality"))  # type: ignore[arg-type]

    total_weight_ref: list[float] = []

    if bool(text_blocks) and not math_cmds:
        _fire("ALL_TEXT_NO_MATH", "error",
              "Output consists only of \\text{} blocks with no math commands.")

    if _STARTS_EQ_RE.search(latex):
        _fire("STARTS_WITH_EQUALS", "warning",
              "Equation begins with '=' — left-hand side may be clipped.")

    if _REPEATED_OP_RE.search(latex):
        _fire("REPEATED_OPERATOR", "warning",
              "Repeated operator sequence detected (e.g. ++, --).")

    if _DANGLING_OP_RE.search(latex):
        _fire("DANGLING_OPERATOR", "warning",
              "Expression ends with an operator; may be incomplete.")

    if _REPEATED_CMD_RE.search(latex):
        _fire("REPEATED_SYMBOL", "warning",
              "Same LaTeX command repeated 3+ times consecutively.")

    if _UNICODE_MATH_RE.search(latex):
        _fire("UNICODE_LEAKAGE", "info",
              "Unicode math symbols found in LaTeX output; mixed encoding.")

    if _ISOLATED_NUM_RE.match(latex):
        _fire("ISOLATED_NUMBER", "info",
              "Output is a lone number with no variables or operators.")

    if _INCOMPLETE_RE.search(latex):
        _fire("INCOMPLETE_EXPRESSION", "warning",
              "Expression ends with \\cdots or \\ldots; may be truncated.")

    if text_ratio > config.high_text_ratio_threshold:
        _fire("HIGH_TEXT_RATIO", "warning",
              f"More than {int(config.high_text_ratio_threshold*100)}% of output is \\text{{}} content.")

    if not math_cmds and not re.search(r"[=<>+\-*/^_]", latex):
        _fire("NO_MATH_COMMANDS", "warning",
              "No LaTeX math commands or operators found; output may be plain text.")

    details.triggered_checks = triggered
    total_weight = sum(total_weight_ref)
    score = max(0.0, 1.0 - total_weight)
    return score, details, issues
