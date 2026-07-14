"""Recognition confidence estimator — uses token log-probs when available, proxy metrics otherwise."""

from __future__ import annotations

import math
import re

from .config import ConfidenceConfig
from .models import Issue, RecognitionDetails

_RELATIONAL_PATTERNS = re.compile(
    r"[=<>]|\\leq|\\geq|\\neq|\\approx|\\sim|\\simeq|\\cong|\\equiv|\\propto|\\ll|\\gg"
    r"|\\subset|\\subseteq|\\supset|\\supseteq|\\in|\\notin"
)
_TEXT_BLOCK_RE = re.compile(r"\\text\{[^}]*\}")
_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|\S")


def estimate_recognition(
    *,
    latex: str,
    token_logprobs: list[float] | None,
    config: ConfidenceConfig,
) -> tuple[float, RecognitionDetails, list[Issue]]:
    issues: list[Issue] = []
    if token_logprobs:
        return _from_logprobs(latex, token_logprobs, config, issues)
    return _from_proxies(latex, config, issues)


def _from_logprobs(
    latex: str,
    logprobs: list[float],
    config: ConfidenceConfig,
    issues: list[Issue],
) -> tuple[float, RecognitionDetails, list[Issue]]:
    probs = [math.exp(lp) for lp in logprobs]
    n = len(probs)
    geo_mean = math.exp(sum(logprobs) / n) if n else 0.0
    min_prob = min(probs) if probs else 0.0
    uncertain_ratio = sum(1 for p in probs if p < config.uncertain_token_threshold) / n if n else 0.0

    normalizer = config.min_token_prob_normalizer
    score = (
        0.50 * geo_mean
        + 0.30 * (1.0 - uncertain_ratio)
        + 0.20 * min(min_prob / normalizer, 1.0)
    )
    details = RecognitionDetails(
        method="logprob",
        geometric_mean_prob=round(geo_mean, 4),
        min_token_prob=round(min_prob, 4),
        uncertain_token_ratio=round(uncertain_ratio, 4),
    )
    return max(0.0, min(1.0, score)), details, issues


def _from_proxies(
    latex: str,
    config: ConfidenceConfig,
    issues: list[Issue],
) -> tuple[float, RecognitionDetails, list[Issue]]:
    issues.append(Issue(
        code="USING_PROXY_RECOGNITION",
        severity="info",
        message="Token log-probabilities unavailable; using deterministic proxy metrics.",
        component="recognition",
    ))

    applied: list[str] = []

    if not latex or not latex.strip():
        details = RecognitionDetails(method="proxy", proxy_penalties=["empty_latex"])
        return 0.10, details, issues

    score = 0.85

    # High \text{} ratio
    text_blocks = _TEXT_BLOCK_RE.findall(latex)
    all_tokens = _TOKEN_RE.findall(latex)
    if all_tokens:
        ratio = len(text_blocks) / len(all_tokens)
        if ratio > config.high_text_ratio_threshold:
            score -= 0.20
            applied.append("high_text_ratio")

    # No LaTeX commands at all — plain text passthrough
    if not re.search(r"\\[a-zA-Z]", latex):
        score -= 0.25
        applied.append("no_backslash_commands")

    # No relational operator in a non-trivial output
    if len(latex.strip()) > 4 and not _RELATIONAL_PATTERNS.search(latex):
        score -= 0.15
        applied.append("no_relational_operator")

    # Single-character output
    if re.match(r"^\s*.\s*$", latex):
        score -= 0.30
        applied.append("single_char_output")

    details = RecognitionDetails(method="proxy", proxy_penalties=applied)
    return max(0.0, min(1.0, score)), details, issues
