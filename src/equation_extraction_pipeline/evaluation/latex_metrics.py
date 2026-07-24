"""LaTeX fidelity metrics — deterministic, stdlib-only.

Turns a (predicted, gold) LaTeX pair into comparable scores so extraction accuracy is
measurable and every future change is A/B-testable.

Metrics
-------
* ``exact_match``       — 1.0 if the two strings are identical after normalization, else 0.0.
* ``char_similarity``   — character-level similarity in [0,1] (difflib ratio on normalized text).
* ``token_similarity``  — LaTeX-token-level similarity in [0,1] (robust to spacing/brace noise).

``token_similarity`` is the recommended headline number: it compares LaTeX *structure*
(``\\frac``, ``{``, ``x`` …) rather than raw characters, so cosmetic differences do not
unfairly penalize a correct transcription.

Note: CDM (Character Detection Matching), the SOTA render-based metric, is a future upgrade
(it requires rendering both sides to images); these string metrics are the immediately
runnable, offline baseline.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

__all__ = [
    "normalize_latex",
    "tokenize",
    "exact_match",
    "char_similarity",
    "token_similarity",
    "score_pair",
]

# Math-mode delimiters and display wrappers the model sometimes leaves in.
_DELIMITERS = [
    (re.compile(r"^\s*\$\$"), ""),
    (re.compile(r"\$\$\s*$"), ""),
    (re.compile(r"^\s*\\\["), ""),
    (re.compile(r"\\\]\s*$"), ""),
    (re.compile(r"^\s*\\\("), ""),
    (re.compile(r"\\\)\s*$"), ""),
    (re.compile(r"^\s*\$"), ""),
    (re.compile(r"\$\s*$"), ""),
]

# Whitespace/spacing commands that carry no semantic content.
_SPACING = re.compile(r"\\[,;:!>]|\\quad|\\qquad|\\ |~")

# Fraction command variants normalised to a canonical form.
_FRAC_VARIANTS = re.compile(r"\\(?:dfrac|tfrac|cfrac)\b")

# A LaTeX token: a control sequence (\alpha, \frac), a single escaped char (\{),
# a brace, or any other single non-space character.
_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|\\.|[{}]|[^\s]")


def normalize_latex(s: str) -> str:
    """Normalize a LaTeX string for comparison.

    Strips math delimiters/display wrappers, canonicalizes fraction variants, removes
    spacing-only commands, drops all whitespace, and unifies ``\\left``/``\\right`` (which
    do not change meaning) so equivalent transcriptions compare equal.
    """
    if not s:
        return ""
    text = s.strip()
    # Equation-number tags are metadata, not math: hybrid crops include the printed margin
    # label, which the VLM renders as \tag{2-4} (or a trailing bare "(2-4)"); gold has neither.
    text = re.sub(r"\\tag\*?\{[^{}]*\}", "", text)
    text = re.sub(r"\(\s*\d{1,3}[.\-]\d{1,3}(?:\.\d{1,3})?\s*\)\s*$", "", text)
    for pat, repl in _DELIMITERS:
        text = pat.sub(repl, text)
    text = _FRAC_VARIANTS.sub(r"\\frac", text)
    text = re.sub(r"\\left\b|\\right\b", "", text)
    text = _SPACING.sub("", text)
    text = re.sub(r"\s+", "", text)
    # Redundant single-character script braces: _{e} ≡ _e, ^{2} ≡ ^2 (true LaTeX
    # equivalence — different VLM prompts brace-quote inconsistently).
    text = re.sub(r"([_^])\{([A-Za-z0-9])\}", r"\1\2", text)
    return text


def tokenize(s: str) -> list[str]:
    """Split *normalized* LaTeX into structural tokens."""
    return _TOKEN_RE.findall(normalize_latex(s))


def exact_match(pred: str, gold: str) -> float:
    """1.0 if ``pred`` equals ``gold`` after normalization, else 0.0."""
    return 1.0 if normalize_latex(pred) == normalize_latex(gold) else 0.0


def char_similarity(pred: str, gold: str) -> float:
    """Character-level similarity in [0,1] on normalized strings."""
    a, b = normalize_latex(pred), normalize_latex(gold)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def token_similarity(pred: str, gold: str) -> float:
    """LaTeX-token-level similarity in [0,1] — the recommended headline metric."""
    a, b = tokenize(pred), tokenize(gold)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def score_pair(pred: str, gold: str) -> dict[str, float]:
    """Return all fidelity metrics for one (pred, gold) pair."""
    return {
        "exact_match": exact_match(pred, gold),
        "char_similarity": round(char_similarity(pred, gold), 4),
        "token_similarity": round(token_similarity(pred, gold), 4),
    }
