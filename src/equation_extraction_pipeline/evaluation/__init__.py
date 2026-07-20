"""Evaluation utilities: LaTeX fidelity metrics and accuracy scoring.

Used by ``scripts/eval_latex_accuracy.py`` to turn "is the extracted LaTeX correct?"
into numbers, and to validate the external GPT judge against a human gold set.
"""

from .latex_metrics import (
    char_similarity,
    exact_match,
    normalize_latex,
    score_pair,
    token_similarity,
    tokenize,
)

__all__ = [
    "normalize_latex",
    "tokenize",
    "exact_match",
    "char_similarity",
    "token_similarity",
    "score_pair",
]
