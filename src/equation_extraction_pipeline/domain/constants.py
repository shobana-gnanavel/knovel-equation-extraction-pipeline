"""Shared regex primitives and named constants for the equation-extraction pipeline.

Single source of truth for character classes, equation label patterns, and
related named constants used by classifier, formula_detector, and recognition
quality scoring.  Keeping them here prevents silent drift across modules.
"""

from __future__ import annotations

import re

__all__ = [
    "GREEK_LOWER",
    "GREEK_UPPER",
    "GREEK_LETTERS",
    "EQUATION_LABEL_PATTERN",
    "INLINE_MATH_PATTERN",
    "LATEX_COMMAND_PATTERN",
    "MATH_OPERATOR_PATTERN",
    "SUPERSCRIPT_SUBSCRIPT_PATTERN",
    "GREEK_UNICODE_PATTERN",
    "FRACTION_PATTERN",
    "INTEGRAL_PATTERN",
    "SUMMATION_PATTERN",
    "MATRIX_PATTERN",
    "CHEMICAL_FORMULA_PATTERN",
    "EQUATION_NUMBER_PATTERN",
]

# ---------------------------------------------------------------------------
# Greek alphabet character sets
# ---------------------------------------------------------------------------

GREEK_LOWER = "αβγδεζηθικλμνξοπρστυφχψω"
GREEK_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
GREEK_LETTERS = GREEK_LOWER + GREEK_UPPER

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Equation label pattern — matches margin labels like '(12.2.1)' or '12.2.1'.
EQUATION_LABEL_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*\(?(\d+(?:\.\d+){1,4})\)?\s*$"
)

# Inline math delimiters: $…$ or \(…\).
INLINE_MATH_PATTERN: re.Pattern[str] = re.compile(
    r"\$[^$\n]+\$|\\\([^)]+\\\)"
)

# LaTeX command: \commandName (optionally followed by braces).
LATEX_COMMAND_PATTERN: re.Pattern[str] = re.compile(
    r"\\[a-zA-Z]+"
)

# Common binary math operators and comparison symbols.
MATH_OPERATOR_PATTERN: re.Pattern[str] = re.compile(
    r"[+\-*/=<>≤≥≠±×÷∝∞∂∇]"
)

# Superscript/subscript indicators (caret and underscore in LaTeX).
SUPERSCRIPT_SUBSCRIPT_PATTERN: re.Pattern[str] = re.compile(
    r"[_^]\{?[^{}]+\}?"
)

# Greek letters in their Unicode form (used for quick presence checks).
GREEK_UNICODE_PATTERN: re.Pattern[str] = re.compile(
    rf"[{re.escape(GREEK_LETTERS)}]"
)

# LaTeX fraction command.
FRACTION_PATTERN: re.Pattern[str] = re.compile(
    r"\\frac\{[^}]*\}\{[^}]*\}"
)

# LaTeX integral variants: \int, \iint, \iiint, \oint.
INTEGRAL_PATTERN: re.Pattern[str] = re.compile(
    r"\\i{1,3}int|\\oint"
)

# LaTeX summation / product: \sum, \prod.
SUMMATION_PATTERN: re.Pattern[str] = re.compile(
    r"\\(?:sum|prod)"
)

# LaTeX matrix environments.
MATRIX_PATTERN: re.Pattern[str] = re.compile(
    r"\\begin\{(?:matrix|pmatrix|bmatrix|vmatrix|Vmatrix|cases)\}"
)

# Simple molecular/chemical formula: e.g. H2O, CO2, C6H12O6.
CHEMICAL_FORMULA_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:[A-Z][a-z]?\d*){2,}\b"
)

# Equation number in parentheses at end of line: (1), (1.2), (A.3).
EQUATION_NUMBER_PATTERN: re.Pattern[str] = re.compile(
    r"\(\s*[A-Za-z]?\d+(?:\.\d+)*\s*\)\s*$"
)
