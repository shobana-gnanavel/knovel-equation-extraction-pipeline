"""Shared low-level regex primitives for the equation-extraction stage (feature 008).

Single source of truth for character classes that were previously copy-pasted across
:mod:`classifier`, :mod:`detection`, and :mod:`formula_detector`. Keeping them here prevents
silent drift — a fix to one Greek/operator set now applies everywhere it is used.

Scope note: this module holds only primitives that are *verifiably identical* everywhere they
appear (the Greek alphabet, at present). The higher-level reaction-arrow regexes deliberately
still diverge between modules (e.g. the classifier accepts ``<-+`` / ``=+>`` variants that the
chemical-arrow detector does not). Unifying those would change classification/detection outcomes
and must be gated on the equation eval harness rather than done as a blind refactor.
"""

from __future__ import annotations

__all__ = ["GREEK_LOWER", "GREEK_UPPER", "GREEK_LETTERS"]

# Greek alphabet, lower- and upper-case. Used inside regex character classes (e.g.
# ``rf"[{GREEK_LETTERS}]"``) and as a membership set for math-signal-character counting.
GREEK_LOWER = "αβγδεζηθικλμνξοπρστυφχψω"
GREEK_UPPER = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
GREEK_LETTERS = GREEK_LOWER + GREEK_UPPER
