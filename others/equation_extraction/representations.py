"""Representation assembly and validity checks (feature 008, FR-013/FR-021).

Applies the category-appropriate representation policy to a provider's :class:`RecognitionResult`:
plain text always; LaTeX (preferred) plus optional MathML for math/engineering/statistical; a
structured form (SMILES/MOL) for chemical structures. Runs a lightweight LaTeX-validity probe
(balanced braces/environments + a ``latex2mathml`` parse) and a MathML well-formedness check
(stdlib ``xml.etree``), flagging ``invalid_latex`` / ``invalid_mathml`` without dropping the equation.
An absent representation is recorded as ``None`` — never fabricated. Pure functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree

from equation_extraction.providers import RecognitionResult

__all__ = ["Representations", "assemble", "is_valid_latex", "is_valid_mathml"]

try:  # core dependency, present in the base install
    import latex2mathml.converter as _latex2mathml
except Exception:  # pragma: no cover - optional-dep fallback
    _latex2mathml = None  # type: ignore[assignment]

_MATH_CATEGORIES = {"mathematical_equation", "engineering_formula", "statistical_expression"}
_CHEM_STRUCTURE = "chemical_structure"
_CHEM_EQUATION = "chemical_equation"

_ENV_BEGIN = re.compile(r"\\begin\{")
_ENV_END = re.compile(r"\\end\{")


@dataclass
class Representations:
    """The assembled, validity-checked representations for one equation (FR-013)."""

    plain_text: str = ""
    latex: str | None = None
    mathml: str | None = None
    structured_form: str | None = None


def is_valid_latex(latex: str) -> bool:
    """Lightweight LaTeX-validity probe: balanced braces/environments + a parse attempt (FR-021)."""
    if not latex or not latex.strip():
        return False
    if latex.count("{") != latex.count("}"):
        return False
    if len(_ENV_BEGIN.findall(latex)) != len(_ENV_END.findall(latex)):
        return False
    if _latex2mathml is None:  # pragma: no cover - parse probe unavailable
        return True
    try:
        _latex2mathml.convert(latex)
        return True
    except Exception:
        return False


def is_valid_mathml(mathml: str) -> bool:
    """MathML well-formedness via a stdlib XML parse (FR-021)."""
    if not mathml or not mathml.strip():
        return False
    try:
        ElementTree.fromstring(mathml)
        return True
    except Exception:
        return False


def _to_mathml(latex: str) -> str | None:
    if _latex2mathml is None:  # pragma: no cover
        return None
    try:
        return str(_latex2mathml.convert(latex))
    except Exception:
        return None


def assemble(
    result: RecognitionResult, *, category: str, config: Any
) -> tuple[Representations, list[str]]:
    """Assemble category-appropriate representations and return ``(representations, flags)``."""
    flags: list[str] = []
    latex_enabled = bool(getattr(config, "KNOVEL_EQUATION_LATEX_ENABLED", True))
    mathml_enabled = bool(getattr(config, "KNOVEL_EQUATION_MATHML_ENABLED", False))
    structured_enabled = bool(getattr(config, "KNOVEL_EQUATION_STRUCTURED_ENABLED", True))

    rep = Representations(plain_text=result.plain_text or "")

    if category in _MATH_CATEGORIES or category == _CHEM_EQUATION:
        if latex_enabled and result.latex:
            if is_valid_latex(result.latex):
                rep.latex = result.latex
            else:
                rep.latex = result.latex  # retained, but flagged (never silently dropped)
                flags.append("invalid_latex")
        if mathml_enabled:
            mathml = result.mathml or (_to_mathml(rep.latex) if rep.latex else None)
            if mathml:
                if is_valid_mathml(mathml):
                    rep.mathml = mathml
                else:
                    rep.mathml = mathml
                    flags.append("invalid_mathml")

    if category in {_CHEM_STRUCTURE, _CHEM_EQUATION} and structured_enabled:
        rep.structured_form = result.structured_form or None

    return rep, flags
