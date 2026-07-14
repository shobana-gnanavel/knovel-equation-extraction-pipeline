"""Rule-based equation classification (feature 008, FR-006..FR-009).

A deterministic, multi-signal classifier maps a detected equation region to one of the six content
categories with a confidence, a human-readable reason, and a recommended provider. Signals: the
region's text and symbol set (mathematical vs chemical glyphs, reaction arrows/stoichiometry,
statistical operators), unit/physical-quantity cues (the ``engineering_formula`` vs
``mathematical_equation`` discriminator), and the surrounding section/chapter context. Unresolved or
sub-threshold regions default to ``unknown`` and are flagged by the caller (FR-009). Pure and
deterministic for a given input + config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from equation_extraction.patterns import GREEK_LETTERS
from pipeline.models import EQUATION_CATEGORIES

__all__ = ["Classification", "classify_region", "DEFAULT_PROVIDER_BY_CATEGORY"]

# Default category → provider-role map (selection.py applies configured overrides on top, FR-010).
DEFAULT_PROVIDER_BY_CATEGORY: dict[str, str] = {
    "mathematical_equation": "qwen_vl",
    "engineering_formula": "qwen_vl",
    "statistical_expression": "qwen_vl",
    "chemical_equation": "qwen_vl",
    "chemical_structure": "qwen_vl",
    "unknown": "generic",
}

_MATH_OPS = re.compile(rf"[{GREEK_LETTERS}∑∫∂√∞±×÷≤≥≠≈∇∆]|[=^_]|\b(sin|cos|tan|log|ln|exp|lim)\b")
_REACTION = re.compile(
    r"[→⇌⇒⇄↔⟶]"  # Unicode reaction arrows
    r"|-+>|<-+|=+>"  # ASCII arrow variants
    r"|--\+"  # OCR corruption: bold '→' read as '--+' in scanned books
)
_MOLECULE = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")  # molecular formula, e.g. H2O / NaCl / CO2
# Stoichiometric coefficient before an element symbol, allowing whitespace and common OCR
# substitutions (subscript digits replaced by letters: 2→z, 3→s/a).
_STOICH = re.compile(r"(?:^|[\s+])\d+\.?\d*\s*[A-Z][a-z]?")
_STRUCTURE_HINT = re.compile(r"\b(benzene|ring|bond|aromatic|cyclo|structure)\b", re.IGNORECASE)
_STAT = re.compile(
    r"\b(Var|Cov|Pr|SD|std|mean|median|variance|distribution|regression|correlation)\b"
    r"|[μσ]"
    r"|\bP\s*\("  # probability P(...)
    r"|\bE\s*\["  # expectation E[...]
    r"|~\s*N\(|\bN\(0"
)
# Unit / physical-quantity cues distinguishing engineering formulae from pure-symbolic math.
_UNIT = re.compile(
    r"\b(\d+(\.\d+)?\s*)?(m|km|cm|mm|kg|g|mg|s|ms|N|Pa|kPa|MPa|J|kJ|W|kW|V|A|Ω|Hz|"
    r"mol|K|°C|°F|rad|bar|psi|lb|ft|in)\b"
)
_ENGINEERING_CONTEXT = re.compile(
    r"\b(engineering|mechanical|thermodynamic|fluid|stress|strain|circuit|"
    r"structural|hydraulic|kinematic|dynamics|material)\b",
    re.IGNORECASE,
)


@dataclass
class Classification:
    """The classifier's verdict for one region (FR-008)."""

    category: str
    confidence: float
    reason: str
    recommended_provider: str


def _result(category: str, confidence: float, reason: str) -> Classification:
    provider = DEFAULT_PROVIDER_BY_CATEGORY.get(category, "generic")
    return Classification(
        category=category,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
        reason=reason,
        recommended_provider=provider,
    )


def classify_region(
    *,
    region_type: str,
    region_text: str,
    section_context: str = "",
    is_inline: bool = False,
) -> Classification:
    """Classify one equation region into a category with confidence, reason, and provider (FR-006/07/08)."""
    text = region_text or ""
    ctx = section_context or ""

    has_reaction = bool(_REACTION.search(text))
    has_molecule = bool(_MOLECULE.search(text))
    has_stoich = bool(_STOICH.search(text))
    has_structure_hint = bool(_STRUCTURE_HINT.search(text))
    has_math = bool(_MATH_OPS.search(text))
    has_stat = bool(_STAT.search(text))
    has_unit = bool(_UNIT.search(text))
    eng_context = bool(_ENGINEERING_CONTEXT.search(ctx))

    # Count stoichiometric groups for multi-group signal (coefficient + element, e.g. "3CO2").
    stoich_count = len(_STOICH.findall(text))

    # Chemical signals dominate when reactions/molecules/structures are present (FR-006).
    # Three upgrade paths to "chemical_equation":
    #   1. Explicit reaction arrow (→ ⇌ -> --+ etc.) — strongest signal
    #   2. Molecular formula + stoichiometric coefficient + no relational math
    #   3. Two or more stoichiometric groups without relational math (e.g. "3CO2 + 2.5H2O")
    if (
        has_reaction
        or (has_molecule and has_stoich and not has_math)
        or (stoich_count >= 2 and not has_math)
    ):
        return _result(
            "chemical_equation",
            0.85,
            "chemical reaction signals (arrow/stoichiometry/molecular formula)",
        )
    if has_structure_hint or (has_molecule and not has_math and not has_stat):
        return _result(
            "chemical_structure",
            0.75,
            "chemical structure signals (molecular formula / ring/bond cues, no relational math)",
        )

    # Statistical expressions.
    if has_stat:
        return _result(
            "statistical_expression",
            0.7,
            "statistical operators/notation (expectation/variance/distribution)",
        )

    # Engineering vs pure mathematics: unit / physical-quantity or engineering section context.
    if has_unit or eng_context:
        return _result(
            "engineering_formula",
            0.7,
            "physical-quantity/unit cues or engineering section context",
        )

    # Default math category: a layout-typed equation region (or an inline math run).
    if has_math:
        return _result("mathematical_equation", 0.8, "mathematical operators/symbols present")
    if region_type == "equation":
        return _result(
            "mathematical_equation",
            0.6,
            "layout-typed display equation region (no chemical/statistical/unit signal)",
        )
    if is_inline:
        return _result("mathematical_equation", 0.55, "inline math run detected in text")

    return _result("unknown", 0.3, "no decisive equation signal")


# Startup check: DEFAULT_PROVIDER_BY_CATEGORY must cover every known category.
_missing_categories = set(EQUATION_CATEGORIES) - set(DEFAULT_PROVIDER_BY_CATEGORY)
if _missing_categories:
    raise RuntimeError(
        f"classifier: DEFAULT_PROVIDER_BY_CATEGORY is missing entries for: {sorted(_missing_categories)}. "
        "Update DEFAULT_PROVIDER_BY_CATEGORY to match EQUATION_CATEGORIES."
    )
