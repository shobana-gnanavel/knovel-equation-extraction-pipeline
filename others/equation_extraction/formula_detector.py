"""Confidence-based display-formula candidate scorer (feature 008, revision).

Every display formula has at least one of three types of evidence:

  Mathematical signals  — relational operators, Greek/math characters, ``(cid:N)``
                          font-glyph placeholders, inline-math regex patterns.
  Layout signals        — block narrower than body text, centered on page,
                          vertical proximity to an equation-number label.
  Structural signals    — fraction layout (multi-line, short lines, no finite verb),
                          explicit formula/equation region type from the layout backend.

Mathematical signals are the highest-quality evidence.  When present, layout signals
are additive bonuses.  When absent, the block must exhibit a fraction layout (two or
more short lines with no finite verb) to remain a formula candidate; single-line
word-only blocks that are very close to a label are placed in the ``needs_llm`` zone
where an LLM oracle can resolve the ambiguity.

Thresholds
----------
``FORMULA_THRESHOLD``    (0.45)  — unconditionally a formula.
``AMBIGUOUS_THRESHOLD``  (0.25)  — grey zone; an LLM oracle resolves these when
                                   available; otherwise treated as non-formula.
``TEXT_ONLY_THRESHOLD``  (0.15)  — lower bar used by the text-only backward-compat
                                   wrapper :func:`looks_like_standalone_formula_score`
                                   so that two-char math expressions like ``"2θ"``
                                   still pass the heuristic.

Usage
-----
::

    from equation_extraction.formula_detector import score_formula_candidate

    fs = score_formula_candidate(
        text,
        bbox=[x0, y0, x1, y1],          # page-point coordinates
        page_dims=(page_w, page_h),       # page dimensions in points
        label_distance_pts=abs(label_cy - block_cy),
        region_type="paragraph",
    )
    if fs.is_formula:
        # treat as display equation
    elif fs.needs_llm:
        # call LLM oracle to resolve ambiguity
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "FormulaScore",
    "score_formula_candidate",
    "FORMULA_THRESHOLD",
    "AMBIGUOUS_THRESHOLD",
    "TEXT_ONLY_THRESHOLD",
]

FORMULA_THRESHOLD: float = 0.45
AMBIGUOUS_THRESHOLD: float = 0.25
TEXT_ONLY_THRESHOLD: float = 0.15

# Finite verb forms: their presence in a no-math-signal block strongly suggests
# the text is a descriptive sentence/heading, not a display formula.
_FINITE_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|have|has|do|does|did|will|would|shall|should"
    r"|may|might|must|can|could)\b",
    re.IGNORECASE,
)

# Relational operators (= and common substitutes / Unicode variants).
# ¼ (U+00BC) is included because engineering PDFs encode the equals sign as ¼ due
# to font-encoding failures.
_RELATIONAL_OP_RE = re.compile(r"[=¼≈≠≤≥]")


@dataclass
class FormulaScore:
    """Confidence evaluation result for a single formula candidate block."""

    score: float
    """Combined confidence value in [0, 1]."""

    signals: dict[str, float] = field(default_factory=dict)
    """Individual signal name → contribution (positive or negative)."""

    is_formula: bool = False
    """True when *score* ≥ :data:`FORMULA_THRESHOLD`."""

    needs_llm: bool = False
    """True when *score* is in [AMBIGUOUS_THRESHOLD, FORMULA_THRESHOLD) — an LLM
    oracle may promote this block to a formula."""


# First line of a two-line fragment is a year or section number (pure digits/dots),
# e.g. "1992.\n\nNiu," or "3.5.2.\n\nForward".  These are reference-list entries or
# section headings, not fractions.
_DIGIT_DOT_ONLY = re.compile(r"^\d[\d\.]*\.?$")


def _fraction_layout_valid(stripped: str) -> bool:
    """True when *stripped* looks like a stacked fraction formula.

    Requirements:
    * 2–4 non-empty lines (numerator/denominator or similar short stacking)
    * Every line is ≤ 80 characters (rules out long prose paragraphs split by newlines)
    * No finite verb — prevents headers like "Mean piston speed is simply" from
      matching when they happen to be near an equation label.
    * First line must not be a pure digit/dot sequence — rules out section numbers
      like "3.5.2." and 4-digit years like "1992." that precede author names.
    """
    lines = [ln.strip() for ln in stripped.split("\n") if ln.strip()]
    if not (2 <= len(lines) <= 4):
        return False
    if any(len(ln) > 80 for ln in lines):
        return False
    if _FINITE_VERB_RE.search(stripped):
        return False
    if _DIGIT_DOT_ONLY.match(lines[0]):
        return False
    return True


def score_formula_candidate(
    text: str,
    *,
    bbox: list[float] | None = None,
    page_dims: tuple[float, float] | None = None,
    label_distance_pts: float | None = None,
    region_type: str = "",
) -> FormulaScore:
    """Compute a confidence score for whether *text* is a display formula.

    Parameters
    ----------
    text:
        Candidate block plain text (may contain ``\\n`` for multi-line blocks).
    bbox:
        ``[x0, y0, x1, y1]`` in page points.  Enables layout signals.
    page_dims:
        ``(width, height)`` of the containing page in points.
    label_distance_pts:
        Absolute vertical distance between the block centre and the nearest
        equation-label centre, in points.  ``None`` when no label is nearby.
    region_type:
        Layout-backend region type string (e.g. ``"paragraph"``, ``"formula"``).
        When the value is ``"equation"`` or ``"formula"`` the score is set to 1.0
        immediately, bypassing all other signal computation.
    """
    stripped = text.strip()

    # ── Fast path: explicit layout-backend label ─────────────────────────────────
    # Checked before the length guard so that a layout-tagged "formula" region always
    # wins even when its text is a single character or very short.
    if region_type in {"equation", "formula"}:
        return FormulaScore(
            score=1.0,
            signals={"region_type_formula": 1.0},
            is_formula=True,
        )

    if not stripped or len(stripped) == 1 or len(stripped) > 250:
        return FormulaScore(score=0.0)

    # ── Import shared patterns lazily to avoid circular imports ─────────────────
    # (formula_detector is imported by detection.py; detection.py is also imported
    # here — so we defer the import to inside this function, not at module level.)
    from equation_extraction.detection import (  # noqa: PLC0415
        _CHEM_STOICH_SEQUENCE,
        _CHEMICAL_REACTION_ARROW,
        _CID_GLYPH_RE,
        _INLINE_MATH,
        _MATH_SIGNAL_CHARS,
        _PROSE_WORDS,
    )

    signals: dict[str, float] = {}

    # ── Chemical equation signals (checked before math signals) ──────────────────
    # Chemical reaction arrows are an unambiguous display-equation signal: they
    # appear in chemical reactions and nowhere else in ordinary text.  A single
    # confirmed arrow is worth as much as a relational operator in a math formula.
    has_chem_arrow = bool(_CHEMICAL_REACTION_ARROW.search(stripped))
    if has_chem_arrow:
        signals["chemical_reaction_arrow"] = 0.40

    # Stoichiometric coefficient sequences (e.g. "3CO2 + 2.5H2O + 1.5N2") indicate
    # a chemical product or reactant list even when no explicit reaction arrow is
    # present.  Two or more groups is a strong signal; three or more is decisive.
    stoich_groups = len(_CHEM_STOICH_SEQUENCE.findall(stripped))
    if stoich_groups >= 3:
        signals["chemical_stoich_sequence"] = 0.25
    elif stoich_groups >= 2:
        signals["chemical_stoich_sequence"] = 0.15

    # ── Mathematical text signals ─────────────────────────────────────────────────
    has_relational = bool(_RELATIONAL_OP_RE.search(stripped))
    if has_relational:
        signals["relational_op"] = 0.40

    math_char_count = sum(1 for c in stripped if c in _MATH_SIGNAL_CHARS)
    if math_char_count >= 6:
        signals["math_signal_char"] = 0.30
    elif math_char_count >= 3:
        signals["math_signal_char"] = 0.20
    elif math_char_count >= 1:
        signals["math_signal_char"] = 0.10

    cid_count = len(_CID_GLYPH_RE.findall(stripped))
    if cid_count >= 3:
        signals["cid_glyph"] = 0.25
    elif cid_count >= 1:
        signals["cid_glyph"] = 0.12

    if _INLINE_MATH.search(stripped):
        signals["inline_math_pattern"] = 0.20

    has_any_math = bool(signals)

    # ── Structural signals ────────────────────────────────────────────────────────
    if _fraction_layout_valid(stripped):
        signals["fraction_layout"] = 0.28

    if len(stripped) < 80:
        signals["short_text"] = 0.08

    # ── Label proximity signal ────────────────────────────────────────────────────
    if label_distance_pts is not None:
        if label_distance_pts < 30:
            signals["label_proximity_close"] = 0.30
        elif label_distance_pts < 80:
            signals["label_proximity_near"] = 0.15

    # ── Layout geometry signals ───────────────────────────────────────────────────
    if bbox and page_dims and len(bbox) == 4:
        page_w, _ = page_dims
        x0, _, x1, _ = bbox[0], bbox[1], bbox[2], bbox[3]
        block_w = x1 - x0
        block_cx = (x0 + x1) / 2.0
        page_cx = page_w / 2.0
        if page_w > 0:
            width_ratio = block_w / page_w
            if width_ratio < 0.55:
                signals["narrow_block"] = 0.15
            elif width_ratio > 0.88:
                signals["full_width_block"] = -0.20
            if abs(block_cx - page_cx) / page_w < 0.20:
                signals["centered_block"] = 0.10

    # ── Prose penalties ──────────────────────────────────────────────────────────
    prose_hits = len(_PROSE_WORDS.findall(stripped))
    if prose_hits >= 5:
        signals["heavy_prose"] = -0.50
    elif prose_hits >= 3 and not has_any_math:
        signals["moderate_prose"] = -0.25

    # ── No-math-signal guard ─────────────────────────────────────────────────────
    # Without any mathematical character evidence, only fraction-layout blocks are
    # valid formula candidates from purely textual content.  Single-line word blocks
    # may still be sent to an LLM when they are within 30 pts of an equation label.
    if not has_any_math:
        has_fraction = bool(signals.get("fraction_layout"))
        if not has_fraction:
            is_ambiguous = (
                label_distance_pts is not None
                and label_distance_pts < 30
                and sum(v for v in signals.values() if v > 0) >= AMBIGUOUS_THRESHOLD
            )
            return FormulaScore(
                score=max(0.0, sum(signals.values())),
                signals=signals,
                is_formula=False,
                needs_llm=is_ambiguous,
            )

    raw_score = sum(signals.values())
    score = max(0.0, min(1.0, raw_score))
    is_formula = score >= FORMULA_THRESHOLD
    needs_llm = not is_formula and score >= AMBIGUOUS_THRESHOLD

    return FormulaScore(score=score, signals=signals, is_formula=is_formula, needs_llm=needs_llm)
