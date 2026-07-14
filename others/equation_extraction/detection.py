"""Equation-region detection (feature 008, FR-003/FR-005).

Display equations are the equation-typed ``LayoutRegion``s produced by feature 005 — this module
selects them; it does not detect or re-type page regions. Inline equations are detected *within*
feature-007 ``TextBlock`` text using a math-signal heuristic, with the spec's inline bbox derivation
policy (span character boxes when available, else the containing block's bbox with a recorded note).
Pure functions; deterministic for a given input + config.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from equation_extraction.patterns import GREEK_LETTERS
from pipeline.models import LayoutRegion, TextBlock

__all__ = [
    "EQUATION_REGION_TYPES",
    "is_equation_region",
    "region_text",
    "detect_inline_spans",
    "derive_inline_bbox",
    "is_isolated_equation_label",
    "looks_like_standalone_formula",
    "extract_label_number",
    "extract_mixed_label_block",
    "InlineMatch",
    "_TRIVIAL_INLINE_FRAGMENT",
    "_CID_GLYPH_RE",
    "_CHEMICAL_REACTION_ARROW",
    "_CHEM_STOICH_SEQUENCE",
]

# Layout region types that are display equations (feature 005 taxonomy).
# "formula" is included as Docling sometimes uses that label instead of "equation".
EQUATION_REGION_TYPES: frozenset[str] = frozenset({"equation", "formula"})

# A fragment produced by detect_inline_spans that matches this pattern is NOT equation-worthy
# even though the surrounding text triggered the inline-math heuristic.
#
# Rejects — in order of the alternation:
#   1. A Greek letter with an optional short alphanumeric/Greek suffix: variable references
#      such as σ1, σND, τf, μm, εhkl, δε, σφφ.  The suffix length cap ({0,6}) keeps the
#      pattern from swallowing compound expressions.
#   2. A single ASCII letter with an optional short alphanum suffix: n, x, qu (alone), …
#   3. A run of bare operator symbols with no operands: ×, ±, ≤, =+∞, …
#
# Keeps (not matched, so kept as inline equations):
#   • Explicit LaTeX-subscript notation (τ_max, k_eff) — "_" is not in the suffix set
#   • Relational expressions (qu=2cu, L=0, a=b) — "=" after the symbol breaks the match
#   • Trig expressions (sinθ, sin2ψ) — θ/ψ after "sin" is not in [a-zA-Z0-9]
#   • Increment-symbol notation (∆x, ∆r) — ∆ (U+2206) is an operator, not a Greek letter;
#     the letter suffix blocks the operator-only branch
#   • Numeric-prefixed expressions (2θ, 4πr) — leading digit matches neither branch
#   • Partial compound expressions (σ3)/2) — trailing "/" is not in the punct set
#   • Properly-subscripted Unicode form (σ₁) — ₁ (U+2081) is in neither suffix set nor
#     trailing punct, so the match fails
_TRIVIAL_INLINE_FRAGMENT = re.compile(
    r"^[.,;:\s()\[\]{}'\"]*"  # optional leading punctuation / whitespace
    r"(?:"
    # Greek letter with an optional short alphanumeric-or-Greek suffix.
    # Covers: σ (bare), σ1 σ3 σ2 (ASCII digit), τf σf (ASCII letter),
    #         σND σLR εhkl (multi-char ASCII), δε σφφ (Greek suffix), μm ρD (unit-like).
    rf"[{GREEK_LETTERS}]"
    rf"[a-zA-Z0-9{GREEK_LETTERS}]{{0,6}}"
    # Lowercase-initial variable reference: x, qu, nD, etc.
    r"|[a-z][a-zA-Z0-9]{0,4}"
    # Uppercase-initial with lowercase-only suffix: Tx, En, Fn — NOT CO2, H2O, NaCl.
    # Molecular formula patterns (uppercase after first char, or uppercase + digit) are
    # intentionally excluded so that chemical formula fragments are not silently dropped.
    r"|[A-Z][a-z]{1,4}"
    r"|[+\-×÷±≤≥≠≈<>·∞∝→↔⇒∇∆∈∉∩∪⊂⊃∀∃∑∫∂√=]+"  # one or more bare operators only
    r")"
    r"[.,;:\s()\[\]{}'\"]*$"  # optional trailing punctuation / whitespace
)

# Broad maths signal for inline detection within running text: relational/arithmetic operators,
# Greek letters, calculus symbols, scientific notation, engineering subscript/fraction notation,
# common math functions, and chemical equation patterns.
_INLINE_MATH = re.compile(
    rf"[{GREEK_LETTERS}]"  # Greek letters
    r"|[∑∫∂√∞±×÷≤≥≠≈→↔⇒⇄⇌⟶∇∆∈∉∩∪⊂⊃∀∃]"  # math + chemical reaction arrows
    r"|[a-zA-Z0-9\)\]\.]\s*[=≈<>≤≥]\s*[a-zA-Z0-9\(\[\-]"  # relations: a = b, x < y
    r"|[a-zA-Z]\s*_\s*[a-zA-Z0-9{]"  # subscripts: k_eff, T_max, σ_y
    r"|[a-zA-Z]\s*\^\s*[a-zA-Z0-9{]"  # superscripts: x^2, E^n
    r"|[0-9]\s*[eE][+-]?[0-9]"  # scientific notation: 9.81e-3
    r"|[A-Z]\s*/\s*[A-Z]"  # engineering ratios: F/A, dL/L, M/I
    r"|\bΔ[A-Za-z]|\bδ[A-Za-z]"  # delta notation: ΔL, δε, δσ
    r"|\b(?:sin|cos|tan|log|ln|exp|lim|max|min)\s*[\(\[]"  # math functions
    # ── Chemical equation patterns ────────────────────────────────────────────
    r"|--\+"  # OCR chemical reaction arrow (bold '→' → '--+')
    r"|(?<![<\-])->(?!-)"  # ASCII reaction arrow -> (not part of <-> or -->)
    # Stoichiometric expression: digit-coefficient before element symbol, e.g. 2H2O, 3CO2, 2.5N2
    r"|(?<![A-Za-z])\d+\.?\d*\s*[A-Z][a-z]?[A-Za-z0-9,]+"
    # Consecutive element symbols (≥2 elements with no spaces), e.g. NaCl, H2SO4, Ca(OH)2
    r"|[A-Z][a-z]?\d*[A-Z][a-z]?\d"
)

# Standalone equation-number label: (12.2.1), (A.3), (1), [12.2], Eq. 12, Equation (5)
# Also handles sub-equation letter suffixes: Eq. 3.9.2(a), Eq. 3.9.2(b)
_EQUATION_LABEL = re.compile(
    r"""
    ^\s*
    (?:
        \(\s*(?:\d{1,3}(?:[.\-]\d{1,3}){0,3}|[A-Z][.\-]\d{1,3}|[ivxlIVXL]{1,5})\s*\)
        | \[\s*(?:\d{1,3}(?:[.\-]\d{1,3}){0,3}|[A-Z][.\-]\d{1,3})\s*\]
        | Eq(?:uation)?[.:]?\s*\(?\s*\d{1,3}(?:[.\-]\d{1,3}){0,3}\s*\)?(?:\s*\([a-z]\))?
    )
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

# OCR confusion: in scanned PDFs the digit '1' is often extracted as letter 'l' when it
# follows a digit-dot sequence (e.g. "Eq. 3.9.l(a)" → "Eq. 3.9.1(a)").
_OCR_DIGIT_L = re.compile(r"(?<=\d\.)l(?=[(\s]|$)")

# (cid:N) — unmapped font glyph placeholder emitted by the PDF text extractor when
# a font has no ToUnicode entry for a glyph.  In engineering PDFs these almost always
# represent mathematical symbols (×, →, √, etc.) that the font renders correctly
# on-screen but whose Unicode mapping is absent from the font descriptor.
_CID_GLYPH_RE = re.compile(r"\(cid:\d+\)")

# Common English prose words — ≥2 hits means the block is running text, not a formula.
_PROSE_WORDS = re.compile(
    r"\b(?:the|a(?:n|nd)?|is|are|was|were|be|been|have|has|had|do|does|did|"
    r"will|would|shall|should|may|might|must|can|could|"
    r"but|or|for|yet|so|because|since|although|while|where|when|which|"
    r"this|these|those|by|in|on|at|to|of|with|from|into|between)\b",
    re.IGNORECASE,
)

# A "where:" / "Note:" line marks the start of a notation or description section that
# follows the formula body.  Stripping this suffix before the prose-count check prevents
# the description from inflating the score and incorrectly classifying a definition block
# as a cross-reference.  Also covers common prose connector phrases used in engineering
# texts (e.g. "from which the value of... may be determined", "Hence, under conditions of",
# "in which V, represents the volume of", "Such a correction, though accurate enough for").
_DESCRIPTION_SECTION_START = re.compile(
    r"^(?:where[:\s]|note[:\s,]|for\b|if\b|let\b|"
    r"from\s+which\b|"
    r"hence[,\s]|therefore[,\s]|thus[,\s]|"
    r"such\s+a\b|in\s+which\b|subsequently[,\s])",
    re.IGNORECASE,
)

# Equation label at the START of a line — same as _EQUATION_LABEL but without the
# trailing \s*$ anchor.  Used to detect labels that have a short connector word
# immediately after them (e.g. "(2-22 )  becomes"), which pdfminer sees as a standalone
# label but pypdfium2 groups with trailing text.
_EQUATION_LABEL_PREFIX = re.compile(
    r"""
    ^\s*
    (?:
        \(\s*(?:\d{1,3}(?:[.\-]\d{1,3}){0,3}|[A-Z][.\-]\d{1,3}|[ivxlIVXL]{1,5})\s*\)
        | \[\s*(?:\d{1,3}(?:[.\-]\d{1,3}){0,3}|[A-Z][.\-]\d{1,3})\s*\]
        | Eq(?:uation)?[.:]?\s*\(?\s*\d{1,3}(?:[.\-]\d{1,3}){0,3}\s*\)?(?:\s*\([a-z]\))?
    )
    \s*
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Short English prose connector words that can follow an equation label in a text block
# when the block renders the label and a brief continuation phrase together (e.g. "(2-22)
# becomes").  The equation itself is image-only; the label is still extractable.
_LABEL_CONNECTOR_WORDS = re.compile(
    r"^(?:becomes?|is|gives?|yields?|then|reduces?|simplifies?|applies?|follows?)\s*$",
    re.IGNORECASE,
)

# Characters that signal math content when counting signal density.
# ¼ (U+00BC) is included because scanned/typeset engineering PDFs frequently use it as
# a stand-in for the equals sign when the font encoding is incomplete.
# Chemical reaction arrows (→⇌⇒⇄↔⟶) are included so that chemistry-only text
# blocks with no Greek letters or calculus symbols still accumulate math-char signal.
_MATH_SIGNAL_CHARS = frozenset(
    GREEK_LETTERS + "∑∫∂√∞±×÷≤≥≠≈→↔⇒⇄⇌⟶∇∆∈∉∩∪⊂⊃∀∃"
    "=+*^_|"  # '/' removed: ratio detection is handled by _INLINE_MATH already;
    # keeping '/' here caused A/B abbreviations (MDO/HFO, VOC/HFO) to
    # score 0.38 and bypass the inline equation scoring gate.
    "¼"
)

# ── Chemical equation detection ───────────────────────────────────────────────

# Chemical reaction arrows — both standard Unicode and common OCR corruptions.
#
# OCR corruption variants documented from scanned chemistry/explosives textbooks:
#   --+   bold reaction arrow '→' read as double-dash + plus (most common)
#   ->    simple ASCII arrow (only matched when not part of <-> or -->  block code)
#
# The pattern is intentionally NOT anchored so it can fire anywhere in a block.
_CHEMICAL_REACTION_ARROW = re.compile(
    r"[→⇌⇒⇄↔⟶]"  # Unicode reaction arrows
    r"|--\+"  # OCR: bold '→' read as --+
    r"|(?<![<\-])->(?!-)"  # ASCII -> not preceded by < or - (avoids <-> matching twice)
)

# A stoichiometric group: an optional coefficient (integer or simple decimal) followed
# immediately by an element-symbol start (uppercase letter).  Examples that match:
#   3CO2   2.5H2O   10KNO3   1.5N2   0.5O2
# Also matches partially OCR-corrupted variants where subscript digits are replaced:
#   3COt   3Coz   1.5Nn   2.5HzO   (subscript '2'→'z', '2'→'t', etc.)
#
# Two or more such groups in a single block are a strong signal that the text
# contains a stoichiometric expression (chemical reaction or product list).
_CHEM_STOICH_SEQUENCE = re.compile(
    r"(?<![A-Za-z0-9])"         # not preceded by letter OR digit (prevents partial 4-digit year match)
    r"\d{1,3}(?:\.\d{1,3})?"   # 1–3 digit coefficient, optional decimal (excludes 4-digit years like 1992)
    r"[ \t]*"                   # horizontal whitespace only — \s* bridged newlines, making section
                                # numbers like "3.5.2.\n\nForward" false-positive stoichiometric matches
    r"[A-Z][a-z]?"              # element symbol start: H, He, Na, Ca, etc.
    r"[A-Za-z0-9,]*"            # rest of the formula (tolerates OCR substitutions)
)


@dataclass
class InlineMatch:
    """One inline-equation span located within a text block."""

    fragment: str
    start: int
    end: int
    bbox: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def is_equation_region(region_type: str) -> bool:
    """True when a layout region type is a display equation (FR-003)."""
    return region_type in EQUATION_REGION_TYPES


def is_isolated_equation_label(text: str) -> bool:
    """True when *text* is exclusively an equation-number label (nothing else).

    Matches patterns like ``(12.2.1)``, ``(A.3)``, ``[12.2]``, ``Eq. 5``,
    and sub-equation variants like ``Eq. 3.9.2(a)``.

    Applies OCR normalization before matching: scanned PDFs frequently extract
    the digit ``1`` as letter ``l`` after a digit-dot sequence
    (e.g. ``Eq. 3.9.l(a)`` → ``Eq. 3.9.1(a)``).
    """
    if not text:
        return False
    normalized = _OCR_DIGIT_L.sub("1", text)
    return bool(_EQUATION_LABEL.match(normalized))


def extract_label_number(label_text: str) -> str:
    """Extract the bare number/identifier from a label string like ``'(12.2.1)'`` → ``'12.2.1'``.

    Preserves sub-equation letter suffixes so ``Eq. 3.9.2(a)`` and ``Eq. 3.9.2(b)``
    produce distinct keys (``'3.9.2a'`` vs ``'3.9.2b'``) and are not deduplicated.
    Applies OCR l→1 normalization before extraction.
    """
    normalized = _OCR_DIGIT_L.sub("1", label_text)
    m = re.search(
        r"\d{1,3}(?:[.\-]\d{1,3}){0,3}|[A-Z][.\-]\d{1,3}|[ivxlIVXL]{1,5}", normalized, re.IGNORECASE
    )
    if not m:
        return normalized.strip("() []")
    base = m.group(0).strip()
    # Include sub-equation letter suffix if immediately following: "(a)", "(b)", ...
    suffix_m = re.match(r"\s*\(([a-z])\)", normalized[m.end() :])
    return f"{base}{suffix_m.group(1)}" if suffix_m else base


def extract_mixed_label_block(text: str) -> tuple[str | None, str]:
    """Detect a text block that contains an equation label on one line and formula on other lines.

    Many technical PDFs (including this Knovel engineering text) render an equation label such
    as ``Eq. 12.4.4`` and its formula body as a single multi-line text block.  Three layouts
    are common:

    * Label-before: ``"Eq. 12.4.4\\nqd,' f\\n.'. t - l.I5(...)"``
    * Label-after:  ``"qd: -+\\nKsE\\nt = (-)\\nEq. 12.2.1"``
    * Label-in-middle: ``"kf, tan ci\\nf\\" =\\nEq. 12.4.10\\nA,\\n-+0.5(1\\n-k)\\ndt"``

    Returns ``(label_line_text, formula_text)`` where *label_line_text* is the raw label line
    (e.g. ``"Eq. 12.4.4"``) and *formula_text* is all remaining non-empty lines joined.

    Returns ``(None, "")`` when:

    * The block has fewer than two non-empty lines — single-line isolated labels are handled
      by :func:`is_isolated_equation_label` in the existing label-scan pass.
    * No single line matches the equation-label pattern (the label is embedded in a prose
      sentence rather than on its own line, indicating a cross-reference).
    * The non-label content contains more than six prose-word matches, which signals a
      cross-reference context block rather than a formula definition.
    """
    if not text:
        return None, ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return None, ""

    label_idx: int | None = None
    label_line: str = ""

    # Primary pass: isolated label on its own line.
    for i, line in enumerate(lines):
        if is_isolated_equation_label(line):
            label_idx = i
            label_line = line
            break

    # Fallback: label at the start of a line followed by ≤1 connecting word
    # (e.g. "(2-22 )  becomes" where pdfminer exposes the label as standalone
    # but pypdfium2 groups it with trailing text).
    if label_idx is None:
        for i, line in enumerate(lines):
            m = _EQUATION_LABEL_PREFIX.match(line)
            if m:
                after_label = line[m.end() :].strip()
                if not after_label or _LABEL_CONNECTOR_WORDS.match(after_label):
                    label_idx = i
                    label_line = m.group(0).strip()
                    break

    if label_idx is None:
        return None, ""

    formula_lines = [ln for i, ln in enumerate(lines) if i != label_idx]

    # Remove lines that are themselves isolated labels or label+connector duplicates
    # (two-column scan artefacts cause the same label to appear twice in one block).
    def _is_label_connector(ln: str) -> bool:
        if is_isolated_equation_label(ln):
            return True
        m = _EQUATION_LABEL_PREFIX.match(ln)
        if m:
            after = ln[m.end() :].strip()
            return not after or bool(_LABEL_CONNECTOR_WORDS.match(after))
        return False

    formula_lines = [ln for ln in formula_lines if not _is_label_connector(ln)]

    # Guard: a high prose-word count indicates the surrounding text is a cross-reference
    # sentence (e.g. "Choose one of the criteria from Eq. 12.4.1 …") not a formula definition.
    # Truncate at the first prose-continuation line (expanded to cover engineering-text
    # patterns such as "from which …", "Hence, …", "in which …", "such a …", "Note, …").
    # The truncated slice is used both for the prose check and as the returned formula text,
    # so that prose continuations are not passed to the OCR / VLM pipeline.
    truncate_at = len(formula_lines)
    for i, ln in enumerate(formula_lines):
        if _DESCRIPTION_SECTION_START.match(ln):
            truncate_at = i
            break
    formula_for_check = "\n".join(formula_lines[:truncate_at])

    if len(_PROSE_WORDS.findall(formula_for_check)) > 6:
        # Even after stripping prose-continuation lines the remaining text is prose-heavy:
        # the label was on its own line but no extractable formula text remains.  Return
        # the label so the caller can still record the equation with an image-only crop.
        return label_line, ""

    return label_line, formula_for_check


def looks_like_standalone_formula(text: str) -> bool:
    """Heuristic: True when *text* looks like a display formula rather than running prose.

    Thin wrapper over :func:`equation_extraction.formula_detector.score_formula_candidate`
    using text-only mode (no layout geometry).  Returns ``True`` when the computed score
    is at or above ``TEXT_ONLY_THRESHOLD`` (0.15) — a deliberately low bar that preserves
    backward compatibility with short math expressions such as ``"2θ"`` that score well
    below the full ``FORMULA_THRESHOLD`` (0.45) without geometry context.

    The lazy import avoids a circular dependency: ``formula_detector`` imports from this
    module; we defer until call time so both modules can be loaded independently.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 250 or len(stripped) == 1:
        return False
    from equation_extraction.formula_detector import (  # noqa: PLC0415
        TEXT_ONLY_THRESHOLD,
        score_formula_candidate,
    )

    return score_formula_candidate(stripped).score >= TEXT_ONLY_THRESHOLD


def region_text(region: LayoutRegion, block: TextBlock | None) -> str:
    """Best-available text for a display equation region: its text block, else layout attributes."""
    if block is not None and (block.text or "").strip():
        return block.text or ""
    value = region.attributes.get("value") if isinstance(region.attributes, dict) else None
    return str(value) if value else ""


def derive_inline_bbox(block: TextBlock) -> tuple[list[float], list[str]]:
    """Inline bbox derivation policy (FR-005).

    A ``TextBlock`` exposes only block-level geometry, so an inline equation's bbox falls back to the
    containing block's ``bbox`` and records ``inline_bbox:block_fallback``. Per-character span boxes
    would be used here if the upstream block exposed them; it does not, so the fallback is the
    deterministic behavior and the matched span offsets (recorded separately) keep the location
    traceable.
    """
    bbox = list(block.bbox) if block.bbox is not None else []
    return bbox, ["inline_bbox:block_fallback"]


# Gap between two inline matches that is composed only of formula operators and connectives.
# Consecutive matches separated by such a gap are merged into one compound span so that
# "10KNOa + 3s + 8C + 3K2S0," is returned as one fragment, not four separate ones.
_FORMULA_GAP_RE = re.compile(r"^[\s+\-×÷→⇌⇒⇄↔⟶,;:/()]+$")


def _merge_adjacent_spans(text: str, matches: list[InlineMatch]) -> list[InlineMatch]:
    """Merge consecutive spans separated only by formula connectives into one compound span.

    Two passes:
    1. Pure-operator gap  — gap is only whitespace / ``+`` / ``-`` / arrows / punctuation.
    2. Short formula gap  — gap is ≤ 25 chars and contains no prose words (e.g. ``" + 3s + 8C + "``
       represents undetected stoichiometric tokens between two detected formula tokens; merging
       them produces the complete balanced equation rather than disjoint fragments).
    """
    if len(matches) <= 1:
        return matches
    merged: list[InlineMatch] = []
    cur_start = matches[0].start
    cur_end = matches[0].end
    base = matches[0]
    for nxt in matches[1:]:
        gap = text[cur_end : nxt.start]
        is_formula_gap = _FORMULA_GAP_RE.match(gap) or (
            len(gap) <= 25 and not _PROSE_WORDS.search(gap)
        )
        if is_formula_gap:
            cur_end = nxt.end  # extend
        else:
            fragment = text[cur_start:cur_end].strip()
            if fragment:
                merged.append(
                    InlineMatch(
                        fragment=fragment,
                        start=cur_start,
                        end=cur_end,
                        bbox=base.bbox,
                        notes=list(base.notes),
                    )
                )
            cur_start = nxt.start
            cur_end = nxt.end
            base = nxt
    fragment = text[cur_start:cur_end].strip()
    if fragment:
        merged.append(
            InlineMatch(
                fragment=fragment,
                start=cur_start,
                end=cur_end,
                bbox=base.bbox,
                notes=list(base.notes),
            )
        )
    return merged


def detect_inline_spans(block: TextBlock, *, enabled: bool) -> list[InlineMatch]:
    """Locate inline-equation spans within a text block's text (FR-003).

    Returns at most one match per contiguous math run. Display/heading/code roles and non-text blocks
    are skipped by the caller; here we only require a non-empty text and the feature toggle.

    Consecutive individual-token matches that are separated only by formula connectives
    (``+``, ``-``, ``→``, whitespace, etc.) are merged into one compound span so that a
    balanced equation like ``10KNOa + 3s + 8C → ...`` is returned as a single fragment
    rather than as five separate single-term matches.
    """
    text = block.text or ""
    if not enabled or not text.strip():
        return []
    matches: list[InlineMatch] = []
    bbox, bbox_notes = derive_inline_bbox(block)
    for match in _INLINE_MATH.finditer(text):
        # Expand the match to the surrounding non-space token run for a more useful fragment.
        start, end = match.start(), match.end()
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        while end < len(text) and not text[end].isspace():
            end += 1
        fragment = text[start:end].strip()
        if not fragment:
            continue
        if _TRIVIAL_INLINE_FRAGMENT.match(fragment):
            continue  # standalone symbol or bare operator — not an equation
        if matches and start < matches[-1].end:  # overlaps the previous run → skip
            continue
        matches.append(
            InlineMatch(fragment=fragment, start=start, end=end, bbox=bbox, notes=list(bbox_notes))
        )
    return _merge_adjacent_spans(text, matches)
