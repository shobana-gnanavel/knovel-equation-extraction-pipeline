"""Equation block detector — merged module.

Sections
--------
1. Visual detection helpers       (from visual_extraction/detection.py)
   Region-type selection, bbox helpers, IoU overlap test, and
   deduplication of overlapping layout regions.

2. Visual content classifier      (from visual_extraction/classifier.py)
   Deterministic multi-signal classifier that maps a detected visual
   region to one of fourteen categories (VisualClassification).
   Note: ``Classification`` is renamed ``VisualClassification`` and
   ``classify_region`` is renamed ``classify_visual_region`` to avoid
   collision with the equation-content classifier in Section 4.

3. Formula confidence scorer      (from formula_detector.py)
   Confidence-based display-formula candidate scorer with mathematical,
   layout, and structural signals.

4. Equation content classifier    (from equation_classifier.py)
   Rule-based 6-category equation content classifier (mathematical_equation,
   engineering_formula, statistical_expression, chemical_equation,
   chemical_structure, unknown).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from equation_extraction_pipeline.domain.constants import GREEK_LETTERS
from equation_extraction_pipeline.domain.models import (
    VISUAL_CATEGORIES,
    LayoutRegion,
    TextBlock,
)

# ---------------------------------------------------------------------------
# Section 1 — Visual detection helpers  (from visual_extraction/detection.py)
# ---------------------------------------------------------------------------
# Visual-region detection utilities. Selects figure/image-typed LayoutRegions
# produced by the layout analysis stage, de-duplicates overlapping/duplicate
# regions (FR-002), and exposes bbox helpers plus the IoU overlap test.

__all__ = [
    # visual detection
    "FIGURE_REGION_TYPES",
    "is_visual_region",
    "region_text",
    "bbox_list",
    "valid_bbox",
    "overlaps",
    "dedupe_regions",
    # visual classifier
    "VisualClassification",
    "classify_visual_region",
    "recommended_provider_for_category",
    # formula scorer
    "FormulaScore",
    "score_formula_candidate",
    "FORMULA_THRESHOLD",
    "AMBIGUOUS_THRESHOLD",
    "TEXT_ONLY_THRESHOLD",
    # equation classifier
    "Classification",
    "classify_region",
    "DEFAULT_PROVIDER_BY_CATEGORY",
    "EQUATION_CATEGORIES",
]

# Layout region types that are visual content (feature 005 graphical taxonomy).
FIGURE_REGION_TYPES: frozenset[str] = frozenset({"figure", "image"})


def is_visual_region(region_type: str) -> bool:
    """True when a layout region type is a figure/image visual (FR-001)."""
    return region_type in FIGURE_REGION_TYPES


def region_text(region: LayoutRegion, block: TextBlock | None) -> str:
    """Best-available text for a visual region: its text block, else layout attributes."""
    if block is not None and block.text.strip():
        return block.text
    attrs = region.attributes if isinstance(region.attributes, dict) else {}
    value = attrs.get("value") or attrs.get("text")
    return str(value) if value else ""


def bbox_list(region: LayoutRegion) -> list[float]:
    """Convert a layout ``{x0,y0,x1,y1}`` bbox dict into a 4-tuple list (FR-001)."""
    bbox = region.bbox if isinstance(region.bbox, dict) else {}
    return [
        float(bbox.get("x0", 0.0)),
        float(bbox.get("y0", 0.0)),
        float(bbox.get("x1", 0.0)),
        float(bbox.get("y1", 0.0)),
    ]


def valid_bbox(bbox: list[float]) -> bool:
    """True when a bbox is a positive-area, in-page 4-tuple (FR-001)."""
    if len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    return x1 > x0 and y1 > y0 and x0 >= 0 and y0 >= 0


def _iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-union of two valid 4-tuple bboxes (0.0 if either is invalid)."""
    if not (valid_bbox(a) and valid_bbox(b)):
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def overlaps(a: list[float], b: list[float], *, threshold: float = 0.7) -> bool:
    """True when two regions overlap enough (IoU ≥ threshold) to be considered duplicates (FR-002)."""
    return _iou(a, b) >= threshold


def dedupe_regions(
    regions: list[LayoutRegion], *, threshold: float = 0.7
) -> tuple[list[LayoutRegion], set[str]]:
    """Drop near-duplicate overlapping regions, keeping the first seen (FR-002).

    Returns ``(kept_regions, duplicate_region_ids)`` — the duplicates are
    reported (not silently lost) so the caller can flag them; the kept region
    is the earliest in the input order.
    """
    kept: list[LayoutRegion] = []
    kept_boxes: list[list[float]] = []
    duplicate_ids: set[str] = set()
    for region in regions:
        box = bbox_list(region)
        if any(overlaps(box, other, threshold=threshold) for other in kept_boxes):
            duplicate_ids.add(region.region_id)
            continue
        kept.append(region)
        kept_boxes.append(box)
    return kept, duplicate_ids


# ---------------------------------------------------------------------------
# Section 2 — Visual content classifier  (from visual_extraction/classifier.py)
# ---------------------------------------------------------------------------
# Rule-based visual classification (feature 010, FR-003/FR-004).
#
# A deterministic, multi-signal classifier maps a detected visual region to
# one of fourteen categories. Renamed to VisualClassification / classify_visual_region
# to avoid collision with the equation content classifier in Section 4.

_VISUAL_CUES: list[tuple[str, "re.Pattern[str]"]] = [
    (
        "chemical_structure",
        re.compile(r"\b(scheme|chemical structure|reaction|compound|moiety|SMILES)\b", re.I),
    ),
    (
        "circuit_diagram",
        re.compile(
            r"\b(circuit|schematic diagram|wiring|resistor|capacitor|transistor)\b", re.I
        ),
    ),
    (
        "flowchart",
        re.compile(
            r"\b(flow\s*chart|flow diagram|process flow|workflow|decision tree)\b", re.I
        ),
    ),
    (
        "cad_drawing",
        re.compile(
            r"\b(CAD|assembly drawing|exploded view|isometric|orthographic)\b", re.I
        ),
    ),
    (
        "engineering_drawing",
        re.compile(
            r"\b(engineering drawing|schematic|blueprint|cross[- ]section|elevation|cutaway)\b",
            re.I,
        ),
    ),
    (
        "map",
        re.compile(
            r"\b(map|geographic|topograph|cartograph|region map|location map)\b", re.I
        ),
    ),
    (
        "screenshot",
        re.compile(
            r"\b(screenshot|screen capture|user interface|UI|dialog box|window)\b", re.I
        ),
    ),
    ("chart", re.compile(r"\b(bar chart|pie chart|histogram|chart)\b", re.I)),
    ("graph", re.compile(r"\b(graph|plot|scatter|line graph|curve|axis|axes)\b", re.I)),
    (
        "photograph",
        re.compile(
            r"\b(photo(graph)?|micrograph|microscopy|SEM|TEM|image of|specimen)\b", re.I
        ),
    ),
    ("diagram", re.compile(r"\b(diagram|block diagram|illustration|schematic of)\b", re.I)),
]

_VISUAL_PROVIDER_BY_CATEGORY: dict[str, str] = {
    "graph": "opencv",
    "chart": "opencv",
    "chemical_structure": "chemical",
    "diagram": "generic",
    "flowchart": "generic",
    "engineering_drawing": "generic",
    "cad_drawing": "generic",
    "circuit_diagram": "generic",
    "map": "generic",
    "screenshot": "generic",
    "figure": "docling",
    "photograph": "docling",
    "composite_figure": "docling",
    "unknown": "default",
}


@dataclass
class VisualClassification:
    """The visual classifier's verdict for one region (FR-003).

    Renamed from ``Classification`` in visual_extraction/classifier.py to avoid
    collision with the equation content ``Classification`` in this module.
    """

    category: str
    confidence: float
    reason: str
    recommended_provider: str


def recommended_provider_for_category(category: str) -> str:
    """Advisory provider for a category (selection.py applies the authoritative routing)."""
    return _VISUAL_PROVIDER_BY_CATEGORY.get(category, "default")


def _visual_result(category: str, confidence: float, reason: str) -> VisualClassification:
    return VisualClassification(
        category=category,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
        reason=reason,
        recommended_provider=recommended_provider_for_category(category),
    )


def classify_visual_region(
    *,
    caption_text: str = "",
    region_text: str = "",
    region_subtype: str | None = None,
    aspect_ratio: float = 0.0,
    is_composite: bool = False,
    is_photo_like: bool | None = None,
    section_context: str = "",
    min_confidence: float = 0.5,
) -> VisualClassification:
    """Classify one visual region into a category with confidence, reason, and provider (FR-003/FR-004).

    Renamed from ``classify_region`` in visual_extraction/classifier.py.

    Precedence: composite → explicit layout subtype → caption/text cues (chemical →
    circuit → flowchart → cad → engineering → map → screenshot → chart → graph →
    photograph → diagram) → image-statistic photograph/figure fallback → ``unknown``.
    """
    if is_composite:
        return _visual_result("composite_figure", 0.85, "multi-panel layout with panel labels")

    if region_subtype and region_subtype in VISUAL_CATEGORIES:
        return _visual_result(region_subtype, 0.9, f"layout subtype '{region_subtype}'")

    haystack = f"{caption_text}\n{region_text}\n{section_context}"
    for category, pattern in _VISUAL_CUES:
        if pattern.search(haystack):
            return _visual_result(category, 0.8, f"caption/text cue for {category}")

    if is_photo_like is True:
        return _visual_result("photograph", 0.6, "continuous-tone image statistics")
    if is_photo_like is False:
        return _visual_result("figure", 0.55, "line-art image statistics")

    if caption_text.strip() or region_text.strip():
        return _visual_result(
            "figure", max(min_confidence, 0.5), "captioned visual, no specific cue"
        )

    return _visual_result("unknown", 0.3, "no decisive visual signal")


# Sanity: every category the classifier can emit is a known visual category.
assert set(_VISUAL_PROVIDER_BY_CATEGORY) == set(VISUAL_CATEGORIES)


# ---------------------------------------------------------------------------
# Section 3 — Formula confidence scorer  (from formula_detector.py)
# ---------------------------------------------------------------------------
# Confidence-based display-formula candidate scorer.
#
# Every display formula has at least one of three types of evidence:
#   Mathematical signals  — relational operators, Greek/math chars, (cid:N)
#                           font-glyph placeholders, inline-math regex patterns.
#   Layout signals        — block narrower than body text, centered on page,
#                           vertical proximity to an equation-number label.
#   Structural signals    — fraction layout (multi-line, short lines, no verb).
#
# Thresholds
# ----------
# FORMULA_THRESHOLD   (0.45) — unconditionally a formula.
# AMBIGUOUS_THRESHOLD (0.25) — grey zone; LLM oracle resolves when available.
# TEXT_ONLY_THRESHOLD (0.15) — lower bar for the text-only backward-compat wrapper.

FORMULA_THRESHOLD: float = 0.45
AMBIGUOUS_THRESHOLD: float = 0.25
TEXT_ONLY_THRESHOLD: float = 0.15

_FINITE_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|have|has|do|does|did|will|would|shall|should"
    r"|may|might|must|can|could)\b",
    re.IGNORECASE,
)

_RELATIONAL_OP_RE = re.compile(r"[=¼≈≠≤≥]")

_INLINE_MATH = re.compile(
    rf"[{GREEK_LETTERS}]"
    r"|[∑∫∂√∞±×÷≤≥≠≈→↔⇒⇄⇌⟶∇∆∈∉∩∪⊂⊃∀∃]"
    r"|[a-zA-Z0-9\)\]\.]\s*[=≈<>≤≥]\s*[a-zA-Z0-9\(\[\-]"
    r"|[a-zA-Z]\s*_\s*[a-zA-Z0-9{]"
    r"|[a-zA-Z]\s*\^\s*[a-zA-Z0-9{]"
    r"|[0-9]\s*[eE][+-]?[0-9]"
    r"|[A-Z]\s*/\s*[A-Z]"
    r"|\bΔ[A-Za-z]|\bδ[A-Za-z]"
    r"|\b(?:sin|cos|tan|log|ln|exp|lim|max|min)\s*[\(\[]"
    r"|--\+"
    r"|(?<![<\-])->(?!-)"
    r"|(?<![A-Za-z])\d+\.?\d*\s*[A-Z][a-z]?[A-Za-z0-9,]+"
    r"|[A-Z][a-z]?\d*[A-Z][a-z]?\d"
)

_CID_GLYPH_RE = re.compile(r"\(cid:\d+\)")

_FORMULA_PROSE_WORDS = re.compile(
    r"\b(?:the|a(?:n|nd)?|is|are|was|were|be|been|have|has|had|do|does|did|"
    r"will|would|shall|should|may|might|must|can|could|"
    r"but|or|for|yet|so|because|since|although|while|where|when|which|"
    r"this|these|those|by|in|on|at|to|of|with|from|into|between)\b",
    re.IGNORECASE,
)

_MATH_SIGNAL_CHARS = frozenset(
    GREEK_LETTERS + "∑∫∂√∞±×÷≤≥≠≈→↔⇒⇄⇌⟶∇∆∈∉∩∪⊂⊃∀∃"
    "=+*^_|"
    "¼"
)

_CHEMICAL_REACTION_ARROW = re.compile(
    r"[→⇌⇒⇄↔⟶]"
    r"|--\+"
    r"|(?<![<\-])->(?!-)"
)

_CHEM_STOICH_SEQUENCE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"\d{1,3}(?:\.\d{1,3})?"
    r"[ \t]*"
    r"[A-Z][a-z]?"
    r"[A-Za-z0-9,]*"
)


@dataclass
class FormulaScore:
    """Confidence evaluation result for a single formula candidate block."""

    score: float
    signals: dict[str, float] = field(default_factory=dict)
    is_formula: bool = False
    needs_llm: bool = False


_DIGIT_DOT_ONLY = re.compile(r"^\d[\d\.]*\.?$")


def _fraction_layout_valid(stripped: str) -> bool:
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
    """Compute a confidence score for whether *text* is a display formula."""
    stripped = text.strip()

    if region_type in {"equation", "formula"}:
        return FormulaScore(
            score=1.0,
            signals={"region_type_formula": 1.0},
            is_formula=True,
        )

    if not stripped or len(stripped) == 1 or len(stripped) > 250:
        return FormulaScore(score=0.0)

    signals: dict[str, float] = {}

    has_chem_arrow = bool(_CHEMICAL_REACTION_ARROW.search(stripped))
    if has_chem_arrow:
        signals["chemical_reaction_arrow"] = 0.40

    stoich_groups = len(_CHEM_STOICH_SEQUENCE.findall(stripped))
    if stoich_groups >= 3:
        signals["chemical_stoich_sequence"] = 0.25
    elif stoich_groups >= 2:
        signals["chemical_stoich_sequence"] = 0.15

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

    if _fraction_layout_valid(stripped):
        signals["fraction_layout"] = 0.28

    if len(stripped) < 80:
        signals["short_text"] = 0.08

    if label_distance_pts is not None:
        if label_distance_pts < 30:
            signals["label_proximity_close"] = 0.30
        elif label_distance_pts < 80:
            signals["label_proximity_near"] = 0.15

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

    prose_hits = len(_FORMULA_PROSE_WORDS.findall(stripped))
    if prose_hits >= 5:
        signals["heavy_prose"] = -0.50
    elif prose_hits >= 3 and not has_any_math:
        signals["moderate_prose"] = -0.25

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


# ---------------------------------------------------------------------------
# Section 4 — Equation content classifier  (from equation_classifier.py)
# ---------------------------------------------------------------------------
# Rule-based equation content classification (6 categories).
#
# Maps a detected equation region to one of the six content categories with
# a confidence, a human-readable reason, and a recommended provider.
# Deterministic for a given input.

# The six equation content categories recognised by this pipeline.
EQUATION_CATEGORIES: tuple[str, ...] = (
    "mathematical_equation",
    "engineering_formula",
    "statistical_expression",
    "chemical_equation",
    "chemical_structure",
    "unknown",
)

DEFAULT_PROVIDER_BY_CATEGORY: dict[str, str] = {
    "mathematical_equation": "qwen_vl",
    "engineering_formula": "qwen_vl",
    "statistical_expression": "qwen_vl",
    "chemical_equation": "qwen_vl",
    "chemical_structure": "qwen_vl",
    "unknown": "generic",
}

_MATH_OPS = re.compile(
    rf"[{GREEK_LETTERS}∑∫∂√∞±×÷≤≥≠≈∇∆]|[=^_]|\b(sin|cos|tan|log|ln|exp|lim)\b"
)
_REACTION = re.compile(
    r"[→⇌⇒⇄↔⟶]"
    r"|-+>|<-+|=+>"
    r"|--\+"
)
_MOLECULE = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
_STOICH = re.compile(r"(?:^|[\s+])\d+\.?\d*\s*[A-Z][a-z]?")
_STRUCTURE_HINT = re.compile(
    r"\b(benzene|ring|bond|aromatic|cyclo|structure)\b", re.IGNORECASE
)
_STAT = re.compile(
    r"\b(Var|Cov|Pr|SD|std|mean|median|variance|distribution|regression|correlation)\b"
    r"|[μσ]"
    r"|\bP\s*\("
    r"|\bE\s*\["
    r"|~\s*N\(|\bN\(0"
)
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
    """The equation classifier's verdict for one region."""

    category: str
    confidence: float
    reason: str
    recommended_provider: str


def _eq_result(category: str, confidence: float, reason: str) -> Classification:
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
    """Classify one equation region into a category with confidence, reason, and provider."""
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

    stoich_count = len(_STOICH.findall(text))

    if (
        has_reaction
        or (has_molecule and has_stoich and not has_math)
        or (stoich_count >= 2 and not has_math)
    ):
        return _eq_result(
            "chemical_equation",
            0.85,
            "chemical reaction signals (arrow/stoichiometry/molecular formula)",
        )
    if has_structure_hint or (has_molecule and not has_math and not has_stat):
        return _eq_result(
            "chemical_structure",
            0.75,
            "chemical structure signals (molecular formula / ring/bond cues, no relational math)",
        )

    if has_stat:
        return _eq_result(
            "statistical_expression",
            0.7,
            "statistical operators/notation (expectation/variance/distribution)",
        )

    if has_unit or eng_context:
        return _eq_result(
            "engineering_formula",
            0.7,
            "physical-quantity/unit cues or engineering section context",
        )

    if has_math:
        return _eq_result("mathematical_equation", 0.8, "mathematical operators/symbols present")

    if region_type:
        return _eq_result(
            "mathematical_equation",
            0.6,
            "detected equation region (no chemical/statistical/unit signal in text)",
        )
    if is_inline:
        return _eq_result("mathematical_equation", 0.55, "inline math run detected in text")

    return _eq_result("unknown", 0.3, "no decisive equation signal")
