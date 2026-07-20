"""Equation label detector.

Detects equation regions across all pages using label-scan (primary) and
Docling ML (fallback) strategies, saves crops, and exposes the public
``detect_equations`` entry point. Document/page *classification* lives in the
sibling ``document_classifier`` module (its former Sections 1-6).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTChar, LTPage, LTTextBox, LTTextLine
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from PIL import Image

try:  # pragma: no cover - optional dependency handling
    import numpy as np
except Exception:  # pragma: no cover - optional dependency handling
    np = None  # type: ignore[assignment]

from equation_extraction_pipeline.config import settings as config
from equation_extraction_pipeline.domain.models import (
    ClassificationResult,
    EquationRegion,
    RenderedPage,
)

logger = logging.getLogger(__name__)

# Math-symbol probe used by image-math page flagging. Kept local to this module
# so detection stays decoupled from document_classifier (which defines an
# identical constant for its own signal collection).
_MATH_SYMBOL = re.compile(r"[=±∑∫∂√≤≥≈×÷∞°µ]")

__all__ = [
    "detect_equations",
    "scan_equation_labels",
]


# ---------------------------------------------------------------------------
# Section 7 — Equation layout detection  (from layout_detection.py)
# ---------------------------------------------------------------------------
# Detects equation regions across all pages using two strategies:
#
# Label mode  — fast regex scan of the PDF text layer for 'Eq. X.X.X' margin
#               labels.  Each label anchors a bounding box for the equation block.
# ML mode     — full Docling layout analysis when no labels are found.  Regions
#               classified as 'formula' or 'equation' are returned as regions.
#
# After detection each region's crop is saved to
# ``<output_dir>/crops/page_NNN/<eq_id>.png`` with CROP_PADDING_PX padding.

# Regex matching "Eq. 12.2.1" or "Eq 12.2.1" (with optional space/period).
# The digit group also accepts 'l' and 'I' which OCR frequently confuses with '1'.
_LABEL_RE = re.compile(
    r"Eq(?:uation)?[.:]?\s*((?:\d|[lI])+(?:\.\s*(?:\d|[lI])+)+(?:\s*\(\s*[a-z]\s*\))?)",
    re.IGNORECASE,
)

_LABEL_OCR_FIX = str.maketrans({"l": "1", "I": "1", "O": "0"})

# Dash-numbered definition labels: "(2-1)", "(2-30)". Unlike _PAREN_LABEL_RE /
# _PAREN_DOTTED_LABEL_RE (which require the label to be the box's ENTIRE text), these
# usually share a line with the equation, so they are scanned with finditer + the same
# cross-reference rejection as _LABEL_RE matches.
_DASH_LABEL_RE = re.compile(r"\(\s*((?:\d|[lI]){1,3}\s*-\s*(?:\d|[lI]){1,3})\s*\)")

# Dotted parenthesized labels that share a line with the equation: "y = mx  (5.2)".
# Anchored at end-of-line because "(5.2)" also occurs mid-sentence as a cross-reference
# and as a plain numeric value; matches are additionally non-explicit (they require
# aligned formula geometry downstream) and reject a digit/dot immediately before the
# paren (decimal values like "0.5(2.3)").
_PAREN_EOL_LABEL_RE = re.compile(
    # First component [1-9]…: chapter numbers never start with 0, whereas parenthesized
    # decimal VALUES ("(0.164)" in tables) do.
    r"\(\s*([1-9](?:\d|[lI]){0,2}\.(?:\d|[lI]){1,3})\s*\)[ \t]*$",
    re.MULTILINE,
)

# Bracketed dotted labels at end-of-line: "y = mx  [1.1]". Same trust model and guards as
# _PAREN_EOL_LABEL_RE (non-explicit; document arbitration + formula geometry). The dotted
# form is REQUIRED so citation brackets "[12]" never match.
_BRACKET_EOL_LABEL_RE = re.compile(
    r"\[\s*([1-9](?:\d|[lI]){0,2}\.(?:\d|[lI]){1,3})\s*\][ \t]*$", re.MULTILINE
)

# Whole-box bracketed dotted label: box text is exactly "[1.1]".
_BRACKET_BOX_LABEL_RE = re.compile(
    r"^\s*\[\s*([1-9]\d{0,2}(?:\.\d{1,3}){1,3})\s*\]\s*$",
)

# Whole-box BARE dotted number: box text is exactly "1.6" (a standalone right-margin label).
# Inline bare numbers are un-disambiguable from values ("= 1.6"), so only the whole-box form
# is ever considered, and it still needs formula-geometry corroboration + arbitration.
_BARE_BOX_LABEL_RE = re.compile(
    r"^\s*([1-9]\d{0,2}(?:\.\d{1,3}){1,3})\s*$",
)


def _iter_label_matches(text: str):
    """Yield ``(match, explicit)`` for definition-label conventions found in ``text``.

    ``explicit`` matches (Eq. X.X.X, dash labels) are trusted on their own; non-explicit
    ones (end-of-line "(5.2)") must be corroborated by aligned formula geometry because
    the same token also appears as cross-references and numeric values.
    """
    for m in _LABEL_RE.finditer(text):
        yield m, True
    for m in _DASH_LABEL_RE.finditer(text):
        yield m, True
    for eol_re in (_PAREN_EOL_LABEL_RE, _BRACKET_EOL_LABEL_RE):
        for m in eol_re.finditer(text):
            prev = text[m.start() - 1] if m.start() > 0 else ""
            if prev.isdigit() or prev == ".":
                continue
            yield m, False


def _allowed_nonexplicit_labels(
    candidates: list[tuple[str, bool]],
) -> set[str]:
    """Document-level arbitration for non-explicit ("(5.2)" end-of-line) label candidates.

    Explicit conventions (Eq. X.X.X, dash) are unambiguous; "(5.2)" also matches numeric
    values, so it is only trusted when it IS the book's convention: when explicit labels
    clearly dominate (>=3 of them and at least as many as the paren candidates), every
    non-explicit candidate is a value/cross-ref and is dropped. Dominance, not an absolute
    count — books using the paren convention still mention "Eq. 4.22" in prose a few times.
    Otherwise a non-explicit label is kept only when its leading component (chapter prefix)
    repeats — one-off hits like "(0.132)" or "(94.39)" do not.
    """
    explicit_labels = {lbl for lbl, exp in candidates if exp}
    nonexplicit = {lbl for lbl, exp in candidates if not exp}
    if len(explicit_labels) >= 3 and len(explicit_labels) >= len(nonexplicit):
        return set()
    prefix_counts: dict[str, int] = {}
    for lbl in nonexplicit:
        prefix = re.split(r"[.\-]", lbl)[0]
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    return {lbl for lbl in nonexplicit if prefix_counts[re.split(r"[.\-]", lbl)[0]] >= 2}


def _normalise_label(raw: str) -> str:
    return re.sub(r"\s+", "", raw).translate(_LABEL_OCR_FIX)


# Standalone connective/prose words that occasionally precede a displayed equation on the same
# typeset line (e.g. "or  P_u = f_u A_u"). When such a word is the leading token of a formula
# fragment its glyphs are trimmed from the crop's left edge so the crop is the equation alone.
# Kept deliberately short and lowercase; a leading single variable (P, f, x) is never in this set.
_LEADING_PROSE = frozenset(
    {"or", "and", "where", "for", "thus", "hence", "so", "then", "with", "if", "when", "use"}
)


def _iter_text_lines(box: LTTextBox) -> list[LTTextLine]:
    """Return the LTTextLine children of a text box (empty if not iterable)."""
    try:
        return [ln for ln in box if isinstance(ln, LTTextLine)]
    except TypeError:  # pragma: no cover - defensive; LTTextBox is iterable
        return []


def _label_anchor_bbox(box: LTTextBox, label_str: str) -> tuple[float, float, float, float]:
    """Return the tightest bbox anchoring ``label_str`` within ``box``.

    Pdfminer sometimes merges a right-margin label ("Eq. 12.4.8") with the adjacent prose
    paragraph into a single wide text box, so the box's own ``bbox`` left edge is the paragraph
    left, not the label. Anchoring on the specific text LINE that carries the label restores the
    correct label geometry (used for the formula search and crop-tightening anchor). Falls back
    to the full box bbox when no line carries the label (the common single-line-label case is
    identical either way).
    """
    for line in _iter_text_lines(box):
        for m, _explicit in _iter_label_matches(line.get_text()):
            if _normalise_label(m.group(1)) == label_str:
                return line.bbox
    return box.bbox


class _BBoxAnchor:
    """Lightweight ``.bbox``-bearing stand-in so bbox helpers can take a tightened label bbox."""

    __slots__ = ("bbox",)

    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        self.bbox = bbox


def _leading_prose_x0(box: LTTextBox) -> float | None:
    """Return the x0 of the first glyph after a leading prose word, or ``None``.

    Scans the leftmost text line: if its first whitespace-delimited token is a standalone prose
    word (see ``_LEADING_PROSE``), the equation actually starts at the next glyph, so the label
    crop should begin there rather than at the prose. Returns ``None`` when there is no such
    leading prose (the overwhelmingly common case), leaving the left edge untouched.
    """
    lines = _iter_text_lines(box)
    if not lines:
        return None
    line = min(lines, key=lambda ln: ln.bbox[0])

    token = ""
    token_done = False
    for obj in line:
        text = obj.get_text()
        if not token_done:
            if text.strip() == "":
                if token:
                    token_done = True
                continue
            token += text
        else:
            if text.strip() == "":
                continue
            if isinstance(obj, LTChar):
                if token.lower() in _LEADING_PROSE:
                    return float(obj.x0)
                return None
    return None


def _formula_bbox_for_label(
    label_box: LTTextBox,
    formula_boxes: list[LTTextBox],
) -> tuple[float, float, float, float] | None:
    """Return the complete formula bbox aligned with a right-margin label.

    Pdfminer commonly splits a typeset fraction into separate LHS, numerator,
    denominator, and RHS boxes.  Selecting only the nearest box produces tiny
    crops.  Merge formula fragments to the left of the label that are within
    the label's immediate vertical neighbourhood so recognition receives the
    whole equation.

    Only boxes whose centroid falls within ±2 label-heights of the label
    centroid are considered, and the merged bbox is further capped at
    ±3 label-heights to avoid merging adjacent equations.
    """
    lx0, ly0, _lx1, ly1 = label_box.bbox
    label_height = max(ly1 - ly0, 8.0)
    label_cy = (ly0 + ly1) / 2.0
    max_centroid_dist = label_height * 2.0

    aligned: list[LTTextBox] = []
    for box in formula_boxes:
        fx0, fy0, fx1, fy1 = box.bbox
        vertical_overlap = min(ly1, fy1) - max(ly0, fy0)
        if vertical_overlap <= 0 or fx0 >= lx0 or fx1 > lx0 + 8.0:
            continue
        box_cy = (fy0 + fy1) / 2.0
        if abs(box_cy - label_cy) > max_centroid_dist:
            continue
        aligned.append(box)

    if not aligned:
        return None
    # Left edge, trimming any leading prose token ("or", "where", …) that shares a fragment's
    # line so the crop starts at the equation rather than the connective word.
    left_edges: list[float] = []
    for box in aligned:
        trimmed = _leading_prose_x0(box)
        left_edges.append(trimmed if trimmed is not None else box.bbox[0])
    x0 = min(left_edges)
    raw_y0 = min(box.bbox[1] for box in aligned)
    raw_y1 = max(box.bbox[3] for box in aligned)
    capped_y0 = max(raw_y0, label_cy - label_height * 3.0)
    capped_y1 = min(raw_y1, label_cy + label_height * 3.0)
    return (
        max(0.0, x0 - 4.0),
        capped_y0,
        lx0 - 12.0,
        capped_y1,
    )


def _image_formula_bbox_for_label(
    label_box: LTTextBox,
) -> tuple[float, float, float, float]:
    """Estimate an image-only formula crop immediately left of its label.

    Some embedded equation fonts have no usable Unicode map, so pdfminer emits
    the right-margin label but no formula text box at all.  Uses a conservative
    band around the label baseline with asymmetric padding so stacked fractions
    are retained without pulling in the following prose.
    """
    lx0, ly0, _lx1, ly1 = label_box.bbox
    label_height = max(ly1 - ly0, 8.0)
    return (
        max(0.0, lx0 * 0.24),
        max(0.0, ly0 - (0.75 * label_height)),
        max(0.0, lx0 - 12.0),
        ly1 + (2.0 * label_height),
    )


# Secondary label: box whose ENTIRE text is a parenthesized integer or Roman numeral
_PAREN_LABEL_RE = re.compile(
    r"^\s*\(\s*([ivxlIVXL]{1,5}|\d{1,3})\s*\)\s*$",
)

# Parenthesized multi-part dotted label: "(5.5.11)", "(3.9.1)"
_PAREN_DOTTED_LABEL_RE = re.compile(
    # First component [1-9]…: rejects whole-box parenthesized decimal values such as
    # "(0.164)" in tables (28120_09a), which are never equation labels.
    r"^\s*\(\s*([1-9]\d{0,2}(?:\.\d{1,3}){1,3})\s*\)\s*$",
)

# Common English prose words — ≥2 hits means the label is a cross-reference
_PROSE_WORDS_RE = re.compile(
    r"\b(?:the|a(?:n|nd)?|is|are|was|were|be|been|have|has|had|do|does|did|"
    r"will|would|shall|should|may|might|must|can|could|"
    r"but|or|for|yet|so|because|since|although|while|where|when|which|"
    r"this|these|those|by|in|on|at|to|of|with|from|into|between|"
    r"shows?|given|see|above|below|follows?|use|using|if|let|"
    r"equation|equations|satisfies?|according|apply|applies)\b",
    re.IGNORECASE,
)


def _crop_ink_bands(crop: Image.Image) -> tuple[list[tuple[int, int]], int, int]:
    """Return (bands, height, width) where bands are (y0,y1) runs of ink rows.

    An "ink row" is a row whose dark-pixel count exceeds a small fraction of the width, so
    isolated specks do not create spurious bands.
    """
    gray = np.asarray(crop.convert("L"))
    h, w = gray.shape
    ink = gray < 200
    row_has_ink = ink.sum(axis=1) > max(2, int(0.01 * w))
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for y in range(h):
        if row_has_ink[y] and start is None:
            start = y
        elif not row_has_ink[y] and start is not None:
            bands.append((start, y - 1))
            start = None
    if start is not None:
        bands.append((start, h - 1))
    return bands, h, w


def _tighten_crop(
    crop: Image.Image,
    anchor_row: int | None = None,
    gap_px: float | None = None,
) -> Image.Image:
    """Trim prose lines that bled into a label-anchored equation crop.

    Using a horizontal ink projection, keep the content band that the equation occupies —
    identified by ``anchor_row`` (the label's vertical position within the crop, since the
    label sits on the equation's baseline) — plus neighbouring bands separated by only a
    small gap (fraction numerator/denominator, aligned multi-line systems). Bands separated
    by a gap larger than a text-line height (a prose paragraph) are dropped.

    ``anchor_row`` is essential: the equation is NOT always vertically centred (prose bleed
    can push it to an edge), so anchoring on the crop centre would keep the wrong band.
    Falls back to the centre only when no anchor is supplied.

    Conservative by design: only leading/trailing bands are removed, never interior ones,
    and if nothing would be trimmed the original crop is returned unchanged. Falls back to
    the untouched crop when numpy is unavailable.
    """
    if np is None:
        return crop
    bands, h, w = _crop_ink_bands(crop)
    if len(bands) <= 1 or h < 20 or w < 20:
        return crop

    # A prose separation is a gap comparable to a text-line height, whereas gaps *within* an
    # equation (fraction numerator/bar/denominator, sub/superscripts) are smaller. The label
    # height is the stable reference for a text line, so derive the threshold from it when
    # available. The old min-band-height heuristic split fractions (tiny sub/superscript
    # bands made the threshold too small), so it is only a fallback.
    if gap_px is not None:
        gap_threshold = max(10, int(gap_px))
    else:
        heights = [b1 - b0 + 1 for b0, b1 in bands]
        gap_threshold = max(8, int(0.8 * min(heights)))

    ref = anchor_row if anchor_row is not None else h // 2
    ref = max(0, min(h - 1, ref))
    anchor = min(
        range(len(bands)),
        key=lambda i: (
            0
            if bands[i][0] <= ref <= bands[i][1]
            else min(abs(bands[i][0] - ref), abs(bands[i][1] - ref))
        ),
    )

    lo = hi = anchor
    while lo - 1 >= 0 and (bands[lo][0] - bands[lo - 1][1] - 1) <= gap_threshold:
        lo -= 1
    while hi + 1 < len(bands) and (bands[hi + 1][0] - bands[hi][1] - 1) <= gap_threshold:
        hi += 1

    top, bot = bands[lo][0], bands[hi][1]
    if top == 0 and bot == h - 1:
        return crop  # nothing to trim

    pad = 4
    return crop.crop((0, max(0, top - pad), w, min(h, bot + pad + 1)))


def _save_crop(
    page_image: Image.Image,
    bbox_points: tuple[float, float, float, float],
    page_number: int,
    dpi: int,
    eq_id: str,
    crops_dir: Path,
    *,
    label_y_pts: float | None = None,
    label_height_pts: float | None = None,
) -> str:
    """Crop the equation region from the page image and save as PNG.

    When tightening is enabled the vertical window is first *expanded* by
    ``CROP_VEXPAND_FACTOR`` label-heights on each side so equations the label-anchored bbox
    under-captured (clipped numerators/denominators) are fully included; ``_tighten_crop``
    then trims the prose/whitespace the expansion pulls in, anchored on the label row
    (``label_y_pts``) so it keeps the equation band even when it is not vertically centred.

    Returns the path relative to the book output directory.
    """
    x0, y0, x1, y1 = bbox_points
    scale = dpi / config.PDF_POINTS_PER_INCH
    pad = config.CROP_PADDING_PX

    # Generous vertical expansion, paired with tightening (never expand without trimming),
    # so equations the bbox under-captured vertically are fully included and the excess is
    # trimmed. (Horizontal expansion was evaluated and rejected — it regressed neighbouring
    # crops without recovering the hard left-clip cases.)
    expand = 0
    if config.CROP_TIGHTEN_ENABLED and label_height_pts:
        expand = int(config.CROP_VEXPAND_FACTOR * label_height_pts * scale)

    px0 = max(0, int(x0 * scale) - pad)
    py0 = max(0, int(y0 * scale) - pad - expand)
    px1 = min(page_image.width, int(x1 * scale) + pad)
    py1 = min(page_image.height, int(y1 * scale) + pad + expand)

    if px1 <= px0 or py1 <= py0:
        logger.warning("invalid_crop eq_id=%s bbox=%s", eq_id, bbox_points)
        return ""

    crop = page_image.crop((px0, py0, px1, py1))
    if config.CROP_TIGHTEN_ENABLED:
        try:
            anchor_row = int(label_y_pts * scale) - py0 if label_y_pts is not None else None
            gap_px = (
                config.CROP_TIGHTEN_GAP_FACTOR * label_height_pts * scale
                if label_height_pts
                else None
            )
            crop = _tighten_crop(crop, anchor_row, gap_px)
        except Exception as exc:  # pragma: no cover - never fail a run over tightening
            logger.debug("crop_tighten_failed eq_id=%s error=%s", eq_id, exc)
    page_dir = crops_dir / f"page_{page_number:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    dest = page_dir / f"{eq_id}.png"
    crop.save(dest, format="PNG")

    return str(Path("crops") / f"page_{page_number:03d}" / f"{eq_id}.png")


def _extract_page_layout(pdf_path: Path) -> list[tuple[int, list[LTTextBox]]]:
    """Return (0-based page index, list of LTTextBox) for all pages."""
    rsrcmgr = PDFResourceManager()
    laparams = LAParams(line_overlap=0.5, char_margin=2.0, line_margin=0.5, word_margin=0.1)
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    results: list[tuple[int, list[LTTextBox]]] = []
    with open(pdf_path, "rb") as fh:
        for page_idx, page in enumerate(PDFPage.get_pages(fh)):
            try:
                interpreter.process_page(page)
                layout: LTPage = device.get_result()
                boxes = [el for el in layout if isinstance(el, LTTextBox)]
                results.append((page_idx, boxes))
            except Exception as exc:
                logger.debug("layout_extract_failed page=%d error=%s", page_idx, exc)
                results.append((page_idx, []))
    return results


def scan_equation_labels(pdf_path: Path) -> list[str]:
    """Return distinct definition labels found in the PDF, in document order.

    Uses the same matching and cross-reference rejection rules as layout
    detection so dashboard coverage is measured against the detector's actual
    label universe (including sub-equations such as ``3.9.1(a)`` and ``(b)``).
    """
    labels: list[str] = []
    seen: set[str] = set()
    # Two-phase: collect candidates per page first, then arbitrate paren-EOL labels
    # document-wide (they collide with numeric values, so they are only trusted when the
    # book's convention — see _allowed_nonexplicit_labels).
    pages_data: list[tuple[list[tuple[str, LTTextBox, str]], list[LTTextBox]]] = []
    arbitration_pool: list[tuple[str, bool]] = []
    for _page_idx, boxes in _extract_page_layout(Path(pdf_path)):
        formula_boxes: list[LTTextBox] = []
        candidates: list[tuple[str, LTTextBox, str]] = []
        for box in boxes:
            text = box.get_text().strip()
            found_any = False
            for m, explicit in _iter_label_matches(text):
                found_any = True
                if not _is_cross_reference_for_match(text, m):
                    label = _normalise_label(m.group(1))
                    candidates.append((label, box, "explicit" if explicit else "paren_eol"))
                    arbitration_pool.append((label, explicit))
            if found_any:
                if not any(c[1] is box for c in candidates):
                    formula_boxes.append(box)
                continue
            dpm = _PAREN_DOTTED_LABEL_RE.match(text) or _BRACKET_BOX_LABEL_RE.match(text)
            if dpm:
                # Whole-box "(4.12)" / "[1.1]" is trusted for itself but counts as paren-
                # convention evidence (False) — it must not suppress its own inline matches.
                candidates.append((dpm.group(1), box, "box_dotted"))
                arbitration_pool.append((dpm.group(1), False))
                continue
            paren = _PAREN_LABEL_RE.match(text)
            bare = _BARE_BOX_LABEL_RE.match(text) if config.BARE_NUMBER_LABELS_ENABLED else None
            if paren:
                candidates.append((paren.group(1), box, "box_paren"))
            elif bare:
                candidates.append((bare.group(1), box, "box_bare"))
                arbitration_pool.append((bare.group(1), False))
            else:
                formula_boxes.append(box)
        pages_data.append((candidates, formula_boxes))

    allowed_eol = _allowed_nonexplicit_labels(arbitration_pool)
    for candidates, formula_boxes in pages_data:
        for label, label_box, origin in candidates:
            if origin in ("paren_eol", "box_bare") and label not in allowed_eol:
                continue
            if (
                origin in ("paren_eol", "box_paren", "box_bare")
                and _formula_bbox_for_label(label_box, formula_boxes) is None
            ):
                continue
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


# A running-sentence continuation immediately after a label: a lowercase alphabetic word of
# length >=2 (e.g. "become", "becomes", "thus"). A genuine right-margin label is line-terminal
# (nothing meaningful follows it on the line), whereas a spelled-out prose cross-reference such
# as "equation 1.29 become" continues into a verb. Length >=2 and all-lowercase deliberately
# excludes trailing math (a single variable "x", or mixed-case tokens like "dV").
_RUNNING_SENTENCE_AFTER_RE = re.compile(r"^[a-z]{2,}\b")

# Sentence-initial words that INTRODUCE a reference to an existing equation ("From Eq. 3.9.1(a):",
# "Substituting equation 1.27 …"). When the line begins with one of these before the label, the
# label is a cross-reference, not an anchor. Deliberately excludes math connectives that legitimately
# precede a displayed equation on its own line ("where", "for", "or", "and", "thus"), which is why
# a curated introducer set is used rather than the broad _PROSE_WORDS_RE.
_REF_INTRO_WORDS = frozenset(
    {
        "from",
        "see",
        "using",
        "use",
        "substituting",
        "substitute",
        "recall",
        "combining",
        "comparing",
        "applying",
        "apply",
        "consider",
        "putting",
        "equating",
        "integrating",
        "differentiating",
        "solving",
        "dividing",
        "multiplying",
        "rearranging",
    }
)


def _is_cross_reference_for_match(text: str, m: re.Match) -> bool:
    """True when this specific label match is embedded in prose (cross-reference).

    Examines the same LINE as the label match.  Requires ≥2 prose-word hits
    rather than 1 so that mathematical qualifiers like "for", "or", "in" do
    not mis-classify equation definitions such as "f(x) = 1  for  x > 0
    Eq. 5.5.11".

    Additionally treats the label as a cross-reference when the line continues into a
    running sentence right after it (a lowercase verb continuation such as "…1.29 become"
    or "…1.65 thus becomes"). Image-based math books spell out "equation N.N" mid-sentence,
    which _LABEL_RE matches; those references are line-internal, not line-terminal margin
    labels, so this rule keeps them out of the label universe.
    """
    label_start, label_end = m.start(), m.end()

    line_start = text.rfind("\n", 0, label_start)
    line_start = line_start + 1 if line_start >= 0 else 0
    line_end = text.find("\n", label_end)
    line_end = line_end if line_end >= 0 else len(text)

    before_on_line = text[line_start:label_start].strip()
    after_on_line = text[label_end:line_end].strip()

    if before_on_line and len(_PROSE_WORDS_RE.findall(before_on_line)) >= 2:
        return True
    if after_on_line and len(_PROSE_WORDS_RE.findall(after_on_line)) >= 2:
        return True
    if after_on_line and _RUNNING_SENTENCE_AFTER_RE.match(after_on_line):
        return True
    first_before = re.match(r"[A-Za-z]+", before_on_line)
    if first_before and first_before.group(0).lower() in _REF_INTRO_WORDS:
        return True
    return False


def _is_cross_reference(text: str) -> bool:
    """True when 'Eq. X.X.X' is embedded in a prose sentence rather than anchoring it."""
    m = _LABEL_RE.search(text)
    if not m:
        return False
    return _is_cross_reference_for_match(text, m)


def _find_labeled_equations(
    pdf_path: Path,
    pages: list[RenderedPage],
    crops_dir: Path,
) -> list[EquationRegion]:
    """Detect equations via 'Eq. X.X.X' label scan.

    Only non-prose label occurrences are accepted as equation anchors.
    Each label number is deduplicated across the whole document.
    """
    page_map = {rp.page_number: rp for rp in pages}
    page_layouts = _extract_page_layout(pdf_path)
    regions: list[EquationRegion] = []
    eq_counter = 0
    seen_labels: set[str] = set()

    for page_idx, boxes in page_layouts:
        page_number = page_idx + 1
        rp = page_map.get(page_number)
        if rp is None:
            continue

        label_boxes: list[tuple[str, LTTextBox, bool]] = []
        formula_boxes: list[LTTextBox] = []

        for box in boxes:
            text = box.get_text().strip()
            found_any = False
            all_cross_refs = True
            for m, explicit in _iter_label_matches(text):
                # Legacy path is single-pass per page and cannot arbitrate paren-EOL
                # labels document-wide, so it accepts explicit conventions only; the
                # hybrid path handles "(5.2)"-convention books.
                if not explicit:
                    continue
                found_any = True
                if not _is_cross_reference_for_match(text, m):
                    all_cross_refs = False
                    label_str = _normalise_label(m.group(1))
                    label_boxes.append((label_str, box, True))
            if found_any:
                if all_cross_refs:
                    formula_boxes.append(box)
                continue
            dpm = _PAREN_DOTTED_LABEL_RE.match(text)
            if dpm:
                label_boxes.append((dpm.group(1), box, True))
                continue
            pm = _PAREN_LABEL_RE.match(text)
            if pm:
                label_boxes.append((pm.group(1), box, False))
            else:
                formula_boxes.append(box)

        for label_str, label_box, explicit_eq_label in label_boxes:
            if label_str in seen_labels:
                logger.debug("label_duplicate_skipped label=%s page=%d", label_str, page_number)
                continue
            # Anchor on the label's own text line, not the whole box: pdfminer sometimes merges
            # a right-margin label with its adjacent prose paragraph, which would otherwise put
            # the label's left edge at the paragraph and send the crop to the wrong region.
            anchor = _BBoxAnchor(_label_anchor_bbox(label_box, label_str))
            lx0, ly0, lx1, ly1 = anchor.bbox

            formula_bbox = _formula_bbox_for_label(anchor, formula_boxes)
            if formula_bbox is not None:
                fx0, fy0, fx1, fy1 = formula_bbox
            elif not explicit_eq_label:
                logger.debug(
                    "parenthesized_list_marker_skipped label=%s page=%d",
                    label_str,
                    page_number,
                )
                continue
            else:
                fx0, fy0, fx1, fy1 = _image_formula_bbox_for_label(anchor)

            seen_labels.add(label_str)

            page_height_pts = rp.height_px / (rp.dpi / config.PDF_POINTS_PER_INCH)
            bbox = (fx0, page_height_pts - fy1, fx1, page_height_pts - fy0)

            # Label geometry (top-left points) for expansion + tightening anchor: the label
            # sits on the equation's baseline, so its row identifies the equation band.
            label_y_pts = page_height_pts - (ly0 + ly1) / 2.0
            label_height_pts = max(ly1 - ly0, 8.0)

            safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label_str).strip("_")
            eq_id = f"eq_{eq_counter}_p{page_number}_{safe_label}"
            eq_counter += 1

            crop_rel = _save_crop(
                rp.load_image(),
                bbox,
                page_number,
                rp.dpi,
                eq_id,
                crops_dir,
                label_y_pts=label_y_pts,
                label_height_pts=label_height_pts,
            )
            regions.append(
                EquationRegion(
                    page_number=page_number,
                    equation_id=eq_id,
                    label=label_str,
                    bbox=bbox,
                    detection_method="label",
                    crop_path=crop_rel or None,
                )
            )

    return regions


def _find_ml_equations(
    pdf_path: Path,
    pages: list[RenderedPage],
    crops_dir: Path,
) -> list[EquationRegion]:
    """Detect equations using Docling layout analysis."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        logger.warning("docling not installed; ML equation detection unavailable")
        return []

    page_map = {rp.page_number: rp for rp in pages}
    regions: list[EquationRegion] = []
    eq_counter = 0

    try:
        _raw_artifacts_path = os.getenv("DOCLING_ARTIFACTS_PATH", "").strip()
        if _raw_artifacts_path and not Path(_raw_artifacts_path).is_dir():
            logger.warning(
                "docling_artifacts_path_missing path=%s; falling back to HF cache",
                _raw_artifacts_path,
            )
            _raw_artifacts_path = ""
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=False,
            artifacts_path=_raw_artifacts_path or None,
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        result = converter.convert(str(pdf_path))
        doc = result.document

        for item, _ in doc.iterate_items():
            label = getattr(item, "label", "")
            if str(label).lower() not in {"formula", "equation"}:
                continue
            prov = getattr(item, "prov", [])
            if not prov:
                continue
            prov_item = prov[0]
            page_number = getattr(prov_item, "page_no", 0)
            bbox_obj = getattr(prov_item, "bbox", None)
            if bbox_obj is None or page_number == 0:
                continue

            rp = page_map.get(page_number)
            if rp is None:
                continue

            # Docling bboxes are bottom-left origin; _save_crop expects top-left points.
            page = doc.pages.get(page_number) if hasattr(doc.pages, "get") else None
            if page is None:
                continue
            tl = bbox_obj.to_top_left_origin(float(page.size.height))
            bbox = (
                float(tl.l),
                float(tl.t),
                float(tl.r),
                float(tl.b),
            )
            eq_id = f"eq_{eq_counter}_p{page_number}_ml"
            eq_counter += 1

            crop_rel = _save_crop(rp.load_image(), bbox, page_number, rp.dpi, eq_id, crops_dir)
            regions.append(
                EquationRegion(
                    page_number=page_number,
                    equation_id=eq_id,
                    label=None,
                    bbox=bbox,
                    detection_method="ml",
                    crop_path=crop_rel or None,
                )
            )
    except Exception as exc:
        logger.error("ml_detection_failed error=%s", exc)

    return regions


def _detect_docling_regions(
    pdf_path: Path, pages: list[RenderedPage]
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Run Docling layout detection and return formula bboxes per page (top-left points).

    Detection ONLY — ``do_formula_enrichment`` stays off so the CPU-bound CodeFormulaV2 LaTeX
    decode never runs (it is 20–40 min/chapter on CPU; recognition stays on the VL/judge path).
    Returns ``{page_no(1-based): [(l, t, r, b), …]}`` with coordinates in PDF points, top-left
    origin — the same space ``_save_crop``/``EquationRegion.bbox`` use.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        logger.warning("docling not installed; hybrid detection unavailable")
        return {}

    out: dict[int, list[tuple[float, float, float, float]]] = {}
    try:
        raw_artifacts = os.getenv("DOCLING_ARTIFACTS_PATH", "").strip()
        if raw_artifacts and not Path(raw_artifacts).is_dir():
            logger.warning("docling_artifacts_path_missing path=%s; using HF cache", raw_artifacts)
            raw_artifacts = ""
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=False,
            do_formula_enrichment=False,
            artifacts_path=raw_artifacts or None,
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        doc = converter.convert(str(pdf_path)).document
        for item, _ in doc.iterate_items():
            if str(getattr(item, "label", "")).lower() not in {"formula", "equation"}:
                continue
            prov = getattr(item, "prov", []) or []
            if not prov:
                continue
            page_no = getattr(prov[0], "page_no", 0)
            bbox_obj = getattr(prov[0], "bbox", None)
            page = doc.pages.get(page_no) if hasattr(doc.pages, "get") else None
            if bbox_obj is None or page_no == 0 or page is None:
                continue
            ph = float(page.size.height)
            tl = bbox_obj.to_top_left_origin(ph)
            out.setdefault(page_no, []).append((float(tl.l), float(tl.t), float(tl.r), float(tl.b)))
    except Exception as exc:
        logger.error("docling_detect_failed error=%s", exc)
    return out


def _associate_labels_to_regions(
    labels: list[tuple[str, tuple[float, float, float, float]]],
    regions: list[tuple[float, float, float, float]],
) -> dict[str, tuple[float, float, float, float]]:
    """Map each label to its Docling crop box (top-left points); labels with none are absent.

    Docling frequently emits a labeled equation as SEVERAL fragments (LHS, numerator, bar,
    denominator) or a single thin baseline strip. Picking one region by nearest edge clips the
    equation (measured: 22/56 crops came out ~6pt tall). So instead, for each label we take the
    UNION of every fragment on its row, which reassembles the whole equation; a tall region
    shared by stacked labels is split vertically first so each label keeps only its band. A
    minimum height derived from the label height guards the remaining thin single-line cases.
    """
    label_info: dict[str, tuple[float, float, float, float]] = {}  # label -> (lx0, cy, h, w)
    reg_labels: dict[int, list[str]] = {i: [] for i in range(len(regions))}
    for label, (lx0, lt, lx1, lb) in labels:
        cy = (lt + lb) / 2.0
        h = max(lb - lt, 8.0)
        label_info[label] = (lx0, cy, h, max(lx1 - lx0, 0.0))
        for i, (rl, rt, rr, rb) in enumerate(regions):
            if rl >= lx0 + h:  # region must start left of the margin label
                continue
            if rt - 1.5 * h <= cy <= rb + 1.5 * h:  # region row overlaps the label baseline
                reg_labels[i].append(label)

    # Each region contributes to the label(s) whose row it overlaps; a region shared by several
    # stacked labels is split vertically (midpoints between adjacent label rows).
    parts: dict[str, list[tuple[float, float, float, float]]] = {lbl: [] for lbl, _ in labels}
    for i, (rl, rt, rr, rb) in enumerate(regions):
        labs = reg_labels[i]
        if not labs:
            continue
        if len(labs) == 1:
            parts[labs[0]].append((rl, rt, rr, rb))
            continue
        labs.sort(key=lambda L: label_info[L][1])
        cys = [label_info[L][1] for L in labs]
        for j, L in enumerate(labs):
            bt = rt if j == 0 else (cys[j - 1] + cys[j]) / 2.0
            bb = rb if j == len(labs) - 1 else (cys[j] + cys[j + 1]) / 2.0
            parts[L].append((rl, bt, rr, bb))

    out: dict[str, tuple[float, float, float, float]] = {}
    for label, (lx0, cy, h, lw) in label_info.items():
        ps = parts[label]
        if not ps:
            continue
        x0 = min(p[0] for p in ps)
        x1 = max(p[2] for p in ps)
        y0 = min(p[1] for p in ps)
        y1 = max(p[3] for p in ps)
        # Label-only association: the union never extends left of the label's own left edge,
        # i.e. Docling only detected the printed label text (typical when the equation itself
        # is an image with no separate formula region). Cropping it would ship a picture of
        # "Eq. 5.5.1" — discard so the reconstruction fallback supplies a real crop instead.
        # Only meaningful for a NARROW standalone-label anchor: when the label is embedded in
        # a prose line the anchor spans the whole line, its left edge is the line start, and
        # this test would wrongly discard a legitimate association (measured: 39896_02 2-2).
        if lw <= 6.0 * h and x0 >= lx0 - 2.0 * h:
            continue
        # Floor so thin/partial detections keep vertical context: fractions/integral limits
        # need ~2 label-heights (1.6 measured too small on degraded scans — 39896_02 2-27).
        min_h = 2.0 * h
        if (y1 - y0) < min_h:
            y0 = min(y0, cy - min_h / 2.0)
            y1 = max(y1, cy + min_h / 2.0)
        out[label] = (x0, y0, x1, y1)
    return out


def _save_crop_simple(
    page_image: Image.Image,
    bbox_points: tuple[float, float, float, float],
    page_number: int,
    dpi: int,
    eq_id: str,
    crops_dir: Path,
) -> str:
    """Crop a model-detected region: fixed fractional pad, NO ink-projection tightening.

    A detector bbox is already tight and prose-free, so the vertical-expansion / band-trimming
    heuristics (`_tighten_crop`, `CROP_VEXPAND_FACTOR`) are unnecessary and off by construction.
    """
    x0, y0, x1, y1 = bbox_points
    x0, x1 = min(x0, x1), max(x0, x1)  # normalise ordering defensively
    y0, y1 = min(y0, y1), max(y0, y1)
    scale = dpi / config.PDF_POINTS_PER_INCH
    px0, py0, px1, py1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
    pf = config.CROP_PAD_FRAC
    # Vertical pad floor recovers clipped integral limits / fraction denominators on thin
    # detector strips; horizontal stays small — sideways clips are detection gaps, and a wide
    # x-pad risks bleeding the neighbouring column in two-column layouts.
    dx = max((px1 - px0) * pf, 2.0 * scale)
    dy = max((py1 - py0) * pf, config.CROP_MIN_PAD_PTS * scale)
    cx0, cy0 = max(0, int(px0 - dx)), max(0, int(py0 - dy))
    cx1 = min(page_image.width, int(px1 + dx))
    cy1 = min(page_image.height, int(py1 + dy))
    if cx1 <= cx0 or cy1 <= cy0:
        logger.warning("invalid_crop eq_id=%s bbox=%s", eq_id, bbox_points)
        return ""
    page_dir = crops_dir / f"page_{page_number:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    page_image.crop((cx0, cy0, cx1, cy1)).save(page_dir / f"{eq_id}.png", format="PNG")
    return str(Path("crops") / f"page_{page_number:03d}" / f"{eq_id}.png")


def _find_hybrid_equations(
    pdf_path: Path,
    pages: list[RenderedPage],
    crops_dir: Path,
    docling_by_page: dict[int, list[tuple[float, float, float, float]]],
) -> list[EquationRegion]:
    """Detect labeled equations, cropping from the Docling box when available.

    Scoping/numbering come from the proven label scan (document-wide dedup via ``seen_labels``);
    the crop geometry comes from Docling (tight, no tuning) when a region associates with the
    label, else falls back to the legacy reconstruction bbox + ``_save_crop`` for the rare miss.
    """
    page_map = {rp.page_number: rp for rp in pages}
    page_layouts = _extract_page_layout(pdf_path)
    seen_labels: set[str] = set()
    arbitration_pool: list[tuple[str, bool]] = []

    # Pass 1 — gather deduped label anchors + formula boxes (for reconstruction fallback) per page.
    per_page: dict[int, dict[str, Any]] = {}
    for page_idx, boxes in page_layouts:
        page_number = page_idx + 1
        rp = page_map.get(page_number)
        if rp is None:
            continue
        ph = rp.height_px / (rp.dpi / config.PDF_POINTS_PER_INCH)

        label_boxes: list[tuple[str, LTTextBox, str]] = []
        formula_boxes: list[LTTextBox] = []
        for box in boxes:
            text = box.get_text().strip()
            found_any = False
            all_cross_refs = True
            for m, explicit in _iter_label_matches(text):
                found_any = True
                if not _is_cross_reference_for_match(text, m):
                    all_cross_refs = False
                    label_boxes.append(
                        (
                            _normalise_label(m.group(1)),
                            box,
                            "explicit" if explicit else "paren_eol",
                        )
                    )
            if found_any:
                if all_cross_refs:
                    formula_boxes.append(box)
                continue
            dpm = _PAREN_DOTTED_LABEL_RE.match(text) or _BRACKET_BOX_LABEL_RE.match(text)
            if dpm:
                # Trusted for itself, but paren-convention evidence for arbitration.
                label_boxes.append((dpm.group(1), box, "box_dotted"))
                continue
            pm = _PAREN_LABEL_RE.match(text)
            bare = _BARE_BOX_LABEL_RE.match(text) if config.BARE_NUMBER_LABELS_ENABLED else None
            if pm:
                label_boxes.append((pm.group(1), box, "box_paren"))
            elif bare:
                label_boxes.append((bare.group(1), box, "box_bare"))
            else:
                formula_boxes.append(box)

        anchors: list[tuple[str, tuple[float, float, float, float], str, LTTextBox]] = []
        for label_str, label_box, origin in label_boxes:
            if label_str in seen_labels:
                continue
            seen_labels.add(label_str)
            arbitration_pool.append((label_str, origin == "explicit"))
            ax0, ay0, ax1, ay1 = _label_anchor_bbox(label_box, label_str)
            anchors.append((label_str, (ax0, ph - ay1, ax1, ph - ay0), origin, label_box))
        per_page[page_number] = {"anchors": anchors, "formula_boxes": formula_boxes, "ph": ph}

    # Arbitrate paren-EOL labels document-wide (they collide with numeric values).
    allowed_eol = _allowed_nonexplicit_labels(arbitration_pool)

    # Pass 2 — per page: associate labels ↔ Docling regions (share/split), then crop.
    regions_out: list[EquationRegion] = []
    eq_counter = 0
    source_counts = {"docling": 0, "reconstruction": 0}
    for page_number in sorted(per_page):
        info = per_page[page_number]
        rp = page_map[page_number]
        ph = info["ph"]
        page_anchors = [
            a
            for a in info["anchors"]
            if a[2] not in ("paren_eol", "box_bare") or a[0] in allowed_eol
        ]
        dl = _associate_labels_to_regions(
            [(lbl, anc) for (lbl, anc, _o, _b) in page_anchors],
            docling_by_page.get(page_number, []),
        )
        for label_str, _anchor_tl, origin, label_box in page_anchors:
            explicit = origin in ("explicit", "box_dotted")
            source = "docling"
            bbox = dl.get(label_str)
            if bbox is not None and (bbox[3] - bbox[1] < 3.0 or bbox[2] - bbox[0] < 3.0):
                # Degenerate association band (label row fell outside the region, or a stacked
                # split collapsed) — treat as a miss so reconstruction supplies a real crop.
                bbox = None
            if bbox is None:
                # Reconstruction fallback (legacy helpers work in bottom-left points).
                anchor = _BBoxAnchor(_label_anchor_bbox(label_box, label_str))
                fb = _formula_bbox_for_label(anchor, info["formula_boxes"])
                if fb is None and explicit:
                    fb = _image_formula_bbox_for_label(anchor)
                if fb is None:
                    logger.debug("hybrid_no_region_or_reconstruction label=%s", label_str)
                    continue
                fx0, fy0, fx1, fy1 = fb
                bbox = (fx0, ph - fy1, fx1, ph - fy0)
                source = "reconstruction"

            safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label_str).strip("_")
            eq_id = f"eq_{eq_counter}_p{page_number}_{safe_label}"
            eq_counter += 1
            source_counts[source] += 1

            if source == "docling":
                crop_rel = _save_crop_simple(
                    rp.load_image(), bbox, page_number, rp.dpi, eq_id, crops_dir
                )
            else:
                lx0, ly0, lx1, ly1 = _label_anchor_bbox(label_box, label_str)
                crop_rel = _save_crop(
                    rp.load_image(),
                    bbox,
                    page_number,
                    rp.dpi,
                    eq_id,
                    crops_dir,
                    label_y_pts=ph - (ly0 + ly1) / 2.0,
                    label_height_pts=max(ly1 - ly0, 8.0),
                )
            regions_out.append(
                EquationRegion(
                    page_number=page_number,
                    equation_id=eq_id,
                    label=label_str,
                    bbox=bbox,
                    detection_method="label",
                    crop_path=crop_rel or None,
                )
            )
    logger.info(
        "hybrid_crop_sources docling=%d reconstruction=%d",
        source_counts["docling"],
        source_counts["reconstruction"],
    )
    return regions_out


def _merge_row_fragments(
    boxes: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Union Docling formula fragments that belong to one display equation.

    Docling emits one equation as several boxes (LHS / numerator / bar / continuation line).
    Two boxes are merged when their vertical bands overlap by >=50% of the smaller band —
    the same row test the labeled path's association uses. After merging, tiny leftovers
    (inline-math slivers far smaller than the page's typical display equation) are dropped.
    """
    if not boxes:
        return []
    merged: list[list[float]] = []
    for b in sorted(boxes, key=lambda b: (b[1], b[0])):
        for m in merged:
            overlap = min(m[3], b[3]) - max(m[1], b[1])
            smaller = min(m[3] - m[1], b[3] - b[1])
            if smaller > 0 and overlap / smaller >= 0.5:
                m[0], m[1] = min(m[0], b[0]), min(m[1], b[1])
                m[2], m[3] = max(m[2], b[2]), max(m[3], b[3])
                break
        else:
            merged.append(list(b))
    heights = sorted(m[3] - m[1] for m in merged)
    med_h = heights[len(heights) // 2]
    widths = sorted(m[2] - m[0] for m in merged)
    med_w = widths[len(widths) // 2]
    kept = [
        tuple(m)
        for m in merged
        if not ((m[3] - m[1]) < 0.6 * med_h and (m[2] - m[0]) < 0.3 * med_w)
    ]
    return kept or [tuple(m) for m in merged]


def _regions_from_docling(
    docling_by_page: dict[int, list[tuple[float, float, float, float]]],
    pages: list[RenderedPage],
    crops_dir: Path,
) -> list[EquationRegion]:
    """Unlabeled-document path: crop merged Docling formula regions (no label scoping)."""
    page_map = {rp.page_number: rp for rp in pages}
    regions: list[EquationRegion] = []
    eq_counter = 0
    for page_number in sorted(docling_by_page):
        rp = page_map.get(page_number)
        if rp is None:
            continue
        for bbox in _merge_row_fragments(docling_by_page[page_number]):
            eq_id = f"eq_{eq_counter}_p{page_number}_ml"
            eq_counter += 1
            crop_rel = _save_crop_simple(
                rp.load_image(), bbox, page_number, rp.dpi, eq_id, crops_dir
            )
            regions.append(
                EquationRegion(
                    page_number=page_number,
                    equation_id=eq_id,
                    label=None,
                    bbox=bbox,
                    detection_method="ml",
                    crop_path=crop_rel or None,
                )
            )
    return regions


def _scan_page_media(pdf_path: Path) -> dict[int, tuple[int, str]]:
    """Return ``{page_no(1-based): (embedded_image_count, page_text)}``.

    Counts LTImage elements (recursing into LTFigure), which is how books that render display
    equations as rasters expose them — many small images per page — versus text/vector math
    (few or none). One pdfminer pass; degrades to an empty map on failure.
    """
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTFigure, LTImage, LTTextContainer

    out: dict[int, tuple[int, str]] = {}
    try:
        for page_no, page in enumerate(extract_pages(str(pdf_path)), start=1):
            n_images = 0
            texts: list[str] = []

            def _walk(container) -> None:
                nonlocal n_images
                for el in container:
                    if isinstance(el, LTImage):
                        n_images += 1
                    elif isinstance(el, LTFigure):
                        _walk(el)
                    elif isinstance(el, LTTextContainer):
                        texts.append(el.get_text())

            _walk(page)
            out[page_no] = (n_images, "".join(texts))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("scan_page_media_failed error=%s", exc)
    return out


def _flag_image_math(media: dict[int, tuple[int, str]]) -> set[int]:
    """Return the set of 'image-math' page numbers, or empty if the document is not image-math.

    Discriminator is the per-page embedded-image COUNT: a page whose display equations (and
    their margin numbers) are rasterised carries many small images, whereas text/vector math
    carries few or none. The math-symbol text cue is deliberately NOT used per page — on exactly
    these pages the math lives in the images, so the text layer is pure prose and would fail it.

    Instead a cheap DOCUMENT-level math-presence gate keeps ordinary image-heavy PDFs (photo
    albums, scanned figures) from tripping the VLM path: the document must mention equations /
    contain math on several pages. When both hold and image-dense pages are a meaningful share
    (``IMAGE_MATH_DOC_PAGE_FRACTION``) of the content pages, every image-dense page is flagged.
    """
    min_images = int(getattr(config, "IMAGE_MATH_MIN_IMAGES_PER_PAGE", 6))
    frac_threshold = float(getattr(config, "IMAGE_MATH_DOC_PAGE_FRACTION", 0.30))

    content_pages = [p for p, (n, _t) in media.items() if n > 0]
    if not content_pages:
        return set()

    # Document-level math presence: several pages reference equations or carry math symbols.
    math_pages = sum(
        1
        for _p, (_n, text) in media.items()
        if _EQUATION_REF_RE.search(text) or _MATH_SYMBOL.search(text)
    )
    if math_pages < 3:
        return set()

    image_dense = {p for p, (n, _t) in media.items() if n >= min_images}
    if len(image_dense) / len(content_pages) < frac_threshold:
        return set()
    return image_dense


# "equation 12" / "Eqn. 3" style math cues in page text (used only as an image-math signal,
# NOT as a label — label matching stays with _LABEL_RE and its cross-reference guards).
_EQUATION_REF_RE = re.compile(r"\bEq(?:uation|n)?s?\.?\s*\d", re.IGNORECASE)


def _vertical_overlap_pts(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Fraction of the shorter box's height that overlaps vertically (top-left points)."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    shorter = min(a[3] - a[1], b[3] - b[1])
    return overlap / shorter if shorter > 0 else 0.0


def _recover_image_math_pages(
    pages: list[RenderedPage],
    crops_dir: Path,
    image_math_pages: set[int],
    existing: list[EquationRegion],
) -> list[EquationRegion]:
    """VLM-enumerate equations on image-math pages, adding those not already captured.

    Docling is empirically unreliable on exactly these pages (validated: 0 regions on the
    image-equation pages), so the VLM is the primary recovery here. Each VLM equation is
    converted from normalised page fractions → PDF points, cropped via ``_save_crop_simple``,
    and carries the VLM transcription as ``seed_latex``. Deduplicated against ``existing``
    regions by matching label or significant vertical overlap so nothing is double-counted.
    """
    try:
        from equation_extraction_pipeline.extraction.gpt_judge import (
            extract_page_equations,
            gpt_judge_available,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("image_math_recovery_import_failed error=%s", exc)
        return []
    if not gpt_judge_available():
        logger.info("image_math_recovery_skipped reason=gpt_judge_unconfigured")
        return []

    page_map = {rp.page_number: rp for rp in pages}
    existing_by_page: dict[int, list[EquationRegion]] = {}
    for r in existing:
        existing_by_page.setdefault(r.page_number, []).append(r)

    new_regions: list[EquationRegion] = []
    counter = 0
    for page_number in sorted(image_math_pages):
        rp = page_map.get(page_number)
        if rp is None:
            continue
        ph_pts = rp.height_px / (rp.dpi / config.PDF_POINTS_PER_INCH)
        pw_pts = rp.width_px / (rp.dpi / config.PDF_POINTS_PER_INCH)
        page_existing = existing_by_page.get(page_number, [])
        existing_labels = {_normalise_label(r.label) for r in page_existing if r.label}

        for eq in extract_page_equations(rp.load_image(), page_number=page_number, mode="labeled"):
            label = eq.get("label")
            norm_label = _normalise_label(label) if label else None
            if norm_label and norm_label in existing_labels:
                continue  # already captured by the label/docling path

            frac = eq.get("bbox_frac")
            bbox_pts: tuple[float, float, float, float] | None = None
            if frac is not None:
                fx0, fy0, fx1, fy1 = frac
                bbox_pts = (fx0 * pw_pts, fy0 * ph_pts, fx1 * pw_pts, fy1 * ph_pts)
                if any(_vertical_overlap_pts(bbox_pts, r.bbox) >= 0.5 for r in page_existing):
                    continue  # spatially overlaps an already-detected region
                # VLM pixel coordinates are approximate (they can sit ~a line off), so give the
                # crop a half-box-height of vertical slack on each side to keep the target
                # equation in-frame. The judge tolerates a neighbouring line clipped at a border.
                box_h = bbox_pts[3] - bbox_pts[1]
                slack = 0.5 * box_h
                bbox_pts = (
                    bbox_pts[0],
                    max(0.0, bbox_pts[1] - slack),
                    bbox_pts[2],
                    min(ph_pts, bbox_pts[3] + slack),
                )

            safe_label = (
                re.sub(r"[^A-Za-z0-9]+", "_", norm_label).strip("_") if norm_label else "vlm"
            )
            eq_id = f"eq_{counter}_p{page_number}_{safe_label}"
            counter += 1

            crop_rel: str | None = None
            if bbox_pts is not None:
                crop_rel = (
                    _save_crop_simple(
                        rp.load_image(), bbox_pts, page_number, rp.dpi, eq_id, crops_dir
                    )
                    or None
                )
            new_regions.append(
                EquationRegion(
                    page_number=page_number,
                    equation_id=eq_id,
                    label=label if label else None,
                    bbox=bbox_pts if bbox_pts is not None else (0.0, 0.0, pw_pts, ph_pts),
                    detection_method="vlm",
                    crop_path=crop_rel,
                    seed_latex=eq.get("latex") or None,
                )
            )
    logger.info(
        "image_math_recovery pages=%d recovered=%d", len(image_math_pages), len(new_regions)
    )
    return new_regions


def detect_equations(
    pdf_path: Path,
    pages: list[RenderedPage],
    classification: ClassificationResult,
    output_dir: Path,
    *,
    detection_meta: dict[str, Any] | None = None,
) -> list[EquationRegion]:
    """Detect all equation regions and save crop images.

    Detector selected by ``config.EQUATION_DETECTOR``:
      * ``hybrid`` (default) — Docling supplies tight crop boxes; the label scan scopes/numbers;
        reconstruction is the fallback for equations Docling misses. Unlabeled docs fall back to
        cropping every Docling region.
      * ``label`` — legacy label-scan + geometry-reconstruction crops.
      * ``docling`` — Docling regions only, no label scoping.

    Returns regions sorted by (page_number, y-coordinate). Crops are written to
    ``<output_dir>/crops/page_NNN/<eq_id>.png``.
    """
    pdf_path = Path(pdf_path)
    crops_dir = output_dir / "crops"
    detector = getattr(config, "EQUATION_DETECTOR", "hybrid")
    logger.info("detecting equations pdf=%s detector=%s", pdf_path.name, detector)

    regions: list[EquationRegion] = []
    if detector in ("hybrid", "docling"):
        docling_by_page = _detect_docling_regions(pdf_path, pages)
        n_regions = sum(len(v) for v in docling_by_page.values())
        logger.info("docling_detection regions=%d", n_regions)

        if detector == "docling":
            regions = _regions_from_docling(docling_by_page, pages, crops_dir)
        else:
            regions = _find_hybrid_equations(pdf_path, pages, crops_dir, docling_by_page)
            if regions:
                logger.info("hybrid_detection found=%d equations (labeled)", len(regions))
            elif docling_by_page:
                logger.info("no labels; cropping all %d docling regions", n_regions)
                regions = _regions_from_docling(docling_by_page, pages, crops_dir)

    if not regions:
        # Legacy path (also the `label` detector's primary path, and the fall-through when the
        # Docling/hybrid path yields nothing).
        regions = _find_labeled_equations(pdf_path, pages, crops_dir)
        if regions:
            logger.info("label_detection found=%d equations", len(regions))
        else:
            logger.info("no labels found; falling back to ML detection")
            regions = _find_ml_equations(pdf_path, pages, crops_dir)
            logger.info("ml_detection found=%d equations", len(regions))

    # Image-based math recovery: equations embedded as rasters are invisible to the text/label
    # scan and unreliable for Docling. On flagged pages a VLM enumerates them from the page image
    # and the results are unioned in (deduplicated against what the label/Docling path found).
    image_math_pages: set[int] = set()
    if getattr(config, "IMAGE_MATH_FALLBACK_ENABLED", True):
        media = _scan_page_media(pdf_path)
        image_math_pages = _flag_image_math(media)
        if image_math_pages:
            logger.info("image_math_flagged pages=%d", len(image_math_pages))
            recovered = _recover_image_math_pages(pages, crops_dir, image_math_pages, regions)
            regions = regions + recovered

    if detection_meta is not None:
        detection_meta["image_math"] = bool(image_math_pages)
        detection_meta["image_math_pages"] = sorted(image_math_pages)

    regions.sort(key=lambda r: (r.page_number, r.bbox[1]))
    return regions
