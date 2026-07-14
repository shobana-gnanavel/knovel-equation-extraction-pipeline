"""Knovel equation toolkit — extract and validate in one script.

Modes
-----
auto     (default) Scan for Eq. X.X.X labels; use labeled mode if found, else
         fall back to the full ML pipeline.
labeled  Label-based extraction — fast, no ML.  Works on PDFs that
         mark equations with right-margin labels "Eq. 12.3.4".
full     Full ML pipeline: ingestion → layout → pix2tex recognition.

Common workflows
----------------
Auto-extract + validate a single PDF::

    python scripts/equations.py --pdf data/input/book.pdf --validate

Force labeled mode (PDF has Eq. X.X.X labels)::

    python scripts/equations.py --pdf data/input/28120_12.pdf --mode labeled --validate

Force full ML pipeline::

    python scripts/equations.py --pdf data/input/HTO_Module1.pdf --mode full --validate

Batch extraction (full pipeline only)::

    python scripts/equations.py --input-dir data/input/ --mode full

Validate an existing sidecar without re-extracting::

    python scripts/equations.py --pdf data/input/book.pdf --validate-only

Open the HTML report immediately after validation::

    python scripts/equations.py --pdf data/input/book.pdf --validate --open-report
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

# Shared detection primitives — same patterns used by the production pipeline.
from equation_extraction.detection import (  # noqa: E402
    extract_label_number as _extract_label_number,
)
from equation_extraction.detection import (
    is_isolated_equation_label as _is_label,
)
from equation_extraction.formula_detector import (
    score_formula_candidate as _score_formula,  # noqa: E402
)

# ── Shared constants ───────────────────────────────────────────────────────────
DATA_DIR       = PROJECT_ROOT / "data"
OUTPUT_DIR     = DATA_DIR / "output"
VALIDATION_DIR = DATA_DIR / "validation"

# Label detection is provided by the shared detection module so the script uses
# the same patterns as the production pipeline extractor.
# _LABEL_RE was removed; use is_isolated_equation_label() and extract_label_number()
# from equation_extraction.detection instead.
_UNITS_RE    = re.compile(r"\b(lbs?|in|ft|psi|ksi|kip|lb|N|kN|MPa|GPa|kg|m|mm|kPa)\b", re.IGNORECASE)


def _confidence_score(eq: dict) -> float:
    """Combined confidence exported with each equation."""
    rec = float(eq.get("recognition_confidence") or 0.0)
    cls = float(eq.get("classification_confidence") or 0.0)
    return round((rec + cls) / 2.0, 3)


# Require at least one digit so "PR", "AE" (variable names) are not confused with molecules.
_CHEM_RE     = re.compile(r"\b(?:[A-Z][a-z]?\d+)+(?:[A-Z][a-z]?\d*)+\b")
_REACTION_RE = re.compile(r"[→⇌⟶]|-{1,3}>|<-{1,3}|={1,2}>")

# LaTeX quality detection patterns
_LABEL_IN_LATEX_RE  = re.compile(r"\bEq\.\s*\d+\.\d+", re.IGNORECASE)
_UNICODE_MATH_CHARS = frozenset("≤≥×÷±∑∫∂∞αβγδεζηθικλμνξπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΠΡΣΤΥΦΧΨΩ√≠≈")

# OCR substitution rules applied before VLM processing.
# Rules are applied in order; place broader substitutions after more specific ones.
_OCR_FIX: list[tuple[re.Pattern, str]] = [
    # ¼ (U+00BC) is used as the equals sign '=' in many Knovel engineering PDFs
    # where the font encoding maps the equals glyph to U+00BC.
    (re.compile(r'¼'),                                   '='),
    # (cid:2) is '×' in this document family; strip other unmapped CID placeholders.
    (re.compile(r'\(cid:2\)'),                           '×'),
    (re.compile(r'\(cid:\d+\)'),                         ''),
    # ﬃ/ﬀ/ﬁ/ﬂ ligature runs (3+ chars) represent the square-root radical '√'.
    (re.compile(r'[ﬃﬀﬁﬂ]{3,}'),                         '√'),
    # ð / Þ are used as '(' / ')' bracket substitutes in some font encodings.
    (re.compile(r'ð'),                                   '('),
    (re.compile(r'Þ'),                                   ')'),
    # l / I confused with digit 1 in numeric contexts
    (re.compile(r'\b([lI])\.(\d)'),                      r'1.\2'),   # l.15 → 1.15
    (re.compile(r'(?<=[\d.])[lI](?=[\d])'),              '1'),       # 1l5 → 115
    (re.compile(r'(?<=[=(\[{+\-*/\s])([lI])(?=[.\d])'), '1'),       # =l.5 → =1.5
    # Multi-line separator artifact from _find_formula_bbox merging
    (re.compile(r'\s*;\s*'),                              ' '),
]
_WHERE_CLAUSE   = re.compile(r'\s*\bwhere\s*[:\-]', re.IGNORECASE)
_NOTE_CLAUSE    = re.compile(r'\s*\bnote\s*[:\-]',  re.IGNORECASE)
_EQ_LABEL_TEXT  = re.compile(r'\bEq\.\s*\d+\.\d+[\.\d]*', re.IGNORECASE)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LABELED MODE  (Eq. X.X.X label-based, no ML)                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _classify(text: str) -> tuple[str, float]:
    """Classify a formula string into a category.

    Uses a conservative ``chemical_equation`` gate — both a reaction arrow AND a
    molecular-formula pattern (with at least one digit) must be present.  This
    prevents short uppercase variable names like ``PR`` or ``KsE`` from being
    misclassified as molecules.
    """
    if not text:
        return "mathematical_equation", 0.5
    if _UNITS_RE.search(text):
        return "engineering_formula", 0.75
    if _REACTION_RE.search(text) and _CHEM_RE.search(text):
        return "chemical_equation", 0.65
    return "mathematical_equation", 0.7


def _dedup_block_text(text: str) -> str:
    """Remove duplicate lines from two-column scanned PDF text duplication artefact."""
    lines = text.splitlines()
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped not in seen:
            seen.add(stripped)
            result.append(line)
    return "\n".join(result).strip()


def _clean_ocr(text: str) -> str:
    """Clean OCR artefacts from formula text before classification or VLM processing.

    Steps in order:
    1. Strip description clauses (``where:``, ``note:``) — not part of the formula.
    2. Remove embedded equation-label cross-references (``Eq. 12.4.1``).
    3. Apply common OCR substitution rules (l/I → 1 in numeric positions,
       multi-line semicolon separators removed).
    4. Collapse whitespace.
    """
    if not text:
        return ""
    t = text.replace("\n", " ")
    # Strip description suffixes
    for pat in (_WHERE_CLAUSE, _NOTE_CLAUSE):
        m = pat.search(t)
        if m:
            t = t[:m.start()]
    # Remove embedded label references
    t = _EQ_LABEL_TEXT.sub("", t)
    # Apply OCR fixes
    for pat, repl in _OCR_FIX:
        t = pat.sub(repl, t)
    return re.sub(r"\s+", " ", t).strip()


def _find_formula_bbox(
    blocks: list,
    label_x0: float,
    label_y0: float,
    label_y1: float,
    page_width: float,
    text_window: float = 8.0,
    gap_search: float = 60.0,
) -> tuple[list[float], str | None]:
    """Return (bbox, plain_text|None) for the formula region near the given label.

    Pass 1: text blocks overlapping the label Y ± text_window.
      text_window is kept tight (8 pt ≈ one text line) so adjacent equation blocks
      on the same page are not accidentally merged into the same crop.
    Pass 2: gap analysis using surrounding paragraph boundaries (image-only fallback).
    """
    label_left = min(label_x0 - 5, page_width * 0.75)

    # Imported once per call rather than per-block iteration.
    from equation_extraction.detection import extract_mixed_label_block as _emb  # noqa: PLC0415

    formula_blocks: list[tuple[float, float, float, float, str]] = []
    for bx0, by0, bx1, by1, btext, *_ in blocks:
        btext_c = btext.strip()
        if not btext_c or _is_label(btext_c):
            continue
        # Skip mixed-label blocks: they contain an equation label as a line and will
        # be handled by Pass 2, so don't fold them into a different equation's crop.
        if _emb(btext_c)[0] is not None:
            continue
        y_overlap = not (by1 < label_y0 - text_window or by0 > label_y1 + text_window)
        if y_overlap and bx0 < label_left:
            # Skip blocks that score negatively (clear prose paragraphs).
            # Pass geometry so layout signals can assist, but omit page_dims since
            # pypdfium2 blocks don't expose page width at this call site.
            label_cy = (label_y0 + label_y1) / 2.0
            block_cy = (by0 + by1) / 2.0
            sc = _score_formula(
                btext_c,
                bbox=[bx0, by0, bx1, by1],
                label_distance_pts=abs(block_cy - label_cy),
            )
            # Exclude blocks that are net-negative OR that look like running prose
            # (a block may score > 0 via a relational operator even when it's mainly
            # explanatory prose — raise the floor to 0.15 to reduce false captures).
            if sc.score >= 0.15:
                formula_blocks.append((bx0, by0, bx1, by1, btext_c))

    if formula_blocks:
        x0 = min(b[0] for b in formula_blocks)
        y0 = min(b[1] for b in formula_blocks)
        x1 = max(b[2] for b in formula_blocks)
        y1 = max(b[3] for b in formula_blocks)
        merged = " ".join(_clean_ocr(b[4]) for b in sorted(formula_blocks, key=lambda b: b[1]))
        # Prose-density cap: if the merged text is long AND prose-heavy, it likely
        # captured a surrounding paragraph. Keep only up to the first sentence
        # boundary (period/newline) that contains a relational operator.
        if len(merged) > 120:
            first_clause = re.split(r'(?<=[.!?])\s+[A-Z]|\n', merged, maxsplit=1)[0].strip()
            if first_clause and any(op in first_clause for op in '=<>≤≥≈±'):
                merged = first_clause
        return [max(0.0, x0 - 5), max(0.0, y0 - 4), x1 + 5, y1 + 4], merged

    blocks_before = [
        (bx0, by0, bx1, by1)
        for bx0, by0, bx1, by1, btext, *_ in blocks
        if by1 < label_y0 and by1 >= label_y0 - gap_search
        and not _is_label(btext.strip()) and btext.strip()
    ]
    blocks_after = [
        (bx0, by0, bx1, by1)
        for bx0, by0, bx1, by1, btext, *_ in blocks
        if by0 > label_y1 and by0 <= label_y1 + gap_search
        and not _is_label(btext.strip()) and btext.strip()
    ]

    top    = max((b[3] for b in blocks_before), default=label_y0 - 25)
    bottom = min((b[1] for b in blocks_after),  default=label_y1 + 15)
    if bottom <= top + 4:
        top, bottom = label_y0 - 10, label_y1 + 10

    return [36.0, top - 4, label_left, bottom + 4], None


def extract_labeled(pdf_path: Path) -> list[dict]:
    """Extract all Eq. X.X.X equations via label scanning (no ML).

    Two detection passes per page:
      Pass 1 — isolated label blocks: blocks whose entire text is ``Eq. X.X.X``.
               The formula is found in a neighbouring block via ``_find_formula_bbox``.
      Pass 2 — mixed-label blocks: blocks that contain *both* a label line and the
               formula body.  Handled by ``extract_mixed_label_block`` from the
               detection module.  Catches the ~57% of Knovel equations whose label
               and formula share one PDF text block.

    ``seen_eq_numbers`` prevents the same equation number being counted twice when
    the label appears in more than one block (e.g. a cross-reference later in the
    chapter).
    """
    from PIL import Image as _PILImage

    from equation_extraction.detection import extract_mixed_label_block, is_isolated_equation_label
    from pipeline import config
    from pipeline.pdf_backend import open_document, render_page_image
    from visual_extraction.assets import crop_region

    equations: list[dict] = []
    eq_counter = 0
    seen_eq_numbers: set[str] = set()
    pad_frac = config.KNOVEL_EQUATION_CROP_PAD_FRAC

    def _make_crop(page: object, page_rect: object, clip_bbox: list[float]) -> bytes | None:
        try:
            # Symmetrically expand the clip box (clamped to the page) so a left-hand-side
            # variable that sits just outside a tight layout box is captured in the crop —
            # the dominant cause of "= …" LaTeX with the LHS clipped off.
            x0, y0, x1, y1 = clip_bbox
            if pad_frac:
                dx, dy = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
                x0, y0 = max(0.0, x0 - dx), max(0.0, y0 - dy)
                x1 = min(float(page_rect.width), x1 + dx)
                y1 = min(float(page_rect.height), y1 + dy)
                clip_bbox = [x0, y0, x1, y1]
            page_img = _PILImage.fromarray(render_page_image(page, zoom=2.0))
            crop = crop_region(page_img, (page_rect.width, page_rect.height), clip_bbox)
            if crop is None:
                return None
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError as exc:
            print(f"  [error] crop: missing dependency — {exc}", file=sys.stderr)
            return None
        except Exception as exc:
            print(f"  [warn] crop failed bbox={clip_bbox}: {exc}", file=sys.stderr)
            return None

    with open_document(str(pdf_path)) as doc:
        total_pages = len(doc)
        print(f"\n[stage:equation_extraction] Scanning {total_pages} pages …")

        for pno in range(total_pages):
            page = doc[pno]
            page_rect  = page.rect
            page_width = page_rect.width
            blocks     = page.get_text("blocks")
            page_found = 0

            # ── Pass 1: isolated label blocks ─────────────────────────────────
            for bx0, by0, bx1, by1, btext, *_ in blocks:
                btext_stripped = _dedup_block_text(btext.strip())
                if not _is_label(btext_stripped):
                    continue

                eq_num_str = _extract_label_number(btext_stripped)
                if eq_num_str in seen_eq_numbers:
                    continue
                seen_eq_numbers.add(eq_num_str)

                eq_id = f"eq_{eq_counter}_p{pno}_{eq_num_str.replace('.', '_')}"

                bbox, plain_text = _find_formula_bbox(
                    blocks, label_x0=bx0, label_y0=by0, label_y1=by1, page_width=page_width,
                )
                category, class_conf = _classify(plain_text or "")
                clip = [bbox[0], max(0.0, bbox[1] - 8.0),
                        min(page_rect.width, bx1 + 5.0), min(page_rect.height, bbox[3] + 8.0)]
                crop_png = _make_crop(page, page_rect, clip)

                equations.append({
                    "equation_id":               eq_id,
                    "equation_number":           eq_num_str,
                    "page_no":                   pno,
                    "page_position":             eq_counter,
                    "reading_position":          eq_counter,
                    "is_inline":                 False,
                    "category":                  category,
                    "classification_confidence": class_conf,
                    "classification_reason": (
                        "engineering units detected" if category == "engineering_formula"
                        else "mathematical operators/symbols"
                    ),
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                    "plain_text":                plain_text,
                    "latex":                     None,
                    "mathml":                    None,
                    "structured_form":           None,
                    "recognition_confidence":    0.0,
                    "selected_provider":         "text_layer" if plain_text else "none",
                    "provenance": {
                        "source_extractors": ["eq_label_scan"],
                        "source_pages":      [pno],
                    },
                    "validation_flags": [] if plain_text else ["no_ocr_text"],
                    "notes": (
                        ["eq_label_detected"] if plain_text
                        else ["eq_label_detected", "formula_image_only"]
                    ),
                    "canonical_element_id": None,
                    "caption_ref":          None,
                    "continuation_ref":     None,
                    "structural_parent_id": None,
                    "region_id":            f"p{pno}-eq{eq_counter}",
                    "text_block_id":        None,
                    "metadata": {"corrected": False, "notes": {}, "crop_png": crop_png},
                })
                eq_counter += 1
                page_found += 1
                print(
                    f"  [pass1] p{pno+1:3d}  Eq.{eq_num_str:<12} "
                    f"{'TEXT' if plain_text else 'IMG '}: "
                    f"{(plain_text or '(image formula)')[:60]}"
                )

            # ── Pass 2: mixed-label blocks (label + formula in the same block) ─
            for bx0, by0, bx1, by1, btext, *_ in blocks:
                deduped_btext = _dedup_block_text(btext.strip())
                label_line, formula_text = extract_mixed_label_block(deduped_btext)
                if label_line is None:
                    continue
                if formula_text and is_isolated_equation_label(formula_text):
                    continue

                eq_num_str = _extract_label_number(label_line)
                if eq_num_str in seen_eq_numbers:
                    continue
                seen_eq_numbers.add(eq_num_str)

                eq_id = f"eq_{eq_counter}_p{pno}_{eq_num_str.replace('.', '_')}_mixed"
                plain_text = _clean_ocr(formula_text) if formula_text else None
                category, class_conf = _classify(plain_text or "")

                # Trim the label line from the crop so the VLM sees only the formula.
                # Estimate line height from the block height divided by its line count.
                block_lines = [ln.strip() for ln in deduped_btext.split("\n") if ln.strip()]
                n_lines = max(1, len(block_lines))
                line_h = max(8.0, (by1 - by0) / n_lines)
                clip_y0, clip_y1 = by0, by1
                if block_lines and _is_label(block_lines[0]):
                    clip_y0 = min(by1 - line_h * 0.5, by0 + line_h)
                elif block_lines and _is_label(block_lines[-1]):
                    clip_y1 = max(by0 + line_h * 0.5, by1 - line_h)
                crop_png = _make_crop(page, page_rect, [bx0, clip_y0, bx1, clip_y1])

                equations.append({
                    "equation_id":               eq_id,
                    "equation_number":           eq_num_str,
                    "page_no":                   pno,
                    "page_position":             eq_counter,
                    "reading_position":          eq_counter,
                    "is_inline":                 False,
                    "category":                  category,
                    "classification_confidence": class_conf,
                    "classification_reason": (
                        "engineering units detected" if category == "engineering_formula"
                        else "mathematical operators/symbols"
                    ),
                    "bbox": {"x0": bx0, "y0": by0, "x1": bx1, "y1": by1},
                    "plain_text":                plain_text,
                    "latex":                     None,
                    "mathml":                    None,
                    "structured_form":           None,
                    "recognition_confidence":    0.0,
                    "selected_provider":         "text_layer" if plain_text else "none",
                    "provenance": {
                        "source_extractors": ["eq_mixed_block_scan"],
                        "source_pages":      [pno],
                    },
                    "validation_flags": [] if plain_text else ["no_ocr_text"],
                    "notes": ["eq_mixed_block_detected"],
                    "canonical_element_id": None,
                    "caption_ref":          None,
                    "continuation_ref":     None,
                    "structural_parent_id": None,
                    "region_id":            f"p{pno}-eq{eq_counter}",
                    "text_block_id":        None,
                    "metadata": {"corrected": False, "notes": {}, "crop_png": crop_png},
                })
                eq_counter += 1
                page_found += 1
                print(
                    f"  [pass2] p{pno+1:3d}  Eq.{eq_num_str:<12} "
                    f"[MIXED] "
                    f"{'TEXT' if plain_text else 'IMG '}: "
                    f"{(plain_text or '(image formula)')[:60]}"
                )

            if page_found:
                print(f"[stage:equation_extraction] page {pno+1}/{total_pages}: {page_found} equations detected (running total: {eq_counter})")

        print(f"[stage:equation_extraction] Scan complete — {eq_counter} equations found across {total_pages} pages")

    return equations


def _postprocess_latex(latex: str) -> str:
    """Fix common VLM output issues in LaTeX strings.

    Applies in order: label bleed removal, unescaped ``sqrt``/function names,
    stray dollar signs, and whitespace normalisation.
    """
    if not latex:
        return latex
    # Remove equation-label bleed (e.g. "Eq. 12.4.3" captured into LaTeX)
    latex = _LABEL_IN_LATEX_RE.sub("", latex)
    # Fix unescaped sqrt: sqrt(...) → \sqrt{...}
    latex = re.sub(r'(?<!\\)\bsqrt\s*\(([^)]+)\)', r'\\sqrt{\1}', latex)
    # Add missing backslash on common math function names
    for _fn in ("sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim", "max", "min"):
        latex = re.sub(rf'(?<!\\)\b{_fn}\b', rf'\\{_fn}', latex)
    # Strip stray wrapper characters
    latex = latex.strip().strip("$").strip()
    return latex


def _latex_to_mathml(latex: str) -> str | None:
    """Convert a LaTeX string to MathML using latex2mathml. Returns None on failure."""
    if not latex:
        return None
    try:
        from latex2mathml.converter import convert
        return convert(latex)
    except Exception:
        return None


def _derive_structured_form(latex: str) -> dict | None:
    """Derive a simple structured representation from a LaTeX expression.

    Splits on the first ``=`` to produce ``{lhs, rhs, form: 'equation'}``; returns
    ``{expression, form: 'expression'}`` when no equals sign is present.
    """
    if not latex:
        return None
    # Strip wrappers like \\[ ... \\] or $...$
    stripped = re.sub(r'^\s*(?:\\\[|\$+)\s*|\s*(?:\\\]|\$+)\s*$', "", latex).strip()
    if not stripped:
        return None
    if "=" in stripped:
        lhs, _, rhs = stripped.partition("=")
        return {"lhs": lhs.strip(), "rhs": rhs.strip(), "form": "equation"}
    return {"expression": stripped, "form": "expression"}


def _enrich_with_latex(equations: list[dict]) -> None:
    """Call the configured VLM backend to fill latex/confidence for each equation in-place.

    Uses the image crop when available (preferred), falling back to the OCR plain_text.
    Silently skips when the inference backend is unreachable so the labeled extraction
    still produces a valid sidecar — just without LaTeX.
    """
    import os
    import tempfile

    from pipeline import config

    _RETRY_ENABLED = config.KNOVEL_EQUATION_RETRY_ENABLED
    _RETRY_THRESHOLD = config.KNOVEL_EQUATION_RECOGNITION_RETRY_THRESHOLD
    # Quality-issue notes that independently warrant a retry even when confidence is
    # above the numeric threshold.
    _RETRY_QUALITY_NOTES = frozenset({
        "quality:label_only", "quality:prose_contamination", "quality:spaced_text",
        "quality:multiple_tags", "quality:multiple_equations",
    })
    _SPLIT_NOTES = frozenset({"quality:multiple_tags", "quality:multiple_equations"})

    from equation_extraction.crop_split import split_stacked_crop as _split_crop

    try:
        from pipeline.inference import get_vision_service
        svc = get_vision_service()
    except Exception as exc:
        print(f"  [info] Inference backend unavailable — skipping LaTeX generation: {exc}",
              file=sys.stderr)
        return

    if not svc._backend.health_check():
        print(
            "  [info] Ollama not reachable — skipping LaTeX generation.\n"
            "         Start Ollama and re-run to get LaTeX.",
            file=sys.stderr,
        )
        return

    print(f"\nGenerating LaTeX via Qwen [{len(equations)} equations] …")
    enriched = 0

    for i, eq in enumerate(equations, 1):
        eq_id      = eq["equation_id"]
        category   = eq.get("category", "unknown")
        crop_bytes = (eq.get("metadata") or {}).get("crop_png")

        try:
            if crop_bytes:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(crop_bytes)
                    tmp_path = Path(tmp.name)
                try:
                    result = svc.extract_equation(tmp_path, category=category)
                    result_note_set = set(result.notes or [])
                    _has_quality_issue = bool(_RETRY_QUALITY_NOTES & result_note_set)
                    _needs_split = bool(_SPLIT_NOTES & result_note_set)

                    # ── Split path: retry on top sub-crop when multiple equations detected ──
                    if _RETRY_ENABLED and _needs_split:
                        try:
                            from PIL import Image as _PILImg
                            _img = _PILImg.open(tmp_path)
                            _sub = _split_crop(_img)
                            if len(_sub) > 1:
                                _buf = io.BytesIO()
                                _sub[0].save(_buf, format="PNG")
                                _buf.seek(0)
                                import tempfile as _tf
                                with _tf.NamedTemporaryFile(suffix=".png", delete=False) as _t:
                                    _t.write(_buf.read())
                                    _split_path = Path(_t.name)
                                try:
                                    split_result = svc.extract_equation(
                                        _split_path, category=category, strict=True
                                    )
                                    n = len(_sub)
                                    split_result.notes = list(split_result.notes) + [
                                        f"split_crop:n={n}", "split_crop:applied"
                                    ]
                                    if split_result.confidence > result.confidence:
                                        split_result.notes = list(split_result.notes) + ["split_crop:improved"]
                                        result = split_result
                                        _has_quality_issue = False  # split fixed it; skip standard retry
                                    else:
                                        result.notes = list(result.notes) + ["split_crop:no_improvement"]
                                finally:
                                    try:
                                        os.unlink(_split_path)
                                    except OSError:
                                        pass
                        except Exception:
                            pass

                    # ── Standard confidence/quality retry (padded full crop) ──
                    if _RETRY_ENABLED and (
                        result.confidence < _RETRY_THRESHOLD or _has_quality_issue
                    ):
                        retry = svc.extract_equation(tmp_path, category=category, strict=True)
                        if _has_quality_issue:
                            retry.notes = list(retry.notes) + ["recognition_retry:quality_issue"]
                        if retry.confidence > result.confidence:
                            retry.notes = list(retry.notes) + ["recognition_retry:improved"]
                            result = retry
                        else:
                            result.notes = list(result.notes) + ["recognition_retry:no_improvement"]
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            elif eq.get("plain_text"):
                result = svc.extract_equation_from_text(eq["plain_text"], category=category)
            else:
                continue  # no source material

            if result.latex or result.structured_form:
                clean_latex = _postprocess_latex(result.latex or "")
                eq["latex"]                  = clean_latex or result.latex
                eq["mathml"]                 = _latex_to_mathml(clean_latex or result.latex or "")
                eq["structured_form"]        = result.structured_form or _derive_structured_form(clean_latex or result.latex or "")
                eq["recognition_confidence"] = result.confidence
                eq["selected_provider"]      = svc._backend.backend_name
                eq["notes"] = (eq.get("notes") or []) + ["vlm_enriched"] + list(result.notes or [])
                enriched += 1
                preview = (clean_latex or result.latex or result.structured_form or "")[:60]
                print(f"  [{i:>2}/{len(equations)}] ✓ {eq_id}: {preview}")
            else:
                note_str = ", ".join(result.notes) if result.notes else "empty response"
                print(f"  [{i:>2}/{len(equations)}] ~ {eq_id}: no output ({note_str})")
                eq["notes"] = (eq.get("notes") or []) + (result.notes or [])

        except Exception as exc:
            print(f"  [{i:>2}/{len(equations)}] ✗ {eq_id}: {exc}", file=sys.stderr)

    print(f"\n  LaTeX generated: {enriched} / {len(equations)} equations")


def _apply_confidence(equations: list[dict]) -> None:
    """Stamp confidence scores onto each equation dict in-place.

    Uses crop_image (PNG bytes from metadata) and bbox when available so that
    confidence_layout is populated rather than null.  Call this BEFORE stripping
    crop_png from metadata.
    """
    from io import BytesIO

    try:
        from PIL import Image
    except ImportError:
        Image = None

    from equation_extraction.confidence_estimation import ConfidenceEstimator
    estimator = ConfidenceEstimator()
    for eq in equations:
        latex = eq.get("latex") or ""
        crop_image = None
        bbox_tuple = None
        try:
            crop_bytes = (eq.get("metadata") or {}).get("crop_png")
            if crop_bytes and Image is not None:
                crop_image = Image.open(BytesIO(crop_bytes))
            raw_bbox = eq.get("bbox")
            if isinstance(raw_bbox, dict):
                bbox_tuple = (raw_bbox["x0"], raw_bbox["y0"], raw_bbox["x1"], raw_bbox["y1"])
            elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
                bbox_tuple = tuple(raw_bbox)
        except Exception:
            pass
        try:
            cr = estimator.estimate(
                latex=latex, crop_image=crop_image, bbox=bbox_tuple,
            )
            confidence_payload = cr.to_dict()
            eq["overall_confidence"]      = cr.overall_confidence
            eq["confidence_recognition"]  = cr.recognition
            eq["confidence_layout"]       = cr.layout
            eq["confidence_syntax"]       = cr.syntax
            eq["confidence_ocr_quality"]  = cr.ocr_quality
            eq["confidence"]              = confidence_payload
            eq["confidence_result"]       = confidence_payload
            # keep recognition_confidence in sync with overall
            eq["recognition_confidence"]  = cr.overall_confidence
        except Exception:
            pass
        eq["confidence_score"] = _confidence_score(eq)


def build_sidecar(equations: list[dict]) -> dict:
    """Build the flat sidecar envelope expected by the validate step."""
    # Apply confidence BEFORE stripping crop_png so layout estimation has the image.
    _apply_confidence(equations)

    for eq in equations:
        eq.get("metadata", {}).pop("crop_png", None)

    metrics = _compute_metrics(equations, layout_equation_count=len(equations))
    stats: dict = {
        "total_pages": len({e["page_no"] for e in equations}),
        **metrics,
    }

    return {
        "version":        "1.0",
        "outcome":        "success",
        "equations":      equations,
        "pages":          [],
        "statistics":     stats,
        "providers":      {"eq_label_scan": {"name": "Eq-label scanner", "version": "1.0"}},
        "config_hash":    "eq_label_extractor_v1",
        "notes":          ["Extracted via Eq.X.X.X label detection from OCR text layer"],
        "failure_reason": None,
    }


def save_sidecar(sidecar: dict, pdf_path: Path) -> Path:
    """Write sidecar to data/output/{stem}/equation_extraction.json."""
    out_dir = OUTPUT_DIR / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "equation_extraction.json"
    out.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✓ Sidecar written → {out}")
    return out


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  FULL PIPELINE MODE  (ingestion → layout → pix2tex)                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_full_pipeline(
    pdf_path: Path,
    output_dir: Path,
    *,
    inline: bool = True,
    mathml: bool = False,
    debug_dump: bool = False,
    provider_map: str = "",
) -> dict:
    """Run all upstream pipeline stages then equation extraction."""
    os.environ["KNOVEL_EQUATION_INLINE_ENABLED"] = "true" if inline else "false"
    os.environ["KNOVEL_EQUATION_MATHML_ENABLED"]  = "true" if mathml else "false"
    os.environ["KNOVEL_EQUATION_DEBUG_DUMP"]       = "true" if debug_dump else "false"
    if provider_map:
        os.environ["KNOVEL_EQUATION_PROVIDER_MAP"] = provider_map
    for flag in (
        "KNOVEL_TABLE_ENABLED", "KNOVEL_VISUAL_ENABLED",
        "KNOVEL_METADATA_ENABLED", "KNOVEL_RELATIONSHIP_ENABLED",
        "KNOVEL_VALIDATION_ENABLED", "KNOVEL_EXPORT_ENABLED",
    ):
        os.environ.setdefault(flag, "false")

    from classifier.doc_manifest import get_or_create_classification
    from classifier.manifest import get_or_create_manifest
    from equation_extraction import get_or_create_equation_extraction
    from ingestion.fingerprint import compute_fingerprint
    from ingestion.ingest import ingest_document
    from layout import get_or_create_layout
    from pipeline import config
    from preprocessing import get_or_create_preprocessing
    from reading_order import get_or_create_reading_order
    from text_extraction import get_or_create_text_extraction
    import importlib; importlib.reload(config)

    pdf_path = pdf_path.expanduser().resolve()

    ingestion    = ingest_document(pdf_path, output_dir=output_dir, pipeline_run_id="standalone")
    manifest     = ingestion.manifest
    sha256       = manifest.identity.fingerprint if manifest.identity else compute_fingerprint(pdf_path)
    page_manifest = get_or_create_manifest(pdf_path, sha256)
    classification = get_or_create_classification(
        pdf_path, sha256, page_manifest=page_manifest, metadata=manifest.metadata
    )
    preprocessing = get_or_create_preprocessing(pdf_path, sha256, classification=classification, page_manifest=page_manifest) \
        if config.KNOVEL_PREPROCESS_ENABLED else None
    layout = get_or_create_layout(
        pdf_path, sha256, classification=classification,
        preprocessing=preprocessing, page_manifest=page_manifest,
    ) if config.KNOVEL_LAYOUT_ENABLED else None
    reading_order = get_or_create_reading_order(
        pdf_path, sha256, layout=layout, classification=classification, page_manifest=page_manifest,
    ) if config.KNOVEL_READING_ORDER_ENABLED else None
    text_extraction = get_or_create_text_extraction(
        pdf_path, sha256, reading_order=reading_order, layout=layout,
        preprocessing=preprocessing, classification=classification, page_manifest=page_manifest,
    ) if config.KNOVEL_TEXT_ENABLED else None

    ctx = get_or_create_equation_extraction(
        pdf_path, sha256, text_extraction=text_extraction, reading_order=reading_order,
        layout=layout, preprocessing=preprocessing, classification=classification,
        page_manifest=page_manifest,
    )

    book_out = output_dir / pdf_path.stem
    book_out.mkdir(parents=True, exist_ok=True)

    from equation_extraction.output_formatter import (
        format_equation_extraction_output,
        save_structured_crops,
    )
    crops_dir = book_out / "crops"
    crop_info = save_structured_crops(ctx, pdf_path, preprocessing, layout, crops_dir)
    output = format_equation_extraction_output(ctx, pdf_path, classification, crop_info, layout)
    result_path = book_out / "equation_extraction.json"
    result_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    stats = ctx.statistics
    return {
        "pdf": pdf_path.name,
        "outcome": ctx.outcome,
        "total_equations": stats.total_equations,
        "total_pages": stats.total_pages,
        "category_distribution": stats.category_distribution,
        "equations_by_provider": stats.equations_by_provider,
        "latex_valid": stats.latex_valid_count,
        "mathml_valid": stats.mathml_valid_count,
        "low_confidence_classification": stats.low_confidence_classification_count,
        "low_confidence_recognition":    stats.low_confidence_recognition_count,
        "failures": stats.failures,
        "result_path": str(result_path),
    }


def print_full_summary(summary: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {summary['pdf']}  [{summary['outcome'].upper()}]")
    print(f"{'='*60}")
    print(f"  Total equations   : {summary['total_equations']}")
    print(f"  Pages processed   : {summary['total_pages']}")
    print(f"  LaTeX valid       : {summary['latex_valid']}")
    if summary.get("mathml_valid"):
        print(f"  MathML valid      : {summary['mathml_valid']}")
    print(f"  Low-conf (class.) : {summary['low_confidence_classification']}")
    print(f"  Low-conf (recog.) : {summary['low_confidence_recognition']}")
    print(f"  Page failures     : {summary['failures']}")
    for cat, count in sorted((summary.get("category_distribution") or {}).items(), key=lambda x: -x[1]):
        print(f"    {cat:<35} {count}")
    print(f"  Result JSON       : {summary['result_path']}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  VALIDATION  (HTML report + CSV from sidecar JSON)                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _render_pdf_crop(pdf_path: Path, page_no: int, bbox: list[float]) -> bytes | None:
    try:
        from PIL import Image

        from pipeline.pdf_backend import RENDER_ZOOM, open_document, render_page_image
        from visual_extraction.assets import crop_region

        with open_document(str(pdf_path)) as doc:
            page = doc[page_no]  # page_no is 0-based (same as PdfDocument)
            rect = page.rect
            img  = Image.fromarray(render_page_image(page)).convert("RGB")

        # page.rect uses pdfminer; fall back to render-derived dims if it returns 0
        pw, ph = rect.width, rect.height
        if pw <= 0 or ph <= 0:
            px_w, px_h = img.size
            pw, ph = px_w / RENDER_ZOOM, px_h / RENDER_ZOOM

        crop = crop_region(img, (pw, ph), bbox)
        if crop is None:
            return None
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError as exc:
        # Missing pypdfium2 / pdfminer.six — add to requirements-dashboard.txt
        print(f"  [error] crop render: missing dependency — {exc}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  [warn] crop render failed p{page_no} bbox={bbox}: {exc}", file=sys.stderr)
        return None


def _png_to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


_REPORT_LABEL_ONLY_RE = re.compile(
    r"""^\s*
    (?:Eq(?:uation)?[.:]?\s*[\d]+(?:\.[\d]+){1,4}
    |\(\s*[\d]+(?:[-.][\d]+){1,4}\s*\)
    |\[\s*[\d]+(?:[-.][\d]+){1,4}\s*\]
    |\\left\s*\(\s*[\d]+(?:\s*[-–]\s*[\d]+){1,4}\s*\\right\s*\)
    )\s*$""",
    re.VERBOSE | re.IGNORECASE,
)
_REPORT_REL_OP_RE = re.compile(
    r"[=<>≤≥≈≅±∝∈∉∪∩→⇌]"
    r"|\\(?:frac|sum|int|prod|leq|geq|neq|approx|sim|simeq|cong|equiv|propto"
    r"|subset|subseteq|supset|supseteq|in|notin|rightarrow|to|Rightarrow|Leftrightarrow)\b"
)
_REPORT_PROSE_RE = re.compile(
    r"\b(?:where|which|from|then|becomes|therefore|the|this|that|"
    r"these|those|for|with|into|using|equation|factor|value|note)\b",
    re.IGNORECASE,
)


def _score_latex_quality(eq: dict) -> tuple[float, list[str]]:
    """Score a display equation's LaTeX output; return (score [0–1], issues).

    Detects VLM failure modes:
    - label_bleed: the margin label (Eq. N.N) was captured into the LaTeX string
    - label_only_output: output is entirely a label, no formula
    - hallucination: prose text wrapped inside a \\begin{equation} block
    - plain_text_like: Unicode math symbols used instead of LaTeX commands
    - prose_contamination: prose words present with no relational operator
    - no_relational_operator: no =/</>/ etc. or LaTeX operator commands
    - multiple_equations_in_crop: multiple lines each containing an operator
    - suspiciously_short: fewer than 3 math characters after stripping commands
    - degenerate: output too short (<3 chars) to be meaningful
    """
    latex = (eq.get("latex") or "").strip()
    if not latex:
        return 0.0, ["no_latex"]
    score = 1.0
    issues: list[str] = []

    # Label-only output: the VLM read the margin label instead of the formula
    if _REPORT_LABEL_ONLY_RE.match(latex):
        score -= 0.8
        issues.append("label_only_output")
        return max(0.0, min(1.0, round(score, 2))), issues

    if _LABEL_IN_LATEX_RE.search(latex):
        score -= 0.6
        issues.append("label_bleed")

    if "\\begin{" in latex:
        stripped = re.sub(r"\\[a-zA-Z]+(?:\{[^}]*\})?", " ", latex)
        plain_words = re.findall(r"\b[a-zA-Z]{4,}\b", stripped)
        if len(plain_words) > 5:
            score -= 0.5
            issues.append("hallucination")

    unicode_hits = sum(1 for c in latex if c in _UNICODE_MATH_CHARS)
    if unicode_hits >= 2:
        score -= min(0.35, unicode_hits * 0.08)
        issues.append("plain_text_like")

    # Prose contamination: prose words without any relational/math operator
    prose_hits = len(_REPORT_PROSE_RE.findall(latex))
    if prose_hits >= 2 and not _REPORT_REL_OP_RE.search(latex):
        score -= 0.4
        issues.append("prose_contamination")

    # Connective sentence fragment inside \text{}: patterns like
    # "\text{ or for standard conditions at }" are prose bleeding from surrounding text.
    # Uses short connectives (or/and/the/for/at/…); quantity labels like
    # "\text{Heat liberated by explosion}" have at most 1 such word and don't fire.
    _CONNECTIVE_RE = re.compile(r'\b(?:or|and|the|for|with|at|of|in|is|are|to|by)\b', re.IGNORECASE)
    _text_blocks = re.findall(r'\\text\{([^}]*)\}', latex)
    prose_in_text = sum(1 for blk in _text_blocks if len(_CONNECTIVE_RE.findall(blk)) >= 2)
    if prose_in_text and "prose_contamination" not in issues:
        score -= 0.35
        issues.append("prose_in_text_block")

    # No relational operator — likely a bare fragment (RHS only, or a name/label)
    if not _REPORT_REL_OP_RE.search(latex) and "prose_contamination" not in issues:
        score -= 0.25
        issues.append("no_relational_operator")

    # Multiple equations in one crop: 2+ lines each containing a relational operator
    op_lines = sum(1 for ln in latex.split("\n") if _REPORT_REL_OP_RE.search(ln))
    if op_lines >= 2:
        score -= 0.2
        issues.append("multiple_equations_in_crop")

    # Suspiciously short after stripping LaTeX commands and whitespace
    math_chars = re.sub(r"\\[a-zA-Z]+|\s", "", latex)
    if len(math_chars) < 3:
        score -= 0.5
        issues.append("suspiciously_short")
    elif len(latex.replace(" ", "")) < 3:
        score -= 0.8
        issues.append("degenerate")

    return max(0.0, min(1.0, round(score, 2))), issues


def _scan_pdf_labeled_count(pdf_path: Path) -> int:
    """Count standalone equation labels in the PDF text layer using pdfminer.six.

    Uses the production `is_isolated_equation_label` / `extract_label_number`
    helpers (already imported) so the count is consistent with the extractor.
    Returns 0 when pdfminer is unavailable or the scan fails.
    """
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextBoxHorizontal

        seen: set[str] = set()
        for page_layout in extract_pages(str(pdf_path)):
            for element in page_layout:
                if not isinstance(element, LTTextBoxHorizontal):
                    continue
                for line in element.get_text().splitlines():
                    s = line.strip()
                    if _is_label(s):
                        label_num = _extract_label_number(s)
                        if label_num:
                            seen.add(label_num)
        return len(seen)
    except ImportError:
        return 0
    except Exception:
        return 0


def _compute_metrics(equations: list[dict], layout_equation_count: int) -> dict:
    display = [e for e in equations if not e["is_inline"]]
    inline  = [e for e in equations if e["is_inline"]]

    # Score every display equation with latex; mutate flags in-place
    quality_scores: list[float] = []
    quality_dist = {"good": 0, "warn": 0, "fail": 0}
    needs_review  = 0
    for eq in display:
        if eq.get("latex"):
            score, issues = _score_latex_quality(eq)
            eq["quality_score"] = score
            if score < 0.5:
                vflags = list(eq.get("validation_flags") or [])
                if "invalid_latex" not in vflags:
                    vflags.append("invalid_latex")
                    eq["validation_flags"] = vflags
            if score >= 0.75:
                quality_dist["good"] += 1
            elif score >= 0.5:
                quality_dist["warn"] += 1
            else:
                quality_dist["fail"] += 1
            quality_scores.append(score)
            if score < 0.6:
                needs_review += 1

    avg_quality = round(sum(quality_scores) / len(quality_scores) * 100, 1) if quality_scores else 0.0

    # When all equations are inline (no display equations), fall back to using all
    # equations for LaTeX / confidence stats so the metrics are not misleadingly 0/0.
    all_inline = len(display) == 0 and len(inline) > 0
    eq_pool = equations if all_inline else display

    has_latex     = [e for e in eq_pool if e.get("latex")]
    valid_latex   = [e for e in eq_pool if e.get("latex") and "invalid_latex" not in (e.get("validation_flags") or [])]
    invalid_latex = [e for e in eq_pool if e.get("latex") and "invalid_latex" in (e.get("validation_flags") or [])]
    high_conf     = [e for e in eq_pool if e.get("recognition_confidence", 0) >= 0.5]
    no_image      = [e for e in display if any("no_region_image" in n for n in (e.get("notes") or []))]

    flags: dict[str, int] = {}
    for e in equations:
        for f in (e.get("validation_flags") or []):
            flags[f] = flags.get(f, 0) + 1

    conf_buckets = {"0.0 (passthrough)": 0, "0.0–0.5": 0, "0.5–0.8": 0, "0.8–1.0": 0}
    for e in equations:
        c = e.get("recognition_confidence", 0)
        if c == 0.0:    conf_buckets["0.0 (passthrough)"] += 1
        elif c < 0.5:   conf_buckets["0.0–0.5"] += 1
        elif c < 0.8:   conf_buckets["0.5–0.8"] += 1
        else:           conf_buckets["0.8–1.0"] += 1

    cat_dist: dict[str, int] = {}
    for e in equations:
        cat_dist.setdefault(e.get("category", "unknown"), 0)
        cat_dist[e.get("category", "unknown")] += 1

    pages_with_equations = len({e["page_no"] for e in (equations if all_inline else display)})

    return {
        "total_equations":            len(equations),
        "display_equations":          len(display),
        "inline_equations":           len(inline),
        "all_inline":                 all_inline,
        "latex_pool_size":            len(eq_pool),
        "layout_equation_regions":    layout_equation_count,
        "detection_rate_pct":         round(len(display) / layout_equation_count * 100, 1) if layout_equation_count else 0,
        "latex_generated":            len(has_latex),
        "latex_generation_rate_pct":  round(len(has_latex)   / len(eq_pool)   * 100, 1) if eq_pool   else 0,
        "latex_valid":                len(valid_latex),
        "latex_validity_rate_pct":    round(len(valid_latex) / len(has_latex)  * 100, 1) if has_latex else 0,
        "latex_invalid":              len(invalid_latex),
        "high_confidence_recognition":len(high_conf),
        "high_confidence_rate_pct":   round(len(high_conf)   / len(eq_pool)   * 100, 1) if eq_pool   else 0,
        "no_image_fallback":          len(no_image),
        "confidence_distribution":    conf_buckets,
        "category_distribution":      cat_dist,
        "validation_flags":           flags,
        "latex_quality_score_pct":    avg_quality,
        "needs_review_count":         needs_review,
        "quality_distribution":       quality_dist,
        "pages_with_equations":       pages_with_equations,
    }


_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Equation Extraction Validation Report</title>
<script>
  window.MathJax = {{
    tex: {{ inlineMath: [['\\\\(','\\\\)'], ['$','$']], displayMath: [['$$','$$']] }},
    options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
  body {{ font-family: sans-serif; margin: 20px; background: #f8f9fa; color: #212529; }}
  h1 {{ color: #343a40; }}
  h2 {{ color: #495057; border-bottom: 2px solid #dee2e6; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; background: white; }}
  th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; vertical-align: top; }}
  th {{ background: #e9ecef; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .metric-value {{ font-weight: bold; color: #0d6efd; }}
  .pass {{ color: #198754; font-weight: bold; }}
  .fail {{ color: #dc3545; font-weight: bold; }}
  .warn {{ color: #fd7e14; font-weight: bold; }}
  .eq-row td {{ padding: 6px 8px; }}
  img.crop {{ max-width: 340px; max-height: 120px; border: 1px solid #ccc; }}
  .mathjax-render {{ min-height: 40px; padding: 6px; background: white; border: 1px solid #ccc;
                      border-radius: 3px; font-size: 13px; overflow-x: auto; max-width: 340px; }}
  code {{ font-size: 11px; background: #f1f3f5; padding: 2px 4px; border-radius: 3px;
           word-break: break-all; display: block; max-width: 320px; }}
  .flag {{ display: inline-block; background: #fff3cd; border: 1px solid #ffc107;
            border-radius: 3px; padding: 1px 5px; font-size: 11px; margin: 1px; }}
  .flag-bad {{ background: #f8d7da; border-color: #dc3545; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px; }}
  .metric-card {{ background: white; border: 1px solid #dee2e6; border-radius: 6px;
                   padding: 14px 18px; }}
  .metric-card h3 {{ margin: 0 0 4px; font-size: 13px; color: #6c757d; }}
  .metric-card .val {{ font-size: 26px; font-weight: bold; color: #0d6efd; }}
  .metric-card .sub {{ font-size: 12px; color: #6c757d; margin-top: 2px; }}
</style>
</head>
<body>
<h1>Equation Extraction Validation Report</h1>
<p><strong>PDF:</strong> {pdf_name} &nbsp;|&nbsp; <strong>Pages:</strong> {total_pages} &nbsp;|&nbsp;
   <strong>Generated:</strong> {timestamp}</p>
"""

_HTML_TAIL = "</body></html>"


def _flag_html(flags: list[str]) -> str:
    out = []
    for f in flags:
        cls = "flag-bad" if f in {"invalid_latex", "invalid_mathml", "unsupported_category", "broken_multiline"} else "flag"
        out.append(f'<span class="{cls}">{f}</span>')
    return " ".join(out)


def _status_class(eq: dict) -> str:
    # Prefer LLM judge verdict when available
    llm = eq.get("llm_verdict")
    if llm == "accept":
        return "pass"
    if llm == "reject":
        return "fail"
    if llm == "review":
        return "warn"
    # Fallback: rule-based
    flags = eq.get("validation_flags") or []
    if "invalid_latex" in flags or "unsupported_category" in flags:
        return "fail"
    if eq.get("latex") and eq.get("recognition_confidence", 0) >= 0.5:
        return "pass"
    return "warn"


def _write_html_report(
    equations: list[dict],
    metrics: dict,
    pdf_path: Path,
    out_path: Path,
    *,
    max_visual: int = 200,
) -> None:
    from datetime import datetime
    display_eqs = [e for e in equations if not e["is_inline"]]
    inline_eqs  = [e for e in equations if e["is_inline"]]

    lines: list[str] = []
    lines.append(_HTML_HEAD.format(
        pdf_name=pdf_path.name,
        total_pages=metrics.get("total_pages", "?"),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
    ))

    lines.append('<div class="summary-grid">')
    pdf_labeled = metrics.get("pdf_labeled_count", 0)
    pdf_cov = metrics.get("pdf_coverage_pct")
    coverage_val = f"{pdf_cov}%" if pdf_cov is not None else "N/A"
    coverage_sub = (
        f"{metrics['display_equations']} extracted / {pdf_labeled} in PDF"
        if pdf_labeled else "PDF label scan returned 0 (may use ML layout only)"
    )
    cards = [
        ("Extracted Equations",  metrics["total_equations"],
         f"display: {metrics['display_equations']}, inline: {metrics['inline_equations']}"),
        ("PDF Labeled Equations", pdf_labeled if pdf_labeled else "N/A",
         "standalone number labels found in the PDF text layer"),
        ("Extraction Coverage",  coverage_val, coverage_sub),
        ("Detection Rate",
         f"{metrics['detection_rate_pct']}%" if metrics['layout_equation_regions'] else "N/A",
         (f"{metrics['display_equations']} / {metrics['layout_equation_regions']} layout regions")
         if metrics['layout_equation_regions']
         else "no layout equation regions (label-scan mode)"),
        ("LaTeX Generation",     f"{metrics['latex_generation_rate_pct']}%",
         (f"{metrics['latex_generated']} / {metrics['latex_pool_size']} have LaTeX (inline only)"
          if metrics.get("all_inline") else
          f"{metrics['latex_generated']} / {metrics['display_equations']} have LaTeX")),
        ("LaTeX Validity",       f"{metrics['latex_validity_rate_pct']}%",
         f"{metrics['latex_valid']} valid, {metrics['latex_invalid']} invalid"),
        ("High Confidence",      f"{metrics['high_confidence_rate_pct']}%",
         (f"{metrics['high_confidence_recognition']} / {metrics['latex_pool_size']} (inline only)"
          if metrics.get("all_inline") else
          f"{metrics['high_confidence_recognition']} / {metrics['display_equations']} display")),
        ("No Image (fallback)",  metrics["no_image_fallback"],
         "equations where PDF render also failed"),
    ]
    for title, val, sub in cards:
        lines.append(
            f'<div class="metric-card"><h3>{title}</h3>'
            f'<div class="val">{val}</div><div class="sub">{sub}</div></div>'
        )
    lines.append("</div>")

    lines.append("<h2>Category Distribution</h2><table><tr><th>Category</th><th>Count</th></tr>")
    for cat, cnt in sorted(metrics["category_distribution"].items(), key=lambda x: -x[1]):
        lines.append(f"<tr><td>{cat}</td><td>{cnt}</td></tr>")
    lines.append("</table>")

    lines.append("<h2>Confidence Distribution</h2><table><tr><th>Bucket</th><th>Count</th></tr>")
    for bucket, cnt in metrics["confidence_distribution"].items():
        lines.append(f"<tr><td>{bucket}</td><td>{cnt}</td></tr>")
    lines.append("</table>")

    if metrics["validation_flags"]:
        lines.append("<h2>Validation Flags</h2><table><tr><th>Flag</th><th>Count</th></tr>")
        for flag, cnt in sorted(metrics["validation_flags"].items(), key=lambda x: -x[1]):
            lines.append(f"<tr><td>{flag}</td><td>{cnt}</td></tr>")
        lines.append("</table>")

    lines.append(
        f"<h2>Display Equations — Visual Comparison "
        f"(showing {min(len(display_eqs), max_visual)} of {len(display_eqs)})</h2>"
    )
    lines.append("""<table>
<tr>
  <th style="width:60px">ID</th>
  <th style="width:50px">Page</th>
  <th style="width:100px">Category</th>
  <th style="width:70px">Status</th>
  <th style="width:80px">Provider</th>
  <th style="width:320px">PDF Crop</th>
  <th style="width:320px">Extracted LaTeX (rendered)</th>
  <th style="width:200px">LaTeX Source</th>
  <th>Flags</th>
</tr>""")

    for eq in display_eqs[:max_visual]:
        page_no = eq["page_no"]
        bbox    = eq.get("bbox") or []
        latex   = eq.get("latex") or ""
        flags   = eq.get("validation_flags") or []
        status_cls   = _status_class(eq)
        status_label = {"pass": "✓ accept", "fail": "✗ reject", "warn": "~ review"}[status_cls]

        # LLM judge scores for the status cell
        llm_overall      = eq.get("llm_overall")
        llm_completeness = eq.get("llm_completeness")
        llm_lq           = eq.get("llm_latex_quality")
        llm_relevance    = eq.get("llm_relevance")
        llm_issues       = eq.get("llm_issues") or []
        if llm_overall is not None:
            issues_html = (
                f'<br><small style="color:#f87171">{"; ".join(llm_issues[:2])}</small>'
                if llm_issues else ""
            )
            status_detail = (
                f'{status_label}<br>'
                f'<small>overall {llm_overall}/10 | '
                f'comp {llm_completeness}/10 | '
                f'latex {llm_lq}/10 | '
                f'rel {llm_relevance}/10</small>'
                f'{issues_html}'
            )
        else:
            conf = eq.get("recognition_confidence", 0)
            status_detail = f'{status_label}<br><small>conf {conf:.2f}</small>'

        bbox_list = [bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]] if isinstance(bbox, dict) else bbox
        crop_bytes = _render_pdf_crop(pdf_path, page_no, bbox_list) if len(bbox_list) == 4 else None
        crop_html  = (
            f'<img class="crop" src="{_png_to_data_uri(crop_bytes)}" alt="eq crop">'
            if crop_bytes else "<em>no crop</em>"
        )

        if latex:
            escaped      = latex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            rendered_html = f'<div class="mathjax-render">\\({escaped}\\)</div>'
        else:
            rendered_html = "<em>no latex</em>"

        latex_code = f"<code>{latex[:200]}</code>" if latex else "<em>—</em>"
        provider   = eq.get("selected_provider") or "—"

        lines.append(f"""<tr class="eq-row">
  <td style="font-size:11px">{eq['equation_id']}</td>
  <td>{page_no}</td>
  <td style="font-size:12px">{eq.get('category','')}</td>
  <td class="{status_cls}">{status_detail}</td>
  <td style="font-size:11px">{provider}</td>
  <td>{crop_html}</td>
  <td>{rendered_html}</td>
  <td>{latex_code}</td>
  <td>{_flag_html(flags)}</td>
</tr>""")

    lines.append("</table>")

    lines.append(f"<h2>Inline Equations ({len(inline_eqs)} total — text-only, no image recognition)</h2>")
    lines.append("<table><tr><th>ID</th><th>Page</th><th>Category</th><th>Fragment</th><th>Flags</th></tr>")
    for eq in inline_eqs[:100]:
        plain = (eq.get("plain_text") or "")[:120]
        lines.append(
            f"<tr><td style='font-size:11px'>{eq['equation_id']}</td><td>{eq['page_no']}</td>"
            f"<td style='font-size:12px'>{eq.get('category','')}</td>"
            f"<td><code>{plain}</code></td><td>{_flag_html(eq.get('validation_flags') or [])}</td></tr>"
        )
    if len(inline_eqs) > 100:
        lines.append(f"<tr><td colspan='5'><em>… {len(inline_eqs)-100} more inline equations not shown</em></td></tr>")
    lines.append("</table>")

    lines.append(_HTML_TAIL)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(equations: list[dict], out_path: Path) -> None:
    fieldnames = [
        "equation_id", "page_no", "is_inline", "category",
        "classification_confidence", "selected_provider",
        "recognition_confidence", "has_latex", "latex_valid",
        "quality_score",
        # LLM-as-a-Judge columns (populated when GPT evaluation ran)
        "llm_verdict", "llm_overall", "llm_completeness",
        "llm_latex_quality", "llm_relevance", "llm_issues",
        "plain_text_len", "latex_len", "validation_flags", "notes",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for eq in equations:
            flags     = eq.get("validation_flags") or []
            has_latex = bool(eq.get("latex"))
            writer.writerow({
                "equation_id":               eq["equation_id"],
                "page_no":                   eq["page_no"],
                "is_inline":                 eq["is_inline"],
                "category":                  eq.get("category", ""),
                "classification_confidence": eq.get("classification_confidence", 0),
                "selected_provider":         eq.get("selected_provider", ""),
                "recognition_confidence":    eq.get("recognition_confidence", 0),
                "has_latex":                 has_latex,
                "latex_valid":               has_latex and "invalid_latex" not in flags,
                "quality_score":             eq.get("quality_score", ""),
                "llm_verdict":               eq.get("llm_verdict", ""),
                "llm_overall":               eq.get("llm_overall", ""),
                "llm_completeness":          eq.get("llm_completeness", ""),
                "llm_latex_quality":         eq.get("llm_latex_quality", ""),
                "llm_relevance":             eq.get("llm_relevance", ""),
                "llm_issues":                "|".join(eq.get("llm_issues") or []),
                "plain_text_len":            len(eq.get("plain_text") or ""),
                "latex_len":                 len(eq.get("latex") or ""),
                "validation_flags":          "|".join(flags),
                "notes":                     "|".join(eq.get("notes") or []),
            })


def run_validate(
    pdf_path: Path,
    sidecar_path: Path,
    out_dir: Path | None = None,
    max_visual: int = 200,
    open_report: bool = False,
) -> Path | None:
    """Generate HTML + CSV validation report from an existing sidecar JSON."""
    out_dir = (out_dir or VALIDATION_DIR) / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PDF              : {pdf_path.name}")
    print(f"Extraction JSON  : {sidecar_path}")

    data      = json.loads(sidecar_path.read_text(encoding="utf-8"))
    equations: list[dict] = data.get("equations") or []
    pages_data: list[dict] = data.get("pages") or []
    stats     = data.get("statistics") or {}

    # Normalise new-format equations (produced by output_formatter) so the
    # rest of run_validate can use the same field names as the old format.
    if data.get("document"):  # new-format sentinel
        total_pages = data["document"].get("total_pages", len(pages_data))
        normalised: list[dict] = []
        for eq in equations:
            orig = (eq.get("detection") or {}).get("original_bbox") or {}
            x, y = float(orig.get("x", 0)), float(orig.get("y", 0))
            w, h = float(orig.get("width", 0)), float(orig.get("height", 0))
            ocr   = eq.get("ocr") or {}
            final = eq.get("final") or {}
            cv    = eq.get("crop_validation") or {}
            normalised.append({
                "equation_id":            eq.get("equation_id", ""),
                "equation_number":        eq.get("label"),
                "page_no":                eq.get("page_number", 0),
                "is_inline":              False,
                "category":               eq.get("category", ""),
                "classification_confidence": (eq.get("detection") or {}).get("confidence", 0.0),
                "bbox":                   {"x0": x, "y0": y, "x1": x + w, "y1": y + h},
                "plain_text":             None,
                "latex":                  ocr.get("latex"),
                "mathml":                 None,
                "recognition_confidence": ocr.get("confidence", 0.0),
                "overall_confidence":     final.get("overall_confidence", 0.0),
                "selected_provider":      ocr.get("model", ""),
                "validation_flags":       list(cv.get("issues") or []),
                "notes":                  [],
                "provenance":             {"source_extractors": [], "source_pages": []},
                "metadata":               {"corrected": False, "notes": {}},
            })
        equations = normalised
    else:
        total_pages = stats.get("total_pages", len(pages_data))

    # Pre-flag label-only extractions before scoring/reporting so the HTML and
    # CSV reflect this as a validation issue even when the LLM judge is not run.
    _label_re = re.compile(
        r"""^\s*(?:\(\s*[\d.A-Za-z-]{1,10}\s*\)|\[\s*[\d.A-Za-z-]{1,10}\s*\]
            |Eq(?:uation)?[.:]?\s*\d[\d.]*)\s*$""",
        re.VERBOSE | re.IGNORECASE,
    )
    for eq in equations:
        pt = (eq.get("plain_text") or "").strip()
        lt = (eq.get("latex") or "").strip()
        if pt or lt:
            pt_only = not pt or bool(_label_re.match(pt))
            lt_only = not lt or bool(_label_re.match(lt))
            if pt_only and lt_only:
                vf = list(eq.get("validation_flags") or [])
                if "label_only_content" not in vf:
                    vf.append("label_only_content")
                    eq["validation_flags"] = vf

    display_count   = sum(1 for e in equations if not e["is_inline"])
    layout_eq_count = display_count
    layout_json     = pdf_path.with_suffix(".layout.json")
    if layout_json.exists():
        try:
            ldata = json.loads(layout_json.read_text(encoding="utf-8"))
            lctx  = ldata.get("context") or ldata
            layout_eq_count = sum(
                1 for page in (lctx.get("pages") or [])
                for region in (page.get("regions") or [])
                if region.get("region_type") == "equation"
            )
        except Exception:
            pass

    metrics = _compute_metrics(equations, layout_equation_count=layout_eq_count)
    metrics["total_pages"] = total_pages

    # Scan the source PDF for standalone equation labels to get the expected count.
    print("Scanning PDF for labeled equations …")
    pdf_labeled = _scan_pdf_labeled_count(pdf_path)
    extracted_display = metrics["display_equations"]
    pdf_coverage_pct = (
        round(min(extracted_display, pdf_labeled) / pdf_labeled * 100, 1)
        if pdf_labeled > 0 else None
    )
    metrics["pdf_labeled_count"] = pdf_labeled
    metrics["pdf_coverage_pct"] = pdf_coverage_pct

    html_path = out_dir / "equation_validation_report.html"
    csv_path  = out_dir / "equation_validation.csv"
    metrics_json_path = out_dir / "equation_validation_metrics.json"

    # ── LLM-as-a-Judge evaluation (runs BEFORE HTML/CSV so scores appear in report) ──
    judge_block: dict | None = None
    try:
        from pipeline.config import KNOVEL_PORTKEY_API_KEY, KNOVEL_PORTKEY_BASE_URL
        if KNOVEL_PORTKEY_API_KEY and KNOVEL_PORTKEY_BASE_URL:
            from quality.llm_judge import LLMJudge
            print("\nRunning LLM-as-a-Judge evaluation …")
            judge = LLMJudge()
            batch = judge.evaluate_equations(equations, pdf_path=pdf_path)
            print(batch.summary())

            # Write per-equation LLM scores back onto equation dicts so the HTML
            # report and CSV reflect GPT verdicts, not just the rule-based scorer.
            judgment_map = {j.equation_id: j for j in batch.judgments}
            for eq in equations:
                j = judgment_map.get(eq.get("equation_id", ""))
                if j:
                    eq["llm_verdict"]       = j.verdict
                    eq["llm_overall"]       = j.overall
                    eq["llm_completeness"]  = j.completeness
                    eq["llm_latex_quality"] = j.latex_quality
                    eq["llm_relevance"]     = j.relevance_score
                    eq["llm_issues"]        = j.issues

            # Recompute quality distribution from LLM verdicts (overrides rule-based)
            lj_accept = batch.accepted
            lj_review = batch.reviewed
            lj_reject = batch.rejected
            metrics["quality_distribution"] = {
                "good": lj_accept, "warn": lj_review, "fail": lj_reject
            }
            metrics["latex_quality_score_pct"] = round(batch.mean_overall * 10, 1)

            judge_block = {
                "coverage_verdict":  batch.coverage_verdict,
                "missing_labels":    batch.missing_labels,
                "missing_count":     len(batch.missing_labels),
                "mean_overall":      batch.mean_overall,
                "mean_relevance":    batch.mean_relevance,
                "mean_confidence":   batch.mean_confidence,
                "accepted":          batch.accepted,
                "reviewed":          batch.reviewed,
                "rejected":          batch.rejected,
                "total":             batch.total,
            }
        else:
            print("\n[LLM judge] Skipped — KNOVEL_PORTKEY_API_KEY / BASE_URL not set.")
    except Exception as _judge_exc:
        print(f"\n[LLM judge] Failed: {_judge_exc}", file=sys.stderr)

    print("\nRendering equation crops from PDF — this may take a moment ...")
    _write_html_report(equations, metrics, pdf_path, html_path, max_visual=max_visual)
    _write_csv(equations, csv_path)

    base_metrics = {
        "pdf_labeled_count": pdf_labeled,
        "pdf_coverage_pct": pdf_coverage_pct,
        "total_equations": metrics["total_equations"],
        "display_equations": metrics["display_equations"],
        "inline_equations": metrics["inline_equations"],
        "latex_validity_rate_pct": metrics["latex_validity_rate_pct"],
        "latex_quality_score_pct": metrics["latex_quality_score_pct"],
        "needs_review_count": metrics["needs_review_count"],
        "quality_distribution": metrics["quality_distribution"],
        "pages_with_equations": metrics["pages_with_equations"],
        "llm_judge": judge_block,
    }
    metrics_json_path.write_text(json.dumps(base_metrics, indent=2), encoding="utf-8")

    _print_validation_report(metrics, html_path, csv_path, pdf_path)

    if open_report:
        import webbrowser
        webbrowser.open(html_path.as_uri())

    return html_path


def _print_validation_report(metrics: dict, html_path: Path, csv_path: Path, pdf_path: Path) -> None:
    print("\n" + "=" * 60)
    print("  EQUATION EXTRACTION ACCURACY REPORT")
    print("=" * 60)
    print(f"  PDF                     : {pdf_path.name}")
    print(f"  Total pages             : {metrics.get('total_pages', '?')}")
    print(f"  Total equations found   : {metrics['total_equations']}")
    print(f"    Display (layout)      : {metrics['display_equations']}")
    print(f"    Inline (text scan)    : {metrics['inline_equations']}")
    pdf_labeled = metrics.get("pdf_labeled_count", 0)
    if pdf_labeled:
        pdf_cov = metrics.get("pdf_coverage_pct")
        cov_str = f"{pdf_cov}%" if pdf_cov is not None else "N/A"
        print(f"  PDF labeled equations   : {pdf_labeled}  (coverage: {cov_str})")
    print()
    print("  --- Detection ---")
    print(f"  Layout equation regions : {metrics['layout_equation_regions']}")
    print(f"  Display equations found : {metrics['display_equations']}")
    rate = metrics["detection_rate_pct"]
    print(f"  Detection rate          : {'✓' if rate == 100.0 else '~'} {rate}%")
    pool_size = metrics.get("latex_pool_size", metrics["display_equations"])
    pool_label = "all equations — inline only" if metrics.get("all_inline") else "display equations only"
    print()
    print(f"  --- LaTeX Recognition ({pool_label}) ---")
    print(f"  LaTeX generated         : {metrics['latex_generated']} / {pool_size}  ({metrics['latex_generation_rate_pct']}%)")
    print(f"  LaTeX valid             : {metrics['latex_valid']} / {metrics['latex_generated']}  ({metrics['latex_validity_rate_pct']}%)")
    print(f"  LaTeX invalid           : {metrics['latex_invalid']}")
    print(f"  High-confidence recog.  : {metrics['high_confidence_recognition']} / {pool_size}  ({metrics['high_confidence_rate_pct']}%)")
    if metrics["no_image_fallback"]:
        print(f"  No image (fallback)     : {metrics['no_image_fallback']} equations")
    print()
    print("  --- Confidence Distribution ---")
    for bucket, cnt in metrics["confidence_distribution"].items():
        pct = round(cnt / metrics["total_equations"] * 100, 1) if metrics["total_equations"] else 0
        print(f"    {bucket:<22} : {cnt:>4}  ({pct}%)")
    print()
    print("  --- Category Distribution ---")
    for cat, cnt in sorted(metrics["category_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {cat:<35} : {cnt}")
    if metrics["validation_flags"]:
        print()
        print("  --- Validation Flags ---")
        for flag, cnt in sorted(metrics["validation_flags"].items(), key=lambda x: -x[1]):
            print(f"    {flag:<35} : {cnt}")
    print()
    print("  --- Output ---")
    print(f"  HTML report             : {html_path}")
    print(f"  CSV summary             : {csv_path}")
    print("=" * 60)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  UTILITIES                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def detect_mode(pdf_path: Path, sample_pages: int = 50) -> str:
    """Return 'labeled' if Eq. X.X.X labels are found, else 'full'."""
    try:
        from pipeline.pdf_backend import open_document
        with open_document(str(pdf_path)) as doc:
            for pno in range(min(len(doc), sample_pages)):
                # PyMuPDF blocks: (x0, y0, x1, y1, text, block_no, block_type)
                for x0, y0, x1, y1, btext, *_ in doc[pno].get_text("blocks"):
                    if _is_label(btext.strip()):
                        return "labeled"
    except Exception:
        pass
    return "full"


def find_sidecar(stem: str) -> Path | None:
    """Locate the equation_extraction.json sidecar, preferring the flat output format."""
    candidates = [
        OUTPUT_DIR / stem / "equation_extraction.json",
        DATA_DIR / "input" / f"{stem}.equation_extraction.json",
    ]
    return next((p for p in candidates if p.exists()), None)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Knovel equation toolkit — extract and/or validate equations in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--pdf",       type=Path, help="Single PDF to process")
    src.add_argument("--input-dir", type=Path, help="Directory of PDFs (batch mode, uses 'full' pipeline)")

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode", choices=["auto", "labeled", "full"], default="auto",
        help="Extraction strategy (default: auto-detect)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Skip extraction; generate report from existing sidecar JSON",
    )

    # ── Post-extraction actions ────────────────────────────────────────────────
    parser.add_argument("--validate",    action="store_true", help="Run validation report after extraction")
    parser.add_argument("--open-report", action="store_true", help="Open the HTML report in a browser")

    # ── Validate options ──────────────────────────────────────────────────────
    parser.add_argument("--extraction-json", type=Path, help="Override sidecar path for --validate-only")
    parser.add_argument("--max-visual",      type=int, default=200,
                        help="Max equations shown in visual comparison (default: 200)")
    parser.add_argument("--output-dir",      type=Path, default=OUTPUT_DIR,
                        help="Root output directory for extraction (default: data/output)")

    # ── Labeled-mode options ──────────────────────────────────────────────────
    parser.add_argument(
        "--no-latex", action="store_true",
        help="Skip VLM LaTeX generation in labeled mode (fast run, no Ollama required)",
    )

    # ── Full-pipeline options ─────────────────────────────────────────────────
    parser.add_argument("--no-inline",    action="store_true", help="Disable inline equation detection")
    parser.add_argument("--mathml",       action="store_true", help="Convert LaTeX to MathML")
    parser.add_argument("--debug-dump",   action="store_true", help="Write debug JSON alongside sidecar")
    parser.add_argument("--provider-map", type=str, default="",
                        help='Override provider per category, e.g. "mathematical_equation=generic"')

    return parser


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    if not args.pdf and not args.input_dir:
        parser.error("Provide --pdf or --input-dir")

    # ── Batch mode (always full pipeline) ─────────────────────────────────────
    if args.input_dir:
        input_dir  = args.input_dir.expanduser().resolve()
        pdfs       = sorted(input_dir.glob("*.pdf"))
        if not pdfs:
            print(f"No PDF files found in {input_dir}", file=sys.stderr)
            return 1
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Batch mode — {len(pdfs)} PDFs — full pipeline")
        total, failures = 0, []
        for pdf_path in pdfs:
            print(f"\nProcessing: {pdf_path.name} ...")
            try:
                summary = run_full_pipeline(
                    pdf_path, output_dir,
                    inline=not args.no_inline, mathml=args.mathml,
                    debug_dump=args.debug_dump, provider_map=args.provider_map,
                )
                print_full_summary(summary)
                total += summary["total_equations"]
                if summary["outcome"] == "failed":
                    failures.append(pdf_path.name)
                if (args.validate or args.open_report) and summary["outcome"] != "failed":
                    sidecar = Path(summary["result_path"])
                    run_validate(pdf_path, sidecar, max_visual=args.max_visual, open_report=args.open_report)
            except Exception as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                failures.append(pdf_path.name)
        print(f"\n{'='*60}\nDone. Total equations : {total}")
        if failures:
            print(f"Failed ({len(failures)}): {', '.join(failures)}")
            return 2
        return 0

    # ── Single PDF ─────────────────────────────────────────────────────────────
    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    # ── Validate-only mode ────────────────────────────────────────────────────
    if args.validate_only:
        sidecar = args.extraction_json or find_sidecar(pdf_path.stem)
        if not sidecar or not sidecar.exists():
            print(
                f"ERROR: no sidecar found for '{pdf_path.stem}'.\n"
                "Run without --validate-only first to extract, or pass --extraction-json.",
                file=sys.stderr,
            )
            return 1
        report = run_validate(pdf_path, sidecar, max_visual=args.max_visual, open_report=args.open_report)
        return 0 if report else 1

    # ── Choose extraction mode ────────────────────────────────────────────────
    mode = args.mode
    if mode == "auto":
        print(f"Auto-detecting mode for {pdf_path.name} ...")
        mode = detect_mode(pdf_path)
        print(f"  → mode: {mode}")

    sidecar_path: Path | None = None

    if mode == "labeled":
        print(f"\nScanning {pdf_path.name} for Eq. X.X.X labels …\n")
        equations = extract_labeled(pdf_path)
        if not equations:
            print("\nNo 'Eq. X.X.X' labeled equations found.", file=sys.stderr)
            return 1
        text_count = sum(1 for e in equations if e.get("plain_text"))
        print(f"\nTotal: {len(equations)} labeled equations")
        print(f"  Text-layer extracted : {text_count}")
        print(f"  Image-only (visual)  : {len(equations) - text_count}")
        if not args.no_latex:
            _enrich_with_latex(equations)

        # Apply confidence estimation (normally done inside build_sidecar).
        # Must run before saving crops so layout scores are available.
        from equation_extraction.confidence_estimation import ConfidenceEstimator
        _apply_confidence(equations)

        # Save crops and write new-format JSON before build_sidecar strips PNGs.
        from equation_extraction.output_formatter import (
            format_labeled_output,
            save_labeled_crops,
        )
        output_dir_labeled = args.output_dir.expanduser().resolve()
        book_out_labeled = output_dir_labeled / pdf_path.stem
        book_out_labeled.mkdir(parents=True, exist_ok=True)
        crops_dir_labeled = book_out_labeled / "crops"
        crop_info_labeled = save_labeled_crops(equations, crops_dir_labeled)
        labeled_output = format_labeled_output(equations, crop_info_labeled, pdf_path)
        result_path_labeled = book_out_labeled / "equation_extraction.json"
        result_path_labeled.write_text(
            json.dumps(labeled_output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n✓ equation_extraction.json → {result_path_labeled}")

        sidecar      = build_sidecar(equations)
        sidecar_path = result_path_labeled  # reuse the new-format file for validation

    else:  # full
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running full ML pipeline on {pdf_path.name} …")
        try:
            summary = run_full_pipeline(
                pdf_path, output_dir,
                inline=not args.no_inline, mathml=args.mathml,
                debug_dump=args.debug_dump, provider_map=args.provider_map,
            )
            print_full_summary(summary)
            sidecar_path = Path(summary["result_path"])
        except Exception as exc:
            print(f"ERROR: full pipeline failed: {exc}", file=sys.stderr)
            return 1

    # ── Optionally validate ───────────────────────────────────────────────────
    if (args.validate or args.open_report) and sidecar_path:
        run_validate(pdf_path, sidecar_path, max_visual=args.max_visual, open_report=args.open_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
