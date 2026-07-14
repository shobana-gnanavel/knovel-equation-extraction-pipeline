"""Equation Extraction stage entrypoint (feature 008).

Consumes a text-extracted document's ``LayoutContext`` (equation-typed regions + geometry),
``ReadingOrderContext`` (order, hierarchy, equation-number/caption associations), and
``TextExtractionContext`` (text blocks for inline detection and classification signals), and produces
an :class:`EquationExtractionContext`: one equation per equation region (display) plus inline
equations detected within text blocks, in the feature-006 reading order — classified, routed to a
configuration-driven provider, recognized into structured representations, numbered, related,
scored, and validated. Region images are sourced from the feature-004 corrected rasters
(``derived_artifact``); the stage never decides or runs OCR and never re-opens the PDF. Per-equation/
page failures are contained; a missing/failed Layout or Reading Order Context yields a document-level
``failed`` outcome without aborting the batch.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from equation_extraction import confidence as confidence_mod
from equation_extraction import validation
from equation_extraction.crop_split import MULTI_EQ_NOTES, split_stacked_crop
from equation_extraction.recognition_quality import RETRY_QUALITY_NOTES
from equation_extraction.classifier import classify_region
from equation_extraction.confidence_estimation import ConfidenceEstimator
from equation_extraction.detection import (
    detect_inline_spans,
    extract_label_number,
    extract_mixed_label_block,
    is_equation_region,
    is_isolated_equation_label,
    looks_like_standalone_formula,
)
from equation_extraction.detection import (
    region_text as region_text_for,
)
from equation_extraction.formula_detector import score_formula_candidate
from equation_extraction.numbering import build_equation_numbers
from equation_extraction.providers import EquationProvider
from equation_extraction.registry import (
    close_providers,
    provider_identities,
    resolve_providers,
)
from equation_extraction.relationships import build_caption_refs, build_continuation_refs
from equation_extraction.representations import assemble
from equation_extraction.selection import select_provider
from pipeline import config
from pipeline.models import (
    ClassificationContext,
    Equation,
    EquationExtractionContext,
    EquationExtractionStatistics,
    LayoutContext,
    LayoutRegion,
    PageEquationExtraction,
    PageReadingOrder,
    PreprocessingContext,
    Provenance,
    ReadingOrderContext,
    TextBlock,
    TextExtractionContext,
)

__all__ = ["extract_equations"]

logger = structlog.get_logger(__name__)

# Minimum score for an inline fragment to be accepted as an equation.
# Set to FORMULA_THRESHOLD (0.45): an inline fragment must clear the same bar as a
# standalone formula. Empirically (24-book Knovel corpus) the 0.35–0.45 band is
# almost entirely false positives — subscripted variable mentions (D_0, ∆εrel),
# unit strings (μm/m, ±5), method names (sin2ψ-method) and radiation labels (Co-Kα)
# all score ~0.38 from a single Greek/subscript char with no relational operator.
# Genuine inline math with an operator scores ≥ 0.48 (x^2 + y^2) and equations with
# a relation score ≥ 0.78 (Tmin = 1.66), so raising 0.35→0.45 removed ~370 junk
# inline "equations" across the corpus with no measurable loss of real ones.
_INLINE_SCORE_GATE: float = 0.45

# Operators that indicate a region text is a self-contained equation rather than a bare fragment.
_EQUATION_OP_RE = re.compile(
    r"[=≈≠≤≥<>+\-*/^]"  # relational / arithmetic operators
    r"|[→⇌⇒⇄↔⟶]"  # reaction arrows (Unicode)
    r"|-+>|<-+|--\+"  # ASCII / OCR reaction arrows
    r"|\\frac|\\sum|\\int"  # common LaTeX constructs
)
_PROSE_HEAVY_WORD_RE = re.compile(
    r"\b(?:the|this|that|these|those|from|with|into|then|becomes|figure|equation|"
    r"calculated|obtained|region|value|using|terms|plastic|elastic|where|for)\b",
    re.IGNORECASE,
)
_PROSE_PREFIX_RE = re.compile(
    r"^(?:figure\b|equation\b|eq\.?\b|in the\b|where\b|from\b|then\b)",
    re.IGNORECASE,
)


def _is_meaningful_equation_text(text: str) -> bool:
    """Return True when *text* is a self-contained equation, not a bare product/reactant token.

    A fragment like ``'3COt'`` or ``'2.5H2O'`` that the layout tagger placed in its own
    equation region is not a useful standalone equation: it is a single stoichiometric term
    broken out of a larger balanced reaction.  We require at least one operator (relational,
    arithmetic, or reaction arrow) to consider a region worth extracting on its own.

    Exception: if the text spans multiple lines, each line may be a term in a stacked
    fraction — keep it regardless of operators. Multi-token text (containing whitespace)
    is also kept: a chemical structure description such as ``'C6H6 benzene ring structure'``
    has no operator but is a legitimate standalone region. The gate targets only bare
    single-token fragments (``'3COt'``, ``'2.5H2O'``) that carry neither an operator nor
    any accompanying descriptive tokens.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Multi-line blocks are always kept (stacked fractions, multi-line reactions).
    if "\n" in stripped:
        return True
    # An operator marks a self-contained equation/reaction.
    if _EQUATION_OP_RE.search(stripped):
        return True
    # Multi-token text (whitespace-separated) is kept: a single bare token with no
    # operator is a broken-out stoichiometric fragment; anything wordier is a real
    # region (e.g. a chemical structure description).
    return bool(re.search(r"\s", stripped))


# Roles whose text blocks are NOT scanned for inline equations.
# Structural/navigational roles (headings, headers, footers, page numbers) are excluded because
# they routinely contain numbered section references (e.g. "2-2.1 GRAM FORMULA WEIGHT") that
# superficially match chemical/mathematical patterns but are not equations.
_INLINE_SKIP_ROLES = frozenset(
    {
        "code_block",
        "heading",
        "document_title",
        "subheading",
        "header",
        "footer",
        "page_number",
    }
)


def _bbox_list(region: LayoutRegion) -> list[float]:
    if isinstance(region.bbox, dict):
        return [
            float(region.bbox.get("x0", 0.0)),
            float(region.bbox.get("y0", 0.0)),
            float(region.bbox.get("x1", 0.0)),
            float(region.bbox.get("y1", 0.0)),
        ]
    if isinstance(region.bbox, (list, tuple)) and len(region.bbox) == 4:
        return [float(v) for v in region.bbox]
    return [0.0, 0.0, 0.0, 0.0]


def _valid_bbox(bbox: list[float]) -> bool:
    if len(bbox) != 4:
        return False
    x0, y0, x1, y1 = bbox
    return x1 > x0 and y1 > y0 and x0 >= 0 and y0 >= 0


@dataclass(frozen=True)
class _CropQuality:
    """Minimal crop diagnostics used to drive retry decisions."""

    touch_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class _LabelScanCandidate:
    """Assembled multi-block candidate recovered from an isolated equation label."""

    region_id: str
    region_ids: tuple[str, ...]
    region_type: str
    bbox: list[float]
    formula_text: str
    label_text: str
    reading_position: int
    page_position: int
    structural_parent_id: str | None


def _retry_padding_fractions(base_pad_frac: float) -> tuple[float, float, float, float]:
    """Return per-side retry padding fractions, preserving the old symmetric knob as a fallback."""
    fallback = max(0.0, float(base_pad_frac))
    left = max(0.0, float(getattr(config, "KNOVEL_EQUATION_CROP_PAD_LEFT_FRAC", fallback)))
    right = max(0.0, float(getattr(config, "KNOVEL_EQUATION_CROP_PAD_RIGHT_FRAC", fallback)))
    top = max(0.0, float(getattr(config, "KNOVEL_EQUATION_CROP_PAD_TOP_FRAC", fallback)))
    bottom = max(0.0, float(getattr(config, "KNOVEL_EQUATION_CROP_PAD_BOTTOM_FRAC", fallback)))
    return left, right, top, bottom


def _first_pass_padding_fractions() -> tuple[float, float, float, float]:
    """Return per-side first-pass padding fractions from config.

    Left is independently configurable because lhs_clipped is the dominant first-pass quality
    failure — Docling bboxes are typically flush with the leftmost ink, so the LHS variable is
    the first thing to be cut off when the crop is tight. Right/top/bottom share the symmetric
    FIRST_PASS_PAD_FRAC; add explicit env vars for those sides if the corpus warrants it.
    """
    sym = max(0.0, float(getattr(config, "KNOVEL_EQUATION_FIRST_PASS_PAD_FRAC", 0.08)))
    left = max(0.0, float(getattr(config, "KNOVEL_EQUATION_FIRST_PASS_PAD_LEFT_FRAC", 0.15)))
    return left, sym, sym, sym  # left, right, top, bottom


def _crop_touch_flags(image: Any) -> tuple[str, ...]:
    """Detect whether dark ink touches a crop border, indicating likely clipping."""
    if image is None or not hasattr(image, "convert"):
        return ()
    try:
        gray = image.convert("L")
        width, height = gray.size
        if width <= 0 or height <= 0:
            return ()
        edge_px = max(1, int(min(width, height) * 0.02))
        threshold = 220
        min_dark = 1
        flags: list[str] = []

        def _has_dark_band(box: tuple[int, int, int, int]) -> bool:
            band = gray.crop(box)
            return sum(1 for value in band.getdata() if value < threshold) >= min_dark

        if _has_dark_band((0, 0, edge_px, height)):
            flags.append("crop_touch:left")
        if _has_dark_band((max(0, width - edge_px), 0, width, height)):
            flags.append("crop_touch:right")
        if _has_dark_band((0, 0, width, edge_px)):
            flags.append("crop_touch:top")
        if _has_dark_band((0, max(0, height - edge_px), width, height)):
            flags.append("crop_touch:bottom")
        return tuple(flags)
    except Exception:
        return ()


def _merge_bboxes(*boxes: list[float]) -> list[float]:
    valid = [box for box in boxes if _valid_bbox(box)]
    if not valid:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _vertical_gap(a: list[float], b: list[float]) -> float:
    if not _valid_bbox(a) or not _valid_bbox(b):
        return float("inf")
    if a[3] < b[1]:
        return b[1] - a[3]
    if b[3] < a[1]:
        return a[1] - b[3]
    return 0.0


def _horizontal_overlap_ratio(a: list[float], b: list[float]) -> float:
    if not _valid_bbox(a) or not _valid_bbox(b):
        return 0.0
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    aw = max(1e-6, a[2] - a[0])
    bw = max(1e-6, b[2] - b[0])
    return overlap / min(aw, bw)


def _join_formula_text(parts: list[str]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = _deduplicate_lines(part or "").strip()
        if not text:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            lines.append(stripped)
    return "\n".join(lines)


def _relation_line_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if _EQUATION_OP_RE.search(line.strip()))


def _looks_prose_heavy(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return len(_PROSE_HEAVY_WORD_RE.findall(stripped)) >= 3 and not _EQUATION_OP_RE.search(stripped)


def _trim_assembled_formula_text(text: str) -> str:
    """Drop leading/trailing narrative lines from an assembled label-scan candidate."""
    raw_lines = [line.strip() for line in _deduplicate_lines(text).splitlines() if line.strip()]
    if not raw_lines:
        return ""

    kept: list[str] = []
    started = False
    for line in raw_lines:
        if is_isolated_equation_label(line):
            continue
        if not started:
            if _looks_prose_heavy(line) or _PROSE_PREFIX_RE.match(line):
                continue
            started = True
            kept.append(line)
            continue
        if _looks_prose_heavy(line):
            break
        kept.append(line)

    while kept and (_looks_prose_heavy(kept[-1]) or _PROSE_PREFIX_RE.match(kept[-1])):
        kept.pop()
    return "\n".join(kept).strip()


def _is_mergeable_formula_neighbor(
    *,
    text: str,
    bbox: list[float],
    current_bbox: list[float],
    current_text: str,
    label_bbox: list[float] | None,
    page_dims: tuple[float, float] | None,
    region_type: str,
) -> bool:
    stripped = _deduplicate_lines(text or "").strip()
    if not stripped or is_isolated_equation_label(stripped):
        return False
    if not _valid_bbox(bbox) or not _valid_bbox(current_bbox):
        return False

    label_distance: float | None = None
    if label_bbox and _valid_bbox(label_bbox):
        label_distance = abs(((bbox[1] + bbox[3]) / 2.0) - ((label_bbox[1] + label_bbox[3]) / 2.0))

    score = score_formula_candidate(
        stripped,
        bbox=bbox,
        page_dims=page_dims,
        label_distance_pts=label_distance,
        region_type=region_type,
    )
    if not (score.is_formula or score.needs_llm or looks_like_standalone_formula(stripped)):
        return False

    gap = _vertical_gap(current_bbox, bbox)
    overlap = _horizontal_overlap_ratio(current_bbox, bbox)
    max_height = max(current_bbox[3] - current_bbox[1], bbox[3] - bbox[1], 1.0)
    if gap > max(18.0, 0.75 * max_height):
        return False

    # If both the assembled candidate and the neighbor already look like standalone
    # equations with their own relation lines, treat a visible gap as a boundary
    # between two numbered equations rather than one fragmented equation.
    if (
        _relation_line_count(current_text) >= 1
        and _relation_line_count(stripped) >= 1
        and gap > 3.0
    ):
        return False

    if _looks_prose_heavy(stripped):
        return False

    if page_dims is not None and page_dims[0] > 0:
        page_w = page_dims[0]
        center_delta = abs(
            (((current_bbox[0] + current_bbox[2]) / 2.0) - ((bbox[0] + bbox[2]) / 2.0)) / page_w
        )
    else:
        center_delta = 0.0
    # Horizontally adjacent blocks (side-by-side with ≤ 8pt gap) form a single split
    # formula region (e.g. "S =" next to "AE/FS") and should always be merged.
    h_gap = max(0.0, bbox[0] - current_bbox[2], current_bbox[0] - bbox[2])
    return overlap >= 0.20 or center_delta <= 0.18 or h_gap <= 8.0


def _assemble_label_candidate(
    *,
    anchor_idx: int,
    label_idx: int,
    entries_sorted: list,
    region_by_id: dict[str, LayoutRegion],
    block_by_region: dict[str, TextBlock],
    extracted_ids: set[str],
    label_region_ids: set[str],
    page_dims: tuple[float, float] | None,
) -> _LabelScanCandidate | None:
    anchor_entry = entries_sorted[anchor_idx]
    anchor_region = region_by_id.get(anchor_entry.region_id)
    anchor_block = block_by_region.get(anchor_entry.region_id)
    if anchor_region is None or anchor_block is None:
        return None

    anchor_bbox = _bbox_list(anchor_region)
    label_region = region_by_id.get(entries_sorted[label_idx].region_id)
    label_bbox = _bbox_list(label_region) if label_region is not None else None
    selected_indices = {anchor_idx}
    current_bbox = list(anchor_bbox)
    current_text = _deduplicate_lines(anchor_block.text or "").strip()

    def _try_attach(idx: int) -> bool:
        nonlocal current_bbox, current_text
        if idx in selected_indices:
            return False
        entry = entries_sorted[idx]
        if entry.region_id in extracted_ids or entry.region_id in label_region_ids:
            return False
        region = region_by_id.get(entry.region_id)
        block = block_by_region.get(entry.region_id)
        if region is None or block is None:
            return False
        bbox = _bbox_list(region)
        if not _is_mergeable_formula_neighbor(
            text=block.text or "",
            bbox=bbox,
            current_bbox=current_bbox,
            current_text=current_text,
            label_bbox=label_bbox,
            page_dims=page_dims,
            region_type=region.region_type,
        ):
            return False
        selected_indices.add(idx)
        current_bbox = _merge_bboxes(current_bbox, bbox)
        current_text = _join_formula_text(
            [block_by_region[entries_sorted[i].region_id].text for i in sorted(selected_indices)]
        )
        return True

    # Grow toward the label first: the most common failure is a narrow anchor block
    # with adjacent numerator/denominator or LHS fragments between the anchor and label.
    for idx in range(anchor_idx + 1, label_idx):
        if not _try_attach(idx):
            break
    for idx in range(anchor_idx - 1, -1, -1):
        if entries_sorted[idx].page_no != anchor_entry.page_no:
            break
        if not _try_attach(idx):
            break

    selected = sorted(selected_indices)
    text = _join_formula_text([block_by_region[entries_sorted[idx].region_id].text for idx in selected])
    text = _trim_assembled_formula_text(text)
    if not text:
        text = _deduplicate_lines(anchor_block.text or "").strip()

    return _LabelScanCandidate(
        region_id=anchor_entry.region_id,
        region_ids=tuple(entries_sorted[idx].region_id for idx in selected),
        region_type=anchor_region.region_type,
        bbox=current_bbox,
        formula_text=text,
        label_text=_deduplicate_lines(block_by_region[entries_sorted[label_idx].region_id].text or "").strip(),
        reading_position=min(entries_sorted[idx].reading_position for idx in selected),
        page_position=min(entries_sorted[idx].page_position for idx in selected),
        structural_parent_id=anchor_entry.structural_parent_id,
    )


def _load_region_image(
    raster_cache: dict[int, Any],
    preprocessing: PreprocessingContext | None,
    page_no: int,
    geom: tuple[float, float] | None,
    bbox: list[float],
    pdf_path: Path | None = None,
    *,
    pad_frac: float = 0.0,
    zoom: float | None = None,
    fp_raster_cache: dict[int, Any] | None = None,
) -> Any:
    """Best-effort region crop from the feature-004 corrected raster with PDF render fallback.

    Three rendering paths, selected by the arguments provided:

    * First-pass (``fp_raster_cache`` is not None): renders fresh at
      ``KNOVEL_EQUATION_FIRST_PASS_ZOOM``, caches the full-page render in ``fp_raster_cache``
      so all equations on the same page share one render, and applies
      ``_first_pass_padding_fractions()`` to the crop.  Falls back to the preprocessing raster
      when ``pdf_path`` is unavailable.

    * Retry (``zoom`` is not None): renders fresh at the given ``zoom`` (not cached — retries
      are rare and each retry may use a different zoom) and applies ``_retry_padding_fractions``
      from ``pad_frac``.

    * Standard (both None): uses the preprocessing corrected raster (``derived_artifact``),
      cached in ``raster_cache``.  For digital passthrough pages where no raster exists, falls
      back to an on-demand render at the default PDF backend zoom.  ``pad_frac`` is ignored.

    Returns a cropped PIL image or ``None`` when all sources fail.
    """
    if geom is None or not _valid_bbox(bbox):
        return None

    if fp_raster_cache is not None:
        # First-pass path: render once per page at FIRST_PASS_ZOOM and cache.
        if page_no not in fp_raster_cache:
            fp_zoom = float(getattr(config, "KNOVEL_EQUATION_FIRST_PASS_ZOOM", 3.0))
            rendered = (
                _render_pdf_page(pdf_path, page_no, zoom=fp_zoom) if pdf_path is not None else None
            )
            if rendered is None:
                rendered = _open_raster(preprocessing, page_no)
            fp_raster_cache[page_no] = rendered
        image = fp_raster_cache[page_no]
    elif zoom is not None:
        # Retry path: render fresh at the requested zoom; not cached.
        image = _render_pdf_page(pdf_path, page_no, zoom=zoom) if pdf_path is not None else None
    else:
        # Standard path: prefer preprocessing raster, fall back to on-demand render.
        if page_no not in raster_cache:
            raster = _open_raster(preprocessing, page_no)
            if raster is None and pdf_path is not None:
                raster = _render_pdf_page(pdf_path, page_no)
            raster_cache[page_no] = raster
        image = raster_cache[page_no]

    if image is None:
        return None
    page_w, page_h = geom
    if page_w <= 0 or page_h <= 0:
        return None
    try:
        px_w, px_h = image.size
        sx, sy = px_w / page_w, px_h / page_h
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        if fp_raster_cache is not None:
            left_pad, right_pad, top_pad, bottom_pad = _first_pass_padding_fractions()
        elif pad_frac:
            left_pad, right_pad, top_pad, bottom_pad = _retry_padding_fractions(pad_frac)
        else:
            left_pad = right_pad = top_pad = bottom_pad = 0.0
        if left_pad or right_pad or top_pad or bottom_pad:
            dx, dy = (x1 - x0), (y1 - y0)
            x0 = max(0.0, x0 - dx * left_pad)
            y0 = max(0.0, y0 - dy * top_pad)
            x1 = min(page_w, x1 + dx * right_pad)
            y1 = min(page_h, y1 + dy * bottom_pad)
        box = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        cropped = image.crop(box)
        if cropped.size[0] == 0 or cropped.size[1] == 0:
            return None
        return cropped
    except Exception:  # cropping is best-effort; a failure just means no image for this region
        return None


def _open_raster(preprocessing: PreprocessingContext | None, page_no: int) -> Any:
    if preprocessing is None:
        return None
    artifact: str | None = None
    for page in preprocessing.pages:
        if page.page_no == page_no:
            artifact = page.derived_artifact
            break
    if not artifact or not Path(artifact).exists():
        return None
    try:
        from PIL import Image

        return Image.open(artifact).convert("RGB")
    except Exception:  # pragma: no cover - raster load is best-effort
        return None


def _render_pdf_page(pdf_path: Path, page_no: int, zoom: float | None = None) -> Any:
    """On-demand render of a single PDF page to a PIL image (fallback for digital passthrough pages).

    ``zoom`` overrides the default render scale for higher-resolution retry crops.
    """
    try:
        from PIL import Image

        from pipeline.pdf_backend import open_document, render_page_image

        with open_document(str(pdf_path)) as doc:
            page = doc[page_no]
            array = render_page_image(page) if zoom is None else render_page_image(page, zoom=zoom)
        return Image.fromarray(array).convert("RGB")
    except Exception as exc:  # pragma: no cover - best-effort render
        logger.debug("equation_page_render_failed", page_no=page_no, error=str(exc))
        return None


def _recognize_with_retry(
    provider: EquationProvider,
    *,
    image: Any,
    region_text: str,
    category: str,
    recrop: Callable[[float, float], Any] | None,
) -> Any:
    """Recognize once; retry with a padded, higher-resolution crop + strict prompt when the
    first pass scores below the retry threshold, keeping the higher-scoring attempt (FR-013/019).

    The retry is a no-op — returning the first result — when it is disabled, no image is
    available (inline), no re-crop callable was supplied, the first pass already scores at or
    above the threshold, or the re-crop yields no image. A retry that does not improve the score
    is discarded so a stronger prompt can never make an equation worse.
    """
    crop_quality = _CropQuality(touch_flags=_crop_touch_flags(image))
    result = provider.recognize(
        region_image=image,
        region_text=region_text,
        category=category,
        config=config,
    )
    if crop_quality.touch_flags:
        result.notes = list(result.notes) + list(crop_quality.touch_flags)
    should_retry_for_confidence = (
        result.confidence < config.KNOVEL_EQUATION_RECOGNITION_RETRY_THRESHOLD
    )
    should_retry_for_crop = bool(crop_quality.touch_flags)
    result_notes_set = set(result.notes)
    should_retry_for_quality = bool(RETRY_QUALITY_NOTES & result_notes_set)
    should_split = bool(MULTI_EQ_NOTES & result_notes_set)

    if (
        image is None
        or not config.KNOVEL_EQUATION_RETRY_ENABLED
        or (not should_retry_for_confidence and not should_retry_for_crop
                and not should_retry_for_quality)
    ):
        return result

    # ── Split path: two stacked equations detected → retry on the top sub-crop ──
    # Run this BEFORE the padded-image retry because the root cause is the crop
    # containing two equations, not crop tightness. If the split gives a better
    # result, return immediately; otherwise fall through to the standard retry.
    if should_split and image is not None:
        try:
            sub_crops = split_stacked_crop(image)
            if len(sub_crops) > 1:
                split_result = provider.recognize(
                    region_image=sub_crops[0],
                    region_text=region_text,
                    category=category,
                    config=config,
                    strict=True,
                )
                n = len(sub_crops)
                split_result.notes = list(split_result.notes) + [
                    f"split_crop:n={n}", "split_crop:applied"
                ]
                if split_result.confidence > result.confidence:
                    split_result.notes = list(split_result.notes) + ["split_crop:improved"]
                    return split_result
                result.notes = list(result.notes) + ["split_crop:no_improvement"]
        except Exception:
            pass

    # ── Standard padded / hi-res retry ──────────────────────────────────────────
    if recrop is None:
        return result
    padded = recrop(config.KNOVEL_EQUATION_CROP_PAD_FRAC, config.KNOVEL_EQUATION_RETRY_ZOOM)
    if padded is None:
        return result
    retry = provider.recognize(
        region_image=padded,
        region_text=region_text,
        category=category,
        config=config,
        strict=True,
    )
    if crop_quality.touch_flags:
        retry.notes = list(retry.notes) + list(crop_quality.touch_flags)
    if should_retry_for_crop:
        retry.notes = list(retry.notes) + ["recognition_retry:crop_touch"]
    if should_retry_for_quality:
        retry.notes = list(retry.notes) + ["recognition_retry:quality_issue"]
    if retry.confidence > result.confidence:
        retry.notes = list(retry.notes) + ["recognition_retry:improved"]
        return retry
    if should_retry_for_crop:
        result.notes = list(result.notes) + ["recognition_retry:crop_touch"]
    if should_retry_for_quality:
        result.notes = list(result.notes) + ["recognition_retry:quality_issue"]
    result.notes = list(result.notes) + ["recognition_retry:no_improvement"]
    return result


def _process_candidate(
    *,
    equation_id: str,
    region_id: str,
    region_type: str,
    page_no: int,
    bbox: list[float],
    is_inline: bool,
    text_block_id: str | None,
    region_text: str,
    section_context: str,
    reading_position: int,
    page_position: int,
    structural_parent_id: str | None,
    caption_ref: str | None,
    equation_number: str | None,
    continuation_ref: str | None,
    region_image: Any,
    providers: dict[str, EquationProvider],
    extra_notes: list[str],
    recrop: Callable[[float, float], Any] | None = None,
    confidence_estimator: ConfidenceEstimator | None = None,
) -> Equation:
    """Classify → select → recognize → assemble → relate → build one ``Equation``."""
    classification = classify_region(
        region_type=region_type,
        region_text=region_text,
        section_context=section_context,
        is_inline=is_inline,
    )
    provider_role = select_provider(classification.category, config=config)
    provider = providers.get(provider_role) or providers.get("generic")
    if provider is None:
        raise RuntimeError(
            f"No provider registered for role '{provider_role}' and 'generic' fallback is absent"
        )
    image = None if is_inline else region_image  # inline equations are text fragments, not crops
    result = _recognize_with_retry(
        provider,
        image=image,
        region_text=region_text,
        category=classification.category,
        recrop=None if is_inline else recrop,
    )

    # Run confidence estimation immediately after recognition.
    # The overall_confidence replaces the flat recognition score; the full
    # structured result is stored on the Equation for downstream consumers.
    confidence_result_dict: dict | None = None
    if confidence_estimator is not None:
        try:
            bbox_tuple = tuple(bbox[:4]) if len(bbox) >= 4 else None
            cr = confidence_estimator.estimate(
                latex=result.latex or "",
                crop_image=image,
                bbox=bbox_tuple,
                image_metadata=None,
                token_logprobs=None,
            )
            result.confidence = cr.overall_confidence
            confidence_result_dict = cr.to_dict()
        except Exception as _ce:
            logger.debug("confidence_estimation_failed", error=str(_ce))

    rep, rep_flags = assemble(result, category=classification.category, config=config)

    flags = list(rep_flags)
    notes = list(extra_notes) + list(result.notes)
    if not _valid_bbox(bbox):
        flags.append("invalid_bbox")
    # An ``unknown`` category has no recognition provider that understands it — it is routed to the
    # generic passthrough / manual review and reported as unsupported (FR-021/FR-027). A merely
    # *absent* optional model is recognition degradation, recorded via the ``provider_absent`` note
    # and the low-recognition-confidence flag, not an unsupported category.
    if classification.category == "unknown":
        flags.append("unsupported_category")
    # Broken multi-line: a multi-line math region the provider could not recognize into LaTeX.
    if (
        classification.category
        in {"mathematical_equation", "engineering_formula", "statistical_expression"}
        and "\n" in region_text
        and rep.latex is None
    ):
        flags.append("broken_multiline")

    return Equation(
        equation_id=equation_id,
        region_id=region_id,
        text_block_id=text_block_id,
        is_inline=is_inline,
        page_no=page_no,
        reading_position=reading_position,
        page_position=page_position,
        structural_parent_id=structural_parent_id,
        caption_ref=caption_ref,
        bbox=list(bbox),
        equation_number=equation_number,
        category=classification.category,
        classification_confidence=confidence_mod.clamp(classification.confidence),
        classification_reason=classification.reason,
        selected_provider=provider_role,
        plain_text=rep.plain_text,
        latex=rep.latex,
        mathml=rep.mathml,
        structured_form=rep.structured_form,
        recognition_confidence=confidence_mod.clamp(result.confidence),
        overall_confidence=confidence_result_dict["overall_confidence"] if confidence_result_dict else None,
        confidence_recognition=confidence_result_dict["recognition"] if confidence_result_dict else None,
        confidence_layout=confidence_result_dict["layout"] if confidence_result_dict else None,
        confidence_syntax=confidence_result_dict["syntax"] if confidence_result_dict else None,
        confidence_ocr_quality=confidence_result_dict["ocr_quality"] if confidence_result_dict else None,
        continuation_ref=continuation_ref,
        validation_flags=flags,
        provenance=Provenance(
            source_extractors=[getattr(provider, "backend", provider_role)], source_pages=[page_no]
        ),
        notes=notes,
        confidence_result=confidence_result_dict,
    )


def _validate_formula_with_llm(
    text: str,
    providers: dict[str, EquationProvider],
) -> bool:
    """Ask a provider's LLM oracle whether *text* is a mathematical formula.

    Routes through the equation-recognition provider layer (``is_equation``) rather than
    calling the backend HTTP API directly, so the orchestrator stays backend-agnostic and
    reuses the provider's pooled client. Falls back to ``False`` (conservative) when no
    provider offers an oracle or the call fails.
    """
    provider = providers.get("qwen_vl") or providers.get("generic")
    oracle = getattr(provider, "is_equation", None)
    if not callable(oracle):
        return False
    try:
        return bool(oracle(text, config=config))
    except Exception:  # provider oracle must never fail the detection pass
        return False


def _deduplicate_lines(text: str) -> str:
    """Remove duplicate adjacent lines that arise from two-column PDF text-layer duplication.

    Many scanned engineering PDFs extract each line of text twice in sequence — one copy per
    column — producing blocks like ``"wRT\\n\\np = -\\n\\nwRT\\n\\np = -"``.  Collapsing
    consecutive duplicate lines to a single occurrence restores the logical single-column text
    without losing any information (the duplicates are identical).  Empty lines are preserved
    only when they separate distinct content.
    """
    lines = text.splitlines()
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped not in seen:
            seen.add(stripped)
            result.append(line)
    return "\n".join(result).strip()


def _scan_label_candidates(
    entries_sorted: list,
    region_by_id: dict[str, LayoutRegion],
    block_by_region: dict[str, TextBlock],
    extracted_ids: set[str],
    *,
    geom_by_page: dict[int, tuple[float, float]] | None = None,
    providers: dict[str, EquationProvider] | None = None,
) -> list[_LabelScanCandidate]:
    """Second-pass label scan: find display equations Docling did not tag as 'equation' regions.

    Looks for text blocks that contain only an equation-number label (e.g. ``(12.2.1)``) and
    then scans backwards in reading order to find the nearest preceding block that passes the
    confidence-based formula scorer.  Returns assembled candidates whose bbox/text may span
    multiple neighboring fragments when the upstream layout split one visual equation across
    several blocks.  Mutates *extracted_ids* to prevent the same region being yielded twice.

    Text blocks are deduplicated via :func:`_deduplicate_lines` before evaluation.  Many
    scanned two-column PDFs produce blocks where each line is repeated twice; deduplication
    restores the single-column view that ``is_isolated_equation_label`` and
    ``score_formula_candidate`` expect.

    When ``geom_by_page`` is provided, the scorer receives the block bbox and page dimensions
    so that layout signals (width ratio, centering, label proximity) can contribute to the
    score.  Without geometry the scorer falls back to text-only mode.

    For blocks in the ambiguous zone (score ≥ ``AMBIGUOUS_THRESHOLD`` but below
    ``FORMULA_THRESHOLD``) the optional ``providers`` dict is used to call
    :func:`_validate_formula_with_llm` for a final yes/no decision.
    """
    label_entries: list[tuple[int, str, str]] = []  # (page_pos, region_id, label_text)
    for entry in entries_sorted:
        if entry.region_id in extracted_ids:
            continue
        block = block_by_region.get(entry.region_id)
        if block is None:
            continue
        text = _deduplicate_lines(block.text or "")
        if is_isolated_equation_label(text):
            label_entries.append((entry.page_position, entry.region_id, text))

    if not label_entries:
        return []

    label_region_ids = {rid for _, rid, _ in label_entries}
    entry_by_region = {entry.region_id: entry for entry in entries_sorted}
    index_by_region = {entry.region_id: idx for idx, entry in enumerate(entries_sorted)}
    candidates: list[_LabelScanCandidate] = []

    for label_pos, _label_rid, label_text in label_entries:
        # Resolve label geometry once per label entry so the inner loop is fast.
        label_region = region_by_id.get(_label_rid)
        label_bbox = _bbox_list(label_region) if label_region is not None else None
        label_cy = ((label_bbox[1] + label_bbox[3]) / 2.0) if label_bbox else None
        # Derive the page_no by looking at the label's entry.
        label_entry = entry_by_region.get(_label_rid)
        page_no = getattr(label_entry, "page_no", None)
        page_dims = geom_by_page.get(page_no) if (geom_by_page and page_no is not None) else None
        label_idx = index_by_region.get(_label_rid)
        previous_label_idx: int | None = None
        if label_idx is not None:
            for prev_idx in range(label_idx - 1, -1, -1):
                prev_entry = entries_sorted[prev_idx]
                if getattr(prev_entry, "page_no", None) != page_no:
                    break
                if prev_entry.region_id in label_region_ids:
                    previous_label_idx = prev_idx
                    break

        # Scan backwards from the label block to find the nearest preceding formula block.
        for idx in range(len(entries_sorted) - 1, -1, -1):
            entry = entries_sorted[idx]
            if getattr(entry, "page_no", None) != page_no:
                continue
            if previous_label_idx is not None and idx <= previous_label_idx:
                break
            if entry.page_position >= label_pos:
                continue
            if entry.region_id in extracted_ids or entry.region_id in label_region_ids:
                continue
            block = block_by_region.get(entry.region_id)
            if block is None:
                continue
            # Deduplicate before scoring to remove two-column line repetition.
            formula_text = _deduplicate_lines(block.text or "")

            # Build scoring context when geometry is available.
            cand_region = region_by_id.get(entry.region_id)
            cand_bbox = _bbox_list(cand_region) if cand_region is not None else None
            cand_region_type = cand_region.region_type if cand_region is not None else ""

            dist: float | None = None
            if label_cy is not None and cand_bbox is not None:
                cand_cy = (cand_bbox[1] + cand_bbox[3]) / 2.0
                dist = abs(cand_cy - label_cy)
                if dist > 120:
                    continue

            formula_score = score_formula_candidate(
                formula_text,
                bbox=cand_bbox,
                page_dims=page_dims,
                label_distance_pts=dist,
                region_type=cand_region_type,
            )

            if formula_score.is_formula:
                assembled = (
                    _assemble_label_candidate(
                        anchor_idx=idx,
                        label_idx=label_idx,
                        entries_sorted=entries_sorted,
                        region_by_id=region_by_id,
                        block_by_region=block_by_region,
                        extracted_ids=extracted_ids,
                        label_region_ids=label_region_ids,
                        page_dims=page_dims,
                    )
                    if label_idx is not None
                    else None
                )
                if assembled is None:
                    assembled = _LabelScanCandidate(
                        region_id=entry.region_id,
                        region_ids=(entry.region_id,),
                        region_type=cand_region_type or "text",
                        bbox=cand_bbox if cand_bbox is not None else [0.0, 0.0, 0.0, 0.0],
                        formula_text=formula_text,
                        label_text=label_text,
                        reading_position=entry.reading_position,
                        page_position=entry.page_position,
                        structural_parent_id=entry.structural_parent_id,
                    )
                candidates.append(assembled)
                extracted_ids.update(assembled.region_ids)
                extracted_ids.add(_label_rid)  # prevent mixed scan from re-processing the label block
                break
            elif formula_score.needs_llm and providers:
                if _validate_formula_with_llm(formula_text, providers):
                    assembled = (
                        _assemble_label_candidate(
                            anchor_idx=idx,
                            label_idx=label_idx,
                            entries_sorted=entries_sorted,
                            region_by_id=region_by_id,
                            block_by_region=block_by_region,
                            extracted_ids=extracted_ids,
                            label_region_ids=label_region_ids,
                            page_dims=page_dims,
                        )
                        if label_idx is not None
                        else None
                    )
                    if assembled is None:
                        assembled = _LabelScanCandidate(
                            region_id=entry.region_id,
                            region_ids=(entry.region_id,),
                            region_type=cand_region_type or "text",
                            bbox=cand_bbox if cand_bbox is not None else [0.0, 0.0, 0.0, 0.0],
                            formula_text=formula_text,
                            label_text=label_text,
                            reading_position=entry.reading_position,
                            page_position=entry.page_position,
                            structural_parent_id=entry.structural_parent_id,
                        )
                    candidates.append(assembled)
                    extracted_ids.update(assembled.region_ids)
                    extracted_ids.add(_label_rid)
                    break

    return candidates


def _scan_unlabeled_chemical_candidates(
    entries_sorted: list,
    region_by_id: dict[str, LayoutRegion],
    block_by_region: dict[str, TextBlock],
    extracted_ids: set[str],
    geom_by_page: dict[int, tuple[float, float]],
) -> list[tuple[str, str]]:
    """Fourth-pass scan: unlabeled standalone chemical equations.

    Targets non-equation-typed layout regions that carry a confirmed chemical reaction arrow
    (``→``, ``⇌``, ``--+``, etc.) AND score at or above ``FORMULA_THRESHOLD`` with the updated
    chemical signals.  The reaction-arrow pre-filter is intentional: it is an unambiguous signal
    that never appears in normal prose, so it protects against hallucinating equations from
    section headings, table cells, or other short non-equation blocks.

    This pass is necessary for chemistry/explosives/materials textbooks where reactions are
    presented as display equations without sequential equation numbers, making them invisible
    to the label-scan (second/third) passes.

    Returns ``(region_id, formula_text)`` tuples.  Mutates *extracted_ids*.
    """
    from equation_extraction.detection import _CHEMICAL_REACTION_ARROW  # noqa: PLC0415
    from equation_extraction.formula_detector import FORMULA_THRESHOLD  # noqa: PLC0415

    candidates: list[tuple[str, str]] = []
    for entry in entries_sorted:
        if entry.region_id in extracted_ids:
            continue
        region = region_by_id.get(entry.region_id)
        if region is None:
            continue
        # Equation-typed regions are processed in the first pass — skip them here.
        if is_equation_region(region.region_type):
            continue
        block = block_by_region.get(entry.region_id)
        if block is None:
            continue
        text = (block.text or "").strip()
        if not text:
            continue

        # Require an unambiguous chemical reaction arrow as a gating condition.
        # This prevents the fourth pass from promoting section headings or prose
        # blocks that happen to score marginally above the formula threshold.
        if not _CHEMICAL_REACTION_ARROW.search(text):
            continue

        page_no = getattr(entry, "page_no", None)
        page_dims = geom_by_page.get(page_no) if page_no is not None else None
        cand_bbox = _bbox_list(region)

        formula_score = score_formula_candidate(
            text,
            bbox=cand_bbox,
            page_dims=page_dims,
        )

        if formula_score.score >= FORMULA_THRESHOLD:
            candidates.append((entry.region_id, text))
            extracted_ids.add(entry.region_id)

    return candidates


def _scan_mixed_label_candidates(
    entries_sorted: list,
    region_by_id: dict[str, LayoutRegion],
    block_by_region: dict[str, TextBlock],
    extracted_ids: set[str],
    seen_eq_numbers: set[str],
) -> list[tuple[str, str, str]]:
    """Third-pass scan: find equations where label and formula share a single text block.

    Complements :func:`_scan_label_candidates` which only handles blocks whose entire text is
    an isolated equation label.  Here, blocks containing the label on one line and the formula
    on other lines (label-before, label-after, or label-in-middle patterns) are detected via
    :func:`extract_mixed_label_block`.

    Returns ``(region_id, formula_text, label_text)`` tuples — same shape as
    ``_scan_label_candidates`` — so the caller can handle both with the same loop.
    Mutates *extracted_ids* and *seen_eq_numbers* to prevent duplicate extractions across
    passes and across pages.
    """
    candidates: list[tuple[str, str, str]] = []
    for entry in entries_sorted:
        if entry.region_id in extracted_ids:
            continue
        block = block_by_region.get(entry.region_id)
        if block is None:
            continue
        # Deduplicate before mixed-block detection.  Two-column PDFs repeat each line
        # twice; without deduplication a label-only block like "(2-26)\n\n(2-26)" has
        # two non-empty lines and `extract_mixed_label_block` incorrectly treats the
        # duplicate as the formula text, producing a label-only extraction.
        text = _deduplicate_lines(block.text or "")
        label_line, formula_text = extract_mixed_label_block(text)
        if label_line is None:
            continue
        # Reject degenerate case: formula_text is also a standalone equation label.
        # This happens when deduplication left a multi-line block where all remaining
        # lines are also equation labels (e.g. a block with two different labels).
        if is_isolated_equation_label(formula_text):
            continue
        # Reject table rows and list-item sequences that happen to start with a
        # parenthesised number or Roman numeral that matches the equation-label
        # pattern (e.g. "(1) ft  0.3048 m", "(i) (j) (k)" list markers, or a
        # column-header block like "Symbol\nk\ncv\nmv").  A real formula definition
        # must contain at least one math operator or signal character.
        if not looks_like_standalone_formula(formula_text):
            continue
        eq_number = extract_label_number(label_line)
        if eq_number in seen_eq_numbers:
            continue
        candidates.append((entry.region_id, formula_text, label_line))
        extracted_ids.add(entry.region_id)
        seen_eq_numbers.add(eq_number)
    return candidates


def _extract_page(
    page_ro: PageReadingOrder,
    *,
    region_by_id: dict[str, LayoutRegion],
    block_by_region: dict[str, TextBlock],
    geom_by_page: dict[int, tuple[float, float]],
    numbers: dict[str, str],
    caption_refs: dict[str, str],
    continuation_refs: dict[str, str],
    providers: dict[str, EquationProvider],
    preprocessing: PreprocessingContext | None,
    inline_enabled: bool,
    pdf_path: Path | None = None,
    seen_eq_numbers: set[str] | None = None,
    confidence_estimator: ConfidenceEstimator | None = None,
) -> PageEquationExtraction:
    page_no = page_ro.page_no
    raster_cache: dict[int, Any] = {}
    fp_raster_cache: dict[int, Any] = {}  # first-pass renders at FIRST_PASS_ZOOM, one per page
    equations: list[Equation] = []
    failure_reason: str | None = None

    try:
        entries_sorted = sorted(page_ro.entries, key=lambda e: e.page_position)
        for entry in entries_sorted:
            region = region_by_id.get(entry.region_id)
            if region is None:
                continue
            parent_block = (
                block_by_region.get(entry.structural_parent_id)
                if entry.structural_parent_id
                else None
            )
            section_context = parent_block.text if parent_block is not None else ""

            if is_equation_region(region.region_type):
                block = block_by_region.get(region.region_id)
                bbox = _bbox_list(region)
                raw_text = region_text_for(region, block)
                # Skip bare single-token fragments (e.g. "3COt", "2.5H2O") that the
                # layout tagger placed in their own equation regions.  A valid display
                # equation must contain at least one relational/reaction operator or
                # span multiple tokens; single-term fragments without such an operator
                # are product/reactant labels broken out of a parent equation block.
                if raw_text and not _is_meaningful_equation_text(raw_text):
                    logger.debug(
                        "equation_region_skipped_fragment",
                        page_no=page_no,
                        region_id=region.region_id,
                        text=raw_text[:60],
                    )
                    continue
                equations.append(
                    _process_candidate(
                        equation_id=f"eq_{page_no}_{region.region_id}",
                        region_id=region.region_id,
                        region_type=region.region_type,
                        page_no=page_no,
                        bbox=bbox,
                        is_inline=False,
                        text_block_id=None,
                        region_text=raw_text,
                        section_context=section_context,
                        reading_position=entry.reading_position,
                        page_position=entry.page_position,
                        structural_parent_id=entry.structural_parent_id,
                        caption_ref=caption_refs.get(region.region_id),
                        equation_number=numbers.get(region.region_id),
                        continuation_ref=continuation_refs.get(region.region_id),
                        region_image=_load_region_image(
                            raster_cache,
                            preprocessing,
                            page_no,
                            geom_by_page.get(page_no),
                            bbox,
                            pdf_path=pdf_path,
                            fp_raster_cache=fp_raster_cache,
                        ),
                        recrop=lambda pad, z, _b=bbox, _p=page_no: _load_region_image(
                            raster_cache,
                            preprocessing,
                            _p,
                            geom_by_page.get(_p),
                            _b,
                            pdf_path=pdf_path,
                            pad_frac=pad,
                            zoom=z,
                        ),
                        providers=providers,
                        extra_notes=[],
                        confidence_estimator=confidence_estimator,
                    )
                )
            elif inline_enabled:
                block = block_by_region.get(entry.region_id)
                if block is None or block.role in _INLINE_SKIP_ROLES:
                    continue
                for index, match in enumerate(detect_inline_spans(block, enabled=True)):
                    # Reject fragments that lack relational operators or math chars —
                    # product codes, engine IDs, dates, and unit strings all score ≈ 0.28.
                    if score_formula_candidate(match.fragment).score < _INLINE_SCORE_GATE:
                        continue
                    inline_eq = _process_candidate(
                        equation_id=f"eq_{page_no}_{entry.region_id}_inline_{index}",
                        region_id=entry.region_id,
                        region_type="inline",
                        page_no=page_no,
                        bbox=list(match.bbox),
                        is_inline=True,
                        text_block_id=block.block_id,
                        region_text=match.fragment,
                        section_context=section_context,
                        reading_position=entry.reading_position,
                        page_position=entry.page_position,
                        structural_parent_id=entry.structural_parent_id,
                        caption_ref=None,
                        equation_number=None,
                        continuation_ref=None,
                        region_image=None,
                        providers=providers,
                        extra_notes=list(match.notes),
                        confidence_estimator=confidence_estimator,
                    )
                    # Skip inline spans where the VLM explicitly rejected the content
                    # (structural_latex / prose refusal) and produced no usable LaTeX.
                    # These are false-positive detections — the VLM correctly identified
                    # the fragment as non-mathematical (e.g. section headers, prose phrases).
                    if (
                        not inline_eq.latex
                        and any(
                            n.startswith("vl_response_rejected")
                            for n in inline_eq.notes
                        )
                    ):
                        logger.debug(
                            "inline_equation_vl_rejected_skipped",
                            page_no=page_no,
                            fragment=match.fragment[:60],
                        )
                        continue
                    equations.append(inline_eq)

        # Second pass: label-scan for display equations Docling did not tag as "equation" regions.
        # Finds text blocks preceded by an isolated equation-number label (e.g. "(12.2.1)").
        # geom_by_page and providers are threaded through so the confidence scorer can use
        # layout geometry and, for ambiguous blocks, optionally call the LLM oracle.
        extracted_ids: set[str] = {eq.region_id for eq in equations}
        label_candidates = _scan_label_candidates(
            entries_sorted,
            region_by_id,
            block_by_region,
            extracted_ids,
            geom_by_page=geom_by_page,
            providers=providers,
        )
        for idx, candidate in enumerate(label_candidates):
            region = region_by_id.get(candidate.region_id)
            bbox = list(candidate.bbox)
            equations.append(
                _process_candidate(
                    equation_id=f"eq_{page_no}_label_{idx}",
                    region_id=candidate.region_id,
                    region_type=candidate.region_type if candidate.region_type else (region.region_type if region is not None else "text"),
                    page_no=page_no,
                    bbox=bbox,
                    is_inline=False,
                    text_block_id=None,
                    region_text=candidate.formula_text,
                    section_context="",
                    reading_position=candidate.reading_position,
                    page_position=candidate.page_position,
                    structural_parent_id=candidate.structural_parent_id,
                    caption_ref=caption_refs.get(candidate.region_id),
                    equation_number=extract_label_number(candidate.label_text),
                    continuation_ref=continuation_refs.get(candidate.region_id),
                    region_image=_load_region_image(
                        raster_cache,
                        preprocessing,
                        page_no,
                        geom_by_page.get(page_no),
                        bbox,
                        pdf_path=pdf_path,
                        fp_raster_cache=fp_raster_cache,
                    ),
                    recrop=lambda pad, z, _b=bbox, _p=page_no: _load_region_image(
                        raster_cache,
                        preprocessing,
                        _p,
                        geom_by_page.get(_p),
                        _b,
                        pdf_path=pdf_path,
                        pad_frac=pad,
                        zoom=z,
                    ),
                    providers=providers,
                    extra_notes=[
                        "eq_label_detected",
                        f"eq_label_assembled:{len(candidate.region_ids)}",
                    ],
                    confidence_estimator=confidence_estimator,
                )
            )
        if label_candidates:
            logger.debug(
                "equation_label_scan_found",
                page_no=page_no,
                count=len(label_candidates),
            )

        # Third pass: mixed-block scan — label and formula share one text block.
        # Handles label-before, label-after, and label-in-middle patterns that the
        # isolated-label backward scan misses (e.g. "Eq. 12.4.4\nqd,' f\n.'. t - l.I5(...)").
        _seen = seen_eq_numbers if seen_eq_numbers is not None else set()
        # Pre-populate with numbers already extracted on this page so the mixed scan
        # respects both cross-page and within-page deduplication.
        for eq in equations:
            if eq.equation_number:
                _seen.add(eq.equation_number)
        mixed_candidates = _scan_mixed_label_candidates(
            entries_sorted, region_by_id, block_by_region, extracted_ids, _seen
        )
        for idx, (region_id, formula_text, label_text) in enumerate(mixed_candidates):
            region = region_by_id.get(region_id)
            bbox = _bbox_list(region) if region is not None else [0.0, 0.0, 0.0, 0.0]
            entry = next((e for e in entries_sorted if e.region_id == region_id), None)
            equations.append(
                _process_candidate(
                    equation_id=f"eq_{page_no}_mixed_{idx}",
                    region_id=region_id,
                    region_type=region.region_type if region is not None else "text",
                    page_no=page_no,
                    bbox=bbox,
                    is_inline=False,
                    text_block_id=None,
                    region_text=formula_text,
                    section_context="",
                    reading_position=entry.reading_position if entry is not None else idx,
                    page_position=entry.page_position if entry is not None else idx,
                    structural_parent_id=entry.structural_parent_id if entry is not None else None,
                    caption_ref=caption_refs.get(region_id),
                    equation_number=extract_label_number(label_text),
                    continuation_ref=continuation_refs.get(region_id),
                    region_image=_load_region_image(
                        raster_cache,
                        preprocessing,
                        page_no,
                        geom_by_page.get(page_no),
                        bbox,
                        pdf_path=pdf_path,
                        fp_raster_cache=fp_raster_cache,
                    ),
                    recrop=lambda pad, z, _b=bbox, _p=page_no: _load_region_image(
                        raster_cache,
                        preprocessing,
                        _p,
                        geom_by_page.get(_p),
                        _b,
                        pdf_path=pdf_path,
                        pad_frac=pad,
                        zoom=z,
                    ),
                    providers=providers,
                    extra_notes=["eq_mixed_block_detected"],
                    confidence_estimator=confidence_estimator,
                )
            )
        if mixed_candidates:
            logger.debug(
                "equation_mixed_block_scan_found",
                page_no=page_no,
                count=len(mixed_candidates),
            )

        # Fourth pass: unlabeled standalone chemical equations.
        # Recovers chemical reactions displayed without equation numbers (common in chemistry,
        # explosives, and materials science textbooks).  Gated on a confirmed reaction arrow
        # to avoid hallucinating equations from non-formula content.
        chem_candidates = _scan_unlabeled_chemical_candidates(
            entries_sorted, region_by_id, block_by_region, extracted_ids, geom_by_page
        )
        for idx, (region_id, formula_text) in enumerate(chem_candidates):
            region = region_by_id.get(region_id)
            bbox = _bbox_list(region) if region is not None else [0.0, 0.0, 0.0, 0.0]
            entry = next((e for e in entries_sorted if e.region_id == region_id), None)
            equations.append(
                _process_candidate(
                    equation_id=f"eq_{page_no}_chem_{idx}",
                    region_id=region_id,
                    region_type=region.region_type if region is not None else "text",
                    page_no=page_no,
                    bbox=bbox,
                    is_inline=False,
                    text_block_id=None,
                    region_text=formula_text,
                    section_context="",
                    reading_position=entry.reading_position if entry is not None else idx,
                    page_position=entry.page_position if entry is not None else idx,
                    structural_parent_id=entry.structural_parent_id if entry is not None else None,
                    caption_ref=caption_refs.get(region_id),
                    equation_number=None,
                    continuation_ref=continuation_refs.get(region_id),
                    region_image=_load_region_image(
                        raster_cache,
                        preprocessing,
                        page_no,
                        geom_by_page.get(page_no),
                        bbox,
                        pdf_path=pdf_path,
                        fp_raster_cache=fp_raster_cache,
                    ),
                    recrop=lambda pad, z, _b=bbox, _p=page_no: _load_region_image(
                        raster_cache,
                        preprocessing,
                        _p,
                        geom_by_page.get(_p),
                        _b,
                        pdf_path=pdf_path,
                        pad_frac=pad,
                        zoom=z,
                    ),
                    providers=providers,
                    extra_notes=["eq_unlabeled_chemical_detected"],
                    confidence_estimator=confidence_estimator,
                )
            )
        if chem_candidates:
            logger.debug(
                "equation_chemical_scan_found",
                page_no=page_no,
                count=len(chem_candidates),
            )

    except Exception as exc:  # contain per-page failure (FR-027)
        failure_reason = f"page_error:{type(exc).__name__}"
        logger.warning("equation_extraction_page_failed", page_no=page_no, error=str(exc))

    category_counts = Counter(eq.category for eq in equations)
    provider_counts = Counter(eq.selected_provider for eq in equations)
    if failure_reason is not None:
        outcome = "degraded"
    elif not equations:
        outcome = "empty"
    elif any(
        "unsupported_category" in eq.validation_flags
        or any(n.startswith(("recognition_failed", "provider_absent")) for n in eq.notes)
        for eq in equations
    ):
        outcome = "partial"
    else:
        outcome = "extracted"

    return PageEquationExtraction(
        page_no=page_no,
        equations=equations,
        outcome=outcome,
        category_counts=dict(category_counts),
        provider_counts=dict(provider_counts),
        confidence=confidence_mod.page_confidence(equations),
        failure_reason=failure_reason,
    )


def _build_statistics(
    pages: list[PageEquationExtraction],
    document_equations: list[Equation],
    validation_counts: dict[str, int],
    low_conf_class: int,
    low_conf_recog: int,
) -> EquationExtractionStatistics:
    category_distribution: Counter[str] = Counter()
    by_provider: Counter[str] = Counter()
    latex_valid = 0
    mathml_valid = 0
    for eq in document_equations:
        category_distribution[eq.category] += 1
        by_provider[eq.selected_provider] += 1
        if eq.latex and "invalid_latex" not in eq.validation_flags:
            latex_valid += 1
        if eq.mathml and "invalid_mathml" not in eq.validation_flags:
            mathml_valid += 1
    return EquationExtractionStatistics(
        total_pages=len(pages),
        total_equations=len(document_equations),
        category_distribution=dict(category_distribution),
        equations_by_provider=dict(by_provider),
        low_confidence_classification_count=low_conf_class,
        low_confidence_recognition_count=low_conf_recog,
        latex_valid_count=latex_valid,
        mathml_valid_count=mathml_valid,
        validation_counts=dict(validation_counts),
        failures=sum(1 for page in pages if page.outcome == "degraded"),
    )


def extract_equations(
    pdf_path: Path,
    *,
    text_extraction: TextExtractionContext | None,
    reading_order: ReadingOrderContext | None,
    layout: LayoutContext | None,
    preprocessing: PreprocessingContext | None,
    classification: ClassificationContext | None,
    page_manifest: list,
    config_hash: str = "",
) -> EquationExtractionContext:
    """Produce an :class:`EquationExtractionContext` for ``pdf_path`` from its upstream contexts."""
    providers = resolve_providers()
    provider_ids = provider_identities(providers)

    if layout is None or reading_order is None or reading_order.outcome == "failed":
        if layout is None:
            reason = "missing_layout"
        elif reading_order is None:
            reason = "missing_reading_order"
        else:
            reason = reading_order.failure_reason or "reading_order_failed"
        logger.info("equation_extraction_failed", pdf=str(pdf_path), failure_reason=reason)
        close_providers(providers)
        return EquationExtractionContext(
            outcome="failed",
            providers=provider_ids,
            config_hash=config_hash,
            statistics=EquationExtractionStatistics(failures=1),
            failure_reason=reason,
        )

    confidence_estimator = ConfidenceEstimator()

    notes: list[str] = []
    inline_enabled = config.KNOVEL_EQUATION_INLINE_ENABLED
    if text_extraction is None:
        inline_enabled = False
        notes.append("text_extraction_absent")
    if reading_order.outcome == "degraded":
        notes.append("reading_order_degraded")

    region_by_id: dict[str, LayoutRegion] = {}
    geom_by_page: dict[int, tuple[float, float]] = {}
    for page_layout in layout.pages:
        geom_by_page[page_layout.page_no] = (page_layout.width, page_layout.height)
        for region in page_layout.regions:
            region_by_id[region.region_id] = region

    block_by_region: dict[str, TextBlock] = {}
    if text_extraction is not None:
        for block in text_extraction.blocks or []:
            if block.region_id in block_by_region:
                logger.warning("duplicate_block_region_id", region_id=block.region_id)
            block_by_region[block.region_id] = block

    numbers = build_equation_numbers(reading_order, region_by_id)
    caption_refs = build_caption_refs(reading_order)
    continuation_refs = build_continuation_refs(reading_order)

    # seen_eq_numbers is threaded across pages so the mixed-block third pass does not
    # re-extract the same equation number from a later cross-reference block.
    seen_eq_numbers: set[str] = set()
    pages: list = []
    for page_ro in reading_order.pages or []:
        page_result = _extract_page(
            page_ro,
            region_by_id=region_by_id,
            block_by_region=block_by_region,
            geom_by_page=geom_by_page,
            numbers=numbers,
            caption_refs=caption_refs,
            continuation_refs=continuation_refs,
            providers=providers,
            preprocessing=preprocessing,
            inline_enabled=inline_enabled,
            pdf_path=pdf_path,
            seen_eq_numbers=seen_eq_numbers,
            confidence_estimator=confidence_estimator,
        )
        pages.append(page_result)
        # Propagate discovered equation numbers so subsequent pages skip them.
        for eq in page_result.equations:
            if eq.equation_number:
                seen_eq_numbers.add(eq.equation_number)

    document_equations = [eq for page in pages for eq in page.equations]
    low_conf_class, low_conf_recog = confidence_mod.flag_low_confidence(
        document_equations,
        classification_threshold=config.KNOVEL_EQUATION_CLASSIFICATION_MIN_CONFIDENCE,
        recognition_threshold=config.KNOVEL_EQUATION_RECOGNITION_MIN_CONFIDENCE,
    )
    validation_counts = validation.validate(
        document_equations,
        reading_order=reading_order,
        valid_region_ids=set(region_by_id),
    )
    statistics = _build_statistics(
        pages, document_equations, validation_counts, low_conf_class, low_conf_recog
    )

    outcome = "degraded" if statistics.failures > 0 else "extracted"
    context = EquationExtractionContext(
        outcome=outcome,
        equations=document_equations,
        pages=pages,
        statistics=statistics,
        providers=provider_ids,
        config_hash=config_hash,
        notes=notes,
    )

    if config.KNOVEL_EQUATION_DEBUG_DUMP:
        try:
            from equation_extraction.debug import resolve_workdir, write_equation_dump

            write_equation_dump(context, pdf_path, resolve_workdir(pdf_path))
        except Exception as exc:  # debug dump is best-effort, never fails the stage
            logger.warning(
                "equation_extraction_debug_dump_failed", pdf=str(pdf_path), error=str(exc)
            )

    logger.info(
        "document_equations_extracted",
        pdf=str(pdf_path),
        outcome=outcome,
        providers=provider_ids,
        total_pages=statistics.total_pages,
        total_equations=statistics.total_equations,
        category_distribution=dict(statistics.category_distribution),
        equations_by_provider=dict(statistics.equations_by_provider),
        low_confidence_classification=statistics.low_confidence_classification_count,
        low_confidence_recognition=statistics.low_confidence_recognition_count,
        failures=statistics.failures,
    )
    close_providers(providers)
    return context


def save_equation_crops(
    context: "EquationExtractionContext",
    pdf_path: Path,
    preprocessing: Any,
    layout: Any,
    crops_dir: Path,
) -> None:
    """Save the first-pass VLM crop image for every non-inline equation to *crops_dir*.

    The crop geometry mirrors exactly what was sent to the VLM during extraction:
    rendered at ``KNOVEL_EQUATION_FIRST_PASS_ZOOM`` with ``_first_pass_padding_fractions()``.
    This gives a faithful before/after comparison when crop-quality fixes are applied.
    """
    crops_dir.mkdir(parents=True, exist_ok=True)

    geom_by_page: dict[int, tuple[float, float]] = {}
    if layout is not None:
        for page_layout in getattr(layout, "pages", []):
            geom_by_page[page_layout.page_no] = (page_layout.width, page_layout.height)

    raster_cache: dict[int, Any] = {}
    fp_raster_cache: dict[int, Any] = {}

    for eq in getattr(context, "equations", []):
        if getattr(eq, "is_inline", False):
            continue
        raw_bbox = getattr(eq, "bbox", None)
        if isinstance(raw_bbox, dict):
            bbox = [
                float(raw_bbox.get("x0", 0)),
                float(raw_bbox.get("y0", 0)),
                float(raw_bbox.get("x1", 0)),
                float(raw_bbox.get("y1", 0)),
            ]
        elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            bbox = [float(v) for v in raw_bbox]
        else:
            continue
        if not _valid_bbox(bbox):
            continue
        page_no = getattr(eq, "page_no", None)
        geom = geom_by_page.get(page_no) if page_no is not None else None
        crop = _load_region_image(
            raster_cache,
            preprocessing,
            page_no,
            geom,
            bbox,
            pdf_path=pdf_path,
            fp_raster_cache=fp_raster_cache,
        )
        if crop is None:
            continue
        eq_id = str(getattr(eq, "equation_id", f"eq_p{page_no}")).replace("/", "_").replace("\\", "_")
        try:
            crop.save(crops_dir / f"{eq_id}.png")
        except Exception as exc:
            logger.debug("equation_crop_save_failed", eq_id=eq_id, error=str(exc))
