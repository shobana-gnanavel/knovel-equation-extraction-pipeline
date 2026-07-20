"""Text extraction, preprocessing, and normalization — merged module.

Applies image enhancement (denoise, sharpen, deskew) to rendered pages,
then extracts, normalizes, roles, scores, and validates text blocks from
reading-ordered layout regions.

Merged sources
--------------
* equation-extraction-pipeline/preprocessing.py                   — image enhancement
* equation-extraction-pipeline/text_extraction/roles.py           — semantic role mapping
* equation-extraction-pipeline/text_extraction/normalize.py       — Unicode/whitespace normalization
* equation-extraction-pipeline/text_extraction/selection.py       — method selection and fallback gate
* equation-extraction-pipeline/text_extraction/confidence.py      — confidence scoring and flagging
* equation-extraction-pipeline/text_extraction/engines.py         — native-PDF and OCR engine protocol
* equation-extraction-pipeline/text_extraction/extractor.py       — document-level orchestration
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np
import structlog
from PIL import Image, ImageFilter

from equation_extraction_pipeline.config import settings as config
from equation_extraction_pipeline.domain.models import (
    ClassificationContext,
    ClassificationResult,
    LayoutContext,
    LayoutRegion,
    PageReadingOrder,
    PageTextExtraction,
    PreprocessingContext,
    Provenance,
    ReadingOrderContext,
    RenderedPage,
    TextBlock,
    TextExtractionContext,
    TextExtractionStatistics,
)

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)

__all__ = [
    # preprocessing.py
    "preprocess_pages",
    # roles.py
    "text_role",
    "is_text_bearing",
    "NON_TEXT_TYPES",
    "CODE_ROLE",
    # normalize.py
    "normalize_block",
    "printable_ratio",
    "has_invalid_unicode",
    # selection.py
    "select_method",
    "should_fallback",
    # confidence.py
    "native_confidence",
    "clamp",
    "page_confidence",
    "flag_low_confidence",
    # engines.py
    "ExtractedText",
    "TextExtractionEngine",
    "NativeEngine",
    "OcrEngine",
    # extractor.py
    "extract_text",
    "resolve_languages",
]


# ============================================================
# SECTION 1: Merged from text_extraction/roles.py
# Semantic text-role mapping and the text-bearing-region filter
# ============================================================

# Layout region type -> semantic text role. Anything unlisted falls back to ``text_block``.
_ROLE_BY_TYPE: dict[str, str] = {
    "document_title": "document_title",
    "chapter": "chapter_title",
    "section": "heading",
    "subsection": "subheading",
    "heading": "heading",
    "references": "reference",
    "appendix": "appendix",
    "paragraph": "paragraph",
    "text_block": "text_block",
    "list": "bullet_list",
    "bullet_list": "bullet_list",
    "numbered_list": "numbered_list",
    "code_block": "code_block",
    "quote": "quote",
    "table_caption": "caption",
    "figure_caption": "caption",
    "footnote": "footnote",
    "endnote": "endnote",
    "sidebar": "text_block",
    "callout": "text_block",
    "equation_number": "text_block",
}

# Region types that carry no extractable running text for this stage (handled by other stages or
# excluded from the body flow): graphics, tables, equation bodies, and navigational furniture.
NON_TEXT_TYPES: frozenset[str] = frozenset(
    {"table", "equation", "figure", "image", "header", "footer", "page_number"}
)

# The role whose text is preserved verbatim (no merging / de-hyphenation / whitespace collapsing).
CODE_ROLE = "code_block"


def text_role(region_type: str) -> str:
    """Return the semantic text role for a layout ``region_type`` (default ``text_block``)."""
    return _ROLE_BY_TYPE.get(region_type, "text_block")


def is_text_bearing(region_type: str) -> bool:
    """Whether a region carries extractable body text."""
    return region_type not in NON_TEXT_TYPES


# ============================================================
# SECTION 2: Merged from text_extraction/normalize.py
# Role-aware text normalization
# ============================================================

_UNICODE_FORMS: dict[str, Literal["NFC", "NFKC", "NFD", "NFKD"]] = {
    "NFC": "NFC",
    "NFKC": "NFKC",
    "NFD": "NFD",
    "NFKD": "NFKD",
}

# A line-final hyphen (optionally followed by trailing spaces) used to split a word across lines.
_HYPHEN_BREAK = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")
# Runs of intra-line spaces/tabs collapse to a single space; newlines are handled separately.
_INTRALINE_WS = re.compile(r"[ \t\f\v]+")
# A blank line (two-or-more newlines) marks a true paragraph boundary.
_PARA_BREAK = re.compile(r"\n[ \t]*\n[ \t\n]*")
# A single (soft) newline inside a paragraph.
_SOFT_BREAK = re.compile(r"[ \t]*\n[ \t]*")


def printable_ratio(text: str) -> float:
    """Fraction of characters that are printable/space (a native-text quality signal)."""
    if not text:
        return 0.0
    good = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    return max(0.0, min(good / len(text), 1.0))


def has_invalid_unicode(text: str) -> bool:
    """True if ``text`` carries replacement chars or unassigned/surrogate code points."""
    for ch in text:
        if ch == "�":
            return True
        category = unicodedata.category(ch)
        if category in {"Cn", "Cs"}:  # unassigned or surrogate
            return True
    return False


def _dehyphenate(text: str, *, use_dict: bool) -> tuple[str, bool]:
    """Rejoin end-of-line hyphenated splits where the continuation begins lowercase."""
    applied = False

    def _join(match: re.Match[str]) -> str:
        nonlocal applied
        left, right = match.group(1), match.group(2)
        if not right.islower():
            return f"{left}-\n{right}"
        applied = True
        return f"{left}{right}"

    result = _HYPHEN_BREAK.sub(_join, text)
    return result, applied


def normalize_block(raw_text: str, *, role: str, config: Any) -> tuple[str, list[str]]:
    """Normalize ``raw_text`` for ``role``; return ``(text, applied_actions)``.

    Code-block roles are returned verbatim with only ``code_verbatim`` recorded.
    """
    if role == CODE_ROLE:
        return raw_text.rstrip("\n"), ["code_verbatim"]

    actions: list[str] = []
    text = raw_text

    form = (getattr(config, "KNOVEL_TEXT_UNICODE_FORM", "NFC") or "NFC").upper()
    literal_form = _UNICODE_FORMS.get(form)
    if literal_form is not None:
        normalized = unicodedata.normalize(literal_form, text)
        if normalized != text:
            text = normalized
        actions.append(f"unicode_{form.lower()}")

    if has_invalid_unicode(text):
        actions.append("invalid_unicode")

    if getattr(config, "KNOVEL_TEXT_DEHYPHENATE", True):
        text, applied = _dehyphenate(
            text, use_dict=getattr(config, "KNOVEL_TEXT_DEHYPHENATE_DICT", False)
        )
        actions.append("dehyphenated" if applied else "hyphen_preserved")

    if getattr(config, "KNOVEL_TEXT_MERGE_SOFT_BREAKS", True):
        paragraphs = _PARA_BREAK.split(text)
        merged = [_SOFT_BREAK.sub(" ", para).strip() for para in paragraphs]
        text = "\n\n".join(part for part in merged if part)
        actions.append("soft_breaks_merged")
        actions.append("paragraph_reconstructed")

    if getattr(config, "KNOVEL_TEXT_NORMALIZE_WHITESPACE", True):
        text = _INTRALINE_WS.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = text.strip()
        actions.append("whitespace")

    return text, actions


# ============================================================
# SECTION 3: Merged from text_extraction/selection.py
# Per-region extraction-method selection and native→OCR fallback gate
# ============================================================


def select_method(page_modality: str, *, config: Any) -> str:
    """Pick the initial extraction method for a region (``native`` | ``ocr``)."""
    strategy = (getattr(config, "KNOVEL_TEXT_STRATEGY", "auto") or "auto").lower()
    if strategy == "native":
        return "native"
    if strategy == "ocr":
        return "ocr"
    # auto / per_region: derive from the page modality; hybrid defaults to native then may fall back.
    if page_modality == "scanned":
        return "ocr"
    return "native"


def should_fallback(text: str, *, config: Any) -> bool:
    """Whether empty/garbage native text should fall back to OCR."""
    if not getattr(config, "KNOVEL_TEXT_FALLBACK_ENABLED", True):
        return False
    if not text.strip():
        return True
    if has_invalid_unicode(text):
        return True
    threshold = float(getattr(config, "KNOVEL_TEXT_NATIVE_MIN_PRINTABLE", 0.60))
    return printable_ratio(text) < threshold


# ============================================================
# SECTION 4: Merged from text_extraction/confidence.py
# Confidence normalization and low-confidence handling
# ============================================================


def clamp(value: float) -> float:
    """Clamp a confidence into ``[0,1]`` and round for stable serialization."""
    return round(max(0.0, min(1.0, float(value))), 4)


def native_confidence(text: str) -> float:
    """Native-text confidence heuristic: the printable/valid-character ratio."""
    return clamp(printable_ratio(text))


def page_confidence(blocks: list[TextBlock]) -> float:
    """Mean of block confidences (0.0 for a page with no blocks)."""
    if not blocks:
        return 0.0
    return round(sum(block.confidence for block in blocks) / len(blocks), 4)


def flag_low_confidence(
    blocks: list[TextBlock],
    *,
    threshold: float,
    policy: str = "flag",
) -> tuple[list[TextBlock], int]:
    """Flag blocks below ``threshold`` as low-confidence and retain them. Returns ``(kept, count)``.

    ``policy`` is accepted for symmetry with other stages, but dropping text is not permitted:
    blocks are always retained; only the ``low_confidence`` flag is set.
    """
    flagged = 0
    for block in blocks:
        if block.confidence < threshold:
            flagged += 1
            block.low_confidence = True
            if "low_confidence" not in block.validation_flags:
                block.validation_flags.append("low_confidence")
    return blocks, flagged


# ============================================================
# SECTION 5: Merged from text_extraction/engines.py
# Interchangeable text-extraction engines behind a common protocol
#
# - NativeEngine: reads native PDF spans (PageView.get_text("dict"))
# - OcrEngine: wraps pipeline.ocr_backend (PaddleOCR) over rendered crops
# ============================================================

_BOLD_FLAG = 16


@dataclass
class ExtractedText:
    """Raw engine output for one region before normalization."""

    text: str = ""
    confidence: float = 0.0
    inline_formats: list[str] = field(default_factory=list)


class TextExtractionEngine(Protocol):
    """Common interface for a text-extraction engine."""

    name: str
    method: str  # "native" | "ocr"

    def extract_region(
        self,
        region: Any,
        *,
        page_text: dict | None,
        page_image: np.ndarray | None,
        language: str,
        config: Any,
    ) -> ExtractedText: ...


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _inside(bbox: list[float], region: dict[str, float]) -> bool:
    cx, cy = _bbox_center(bbox)
    return region.get("x0", 0.0) <= cx <= region.get("x1", 0.0) and region.get(
        "y0", 0.0
    ) <= cy <= region.get("y1", 0.0)


class NativeEngine:
    """Native PDF text engine reading spans inside a region bbox."""

    name = "pdf_backend"
    method = "native"

    def extract_region(
        self,
        region: Any,
        *,
        page_text: dict | None,
        page_image: np.ndarray | None,
        language: str,
        config: Any,
    ) -> ExtractedText:
        if not page_text:
            return ExtractedText(text="", confidence=0.0)
        region_bbox = region.bbox if isinstance(region.bbox, dict) else {}
        lines_out: list[str] = []
        formats: set[str] = set()
        for block in page_text.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            bbox = [float(v) for v in block.get("bbox", [0.0, 0.0, 0.0, 0.0])]
            if region_bbox and not _inside(bbox, region_bbox):
                continue
            for line in block.get("lines", []):
                parts: list[str] = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if not span_text:
                        continue
                    parts.append(span_text)
                    if int(span.get("flags", 0)) & _BOLD_FLAG:
                        formats.add("bold")
                    font = str(span.get("font", "")).lower()
                    if "italic" in font or "oblique" in font:
                        formats.add("italic")
                line_text = "".join(parts)
                if line_text.strip():
                    lines_out.append(line_text)
        text = "\n".join(lines_out)
        return ExtractedText(
            text=text,
            confidence=native_confidence(text),
            inline_formats=sorted(formats),
        )


class OcrEngine:
    """OCR engine wrapping ``pipeline.ocr_backend`` over the region's raster crop."""

    name = "paddleocr"
    method = "ocr"

    def extract_region(
        self,
        region: Any,
        *,
        page_text: dict | None,
        page_image: np.ndarray | None,
        language: str,
        config: Any,
    ) -> ExtractedText:
        # Import from the merged ocr_extractor for the PaddleOCR wrapper.
        try:
            from equation_extraction_pipeline.extraction.ocr_extractor import (
                OCR_AVAILABLE,
                ocr_page,
            )
        except ImportError:
            # Fallback to original pipeline location during transition.
            from pipeline.ocr_backend import OCR_AVAILABLE, ocr_page  # type: ignore[import]

        try:
            from pipeline.pdf_backend import RENDER_ZOOM  # type: ignore[import]
        except ImportError:
            RENDER_ZOOM = 1.0  # type: ignore[assignment]

        if page_image is None or not OCR_AVAILABLE:
            return ExtractedText(text="", confidence=0.0)
        region_bbox = region.bbox if isinstance(region.bbox, dict) else {}
        height, width = page_image.shape[0], page_image.shape[1]
        x0 = max(0, int(region_bbox.get("x0", 0.0) * RENDER_ZOOM))
        y0 = max(0, int(region_bbox.get("y0", 0.0) * RENDER_ZOOM))
        x1 = min(width, int(region_bbox.get("x1", 0.0) * RENDER_ZOOM))
        y1 = min(height, int(region_bbox.get("y1", 0.0) * RENDER_ZOOM))
        if x1 <= x0 or y1 <= y0:
            crop = page_image
        else:
            crop = page_image[y0:y1, x0:x1]
        lines = ocr_page(crop, language or "en")
        texts = [str(line.get("text", "")) for line in lines if line.get("text")]
        confidences = [float(line.get("confidence", 0.0)) for line in lines]
        text = "\n".join(texts)
        mean_conf = clamp(sum(confidences) / len(confidences)) if confidences else 0.0
        return ExtractedText(text=text, confidence=mean_conf)


# ============================================================
# SECTION 6: Merged from preprocessing.py
# Image enhancement: denoise, sharpen, deskew
#
# Applies to rendered page images produced by page_renderer.render_pages().
# ============================================================


def _to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _denoise(arr: np.ndarray, modality: str) -> np.ndarray:
    if modality == "scanned":
        return cv2.fastNlMeansDenoisingColored(
            arr, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21
        )
    return cv2.GaussianBlur(arr, (3, 3), 0)


def _sharpen(image: Image.Image) -> Image.Image:
    if config.SHARPEN_AMOUNT <= 0:
        return image
    return image.filter(
        ImageFilter.UnsharpMask(
            radius=config.SHARPEN_RADIUS,
            percent=int(config.SHARPEN_AMOUNT * 100),
            threshold=3,
        )
    )


def _detect_skew_angle(arr: np.ndarray) -> float:
    """Return estimated skew angle in degrees using Hough line detection."""
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV builds differ in whether minAreaRect reports approximately -90 or
    # +90 degrees for a nearly horizontal rectangle.  Map both conventions to
    # the small correction range [-45, 45]; otherwise a level page reported as
    # 89.9 degrees is incorrectly rotated sideways.
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return float(angle)


def _deskew(arr: np.ndarray, angle: float) -> np.ndarray:
    h, w = arr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _compute_quality_score_preprocessing(image: Image.Image) -> float:
    gray = np.array(image.convert("L"))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return round(min(lap_var, 500.0) / 500.0, 4)


def _enhance_page(rp: RenderedPage, modality: str) -> tuple[Image.Image, float]:
    """Apply denoise / deskew / sharpen to a single page and return ``(image, quality)``."""
    arr = _to_cv(rp.image)
    arr = _denoise(arr, modality)

    angle = _detect_skew_angle(arr)
    if abs(angle) > config.DESKEW_THRESHOLD_DEG:
        logger.debug("deskew page=%d angle=%.2f°", rp.page_number, angle)
        arr = _deskew(arr, angle)

    enhanced_image = _to_pil(arr)
    enhanced_image = _sharpen(enhanced_image)
    return enhanced_image, _compute_quality_score_preprocessing(enhanced_image)


def render_and_preprocess_pages(
    pdf_path: Path,
    classification: ClassificationResult,
    pages_dir: Path,
) -> list[RenderedPage]:
    """Render, enhance, and persist each page one at a time (flat-memory ingestion).

    Streams pages through :func:`page_renderer.iter_rendered_pages`, enhances each
    (denoise / deskew / sharpen), writes the result to ``pages_dir/page_NNN.png``, and
    returns lightweight :class:`RenderedPage` objects whose pixels live on disk
    (``image=None``, ``raster_path`` set). Downstream stages load rasters on demand via
    :meth:`RenderedPage.load_image`, so peak memory stays bounded to a few pages
    regardless of page count — unlike ``render_pages`` + ``preprocess_pages``, which
    hold every raw and enhanced page in RAM simultaneously.
    """
    from equation_extraction_pipeline.extraction.page_renderer import iter_rendered_pages

    pages_dir = Path(pages_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)
    modality = classification.modality
    results: list[RenderedPage] = []

    for rp in iter_rendered_pages(pdf_path, classification):
        enhanced_image, new_quality = _enhance_page(rp, modality)
        # Release the raw render immediately; only the enhanced page proceeds.
        rp.image = None

        raster_path = pages_dir / f"page_{rp.page_number:03d}.png"
        enhanced_image.save(raster_path, format="PNG")

        logger.debug(
            "preprocess page=%d quality %.3f → %.3f",
            rp.page_number,
            rp.quality_score,
            new_quality,
        )
        results.append(
            RenderedPage(
                page_number=rp.page_number,
                image=None,
                dpi=rp.dpi,
                quality_score=new_quality,
                width_px=rp.width_px,
                height_px=rp.height_px,
                raster_path=str(raster_path),
            )
        )
        # Drop the enhanced raster from RAM; it is reloaded lazily from disk.
        enhanced_image.close()

    logger.info("render_preprocess_done pages=%d dir=%s", len(results), pages_dir)
    return results


def preprocess_pages(
    pages: list[RenderedPage],
    classification: ClassificationResult,
) -> list[RenderedPage]:
    """Apply denoise, sharpen, and optional deskew to each rendered page.

    Parameters
    ----------
    pages:
        Output from ``page_renderer.render_pages()``.
    classification:
        Used to select the appropriate denoise strategy.

    Returns
    -------
    list[RenderedPage]
        Same list with enhanced images and recomputed quality scores.
        The original page objects are replaced — pixel data is not shared.
    """
    modality = classification.modality
    results: list[RenderedPage] = []

    for rp in pages:
        arr = _to_cv(rp.image)

        # Denoise
        arr = _denoise(arr, modality)

        # Deskew (only when tilt is significant)
        angle = _detect_skew_angle(arr)
        if abs(angle) > config.DESKEW_THRESHOLD_DEG:
            logger.debug("deskew page=%d angle=%.2f°", rp.page_number, angle)
            arr = _deskew(arr, angle)

        enhanced_image = _to_pil(arr)

        # Sharpen
        enhanced_image = _sharpen(enhanced_image)

        new_quality = _compute_quality_score_preprocessing(enhanced_image)
        logger.debug(
            "preprocess page=%d quality %.3f → %.3f",
            rp.page_number,
            rp.quality_score,
            new_quality,
        )

        results.append(
            RenderedPage(
                page_number=rp.page_number,
                image=enhanced_image,
                dpi=rp.dpi,
                quality_score=new_quality,
                width_px=rp.width_px,
                height_px=rp.height_px,
            )
        )

    logger.info("preprocessing_done pages=%d", len(results))
    return results


# ============================================================
# SECTION 7: Merged from text_extraction/extractor.py
# Text extraction stage entrypoint — document-level orchestration
#
# Stubs for text_extraction.registry (not in source files):
#   resolve_native_engine, resolve_ocr_engine,
#   selected_native_name, selected_ocr_name
#
# Stub for text_extraction.validation (not in source files):
#   validate
# ============================================================

_HEADING_TYPES = frozenset({"document_title", "chapter", "section", "subsection", "heading"})


# ---------------------------------------------------------------------------
# Registry stubs (text_extraction.registry not provided as source)
# ---------------------------------------------------------------------------

def resolve_native_engine() -> TextExtractionEngine:
    """Return the configured native text-extraction engine (defaults to NativeEngine)."""
    return NativeEngine()


def resolve_ocr_engine() -> TextExtractionEngine:
    """Return the configured OCR text-extraction engine (defaults to OcrEngine)."""
    return OcrEngine()


def selected_native_name() -> str:
    """Return the name of the active native engine."""
    return NativeEngine.name


def selected_ocr_name() -> str:
    """Return the name of the active OCR engine."""
    return OcrEngine.name


# ---------------------------------------------------------------------------
# Validation stub (text_extraction.validation not provided as source)
# ---------------------------------------------------------------------------

class _ValidationStub:
    """Minimal stub for text_extraction.validation until that module is merged."""

    @staticmethod
    def validate(
        blocks: list[TextBlock],
        *,
        reading_order: Any = None,
    ) -> dict[str, int]:
        """Return an empty validation-count dict (stub implementation)."""
        return {}


validation = _ValidationStub()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_languages(classification: ClassificationContext | None) -> list[str]:
    """Resolve OCR language(s) from config, else the document's dominant language."""
    configured = (config.KNOVEL_TEXT_OCR_LANGUAGES or "auto").strip()
    if configured and configured.lower() != "auto":
        return [part.strip() for part in configured.split(",") if part.strip()]
    language = (classification.dominant_language if classification else "") or ""
    code = language.lower()[:2]
    if not code or code == "un":  # "und" / empty → default English
        code = "en"
    return [code]


def _bbox_list(region: LayoutRegion) -> list[float]:
    bbox = region.bbox if isinstance(region.bbox, dict) else {}
    return [
        float(bbox.get("x0", 0.0)),
        float(bbox.get("y0", 0.0)),
        float(bbox.get("x1", 0.0)),
        float(bbox.get("y1", 0.0)),
    ]


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if limit and len(text) > limit:
        return text[:limit], True
    return text, False


def _extract_region_block(
    entry: Any,
    region: LayoutRegion,
    *,
    page_no: int,
    modality: str,
    language: str,
    page_text: dict | None,
    image_holder: dict[str, Any],
    page_view: Any,
    native_engine: TextExtractionEngine,
    ocr_engine: TextExtractionEngine,
    native_name: str,
    ocr_name: str,
) -> TextBlock:
    """Extract, normalize, role, score, and validate one region into a ``TextBlock``."""
    role = text_role(region.region_type)
    method = select_method(modality, config=config)
    notes: list[str] = []

    try:
        from pipeline.pdf_backend import RENDER_ZOOM  # type: ignore[import]
    except ImportError:
        RENDER_ZOOM = 1.0  # type: ignore[assignment]

    def _render_image() -> Any:
        if "image" not in image_holder:
            try:
                image_holder["image"] = page_view.render(RENDER_ZOOM) if page_view else None
            except Exception:
                image_holder["image"] = None
        return image_holder["image"]

    def _run(engine: TextExtractionEngine) -> ExtractedText:
        page_image = _render_image() if engine.method == "ocr" else None
        return engine.extract_region(
            region,
            page_text=page_text,
            page_image=page_image,
            language=language,
            config=config,
        )

    if method == "ocr":
        result = _run(ocr_engine)
        engine_name = ocr_name
        used_language: str | None = language
    else:
        result = _run(native_engine)
        engine_name = native_name
        used_language = None
        if should_fallback(result.text, config=config):
            ocr_result = _run(ocr_engine)
            if ocr_result.text.strip():
                result = ocr_result
                method = "fallback"
                engine_name = ocr_name
                used_language = language
                notes.append("fallback:native_empty")

    text, actions = normalize_block(result.text, role=role, config=config)
    text, truncated = _truncate(text, config.KNOVEL_TEXT_MAX_BLOCK_CHARS)
    if truncated:
        notes.append("truncated:size_cap")

    block = TextBlock(
        block_id=f"text_{page_no}_{region.region_id}",
        region_id=region.region_id,
        page_no=page_no,
        reading_position=entry.reading_position,
        page_position=entry.page_position,
        structural_parent_id=entry.structural_parent_id,
        column_index=entry.column_index,
        bbox=_bbox_list(region),
        text=text,
        role=role,
        method=method,
        language=used_language,
        confidence=clamp(result.confidence),
        char_count=len(text),
        normalization=[a for a in actions if a != "invalid_unicode"],
        inline_formats=list(result.inline_formats),
        provenance=Provenance(source_extractors=[engine_name], source_pages=[page_no]),
        notes=notes,
    )

    # Per-block validation findings that need region context.
    if "invalid_unicode" in actions:
        block.validation_flags.append("invalid_unicode")
    if not text.strip():
        block.validation_flags.append("empty")
        block.validation_flags.append("missing_text")
        if region.region_type in _HEADING_TYPES:
            block.validation_flags.append("missing_heading")
    elif method != "ocr" and printable_ratio(text) < config.KNOVEL_TEXT_NATIVE_MIN_PRINTABLE:
        block.validation_flags.append("corrupted")
    if entry.page_no and entry.page_no != page_no:
        block.validation_flags.append("page_ref_mismatch")
    return block


def _extract_page(
    page_ro: PageReadingOrder,
    page_view: Any,
    region_by_id: dict[str, LayoutRegion],
    *,
    modality: str,
    language: str,
    native_engine: TextExtractionEngine,
    ocr_engine: TextExtractionEngine,
    native_name: str,
    ocr_name: str,
) -> PageTextExtraction:
    page_no = page_ro.page_no
    failure_reason: str | None = None
    page_text: dict | None = None
    if page_view is not None:
        try:
            page_text = page_view.get_text("dict")
        except Exception as exc:
            failure_reason = f"page_read_error:{type(exc).__name__}"
            _slog.warning("text_extraction_page_failed", page_no=page_no, error=str(exc))

    image_holder: dict[str, Any] = {}
    blocks: list[TextBlock] = []
    entries = sorted(page_ro.entries, key=lambda e: e.page_position)
    for entry in entries:
        region = region_by_id.get(entry.region_id)
        if region is None or not is_text_bearing(region.region_type):
            continue
        block = _extract_region_block(
            entry,
            region,
            page_no=page_no,
            modality=modality,
            language=language,
            page_text=page_text,
            image_holder=image_holder,
            page_view=page_view,
            native_engine=native_engine,
            ocr_engine=ocr_engine,
            native_name=native_name,
            ocr_name=ocr_name,
        )
        blocks.append(block)

    flag_low_confidence(
        blocks,
        threshold=config.KNOVEL_TEXT_MIN_CONFIDENCE,
        policy=config.KNOVEL_TEXT_LOW_CONFIDENCE_POLICY,
    )

    method_counts = Counter(block.method for block in blocks)
    fallback_used = method_counts.get("fallback", 0) > 0
    if failure_reason is not None:
        outcome = "degraded"
    elif not blocks:
        outcome = "empty"
    elif any("missing_text" in block.validation_flags for block in blocks):
        outcome = "partial"
    else:
        outcome = "extracted"

    return PageTextExtraction(
        page_no=page_no,
        blocks=blocks,
        outcome=outcome,
        method_counts=dict(method_counts),
        confidence=page_confidence(blocks),
        fallback_used=fallback_used,
        failure_reason=failure_reason,
    )


def _build_statistics(
    pages: list[PageTextExtraction],
    document_blocks: list[TextBlock],
    validation_counts: dict[str, int],
) -> TextExtractionStatistics:
    by_method: Counter[str] = Counter()
    by_role: Counter[str] = Counter()
    norm_counts: Counter[str] = Counter()
    total_chars = 0
    low_conf = 0
    fallback = 0
    for block in document_blocks:
        by_method[block.method] += 1
        by_role[block.role] += 1
        total_chars += block.char_count
        if block.low_confidence:
            low_conf += 1
        if block.method == "fallback":
            fallback += 1
        for action in block.normalization:
            norm_counts[action] += 1
    return TextExtractionStatistics(
        total_pages=len(pages),
        total_blocks=len(document_blocks),
        total_characters=total_chars,
        blocks_by_method=dict(by_method),
        blocks_by_role=dict(by_role),
        low_confidence_count=low_conf,
        normalization_counts=dict(norm_counts),
        validation_counts=dict(validation_counts),
        fallback_count=fallback,
        failures=sum(1 for page in pages if page.outcome == "degraded"),
    )


def extract_text(
    pdf_path: Path,
    *,
    reading_order: ReadingOrderContext | None,
    layout: LayoutContext | None,
    preprocessing: PreprocessingContext | None,
    classification: ClassificationContext | None,
    page_manifest: list,
    config_hash: str = "",
) -> TextExtractionContext:
    """Produce a :class:`TextExtractionContext` for ``pdf_path`` from its Reading Order Context."""
    native_engine = resolve_native_engine()
    ocr_engine = resolve_ocr_engine()
    native_name = selected_native_name()
    ocr_name = selected_ocr_name()
    languages = resolve_languages(classification)
    primary_language = languages[0] if languages else "en"

    if reading_order is None or reading_order.outcome == "failed":
        reason = (
            "missing_reading_order"
            if reading_order is None
            else (reading_order.failure_reason or "reading_order_failed")
        )
        _slog.info("text_extraction_failed", pdf=str(pdf_path), failure_reason=reason)
        return TextExtractionContext(
            outcome="failed",
            native_engine=native_name,
            ocr_engine=ocr_name,
            languages=languages,
            config_hash=config_hash,
            statistics=TextExtractionStatistics(failures=1),
            failure_reason=reason,
        )

    region_by_id: dict[str, LayoutRegion] = {}
    if layout is not None:
        for page_layout in layout.pages:
            for region in page_layout.regions:
                region_by_id[region.region_id] = region

    modality_by_page = {
        getattr(meta, "page_no", index): getattr(meta, "page_type", "")
        for index, meta in enumerate(page_manifest or [])
    }
    default_modality = classification.modality if classification else "digital"

    # When the document has an embedded text layer (PDF has real fonts / a text layer), use
    # native-first extraction even for pages the per-page classifier marked as 'scanned'.
    if classification is not None and classification.characteristics.get("has_text_layer"):
        modality_by_page = {
            k: ("digital" if v == "scanned" else v) for k, v in modality_by_page.items()
        }
        if default_modality == "scanned":
            default_modality = "digital"

    document = None
    try:
        from pipeline.pdf_backend import open_document  # type: ignore[import]

        document = open_document(str(pdf_path))
        page_count = len(document)
    except Exception as exc:
        _slog.warning("text_extraction_open_failed", pdf=str(pdf_path), error=str(exc))
        page_count = 0

    pages: list[PageTextExtraction] = []
    try:
        for page_ro in reading_order.pages:
            page_no = page_ro.page_no
            page_view = (
                document[page_no] if document is not None and page_no < page_count else None
            )
            modality = modality_by_page.get(page_no, default_modality)
            pages.append(
                _extract_page(
                    page_ro,
                    page_view,
                    region_by_id,
                    modality=modality,
                    language=primary_language,
                    native_engine=native_engine,
                    ocr_engine=ocr_engine,
                    native_name=native_name,
                    ocr_name=ocr_name,
                )
            )
    finally:
        if document is not None:
            try:
                document.close()
            except Exception:
                pass

    document_blocks = [block for page in pages for block in page.blocks]
    validation_counts = validation.validate(document_blocks, reading_order=reading_order)
    statistics = _build_statistics(pages, document_blocks, validation_counts)

    outcome = "degraded" if statistics.failures > 0 else "extracted"
    context = TextExtractionContext(
        outcome=outcome,
        blocks=document_blocks,
        pages=pages,
        statistics=statistics,
        native_engine=native_name,
        ocr_engine=ocr_name,
        languages=languages,
        config_hash=config_hash,
    )

    if config.KNOVEL_TEXT_DEBUG_DUMP:
        try:
            from text_extraction.debug import (  # type: ignore[import]
                resolve_workdir,
                write_text_dump,
            )

            write_text_dump(context, pdf_path, resolve_workdir(pdf_path))
        except Exception as exc:
            _slog.warning(
                "text_extraction_debug_dump_failed", pdf=str(pdf_path), error=str(exc)
            )

    _slog.info(
        "document_text_extracted",
        pdf=str(pdf_path),
        outcome=outcome,
        native_engine=native_name,
        ocr_engine=ocr_name,
        languages=languages,
        total_pages=statistics.total_pages,
        total_blocks=statistics.total_blocks,
        total_characters=statistics.total_characters,
        blocks_by_method=dict(statistics.blocks_by_method),
        low_confidence=statistics.low_confidence_count,
        fallbacks=statistics.fallback_count,
    )
    return context
