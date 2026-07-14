"""Equation label detector — merged module.

Sections
--------
1. Classification signals         (from classifier/signals.py)
   Page-level signals: text density, font metadata, image coverage,
   render similarity.

2. Language detection             (from classifier/language.py)
   Optional langdetect backend with graceful degradation.

3. Document signals               (from classifier/doc_signals.py)
   Document-level signal collection over a sampled set of pages.

4. Classification rules           (from classifier/doc_rules.py)
   Pure deterministic rules: modality, layout complexity, category
   scoring/selection, strategy recommendation.

5. Page classifier                (from classifier/page_classifier.py)
   Per-page digital/scanned/hybrid classification.

6. Document classifier            (from classifier/doc_classifier.py)
   Orchestrates signals → rules → language into ClassificationContext.

7. Equation layout detection      (from layout_detection.py)
   Label-scan (primary) and Docling ML (fallback) equation detection,
   crop saving, and the public ``detect_equations`` entry point.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

import structlog
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTPage, LTTextBox
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from PIL import Image

try:  # pragma: no cover - optional dependency handling
    import numpy as np
except Exception:  # pragma: no cover - optional dependency handling
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency handling
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0  # deterministic results (constitution VIII)
except Exception:  # pragma: no cover - optional dependency handling
    detect_langs = None

from equation_extraction_pipeline.config import settings as config
from equation_extraction_pipeline.domain.models import (
    CategoryCandidate,
    ClassificationContext,
    ClassificationResult,
    DetectedLanguage,
    DocumentMetadata,
    EquationRegion,
    PageMeta,
    RenderedPage,
)
from equation_extraction_pipeline.extraction.ocr_extractor import OCR_AVAILABLE, ocr_text
from equation_extraction_pipeline.ingestion.pdf_loader import (
    PageView,
    open_document,
    render_page_image,
)

logger = logging.getLogger(__name__)
_structlog = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Section 1 — Classification signals  (from classifier/signals.py)
# ---------------------------------------------------------------------------
# Signals used to classify PDF pages and document regions.

FAKE_FONTS = {
    "",
    "GlyphLessFont",
    "Arial-BoldMT",
    "ArialMT",
    "FFIAEA+GlyphLessFont",
}

__all__ = [
    # signals
    "TEXT_DENSITY",
    "FONT_METADATA",
    "IMAGE_COVERAGE",
    "RENDER_SIMILARITY",
    # language
    "LANGDETECT_AVAILABLE",
    "LanguageResult",
    "detect_languages",
    # doc_signals
    "DocumentSignals",
    "sample_pages",
    "estimate_columns",
    "collect_signals",
    # doc_rules
    "determine_modality",
    "assess_layout_complexity",
    "score_categories",
    "select_category",
    "recommend_strategy",
    # page_classifier
    "classify_page",
    # doc_classifier
    "classify_document",
    # layout detection
    "detect_equations",
    "scan_equation_labels",
]


def _signal_text_density(page: PageView) -> tuple[int, list[str]]:
    words = page.get_text("words")
    word_count = len(words)
    signals_used = ["text_density"]
    return word_count, signals_used


def _signal_font_metadata(page: PageView) -> tuple[bool, list[str]]:
    rawdict = page.get_text("rawdict")
    fonts: set[str] = set()

    for block in rawdict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_name = span.get("font")
                if font_name is not None:
                    fonts.add(font_name)

    has_real_fonts = any(font not in FAKE_FONTS for font in fonts)
    signals_used = ["font_metadata"]
    return has_real_fonts, signals_used


def _signal_image_coverage(page: PageView) -> tuple[float, list[str]]:
    images = page.get_image_info()
    page_area = page.rect.width * page.rect.height
    if page_area == 0:
        return 0.0, ["image_coverage"]

    covered = sum(
        abs((image["bbox"][2] - image["bbox"][0]) * (image["bbox"][3] - image["bbox"][1]))
        for image in images
    )
    coverage = min(covered / page_area, 1.0)
    return coverage, ["image_coverage"]


def _signal_render_similarity(page: PageView, embedded_text: str) -> tuple[float, list[str]]:
    if np is None or not OCR_AVAILABLE:
        return 0.5, ["render_similarity_skipped"]

    image = render_page_image(page)

    height, width = image.shape[:2]
    crop = image[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3]

    recognized = ocr_text(crop)
    similarity = SequenceMatcher(None, embedded_text, recognized).ratio()
    return similarity, ["render_similarity"]


TEXT_DENSITY = _signal_text_density
FONT_METADATA = _signal_font_metadata
IMAGE_COVERAGE = _signal_image_coverage
RENDER_SIMILARITY = _signal_render_similarity


# ---------------------------------------------------------------------------
# Section 2 — Language detection  (from classifier/language.py)
# ---------------------------------------------------------------------------
# Pluggable language-detection backend for document classification.
# The optional library import is guarded; when absent the stage degrades
# to a deterministic stdlib heuristic and records the degradation.

LANGDETECT_AVAILABLE = detect_langs is not None


@dataclass
class LanguageResult:
    """Ranked detected languages plus any degradation note."""

    languages: list[DetectedLanguage] = field(default_factory=list)
    degraded: bool = False
    note: str | None = None


def _undetermined(degraded: bool, note: str | None) -> LanguageResult:
    return LanguageResult(
        languages=[DetectedLanguage(language="und", confidence=0.0)],
        degraded=degraded,
        note=note,
    )


def _detect_with_library(text: str, min_confidence: float) -> LanguageResult:
    """Use the optional langdetect backend; sorted desc, filtered by ``min_confidence``."""
    try:
        ranked = detect_langs(text)
    except Exception:
        return _undetermined(degraded=False, note="no_language_features")

    languages = [
        DetectedLanguage(language=str(item.lang), confidence=float(item.prob))
        for item in ranked
        if float(item.prob) >= min_confidence
    ]
    languages.sort(key=lambda detected: (-detected.confidence, detected.language))
    if not languages:
        return _undetermined(degraded=False, note="below_min_confidence")
    return LanguageResult(languages=languages, degraded=False, note=None)


def detect_languages(
    text: str,
    *,
    backend: str = "default",
    min_confidence: float = 0.10,
) -> LanguageResult:
    """Detect language(s) in ``text``, returning a ranked :class:`LanguageResult`.

    ``backend == "none"`` (or an unavailable library) yields a degraded ``und`` result.
    Empty/whitespace text yields ``und`` without marking degradation (no signal to detect).
    """
    if not text or not text.strip():
        return _undetermined(degraded=False, note="empty_text")

    if backend == "none" or not LANGDETECT_AVAILABLE:
        return _undetermined(degraded=True, note="language_backend_unavailable")

    return _detect_with_library(text, min_confidence)


# ---------------------------------------------------------------------------
# Section 3 — Document signals  (from classifier/doc_signals.py)
# ---------------------------------------------------------------------------
# Aggregates per-page classification results plus lightweight structural
# signals and a bounded prose sample for language detection.

_MAX_PROSE_CHARS = 4000
_NUMERIC_TOKEN = re.compile(r"\d")
_MATH_SYMBOL = re.compile(r"[=±∑∫∂√≤≥≈×÷∞°µ]")


@dataclass
class DocumentSignals:
    """Collected document-level signals feeding the classification rules."""

    total_pages: int = 0
    sampled_pages: list[int] = field(default_factory=list)
    sample_strategy: str = "stratified"
    page_type_counts: dict[str, int] = field(default_factory=dict)
    has_text_layer: bool = False
    multi_column: bool = False
    column_estimate: int = 1
    table_density: float = 0.0
    figure_density: float = 0.0
    equation_density: float = 0.0
    word_count_mean: float = 0.0
    prose_text: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    signals_used: list[str] = field(default_factory=list)


def sample_pages(total_pages: int, strategy: str, cap: int) -> list[int]:
    """Return a deterministic list of page indices to analyze.

    ``total_pages <= cap`` → every page.  Otherwise a stratified, evenly-spaced
    sample of ``cap`` pages that always includes the first and last page.  Any
    unknown strategy falls back to ``stratified`` so the caller always gets a
    valid, repeatable sample.
    """
    if total_pages <= 0:
        return []
    if cap <= 0 or total_pages <= cap:
        return list(range(total_pages))

    if strategy not in {"stratified", "all"}:
        strategy = "stratified"
    if strategy == "all":
        return list(range(total_pages))

    step = (total_pages - 1) / (cap - 1)
    indices = sorted({int(round(i * step)) for i in range(cap)})
    return indices


def estimate_columns(block_x_centers: list[float], page_width: float) -> int:
    """Estimate the number of text columns from block horizontal centers.

    Reports 2 columns when each half holds a meaningful share (≥ 25%) of
    blocks, else 1.  Deterministic and dependency-free.
    """
    if page_width <= 0 or len(block_x_centers) < 4:
        return 1
    mid = page_width / 2.0
    left = sum(1 for x in block_x_centers if x < mid)
    right = len(block_x_centers) - left
    total = left + right
    if total == 0:
        return 1
    if min(left, right) / total >= 0.25:
        return 2
    return 1


def _page_density_indicators(text: str) -> tuple[float, float]:
    """Return (numeric-line ratio, math-symbol-line ratio) for a page's text."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0, 0.0
    numeric = sum(1 for line in lines if len(_NUMERIC_TOKEN.findall(line)) >= 3)
    mathy = sum(1 for line in lines if _MATH_SYMBOL.search(line))
    return numeric / len(lines), mathy / len(lines)


def collect_signals(
    pdf_path: Path,
    *,
    page_manifest: list[PageMeta],
    metadata: DocumentMetadata | None,
    strategy: str,
    cap: int,
    column_threshold: int,
) -> DocumentSignals:
    """Open the document over the sampled pages and assemble :class:`DocumentSignals`."""
    total_pages = len(page_manifest)
    sampled = sample_pages(total_pages, strategy, cap)
    signals = DocumentSignals(
        total_pages=total_pages,
        sampled_pages=sampled,
        sample_strategy=strategy,
        metadata=dict(metadata.raw) if metadata is not None else {},
        signals_used=["page_modality_aggregate"],
    )
    if not sampled:
        return signals

    page_type_counts: Counter[str] = Counter()
    word_counts: list[int] = []
    has_text_layer = False
    image_coverages: list[float] = []
    for index in sampled:
        meta = page_manifest[index]
        page_type_counts[meta.page_type] += 1
        word_counts.append(meta.word_count)
        image_coverages.append(meta.image_coverage)
        if meta.word_count > 0 and meta.has_real_fonts:
            has_text_layer = True
    signals.page_type_counts = dict(page_type_counts)
    signals.has_text_layer = has_text_layer
    signals.figure_density = (
        sum(image_coverages) / len(image_coverages) if image_coverages else 0.0
    )
    signals.word_count_mean = sum(word_counts) / len(word_counts) if word_counts else 0.0

    prose_parts: list[str] = []
    max_columns = 1
    table_ratios: list[float] = []
    math_ratios: list[float] = []
    try:
        with open_document(str(pdf_path)) as document:
            for index in sampled:
                if index >= len(document):
                    continue
                page = document[index]
                text = page.get_text("text") or ""
                if len(" ".join(prose_parts)) < _MAX_PROSE_CHARS:
                    prose_parts.append(text)
                table_ratio, math_ratio = _page_density_indicators(text)
                table_ratios.append(table_ratio)
                math_ratios.append(math_ratio)

                rawdict = page.get_text("dict") or {}
                centers = [
                    (float(block["bbox"][0]) + float(block["bbox"][2])) / 2.0
                    for block in rawdict.get("blocks", [])
                    if block.get("type", 0) == 0 and block.get("bbox")
                ]
                max_columns = max(max_columns, estimate_columns(centers, page.rect.width))
        signals.signals_used.extend(["block_columns", "content_density"])
    except Exception as exc:
        _structlog.warning("doc_signal_collection_degraded", pdf=str(pdf_path), error=str(exc))
        signals.signals_used.append("structural_signals_skipped")

    signals.column_estimate = max_columns
    signals.multi_column = max_columns >= column_threshold
    signals.table_density = sum(table_ratios) / len(table_ratios) if table_ratios else 0.0
    signals.equation_density = sum(math_ratios) / len(math_ratios) if math_ratios else 0.0
    signals.prose_text = " ".join(prose_parts)[:_MAX_PROSE_CHARS]
    if metadata is not None:
        signals.signals_used.append("metadata_cues")
    return signals


# ---------------------------------------------------------------------------
# Section 4 — Classification rules  (from classifier/doc_rules.py)
# ---------------------------------------------------------------------------
# Pure, deterministic classification rules. All functions are side-effect
# free and operate on plain data so they are trivially unit-testable.

_MODALITY_DEFAULT_STRATEGY = {
    "digital": "digital_text_first",
    "scanned": "ocr_first",
    "hybrid": "hybrid_per_page_routed",
}

_CATEGORY_KEYWORDS = {
    "textbook": ("textbook", "introduction to", "lecture", "edition"),
    "reference_handbook": ("handbook", "reference", "encyclopedia", "manual"),
    "journal_article": ("journal", "doi", "abstract", "vol."),
    "conference_paper": ("proceedings", "conference", "symposium", "workshop"),
    "technical_report": ("report", "technical memorandum", "white paper"),
    "standard_specification": ("standard", "specification", "iso ", "astm", "ansi", "ieee std"),
    "datasheet": ("datasheet", "data sheet", "product specification"),
    "patent": ("patent", "claims", "united states patent", "application no"),
}


def determine_modality(
    page_type_counts: dict[str, int],
    *,
    digital_threshold: float,
    scanned_threshold: float,
) -> tuple[str, float, dict[str, float]]:
    """Aggregate per-page modality into a document modality with banding (FR-002).

    Returns ``(modality, confidence, proportions)``.  ``hybrid`` is the mixed
    band when neither digital nor scanned share clears its threshold.
    """
    total = sum(page_type_counts.values())
    if total == 0:
        return "scanned", 0.0, {}

    proportions = {
        ptype: round(count / total, 6)
        for ptype, count in sorted(page_type_counts.items())
    }
    digital = proportions.get("digital", 0.0)
    scanned = proportions.get("scanned", 0.0)

    if digital >= digital_threshold:
        return "digital", digital, proportions
    if scanned >= scanned_threshold:
        return "scanned", scanned, proportions
    confidence = round(1.0 - max(digital, scanned), 6)
    return "hybrid", confidence, proportions


def assess_layout_complexity(
    *,
    multi_column: bool,
    table_density: float,
    figure_density: float,
    equation_density: float,
    moderate_density: float,
    complex_density: float,
) -> tuple[str, float]:
    """Map column structure + content density to a complexity level (FR-005)."""
    density = max(table_density, figure_density, equation_density)
    if multi_column or density >= complex_density:
        if density >= complex_density or (multi_column and density >= moderate_density):
            return "complex", round(min(1.0, 0.6 + density), 6)
        return "moderate", round(min(1.0, 0.5 + density), 6)
    if density >= moderate_density:
        return "moderate", round(min(1.0, 0.5 + density), 6)
    return "simple", round(min(1.0, 0.7 + (moderate_density - density)), 6)


def _category_raw_score(category: str, signals: DocumentSignals) -> float:
    """Heuristic non-negative score for a single category from document signals."""
    title = (
        signals.metadata.get("Title", "") + " " + signals.metadata.get("title", "")
    ).lower()
    keywords = _CATEGORY_KEYWORDS.get(category, ())
    keyword_hits = sum(1 for kw in keywords if kw in title)
    score = 2.5 * keyword_hits

    pages = signals.total_pages
    if category == "textbook":
        score += 1.0 if pages >= 150 else 0.2
        score += 0.4 if signals.equation_density >= 0.05 else 0.0
    elif category == "reference_handbook":
        score += 1.0 if (pages >= 200 and signals.table_density >= 0.15) else 0.2
    elif category == "journal_article":
        score += 1.0 if (pages <= 40 and signals.multi_column) else 0.1
    elif category == "conference_paper":
        score += 0.9 if (pages <= 30 and signals.multi_column) else 0.1
        score += 0.3 if signals.figure_density >= 0.1 else 0.0
    elif category == "technical_report":
        score += 0.6 if 20 <= pages <= 200 else 0.2
    elif category == "standard_specification":
        score += 0.5 if signals.table_density >= 0.1 else 0.1
    elif category == "datasheet":
        score += 1.0 if (pages <= 20 and signals.table_density >= 0.25) else 0.1
    elif category == "patent":
        score += 0.4
    return max(0.0, score)


def score_categories(
    signals: DocumentSignals, taxonomy: list[str]
) -> list[CategoryCandidate]:
    """Score every taxonomy category and return candidates sorted by confidence desc.

    Confidence is each category's share of the total score (relative), so a
    flat distribution yields low confidences that fall back to ``unknown``
    downstream (FR-003).
    """
    raw = {category: _category_raw_score(category, signals) for category in taxonomy}
    total = sum(raw.values())
    if total <= 0:
        return []
    candidates = [
        CategoryCandidate(category=category, confidence=round(score / total, 6))
        for category, score in raw.items()
        if score > 0
    ]
    candidates.sort(key=lambda c: (-c.confidence, c.category))
    return candidates


def select_category(
    candidates: list[CategoryCandidate],
    *,
    threshold: float,
    margin: float,
) -> tuple[str, float, str | None]:
    """Pick the top category if it clears threshold AND beats the runner-up by margin.

    Returns ``(category, confidence, fallback_reason)``; ``fallback_reason`` is
    non-None exactly when the result is ``unknown`` (FR-003).
    """
    if not candidates:
        return "unknown", 0.0, "insufficient_signal"
    top = candidates[0]
    if top.confidence < threshold:
        return "unknown", top.confidence, "below_threshold"
    runner_up = candidates[1].confidence if len(candidates) > 1 else 0.0
    if (top.confidence - runner_up) < margin:
        return "unknown", top.confidence, "below_margin"
    return top.category, top.confidence, None


def recommend_strategy(
    *,
    modality: str,
    category: str,
    layout_complexity: str,
    strategy_map_json: str,
    default_strategy: str,
) -> str:
    """Map (modality, category, layout) → a defined strategy id; always returns one (FR-007).

    Lookup order: exact ``"<modality>|<category>|<layout>"`` → ``"<modality>"``
    → the per-modality built-in default → ``default_strategy``.
    """
    try:
        strategy_map = json.loads(strategy_map_json) if strategy_map_json else {}
        if not isinstance(strategy_map, dict):
            strategy_map = {}
    except (ValueError, TypeError):
        strategy_map = {}

    exact = f"{modality}|{category}|{layout_complexity}"
    if exact in strategy_map:
        return str(strategy_map[exact])
    if modality in strategy_map:
        return str(strategy_map[modality])
    return _MODALITY_DEFAULT_STRATEGY.get(modality, default_strategy)


# ---------------------------------------------------------------------------
# Section 5 — Page classifier  (from classifier/page_classifier.py)
# ---------------------------------------------------------------------------
# Page classification logic for digital, scanned, and hybrid PDFs.


def classify_page(page: PageView, page_no: int) -> PageMeta:
    """Classify a single PDF page as digital, scanned, or hybrid."""
    try:
        word_count, signals_used = TEXT_DENSITY(page)
        has_real_fonts, font_signals = FONT_METADATA(page)
        image_coverage, image_signals = IMAGE_COVERAGE(page)
        signals_used = signals_used + font_signals + image_signals

        page_type: str
        confidence: float
        render_similarity: float | None = None

        if (
            word_count > config.CLASSIFIER_WORD_COUNT_THRESHOLD
            and has_real_fonts
            and image_coverage < 0.30
        ):
            page_type = "digital"
            confidence = 0.95
        elif (
            word_count > config.CLASSIFIER_WORD_COUNT_THRESHOLD
            and has_real_fonts
            and 0.30 <= image_coverage < config.CLASSIFIER_IMAGE_COVERAGE_THRESHOLD
        ):
            embedded_text = page.get_text("text")
            render_similarity, render_signals = RENDER_SIMILARITY(page, embedded_text)
            signals_used += render_signals
            if render_similarity > config.CLASSIFIER_RENDER_SIMILARITY_THRESHOLD:
                page_type = "digital"
                confidence = render_similarity
            elif render_similarity < config.CLASSIFIER_AMBIGUOUS_LOWER:
                page_type = "scanned"
                confidence = 1 - render_similarity
            else:
                page_type = "hybrid"
                confidence = 0.65
        elif word_count > config.CLASSIFIER_WORD_COUNT_THRESHOLD and not has_real_fonts:
            page_type = "hybrid"
            confidence = 0.75
        elif (
            word_count <= config.CLASSIFIER_WORD_COUNT_THRESHOLD
            and image_coverage >= config.CLASSIFIER_IMAGE_COVERAGE_THRESHOLD
        ):
            page_type = "scanned"
            confidence = 0.92
        else:
            page_type = "scanned"
            confidence = 0.70

        return PageMeta(
            page_no=page_no,
            page_type=page_type,
            word_count=word_count,
            has_real_fonts=has_real_fonts,
            image_coverage=image_coverage,
            render_similarity=render_similarity,
            orientation=page.rotation,
            classification_confidence=confidence,
            signals_used=signals_used,
        )
    except Exception:
        return PageMeta(
            page_no=page_no,
            page_type="scanned",
            word_count=0,
            has_real_fonts=False,
            image_coverage=0.0,
            render_similarity=None,
            orientation=page.rotation if hasattr(page, "rotation") else 0,
            classification_confidence=0.0,
            signals_used=["error_fallback"],
        )


# ---------------------------------------------------------------------------
# Section 6 — Document classifier  (from classifier/doc_classifier.py)
# ---------------------------------------------------------------------------
# Orchestrates signal collection → rules → language detection into a single
# ClassificationContext. Per-document failures are contained (outcome="failed")
# and analyzer unavailability degrades gracefully (outcome="degraded").


def _taxonomy() -> list[str]:
    return [
        c.strip()
        for c in config.CLASSIFIER_DOC_CATEGORIES.split(",")
        if c.strip()
    ]


def classify_document(
    pdf_path: Path,
    *,
    page_manifest: list[PageMeta],
    metadata: DocumentMetadata | None,
    config_hash: str = "",
) -> ClassificationContext:
    """Produce the document-level Classification Context for ``pdf_path``."""
    try:
        signals = collect_signals(
            pdf_path,
            page_manifest=page_manifest,
            metadata=metadata,
            strategy=config.CLASSIFIER_DOC_PAGE_SAMPLE_STRATEGY,
            cap=config.CLASSIFIER_DOC_PAGE_SAMPLE_CAP,
            column_threshold=config.CLASSIFIER_DOC_LAYOUT_COLUMN_THRESHOLD,
        )

        modality, modality_conf, proportions = determine_modality(
            signals.page_type_counts,
            digital_threshold=config.CLASSIFIER_DOC_MODALITY_DIGITAL_THRESHOLD,
            scanned_threshold=config.CLASSIFIER_DOC_MODALITY_SCANNED_THRESHOLD,
        )

        layout, layout_conf = assess_layout_complexity(
            multi_column=signals.multi_column,
            table_density=signals.table_density,
            figure_density=signals.figure_density,
            equation_density=signals.equation_density,
            moderate_density=config.CLASSIFIER_DOC_LAYOUT_MODERATE_DENSITY,
            complex_density=config.CLASSIFIER_DOC_LAYOUT_COMPLEX_DENSITY,
        )

        candidates = score_categories(signals, _taxonomy())
        category, category_conf, fallback_reason = select_category(
            candidates,
            threshold=config.CLASSIFIER_DOC_CATEGORY_THRESHOLD,
            margin=config.CLASSIFIER_DOC_CATEGORY_MARGIN,
        )

        language_result = detect_languages(
            signals.prose_text,
            backend=config.CLASSIFIER_DOC_LANGUAGE_BACKEND,
            min_confidence=config.CLASSIFIER_DOC_LANGUAGE_MIN_CONFIDENCE,
        )
        detected = language_result.languages
        dominant = detected[0].language if detected else "und"

        strategy = recommend_strategy(
            modality=modality,
            category=category,
            layout_complexity=layout,
            strategy_map_json=config.CLASSIFIER_DOC_STRATEGY_MAP,
            default_strategy=config.CLASSIFIER_DOC_STRATEGY_DEFAULT,
        )

        signals_used = list(signals.signals_used)
        if detected and dominant != "und":
            signals_used.append("language_backend")

        outcome = "classified"
        degradation_notes: list[str] = []
        if language_result.degraded:
            outcome = "degraded"
            if language_result.note:
                degradation_notes.append(language_result.note)

        verdict_confidences = [modality_conf, layout_conf]
        if category != "unknown":
            verdict_confidences.append(category_conf)
        if detected and dominant != "und":
            verdict_confidences.append(detected[0].confidence)
        overall = round(mean(verdict_confidences), 6) if verdict_confidences else 0.0

        context = ClassificationContext(
            modality=modality,
            modality_confidence=round(modality_conf, 6),
            page_type_proportions=proportions,
            category=category,
            category_confidence=round(category_conf, 6),
            category_candidates=candidates,
            category_fallback_reason=fallback_reason,
            dominant_language=dominant,
            detected_languages=detected,
            layout_complexity=layout,
            layout_confidence=round(layout_conf, 6),
            characteristics={
                "page_count": signals.total_pages,
                "has_text_layer": signals.has_text_layer,
                "multi_column": signals.multi_column,
                "column_estimate": signals.column_estimate,
                "table_density": round(signals.table_density, 6),
                "figure_density": round(signals.figure_density, 6),
                "equation_density": round(signals.equation_density, 6),
                "word_count_mean": round(signals.word_count_mean, 6),
            },
            recommended_strategy=strategy,
            overall_confidence=overall,
            sampling={
                "strategy": signals.sample_strategy,
                "cap": config.CLASSIFIER_DOC_PAGE_SAMPLE_CAP,
                "total_pages": signals.total_pages,
                "sampled_pages": signals.sampled_pages,
            },
            signals_used=signals_used,
            outcome=outcome,
            degradation_notes=degradation_notes,
            config_hash=config_hash,
        )
        _structlog.info(
            "document_classified",
            pdf=str(pdf_path),
            modality=modality,
            category=category,
            dominant_language=dominant,
            layout_complexity=layout,
            recommended_strategy=strategy,
            overall_confidence=overall,
            outcome=outcome,
        )
        return context
    except Exception as exc:
        _structlog.error(
            "document_classification_failed", pdf=str(pdf_path), error=str(exc)
        )
        strategy = recommend_strategy(
            modality="scanned",
            category="unknown",
            layout_complexity="simple",
            strategy_map_json=config.CLASSIFIER_DOC_STRATEGY_MAP,
            default_strategy=config.CLASSIFIER_DOC_STRATEGY_DEFAULT,
        )
        return ClassificationContext(
            modality="scanned",
            category="unknown",
            category_fallback_reason="insufficient_signal",
            dominant_language="und",
            layout_complexity="simple",
            recommended_strategy=strategy,
            overall_confidence=0.0,
            outcome="failed",
            degradation_notes=["classification_error"],
            config_hash=config_hash,
        )


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


def _normalise_label(raw: str) -> str:
    return re.sub(r"\s+", "", raw).translate(_LABEL_OCR_FIX)


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
    x0 = min(box.bbox[0] for box in aligned)
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
    r"^\s*\(\s*(\d{1,3}(?:\.\d{1,3}){1,3})\s*\)\s*$",
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


def _save_crop(
    page_image: Image.Image,
    bbox_points: tuple[float, float, float, float],
    page_number: int,
    dpi: int,
    eq_id: str,
    crops_dir: Path,
) -> str:
    """Crop the equation region from the page image and save as PNG.

    Returns the path relative to the book output directory.
    """
    x0, y0, x1, y1 = bbox_points
    scale = dpi / config.PDF_POINTS_PER_INCH
    pad = config.CROP_PADDING_PX

    px0 = max(0, int(x0 * scale) - pad)
    py0 = max(0, int(y0 * scale) - pad)
    px1 = min(page_image.width,  int(x1 * scale) + pad)
    py1 = min(page_image.height, int(y1 * scale) + pad)

    if px1 <= px0 or py1 <= py0:
        logger.warning("invalid_crop eq_id=%s bbox=%s", eq_id, bbox_points)
        return ""

    crop = page_image.crop((px0, py0, px1, py1))
    page_dir = crops_dir / f"page_{page_number:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    dest = page_dir / f"{eq_id}.png"
    crop.save(dest, format="PNG")

    return str(Path("crops") / f"page_{page_number:03d}" / f"{eq_id}.png")


def _extract_page_layout(pdf_path: Path) -> list[tuple[int, list[LTTextBox]]]:
    """Return (0-based page index, list of LTTextBox) for all pages."""
    rsrcmgr = PDFResourceManager()
    laparams = LAParams(
        line_overlap=0.5, char_margin=2.0, line_margin=0.5, word_margin=0.1
    )
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
                logger.debug(
                    "layout_extract_failed page=%d error=%s", page_idx, exc
                )
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
    for _page_idx, boxes in _extract_page_layout(Path(pdf_path)):
        formula_boxes: list[LTTextBox] = []
        candidates: list[tuple[str, LTTextBox, bool]] = []
        for box in boxes:
            text = box.get_text().strip()
            found_any = False
            for m in _LABEL_RE.finditer(text):
                found_any = True
                if not _is_cross_reference_for_match(text, m):
                    candidates.append((_normalise_label(m.group(1)), box, True))
            if found_any:
                if not any(c[1] is box for c in candidates):
                    formula_boxes.append(box)
                continue
            dpm = _PAREN_DOTTED_LABEL_RE.match(text)
            if dpm:
                candidates.append((dpm.group(1), box, True))
                continue
            paren = _PAREN_LABEL_RE.match(text)
            if paren:
                candidates.append((paren.group(1), box, False))
            else:
                formula_boxes.append(box)

        for label, label_box, explicit_eq_label in candidates:
            if (
                not explicit_eq_label
                and _formula_bbox_for_label(label_box, formula_boxes) is None
            ):
                continue
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def _is_cross_reference_for_match(text: str, m: re.Match) -> bool:
    """True when this specific label match is embedded in prose (cross-reference).

    Examines the same LINE as the label match.  Requires ≥2 prose-word hits
    rather than 1 so that mathematical qualifiers like "for", "or", "in" do
    not mis-classify equation definitions such as "f(x) = 1  for  x > 0
    Eq. 5.5.11".
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
            for m in _LABEL_RE.finditer(text):
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
                logger.debug(
                    "label_duplicate_skipped label=%s page=%d", label_str, page_number
                )
                continue
            lx0, ly0, lx1, ly1 = label_box.bbox

            formula_bbox = _formula_bbox_for_label(label_box, formula_boxes)
            if formula_bbox is not None:
                fx0, fy0, fx1, fy1 = formula_bbox
            elif not explicit_eq_label:
                logger.debug(
                    "parenthesized_list_marker_skipped label=%s page=%d",
                    label_str, page_number,
                )
                continue
            else:
                fx0, fy0, fx1, fy1 = _image_formula_bbox_for_label(label_box)

            seen_labels.add(label_str)

            page_height_pts = rp.height_px / (rp.dpi / config.PDF_POINTS_PER_INCH)
            bbox = (fx0, page_height_pts - fy1, fx1, page_height_pts - fy0)

            safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label_str).strip("_")
            eq_id = f"eq_{eq_counter}_p{page_number}_{safe_label}"
            eq_counter += 1

            crop_rel = _save_crop(rp.image, bbox, page_number, rp.dpi, eq_id, crops_dir)
            regions.append(EquationRegion(
                page_number=page_number,
                equation_id=eq_id,
                label=label_str,
                bbox=bbox,
                detection_method="label",
                crop_path=crop_rel or None,
            ))

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

            bbox = (
                float(bbox_obj.l),
                float(bbox_obj.t),
                float(bbox_obj.r),
                float(bbox_obj.b),
            )
            eq_id = f"eq_{eq_counter}_p{page_number}_ml"
            eq_counter += 1

            crop_rel = _save_crop(rp.image, bbox, page_number, rp.dpi, eq_id, crops_dir)
            regions.append(EquationRegion(
                page_number=page_number,
                equation_id=eq_id,
                label=None,
                bbox=bbox,
                detection_method="ml",
                crop_path=crop_rel or None,
            ))
    except Exception as exc:
        logger.error("ml_detection_failed error=%s", exc)

    return regions


def detect_equations(
    pdf_path: Path,
    pages: list[RenderedPage],
    classification: ClassificationResult,
    output_dir: Path,
) -> list[EquationRegion]:
    """Detect all equation regions and save crop images.

    Parameters
    ----------
    pdf_path:
        Path to the input PDF.
    pages:
        Preprocessed page images from preprocessing.preprocess_pages().
    classification:
        Drives mode selection (label vs ML).
    output_dir:
        Book-level output directory (e.g. ``data/output/28120_12/``).
        Crops are written to ``<output_dir>/crops/page_NNN/<eq_id>.png``.

    Returns
    -------
    list[EquationRegion]
        Detected regions sorted by (page_number, y-coordinate).
    """
    pdf_path = Path(pdf_path)
    crops_dir = output_dir / "crops"

    logger.info("detecting equations pdf=%s", pdf_path.name)

    regions = _find_labeled_equations(pdf_path, pages, crops_dir)

    if regions:
        logger.info("label_detection found=%d equations", len(regions))
    else:
        logger.info("no labels found; falling back to ML detection")
        regions = _find_ml_equations(pdf_path, pages, crops_dir)
        logger.info("ml_detection found=%d equations", len(regions))

    regions.sort(key=lambda r: (r.page_number, r.bbox[1]))
    return regions
