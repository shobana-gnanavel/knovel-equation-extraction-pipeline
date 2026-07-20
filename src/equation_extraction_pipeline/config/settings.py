"""Central configuration for the equation_extraction_pipeline package.

All settings are read from environment variables with documented defaults.
Copy .env.example to .env and adjust values for your environment.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # pragma: no cover - optional dependency fallback

# ---------------------------------------------------------------------------
# Base paths (standalone)
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent

INPUT_DIR: Path = PROJECT_ROOT / os.getenv("INPUT_DIR", "data/input")
OUTPUT_DIR: Path = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "data/output")

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# KNOVEL paths (pipeline)
# ---------------------------------------------------------------------------

KNOVEL_INPUT_DIR: Path = Path(os.getenv("KNOVEL_INPUT_DIR", "data/input"))
KNOVEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)

KNOVEL_OUTPUT_DIR: Path = Path(os.getenv("KNOVEL_OUTPUT_DIR", "data/output"))
KNOVEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stage sidecar cache directory. When set, all per-stage JSON sidecars (classification,
# preprocessing, layout, etc.) are written here as <cache_dir>/<pdf_stem>/<stage>.json
# instead of alongside the PDF. The output directory then holds only the final
# document.json and CSVs — no intermediate files. Defaults to data/_cache.
KNOVEL_CACHE_DIR: Path = Path(os.getenv("KNOVEL_CACHE_DIR", "data/_cache"))
KNOVEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def resolve_sidecar_path(pdf_path: Path, stage: str) -> Path:
    """Return the sidecar cache path for *stage* relative to *pdf_path*.

    Sidecars go to ``KNOVEL_CACHE_DIR/<pdf_stem>/<stage>.json`` so that the input
    directory stays clean and the output directory contains only final artifacts.
    """
    cache_dir = KNOVEL_CACHE_DIR / pdf_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{stage}.json"


# ---------------------------------------------------------------------------
# Ollama / VL model
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VL_MODEL: str = os.getenv("OLLAMA_VL_MODEL", "qwen2.5vl:7b")
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# DPI used when rendering scanned PDFs (higher → sharper crops)
RENDER_DPI_SCANNED: int = int(os.getenv("RENDER_DPI_SCANNED", "300"))

# DPI used when rendering digital (born-digital) PDFs
RENDER_DPI_DIGITAL: int = int(os.getenv("RENDER_DPI_DIGITAL", "216"))

# ---------------------------------------------------------------------------
# Preprocessing (standalone)
# ---------------------------------------------------------------------------

# Deskew a page only when detected skew angle exceeds this threshold (degrees)
DESKEW_THRESHOLD_DEG: float = float(os.getenv("DESKEW_THRESHOLD_DEG", "0.3"))

# Unsharp-mask radius for sharpening (pixels)
SHARPEN_RADIUS: int = int(os.getenv("SHARPEN_RADIUS", "1"))

# Unsharp-mask amount (strength), 0.0 = off
SHARPEN_AMOUNT: float = float(os.getenv("SHARPEN_AMOUNT", "1.5"))

# ---------------------------------------------------------------------------
# Crop / layout
# ---------------------------------------------------------------------------

# Fixed padding added on each side of every equation crop (pixels)
CROP_PADDING_PX: int = int(os.getenv("CROP_PADDING_PX", "8"))

# Trim prose lines that bleed into a label-anchored crop via horizontal ink projection.
CROP_TIGHTEN_ENABLED: bool = os.getenv("CROP_TIGHTEN_ENABLED", "true").lower() == "true"

# Expand the crop's vertical window by this many label-heights on each side before
# tightening, to recover equations the bbox under-captured (clipped). DISABLED by default
# (0.0): expansion adds prose that tightening then over-trims (splitting fractions) or
# leaves in — a coupled trade-off that is unstable to tune. The robust fix for clipped
# bboxes is detector-driven regions; until then this stays off. Set >0 to experiment.
# (Measured on 28120_12: 0.6 recovered ~6 clean clips but regressed ~3 accepted equations
# by pulling in adjacent prose; lowering the gap factor to trim it split fraction
# denominators. No safe global value — hence 0.0.)
CROP_VEXPAND_FACTOR: float = float(os.getenv("CROP_VEXPAND_FACTOR", "0.0"))

# A whitespace gap larger than this many label-heights separates prose from the equation;
# smaller gaps (fraction bar, sub/superscripts) are kept so equations are never split.
CROP_TIGHTEN_GAP_FACTOR: float = float(os.getenv("CROP_TIGHTEN_GAP_FACTOR", "0.5"))

# PDF points per inch (constant — do not change)
PDF_POINTS_PER_INCH: float = 72.0

# Equation-region detector for detect_equations():
#   "label"  — label-scan + geometry-reconstruction crops. DEFAULT: controlled A/B on 28120_12
#              (judge off, identical settings) measured label crops at 0.825 mean token-sim vs
#              0.731 for hybrid/Docling crops — the legacy crops recognize better on
#              born-digital books, so they keep the default until hybrid closes the gap.
#   "hybrid" — Docling vision detector supplies the crop bbox; label scan scopes/numbers;
#              label-reconstruction is the fallback. Same 56/56 recall. Use for SCANNED or
#              unlabeled books, where the label/geometry path cannot work at all (no text
#              layer) but Docling still detects (validated on an image-only proxy).
#   "docling"— Docling regions only (no label scoping); every detected formula is cropped.
# Default flipped to "hybrid" (2026-07-17) after the 17-book gate: detection recall 452/452
# labeled equations across 4 label conventions, 0 blank / 7 sliver crops corpus-wide, and
# 28120_12 gold token-sim 0.828 vs 0.836 for legacy label crops (within the 0.02 gate).
# The legacy path remains available via EQUATION_DETECTOR=label.
EQUATION_DETECTOR: str = os.getenv("EQUATION_DETECTOR", "hybrid")

# Fixed fractional pad added on each side of a Docling-sourced crop (no ink-projection
# tightening — a model bbox is already tight, so the pad is the only knob). Fraction of the
# region's own width/height.
CROP_PAD_FRAC: float = float(os.getenv("CROP_PAD_FRAC", "0.05"))

# Judge-assisted repair: when the GPT judge rejects a transcription but confirms the crop is
# valid and supplies its own corrected reading, adopt the correction IF a fresh judge pass
# accepts it (bounded to one attempt; provenance recorded via the JUDGE_REPAIR ocr flag).
# Main effect: recovers derivation-group transcriptions where the VLM omitted a line.
JUDGE_REPAIR_ENABLED: bool = os.getenv("JUDGE_REPAIR_ENABLED", "true").lower() == "true"

# Bare whole-box numbers ("1.6" alone in a text box) as equation labels. Default OFF:
# measured on 79462_02/04, table values and chart axis ticks satisfy every text-level guard
# (40 -> 122 phantom labels). Enable only for a book that genuinely uses this convention,
# and validate its scan count + contact sheet first.
BARE_NUMBER_LABELS_ENABLED: bool = (
    os.getenv("BARE_NUMBER_LABELS_ENABLED", "false").lower() == "true"
)

# Absolute floor for that pad, in PDF points. A fractional pad collapses to <1pt on the thin
# strip regions Docling emits for degraded scans (measured: a 16pt band on a 30pt fraction,
# 39896_02 eq 2-27), clipping numerators/limits; a few points of guaranteed pad recovers them.
CROP_MIN_PAD_PTS: float = float(os.getenv("CROP_MIN_PAD_PTS", "4.0"))

# ---------------------------------------------------------------------------
# Image-based math documents (equations embedded as raster images)
# ---------------------------------------------------------------------------
# Some books (e.g. IDOSR-JAS-E.-BOOK-1-003) render every display equation AND its margin
# number as an embedded raster image; only the surrounding prose is in the text layer. The
# label scanner is blind to such equations and Docling is unreliable on exactly those pages
# (validated: 148 formula regions doc-wide but 0 on the image-equation pages). For these pages
# a VLM enumerates equations from the rendered page image (reuses the Portkey vision infra).

# Turn the whole image-math handling on/off. Default ON; degrades safely to the existing
# behaviour when the VLM (Portkey) is unconfigured.
IMAGE_MATH_FALLBACK_ENABLED: bool = (
    os.getenv("IMAGE_MATH_FALLBACK_ENABLED", "true").lower() == "true"
)

# A page is "image-math" when it carries at least this many embedded images (pdfminer
# LTImage/LTFigure count). Display equations rendered as images produce many small rasters;
# ordinary text/figure pages have few.
IMAGE_MATH_MIN_IMAGES_PER_PAGE: int = int(
    os.getenv("IMAGE_MATH_MIN_IMAGES_PER_PAGE", "6")
)

# The document is flagged image-math when at least this fraction of its content pages are
# image-math AND the label/hybrid scan produced few confidently-labeled regions. Fraction of
# pages that have any embedded images (content pages), not of all pages.
IMAGE_MATH_DOC_PAGE_FRACTION: float = float(
    os.getenv("IMAGE_MATH_DOC_PAGE_FRACTION", "0.30")
)

# Max tokens for the VLM page-equation enumeration response (larger than the judge default
# because a page can hold many equations, each with a LaTeX transcription).
IMAGE_MATH_MAX_TOKENS: int = int(os.getenv("IMAGE_MATH_MAX_TOKENS", "3000"))

# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

# Equations with OCR confidence below this threshold trigger a retry
RECOGNITION_RETRY_THRESHOLD: float = float(os.getenv("RECOGNITION_RETRY_THRESHOLD", "0.60"))

# Equations with final confidence below this are marked UNCERTAIN in output
RECOGNITION_MIN_CONFIDENCE: float = float(os.getenv("RECOGNITION_MIN_CONFIDENCE", "0.65"))

# Zoom factor applied to crop image for the retry pass (higher → better detail)
RECOGNITION_RETRY_ZOOM: float = float(os.getenv("RECOGNITION_RETRY_ZOOM", "1.5"))

# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

# Set to false to skip judge step (faster runs, no quality gating)
JUDGE_ENABLED: bool = os.getenv("JUDGE_ENABLED", "true").lower() == "true"

# Judge scores at or above this threshold cause the equation to be accepted
JUDGE_ACCEPT_THRESHOLD: float = float(os.getenv("JUDGE_ACCEPT_THRESHOLD", "0.70"))

# Which judge backend to use: "portkey" (external GPT vision judge, authoritative)
# or "legacy" (local Qwen self-judge via Ollama /api/generate).
JUDGE_BACKEND: str = os.getenv("JUDGE_BACKEND", "portkey")

# Max tokens for the GPT judge responses (equation verdict + page completeness JSON).
JUDGE_MAX_TOKENS: int = int(os.getenv("JUDGE_MAX_TOKENS", "1024"))

# Run the per-page completeness audit ("were all equations extracted?").
JUDGE_PAGE_COMPLETENESS_ENABLED: bool = (
    os.getenv("JUDGE_PAGE_COMPLETENESS_ENABLED", "true").lower() == "true"
)

# Production audit policy. When gated (default), the inline per-page completeness audit runs
# ONLY when a cheap risk signal trips — the document was flagged image-math, or a page's
# extracted-equation count is anomalously low — instead of on every page of every book. When
# the audit reports missed equations, one bounded VLM re-extract pass recovers them (no loop).
# Set to false to restore the unconditional audit (audits every page when COMPLETENESS enabled).
JUDGE_COMPLETENESS_GATED: bool = (
    os.getenv("JUDGE_COMPLETENESS_GATED", "true").lower() == "true"
)

# Bounded re-extract: cap of VLM re-extract passes triggered by the gated audit (safety, no
# unbounded loop). 1 = a single recovery pass over pages the audit flagged incomplete.
JUDGE_REEXTRACT_MAX_PASSES: int = int(os.getenv("JUDGE_REEXTRACT_MAX_PASSES", "1"))

# ---------------------------------------------------------------------------
# Provider aliases used by providers.py (QwenVLProvider)
# These mirror the OLLAMA_* vars under the KNOVEL_EQUATION_* namespace so
# the production providers.py and the standalone providers.py use the same
# attribute names when reading from the config object.
# ---------------------------------------------------------------------------

KNOVEL_OLLAMA_BASE_URL: str = os.getenv("KNOVEL_OLLAMA_BASE_URL", OLLAMA_BASE_URL)
KNOVEL_EQUATION_VL_MODEL: str = os.getenv("KNOVEL_EQUATION_VL_MODEL", OLLAMA_VL_MODEL)
KNOVEL_EQUATION_VL_TIMEOUT: float = float(os.getenv("KNOVEL_EQUATION_VL_TIMEOUT", str(OLLAMA_TIMEOUT)))
KNOVEL_EQUATION_VL_MAX_TOKENS: int = int(os.getenv("KNOVEL_EQUATION_VL_MAX_TOKENS", "512"))
# Longest-side pixel cap for images sent to the VL OCR model. Full-page fallback
# crops (~2480×3508 @300 DPI) otherwise swamp the model with vision tokens and hit
# the request timeout. Qwen2.5-VL downsamples internally, so equation legibility is
# unaffected at this cap. Set ≤0 to disable capping.
KNOVEL_EQUATION_VL_MAX_IMAGE_PX: int = int(os.getenv("KNOVEL_EQUATION_VL_MAX_IMAGE_PX", "1600"))
KNOVEL_EQUATION_PROVIDER_MAP: str = os.getenv("KNOVEL_EQUATION_PROVIDER_MAP", "")

# Representation flags (read by representations.py)
KNOVEL_EQUATION_LATEX_ENABLED: bool = os.getenv("KNOVEL_EQUATION_LATEX_ENABLED", "true").lower() == "true"
KNOVEL_EQUATION_MATHML_ENABLED: bool = os.getenv("KNOVEL_EQUATION_MATHML_ENABLED", "false").lower() == "true"
KNOVEL_EQUATION_STRUCTURED_ENABLED: bool = os.getenv("KNOVEL_EQUATION_STRUCTURED_ENABLED", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "5000"))
DASHBOARD_DEBUG: bool = os.getenv("DASHBOARD_DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Pipeline: LLM / concurrency settings
# ---------------------------------------------------------------------------

KNOVEL_LLM_BACKEND: str = os.getenv("KNOVEL_LLM_BACKEND", "ollama")
KNOVEL_OLLAMA_FAST_MODEL: str = (
    os.getenv("KNOVEL_OLLAMA_FAST_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
)
KNOVEL_OLLAMA_COMPLEX_MODEL: str = os.getenv("KNOVEL_OLLAMA_COMPLEX_MODEL", "qwen2.5vl:72b")
KNOVEL_PORTKEY_API_KEY: str = os.getenv("KNOVEL_PORTKEY_API_KEY", "")
KNOVEL_PORTKEY_BASE_URL: str = os.getenv("KNOVEL_PORTKEY_BASE_URL", "")
KNOVEL_PORTKEY_MODEL: str = os.getenv("KNOVEL_PORTKEY_MODEL", "")
KNOVEL_MAX_WORKERS: int = int(os.getenv("KNOVEL_MAX_WORKERS", "4"))
KNOVEL_LLM_MAX_WORKERS: int = int(os.getenv("KNOVEL_LLM_MAX_WORKERS", "2"))
KNOVEL_LLM_TIMEOUT: int = int(os.getenv("KNOVEL_LLM_TIMEOUT", "120"))
KNOVEL_LLM_MAX_RETRIES: int = int(os.getenv("KNOVEL_LLM_MAX_RETRIES", "3"))

CLASSIFIER_WORD_COUNT_THRESHOLD: int = int(os.getenv("CLASSIFIER_WORD_COUNT_THRESHOLD", "20"))
CLASSIFIER_IMAGE_COVERAGE_THRESHOLD: float = float(
    os.getenv("CLASSIFIER_IMAGE_COVERAGE_THRESHOLD", "0.70")
)
CLASSIFIER_RENDER_SIMILARITY_THRESHOLD: float = float(
    os.getenv("CLASSIFIER_RENDER_SIMILARITY_THRESHOLD", "0.85")
)
CLASSIFIER_AMBIGUOUS_LOWER: float = float(os.getenv("CLASSIFIER_AMBIGUOUS_LOWER", "0.50"))

# Ingestion stage (feature 002). MANIFEST_DIR / INDEX_PATH empty => derived from the
# per-run output directory (so batch runs and tests stay self-contained).
KNOVEL_INGESTION_DUPLICATE_POLICY: str = os.getenv("KNOVEL_INGESTION_DUPLICATE_POLICY", "skip")
KNOVEL_INGESTION_MANIFEST_DIR: str = os.getenv("KNOVEL_INGESTION_MANIFEST_DIR", "")
KNOVEL_INGESTION_INDEX_PATH: str = os.getenv("KNOVEL_INGESTION_INDEX_PATH", "")
KNOVEL_INGESTION_MAX_FILE_MB: int = int(os.getenv("KNOVEL_INGESTION_MAX_FILE_MB", "0"))

# Document-level classification stage (feature 003). All defaults are tunable; calibrate
# against the benchmark set. Categories is a comma-separated taxonomy ("unknown" is always
# an implicit fallback). The strategy map is an optional JSON object keyed by
# "<modality>|<category>|<layout>" or "<modality>"; unmatched tuples fall back to the
# per-modality default and finally CLASSIFIER_DOC_STRATEGY_DEFAULT.
CLASSIFIER_DOC_CATEGORIES: str = os.getenv(
    "CLASSIFIER_DOC_CATEGORIES",
    "textbook,reference_handbook,journal_article,conference_paper,"
    "technical_report,standard_specification,datasheet,patent",
)
CLASSIFIER_DOC_MODALITY_DIGITAL_THRESHOLD: float = float(
    os.getenv("CLASSIFIER_DOC_MODALITY_DIGITAL_THRESHOLD", "0.85")
)
CLASSIFIER_DOC_MODALITY_SCANNED_THRESHOLD: float = float(
    os.getenv("CLASSIFIER_DOC_MODALITY_SCANNED_THRESHOLD", "0.85")
)
CLASSIFIER_DOC_CATEGORY_THRESHOLD: float = float(
    os.getenv("CLASSIFIER_DOC_CATEGORY_THRESHOLD", "0.50")
)
CLASSIFIER_DOC_CATEGORY_MARGIN: float = float(os.getenv("CLASSIFIER_DOC_CATEGORY_MARGIN", "0.10"))
CLASSIFIER_DOC_LANGUAGE_MIN_CONFIDENCE: float = float(
    os.getenv("CLASSIFIER_DOC_LANGUAGE_MIN_CONFIDENCE", "0.10")
)
CLASSIFIER_DOC_LAYOUT_MODERATE_DENSITY: float = float(
    os.getenv("CLASSIFIER_DOC_LAYOUT_MODERATE_DENSITY", "0.15")
)
CLASSIFIER_DOC_LAYOUT_COMPLEX_DENSITY: float = float(
    os.getenv("CLASSIFIER_DOC_LAYOUT_COMPLEX_DENSITY", "0.35")
)
CLASSIFIER_DOC_LAYOUT_COLUMN_THRESHOLD: int = int(
    os.getenv("CLASSIFIER_DOC_LAYOUT_COLUMN_THRESHOLD", "2")
)
CLASSIFIER_DOC_PAGE_SAMPLE_STRATEGY: str = os.getenv(
    "CLASSIFIER_DOC_PAGE_SAMPLE_STRATEGY", "stratified"
)
CLASSIFIER_DOC_PAGE_SAMPLE_CAP: int = int(os.getenv("CLASSIFIER_DOC_PAGE_SAMPLE_CAP", "40"))
CLASSIFIER_DOC_LANGUAGE_BACKEND: str = os.getenv("CLASSIFIER_DOC_LANGUAGE_BACKEND", "default")
CLASSIFIER_DOC_ANALYZER_TIMEOUT: int = int(os.getenv("CLASSIFIER_DOC_ANALYZER_TIMEOUT", "30"))
CLASSIFIER_DOC_STRATEGY_MAP: str = os.getenv("CLASSIFIER_DOC_STRATEGY_MAP", "{}")
CLASSIFIER_DOC_STRATEGY_DEFAULT: str = os.getenv(
    "CLASSIFIER_DOC_STRATEGY_DEFAULT", "hybrid_per_page_routed"
)

# PDF preprocessing stage (feature 004). Conditional, passthrough-biased defaults: clean
# digital pages are passed through untouched; degraded scanned/hybrid pages are corrected.
# All thresholds are tunable and calibrated against the benchmark set. OPERATIONS is a
# comma-separated allow-list of enabled operations; an empty value disables derivation.
KNOVEL_PREPROCESS_ENABLED: bool = os.getenv("KNOVEL_PREPROCESS_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_PREPROCESS_OPERATIONS: str = os.getenv(
    "KNOVEL_PREPROCESS_OPERATIONS",
    "rotation_correction,deskew,denoise,enhance,resolution_normalize,dimension_normalize",
)
KNOVEL_PREPROCESS_ROTATION_CONFIDENCE: float = float(
    os.getenv("KNOVEL_PREPROCESS_ROTATION_CONFIDENCE", "0.80")
)
KNOVEL_PREPROCESS_DESKEW_THRESHOLD: float = float(
    os.getenv("KNOVEL_PREPROCESS_DESKEW_THRESHOLD", "0.5")
)
KNOVEL_PREPROCESS_DESKEW_MAX_ANGLE: float = float(
    os.getenv("KNOVEL_PREPROCESS_DESKEW_MAX_ANGLE", "15.0")
)
KNOVEL_PREPROCESS_BLANK_COVERAGE: float = float(
    os.getenv("KNOVEL_PREPROCESS_BLANK_COVERAGE", "0.005")
)
KNOVEL_PREPROCESS_LOWQUALITY_THRESHOLD: float = float(
    os.getenv("KNOVEL_PREPROCESS_LOWQUALITY_THRESHOLD", "0.02")
)
KNOVEL_PREPROCESS_ENHANCE_THRESHOLD: float = float(
    os.getenv("KNOVEL_PREPROCESS_ENHANCE_THRESHOLD", "0.40")
)
KNOVEL_PREPROCESS_RESOLUTION_MIN: float = float(
    os.getenv("KNOVEL_PREPROCESS_RESOLUTION_MIN", "150")
)
KNOVEL_PREPROCESS_RESOLUTION_MAX: float = float(
    os.getenv("KNOVEL_PREPROCESS_RESOLUTION_MAX", "400")
)
KNOVEL_PREPROCESS_DIMENSION_MAX: int = int(os.getenv("KNOVEL_PREPROCESS_DIMENSION_MAX", "10000"))
KNOVEL_PREPROCESS_OCR_DETECT_THRESHOLD: float = float(
    os.getenv("KNOVEL_PREPROCESS_OCR_DETECT_THRESHOLD", "0.50")
)
KNOVEL_PREPROCESS_BACKEND: str = os.getenv("KNOVEL_PREPROCESS_BACKEND", "numpy")
KNOVEL_PREPROCESS_WORKDIR: str = os.getenv("KNOVEL_PREPROCESS_WORKDIR", "")
KNOVEL_PREPROCESS_PERSIST_ARTIFACTS: bool = os.getenv(
    "KNOVEL_PREPROCESS_PERSIST_ARTIFACTS", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_PREPROCESS_REUSE: bool = os.getenv("KNOVEL_PREPROCESS_REUSE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_PREPROCESS_WORKERS: int = int(
    os.getenv("KNOVEL_PREPROCESS_WORKERS", str(KNOVEL_MAX_WORKERS))
)

# Layout analysis stage (feature 005). Detects, types, locates, relates, and scores layout
# regions per page, producing a reusable Layout Context. The default ``heuristic`` backend uses
# PDF text/vector primitives (no new dependency); alternative detectors register via the
# LayoutDetector protocol. Every threshold/tolerance/band is tunable; nothing is hardcoded.
KNOVEL_LAYOUT_ENABLED: bool = os.getenv("KNOVEL_LAYOUT_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_LAYOUT_BACKEND: str = os.getenv("KNOVEL_LAYOUT_BACKEND", "heuristic")
KNOVEL_LAYOUT_MIN_CONFIDENCE: float = float(os.getenv("KNOVEL_LAYOUT_MIN_CONFIDENCE", "0.30"))
KNOVEL_LAYOUT_LOW_CONFIDENCE_POLICY: str = os.getenv(
    "KNOVEL_LAYOUT_LOW_CONFIDENCE_POLICY", "flag"
)  # flag | drop
KNOVEL_LAYOUT_OVERLAP_TOLERANCE: float = float(os.getenv("KNOVEL_LAYOUT_OVERLAP_TOLERANCE", "0.50"))
KNOVEL_LAYOUT_OVERLAP_POLICY: str = os.getenv(
    "KNOVEL_LAYOUT_OVERLAP_POLICY", "demote"
)  # merge | demote
KNOVEL_LAYOUT_DUPLICATE_TOLERANCE: float = float(
    os.getenv("KNOVEL_LAYOUT_DUPLICATE_TOLERANCE", "0.90")
)
KNOVEL_LAYOUT_CONTAINMENT_RATIO: float = float(os.getenv("KNOVEL_LAYOUT_CONTAINMENT_RATIO", "0.80"))
KNOVEL_LAYOUT_CAPTION_MAX_DISTANCE: float = float(
    os.getenv("KNOVEL_LAYOUT_CAPTION_MAX_DISTANCE", "40.0")
)
KNOVEL_LAYOUT_COLUMN_MIN_GUTTER: float = float(os.getenv("KNOVEL_LAYOUT_COLUMN_MIN_GUTTER", "18.0"))
KNOVEL_LAYOUT_COLUMN_MIN: int = int(os.getenv("KNOVEL_LAYOUT_COLUMN_MIN", "1"))
KNOVEL_LAYOUT_COLUMN_MAX: int = int(os.getenv("KNOVEL_LAYOUT_COLUMN_MAX", "3"))
KNOVEL_LAYOUT_HEADER_BAND: float = float(os.getenv("KNOVEL_LAYOUT_HEADER_BAND", "0.08"))
KNOVEL_LAYOUT_FOOTER_BAND: float = float(os.getenv("KNOVEL_LAYOUT_FOOTER_BAND", "0.08"))
KNOVEL_LAYOUT_OOB_POLICY: str = os.getenv("KNOVEL_LAYOUT_OOB_POLICY", "clamp")  # clamp | exclude
KNOVEL_LAYOUT_WORKERS: int = int(os.getenv("KNOVEL_LAYOUT_WORKERS", str(KNOVEL_MAX_WORKERS)))
KNOVEL_LAYOUT_PAGE_TIMEOUT: int = int(os.getenv("KNOVEL_LAYOUT_PAGE_TIMEOUT", "60"))
KNOVEL_LAYOUT_REUSE: bool = os.getenv("KNOVEL_LAYOUT_REUSE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_LAYOUT_WORKDIR: str = os.getenv("KNOVEL_LAYOUT_WORKDIR", "")
KNOVEL_LAYOUT_VISUALIZE: bool = os.getenv("KNOVEL_LAYOUT_VISUALIZE", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Reading order detection stage (feature 006). Consumes the Layout Context and produces the
# authoritative reading sequence, hierarchy, and cross-reference associations. The default
# ``geometric`` strategy orders purely over layout regions (no new dependency); alternative
# strategies register via the ReadingOrderStrategy protocol. Every threshold/policy is tunable.
KNOVEL_READING_ORDER_ENABLED: bool = os.getenv("KNOVEL_READING_ORDER_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_READING_ORDER_STRATEGY: str = os.getenv(
    "KNOVEL_READING_ORDER_STRATEGY", "geometric"
)  # geometric | (future: model | hybrid)
KNOVEL_READING_ORDER_DIRECTION: str = os.getenv(
    "KNOVEL_READING_ORDER_DIRECTION", "auto"
)  # auto | ltr | rtl
KNOVEL_READING_ORDER_FLOAT_POLICY: str = os.getenv(
    "KNOVEL_READING_ORDER_FLOAT_POLICY", "column_break"
)  # column_break | anchor | page_end
KNOVEL_READING_ORDER_CAPTION_PLACEMENT: str = os.getenv(
    "KNOVEL_READING_ORDER_CAPTION_PLACEMENT", "after_target"
)  # after_target | before_target
KNOVEL_READING_ORDER_SIDEBAR_POLICY: str = os.getenv(
    "KNOVEL_READING_ORDER_SIDEBAR_POLICY", "page_end"
)  # page_end | column_anchor
KNOVEL_READING_ORDER_FOOTNOTE_POLICY: str = os.getenv(
    "KNOVEL_READING_ORDER_FOOTNOTE_POLICY", "page_end"
)  # page_end | inline_anchor
KNOVEL_READING_ORDER_ENDNOTE_POLICY: str = os.getenv(
    "KNOVEL_READING_ORDER_ENDNOTE_POLICY", "document_end"
)  # document_end | section_end
KNOVEL_READING_ORDER_HEADER_FOOTER_FLOW: str = os.getenv(
    "KNOVEL_READING_ORDER_HEADER_FOOTER_FLOW", "exclude"
)  # exclude | include
KNOVEL_READING_ORDER_CAPTION_MAX_DISTANCE: float = float(
    os.getenv("KNOVEL_READING_ORDER_CAPTION_MAX_DISTANCE", "40.0")
)
KNOVEL_READING_ORDER_EQUATION_NUMBER_MAX_DISTANCE: float = float(
    os.getenv("KNOVEL_READING_ORDER_EQUATION_NUMBER_MAX_DISTANCE", "60.0")
)
KNOVEL_READING_ORDER_FOOTNOTE_MARKER_MAX_DISTANCE: float = float(
    os.getenv("KNOVEL_READING_ORDER_FOOTNOTE_MARKER_MAX_DISTANCE", "400.0")
)
KNOVEL_READING_ORDER_CROSS_PAGE_ENABLED: bool = os.getenv(
    "KNOVEL_READING_ORDER_CROSS_PAGE_ENABLED", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_READING_ORDER_MIN_CONFIDENCE: float = float(
    os.getenv("KNOVEL_READING_ORDER_MIN_CONFIDENCE", "0.30")
)
KNOVEL_READING_ORDER_LOW_CONFIDENCE_POLICY: str = os.getenv(
    "KNOVEL_READING_ORDER_LOW_CONFIDENCE_POLICY", "flag"
)  # flag | drop
KNOVEL_READING_ORDER_WORKERS: int = int(
    os.getenv("KNOVEL_READING_ORDER_WORKERS", str(KNOVEL_MAX_WORKERS))
)
KNOVEL_READING_ORDER_PAGE_TIMEOUT: int = int(os.getenv("KNOVEL_READING_ORDER_PAGE_TIMEOUT", "60"))
KNOVEL_READING_ORDER_REUSE: bool = os.getenv("KNOVEL_READING_ORDER_REUSE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_READING_ORDER_WORKDIR: str = os.getenv("KNOVEL_READING_ORDER_WORKDIR", "")
KNOVEL_READING_ORDER_VISUALIZE: bool = os.getenv(
    "KNOVEL_READING_ORDER_VISUALIZE", "false"
).lower() in ("1", "true", "yes", "on")

# Text extraction stage (feature 007). Consumes the Reading Order Context and extracts the textual
# content of every text-bearing region in reading order — native PDF text, OCR for scanned regions,
# hybrid per-region selection — normalizes it, assigns semantic text roles, and records provenance
# and confidence. Engines are interchangeable behind a common interface; every policy is tunable.
KNOVEL_TEXT_ENABLED: bool = os.getenv("KNOVEL_TEXT_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_TEXT_NATIVE_ENGINE: str = os.getenv("KNOVEL_TEXT_NATIVE_ENGINE", "pdf_backend")
KNOVEL_TEXT_OCR_ENGINE: str = os.getenv("KNOVEL_TEXT_OCR_ENGINE", "paddleocr")
KNOVEL_TEXT_OCR_LANGUAGES: str = os.getenv("KNOVEL_TEXT_OCR_LANGUAGES", "auto")
KNOVEL_TEXT_STRATEGY: str = os.getenv("KNOVEL_TEXT_STRATEGY", "auto")
KNOVEL_TEXT_FALLBACK_ENABLED: bool = os.getenv("KNOVEL_TEXT_FALLBACK_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_TEXT_NATIVE_MIN_PRINTABLE: float = float(
    os.getenv("KNOVEL_TEXT_NATIVE_MIN_PRINTABLE", "0.60")
)
KNOVEL_TEXT_MIN_CONFIDENCE: float = float(os.getenv("KNOVEL_TEXT_MIN_CONFIDENCE", "0.50"))
KNOVEL_TEXT_LOW_CONFIDENCE_POLICY: str = os.getenv("KNOVEL_TEXT_LOW_CONFIDENCE_POLICY", "flag")
KNOVEL_TEXT_UNICODE_FORM: str = os.getenv("KNOVEL_TEXT_UNICODE_FORM", "NFC")
KNOVEL_TEXT_NORMALIZE_WHITESPACE: bool = os.getenv(
    "KNOVEL_TEXT_NORMALIZE_WHITESPACE", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_TEXT_MERGE_SOFT_BREAKS: bool = os.getenv(
    "KNOVEL_TEXT_MERGE_SOFT_BREAKS", "true"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_TEXT_DEHYPHENATE: bool = os.getenv("KNOVEL_TEXT_DEHYPHENATE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_TEXT_DEHYPHENATE_DICT: bool = os.getenv("KNOVEL_TEXT_DEHYPHENATE_DICT", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_TEXT_MAX_BLOCK_CHARS: int = int(os.getenv("KNOVEL_TEXT_MAX_BLOCK_CHARS", "0"))
KNOVEL_TEXT_WORKERS: int = int(os.getenv("KNOVEL_TEXT_WORKERS", str(KNOVEL_MAX_WORKERS)))
KNOVEL_TEXT_PAGE_TIMEOUT: int = int(os.getenv("KNOVEL_TEXT_PAGE_TIMEOUT", "120"))
KNOVEL_TEXT_REUSE: bool = os.getenv("KNOVEL_TEXT_REUSE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_TEXT_WORKDIR: str = os.getenv("KNOVEL_TEXT_WORKDIR", "")
KNOVEL_TEXT_DEBUG_DUMP: bool = os.getenv("KNOVEL_TEXT_DEBUG_DUMP", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Equation extraction stage (feature 008). Detects equation regions (display from layout, inline within
# text), classifies each into a content category, selects a configuration-driven recognition provider,
# recognizes structured representations (plain text/LaTeX/MathML/structured form), and preserves
# numbering, reading order, hierarchy, relationships, provenance, and confidence. Providers are
# interchangeable behind a common interface and import-guarded (graceful degradation when absent).
KNOVEL_EQUATION_ENABLED: bool = os.getenv("KNOVEL_EQUATION_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# KNOVEL_EQUATION_VL_MODEL defaults to OLLAMA_VL_MODEL when unset (see provider aliases above).
# Set KNOVEL_OLLAMA_BASE_URL (shared with the LLM correction stage) to point at your
# Ollama or vLLM server, e.g. http://localhost:11434 (Ollama) or http://localhost:8000 (vLLM).
# KNOVEL_EQUATION_VL_MODEL, KNOVEL_EQUATION_VL_TIMEOUT, KNOVEL_EQUATION_VL_MAX_TOKENS, and
# KNOVEL_EQUATION_PROVIDER_MAP are defined in the "Provider aliases" section above.
KNOVEL_EQUATION_INLINE_ENABLED: bool = os.getenv(
    "KNOVEL_EQUATION_INLINE_ENABLED", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_EQUATION_CLASSIFICATION_MIN_CONFIDENCE: float = float(
    os.getenv("KNOVEL_EQUATION_CLASSIFICATION_MIN_CONFIDENCE", "0.50")
)
KNOVEL_EQUATION_RECOGNITION_MIN_CONFIDENCE: float = float(
    os.getenv("KNOVEL_EQUATION_RECOGNITION_MIN_CONFIDENCE", "0.50")
)
# Confidence-gated recognition retry. When a first-pass VL recognition scores below the
# retry threshold (heuristic quality score — clipped left-hand side, bare right-hand-side
# fragment, two equations merged into one crop, empty output), the extractor re-crops the
# region with padding at a higher render zoom and re-runs the provider with a stricter
# prompt, keeping the higher-scoring output. All knobs are tunable; retry is a no-op when
# disabled, when no image is available (inline), or when the first pass already scores high.
KNOVEL_EQUATION_RETRY_ENABLED: bool = os.getenv(
    "KNOVEL_EQUATION_RETRY_ENABLED", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_EQUATION_RECOGNITION_RETRY_THRESHOLD: float = float(
    os.getenv("KNOVEL_EQUATION_RECOGNITION_RETRY_THRESHOLD", "0.60")
)
# Symmetric bbox expansion (fraction of region width/height) applied to the retry crop so a
# left-hand-side variable clipped by a tight layout box is recovered.
KNOVEL_EQUATION_CROP_PAD_FRAC: float = float(
    os.getenv("KNOVEL_EQUATION_CROP_PAD_FRAC", "0.15")
)
# Directional retry-crop padding. Left defaults slightly larger than the base symmetric pad because
# clipped left-hand-side variables are the dominant real-world failure in engineering equations.
KNOVEL_EQUATION_CROP_PAD_LEFT_FRAC: float = float(
    os.getenv("KNOVEL_EQUATION_CROP_PAD_LEFT_FRAC", str(KNOVEL_EQUATION_CROP_PAD_FRAC + 0.05))
)
KNOVEL_EQUATION_CROP_PAD_RIGHT_FRAC: float = float(
    os.getenv("KNOVEL_EQUATION_CROP_PAD_RIGHT_FRAC", str(KNOVEL_EQUATION_CROP_PAD_FRAC))
)
KNOVEL_EQUATION_CROP_PAD_TOP_FRAC: float = float(
    os.getenv("KNOVEL_EQUATION_CROP_PAD_TOP_FRAC", str(KNOVEL_EQUATION_CROP_PAD_FRAC))
)
KNOVEL_EQUATION_CROP_PAD_BOTTOM_FRAC: float = float(
    os.getenv("KNOVEL_EQUATION_CROP_PAD_BOTTOM_FRAC", str(KNOVEL_EQUATION_CROP_PAD_FRAC))
)
# Page render zoom used only for the retry crop (higher resolution than the default raster).
KNOVEL_EQUATION_RETRY_ZOOM: float = float(os.getenv("KNOVEL_EQUATION_RETRY_ZOOM", "3.0"))
# First-pass crop quality — controls the initial crop sent to the VL provider before any retry.
# Rendering fresh at a higher zoom (rather than reusing the preprocessing raster at ~144 DPI)
# makes subscripts and small symbols readable on the first attempt, reducing retry frequency.
# Left pad is larger than the symmetric pad because lhs_clipped is the dominant first-pass failure.
KNOVEL_EQUATION_FIRST_PASS_ZOOM: float = float(os.getenv("KNOVEL_EQUATION_FIRST_PASS_ZOOM", "3.0"))
KNOVEL_EQUATION_FIRST_PASS_PAD_FRAC: float = float(
    os.getenv("KNOVEL_EQUATION_FIRST_PASS_PAD_FRAC", "0.08")
)
KNOVEL_EQUATION_FIRST_PASS_PAD_LEFT_FRAC: float = float(
    os.getenv("KNOVEL_EQUATION_FIRST_PASS_PAD_LEFT_FRAC", "0.15")
)
# KNOVEL_EQUATION_LATEX_ENABLED, KNOVEL_EQUATION_MATHML_ENABLED, and
# KNOVEL_EQUATION_STRUCTURED_ENABLED are defined in the "Provider aliases" section above.
KNOVEL_EQUATION_VALIDATION_STRICTNESS: str = os.getenv(
    "KNOVEL_EQUATION_VALIDATION_STRICTNESS", "normal"
)
KNOVEL_EQUATION_WORKERS: int = int(os.getenv("KNOVEL_EQUATION_WORKERS", str(KNOVEL_MAX_WORKERS)))
KNOVEL_EQUATION_TIMEOUT: int = int(os.getenv("KNOVEL_EQUATION_TIMEOUT", "120"))
KNOVEL_EQUATION_REUSE: bool = os.getenv("KNOVEL_EQUATION_REUSE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_EQUATION_WORKDIR: str = os.getenv("KNOVEL_EQUATION_WORKDIR", "")
KNOVEL_EQUATION_DEBUG_DUMP: bool = os.getenv("KNOVEL_EQUATION_DEBUG_DUMP", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Table extraction stage (feature 009). Detects table regions (table-typed regions from layout),
# classifies each into a category, selects a configuration-driven extraction provider by category and
# page modality, extracts the complete cell matrix (row/column ordering, header rows/columns,
# merged-cell spans, explicit empty cells), merges multi-page continuations, and preserves numbering,
# caption/footnotes, reading order, hierarchy, relationships, provenance, and confidence. Providers are
# interchangeable behind a common ITableExtractor interface and import-guarded (graceful degradation).
KNOVEL_TABLE_ENABLED: bool = os.getenv("KNOVEL_TABLE_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_TABLE_DIGITAL_PROVIDER: str = os.getenv("KNOVEL_TABLE_DIGITAL_PROVIDER", "docling")
KNOVEL_TABLE_OCR_PROVIDER: str = os.getenv("KNOVEL_TABLE_OCR_PROVIDER", "img2table")
KNOVEL_TABLE_FALLBACK_PROVIDER: str = os.getenv("KNOVEL_TABLE_FALLBACK_PROVIDER", "pdfplumber")
# Optional category/modality->provider overrides, e.g. "financial=camelot,scanned=img2table".
KNOVEL_TABLE_PROVIDER_MAP: str = os.getenv("KNOVEL_TABLE_PROVIDER_MAP", "")
KNOVEL_TABLE_DETECTION_MIN_CONFIDENCE: float = float(
    os.getenv("KNOVEL_TABLE_DETECTION_MIN_CONFIDENCE", "0.50")
)
KNOVEL_TABLE_CLASSIFICATION_MIN_CONFIDENCE: float = float(
    os.getenv("KNOVEL_TABLE_CLASSIFICATION_MIN_CONFIDENCE", "0.50")
)
KNOVEL_TABLE_EXTRACTION_MIN_CONFIDENCE: float = float(
    os.getenv("KNOVEL_TABLE_EXTRACTION_MIN_CONFIDENCE", "0.50")
)
KNOVEL_TABLE_HEADER_DETECTION: bool = os.getenv(
    "KNOVEL_TABLE_HEADER_DETECTION", "true"
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_TABLE_CAPTION_DETECTION: bool = os.getenv(
    "KNOVEL_TABLE_CAPTION_DETECTION", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_TABLE_MULTIPAGE_MERGE: bool = os.getenv("KNOVEL_TABLE_MULTIPAGE_MERGE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Empty-cell handling policy: "preserve" keeps empty rows/columns; "collapse" drops fully-empty ones.
KNOVEL_TABLE_EMPTY_CELL_POLICY: str = os.getenv("KNOVEL_TABLE_EMPTY_CELL_POLICY", "preserve")
KNOVEL_TABLE_VALIDATION_STRICTNESS: str = os.getenv("KNOVEL_TABLE_VALIDATION_STRICTNESS", "normal")
KNOVEL_TABLE_WORKERS: int = int(os.getenv("KNOVEL_TABLE_WORKERS", str(KNOVEL_MAX_WORKERS)))
KNOVEL_TABLE_TIMEOUT: int = int(os.getenv("KNOVEL_TABLE_TIMEOUT", "120"))
KNOVEL_TABLE_REUSE: bool = os.getenv("KNOVEL_TABLE_REUSE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Visual content extraction stage (feature 010): detect figure/image-typed regions from layout, classify
# each into a visual category, select a configuration-driven provider by category/page-modality, crop and
# materialize each visual asset from the corrected page rasters (or an on-demand pdf_backend render) at
# preserved quality, capture image metadata, resolve composite figures, associate captions/numbers, and
# preserve reading order, hierarchy, relationships, provenance, and confidence. Providers are
# interchangeable behind a common IVisualContentExtractor interface and import-guarded.
KNOVEL_VISUAL_ENABLED: bool = os.getenv("KNOVEL_VISUAL_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_VISUAL_GENERAL_PROVIDER: str = os.getenv("KNOVEL_VISUAL_GENERAL_PROVIDER", "docling")
KNOVEL_VISUAL_GRAPH_PROVIDER: str = os.getenv("KNOVEL_VISUAL_GRAPH_PROVIDER", "opencv")
KNOVEL_VISUAL_CHEMICAL_PROVIDER: str = os.getenv("KNOVEL_VISUAL_CHEMICAL_PROVIDER", "chemical")
KNOVEL_VISUAL_DIAGRAM_PROVIDER: str = os.getenv("KNOVEL_VISUAL_DIAGRAM_PROVIDER", "generic")
KNOVEL_VISUAL_FALLBACK_PROVIDER: str = os.getenv("KNOVEL_VISUAL_FALLBACK_PROVIDER", "default")
# Optional category/modality->provider overrides, e.g. "map=generic,chemical_structure=chemical".
KNOVEL_VISUAL_PROVIDER_MAP: str = os.getenv("KNOVEL_VISUAL_PROVIDER_MAP", "")
# Chemical-structure recognition (MolScribe/DECIMER) is OFF by default; image extraction is unaffected.
KNOVEL_VISUAL_CHEM_RECOGNITION: bool = os.getenv(
    "KNOVEL_VISUAL_CHEM_RECOGNITION", "false"
).lower() in ("1", "true", "yes", "on")
KNOVEL_VISUAL_DETECTION_MIN_CONFIDENCE: float = float(
    os.getenv("KNOVEL_VISUAL_DETECTION_MIN_CONFIDENCE", "0.50")
)
KNOVEL_VISUAL_CLASSIFICATION_MIN_CONFIDENCE: float = float(
    os.getenv("KNOVEL_VISUAL_CLASSIFICATION_MIN_CONFIDENCE", "0.50")
)
KNOVEL_VISUAL_EXTRACTION_MIN_CONFIDENCE: float = float(
    os.getenv("KNOVEL_VISUAL_EXTRACTION_MIN_CONFIDENCE", "0.50")
)
KNOVEL_VISUAL_MIN_IMAGE_SIZE: int = int(os.getenv("KNOVEL_VISUAL_MIN_IMAGE_SIZE", "16"))
KNOVEL_VISUAL_MAX_IMAGE_SIZE: int = int(os.getenv("KNOVEL_VISUAL_MAX_IMAGE_SIZE", "10000"))
KNOVEL_VISUAL_RENDER_DPI: int = int(os.getenv("KNOVEL_VISUAL_RENDER_DPI", "200"))
KNOVEL_VISUAL_IMAGE_FORMAT: str = os.getenv("KNOVEL_VISUAL_IMAGE_FORMAT", "png")
KNOVEL_VISUAL_IMAGE_QUALITY: int = int(os.getenv("KNOVEL_VISUAL_IMAGE_QUALITY", "95"))
KNOVEL_VISUAL_CAPTION_DETECTION: bool = os.getenv(
    "KNOVEL_VISUAL_CAPTION_DETECTION", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_VISUAL_COMPOSITE_DETECTION: bool = os.getenv(
    "KNOVEL_VISUAL_COMPOSITE_DETECTION", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_VISUAL_VALIDATION_STRICTNESS: str = os.getenv(
    "KNOVEL_VISUAL_VALIDATION_STRICTNESS", "normal"
)
KNOVEL_VISUAL_WORKERS: int = int(os.getenv("KNOVEL_VISUAL_WORKERS", str(KNOVEL_MAX_WORKERS)))
KNOVEL_VISUAL_TIMEOUT: int = int(os.getenv("KNOVEL_VISUAL_TIMEOUT", "120"))
KNOVEL_VISUAL_REUSE: bool = os.getenv("KNOVEL_VISUAL_REUSE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Document metadata extraction stage (feature 011): consolidate document/bibliographic/structural/page/
# section/element/semantic/technical/processing/provenance metadata from ingestion (002), classification
# (003), and the layout/reading-order/text/equation/table/visual contexts into a reusable Metadata Context;
# normalize, validate, and attach provenance + confidence. Providers are interchangeable behind a common
# IMetadataExtractor interface and import-guarded; no new *required* dependency.
KNOVEL_METADATA_ENABLED: bool = os.getenv("KNOVEL_METADATA_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_METADATA_DOCUMENT_PROVIDER: str = os.getenv("KNOVEL_METADATA_DOCUMENT_PROVIDER", "default")
KNOVEL_METADATA_TECHNICAL_PROVIDER: str = os.getenv("KNOVEL_METADATA_TECHNICAL_PROVIDER", "default")
KNOVEL_METADATA_SEMANTIC_PROVIDER: str = os.getenv("KNOVEL_METADATA_SEMANTIC_PROVIDER", "default")
# Optional category->provider overrides, e.g. "technical=pypdf,semantic=default".
KNOVEL_METADATA_PROVIDER_MAP: str = os.getenv("KNOVEL_METADATA_PROVIDER_MAP", "")
KNOVEL_METADATA_NORMALIZE: bool = os.getenv("KNOVEL_METADATA_NORMALIZE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_METADATA_LANGUAGE_DETECTION: bool = os.getenv(
    "KNOVEL_METADATA_LANGUAGE_DETECTION", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_METADATA_KEYWORD_EXTRACTION: bool = os.getenv(
    "KNOVEL_METADATA_KEYWORD_EXTRACTION", "true"
).lower() in ("1", "true", "yes", "on")
# Default keyword provider is the deterministic, dependency-free "frequency" extractor; "yake" is opt-in
# (LGPL-3.0 optional extra) and degrades to "frequency" when the dependency is absent.
KNOVEL_METADATA_KEYWORD_PROVIDER: str = os.getenv("KNOVEL_METADATA_KEYWORD_PROVIDER", "frequency")
KNOVEL_METADATA_KEYWORD_MAX: int = int(os.getenv("KNOVEL_METADATA_KEYWORD_MAX", "20"))
KNOVEL_METADATA_TOPIC_EXTRACTION: bool = os.getenv(
    "KNOVEL_METADATA_TOPIC_EXTRACTION", "false"
).lower() in ("1", "true", "yes", "on")
KNOVEL_METADATA_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("KNOVEL_METADATA_CONFIDENCE_THRESHOLD", "0.50")
)
# Deterministic source-precedence order for conflict resolution (research D5).
KNOVEL_METADATA_SOURCE_PRECEDENCE: str = os.getenv(
    "KNOVEL_METADATA_SOURCE_PRECEDENCE", "pdf_info,text,classification"
)
KNOVEL_METADATA_VALIDATION_STRICTNESS: str = os.getenv(
    "KNOVEL_METADATA_VALIDATION_STRICTNESS", "normal"
)
KNOVEL_METADATA_WORKERS: int = int(os.getenv("KNOVEL_METADATA_WORKERS", str(KNOVEL_MAX_WORKERS)))
KNOVEL_METADATA_TIMEOUT: int = int(os.getenv("KNOVEL_METADATA_TIMEOUT", "120"))
KNOVEL_METADATA_REUSE: bool = os.getenv("KNOVEL_METADATA_REUSE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# --- Document relationship builder (feature 012) ---
# Links/validates the upstream contexts (layout/reading-order/text/equation/table/visual/metadata) into a
# Canonical Document Graph (Relationship Context). Re-derives nothing. The default graph backend is
# pure-Python stdlib; "networkx" is an optional, import-guarded alternative that degrades to "python".
KNOVEL_RELATIONSHIP_ENABLED: bool = os.getenv("KNOVEL_RELATIONSHIP_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
KNOVEL_RELATIONSHIP_PROVIDER: str = os.getenv("KNOVEL_RELATIONSHIP_PROVIDER", "python")
# Optional family->provider overrides, e.g. "graph=networkx".
KNOVEL_RELATIONSHIP_PROVIDER_MAP: str = os.getenv("KNOVEL_RELATIONSHIP_PROVIDER_MAP", "")
KNOVEL_RELATIONSHIP_CROSS_REFERENCE: bool = os.getenv(
    "KNOVEL_RELATIONSHIP_CROSS_REFERENCE", "true"
).lower() in ("1", "true", "yes", "on")
# Optional override of the cross-reference regex set (CSV of kind=pattern); empty = built-in patterns.
KNOVEL_RELATIONSHIP_REFERENCE_PATTERNS: str = os.getenv(
    "KNOVEL_RELATIONSHIP_REFERENCE_PATTERNS", ""
)
KNOVEL_RELATIONSHIP_CAPTION_MATCHING: bool = os.getenv(
    "KNOVEL_RELATIONSHIP_CAPTION_MATCHING", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_RELATIONSHIP_FOOTNOTE_MATCHING: bool = os.getenv(
    "KNOVEL_RELATIONSHIP_FOOTNOTE_MATCHING", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_RELATIONSHIP_CITATION_MATCHING: bool = os.getenv(
    "KNOVEL_RELATIONSHIP_CITATION_MATCHING", "true"
).lower() in ("1", "true", "yes", "on")
KNOVEL_RELATIONSHIP_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("KNOVEL_RELATIONSHIP_CONFIDENCE_THRESHOLD", "0.50")
)
KNOVEL_RELATIONSHIP_VALIDATION_STRICTNESS: str = os.getenv(
    "KNOVEL_RELATIONSHIP_VALIDATION_STRICTNESS", "normal"
)
KNOVEL_RELATIONSHIP_WORKERS: int = int(
    os.getenv("KNOVEL_RELATIONSHIP_WORKERS", str(KNOVEL_MAX_WORKERS))
)
KNOVEL_RELATIONSHIP_TIMEOUT: int = int(os.getenv("KNOVEL_RELATIONSHIP_TIMEOUT", "120"))
KNOVEL_RELATIONSHIP_REUSE: bool = os.getenv("KNOVEL_RELATIONSHIP_REUSE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# ---------------------------------------------------------------------------
# Shared truthy-string set (used by export, validation, and orch sections)
# ---------------------------------------------------------------------------

_TRUE_SET = ("1", "true", "yes", "on")

# Canonical Document Serialization & Export (feature 013). Drives the terminal
# serialization stage that emits the authoritative package (JSON master + HTML +
# normalized CSV + manifest) from the fully assembled CanonicalDocument.
KNOVEL_EXPORT_ENABLED: bool = os.getenv("KNOVEL_EXPORT_ENABLED", "true").lower() in _TRUE_SET
KNOVEL_EXPORT_FORMATS: str = os.getenv("KNOVEL_EXPORT_FORMATS", "json,html,csv")
KNOVEL_EXPORT_PRETTY: bool = os.getenv("KNOVEL_EXPORT_PRETTY", "true").lower() in _TRUE_SET
KNOVEL_EXPORT_COMPRESSION: str = os.getenv("KNOVEL_EXPORT_COMPRESSION", "none")
KNOVEL_EXPORT_ENCODING: str = os.getenv("KNOVEL_EXPORT_ENCODING", "utf-8")
KNOVEL_EXPORT_DIR: str = os.getenv("KNOVEL_EXPORT_DIR", "")
KNOVEL_EXPORT_SCHEMA_VERSION: str = os.getenv("KNOVEL_EXPORT_SCHEMA_VERSION", "1.0.0")
KNOVEL_EXPORT_INCLUDE_RELATIONSHIPS: bool = (
    os.getenv("KNOVEL_EXPORT_INCLUDE_RELATIONSHIPS", "true").lower() in _TRUE_SET
)
KNOVEL_EXPORT_INCLUDE_METADATA: bool = (
    os.getenv("KNOVEL_EXPORT_INCLUDE_METADATA", "true").lower() in _TRUE_SET
)
KNOVEL_EXPORT_WORKERS: int = int(os.getenv("KNOVEL_EXPORT_WORKERS", "1"))
KNOVEL_EXPORT_STREAM_THRESHOLD_PAGES: int = int(
    os.getenv("KNOVEL_EXPORT_STREAM_THRESHOLD_PAGES", "2000")
)
KNOVEL_EXPORT_VALIDATION_STRICTNESS: str = os.getenv(
    "KNOVEL_EXPORT_VALIDATION_STRICTNESS", "lenient"
)
KNOVEL_EXPORT_PROVIDER: str = os.getenv("KNOVEL_EXPORT_PROVIDER", "default")
KNOVEL_EXPORT_REUSE: bool = os.getenv("KNOVEL_EXPORT_REUSE", "true").lower() in _TRUE_SET

# Validation, Quality Assurance & Metrics framework (feature 014). Inspects/measures/reports the
# fully assembled CanonicalDocument after the relationship builder and before serialization; never
# aborts the batch. All knobs have safe defaults; the default validator path is pure-Python.
KNOVEL_VALIDATION_ENABLED: bool = (
    os.getenv("KNOVEL_VALIDATION_ENABLED", "true").lower() in _TRUE_SET
)
KNOVEL_VALIDATION_STRICTNESS: str = os.getenv("KNOVEL_VALIDATION_STRICTNESS", "normal")
KNOVEL_VALIDATION_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("KNOVEL_VALIDATION_CONFIDENCE_THRESHOLD", "0.50")
)
KNOVEL_VALIDATION_IOU_THRESHOLD: float = float(os.getenv("KNOVEL_VALIDATION_IOU_THRESHOLD", "0.50"))
# Comma-separated rule/validator categories to disable (empty = all enabled).
KNOVEL_VALIDATION_DISABLED_RULES: str = os.getenv("KNOVEL_VALIDATION_DISABLED_RULES", "")
KNOVEL_VALIDATION_BENCHMARK_MODE: bool = (
    os.getenv("KNOVEL_VALIDATION_BENCHMARK_MODE", "false").lower() in _TRUE_SET
)
KNOVEL_VALIDATION_PARALLEL: bool = (
    os.getenv("KNOVEL_VALIDATION_PARALLEL", "false").lower() in _TRUE_SET
)
KNOVEL_VALIDATION_WORKERS: int = int(
    os.getenv("KNOVEL_VALIDATION_WORKERS", str(KNOVEL_MAX_WORKERS))
)
KNOVEL_VALIDATION_REPORT_FORMATS: str = os.getenv("KNOVEL_VALIDATION_REPORT_FORMATS", "json,csv")
KNOVEL_VALIDATION_REPORT_DIR: str = os.getenv("KNOVEL_VALIDATION_REPORT_DIR", "")
KNOVEL_VALIDATION_REFERENCE_DIR: str = os.getenv("KNOVEL_VALIDATION_REFERENCE_DIR", "")
KNOVEL_VALIDATION_PROVIDER: str = os.getenv("KNOVEL_VALIDATION_PROVIDER", "python")
KNOVEL_VALIDATION_SCHEMA_VERSION: str = os.getenv("KNOVEL_VALIDATION_SCHEMA_VERSION", "1.0.0")
KNOVEL_VALIDATION_REUSE: bool = os.getenv("KNOVEL_VALIDATION_REUSE", "true").lower() in _TRUE_SET
KNOVEL_VALIDATION_TIMEOUT: int = int(os.getenv("KNOVEL_VALIDATION_TIMEOUT", "120"))

# Pipeline Orchestration & Configuration (feature 015). Formalizes the orchestration core that wires
# stages 002-014: declared execution plan, layered config + snapshot, provider/plugin registry, run-level
# checkpoint/resume/restart, retry policy, document/batch parallelism, metrics, and the report family.
# Defaults preserve today's behavior (DOC_WORKERS=1 => the existing per-page pooling owns the cores;
# RETRY_MAX_ATTEMPTS=1 => no retry; RESUME/INCREMENTAL off). All knobs are tunable; nothing is hardcoded.
KNOVEL_ORCH_CONFIG: str = os.getenv("KNOVEL_ORCH_CONFIG", "")
KNOVEL_ORCH_PLAN: str = os.getenv("KNOVEL_ORCH_PLAN", "full")
KNOVEL_ORCH_DISABLED_STAGES: str = os.getenv("KNOVEL_ORCH_DISABLED_STAGES", "")
KNOVEL_ORCH_DOC_WORKERS: int = int(os.getenv("KNOVEL_ORCH_DOC_WORKERS", "1"))
KNOVEL_ORCH_BATCH_WORKERS: int = int(
    os.getenv("KNOVEL_ORCH_BATCH_WORKERS", str(KNOVEL_ORCH_DOC_WORKERS))
)
KNOVEL_ORCH_MEMORY_LIMIT_MB: int = int(os.getenv("KNOVEL_ORCH_MEMORY_LIMIT_MB", "0"))
KNOVEL_ORCH_RETRY_MAX_ATTEMPTS: int = int(os.getenv("KNOVEL_ORCH_RETRY_MAX_ATTEMPTS", "1"))
KNOVEL_ORCH_RETRY_BACKOFF: float = float(os.getenv("KNOVEL_ORCH_RETRY_BACKOFF", "2.0"))
KNOVEL_ORCH_RETRY_BASE_DELAY_S: float = float(os.getenv("KNOVEL_ORCH_RETRY_BASE_DELAY_S", "1.0"))
KNOVEL_ORCH_RETRY_MAX_DELAY_S: float = float(os.getenv("KNOVEL_ORCH_RETRY_MAX_DELAY_S", "30.0"))
KNOVEL_ORCH_CRITICAL_STAGES: str = os.getenv(
    "KNOVEL_ORCH_CRITICAL_STAGES", "ingestion,classification"
)
KNOVEL_ORCH_CHECKPOINT_ENABLED: bool = (
    os.getenv("KNOVEL_ORCH_CHECKPOINT_ENABLED", "true").lower() in _TRUE_SET
)
KNOVEL_ORCH_RESUME: bool = os.getenv("KNOVEL_ORCH_RESUME", "false").lower() in _TRUE_SET
KNOVEL_ORCH_INCREMENTAL: bool = os.getenv("KNOVEL_ORCH_INCREMENTAL", "false").lower() in _TRUE_SET
KNOVEL_ORCH_BENCHMARK_MODE: bool = (
    os.getenv("KNOVEL_ORCH_BENCHMARK_MODE", "false").lower() in _TRUE_SET
)
KNOVEL_ORCH_VALIDATION_MODE: bool = (
    os.getenv("KNOVEL_ORCH_VALIDATION_MODE", "false").lower() in _TRUE_SET
)
KNOVEL_ORCH_REPORT_DIR: str = os.getenv("KNOVEL_ORCH_REPORT_DIR", "")
KNOVEL_ORCH_REPORT_FORMATS: str = os.getenv("KNOVEL_ORCH_REPORT_FORMATS", "json,csv")
KNOVEL_ORCH_CONFIG_VALIDATOR: str = os.getenv("KNOVEL_ORCH_CONFIG_VALIDATOR", "python")
KNOVEL_ORCH_METRICS_BACKEND: str = os.getenv("KNOVEL_ORCH_METRICS_BACKEND", "auto")
KNOVEL_ORCH_SCHEMA_VERSION: str = os.getenv("KNOVEL_ORCH_SCHEMA_VERSION", "1.0.0")


def get_config_summary() -> dict[str, object]:
    """Return the effective configuration for startup logging."""

    return {
        "KNOVEL_INPUT_DIR": KNOVEL_INPUT_DIR,
        "KNOVEL_OUTPUT_DIR": KNOVEL_OUTPUT_DIR,
        "KNOVEL_CACHE_DIR": KNOVEL_CACHE_DIR,
        "KNOVEL_LLM_BACKEND": KNOVEL_LLM_BACKEND,
        "KNOVEL_OLLAMA_BASE_URL": KNOVEL_OLLAMA_BASE_URL,
        "KNOVEL_OLLAMA_FAST_MODEL": KNOVEL_OLLAMA_FAST_MODEL,
        "KNOVEL_OLLAMA_COMPLEX_MODEL": KNOVEL_OLLAMA_COMPLEX_MODEL,
        "KNOVEL_PORTKEY_API_KEY": KNOVEL_PORTKEY_API_KEY,
        "KNOVEL_PORTKEY_BASE_URL": KNOVEL_PORTKEY_BASE_URL,
        "KNOVEL_PORTKEY_MODEL": KNOVEL_PORTKEY_MODEL,
        "KNOVEL_MAX_WORKERS": KNOVEL_MAX_WORKERS,
        "KNOVEL_LLM_MAX_WORKERS": KNOVEL_LLM_MAX_WORKERS,
        "KNOVEL_LLM_TIMEOUT": KNOVEL_LLM_TIMEOUT,
        "KNOVEL_LLM_MAX_RETRIES": KNOVEL_LLM_MAX_RETRIES,
        "CLASSIFIER_WORD_COUNT_THRESHOLD": CLASSIFIER_WORD_COUNT_THRESHOLD,
        "CLASSIFIER_IMAGE_COVERAGE_THRESHOLD": CLASSIFIER_IMAGE_COVERAGE_THRESHOLD,
        "CLASSIFIER_RENDER_SIMILARITY_THRESHOLD": CLASSIFIER_RENDER_SIMILARITY_THRESHOLD,
        "CLASSIFIER_AMBIGUOUS_LOWER": CLASSIFIER_AMBIGUOUS_LOWER,
        "KNOVEL_INGESTION_DUPLICATE_POLICY": KNOVEL_INGESTION_DUPLICATE_POLICY,
        "KNOVEL_INGESTION_MANIFEST_DIR": KNOVEL_INGESTION_MANIFEST_DIR,
        "KNOVEL_INGESTION_INDEX_PATH": KNOVEL_INGESTION_INDEX_PATH,
        "KNOVEL_INGESTION_MAX_FILE_MB": KNOVEL_INGESTION_MAX_FILE_MB,
        "CLASSIFIER_DOC_CATEGORIES": CLASSIFIER_DOC_CATEGORIES,
        "CLASSIFIER_DOC_MODALITY_DIGITAL_THRESHOLD": CLASSIFIER_DOC_MODALITY_DIGITAL_THRESHOLD,
        "CLASSIFIER_DOC_MODALITY_SCANNED_THRESHOLD": CLASSIFIER_DOC_MODALITY_SCANNED_THRESHOLD,
        "CLASSIFIER_DOC_CATEGORY_THRESHOLD": CLASSIFIER_DOC_CATEGORY_THRESHOLD,
        "CLASSIFIER_DOC_CATEGORY_MARGIN": CLASSIFIER_DOC_CATEGORY_MARGIN,
        "CLASSIFIER_DOC_LANGUAGE_MIN_CONFIDENCE": CLASSIFIER_DOC_LANGUAGE_MIN_CONFIDENCE,
        "CLASSIFIER_DOC_LAYOUT_MODERATE_DENSITY": CLASSIFIER_DOC_LAYOUT_MODERATE_DENSITY,
        "CLASSIFIER_DOC_LAYOUT_COMPLEX_DENSITY": CLASSIFIER_DOC_LAYOUT_COMPLEX_DENSITY,
        "CLASSIFIER_DOC_LAYOUT_COLUMN_THRESHOLD": CLASSIFIER_DOC_LAYOUT_COLUMN_THRESHOLD,
        "CLASSIFIER_DOC_PAGE_SAMPLE_STRATEGY": CLASSIFIER_DOC_PAGE_SAMPLE_STRATEGY,
        "CLASSIFIER_DOC_PAGE_SAMPLE_CAP": CLASSIFIER_DOC_PAGE_SAMPLE_CAP,
        "CLASSIFIER_DOC_LANGUAGE_BACKEND": CLASSIFIER_DOC_LANGUAGE_BACKEND,
        "CLASSIFIER_DOC_ANALYZER_TIMEOUT": CLASSIFIER_DOC_ANALYZER_TIMEOUT,
        "CLASSIFIER_DOC_STRATEGY_MAP": CLASSIFIER_DOC_STRATEGY_MAP,
        "CLASSIFIER_DOC_STRATEGY_DEFAULT": CLASSIFIER_DOC_STRATEGY_DEFAULT,
        "KNOVEL_PREPROCESS_ENABLED": KNOVEL_PREPROCESS_ENABLED,
        "KNOVEL_PREPROCESS_OPERATIONS": KNOVEL_PREPROCESS_OPERATIONS,
        "KNOVEL_PREPROCESS_ROTATION_CONFIDENCE": KNOVEL_PREPROCESS_ROTATION_CONFIDENCE,
        "KNOVEL_PREPROCESS_DESKEW_THRESHOLD": KNOVEL_PREPROCESS_DESKEW_THRESHOLD,
        "KNOVEL_PREPROCESS_DESKEW_MAX_ANGLE": KNOVEL_PREPROCESS_DESKEW_MAX_ANGLE,
        "KNOVEL_PREPROCESS_BLANK_COVERAGE": KNOVEL_PREPROCESS_BLANK_COVERAGE,
        "KNOVEL_PREPROCESS_LOWQUALITY_THRESHOLD": KNOVEL_PREPROCESS_LOWQUALITY_THRESHOLD,
        "KNOVEL_PREPROCESS_ENHANCE_THRESHOLD": KNOVEL_PREPROCESS_ENHANCE_THRESHOLD,
        "KNOVEL_PREPROCESS_RESOLUTION_MIN": KNOVEL_PREPROCESS_RESOLUTION_MIN,
        "KNOVEL_PREPROCESS_RESOLUTION_MAX": KNOVEL_PREPROCESS_RESOLUTION_MAX,
        "KNOVEL_PREPROCESS_DIMENSION_MAX": KNOVEL_PREPROCESS_DIMENSION_MAX,
        "KNOVEL_PREPROCESS_OCR_DETECT_THRESHOLD": KNOVEL_PREPROCESS_OCR_DETECT_THRESHOLD,
        "KNOVEL_PREPROCESS_BACKEND": KNOVEL_PREPROCESS_BACKEND,
        "KNOVEL_PREPROCESS_WORKDIR": KNOVEL_PREPROCESS_WORKDIR,
        "KNOVEL_PREPROCESS_PERSIST_ARTIFACTS": KNOVEL_PREPROCESS_PERSIST_ARTIFACTS,
        "KNOVEL_PREPROCESS_REUSE": KNOVEL_PREPROCESS_REUSE,
        "KNOVEL_PREPROCESS_WORKERS": KNOVEL_PREPROCESS_WORKERS,
        "KNOVEL_LAYOUT_ENABLED": KNOVEL_LAYOUT_ENABLED,
        "KNOVEL_LAYOUT_BACKEND": KNOVEL_LAYOUT_BACKEND,
        "KNOVEL_LAYOUT_MIN_CONFIDENCE": KNOVEL_LAYOUT_MIN_CONFIDENCE,
        "KNOVEL_LAYOUT_LOW_CONFIDENCE_POLICY": KNOVEL_LAYOUT_LOW_CONFIDENCE_POLICY,
        "KNOVEL_LAYOUT_OVERLAP_TOLERANCE": KNOVEL_LAYOUT_OVERLAP_TOLERANCE,
        "KNOVEL_LAYOUT_OVERLAP_POLICY": KNOVEL_LAYOUT_OVERLAP_POLICY,
        "KNOVEL_LAYOUT_DUPLICATE_TOLERANCE": KNOVEL_LAYOUT_DUPLICATE_TOLERANCE,
        "KNOVEL_LAYOUT_CONTAINMENT_RATIO": KNOVEL_LAYOUT_CONTAINMENT_RATIO,
        "KNOVEL_LAYOUT_CAPTION_MAX_DISTANCE": KNOVEL_LAYOUT_CAPTION_MAX_DISTANCE,
        "KNOVEL_LAYOUT_COLUMN_MIN_GUTTER": KNOVEL_LAYOUT_COLUMN_MIN_GUTTER,
        "KNOVEL_LAYOUT_COLUMN_MIN": KNOVEL_LAYOUT_COLUMN_MIN,
        "KNOVEL_LAYOUT_COLUMN_MAX": KNOVEL_LAYOUT_COLUMN_MAX,
        "KNOVEL_LAYOUT_HEADER_BAND": KNOVEL_LAYOUT_HEADER_BAND,
        "KNOVEL_LAYOUT_FOOTER_BAND": KNOVEL_LAYOUT_FOOTER_BAND,
        "KNOVEL_LAYOUT_OOB_POLICY": KNOVEL_LAYOUT_OOB_POLICY,
        "KNOVEL_LAYOUT_WORKERS": KNOVEL_LAYOUT_WORKERS,
        "KNOVEL_LAYOUT_PAGE_TIMEOUT": KNOVEL_LAYOUT_PAGE_TIMEOUT,
        "KNOVEL_LAYOUT_REUSE": KNOVEL_LAYOUT_REUSE,
        "KNOVEL_LAYOUT_WORKDIR": KNOVEL_LAYOUT_WORKDIR,
        "KNOVEL_LAYOUT_VISUALIZE": KNOVEL_LAYOUT_VISUALIZE,
        "KNOVEL_EXPORT_ENABLED": KNOVEL_EXPORT_ENABLED,
        "KNOVEL_EXPORT_FORMATS": KNOVEL_EXPORT_FORMATS,
        "KNOVEL_EXPORT_PRETTY": KNOVEL_EXPORT_PRETTY,
        "KNOVEL_EXPORT_COMPRESSION": KNOVEL_EXPORT_COMPRESSION,
        "KNOVEL_EXPORT_ENCODING": KNOVEL_EXPORT_ENCODING,
        "KNOVEL_EXPORT_DIR": KNOVEL_EXPORT_DIR,
        "KNOVEL_EXPORT_SCHEMA_VERSION": KNOVEL_EXPORT_SCHEMA_VERSION,
        "KNOVEL_EXPORT_INCLUDE_RELATIONSHIPS": KNOVEL_EXPORT_INCLUDE_RELATIONSHIPS,
        "KNOVEL_EXPORT_INCLUDE_METADATA": KNOVEL_EXPORT_INCLUDE_METADATA,
        "KNOVEL_EXPORT_WORKERS": KNOVEL_EXPORT_WORKERS,
        "KNOVEL_EXPORT_STREAM_THRESHOLD_PAGES": KNOVEL_EXPORT_STREAM_THRESHOLD_PAGES,
        "KNOVEL_EXPORT_VALIDATION_STRICTNESS": KNOVEL_EXPORT_VALIDATION_STRICTNESS,
        "KNOVEL_EXPORT_PROVIDER": KNOVEL_EXPORT_PROVIDER,
        "KNOVEL_EXPORT_REUSE": KNOVEL_EXPORT_REUSE,
        "KNOVEL_VALIDATION_ENABLED": KNOVEL_VALIDATION_ENABLED,
        "KNOVEL_VALIDATION_STRICTNESS": KNOVEL_VALIDATION_STRICTNESS,
        "KNOVEL_VALIDATION_CONFIDENCE_THRESHOLD": KNOVEL_VALIDATION_CONFIDENCE_THRESHOLD,
        "KNOVEL_VALIDATION_IOU_THRESHOLD": KNOVEL_VALIDATION_IOU_THRESHOLD,
        "KNOVEL_VALIDATION_DISABLED_RULES": KNOVEL_VALIDATION_DISABLED_RULES,
        "KNOVEL_VALIDATION_BENCHMARK_MODE": KNOVEL_VALIDATION_BENCHMARK_MODE,
        "KNOVEL_VALIDATION_PARALLEL": KNOVEL_VALIDATION_PARALLEL,
        "KNOVEL_VALIDATION_WORKERS": KNOVEL_VALIDATION_WORKERS,
        "KNOVEL_VALIDATION_REPORT_FORMATS": KNOVEL_VALIDATION_REPORT_FORMATS,
        "KNOVEL_VALIDATION_REPORT_DIR": KNOVEL_VALIDATION_REPORT_DIR,
        "KNOVEL_VALIDATION_REFERENCE_DIR": KNOVEL_VALIDATION_REFERENCE_DIR,
        "KNOVEL_VALIDATION_PROVIDER": KNOVEL_VALIDATION_PROVIDER,
        "KNOVEL_VALIDATION_SCHEMA_VERSION": KNOVEL_VALIDATION_SCHEMA_VERSION,
        "KNOVEL_VALIDATION_REUSE": KNOVEL_VALIDATION_REUSE,
        "KNOVEL_VALIDATION_TIMEOUT": KNOVEL_VALIDATION_TIMEOUT,
        "KNOVEL_ORCH_CONFIG": KNOVEL_ORCH_CONFIG,
        "KNOVEL_ORCH_PLAN": KNOVEL_ORCH_PLAN,
        "KNOVEL_ORCH_DISABLED_STAGES": KNOVEL_ORCH_DISABLED_STAGES,
        "KNOVEL_ORCH_DOC_WORKERS": KNOVEL_ORCH_DOC_WORKERS,
        "KNOVEL_ORCH_BATCH_WORKERS": KNOVEL_ORCH_BATCH_WORKERS,
        "KNOVEL_ORCH_MEMORY_LIMIT_MB": KNOVEL_ORCH_MEMORY_LIMIT_MB,
        "KNOVEL_ORCH_RETRY_MAX_ATTEMPTS": KNOVEL_ORCH_RETRY_MAX_ATTEMPTS,
        "KNOVEL_ORCH_RETRY_BACKOFF": KNOVEL_ORCH_RETRY_BACKOFF,
        "KNOVEL_ORCH_RETRY_BASE_DELAY_S": KNOVEL_ORCH_RETRY_BASE_DELAY_S,
        "KNOVEL_ORCH_RETRY_MAX_DELAY_S": KNOVEL_ORCH_RETRY_MAX_DELAY_S,
        "KNOVEL_ORCH_CRITICAL_STAGES": KNOVEL_ORCH_CRITICAL_STAGES,
        "KNOVEL_ORCH_CHECKPOINT_ENABLED": KNOVEL_ORCH_CHECKPOINT_ENABLED,
        "KNOVEL_ORCH_RESUME": KNOVEL_ORCH_RESUME,
        "KNOVEL_ORCH_INCREMENTAL": KNOVEL_ORCH_INCREMENTAL,
        "KNOVEL_ORCH_BENCHMARK_MODE": KNOVEL_ORCH_BENCHMARK_MODE,
        "KNOVEL_ORCH_VALIDATION_MODE": KNOVEL_ORCH_VALIDATION_MODE,
        "KNOVEL_ORCH_REPORT_DIR": KNOVEL_ORCH_REPORT_DIR,
        "KNOVEL_ORCH_REPORT_FORMATS": KNOVEL_ORCH_REPORT_FORMATS,
        "KNOVEL_ORCH_CONFIG_VALIDATOR": KNOVEL_ORCH_CONFIG_VALIDATOR,
        "KNOVEL_ORCH_METRICS_BACKEND": KNOVEL_ORCH_METRICS_BACKEND,
        "KNOVEL_ORCH_SCHEMA_VERSION": KNOVEL_ORCH_SCHEMA_VERSION,
    }
