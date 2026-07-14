"""OCR, LLM inference, and provider management — merged module.

The ML boundary: all calls to vision-language models, PaddleOCR, and related
quality-scoring utilities live here.

Merged sources
--------------
* pipeline/inference/exceptions.py     — InferenceError exception hierarchy
* pipeline/ocr_backend.py              — PaddleOCR wrapper (OCR_AVAILABLE, ocr_page, ocr_text)
* recognition_quality.py               — heuristic VLM output quality scoring
* pipeline/inference/prompts.py        — centralised prompt library
* pipeline/inference/client.py         — InferenceBackend protocol and OllamaBackend
* pipeline/inference/vision_service.py — VisionService (provider-agnostic VLM facade)
* providers.py                         — EquationProvider protocol and concrete implementations
* registry.py                          — provider registry and resolution
* selection.py                         — category → provider selection
* llm_judge.py                         — LLM-based quality judgement for LaTeX
* qwen.py                              — legacy Ollama /api/generate OCR path
"""

from __future__ import annotations

import abc
import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import requests
import structlog
from PIL import Image

from equation_extraction_pipeline.config import settings as config
from equation_extraction_pipeline.domain.models import (
    JudgeVerdict,
    OcrResult,
)

try:
    import httpx
    from httpx import ConnectError, HTTPStatusError, TimeoutException
except ImportError:  # pragma: no cover

    class _HttpxStub:  # type: ignore[no-redef]
        ConnectError = Exception
        TimeoutException = Exception
        HTTPStatusError = Exception

        @staticmethod
        def get(*_a, **_k):
            raise RuntimeError("httpx is not installed")

        @staticmethod
        def post(*_a, **_k):
            raise RuntimeError("httpx is not installed")

    httpx = _HttpxStub()  # type: ignore[assignment]
    ConnectError = Exception  # type: ignore[misc,assignment]
    TimeoutException = Exception  # type: ignore[misc,assignment]
    HTTPStatusError = Exception  # type: ignore[misc,assignment]

try:
    from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover

    class RetryError(Exception):  # type: ignore[no-redef]
        last_attempt = None

    def retry(**_kw):  # type: ignore[misc]
        def decorator(fn: Callable) -> Callable:
            return fn

        return decorator

    def stop_after_attempt(_n: int):  # type: ignore[misc]
        return _n

    def wait_exponential(**_kw):  # type: ignore[misc]
        return {}


try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - optional dependency
    PaddleOCR = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)

__all__ = [
    # exceptions
    "InferenceError",
    "InferenceUnavailableError",
    "InferenceTimeoutError",
    "InferenceResponseError",
    "ModelNotInstalledError",
    "MalformedResponseError",
    # ocr_backend
    "OCR_AVAILABLE",
    "ocr_page",
    "ocr_text",
    # recognition_quality
    "score_recognition",
    "MATH_CATEGORIES",
    "RETRY_QUALITY_NOTES",
    # prompts
    "EQUATION_SYSTEM_PROMPT",
    "EQUATION_IMAGE_PROMPTS",
    "EQUATION_IMAGE_PROMPTS_STRICT",
    "EQUATION_TEXT_PROMPTS",
    "EQUATION_VALIDATE_SYSTEM",
    "PAGE_EQUATIONS_SYSTEM",
    "PAGE_EQUATIONS_USER",
    "STRUCTURED_JSON_SYSTEM",
    "get_equation_image_prompt",
    "get_equation_text_prompt",
    "build_equation_validate_user",
    "build_structured_json_user",
    # client
    "InferenceBackend",
    "OllamaBackend",
    "create_backend",
    # vision_service
    "EquationResult",
    "ValidationResult",
    "PageEquationsResult",
    "VisionService",
    # providers
    "RecognitionResult",
    "EquationProvider",
    "QwenVLProvider",
    "GenericProvider",
    # registry
    "register_provider",
    "resolve_provider",
    "resolve_providers",
    "provider_identities",
    "close_providers",
    # selection
    "parse_provider_map",
    "select_provider",
    # llm_judge
    "judge_latex",
    # qwen (legacy)
    "recognize_equation",
    "recognize_with_retry",
]


# ============================================================
# SECTION 1: Merged from pipeline/inference/exceptions.py
# Inference-layer exception hierarchy
# ============================================================


class _PipelineError(Exception):
    """Base exception for Knovel pipeline failures (local alias)."""

    def __init__(self, message: str, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}


# Re-use the project's common exception base when available.
try:
    from equation_extraction_pipeline.common.exceptions import (
        PipelineError as _PipelineError,  # type: ignore[no-redef]
    )
except ImportError:
    pass  # use the local definition above


class InferenceError(_PipelineError):
    """Base class for all inference-layer failures.

    All concrete inference exceptions are subclasses of this, so a single
    ``except InferenceError`` will catch any backend failure.
    """


class InferenceUnavailableError(InferenceError):
    """The inference backend cannot be reached."""


class InferenceTimeoutError(InferenceError):
    """A request exceeded the configured per-request timeout."""


class InferenceResponseError(InferenceError):
    """The backend returned an unexpected or invalid HTTP response."""


class ModelNotInstalledError(InferenceError):
    """The requested model is not available on the backend.

    For Ollama: run ``ollama pull <model>``.
    """


class MalformedResponseError(InferenceError):
    """The response content could not be parsed as the expected format."""


# ============================================================
# SECTION 2: Merged from pipeline/ocr_backend.py
# PaddleOCR wrapper — permissively-licensed OCR backend
#
# PaddleOCR is a heavy optional dependency (Apache-2.0).
# Model load is lazy and cached per language.
# ============================================================

OCR_AVAILABLE: bool = PaddleOCR is not None

_DEFAULT_LANG = "en"
_ENGINES: dict[str, object] = {}
_OCR_RUNTIME_DISABLED = False


def _ocr_engine(lang: str = _DEFAULT_LANG):
    """Lazily build and cache a PaddleOCR engine for ``lang``.

    Returns ``None`` if OCR is unavailable or the engine cannot be constructed,
    in which case callers degrade to empty OCR output rather than aborting the run.
    """
    global _OCR_RUNTIME_DISABLED
    if PaddleOCR is None or _OCR_RUNTIME_DISABLED:  # pragma: no cover
        return None
    engine = _ENGINES.get(lang)
    if engine is None:
        try:
            engine = PaddleOCR(use_textline_orientation=True, lang=lang)
        except Exception as exc:  # pragma: no cover
            _OCR_RUNTIME_DISABLED = True
            logger.warning(
                "OCR disabled: PaddleOCR engine unavailable (%s). "
                "Scanned/hybrid regions will have no extracted text. "
                "Install 'paddlepaddle' to enable OCR.",
                exc,
            )
            return None
        _ENGINES[lang] = engine
    return engine


def _quad_to_bbox(quad) -> list[float]:
    xs = [float(point[0]) for point in quad]
    ys = [float(point[1]) for point in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def _parse_ocr_result(result) -> list[dict]:
    """Normalize a PaddleOCR 3.x predict() result into line dicts.

    3.x returns one dict-like ``OCRResult`` per image with parallel arrays
    (``rec_texts``/``rec_scores`` plus ``rec_polys`` or ``rec_boxes``). Falls
    back to the legacy 2.x nested-list shape for older installs.
    """

    lines: list[dict] = []
    for page in result or []:
        # Legacy 2.x: page is a list of ``[quad, (text, conf)]`` entries.
        if isinstance(page, list):
            for entry in page or []:
                quad, (text, confidence) = entry
                lines.append(
                    {
                        "text": str(text),
                        "bbox": _quad_to_bbox(quad),
                        "confidence": float(confidence),
                    }
                )
            continue
        # 3.x: dict-like result with parallel arrays.
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        polys = page.get("rec_polys")
        boxes = page.get("rec_boxes")
        for i, text in enumerate(texts):
            confidence = float(scores[i]) if i < len(scores) else 0.0
            if polys is not None and i < len(polys):
                bbox = _quad_to_bbox(polys[i])
            elif boxes is not None and i < len(boxes):
                box = boxes[i]
                bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
            lines.append({"text": str(text), "bbox": bbox, "confidence": confidence})
    return lines


def ocr_page(image, lang: str = _DEFAULT_LANG) -> list[dict]:
    """OCR a page image (numpy array) into text lines with bounding boxes and confidences."""
    if not OCR_AVAILABLE:
        return []
    engine = _ocr_engine(lang)
    if engine is None:
        return []
    try:
        raw = engine.predict(image)
    except Exception as exc:  # pragma: no cover
        logger.warning("OCR prediction failed (%s); returning no text for region.", exc)
        return []
    return _parse_ocr_result(raw)


def ocr_text(image, lang: str = _DEFAULT_LANG) -> str:
    """Return the concatenated OCR text for a page image (classification signal)."""
    return " ".join(line["text"] for line in ocr_page(image, lang) if line["text"])


# ============================================================
# SECTION 3: Merged from recognition_quality.py
# Heuristic recognition-quality scoring for VL equation output
#
# Derives a [0,1] quality score from the recognised representation to gate
# a stronger-prompt / padded higher-resolution re-crop retry and to make
# the low-confidence flagging reflect real fidelity.
# ============================================================

RETRY_QUALITY_NOTES: frozenset[str] = frozenset(
    {
        "quality:label_only",
        "quality:prose_contamination",
        "quality:prose_in_text_block",
        "quality:spaced_text",
        "quality:multiple_tags",
        "quality:multiple_equations",
    }
)

MATH_CATEGORIES: frozenset[str] = frozenset(
    {"mathematical_equation", "engineering_formula", "statistical_expression"}
)

_BASE_CONFIDENCE = 0.85

_TAG_RE = re.compile(r"\\tag\b")
_TAG_BLOCK_RE = re.compile(r"\\tag\s*\{[^}]*\}")
_REL_OP_RE = re.compile(
    r"(=|\\leq|\\geq|\\neq|\\approx|\\equiv|\\propto|\\sim|\\simeq|\\cong|\\subset|\\subseteq"
    r"|\\supset|\\supseteq|\\in|\\notin|\\rightarrow|\\to|\\Rightarrow|\\Leftrightarrow|[<>])"
)
_LHS_CLIPPED_RE = re.compile(r"^\s*(?:\\[a-zA-Z]+\s*)?=")
_SPACED_CHARS_RE = re.compile(r"\b([A-Za-z]\s){4,}[A-Za-z]\b")
_LABEL_ONLY_RE = re.compile(
    r"""^\s*
    (?:Eq(?:uation)?[.:]?\s*[\d]+(?:\.[\d]+){1,4}
    |\(\s*[\d]+(?:[-.][\d]+){1,4}\s*\)
    |\[\s*[\d]+(?:[-.][\d]+){1,4}\s*\]
    |\\left\s*\(\s*[\d]+(?:\s*[-–]\s*[\d]+){1,4}\s*\\right\s*\)
    )\s*$""",
    re.VERBOSE | re.IGNORECASE,
)
_PROSE_WORDS_RE = re.compile(
    r"\b(?:where|which|from|then|becomes|therefore|the|this|that|"
    r"these|those|for|with|into|using|equation|factor|value|note)\b",
    re.IGNORECASE,
)


def score_recognition(
    *, latex: str | None, plain_text: str, category: str
) -> tuple[float, list[str]]:
    """Return ``(confidence, notes)`` for one recognition result.

    ``notes`` are ``quality:<reason>`` markers for every penalty applied (empty when clean).
    """
    if category in ("chemical_structure", "chemical_equation"):
        has_content = bool((plain_text or "").strip())
        return (_BASE_CONFIDENCE if has_content else 0.0), []

    text = (latex or "").strip()
    if not text:
        return 0.1, ["quality:empty_latex"]

    notes: list[str] = []
    score = _BASE_CONFIDENCE

    if category in MATH_CATEGORIES and _SPACED_CHARS_RE.search(text):
        score = min(score, 0.10)
        notes.append("quality:spaced_text")
        return round(score, 4), notes

    if _LABEL_ONLY_RE.match(text):
        score = min(score, 0.10)
        notes.append("quality:label_only")
        return round(score, 4), notes

    if category in MATH_CATEGORIES:
        prose_hits = len(_PROSE_WORDS_RE.findall(text))
        if prose_hits >= 2 and not _REL_OP_RE.search(text):
            score = min(score, 0.20)
            notes.append("quality:prose_contamination")

    if category in MATH_CATEGORIES and "quality:prose_contamination" not in notes:
        _CONNECTIVE_RE = re.compile(
            r"\b(?:or|and|the|for|with|at|of|in|is|are|to|by)\b", re.IGNORECASE
        )
        _text_blocks = re.findall(r"\\text\{([^}]*)\}", text)
        prose_in_text = sum(
            1 for blk in _text_blocks if len(_CONNECTIVE_RE.findall(blk)) >= 2
        )
        if prose_in_text:
            score = min(score, 0.30)
            notes.append("quality:prose_in_text_block")

    math_chars = re.sub(r"\\[a-zA-Z]+|\s", "", text)
    if len(math_chars) < 3:
        score = min(score, 0.30)
        notes.append("quality:suspiciously_short")

    if len(_TAG_RE.findall(text)) >= 2:
        score = min(score, 0.35)
        notes.append("quality:multiple_tags")

    if category in MATH_CATEGORIES and "quality:multiple_tags" not in notes:
        op_lines = sum(1 for ln in text.split("\n") if _REL_OP_RE.search(ln.strip()))
        if op_lines >= 2:
            score = min(score, 0.45)
            notes.append("quality:multiple_equations")

    body = _TAG_BLOCK_RE.sub("", text).strip()
    if _LHS_CLIPPED_RE.match(body):
        score = min(score, 0.35)
        notes.append("quality:lhs_clipped")
    elif category in MATH_CATEGORIES and not _REL_OP_RE.search(body):
        score = min(score, 0.45)
        notes.append("quality:no_relational_operator")

    return round(score, 4), notes


# ============================================================
# SECTION 4: Merged from pipeline/inference/prompts.py
# Centralised prompt library for the inference layer
#
# All strings sent to a vision-language model originate here.
# ============================================================

EQUATION_SYSTEM_PROMPT: str = (
    "You are an expert mathematical typesetter specialising in engineering and scientific literature, "
    "including thermodynamics, fluid mechanics, structural analysis, and chemical engineering. "
    "Transcribe equations exactly as they appear in the source — do not simplify, rearrange, reorder terms, "
    "or silently correct what appears to be a typographic error; transcribe an apparent error faithfully. "
    "Output only the requested format: the bare expression and nothing else. "
    "Do not wrap output in markdown fences, do not add prose explanations, "
    "and do not enclose LaTeX in dollar signs or \\begin{equation} blocks. "
    "When a symbol is ambiguous due to image quality or typographic similarity "
    "(e.g. η vs n, ∂ vs δ, × vs ·), choose the most contextually appropriate symbol. "
    "If no equation or formula is present, return nothing — never fabricate or infer an equation "
    "that is not visually present."
)

_EQUATION_MATH_IMAGE_PROMPT: str = (
    "Transcribe the equation or formula in this image as LaTeX.\n"
    "- Preserve the exact form — do not simplify or rearrange terms.\n"
    "- Preserve subscripts, superscripts, Greek letters "
    "(α β γ δ ε ζ η θ λ μ ν ξ π ρ σ τ φ ω Δ Σ Ω), "
    "and dimensional units (e.g. [Pa], [m/s²]) exactly as shown.\n"
    "- For integrals: treat adjacent symbols as a product unless clearly separated — "
    "e.g. '∫ ps du' should be \\int ps\\,du (not \\int p\\,s\\,d\\,u).\n"
    "- Parenthesised numbers at the end of a line (e.g. '(2-7)', '(3.14)') are equation "
    "reference labels — do NOT include them in the LaTeX output.\n"
    "- If the image contains multiple equations, output each on its own line.\n"
    "- Output only the LaTeX — no dollar signs, no \\[ \\], no markdown fences, no explanation."
)

_EQUATION_MATH_IMAGE_PROMPT_STRICT: str = (
    "Transcribe the equation or formula in this image as LaTeX.\n"
    "STRICT MODE — pay close attention to every symbol in the image:\n"
    "- If the left-hand side is not visible (crop starts at '='), begin your output with '=' "
    "— do NOT guess or invent the missing LHS variable.\n"
    "- Parenthesised numbers at the end of a line (e.g. '(2-7)', '(3.14)') are equation "
    "reference labels — do NOT include them in the LaTeX output.\n"
    "- Transcribe ONLY the single primary equation. If several equations are stacked, "
    "output just the topmost/main one.\n"
    "- Distinguish T (temperature, upright seriffed) from I (current/moment, two horizontal "
    "serifs); in thermodynamic/heat contexts, the variable is almost always T.\n"
    "- Preserve the exact form — do not simplify or rearrange terms.\n"
    "- Preserve subscripts, superscripts, Greek letters "
    "(α β γ δ ε ζ η θ λ μ ν ξ π ρ σ τ φ ω Δ Σ Ω), "
    "and dimensional units (e.g. [Pa], [m/s²]) exactly as shown.\n"
    "- Output only the LaTeX — no dollar signs, no \\[ \\], no markdown fences, no explanation."
)

EQUATION_IMAGE_PROMPTS_STRICT: dict[str, str] = {
    "mathematical_equation": _EQUATION_MATH_IMAGE_PROMPT_STRICT,
    "engineering_formula": _EQUATION_MATH_IMAGE_PROMPT_STRICT,
    "statistical_expression": _EQUATION_MATH_IMAGE_PROMPT_STRICT,
}

EQUATION_IMAGE_PROMPTS: dict[str, str] = {
    "mathematical_equation": _EQUATION_MATH_IMAGE_PROMPT,
    "engineering_formula": _EQUATION_MATH_IMAGE_PROMPT,
    "statistical_expression": _EQUATION_MATH_IMAGE_PROMPT,
    "chemical_equation": (
        "Transcribe the chemical equation in this image using standard notation.\n"
        "- Use → for reaction arrows, ⇌ for equilibrium, + between species.\n"
        "- Preserve subscript numbers, charge symbols (²⁺, ³⁻), and state labels "
        "(s), (l), (g), (aq).\n"
        "- If structural formulas are present, add SMILES on a second line "
        "prefixed exactly with 'SMILES:'.\n"
        "- Output only the equation — no explanation."
    ),
    "chemical_structure": (
        "Extract the chemical structure from this image as a SMILES string.\n"
        "- Output only the SMILES — no name, no explanation, no other text."
    ),
    "unknown": (
        "Identify and transcribe the equation or formula in this image.\n"
        "- If mathematical: output as LaTeX (no dollar signs, no fences).\n"
        "- If chemical: output plain text with → for arrows.\n"
        "- Output only the equation — no explanation."
    ),
}

_EQUATION_MATH_TEXT_PROMPT: str = (
    "Convert this equation or formula to LaTeX. "
    "Note: some characters may be encoding artifacts "
    "(¼ means =, (cid:N) means an unmapped symbol). "
    "Output only the LaTeX — no dollar signs, no fences.\n"
    "Equation: {text}"
)

EQUATION_TEXT_PROMPTS: dict[str, str] = {
    "mathematical_equation": _EQUATION_MATH_TEXT_PROMPT,
    "engineering_formula": _EQUATION_MATH_TEXT_PROMPT,
    "statistical_expression": _EQUATION_MATH_TEXT_PROMPT,
    "chemical_equation": (
        "Write this chemical equation in standard notation "
        "(→ for reactions, ⇌ for equilibrium). "
        "Output only the equation.\nEquation: {text}"
    ),
    "chemical_structure": (
        "Convert this chemical structure description to a SMILES string. "
        "Output only the SMILES.\nStructure: {text}"
    ),
    "unknown": (
        "Format this as LaTeX if mathematical, or plain text if chemical. "
        "Output only the result — no explanation.\nContent: {text}"
    ),
}

EQUATION_VALIDATE_SYSTEM: str = (
    "You are an expert at validating mathematical LaTeX. "
    "You will be given an equation image and a candidate LaTeX string. "
    "Return ONLY valid JSON with no explanation and no markdown fences. "
    'Schema: {"is_valid": bool, "corrected_latex": string|null, '
    '"issues": [string], "confidence": float}'
)

_EQUATION_VALIDATE_USER_TEMPLATE: str = (
    "Does this LaTeX correctly represent the equation in the image?\n"
    "Candidate LaTeX: {latex}\n\n"
    "Return JSON only."
)

PAGE_EQUATIONS_SYSTEM: str = (
    "You are an expert at identifying and extracting equations from scientific "
    "and engineering document pages. "
    "Return ONLY valid JSON — no explanation, no markdown fences. "
    "Schema: a JSON object with a single key 'equations' whose value is an "
    "array. Each element: "
    '{"latex": string|null, "plain_text": string, '
    '"category": string, "confidence": float, '
    '"structured_form": string|null}'
    ". category must be one of: mathematical_equation, engineering_formula, "
    "statistical_expression, chemical_equation, chemical_structure, unknown."
)

PAGE_EQUATIONS_USER: str = (
    "Find and extract every equation, formula, or chemical structure on this "
    "page. For mathematical content return LaTeX; for chemical structures "
    "return SMILES in structured_form. Return JSON only."
)

STRUCTURED_JSON_SYSTEM: str = (
    "You are an expert at extracting structured information from document "
    "images. Return ONLY valid JSON — no explanation, no markdown fences, "
    "no preamble."
)

_STRUCTURED_JSON_USER_TEMPLATE: str = (
    "Extract the information from this image as JSON.{schema_clause} "
    "Return JSON only."
)


def get_equation_image_prompt(category: str, strict: bool = False) -> str:
    """Return the image prompt for ``category``, falling back to 'unknown'."""
    if strict and category in EQUATION_IMAGE_PROMPTS_STRICT:
        return EQUATION_IMAGE_PROMPTS_STRICT[category]
    return EQUATION_IMAGE_PROMPTS.get(category, EQUATION_IMAGE_PROMPTS["unknown"])


def get_equation_text_prompt(category: str, text: str) -> str:
    """Return the text-only prompt for ``category`` with ``text`` substituted."""
    template = EQUATION_TEXT_PROMPTS.get(category, EQUATION_TEXT_PROMPTS["unknown"])
    return template.format(text=text)


def build_equation_validate_user(latex: str) -> str:
    """Format the validation user message with the candidate ``latex``."""
    return _EQUATION_VALIDATE_USER_TEMPLATE.format(latex=latex)


def build_structured_json_user(schema_hint: str = "") -> str:
    """Format the structured-JSON user message."""
    schema_clause = f" {schema_hint.strip()}" if schema_hint.strip() else ""
    return _STRUCTURED_JSON_USER_TEMPLATE.format(schema_clause=schema_clause)


# ============================================================
# SECTION 5: Merged from pipeline/inference/client.py
# InferenceBackend protocol and OllamaBackend implementation
# ============================================================


class InferenceBackend(abc.ABC):
    """Contract every inference backend must fulfill.

    The three abstract methods are deliberately minimal so that new backends
    (vLLM, Azure, HuggingFace TGI) can be added without changing any caller.
    """

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Stable short identifier used in logs and metrics."""

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Probe reachability.

        Must always return a bool and must never raise. Uses a short dedicated
        timeout (≤2 s) so callers can skip a whole batch quickly.
        """

    @abc.abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Send ``messages`` and return the assistant content string.

        Raises:
            InferenceUnavailableError, InferenceTimeoutError,
            ModelNotInstalledError, InferenceResponseError, InferenceError.
        """


def _unwrap_retry_error(exc: Exception) -> Exception:
    """Extract the real cause from a tenacity RetryError."""
    last = getattr(exc, "last_attempt", None)
    if last is not None:
        try:
            inner = last.exception()
            if inner is not None:
                return inner
        except Exception:
            pass
    return exc


class OllamaBackend(InferenceBackend):
    """Ollama via the OpenAI-compatible ``/v1/chat/completions`` endpoint.

    Args:
        host:        Base URL of the Ollama server.
        model:       Default model name when the caller does not override.
        timeout:     Per-request timeout in seconds.
        max_retries: Tenacity retry attempts on transient failures.
    """

    def __init__(
        self,
        *,
        host: str,
        model: str,
        timeout: float,
        max_retries: int,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._retrying_call: Callable = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=8),
        )(self._call_once)

    @property
    def backend_name(self) -> str:
        return "ollama"

    def health_check(self) -> bool:
        """GET /api/tags with a 2 s timeout — returns False on any failure."""
        try:
            resp = httpx.get(f"{self._host}/api/tags", timeout=2.0)
            return getattr(resp, "status_code", 500) < 500
        except Exception:
            return False

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Dispatch to ``_retrying_call`` and log duration + retry outcome."""
        effective_model = model or self._model
        bound_log = _slog.bind(
            backend=self.backend_name,
            model=effective_model,
            host=self._host,
        )
        start = time.monotonic()
        try:
            raw = self._retrying_call(messages, effective_model, max_tokens, temperature)
            bound_log.info(
                "inference_complete",
                duration_s=round(time.monotonic() - start, 3),
            )
            return raw
        except RetryError as exc:
            cause = _unwrap_retry_error(exc)
            bound_log.error(
                "inference_failed_after_retries",
                error=str(cause),
                retries=self._max_retries,
                duration_s=round(time.monotonic() - start, 3),
            )
            raise cause from exc
        except InferenceError:
            bound_log.error(
                "inference_error",
                duration_s=round(time.monotonic() - start, 3),
            )
            raise

    def _call_once(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """One HTTP POST to ``/v1/chat/completions``. Raises on any failure."""
        url = f"{self._host}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            response = httpx.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self._timeout,
            )
        except ConnectError as exc:
            raise InferenceUnavailableError(
                f"cannot reach Ollama at {self._host}",
                {"host": self._host, "error": str(exc)},
            ) from exc
        except TimeoutException as exc:
            raise InferenceTimeoutError(
                f"Ollama request timed out after {self._timeout}s",
                {"host": self._host, "model": model, "timeout": self._timeout},
            ) from exc
        except Exception as exc:
            raise InferenceError(
                f"unexpected HTTP error: {exc}",
                {"host": self._host, "model": model},
            ) from exc

        status = getattr(response, "status_code", 500)

        if status == 404:
            raise ModelNotInstalledError(
                f"model '{model}' not found — run: ollama pull {model}",
                {"model": model, "host": self._host},
            )

        try:
            response.raise_for_status()
        except HTTPStatusError as exc:
            raise InferenceResponseError(
                f"Ollama returned HTTP {status}",
                {"status_code": status, "body_excerpt": response.text[:200]},
            ) from exc

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise InferenceResponseError(
                "unexpected Ollama response shape",
                {"body_excerpt": response.text[:200]},
            ) from exc


# Backend registry and factory

_BackendFactory = Callable[[], InferenceBackend]

_BACKEND_REGISTRY: dict[str, _BackendFactory] = {
    "ollama": lambda: OllamaBackend(
        host=config.KNOVEL_OLLAMA_BASE_URL,
        model=config.KNOVEL_OLLAMA_FAST_MODEL,
        timeout=float(getattr(config, "OLLAMA_TIMEOUT", config.KNOVEL_LLM_TIMEOUT)),
        max_retries=config.KNOVEL_LLM_MAX_RETRIES,
    ),
}


def create_backend(backend_type: str | None = None) -> InferenceBackend:
    """Instantiate and return a configured ``InferenceBackend``.

    Args:
        backend_type: Registry key (e.g. ``'ollama'``). Reads
            ``KNOVEL_LLM_BACKEND`` from config when ``None``.

    Raises:
        ValueError: Unknown backend key not present in the registry.
    """
    key = (backend_type or config.KNOVEL_LLM_BACKEND).lower()
    factory = _BACKEND_REGISTRY.get(key)
    if factory is None:
        supported = ", ".join(sorted(_BACKEND_REGISTRY))
        raise ValueError(
            f"unknown inference backend '{key}'. Supported: {supported}"
        )
    return factory()


# ============================================================
# SECTION 6: Merged from pipeline/inference/vision_service.py
# VisionService — provider-agnostic VLM facade
# ============================================================

# Display-only math environments the VLM sometimes wraps output in despite the
# prompt forbidding it. Strip the begin/end pair (structural environments are preserved).
_MATH_ENV_WRAPPER_RE = re.compile(
    r"\\(?:begin|end)\{(?:equation|displaymath|math|gather|multline)\*?\}",
    re.IGNORECASE,
)


def _clean_latex(text: str) -> str:
    """Strip markdown fences, math delimiters, and display-environment wrappers.

    Qwen2.5-VL frequently ignores the "no dollar signs / no environment" instruction and
    returns ``\\begin{equation} … \\end{equation}`` or orphaned ``\\[ … \\]`` / ``\\( … \\)``
    delimiters (sometimes several blocks in one response). Strip all of them so the stored
    LaTeX is the bare expression.
    """
    text = text.strip()
    text = re.sub(r"^```(?:latex)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text)
    text = _MATH_ENV_WRAPPER_RE.sub("", text)
    text = re.sub(r"^\$\$\s*", "", text)
    text = re.sub(r"\s*\$\$$", "", text)
    text = re.sub(r"^\\\[\s*", "", text)
    text = re.sub(r"\s*\\\]$", "", text)
    if text.startswith("$") and text.endswith("$") and len(text) > 2:
        text = text[1:-1]
    text = re.sub(r"\s*\\[\[\]()]\s*", "\n", text)
    text = re.sub(r"\n*\\beg?i?n?\{?[a-z]*\*?\}?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _encode_image_file(image_path: Path) -> str:
    """Read ``image_path`` and return a base64-encoded PNG string."""
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _encode_pil_image(image: Image.Image, zoom: float = 1.0) -> str:
    """Return base64-encoded PNG of *image*, optionally upscaled by *zoom*."""
    if zoom != 1.0:
        new_w = int(image.width * zoom)
        new_h = int(image.height * zoom)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_image_messages(
    *,
    system: str | None,
    user_text: str,
    image_path: Path,
) -> list[dict[str, Any]]:
    """Build an OpenAI-style messages list with an image attachment."""
    b64 = _encode_image_file(image_path)
    user_content: list[dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        },
        {"type": "text", "text": user_text},
    ]
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    return messages


def _image_to_b64(image: Any) -> str:
    """Encode a PIL Image to a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract and parse a JSON object from ``raw``, stripping markdown fences."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text).strip()
    return json.loads(text)


@dataclass
class EquationResult:
    """Extraction result for a single equation region."""

    plain_text: str
    latex: str | None = None
    structured_form: str | None = None
    confidence: float = 0.0
    category: str = "unknown"
    model_used: str = ""
    backend: str = ""
    duration_s: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of validating a LaTeX string against its source image."""

    is_valid: bool
    corrected_latex: str | None = None
    issues: list[str] = field(default_factory=list)
    confidence: float = 0.0
    model_used: str = ""
    backend: str = ""


@dataclass
class PageEquationsResult:
    """All equations extracted from a single page image."""

    equations: list[EquationResult] = field(default_factory=list)
    page_index: int | None = None
    model_used: str = ""
    backend: str = ""


def _parse_equation_response(
    *,
    raw: str,
    category: str,
    model_used: str,
    backend: str,
    duration_s: float,
) -> EquationResult:
    """Convert a raw model string into a structured ``EquationResult``."""
    if not raw:
        return EquationResult(
            plain_text="",
            notes=["empty_response"],
            model_used=model_used,
            backend=backend,
            duration_s=duration_s,
            category=category,
        )

    if category == "chemical_structure":
        return EquationResult(
            plain_text=raw,
            structured_form=raw.strip(),
            confidence=0.85,
            category=category,
            model_used=model_used,
            backend=backend,
            duration_s=duration_s,
        )

    if category == "chemical_equation":
        smiles: str | None = None
        plain_lines: list[str] = []
        for line in raw.splitlines():
            if line.strip().upper().startswith("SMILES:"):
                smiles = line.split(":", 1)[1].strip() or None
            else:
                plain_lines.append(line)
        plain = " ".join(plain_lines).strip()
        return EquationResult(
            plain_text=plain,
            structured_form=smiles,
            confidence=0.85 if plain else 0.0,
            category=category,
            model_used=model_used,
            backend=backend,
            duration_s=duration_s,
        )

    latex = _clean_latex(raw)
    confidence, quality_notes = score_recognition(
        latex=latex, plain_text=latex, category=category
    )
    return EquationResult(
        plain_text=latex,
        latex=latex or None,
        confidence=confidence,
        notes=quality_notes,
        category=category,
        model_used=model_used,
        backend=backend,
        duration_s=duration_s,
    )


def _parse_validation_response(
    *,
    raw: str,
    model_used: str,
    backend: str,
) -> ValidationResult:
    """Parse a JSON validation response into a ``ValidationResult``."""
    try:
        data = _extract_json(raw)
        return ValidationResult(
            is_valid=bool(data.get("is_valid", False)),
            corrected_latex=data.get("corrected_latex"),
            issues=data.get("issues", []),
            confidence=float(data.get("confidence", 0.0)),
            model_used=model_used,
            backend=backend,
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        return ValidationResult(
            is_valid=False,
            issues=["malformed_validation_response"],
            model_used=model_used,
            backend=backend,
        )


def _parse_page_equations_response(
    *,
    raw: str,
    model_used: str,
    backend: str,
) -> list[EquationResult]:
    """Parse a JSON list of equations from a page-level extraction response."""
    try:
        data = _extract_json(raw)
        items: list[dict[str, Any]] = data.get("equations", [])
    except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
        _slog.warning("page_equations_parse_failed", raw_excerpt=raw[:200])
        return []

    results: list[EquationResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            EquationResult(
                plain_text=item.get("plain_text", ""),
                latex=item.get("latex"),
                structured_form=item.get("structured_form"),
                confidence=float(item.get("confidence", 0.0)),
                category=item.get("category", "unknown"),
                model_used=model_used,
                backend=backend,
            )
        )
    return results


class VisionService:
    """Provider-agnostic facade for vision-language model operations.

    Args:
        backend: A configured ``InferenceBackend`` instance.
    """

    def __init__(self, backend: InferenceBackend) -> None:
        self._backend = backend

    def extract_equation(
        self,
        image_path: Path,
        category: str = "unknown",
        model: str | None = None,
        max_tokens: int = 512,
        strict: bool = False,
    ) -> EquationResult:
        """Extract a single equation from a cropped region image."""
        prompt = get_equation_image_prompt(category, strict=strict)
        messages = _build_image_messages(system=None, user_text=prompt, image_path=image_path)
        bound_log = _slog.bind(
            method="extract_equation",
            category=category,
            backend=self._backend.backend_name,
        )
        start = time.monotonic()
        try:
            raw = self._backend.chat_completion(messages, model=model, max_tokens=max_tokens)
            duration = round(time.monotonic() - start, 3)
            bound_log.info("extract_equation_ok", duration_s=duration)
            return _parse_equation_response(
                raw=raw,
                category=category,
                model_used=model or "",
                backend=self._backend.backend_name,
                duration_s=duration,
            )
        except InferenceError as exc:
            duration = round(time.monotonic() - start, 3)
            bound_log.warning("extract_equation_failed", error=str(exc), duration_s=duration)
            return EquationResult(
                plain_text="",
                notes=[f"inference_error:{type(exc).__name__}"],
                model_used=model or "",
                backend=self._backend.backend_name,
                duration_s=duration,
                category=category,
            )

    def validate_equation(
        self,
        image_path: Path,
        latex: str,
        model: str | None = None,
    ) -> ValidationResult:
        """Validate whether ``latex`` correctly represents the equation in the image."""
        user_text = build_equation_validate_user(latex)
        messages = _build_image_messages(
            system=EQUATION_VALIDATE_SYSTEM,
            user_text=user_text,
            image_path=image_path,
        )
        bound_log = _slog.bind(
            method="validate_equation",
            backend=self._backend.backend_name,
        )
        start = time.monotonic()
        try:
            raw = self._backend.chat_completion(messages, model=model, max_tokens=512)
            bound_log.info(
                "validate_equation_ok",
                duration_s=round(time.monotonic() - start, 3),
            )
            return _parse_validation_response(
                raw=raw,
                model_used=model or "",
                backend=self._backend.backend_name,
            )
        except InferenceError as exc:
            bound_log.warning(
                "validate_equation_failed",
                error=str(exc),
                duration_s=round(time.monotonic() - start, 3),
            )
            return ValidationResult(
                is_valid=False,
                issues=[f"inference_error:{type(exc).__name__}"],
                model_used=model or "",
                backend=self._backend.backend_name,
            )

    def extract_equations_from_page(
        self,
        image_path: Path,
        page_index: int | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> PageEquationsResult:
        """Find and extract all equations from a full-page image in one call."""
        messages = _build_image_messages(
            system=PAGE_EQUATIONS_SYSTEM,
            user_text=PAGE_EQUATIONS_USER,
            image_path=image_path,
        )
        bound_log = _slog.bind(
            method="extract_equations_from_page",
            page_index=page_index,
            backend=self._backend.backend_name,
        )
        start = time.monotonic()
        try:
            raw = self._backend.chat_completion(
                messages, model=model, max_tokens=max_tokens
            )
            duration = round(time.monotonic() - start, 3)
            bound_log.info("extract_page_equations_ok", duration_s=duration)
            equations = _parse_page_equations_response(
                raw=raw,
                model_used=model or "",
                backend=self._backend.backend_name,
            )
            return PageEquationsResult(
                equations=equations,
                page_index=page_index,
                model_used=model or "",
                backend=self._backend.backend_name,
            )
        except InferenceError as exc:
            duration = round(time.monotonic() - start, 3)
            bound_log.warning(
                "extract_page_equations_failed",
                error=str(exc),
                duration_s=duration,
            )
            return PageEquationsResult(
                page_index=page_index,
                model_used=model or "",
                backend=self._backend.backend_name,
            )

    def generate_structured_json(
        self,
        image_path: Path,
        schema_hint: str = "",
        system_prompt: str = "",
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Extract structured JSON from an image using an optional schema hint.

        Raises:
            MalformedResponseError: The model did not return valid JSON.
        """
        system = system_prompt.strip() or STRUCTURED_JSON_SYSTEM
        user_text = build_structured_json_user(schema_hint)
        messages = _build_image_messages(
            system=system,
            user_text=user_text,
            image_path=image_path,
        )
        bound_log = _slog.bind(
            method="generate_structured_json",
            backend=self._backend.backend_name,
        )
        start = time.monotonic()
        raw = self._backend.chat_completion(messages, model=model, max_tokens=max_tokens)
        duration = round(time.monotonic() - start, 3)
        try:
            result = _extract_json(raw)
            bound_log.info("generate_structured_json_ok", duration_s=duration)
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            bound_log.error(
                "generate_structured_json_parse_failed",
                raw_excerpt=raw[:200],
                duration_s=duration,
            )
            raise MalformedResponseError(
                "model did not return valid JSON",
                {"raw_excerpt": raw[:200]},
            ) from exc

    def extract_equation_from_text(
        self,
        text: str,
        category: str = "unknown",
        model: str | None = None,
        max_tokens: int = 256,
    ) -> EquationResult:
        """Extract/normalise an equation from embedded text (no image available)."""
        user_text = get_equation_text_prompt(category, text)
        messages = [{"role": "user", "content": user_text}]
        bound_log = _slog.bind(
            method="extract_equation_from_text",
            category=category,
            backend=self._backend.backend_name,
        )
        start = time.monotonic()
        try:
            raw = self._backend.chat_completion(messages, model=model, max_tokens=max_tokens)
            duration = round(time.monotonic() - start, 3)
            bound_log.info("extract_equation_text_ok", duration_s=duration)
            return _parse_equation_response(
                raw=raw,
                category=category,
                model_used=model or "",
                backend=self._backend.backend_name,
                duration_s=duration,
            )
        except InferenceError as exc:
            duration = round(time.monotonic() - start, 3)
            bound_log.warning(
                "extract_equation_text_failed",
                error=str(exc),
                duration_s=duration,
            )
            return EquationResult(
                plain_text=text,
                notes=[f"inference_error:{type(exc).__name__}"],
                model_used=model or "",
                backend=self._backend.backend_name,
                duration_s=duration,
                category=category,
            )


# ============================================================
# SECTION 7: Merged from providers.py
# EquationProvider protocol and concrete implementations
#
# QwenVLProvider: Qwen2.5-VL via OpenAI-compatible API
# GenericProvider: plain-text passthrough fallback
# ============================================================

# Knovel-specific OCR character-substitution table
_OCR_FIX: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"¼"), "="),
    (re.compile(r"\(cid:2\)"), "×"),
    (re.compile(r"\(cid:\d+\)"), ""),
    (re.compile(r"[ﬃﬀﬁﬂ]{3,}"), "√"),
    (re.compile(r"ð"), "("),
    (re.compile(r"Þ"), ")"),
    (re.compile(r"\b([lI])\.(\d)"), r"1.\2"),
    (re.compile(r"(?<=[\d.])[lI](?=[\d])"), "1"),
    (re.compile(r"(?<=[=(\[{+\-*/\s])([lI])(?=[.\d])"), "1"),
    (re.compile(r"\s*;\s*"), " "),
]

_EQ_LABEL_IN_TEXT_RE = re.compile(r"\bEq(?:uation)?[.:]?\s*\d[\d.]*", re.IGNORECASE)

_PROSE_RESPONSE_RE = re.compile(
    r"^(nothing\b"
    r"|it\s|the\s+(?:expression|equation|instruction|formula|given|notation|request|text|image)"
    r"|i\s|certainly|please|note\s|you\s|based\s|as\s+provided|this\s+(?:expression|equation|notation|text)"
    r"|there\s+(?:is|are|was|were|might|seem)|sorry|unfortunately)",
    re.IGNORECASE,
)

_LATEX_STRUCTURAL_CMD_RE = re.compile(
    r"^\\(?:section|subsection|subsubsection|chapter|part|paragraph|subparagraph"
    r"|newcommand|renewcommand|def|let|providecommand"
    r"|begin\{document\}|end\{document\}|documentclass|usepackage|maketitle"
    r"|label|ref|cite|bibliography|bibitem|footnote|caption)\b",
    re.IGNORECASE,
)


def _apply_ocr_fixes(text: str) -> str:
    """Apply Knovel-specific OCR character substitutions to raw PDF text."""
    if not text:
        return text
    t = text.replace("\n", " ")
    t = _EQ_LABEL_IN_TEXT_RE.sub("", t)
    for pat, repl in _OCR_FIX:
        t = pat.sub(repl, t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class RecognitionResult:
    """The structured output of one provider for one equation."""

    plain_text: str = ""
    latex: str | None = None
    mathml: str | None = None
    structured_form: str | None = None
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


@runtime_checkable
class EquationProvider(Protocol):
    """Common recognition interface; concrete providers are selected by configuration."""

    name: str
    backend: str
    available: bool

    def recognize(
        self,
        *,
        region_image: Any,
        region_text: str,
        category: str,
        config: Any,
        strict: bool = False,
    ) -> RecognitionResult: ...


class QwenVLProvider:
    """Equation recognition via Qwen2.5-VL through an OpenAI-compatible API.

    Configuration keys (read from the pipeline config object at recognize-time):

    * ``KNOVEL_OLLAMA_BASE_URL``        — API server root
    * ``KNOVEL_EQUATION_VL_MODEL``      — model name
    * ``KNOVEL_EQUATION_VL_TIMEOUT``    — per-request timeout in seconds
    * ``KNOVEL_EQUATION_VL_MAX_TOKENS`` — maximum response tokens
    """

    name = "qwen_vl"
    backend = "qwen2.5-vl"
    available = True

    def __init__(self) -> None:
        self._client: Any = None

    def _get_client(self, httpx_mod: Any) -> Any:
        if self._client is None:
            self._client = httpx_mod.Client(
                headers={"Content-Type": "application/json"},
                limits=httpx_mod.Limits(max_keepalive_connections=4, max_connections=8),
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    def recognize(
        self,
        *,
        region_image: Any,
        region_text: str,
        category: str,
        config: Any,
        strict: bool = False,
    ) -> RecognitionResult:
        try:
            import httpx as _httpx
        except ImportError:
            return RecognitionResult(
                plain_text=region_text, confidence=0.0, notes=["provider_absent:httpx"]
            )

        base_url = getattr(config, "KNOVEL_OLLAMA_BASE_URL", "http://localhost:11434").rstrip(
            "/"
        )
        model = getattr(
            config,
            "KNOVEL_EQUATION_VL_MODEL",
            getattr(config, "OLLAMA_VL_MODEL", "qwen2.5vl:7b"),
        )
        timeout = float(getattr(config, "KNOVEL_EQUATION_VL_TIMEOUT", 60))
        max_tokens = int(getattr(config, "KNOVEL_EQUATION_VL_MAX_TOKENS", 512))

        messages = self._build_messages(region_image, region_text, category, strict=strict)

        try:
            client = self._get_client(_httpx)
            resp = client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw: str = resp.json()["choices"][0]["message"]["content"].strip()
        except _httpx.ConnectError:
            _slog.warning("qwen_vl_server_unreachable", base_url=base_url, model=model)
            return RecognitionResult(
                plain_text=region_text,
                confidence=0.0,
                notes=["provider_absent:qwen_vl"],
            )
        except Exception as exc:
            _slog.warning("qwen_vl_request_failed", error=str(exc))
            return RecognitionResult(
                plain_text=region_text,
                confidence=0.0,
                notes=[f"recognition_failed:{type(exc).__name__}"],
            )

        return self._parse_response(raw, region_text, category)

    def is_equation(self, text: str, *, config: Any) -> bool:
        """Yes/no oracle: is *text* a mathematical/engineering formula?"""
        stripped = (text or "").strip()
        if not stripped:
            return False
        try:
            import httpx as _httpx
        except ImportError:
            return False

        base_url = getattr(config, "KNOVEL_OLLAMA_BASE_URL", "http://localhost:11434").rstrip(
            "/"
        )
        model = getattr(
            config,
            "KNOVEL_EQUATION_VL_MODEL",
            getattr(config, "OLLAMA_VL_MODEL", "qwen2.5vl:7b"),
        )
        timeout = float(getattr(config, "KNOVEL_EQUATION_VL_TIMEOUT", 30))
        messages = [
            {
                "role": "user",
                "content": (
                    "Is the following text a mathematical or engineering formula or equation? "
                    "Answer with exactly one word: yes or no.\n\nText: " + stripped[:200]
                ),
            }
        ]
        try:
            client = self._get_client(_httpx)
            resp = client.post(
                f"{base_url}/v1/chat/completions",
                json={"model": model, "messages": messages, "max_tokens": 5},
                timeout=timeout,
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
        except Exception as exc:
            _slog.debug("qwen_vl_is_equation_failed", error=str(exc))
            return False
        return answer.startswith("yes")

    def _build_messages(
        self, region_image: Any, region_text: str, category: str, strict: bool = False
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": EQUATION_SYSTEM_PROMPT}]
        if region_image is not None:
            prompt = get_equation_image_prompt(category, strict=strict)
            hint = _apply_ocr_fixes((region_text or "").strip())
            if hint:
                prompt = (
                    prompt + f'\n\nRaw text extracted from this region: "{hint}"\n'
                    "Use this as a reference only — trust the image for correct notation."
                )
            b64 = _image_to_b64(region_image)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            )
        else:
            tmpl = EQUATION_TEXT_PROMPTS.get(category, EQUATION_TEXT_PROMPTS["unknown"])
            messages.append(
                {
                    "role": "user",
                    "content": tmpl.format(text=_apply_ocr_fixes(region_text or "")),
                }
            )
        return messages

    def _parse_response(self, raw: str, region_text: str, category: str) -> RecognitionResult:
        if not raw:
            return RecognitionResult(
                plain_text=region_text, confidence=0.0, notes=["empty_response"]
            )

        if _PROSE_RESPONSE_RE.match(raw.strip()):
            _slog.debug(
                "vl_response_rejected_prose",
                category=category,
                preview=raw.strip()[:80],
            )
            return RecognitionResult(
                plain_text=region_text,
                confidence=0.1,
                notes=["vl_response_rejected:prose"],
            )

        if _LATEX_STRUCTURAL_CMD_RE.match(raw.strip()):
            _slog.debug(
                "vl_response_rejected_structural_latex",
                category=category,
                preview=raw.strip()[:80],
            )
            return RecognitionResult(
                plain_text=region_text,
                confidence=0.1,
                notes=["vl_response_rejected:structural_latex"],
            )

        if category == "chemical_structure":
            return RecognitionResult(
                plain_text=region_text or raw,
                structured_form=raw,
                confidence=0.85,
            )

        if category == "chemical_equation":
            smiles: str | None = None
            plain_lines: list[str] = []
            for line in raw.splitlines():
                if line.strip().upper().startswith("SMILES:"):
                    smiles = line.split(":", 1)[1].strip() or None
                else:
                    plain_lines.append(line)
            plain = " ".join(plain_lines).strip() or region_text
            return RecognitionResult(
                plain_text=plain,
                structured_form=smiles,
                confidence=0.85 if plain else 0.0,
            )

        latex = _clean_latex(raw)
        confidence, quality_notes = score_recognition(
            latex=latex, plain_text=latex or region_text, category=category
        )
        return RecognitionResult(
            plain_text=latex or region_text,
            latex=latex or None,
            confidence=confidence,
            notes=quality_notes,
        )


class GenericProvider:
    """Pure passthrough of the embedded plain text — no model, always available."""

    name = "generic"
    backend = "passthrough"
    available = True

    def recognize(
        self,
        *,
        region_image: Any,
        region_text: str,
        category: str,
        config: Any,
        strict: bool = False,
    ) -> RecognitionResult:
        text = _apply_ocr_fixes((region_text or "").strip())
        return RecognitionResult(plain_text=text, confidence=0.2 if text else 0.0)

    def is_equation(self, text: str, *, config: Any) -> bool:
        return False

    def close(self) -> None:
        pass


# ============================================================
# SECTION 8: Merged from registry.py
# Config-driven selection of equation-recognition providers
# ============================================================

_REGISTRY: dict[str, Callable[[], EquationProvider]] = {
    "qwen_vl": QwenVLProvider,
    "generic": GenericProvider,
}


def register_provider(role: str, factory: Callable[[], EquationProvider]) -> None:
    """Register a provider factory under ``role`` (for tests or alternative backends)."""
    _REGISTRY[role] = factory


def resolve_provider(role: str) -> EquationProvider:
    """Return a provider instance for ``role``; unknown roles fall back to generic."""
    factory = _REGISTRY.get(role, GenericProvider)
    return factory()


def resolve_providers() -> dict[str, EquationProvider]:
    """Instantiate one provider per role for the document run."""
    return {role: factory() for role, factory in _REGISTRY.items()}


def provider_identities(providers: dict[str, EquationProvider]) -> dict[str, str]:
    """Map provider role → backend identity for logging and cache key."""
    return {role: getattr(prov, "backend", role) for role, prov in providers.items()}


def close_providers(providers: dict[str, EquationProvider]) -> None:
    """Release any resources (e.g. pooled HTTP clients) held by resolved providers."""
    for prov in providers.values():
        close = getattr(prov, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


# ============================================================
# SECTION 9: Merged from selection.py
# Category → provider selection
# ============================================================


def parse_provider_map(raw: str) -> dict[str, str]:
    """Parse ``"cat=role,cat2=role2"`` into a category→role override map."""
    _VALID_ROLES = {"qwen_vl", "generic"}
    overrides: dict[str, str] = {}
    try:
        from equation_classifier import DEFAULT_PROVIDER_BY_CATEGORY  # type: ignore[import]
    except ImportError:
        DEFAULT_PROVIDER_BY_CATEGORY = {}  # type: ignore[assignment]

    for pair in (raw or "").split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        category, _, role = pair.partition("=")
        category, role = category.strip(), role.strip()
        if category in DEFAULT_PROVIDER_BY_CATEGORY and role in _VALID_ROLES:
            overrides[category] = role
    return overrides


def select_provider(category: str, *, config: Any) -> str:
    """Return the provider role for ``category`` (configured override beats the default map)."""
    try:
        from equation_classifier import DEFAULT_PROVIDER_BY_CATEGORY  # type: ignore[import]
    except ImportError:
        DEFAULT_PROVIDER_BY_CATEGORY = {}  # type: ignore[assignment]

    overrides = parse_provider_map(getattr(config, "KNOVEL_EQUATION_PROVIDER_MAP", ""))
    if category in overrides:
        return overrides[category]
    return DEFAULT_PROVIDER_BY_CATEGORY.get(category, "generic")


# ============================================================
# SECTION 10: Merged from llm_judge.py
# LLM judge for LaTeX quality
#
# Sends the equation crop image AND the candidate LaTeX to Qwen VL and asks it
# to judge whether the LaTeX correctly represents the equation.
# ============================================================

_JUDGE_PROMPT_TEMPLATE = (
    "You are a strict LaTeX quality inspector.\n"
    "I will show you an equation image and a candidate LaTeX transcription.\n\n"
    "Candidate LaTeX:\n"
    "{latex}\n\n"
    "Task:\n"
    "1. Compare the candidate LaTeX to the equation in the image carefully.\n"
    "2. Check every symbol, subscript, superscript, operator, and bracket.\n"
    "3. Account for normal mathematical font variants. In particular, a closed "
    "contour-integral symbol (\\oint) can resemble the Greek letter phi; use "
    "surrounding differential notation such as ds or dx to distinguish it.\n"
    "4. Respond with a JSON object only — no other text:\n"
    "{{\n"
    '  "accepted": true | false,\n'
    '  "score": 0.0 to 1.0,\n'
    '  "reason": "one sentence explaining your verdict"\n'
    "}}"
)

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_EQUATION_SIGNAL_RE = re.compile(
    r"[=<>≤≥≈≠±→⇌∼∝]"
    r"|\\(?:frac|dfrac|tfrac|sum|prod|int|sqrt|lim|begin|rightarrow|leftrightarrow"
    r"|approx|over|sim|le|ge|neq|propto|pm|cdot|times|div|equiv|subset|supset"
    r"|alpha|beta|gamma|delta|theta|lambda|mu|sigma|tau|phi|omega)\b"
)


def _parse_judge_verdict(raw: str) -> dict[str, Any]:
    m = _JSON_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    accepted = "true" in raw.lower() and "false" not in raw.lower()
    score_m = re.search(r"\b(0\.\d+|1\.0)\b", raw)
    score = float(score_m.group(1)) if score_m else (0.8 if accepted else 0.2)
    return {"accepted": accepted, "score": score, "reason": raw[:120].strip()}


def _call_ollama_generate(prompt: str, image_b64: str) -> str:
    """Send a single request to the Ollama /api/generate endpoint and return raw text."""
    payload: dict[str, Any] = {
        "model": config.OLLAMA_VL_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    try:
        response = requests.post(url, json=payload, timeout=config.OLLAMA_TIMEOUT)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.RequestException as exc:
        logger.error("ollama_generate_failed url=%s error=%s", url, exc)
        raise


def judge_latex(latex: str, crop_image: Image.Image) -> JudgeVerdict:
    """Ask the VL model whether *latex* correctly represents the equation in *crop_image*.

    Parameters
    ----------
    latex:
        Candidate LaTeX string from the OCR stage.
    crop_image:
        PIL image of the cropped equation region.

    Returns
    -------
    JudgeVerdict
        Verdict with accepted flag, score (0–1), and a one-sentence reason.
    """
    if not latex or latex == "\\text{UNREADABLE}":
        return JudgeVerdict(accepted=False, score=0.0, reason="empty or unreadable OCR output")

    # The vision judge can be over-confident on tiny crop fragments.  Such a
    # token is not a complete equation regardless of visual similarity.
    if not _EQUATION_SIGNAL_RE.search(latex):
        return JudgeVerdict(
            accepted=False,
            score=0.0,
            reason="candidate is a fragment, not a complete equation",
        )

    if not config.JUDGE_ENABLED:
        return JudgeVerdict(accepted=True, score=1.0, reason="judge disabled")

    prompt = _JUDGE_PROMPT_TEMPLATE.format(latex=latex)
    image_b64 = _encode_pil_image(crop_image)

    try:
        raw = _call_ollama_generate(prompt, image_b64)
    except Exception as exc:
        logger.warning("judge_failed — defaulting to accepted=True error=%s", exc)
        return JudgeVerdict(accepted=True, score=0.5, reason=f"judge unavailable: {exc}")

    parsed = _parse_judge_verdict(raw)
    verdict = JudgeVerdict(
        accepted=bool(parsed.get("accepted", True)),
        score=float(parsed.get("score", 0.5)),
        reason=str(parsed.get("reason", ""))[:200],
    )

    if verdict.score < config.JUDGE_ACCEPT_THRESHOLD:
        verdict.accepted = False

    logger.debug(
        "judge accepted=%s score=%.2f reason=%s",
        verdict.accepted,
        verdict.score,
        verdict.reason,
    )
    return verdict


# ============================================================
# SECTION 11: Merged from qwen.py
# Legacy Qwen VL OCR via Ollama /api/generate endpoint
#
# Uses the older requests-based path to Ollama's /api/generate.
# New code should prefer QwenVLProvider (OpenAI-compatible /v1/chat/completions).
# ============================================================

_LATEX_BLOCK_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_LATEX_INLINE_RE = re.compile(r"\$(.*?)\$", re.DOTALL)
_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+")

_LABEL_BLEED_RE = re.compile(r"\bEq(?:uation)?[.:]?\s*\d[\d.]*", re.IGNORECASE)

_EXTRACTION_PROMPT = (
    "You are a LaTeX equation transcription expert.\n"
    "Look at the equation image and write the complete LaTeX code.\n"
    "Rules:\n"
    "- Wrap the entire equation in $$ ... $$ delimiters.\n"
    "- Use standard LaTeX notation only.\n"
    "- Do not include any explanation, only the LaTeX.\n"
    "- If the image is unclear or contains no equation, respond with: $$\\text{UNREADABLE}$$"
)

_RETRY_PROMPT = (
    "You are a LaTeX equation transcription expert performing a careful second pass.\n"
    "The first transcription was rejected as low-confidence.\n"
    "Look very carefully at every symbol in the equation image and write the exact LaTeX.\n"
    "Rules:\n"
    "- Wrap the entire equation in $$ ... $$ delimiters.\n"
    "- Be precise — check subscripts, superscripts, Greek letters, and operators.\n"
    "- Respond with LaTeX only, no explanation.\n"
    "- If unreadable: $$\\text{UNREADABLE}$$"
)


def _postprocess_latex_legacy(latex: str) -> str:
    """Fix common VLM output issues in a LaTeX string (legacy Qwen path).

    * Label bleed — removes ``Eq. 12.4.3`` captured from the margin label.
    * Unescaped sqrt — ``sqrt(x)`` → ``\\sqrt{x}``.
    * Missing function backslashes — bare ``sin``, ``cos``, etc.
    * Stray wrapper ``$`` characters.
    """
    if not latex:
        return latex
    latex = _LABEL_BLEED_RE.sub("", latex)
    latex = re.sub(r"(?<!\\)\bsqrt\s*\(([^)]+)\)", r"\\sqrt{\1}", latex)
    for _fn in ("sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim", "max", "min"):
        latex = re.sub(rf"(?<!\\)\b{_fn}\b", rf"\\{_fn}", latex)
    return latex.strip().strip("$").strip()


def _parse_latex_legacy(raw: str) -> str:
    """Extract LaTeX from model response (legacy Qwen path)."""
    m = _LATEX_BLOCK_RE.search(raw)
    if m:
        return m.group(1).strip()
    m = _LATEX_INLINE_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _estimate_confidence_legacy(
    latex: str, category: str = "mathematical_equation"
) -> tuple[float, list[str]]:
    """Return (confidence, flags) using score_recognition heuristics."""
    score, notes = score_recognition(latex=latex, plain_text=latex, category=category)
    flags = ["USING_PROXY_CONFIDENCE"] + [
        f"quality:{n}" if not n.startswith("quality:") else n for n in notes
    ]
    return score, flags


def recognize_equation(
    crop_image: Image.Image,
    *,
    retry: bool = False,
    provider_label: str | None = None,
) -> OcrResult:
    """Send a crop image to Qwen VL via Ollama and return the LaTeX OCR result.

    Parameters
    ----------
    crop_image:
        PIL image of the cropped equation region.
    retry:
        When True, applies higher zoom and a stricter prompt (second-pass).
    provider_label:
        Override the provider string in the result (defaults to the configured model name).

    Returns
    -------
    OcrResult
        Parsed LaTeX, estimated confidence, flags, and provider.
    """
    zoom = config.RECOGNITION_RETRY_ZOOM if retry else 1.0
    prompt = _RETRY_PROMPT if retry else _EXTRACTION_PROMPT

    image_b64 = _encode_pil_image(crop_image, zoom=zoom)

    try:
        raw = _call_ollama_generate(prompt, image_b64)
    except Exception as exc:
        return OcrResult(
            latex="",
            confidence=0.0,
            provider=provider_label or config.OLLAMA_VL_MODEL,
            flags=["OLLAMA_ERROR", str(exc)[:80]],
        )

    latex = _postprocess_latex_legacy(_parse_latex_legacy(raw))
    confidence, flags = _estimate_confidence_legacy(latex)

    if retry:
        flags.append("RETRY_PASS")

    return OcrResult(
        latex=latex,
        confidence=confidence,
        provider=provider_label or config.OLLAMA_VL_MODEL,
        flags=flags,
        raw_response=raw,
    )


def recognize_with_retry(crop_image: Image.Image) -> tuple[OcrResult, OcrResult | None]:
    """Run first-pass OCR; retry if confidence is below threshold.

    Returns
    -------
    (first_pass, retry_pass)
        retry_pass is None when the first pass was confident enough.
    """
    first = recognize_equation(crop_image, retry=False)
    if first.confidence >= config.RECOGNITION_RETRY_THRESHOLD:
        return first, None

    logger.debug(
        "low confidence %.3f < %.3f — retrying",
        first.confidence,
        config.RECOGNITION_RETRY_THRESHOLD,
    )
    retry_result = recognize_equation(crop_image, retry=True)
    return first, retry_result
