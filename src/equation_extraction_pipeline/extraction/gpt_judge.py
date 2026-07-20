"""External GPT vision judge (via the Elsevier Portkey gateway).

This module is the *authoritative* verification layer for the pipeline. Unlike the
legacy ``judge_latex`` (which asks the same local Qwen model to grade its own output),
an independent GPT vision model inspects the evidence and returns a grounded score.

It judges at three levels:

1. ``judge_equation(crop_image, representation, category)`` — is the *crop* a clean,
   complete single equation, and does the recognized *representation* (LaTeX / SMILES /
   reaction notation, per category) faithfully match it? Returns a :class:`JudgeVerdict`
   whose ``ai_score`` becomes the equation's authoritative confidence.
2. ``judge_page(page_image, extracted_on_page)`` — the per-page completeness audit:
   given the full page image and everything we extracted from it, what did we MISS?
3. ``summarize_document(page_verdicts, equation_verdicts)`` — a document-level roll-up
   (``ai_score`` + ``completeness``) for ``document.json``.

The model / endpoint / key come from the ``KNOVEL_PORTKEY_*`` settings, which point at
an OpenAI-compatible ``/chat/completions`` endpoint. If Portkey is not configured the
judge degrades gracefully (marks results "unavailable" without blocking the run).

Strict rubric: the prompts instruct the judge to REJECT when uncertain — the opposite of
the legacy judge's accept-on-doubt default.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from equation_extraction_pipeline.config import settings as config
from equation_extraction_pipeline.domain.models import JudgeVerdict

logger = logging.getLogger(__name__)

__all__ = [
    "gpt_judge_available",
    "judge_equation",
    "judge_page",
    "extract_page_equations",
    "PageCompletenessVerdict",
    "summarize_document",
]

# Category → short description of the representation the judge should expect, so the
# judge validates SMILES/reactions for chemistry rather than assuming LaTeX everywhere.
_REPRESENTATION_BY_CATEGORY: dict[str, str] = {
    "mathematical_equation": "LaTeX",
    "engineering_formula": "LaTeX (may include units such as [Pa] or [m/s^2])",
    "statistical_expression": "LaTeX",
    "chemical_equation": "chemical reaction notation (→ / ⇌, species, state labels)",
    "chemical_structure": "a SMILES string",
    "unknown": "LaTeX if mathematical, or SMILES / reaction notation if chemical",
}

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Simple in-process cache so identical (crop, representation) pairs are judged once.
_VERDICT_CACHE: dict[str, JudgeVerdict] = {}


def gpt_judge_available() -> bool:
    """True when the Portkey gateway is configured (base URL + model present)."""
    return bool(
        getattr(config, "KNOVEL_PORTKEY_BASE_URL", "")
        and getattr(config, "KNOVEL_PORTKEY_MODEL", "")
    )


# ---------------------------------------------------------------------------
# Low-level HTTP + encoding helpers
# ---------------------------------------------------------------------------

def _encode_image(image: Any) -> str:
    """Return a base64 PNG string for a PIL image."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _image_content(image: Any) -> dict[str, Any]:
    """Build an OpenAI-style image_url content part from a PIL image."""
    b64 = _encode_image(image)
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def _post_chat(messages: list[dict[str, Any]], *, max_tokens: int) -> str:
    """POST to the Portkey OpenAI-compatible endpoint; return the assistant content.

    Retries transient failures with a short backoff. Raises on unrecoverable errors.
    """
    base_url = getattr(config, "KNOVEL_PORTKEY_BASE_URL", "").rstrip("/")
    api_key = getattr(config, "KNOVEL_PORTKEY_API_KEY", "")
    model = getattr(config, "KNOVEL_PORTKEY_MODEL", "")
    timeout = float(getattr(config, "KNOVEL_LLM_TIMEOUT", 120))
    max_retries = int(getattr(config, "KNOVEL_LLM_MAX_RETRIES", 3))

    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        # Send both header styles: Portkey accepts x-portkey-api-key and, on the
        # OpenAI-compatible route, an Authorization bearer token.
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-portkey-api-key"] = api_key

    # GPT-5-family models (Azure OpenAI) require ``max_completion_tokens``; older models
    # accept only ``max_tokens``. Start with the modern name and fall back on a 400 that
    # says it is unsupported, so the client works across model generations.
    token_param = "max_completion_tokens"

    def _payload() -> dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            token_param: max_tokens,
            "temperature": 0,
        }

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=_payload(), timeout=timeout)
            if resp.status_code == 400 and token_param == "max_completion_tokens" and (
                "max_completion_tokens" in resp.text or "max_tokens" in resp.text
            ):
                token_param = "max_tokens"
                resp = requests.post(url, headers=headers, json=_payload(), timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 — bounded retry over any transient failure
            last_exc = exc
            logger.warning(
                "gpt_judge_request_failed attempt=%d/%d error=%s", attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"GPT judge request failed after {max_retries} attempts: {last_exc}")


def _normalize_backslashes(s: str) -> str:
    r"""Normalise lone backslashes so LaTeX embedded in JSON string values parses correctly.

    Models emit LaTeX (``\frac``, ``\left``, ``\underline``) inside JSON string values with
    single backslashes, which is invalid JSON. Worse, ``\f`` / ``\b`` / ``\n`` / ``\r`` / ``\t``
    are *valid* JSON escapes, so ``\frac`` would silently decode to a formfeed + "rac" without
    error. This doubles every backslash EXCEPT those beginning ``\"`` / ``\\`` / ``\/`` (kept so
    escaped quotes and string structure survive, and already-doubled pairs pass through as one
    unit). The result is valid JSON whose string values hold the intended LaTeX.
    """
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt in '"\\/':
                out.append(c)
                out.append(nxt)
                i += 2
                continue
            out.append("\\\\")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _parse_json(raw: str) -> dict[str, Any]:
    r"""Extract and parse the first JSON object in ``raw`` (tolerates markdown fences).

    Robust to LaTeX in string values (``"\frac{a}{b}"``): a backslash-normalisation pass runs
    first (see :func:`_normalize_backslashes`), with a fallback to the raw text so any already
    well-formed JSON still parses.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text).strip()

    m = _JSON_RE.search(text)
    candidate = m.group(0) if m else text
    for attempt in (_normalize_backslashes(candidate), candidate):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return json.loads(candidate)


# ---------------------------------------------------------------------------
# 1. Equation-level judge
# ---------------------------------------------------------------------------

_EQUATION_SYSTEM = (
    "You are a strict, independent quality inspector for an equation-extraction pipeline. "
    "You are shown a cropped image of one printed equation (or one labeled derivation group) "
    "and a candidate machine transcription. "
    "Your job is to judge, WITHOUT being lenient, whether (a) the crop is a clean, complete "
    "crop of that equation or group and (b) the transcription faithfully matches it, symbol "
    "by symbol. "
    "If you are uncertain, REJECT. Return ONLY a JSON object, no prose, no markdown fences."
)

_EQUATION_USER_TEMPLATE = (
    "The expected representation for this content is: {expected}.\n\n"
    "Candidate transcription:\n{representation}\n\n"
    "Check carefully:\n"
    "1. CROP: is it exactly one complete equation or one labeled equation group? Penalise "
    "if it is clipped (missing a left/right side), contains paragraph prose, or contains "
    "equations belonging to OTHER labels. Three things are EXPECTED in a correct crop and are "
    "NOT defects: (a) a printed equation-reference label at the margin, such as '(2-7)' or "
    "'Eq. 5.1.2'; (b) a labeled derivation group — several stacked math lines under ONE "
    "printed label, possibly joined by short connective words like 'or', 'since', 'where', "
    "'and thus'; (c) a partially-visible line of neighbouring text cut through by the top or "
    "bottom border — that is a padding artifact, penalise only if the LABELED equation itself "
    "is cut.\n"
    "2. TRANSCRIPTION: does every symbol, subscript, superscript, operator, fraction and "
    "bracket match the image? For a derivation group, every math line must be transcribed. "
    "Account for equivalent notations. The reference label is not part of the equation: the "
    "transcription may omit it or render it as \\tag{{...}} — neither is an error.\n\n"
    "Return JSON with EXACTLY these keys:\n"
    "{{\n"
    '  "crop_valid": true|false,\n'
    '  "crop_issues": [string],\n'
    '  "representation_correct": true|false,\n'
    '  "corrected_representation": string|null,\n'
    '  "ai_score": 0.0-1.0,\n'
    '  "reason": "one sentence"\n'
    "}}"
)


def _cache_key(image: Any, representation: str, category: str) -> str:
    h = hashlib.sha256()
    h.update(_encode_image(image).encode("ascii"))
    h.update(b"\x00")
    h.update(representation.encode("utf-8"))
    h.update(b"\x00")
    h.update(category.encode("utf-8"))
    return h.hexdigest()


def _unavailable_verdict(reason: str) -> JudgeVerdict:
    """Verdict used when the judge cannot run — never blocks the pipeline."""
    return JudgeVerdict(
        accepted=True,
        score=0.5,
        reason=reason,
        ai_score=None,
        issues=["judge_unavailable"],
        judge_model=getattr(config, "KNOVEL_PORTKEY_MODEL", ""),
    )


def judge_equation(crop_image: Any, representation: str, category: str = "unknown") -> JudgeVerdict:
    """Judge one equation crop + its recognized representation via the external GPT model.

    Returns a :class:`JudgeVerdict` whose ``ai_score`` is the authoritative confidence.
    Degrades to an "unavailable" (non-blocking) verdict if Portkey is unconfigured or errors.
    """
    if not (representation or "").strip():
        return JudgeVerdict(
            accepted=False,
            score=0.0,
            reason="empty transcription",
            ai_score=0.0,
            crop_valid=None,
            representation_correct=False,
            issues=["empty_representation"],
            judge_model=getattr(config, "KNOVEL_PORTKEY_MODEL", ""),
        )

    if not gpt_judge_available():
        return _unavailable_verdict("GPT judge not configured (KNOVEL_PORTKEY_* unset)")

    key = _cache_key(crop_image, representation, category)
    cached = _VERDICT_CACHE.get(key)
    if cached is not None:
        return cached

    expected = _REPRESENTATION_BY_CATEGORY.get(category, _REPRESENTATION_BY_CATEGORY["unknown"])
    user_text = _EQUATION_USER_TEMPLATE.format(expected=expected, representation=representation)
    messages = [
        {"role": "system", "content": _EQUATION_SYSTEM},
        {
            "role": "user",
            "content": [_image_content(crop_image), {"type": "text", "text": user_text}],
        },
    ]

    try:
        raw = _post_chat(messages, max_tokens=int(getattr(config, "JUDGE_MAX_TOKENS", 1024)))
        data = _parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("gpt_judge_equation_failed error=%s", exc)
        return _unavailable_verdict(f"GPT judge error: {str(exc)[:120]}")

    ai_score = _clamp01(data.get("ai_score", 0.0))
    accept_threshold = float(getattr(config, "JUDGE_ACCEPT_THRESHOLD", 0.70))
    crop_valid = bool(data.get("crop_valid", False))
    representation_correct = bool(data.get("representation_correct", False))
    # Acceptance keeps transcription in the decision, but via the *continuous* ai_score rather
    # than the binary representation_correct flag: the crop must be a clean, complete equation
    # (the detect-and-crop goal) AND the fidelity score must clear the threshold. This stops a
    # single minor OCR slip on an otherwise-correct crop (e.g. 0.414 read as 0.4l4, or a spacing
    # difference at ai_score 0.88) from hard-rejecting the equation. representation_correct is
    # still recorded below as an advisory signal for human review.
    accepted = crop_valid and ai_score >= accept_threshold
    issues = list(data.get("crop_issues") or [])

    verdict = JudgeVerdict(
        accepted=accepted,
        score=ai_score,
        reason=str(data.get("reason", ""))[:300],
        ai_score=ai_score,
        crop_valid=crop_valid,
        representation_correct=representation_correct,
        corrected_representation=data.get("corrected_representation") or None,
        issues=issues,
        judge_model=getattr(config, "KNOVEL_PORTKEY_MODEL", ""),
    )
    _VERDICT_CACHE[key] = verdict
    logger.debug(
        "gpt_judge_equation accepted=%s ai_score=%.3f crop_valid=%s repr_ok=%s",
        accepted, ai_score, crop_valid, representation_correct,
    )
    return verdict


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# 2. Page-level completeness judge
# ---------------------------------------------------------------------------

@dataclass
class PageCompletenessVerdict:
    """Result of the per-page 'did we extract all equations?' audit."""

    page_number: int
    all_equations_extracted: bool
    missed_equations: list[dict[str, Any]] = field(default_factory=list)
    spurious_regions: list[str] = field(default_factory=list)
    page_ai_score: float = 0.0
    reason: str = ""
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "all_equations_extracted": self.all_equations_extracted,
            "missed_equations": self.missed_equations,
            "spurious_regions": self.spurious_regions,
            "page_ai_score": round(self.page_ai_score, 4),
            "reason": self.reason,
            "available": self.available,
        }


# Unlabeled documents: capture EVERY standalone equation on the page.
_PAGE_SYSTEM_UNLABELED = (
    "You are a strict auditor checking whether an equation-extraction pipeline found EVERY "
    "equation on a document page. You are shown the full page image and a JSON list of the "
    "equations the pipeline already extracted (with their approximate bounding boxes). "
    "Identify any displayed equation, formula, or chemical structure on the page that is NOT "
    "already in the extracted list. Ignore inline symbols inside running prose. If uncertain "
    "whether something is a genuine standalone equation, include it as missed. "
    "Return ONLY a JSON object, no prose, no markdown fences."
)

# Labeled documents: scope is LABELED equations only. Only a numbered/labeled equation that
# is absent from the extracted list counts as missed — unlabeled displayed equations are out
# of scope by design and must NOT be reported.
_PAGE_SYSTEM_LABELED = (
    "You are a strict auditor for an equation-extraction pipeline that, for this document, "
    "only extracts equations carrying a printed REFERENCE LABEL such as 'Eq. 12.2.1', "
    "'(2-7)', or '(5.5.11)'. You are shown the full page image and a JSON list of the labeled "
    "equations already extracted.\n"
    "Report an equation as MISSED only if ALL of the following hold:\n"
    "  1. It has a reference label that is ACTUALLY PRINTED and legible on the page image "
    "(usually in the right margin) — you can read the exact label text.\n"
    "  2. That exact label is NOT already in the extracted list.\n"
    "CRITICAL RULES:\n"
    "  - Do NOT infer, guess, or extrapolate a label. If Eq. 12.4.23 is the last visible "
    "label, do NOT invent 'Eq. 12.4.24' for a nearby unlabeled equation. Only report a label "
    "you can literally see printed.\n"
    "  - Procedural/step markers like '(f4)', '(iii)', '(a)', '(1)', 'Step 3', or bullet "
    "markers are NOT equation reference labels. Equations marked only by these, or with no "
    "label at all, are OUT OF SCOPE — never report them.\n"
    "  - In each missed entry's 'note', quote the EXACT visible label text you read from the "
    "image.\n"
    "Return ONLY a JSON object, no prose, no markdown fences."
)

_PAGE_USER_TEMPLATE = (
    "Equations already extracted from this page (JSON):\n{extracted}\n\n"
    "Return JSON with EXACTLY these keys:\n"
    "{{\n"
    '  "all_equations_extracted": true|false,\n'
    '  "missed_equations": [{{"approx_bbox": [x0,y0,x1,y1]|null, '
    '"latex_if_readable": string|null, "note": string}}],\n'
    '  "spurious_regions": [equation_id, ...],\n'
    '  "page_ai_score": 0.0-1.0,\n'
    '  "reason": "one sentence"\n'
    "}}"
)


def judge_page(
    page_image: Any,
    extracted_on_page: list[dict[str, Any]],
    *,
    page_number: int,
    mode: str = "unlabeled",
) -> PageCompletenessVerdict:
    """Audit a page for missed equations via the external GPT model.

    ``mode`` selects the scope of the audit:
      * ``"labeled"``   — the document extracts only reference-labeled equations, so only a
        missed *labeled* equation counts (unlabeled displayed equations are out of scope).
      * ``"unlabeled"`` — the document extracts everything, so any missed equation counts.
    """
    if not gpt_judge_available():
        return PageCompletenessVerdict(
            page_number=page_number,
            all_equations_extracted=True,
            reason="GPT judge not configured",
            available=False,
        )

    system = _PAGE_SYSTEM_LABELED if mode == "labeled" else _PAGE_SYSTEM_UNLABELED
    extracted_json = json.dumps(extracted_on_page, ensure_ascii=False)
    user_text = _PAGE_USER_TEMPLATE.format(extracted=extracted_json)
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [_image_content(page_image), {"type": "text", "text": user_text}],
        },
    ]

    try:
        raw = _post_chat(messages, max_tokens=int(getattr(config, "JUDGE_MAX_TOKENS", 1024)))
        data = _parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("gpt_judge_page_failed page=%d error=%s", page_number, exc)
        return PageCompletenessVerdict(
            page_number=page_number,
            all_equations_extracted=True,
            reason=f"GPT judge error: {str(exc)[:120]}",
            available=False,
        )

    missed = [m for m in (data.get("missed_equations") or []) if isinstance(m, dict)]
    verdict = PageCompletenessVerdict(
        page_number=page_number,
        all_equations_extracted=bool(data.get("all_equations_extracted", not missed)),
        missed_equations=missed,
        spurious_regions=[str(s) for s in (data.get("spurious_regions") or [])],
        page_ai_score=_clamp01(data.get("page_ai_score", 0.0)),
        reason=str(data.get("reason", ""))[:300],
        available=True,
    )
    logger.info(
        "gpt_judge_page page=%d complete=%s missed=%d",
        page_number, verdict.all_equations_extracted, len(missed),
    )
    return verdict


# ---------------------------------------------------------------------------
# 2b. Page-level equation extraction (image-based math recovery)
# ---------------------------------------------------------------------------
# For pages whose equations are embedded raster images, the text-layer label scanner is
# blind and Docling is unreliable. The VLM reads the rendered page and enumerates every
# display equation directly — label (verbatim printed number), transcription, and a
# NORMALISED bounding box (fractions of page width/height, which VLMs place far more
# reliably than absolute pixels). The caller converts fractions → a crop.

_EXTRACT_SYSTEM = (
    "You are a precise equation extractor. You are shown the full image of ONE document page. "
    "Enumerate EVERY displayed/standalone mathematical equation on the page, in top-to-bottom "
    "reading order. Include equations rendered as images and unlabeled display equations. "
    "Do NOT include inline math inside running prose, section headings, page numbers, or "
    "figure/table content. Return ONLY a JSON object, no prose, no markdown fences."
)

_EXTRACT_USER_TEMPLATE = (
    "Return JSON with EXACTLY this shape:\n"
    '{\n'
    '  "equations": [\n'
    '    {\n'
    '      "label": <the equation\'s printed reference number exactly as shown, e.g. "1.30" '
    'or "(2-7)", or null if it has no printed number>,\n'
    '      "latex": <a faithful LaTeX transcription of the equation, no surrounding text and '
    'no \\tag>,\n'
    '      "bbox": [x0, y0, x1, y1]\n'
    '    }\n'
    '  ]\n'
    '}\n'
    "bbox coordinates are FRACTIONS of the page image in [0,1]: x is left-to-right, y is "
    "top-to-bottom, so [0,0,1,1] is the whole page. Make the box tightly enclose the equation "
    "(its whole width and height, including fractions/limits) but EXCLUDE the margin label. "
    "If the page has no displayed equation, return an empty list."
)


def extract_page_equations(
    page_image: Any,
    *,
    page_number: int,
    mode: str = "labeled",
) -> list[dict[str, Any]]:
    """VLM enumeration of every equation on a rendered page image.

    Returns a list of ``{"label": str|None, "latex": str, "bbox_frac": (x0,y0,x1,y1)}``
    with ``bbox_frac`` in [0,1] page fractions (top-left origin). Empty list when the judge
    is unconfigured, errors, or the page has no equation. ``mode`` is accepted for symmetry
    with :func:`judge_page` (the enumeration prompt is the same for labeled/unlabeled docs).
    """
    if not gpt_judge_available():
        logger.info("extract_page_equations_skipped page=%d reason=unconfigured", page_number)
        return []

    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {
            "role": "user",
            "content": [_image_content(page_image), {"type": "text", "text": _EXTRACT_USER_TEMPLATE}],
        },
    ]
    try:
        raw = _post_chat(
            messages, max_tokens=int(getattr(config, "IMAGE_MATH_MAX_TOKENS", 3000))
        )
        data = _parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("extract_page_equations_failed page=%d error=%s", page_number, exc)
        return []

    out: list[dict[str, Any]] = []
    for item in data.get("equations") or []:
        if not isinstance(item, dict):
            continue
        latex = str(item.get("latex") or "").strip()
        if not latex:
            continue
        bbox = item.get("bbox")
        frac: tuple[float, float, float, float] | None = None
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x0, y0, x1, y1 = (_clamp01(v) for v in bbox)
                if x1 > x0 and y1 > y0:
                    frac = (x0, y0, x1, y1)
            except (TypeError, ValueError):
                frac = None
        label = item.get("label")
        out.append({
            "label": str(label).strip() if label else None,
            "latex": latex,
            "bbox_frac": frac,
        })
    logger.info("extract_page_equations page=%d equations=%d", page_number, len(out))
    return out


# ---------------------------------------------------------------------------
# 3. Document roll-up
# ---------------------------------------------------------------------------

def summarize_document(
    page_verdicts: list[PageCompletenessVerdict],
    equation_ai_scores: list[float],
) -> dict[str, Any]:
    """Aggregate page + equation verdicts into a document-level completeness summary."""
    audited_pages = [v for v in page_verdicts if v.available]
    total_missed = sum(len(v.missed_equations) for v in audited_pages)
    incomplete_pages = [v.page_number for v in audited_pages if not v.all_equations_extracted]
    scored = [s for s in equation_ai_scores if s is not None]
    doc_ai_score = round(sum(scored) / len(scored), 4) if scored else None

    return {
        "complete": total_missed == 0 and not incomplete_pages,
        "document_ai_score": doc_ai_score,
        "pages_audited": len(audited_pages),
        "incomplete_pages": incomplete_pages,
        "total_missed_equations": total_missed,
        "page_audits": [v.to_dict() for v in page_verdicts],
    }
