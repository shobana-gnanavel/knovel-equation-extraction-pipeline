"""Interchangeable equation-recognition providers (feature 008, FR-010/FR-011/FR-013).

Every provider implements the :class:`EquationProvider` protocol and returns a
:class:`RecognitionResult`. The primary provider is :class:`QwenVLProvider`, which routes
equation region images to a Qwen2.5-VL (or Qwen2-VL) model served through an
OpenAI-compatible API — either **Ollama** (``http://localhost:11434/v1``) or **vLLM**
(``http://localhost:8000/v1``). It handles all equation categories with category-specific
prompts and degrades gracefully when the API server is unreachable. The
:class:`GenericProvider` is a pure text passthrough used as a last-resort fallback.

Quick-start (Ollama)::

    ollama pull qwen2.5vl:7b
    # Set in .env: KNOVEL_OLLAMA_BASE_URL=http://localhost:11434
    #              KNOVEL_EQUATION_VL_MODEL=qwen2.5vl:7b

Quick-start (vLLM with 4-bit AWQ)::

    vllm serve Qwen/Qwen2-VL-7B-Instruct-AWQ --quantization awq --port 8000
    # Set in .env: KNOVEL_OLLAMA_BASE_URL=http://localhost:8000
    #              KNOVEL_EQUATION_VL_MODEL=Qwen/Qwen2-VL-7B-Instruct-AWQ
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import structlog

from equation_extraction.recognition_quality import score_recognition

# ---------------------------------------------------------------------------
# Knovel-specific OCR character-substitution table.
# Many Knovel PDFs use non-standard font encodings that map common math
# symbols to unexpected code points.  Apply these before sending OCR text
# to the VLM so the model sees readable input rather than artefacts.
# ---------------------------------------------------------------------------
_OCR_FIX: list[tuple[re.Pattern[str], str]] = [
    # U+00BC (¼) is used as the equals sign in Knovel engineering font encodings.
    (re.compile(r"¼"), "="),
    # (cid:2) maps to × in this document family; drop all other unmapped CID placeholders.
    (re.compile(r"\(cid:2\)"), "×"),
    (re.compile(r"\(cid:\d+\)"), ""),
    # ﬃ/ﬀ/ﬁ/ﬂ ligature runs (3+ chars) represent the square-root radical √.
    (re.compile(r"[ﬃﬀﬁﬂ]{3,}"), "√"),
    # ð / Þ used as bracket substitutes in some font encodings.
    (re.compile(r"ð"), "("),
    (re.compile(r"Þ"), ")"),
    # l / I confused with digit 1 in numeric contexts.
    (re.compile(r"\b([lI])\.(\d)"), r"1.\2"),          # l.15 → 1.15
    (re.compile(r"(?<=[\d.])[lI](?=[\d])"), "1"),      # 1l5  → 115
    (re.compile(r"(?<=[=(\[{+\-*/\s])([lI])(?=[.\d])"), "1"),  # =l.5 → =1.5
    # Multi-line separator artifact from two-column PDF block merging.
    (re.compile(r"\s*;\s*"), " "),
]

# Equation label pattern used to strip "Eq. 12.4.1" from OCR text and LaTeX output.
_EQ_LABEL_IN_TEXT_RE = re.compile(r"\bEq(?:uation)?[.:]?\s*\d[\d.]*", re.IGNORECASE)


def _apply_ocr_fixes(text: str) -> str:
    """Apply Knovel-specific OCR character substitutions to raw PDF text.

    Call before sending OCR text to the VLM so the model receives readable
    input rather than encoding artefacts (¼ = equals, CID placeholders, etc.).
    Also strips embedded equation-label cross-references (``Eq. 12.4.1``)
    that are part of the surrounding paragraph, not the formula itself.
    """
    if not text:
        return text
    t = text.replace("\n", " ")
    t = _EQ_LABEL_IN_TEXT_RE.sub("", t)
    for pat, repl in _OCR_FIX:
        t = pat.sub(repl, t)
    return re.sub(r"\s+", " ", t).strip()


# Matches an equation-label reference that bled into the VLM-produced LaTeX
# (e.g. "Eq. 12.4.3" appended because it appeared at the right margin of the crop).
_LABEL_BLEED_RE = re.compile(r"\bEq(?:uation)?[.:]?\s*\d[\d.]*", re.IGNORECASE)


def _postprocess_latex(latex: str) -> str:
    """Fix common VLM output issues in a LaTeX string.

    Applied after ``_clean_latex`` strips delimiters/fences.  Handles:

    * **Label bleed** — ``Eq. 12.4.3`` captured from the right-margin label into the
      LaTeX because the crop included the label column.
    * **Unescaped sqrt** — ``sqrt(x)`` → ``\\sqrt{x}``.
    * **Missing function backslashes** — bare ``sin``, ``cos``, ``log``, etc.
      → ``\\sin``, ``\\cos``, ``\\log``.
    * **Stray wrapper characters** — leading/trailing ``$`` left after ``_clean_latex``.
    """
    if not latex:
        return latex
    latex = _LABEL_BLEED_RE.sub("", latex)
    # sqrt(x) → \sqrt{x}
    latex = re.sub(r"(?<!\\)\bsqrt\s*\(([^)]+)\)", r"\\sqrt{\1}", latex)
    # Add missing backslash on common math function names.
    for _fn in ("sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim", "max", "min"):
        latex = re.sub(rf"(?<!\\)\b{_fn}\b", rf"\\{_fn}", latex)
    return latex.strip().strip("$").strip()


# LLM responses that begin with natural-language prose are explanatory refusals, not equations.
# Matching is case-insensitive on the first word/phrase of the stripped response.
_PROSE_RESPONSE_RE = re.compile(
    r"^(nothing\b"  # VLM "nothing found" / "nothing to transcribe" (follows system-prompt instruction)
    r"|it\s|the\s+(?:expression|equation|instruction|formula|given|notation|request|text|image)"
    r"|i\s|certainly|please|note\s|you\s|based\s|as\s+provided|this\s+(?:expression|equation|notation|text)"
    r"|there\s+(?:is|are|was|were|might|seem)|sorry|unfortunately)",
    re.IGNORECASE,
)

# LaTeX structural/document commands that are never valid equation content.
# These appear when the VL model receives a non-equation image region and hallucinates
# a LaTeX document snippet instead of a mathematical expression.
_LATEX_STRUCTURAL_CMD_RE = re.compile(
    r"^\\(?:section|subsection|subsubsection|chapter|part|paragraph|subparagraph"
    r"|newcommand|renewcommand|def|let|providecommand"
    r"|begin\{document\}|end\{document\}|documentclass|usepackage|maketitle"
    r"|label|ref|cite|bibliography|bibitem|footnote|caption)\b",
    re.IGNORECASE,
)

__all__ = [
    "RecognitionResult",
    "EquationProvider",
    "QwenVLProvider",
    "GenericProvider",
    "_apply_ocr_fixes",
    "_postprocess_latex",
]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# System prompt — sets a stable expert role for every Qwen2.5-VL call.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT: str = (
    # Role
    "You are an expert mathematical typesetter specialising in engineering and scientific literature, "
    "including thermodynamics, fluid mechanics, structural analysis, and chemical engineering. "
    # Core fidelity rule
    "Transcribe equations exactly as they appear in the source — do not simplify, rearrange, reorder terms, "
    "or silently correct what appears to be a typographic error; transcribe an apparent error faithfully. "
    # Output hygiene — matches the raw-string parser (no JSON, no notes field on this path).
    "Output only the requested format: the bare expression and nothing else. "
    "Do not wrap output in markdown fences, do not add prose explanations, "
    "and do not enclose LaTeX in dollar signs or \\begin{equation} blocks. "
    # Symbol resolution — a decision rule instead of a silent guess.
    "When a symbol is ambiguous due to image quality or typographic similarity "
    "(e.g. η vs n, ∂ vs δ, × vs ·), choose the most contextually appropriate symbol. "
    # Failure mode
    "If no equation or formula is present, return nothing — never fabricate or infer an equation "
    "that is not visually present."
)

# ---------------------------------------------------------------------------
# Shared math prompt — covers mathematical_equation, engineering_formula, and
# statistical_expression. All three need LaTeX output with identical constraints;
# splitting them produced no measurable quality difference but added complexity.
# ---------------------------------------------------------------------------
_MATH_IMAGE_PROMPT: str = (
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

# Stricter variant used on the confidence-gated retry (padded, higher-resolution crop). Adds
# explicit guards against the two dominant first-pass failures: a clipped left-hand side and
# two stacked equations merged into one transcription.
_MATH_IMAGE_PROMPT_STRICT: str = (
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

_MATH_TEXT_PROMPT: str = (
    "Convert this equation or formula to LaTeX. "
    "Note: some characters may be encoding artifacts "
    "(¼ means =, (cid:N) means an unmapped symbol). "
    "Output only the LaTeX — no dollar signs, no fences.\n"
    "Equation: {text}"
)

# ---------------------------------------------------------------------------
# Category-specific prompts sent to Qwen2.5-VL when a region image is available.
# Math categories share _MATH_IMAGE_PROMPT; chemical categories differ in output format.
# ---------------------------------------------------------------------------
_IMAGE_PROMPTS: dict[str, str] = {
    "mathematical_equation": _MATH_IMAGE_PROMPT,
    "engineering_formula": _MATH_IMAGE_PROMPT,
    "statistical_expression": _MATH_IMAGE_PROMPT,
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

# Stricter image prompts for the retry pass. Only the math categories have a dedicated strict
# variant (the observed failures are all math LHS-clipping / equation-merging); other categories
# fall back to their normal prompt.
_IMAGE_PROMPTS_STRICT: dict[str, str] = {
    "mathematical_equation": _MATH_IMAGE_PROMPT_STRICT,
    "engineering_formula": _MATH_IMAGE_PROMPT_STRICT,
    "statistical_expression": _MATH_IMAGE_PROMPT_STRICT,
}

# Text-only prompts used for inline equations (no region image available).
# Math categories share _MATH_TEXT_PROMPT.
_TEXT_PROMPTS: dict[str, str] = {
    "mathematical_equation": _MATH_TEXT_PROMPT,
    "engineering_formula": _MATH_TEXT_PROMPT,
    "statistical_expression": _MATH_TEXT_PROMPT,
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


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RecognitionResult:
    """The structured output of one provider for one equation (FR-013)."""

    plain_text: str = ""
    latex: str | None = None
    mathml: str | None = None
    structured_form: str | None = None
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


@runtime_checkable
class EquationProvider(Protocol):
    """Common recognition interface; concrete providers are selected by configuration (FR-011)."""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Display-only math environments the VLM wraps output in despite the prompt forbidding it.
# These add no semantic content; strip the begin/end pair. Structural environments
# (align/matrix/cases/array) are preserved — they carry layout meaning.
_MATH_ENV_WRAPPER_RE = re.compile(
    r"\\(?:begin|end)\{(?:equation|displaymath|math|gather|multline)\*?\}",
    re.IGNORECASE,
)


def _clean_latex(text: str) -> str:
    """Strip markdown fences, math delimiters, and display-environment wrappers.

    Qwen2.5-VL frequently ignores the "no delimiters / no environment" instruction and returns
    ``\\begin{equation} … \\end{equation}`` or orphaned ``\\[ … \\]`` blocks (sometimes several
    in one response — definition form plus substituted form). Strip all of them so the stored
    LaTeX is the bare expression.
    """
    text = text.strip()
    # Markdown code fences: ```latex ... ``` or ``` ... ```
    text = re.sub(r"^```(?:latex)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text)
    # Display-only environment wrappers, anywhere in the response.
    text = _MATH_ENV_WRAPPER_RE.sub("", text)
    # Display delimiters: $$...$$ or \[...\]
    text = re.sub(r"^\$\$\s*", "", text)
    text = re.sub(r"\s*\$\$$", "", text)
    text = re.sub(r"^\\\[\s*", "", text)
    text = re.sub(r"\s*\\\]$", "", text)
    # Inline math: $...$
    if text.startswith("$") and text.endswith("$") and len(text) > 2:
        text = text[1:-1]
    # Collapse orphaned display/inline delimiters left when the VLM returned multiple
    # blocks in one response (e.g. definition form + i.e. substituted form).
    text = re.sub(r"\s*\\[\[\]()]\s*", "\n", text)
    # Drop a trailing partial environment opener left when the response was cut off at
    # max_tokens mid-second-block (e.g. "…\\end{equation}\n\n\\be").
    text = re.sub(r"\n*\\beg?i?n?\{?[a-z]*\*?\}?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _image_to_b64(image: Any) -> str:
    """Encode a PIL Image to a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class QwenVLProvider:
    """Equation recognition via Qwen2.5-VL through an OpenAI-compatible API (Ollama or vLLM).

    Handles all equation categories with category-specific prompts. Sends the equation
    region image when available; falls back to a text-only prompt for inline equations.
    Degrades gracefully (returns plain text with a note) when the API server is unreachable.

    Configuration (read from the pipeline config object at recognize-time):

    * ``KNOVEL_OLLAMA_BASE_URL``      — API server root, e.g. ``http://localhost:11434``
    * ``KNOVEL_EQUATION_VL_MODEL``    — model name, e.g. ``qwen2.5vl:7b`` (Ollama) or
                                        ``Qwen/Qwen2-VL-7B-Instruct-AWQ`` (vLLM)
    * ``KNOVEL_EQUATION_VL_TIMEOUT``  — per-request timeout in seconds (default 60)
    * ``KNOVEL_EQUATION_VL_MAX_TOKENS`` — maximum response tokens (default 512)
    """

    name = "qwen_vl"
    backend = "qwen2.5-vl"
    available = True  # verified lazily on first request

    def __init__(self) -> None:
        # One pooled HTTP client is reused across every equation in a document run.
        # A book can have hundreds of equations; opening a fresh connection per call
        # paid a TCP+TLS handshake every time. The client is created lazily on first
        # recognize() so an import-guarded httpx never blocks provider construction,
        # and closed via close() at the end of the run.
        self._client: Any = None

    def _get_client(self, httpx_mod: Any) -> Any:
        """Return the pooled ``httpx.Client``, creating it (with keep-alive) on first use."""
        if self._client is None:
            self._client = httpx_mod.Client(
                headers={"Content-Type": "application/json"},
                limits=httpx_mod.Limits(max_keepalive_connections=4, max_connections=8),
            )
        return self._client

    def close(self) -> None:
        """Close the pooled HTTP client. Safe to call multiple times / when never opened."""
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
            import httpx
        except ImportError:  # pragma: no cover — httpx is in requirements.txt
            return RecognitionResult(
                plain_text=region_text, confidence=0.0, notes=["provider_absent:httpx"]
            )

        base_url = getattr(config, "KNOVEL_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        # KNOVEL_EQUATION_VL_MODEL overrides the global fast model for equation work.
        model = getattr(
            config,
            "KNOVEL_EQUATION_VL_MODEL",
            getattr(config, "KNOVEL_OLLAMA_FAST_MODEL", "qwen2.5vl:7b"),
        )
        timeout = float(getattr(config, "KNOVEL_EQUATION_VL_TIMEOUT", 60))
        max_tokens = int(getattr(config, "KNOVEL_EQUATION_VL_MAX_TOKENS", 512))

        messages = self._build_messages(region_image, region_text, category, strict=strict)

        try:
            client = self._get_client(httpx)
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
        except httpx.ConnectError:
            logger.warning("qwen_vl_server_unreachable", base_url=base_url, model=model)
            return RecognitionResult(
                plain_text=region_text,
                confidence=0.0,
                notes=["provider_absent:qwen_vl"],
            )
        except Exception as exc:
            logger.warning("qwen_vl_request_failed", error=str(exc))
            return RecognitionResult(
                plain_text=region_text,
                confidence=0.0,
                notes=[f"recognition_failed:{type(exc).__name__}"],
            )

        return self._parse_response(raw, region_text, category)

    def is_equation(self, text: str, *, config: Any) -> bool:
        """Yes/no oracle: is *text* a mathematical/engineering formula? (ambiguous-block resolver).

        Used by the detection cascade to promote borderline text blocks. Reuses the pooled
        client and the configured VL model. Conservative: returns ``False`` on any error,
        timeout, empty input, or unreachable server so an outage never fabricates equations.
        """
        stripped = (text or "").strip()
        if not stripped:
            return False
        try:
            import httpx
        except ImportError:  # pragma: no cover — httpx is in requirements.txt
            return False

        base_url = getattr(config, "KNOVEL_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = getattr(
            config,
            "KNOVEL_EQUATION_VL_MODEL",
            getattr(config, "KNOVEL_OLLAMA_FAST_MODEL", "qwen2.5vl:7b"),
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
            client = self._get_client(httpx)
            resp = client.post(
                f"{base_url}/v1/chat/completions",
                json={"model": model, "messages": messages, "max_tokens": 5},
                timeout=timeout,
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
        except Exception as exc:
            logger.debug("qwen_vl_is_equation_failed", error=str(exc))
            return False
        return answer.startswith("yes")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self, region_image: Any, region_text: str, category: str, strict: bool = False
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if region_image is not None:
            if strict:
                prompt = _IMAGE_PROMPTS_STRICT.get(
                    category, _IMAGE_PROMPTS.get(category, _IMAGE_PROMPTS["unknown"])
                )
            else:
                prompt = _IMAGE_PROMPTS.get(category, _IMAGE_PROMPTS["unknown"])
            # Append OCR text as a verification anchor when available. Apply
            # Knovel OCR fixes first so the hint is readable (¼→=, CID drops, etc.).
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
            # Text-only path for inline equations (no crop available).
            # Apply OCR fixes so artefacts don't confuse the model.
            tmpl = _TEXT_PROMPTS.get(category, _TEXT_PROMPTS["unknown"])
            messages.append(
                {"role": "user", "content": tmpl.format(text=_apply_ocr_fixes(region_text or ""))}
            )
        return messages

    def _parse_response(self, raw: str, region_text: str, category: str) -> RecognitionResult:
        if not raw:
            return RecognitionResult(
                plain_text=region_text, confidence=0.0, notes=["empty_response"]
            )

        # Reject explanatory prose responses: the VLM is refusing or hedging instead of
        # returning the equation.  Fall back to the embedded region text so at minimum the
        # raw OCR content is preserved rather than hallucinated explanatory text.
        if _PROSE_RESPONSE_RE.match(raw.strip()):
            logger.debug(
                "vl_response_rejected_prose",
                category=category,
                preview=raw.strip()[:80],
            )
            return RecognitionResult(
                plain_text=region_text,
                confidence=0.1,
                notes=["vl_response_rejected:prose"],
            )

        # Reject LaTeX structural/document commands — the VLM hallucinated a document
        # snippet (\section{...}, \newcommand{...}) instead of a mathematical expression.
        if _LATEX_STRUCTURAL_CMD_RE.match(raw.strip()):
            logger.debug(
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

        # All math categories: treat response as LaTeX.
        # Prefer the VLM-produced LaTeX as plain_text: it is the recognised equation
        # content, whereas region_text is raw OCR that often includes surrounding
        # paragraph prose when the layout region spans more than the equation itself.
        latex = _postprocess_latex(_clean_latex(raw))
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
    """Pure passthrough of the embedded plain text — no model, always available (FR-013)."""

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
        """No model to ask — conservatively answer ``False`` (never promotes ambiguous blocks)."""
        return False

    def close(self) -> None:
        """No-op: the passthrough provider holds no resources (symmetry with QwenVLProvider)."""


