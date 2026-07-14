"""Heuristic recognition-quality scoring for VL equation output (feature 008, retry gating).

The VL provider otherwise returns a flat constant confidence, which is useless as a retry
signal. This module derives a ``[0,1]`` quality score from the recognised representation so
the extractor can gate a stronger-prompt / padded higher-resolution re-crop retry and so the
existing low-confidence flagging (:func:`equation_extraction.confidence.flag_low_confidence`)
reflects real fidelity.

Pure and deterministic. It penalises exactly the failure modes observed on the Knovel
engineering corpus (verified on 79462_09.pdf, where the LLM judge rejected 13/65 equations):

* **lhs_clipped** — LaTeX begins at the ``=`` sign; the left-hand-side variable was clipped
  out of a tight layout crop (e.g. ``= {\\frac{\\varepsilon_c E}{FS}}`` for ``S_c = …``).
* **no_relational_operator** — a math equation reduced to a bare right-hand-side fragment
  with no ``=``/inequality at all (e.g. ``\\frac{\\pi^2}{(KL/r)^2}``).
* **multiple_tags** — two stacked equations merged into one record (two ``\\tag{}`` markers).
* **empty_latex** — no LaTeX produced.

A penalty only lowers confidence; it never fabricates a high score. Callers use the score to
*decide whether to retry* and keep the higher-scoring attempt, so a false penalty costs at
most one extra recognition call, never output quality.
"""

from __future__ import annotations

import re

__all__ = ["score_recognition", "MATH_CATEGORIES"]

# Notes that indicate the VLM output is unsuitable and should trigger a retry.
# quality:multiple_tags / quality:multiple_equations → also try a split-crop retry.
RETRY_QUALITY_NOTES: frozenset[str] = frozenset({
    "quality:label_only",
    "quality:prose_contamination",
    "quality:prose_in_text_block",
    "quality:spaced_text",
    "quality:multiple_tags",
    "quality:multiple_equations",
})

MATH_CATEGORIES: frozenset[str] = frozenset(
    {"mathematical_equation", "engineering_formula", "statistical_expression"}
)

# Baseline confidence for a recognition with no detected quality problem. Matches the
# provider's historical constant so unaffected equations keep their previous score.
_BASE_CONFIDENCE = 0.85

_TAG_RE = re.compile(r"\\tag\b")
_TAG_BLOCK_RE = re.compile(r"\\tag\s*\{[^}]*\}")
# A relational/definition operator anywhere marks a complete equation rather than a fragment.
_REL_OP_RE = re.compile(
    r"(=|\\leq|\\geq|\\neq|\\approx|\\equiv|\\propto|\\sim|\\simeq|\\cong|\\subset|\\subseteq"
    r"|\\supset|\\supseteq|\\in|\\notin|\\rightarrow|\\to|\\Rightarrow|\\Leftrightarrow|[<>])"
)
# LaTeX whose first non-space token is the equals sign (optionally after a stray leading
# command left by a clipped subscript): the left-hand side was cut off.
_LHS_CLIPPED_RE = re.compile(r"^\s*(?:\\[a-zA-Z]+\s*)?=")
# Spaced-character sequences: 4+ single ASCII letters each separated by only whitespace
# (or a backslash-space).  This pattern "U n e \ e q u a t i o n" is produced when
# UniMERNet's band isolation picks a prose text line instead of the equation — the model
# reads each glyph individually because the text was typeset with wide letter-spacing.
_SPACED_CHARS_RE = re.compile(r"\b([A-Za-z]\s){4,}[A-Za-z]\b")

# Output is just an equation label (e.g. "Eq. 12.4.3", "(12.4.3)", "[2-26]") — the VLM
# read the margin label instead of the formula body.  Covers plain-text forms and LaTeX
# \left( N-N \right) which the VLM emits when it reads the parenthesised equation number.
_LABEL_ONLY_RE = re.compile(
    r"""^\s*
    (?:Eq(?:uation)?[.:]?\s*[\d]+(?:\.[\d]+){1,4}  # Eq. 12.4.3 / Equation 12.4.3
    |\(\s*[\d]+(?:[-.][\d]+){1,4}\s*\)              # (12.4.3) or (2-26) plain text
    |\[\s*[\d]+(?:[-.][\d]+){1,4}\s*\]              # [12.4.3] plain text
    |\\left\s*\(\s*[\d]+(?:\s*[-–]\s*[\d]+){1,4}\s*\\right\s*\)  # \left( 2-21 \right)
    )\s*$""",
    re.VERBOSE | re.IGNORECASE,
)

# Prose words that should not appear in a math output unless operators are also present.
_PROSE_WORDS_RE = re.compile(
    r"\b(?:where|which|from|then|becomes|therefore|the|this|that|"
    r"these|those|for|with|into|using|equation|factor|value|note)\b",
    re.IGNORECASE,
)


def score_recognition(
    *, latex: str | None, plain_text: str, category: str
) -> tuple[float, list[str]]:
    """Return ``(confidence, notes)`` for one recognition result.

    ``notes`` are ``quality:<reason>`` markers for every penalty applied (empty when clean),
    surfaced on the equation so the retry decision and any review are explainable.
    """
    # Chemical categories use a structured (SMILES) path, not LaTeX — defer to presence of
    # any recognised content and never apply the math-shape penalties below.
    if category in ("chemical_structure", "chemical_equation"):
        has_content = bool((plain_text or "").strip())
        return (_BASE_CONFIDENCE if has_content else 0.0), []

    text = (latex or "").strip()
    if not text:
        return 0.1, ["quality:empty_latex"]

    notes: list[str] = []
    score = _BASE_CONFIDENCE

    # Spaced-text detection: UniMERNet reading a prose line produces "T h e r e f o r e"
    # style spaced characters.  This fires a severe penalty so the caller falls back to
    # Qwen rather than storing word-soup as the equation LaTeX.
    if category in MATH_CATEGORIES and _SPACED_CHARS_RE.search(text):
        score = min(score, 0.10)
        notes.append("quality:spaced_text")
        return round(score, 4), notes

    # Label-only output: the VLM returned the margin label (e.g. "Eq. 12.4.3") instead
    # of the formula body — the crop was too tight and the label dominated the image.
    if _LABEL_ONLY_RE.match(text):
        score = min(score, 0.10)
        notes.append("quality:label_only")
        return round(score, 4), notes

    # Prose contamination: output contains prose words but no relational operator — the
    # crop captured surrounding explanation text rather than the formula.
    if category in MATH_CATEGORIES:
        prose_hits = len(_PROSE_WORDS_RE.findall(text))
        if prose_hits >= 2 and not _REL_OP_RE.search(text):
            score = min(score, 0.20)
            notes.append("quality:prose_contamination")

    # Connective sentence fragment inside \text{}: patterns like
    # "\text{ or for standard conditions at }" are prose bleeding from surrounding text.
    # Short connectives distinguish these from quantity labels like "\text{Heat liberated}".
    if category in MATH_CATEGORIES and "quality:prose_contamination" not in notes:
        _CONNECTIVE_RE = re.compile(
            r'\b(?:or|and|the|for|with|at|of|in|is|are|to|by)\b', re.IGNORECASE
        )
        _text_blocks = re.findall(r'\\text\{([^}]*)\}', text)
        prose_in_text = sum(1 for blk in _text_blocks if len(_CONNECTIVE_RE.findall(blk)) >= 2)
        if prose_in_text:
            score = min(score, 0.30)
            notes.append("quality:prose_in_text_block")

    # Suspiciously short: strip LaTeX commands and whitespace; fewer than 3 math characters
    # left means the output carries essentially no equation content.
    math_chars = re.sub(r"\\[a-zA-Z]+|\s", "", text)
    if len(math_chars) < 3:
        score = min(score, 0.30)
        notes.append("quality:suspiciously_short")

    if len(_TAG_RE.findall(text)) >= 2:
        score = min(score, 0.35)
        notes.append("quality:multiple_tags")

    # Multiple operator lines without \tag{}: the VLM transcribed two stacked
    # equations onto separate lines (one operator per line) instead of one.
    # Only penalise math categories; chemical equations legitimately span lines.
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
