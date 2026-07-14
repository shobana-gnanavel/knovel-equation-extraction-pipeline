"""Common utility functions for the equation-extraction pipeline.

This module merges three utility concerns:

* **Crop splitting** (``split_stacked_crop``, ``MULTI_EQ_NOTES``) —
  horizontal white-band splitting for stacked multi-equation image crops.
* **Representation assembly** (``Representations``, ``assemble``,
  ``is_valid_latex``, ``is_valid_mathml``) — category-appropriate LaTeX /
  MathML assembly with lightweight validity checks.
* **Stage guard** (``stage_guard``) — context-manager that catches stage
  exceptions, wraps them as :class:`~equation_extraction_pipeline.domain.models.StageFailure`
  records, and logs them to disk.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Crop splitting
# ---------------------------------------------------------------------------
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

from equation_extraction_pipeline.config.logging import log_stage_failure
from equation_extraction_pipeline.domain.models import StageFailure

if TYPE_CHECKING:
    from equation_extraction_pipeline.extraction.ocr_extractor import RecognitionResult

__all__ = [
    # crop splitting
    "split_stacked_crop",
    "MULTI_EQ_NOTES",
    # representation helpers
    "Representations",
    "assemble",
    "is_valid_latex",
    "is_valid_mathml",
    # stage guard
    "stage_guard",
]

# ---------------------------------------------------------------------------
# Crop-split constants
# ---------------------------------------------------------------------------

# Notes emitted by score_recognition that signal multiple equations in one crop.
# Wire these into RETRY_QUALITY_NOTES so _recognize_with_retry triggers a split.
MULTI_EQ_NOTES: frozenset[str] = frozenset({
    "quality:multiple_tags",
    "quality:multiple_equations",
})

# Pixel darkness threshold: values below this are treated as "ink".
_DARK_THRESHOLD: int = 220
# A row is "white" when fewer than this fraction of its pixels are dark.
_DARK_ROW_FRAC: float = 0.05
# Minimum consecutive white rows to count as a valid inter-equation gap.
# At zoom=3 (default first-pass), 1 pt ≈ 3 px; a real inter-equation gap is
# at least 4–6 pt, so 12 px is a conservative lower bound.  At zoom=2, 8 px.
# Using 6 px as the floor lets both zoom levels detect most gaps without
# splitting on the thin whitespace between a fraction bar and its numerator.
_MIN_BAND_PX: int = 6
# A sub-crop shorter than this fraction of the total height is noise.
_MIN_BLOCK_FRAC: float = 0.12
# Every retained band must span a meaningful portion of the crop width.  A
# fraction denominator (for example the lone ``g`` in ``R phi / g``) is narrow
# and must remain attached to the equation above, not become a second equation.
_MIN_INK_WIDTH_FRAC: float = 0.12


def split_stacked_crop(
    image: object,
    *,
    min_band_px: int = _MIN_BAND_PX,
    dark_threshold: int = _DARK_THRESHOLD,
    dark_row_frac: float = _DARK_ROW_FRAC,
) -> list:
    """Split a PIL image at horizontal whitespace bands into equation sub-crops.

    Scans every row in the grayscale image; a run of consecutive rows where
    fewer than *dark_row_frac* of pixels are darker than *dark_threshold* and
    the run is at least *min_band_px* rows long is treated as the whitespace
    gap between two stacked equations.

    Returns:
        A list of PIL images (fresh crops from the original).  If no qualifying
        split is found the list contains the original image unchanged so callers
        can always do ``sub_crops[0]`` safely.
    """
    try:
        import numpy as np
        gray = np.array(image.convert("L"))  # type: ignore[union-attr]
    except Exception:
        return [image]

    height, width = gray.shape
    if height < 30 or width == 0:
        return [image]

    dark_per_row = (gray < dark_threshold).sum(axis=1) / width
    is_white = dark_per_row < dark_row_frac

    blocks: list[tuple[int, int]] = []
    in_white = False
    band_start = 0
    content_start = 0

    for r in range(height):
        if is_white[r]:
            if not in_white:
                in_white = True
                band_start = r
        else:
            if in_white:
                in_white = False
                if r - band_start >= min_band_px:
                    blocks.append((content_start, band_start))
                    content_start = r
    blocks.append((content_start, height))

    min_h = max(10, int(height * _MIN_BLOCK_FRAC))
    valid: list[tuple[int, int]] = []
    for s, e in blocks:
        if e - s < min_h:
            continue
        ink = gray[s:e] < dark_threshold
        ink_cols = ink.any(axis=0).nonzero()[0]
        ink_width = int(ink_cols[-1] - ink_cols[0] + 1) if len(ink_cols) else 0
        if ink_width < max(8, int(width * _MIN_INK_WIDTH_FRAC)):
            continue
        valid.append((s, e))

    if len(valid) <= 1:
        return [image]

    return [image.crop((0, s, width, e)) for s, e in valid]


# ---------------------------------------------------------------------------
# Representation assembly
# ---------------------------------------------------------------------------

try:
    import latex2mathml.converter as _latex2mathml
except Exception:
    _latex2mathml = None  # type: ignore[assignment]

_MATH_CATEGORIES = {"mathematical_equation", "engineering_formula", "statistical_expression"}
_CHEM_STRUCTURE = "chemical_structure"
_CHEM_EQUATION = "chemical_equation"

_ENV_BEGIN = re.compile(r"\\begin\{")
_ENV_END = re.compile(r"\\end\{")


@dataclass
class Representations:
    """The assembled, validity-checked representations for one equation."""

    plain_text: str = ""
    latex: str | None = None
    mathml: str | None = None
    structured_form: str | None = None


def is_valid_latex(latex: str) -> bool:
    """Lightweight LaTeX-validity probe: balanced braces/environments + a parse attempt."""
    if not latex or not latex.strip():
        return False
    if latex.count("{") != latex.count("}"):
        return False
    if len(_ENV_BEGIN.findall(latex)) != len(_ENV_END.findall(latex)):
        return False
    if _latex2mathml is None:
        return True
    try:
        _latex2mathml.convert(latex)
        return True
    except Exception:
        return False


def is_valid_mathml(mathml: str) -> bool:
    """MathML well-formedness via a stdlib XML parse."""
    if not mathml or not mathml.strip():
        return False
    try:
        ElementTree.fromstring(mathml)
        return True
    except Exception:
        return False


def _to_mathml(latex: str) -> str | None:
    if _latex2mathml is None:
        return None
    try:
        return str(_latex2mathml.convert(latex))
    except Exception:
        return None


def assemble(
    result: RecognitionResult, *, category: str, config: Any
) -> tuple[Representations, list[str]]:
    """Assemble category-appropriate representations and return ``(representations, flags)``.

    Args:
        result:   Provider recognition output containing raw LaTeX/MathML/plain text.
        category: Equation category string (e.g. ``"mathematical_equation"``).
        config:   Any object that exposes ``KNOVEL_EQUATION_LATEX_ENABLED``,
                  ``KNOVEL_EQUATION_MATHML_ENABLED``, and
                  ``KNOVEL_EQUATION_STRUCTURED_ENABLED`` as attributes.

    Returns:
        A tuple of (:class:`Representations`, list-of-flag-strings).  Flags
        include ``"invalid_latex"`` and ``"invalid_mathml"`` when the
        corresponding representation fails validation but is still retained.
    """
    flags: list[str] = []
    latex_enabled = bool(getattr(config, "KNOVEL_EQUATION_LATEX_ENABLED", True))
    mathml_enabled = bool(getattr(config, "KNOVEL_EQUATION_MATHML_ENABLED", False))
    structured_enabled = bool(getattr(config, "KNOVEL_EQUATION_STRUCTURED_ENABLED", True))

    rep = Representations(plain_text=result.plain_text or "")

    if category in _MATH_CATEGORIES or category == _CHEM_EQUATION:
        if latex_enabled and result.latex:
            if is_valid_latex(result.latex):
                rep.latex = result.latex
            else:
                rep.latex = result.latex  # retained but flagged
                flags.append("invalid_latex")
        if mathml_enabled:
            mathml = result.mathml or (_to_mathml(rep.latex) if rep.latex else None)
            if mathml:
                if is_valid_mathml(mathml):
                    rep.mathml = mathml
                else:
                    rep.mathml = mathml
                    flags.append("invalid_mathml")

    if category in {_CHEM_STRUCTURE, _CHEM_EQUATION} and structured_enabled:
        rep.structured_form = result.structured_form or None

    return rep, flags


# ---------------------------------------------------------------------------
# Stage guard
# ---------------------------------------------------------------------------


@contextmanager
def stage_guard(
    stage_name: str,
    book_id: str,
    pipeline_run_id: str,
    output_dir: Path,
    table_id: str | None = None,
    page_no: int | None = None,
):
    """Context manager that catches stage exceptions and records them as failures.

    On exception, constructs a :class:`~equation_extraction_pipeline.domain.models.StageFailure`
    and delegates to
    :func:`~equation_extraction_pipeline.config.logging.log_stage_failure`
    so that all stage errors are written to a consistent failure log.  The
    exception is **not** re-raised, allowing the pipeline to continue with the
    next item.

    Args:
        stage_name:       Human-readable identifier for the pipeline stage.
        book_id:          Identifier of the book being processed.
        pipeline_run_id:  Unique identifier for the current pipeline run.
        output_dir:       Directory where failure logs are written.
        table_id:         Optional table identifier (for table-extraction stages).
        page_no:          Optional page number being processed.
    """
    try:
        yield
    except Exception as e:
        failure = StageFailure(
            pipeline_run_id=pipeline_run_id,
            book_id=book_id,
            table_id=table_id,
            page_no=page_no,
            stage=stage_name,
            error_type=type(e).__name__,
            error_msg=str(e),
            retry_count=0,
            is_gold_candidate=stage_name in ("table_extraction", "llm_correction"),
            timestamp=datetime.utcnow(),
        )
        log_stage_failure(failure, output_dir)
