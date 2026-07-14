"""Horizontal white-band splitting for stacked multi-equation image crops.

When Docling or the label-scan places two adjacent equations in a single region,
the VLM crop includes both. Sending a tall crop to Qwen causes it to transcribe
both equations, producing ``\\tag{}`` markers or multiple operator lines. This
module detects the horizontal whitespace gap between two stacked equations and
splits the crop so each VLM call sees exactly one equation.

Usage::

    from equation_extraction.crop_split import split_stacked_crop, MULTI_EQ_NOTES

    sub_crops = split_stacked_crop(image)
    if len(sub_crops) > 1:
        top = sub_crops[0]          # first (topmost) equation
        ...
"""

from __future__ import annotations

__all__ = ["split_stacked_crop", "MULTI_EQ_NOTES"]

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
# Reject narrow numerator/denominator fragments as standalone equations.
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
