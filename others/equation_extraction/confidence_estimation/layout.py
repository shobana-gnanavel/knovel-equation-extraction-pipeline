"""Layout confidence estimator — evaluates crop quality from the PIL image."""

from __future__ import annotations

import logging
from typing import Any

from .config import ConfidenceConfig
from .models import Issue, LayoutDetails

log = logging.getLogger(__name__)


def estimate_layout(
    *,
    crop_image: Any | None,
    bbox: tuple[float, float, float, float] | None,
    image_metadata: dict | None,
    config: ConfidenceConfig,
) -> tuple[float | None, LayoutDetails, list[Issue]]:
    issues: list[Issue] = []
    details = LayoutDetails()

    if crop_image is None:
        issues.append(Issue(
            code="MISSING_CROP_IMAGE",
            severity="error",
            message="Crop image is None; layout confidence unavailable.",
            component="layout",
        ))
        return None, details, issues

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            issues.append(Issue(
                code="INVALID_BOUNDING_BOX",
                severity="error",
                message=f"Degenerate bounding box {bbox}; zero or negative area.",
                component="layout",
            ))
            return None, details, issues

    try:
        import numpy as np
        from PIL import Image as PILImage
    except ImportError:
        issues.append(Issue(
            code="IMAGING_DEPS_UNAVAILABLE",
            severity="warning",
            message="PIL/numpy not installed; layout confidence unavailable.",
            component="layout",
        ))
        return None, details, issues

    try:
        if isinstance(crop_image, (str, bytes)):
            img = PILImage.open(crop_image).convert("RGB")
        else:
            img = crop_image.convert("RGB")
        arr = np.array(img, dtype=np.float32)
    except Exception as exc:
        log.warning("layout_estimator_image_load_failed: %s", exc)
        issues.append(Issue(
            code="CORRUPT_IMAGE",
            severity="error",
            message=f"Could not load crop image: {exc}",
            component="layout",
        ))
        return None, details, issues

    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return None, details, issues

    score = 1.0
    strip = config.border_strip_width_px
    threshold = config.border_darkness_threshold

    # --- Border touching (clipping) ---
    borders = {
        "left":   arr[:, :strip, :].mean() < threshold,
        "right":  arr[:, -strip:, :].mean() < threshold,
        "top":    arr[:strip, :, :].mean() < threshold,
        "bottom": arr[-strip:, :, :].mean() < threshold,
    }
    details.touches_left_border = bool(borders["left"])
    details.touches_right_border = bool(borders["right"])
    details.touches_top_border = bool(borders["top"])
    details.touches_bottom_border = bool(borders["bottom"])

    if borders["left"]:
        score -= 0.25
        issues.append(Issue("CLIPPED_LEFT_BORDER", "warning",
                            "Dark content at left edge; equation may be truncated.", "layout"))
    if borders["right"]:
        score -= 0.25
        issues.append(Issue("CLIPPED_RIGHT_BORDER", "warning",
                            "Dark content at right edge; equation may be truncated.", "layout"))
    if borders["top"]:
        score -= 0.10
        issues.append(Issue("CLIPPED_TOP_BORDER", "warning",
                            "Dark content at top edge; multi-line crop may be truncated.", "layout"))
    if borders["bottom"]:
        score -= 0.10
        issues.append(Issue("CLIPPED_BOTTOM_BORDER", "warning",
                            "Dark content at bottom edge; multi-line crop may be truncated.", "layout"))

    # --- DPI ---
    dpi: float | None = None
    if image_metadata:
        dpi = image_metadata.get("dpi") or image_metadata.get("resolution")
    if dpi is None:
        img_info = getattr(img, "info", {})
        dpi_info = img_info.get("dpi") if img_info else None
        if dpi_info is not None:
            dpi = float(dpi_info[0]) if isinstance(dpi_info, tuple) else float(dpi_info)

    details.estimated_dpi = dpi
    if dpi is not None:
        if dpi < config.critical_min_dpi:
            score = min(score, 0.20)
            issues.append(Issue("CRITICALLY_LOW_DPI", "error",
                                f"DPI {dpi:.0f} is critically low; OCR quality severely impaired.", "layout"))
        elif dpi < config.min_recommended_dpi:
            res_penalty = (1.0 - dpi / config.min_recommended_dpi) * 0.30
            score -= res_penalty
            issues.append(Issue("ESTIMATED_DPI_LOW", "warning",
                                f"Estimated DPI {dpi:.0f} below recommended {config.min_recommended_dpi:.0f}.", "layout"))

    # --- Aspect ratio ---
    ratio = w / h if h > 0 else 0.0
    details.aspect_ratio = round(ratio, 2)
    if ratio < config.min_plausible_aspect_ratio:
        score -= 0.15
        issues.append(Issue("UNUSUAL_ASPECT_RATIO_NARROW", "warning",
                            f"Aspect ratio {ratio:.2f} is too narrow; crop may be fragmented.", "layout"))
    elif ratio > config.max_plausible_aspect_ratio:
        score -= 0.10
        issues.append(Issue("UNUSUAL_ASPECT_RATIO_WIDE", "warning",
                            f"Aspect ratio {ratio:.2f} is too wide; crop may span multiple equations.", "layout"))

    # --- Multiple-equations detection (horizontal whitespace bands) ---
    row_means = arr.mean(axis=(1, 2))
    band_min = config.whitespace_band_min_height_px
    in_band = False
    band_count = 0
    current = 0
    for mean_val in row_means:
        if mean_val > threshold:
            current += 1
            if not in_band and current >= band_min:
                in_band = True
        else:
            if in_band:
                band_count += 1
            in_band = False
            current = 0

    multi = band_count >= 1
    details.multiple_equations_suspected = multi
    if multi:
        score -= 0.35
        issues.append(Issue("MULTIPLE_EQUATIONS_IN_CROP", "warning",
                            "Whitespace band detected; crop may contain multiple equations.", "layout"))

    # --- Padding adequacy ---
    margins = [
        float(arr[:strip, :, :].mean()),
        float(arr[-strip:, :, :].mean()),
        float(arr[:, :strip, :].mean()),
        float(arr[:, -strip:, :].mean()),
    ]
    min_margin_brightness = min(margins)
    min_margin_px = (min_margin_brightness / 255.0) * strip
    details.min_margin_px = round(min_margin_px, 1)
    padding_score = min(min_margin_px / config.min_padding_px, 1.0)
    score *= padding_score

    return max(0.0, min(1.0, score)), details, issues
