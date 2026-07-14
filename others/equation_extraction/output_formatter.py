"""Canonical output formatter for the equation extraction pipeline.

Produces a single ``equation_extraction.json`` with this top-level structure::

    {
      "document": { ... },
      "equations": [ ... ]
    }

Crop images are written to ``<crops_dir>/page_<NNN>/<eq_id>.png`` and referenced
by that relative path inside each equation's ``crop.path`` field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from pipeline import config as pipeline_config

logger = structlog.get_logger(__name__)

_PIPELINE_VERSION = "1.0"
_BASE_DPI = 72.0  # PDF points per inch
_LABELED_DPI = 216  # 72 × 3 default first-pass zoom, used for labeled-mode crops


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bbox_to_xywh(bbox: list[float]) -> dict[str, float]:
    """Convert [x0, y0, x1, y1] to {x, y, width, height}."""
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    return {
        "x": round(x0, 2),
        "y": round(y0, 2),
        "width": round(x1 - x0, 2),
        "height": round(y1 - y0, 2),
    }


def _to_bbox_list(raw_bbox: Any) -> list[float]:
    if isinstance(raw_bbox, dict):
        return [
            float(raw_bbox.get("x0", 0)),
            float(raw_bbox.get("y0", 0)),
            float(raw_bbox.get("x1", 0)),
            float(raw_bbox.get("y1", 0)),
        ]
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
        return [float(v) for v in raw_bbox[:4]]
    return [0.0, 0.0, 0.0, 0.0]


def _valid_bbox(bbox: list[float]) -> bool:
    if len(bbox) < 4:
        return False
    x0, y0, x1, y1 = bbox[:4]
    return x1 > x0 and y1 > y0 and x0 >= 0 and y0 >= 0


def _refined_bbox(
    bbox: list[float],
    page_dims: tuple[float, float] | None,
    fp_zoom: float,
) -> dict[str, float]:
    """Compute the padded bbox in page-coordinate space using first-pass padding fractions."""
    if not _valid_bbox(bbox) or page_dims is None:
        return _bbox_to_xywh(bbox)
    from equation_extraction.extractor import _first_pass_padding_fractions

    lf, rf, tf, bf = _first_pass_padding_fractions()
    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
    pw, ph = page_dims
    dx, dy = x1 - x0, y1 - y0
    rx0 = max(0.0, x0 - dx * lf)
    ry0 = max(0.0, y0 - dy * tf)
    rx1 = min(pw, x1 + dx * rf)
    ry1 = min(ph, y1 + dy * bf)
    return {
        "x": round(rx0, 2),
        "y": round(ry0, 2),
        "width": round(rx1 - rx0, 2),
        "height": round(ry1 - ry0, 2),
    }


def _refinement_actions(notes: list[str]) -> list[str]:
    """Derive bbox refinement labels from crop-touch notes; always include padding_applied."""
    touch_map = {
        "crop_touch:left": "expand_left",
        "crop_touch:right": "expand_right",
        "crop_touch:top": "expand_top",
        "crop_touch:bottom": "expand_bottom",
    }
    actions = [touch_map[n] for n in notes if n in touch_map]
    if not actions:
        actions.append("padding_applied")
    return actions


def _crop_issues(eq: Any) -> list[str]:
    issues: list[str] = []
    for flag in getattr(eq, "validation_flags", []) or []:
        if flag in {"invalid_bbox", "broken_multiline"}:
            issues.append(flag)
    for note in getattr(eq, "notes", []) or []:
        if note.startswith("crop_touch:"):
            issues.append(note)
    return issues


def _crop_status(eq: Any) -> str:
    if "invalid_bbox" in (getattr(eq, "validation_flags", []) or []):
        return "FAIL"
    if any(n.startswith("crop_touch:") for n in (getattr(eq, "notes", []) or [])):
        return "WARNING"
    return "PASS"


def _retry_info(eq: Any) -> dict[str, Any]:
    notes = getattr(eq, "notes", []) or []
    performed = any(n.startswith("recognition_retry:") for n in notes)
    return {"performed": performed, "retry_count": 1 if performed else 0}


def _final_status(eq: Any) -> str:
    flags = set(getattr(eq, "validation_flags", []) or [])
    if "unsupported_category" in flags or "broken_multiline" in flags:
        return "PARTIAL"
    if any(n.startswith("recognition_failed") for n in (getattr(eq, "notes", []) or [])):
        return "FAILED"
    if float(getattr(eq, "recognition_confidence", 0.0) or 0.0) < 0.5:
        return "LOW_CONFIDENCE"
    return "SUCCESS"


def _provider_model_name(selected_provider: str) -> str:
    mapping = {
        "math": "Qwen2.5-VL-7B",
        "generic": "Qwen2.5-VL-7B",
        "chemical": "UniMERNet",
        "chemical_structure": "MolScribe",
    }
    return mapping.get(selected_provider, selected_provider)


def _safe_id(equation_id: str) -> str:
    return str(equation_id).replace("/", "_").replace("\\", "_")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_structured_crops(
    context: Any,
    pdf_path: Path,
    preprocessing: Any,
    layout: Any,
    crops_dir: Path,
) -> dict[str, tuple[str, int, int]]:
    """Save per-equation crop images into page-organised subdirectories.

    Each crop is written to ``<crops_dir>/page_<NNN>/<eq_id>.png``.

    Returns a mapping of ``eq_id → (relative_path, pixel_width, pixel_height)``.
    """
    from equation_extraction.extractor import (
        _load_region_image,
        _valid_bbox as _ext_valid_bbox,
    )

    crop_info: dict[str, tuple[str, int, int]] = {}

    geom_by_page: dict[int, tuple[float, float]] = {}
    if layout is not None:
        for page_layout in getattr(layout, "pages", []):
            geom_by_page[page_layout.page_no] = (page_layout.width, page_layout.height)

    raster_cache: dict[int, Any] = {}
    fp_raster_cache: dict[int, Any] = {}

    for eq in getattr(context, "equations", []):
        if getattr(eq, "is_inline", False):
            continue
        bbox = _to_bbox_list(getattr(eq, "bbox", None))
        if not _ext_valid_bbox(bbox):
            continue
        page_no = getattr(eq, "page_no", None)
        geom = geom_by_page.get(page_no) if page_no is not None else None
        crop = _load_region_image(
            raster_cache,
            preprocessing,
            page_no,
            geom,
            bbox,
            pdf_path=pdf_path,
            fp_raster_cache=fp_raster_cache,
        )
        if crop is None:
            continue
        eq_id = _safe_id(getattr(eq, "equation_id", f"eq_p{page_no}"))
        page_subdir = crops_dir / f"page_{page_no:03d}"
        page_subdir.mkdir(parents=True, exist_ok=True)
        out_path = page_subdir / f"{eq_id}.png"
        try:
            crop.save(out_path)
            w, h = crop.size
            rel_path = f"page_{page_no:03d}/{eq_id}.png"
            crop_info[eq_id] = (rel_path, w, h)
        except Exception as exc:
            logger.debug("equation_crop_save_failed", eq_id=eq_id, error=str(exc))

    return crop_info


def format_equation_extraction_output(
    context: Any,
    pdf_path: Path,
    classification: Any | None,
    crop_info: dict[str, tuple[str, int, int]],
    layout: Any,
) -> dict[str, Any]:
    """Build the canonical ``equation_extraction.json`` dict from pipeline contexts.

    ``crop_info`` is the dict returned by :func:`save_structured_crops`:
    ``eq_id → (relative_path, pixel_width, pixel_height)``.
    """
    pdf_type = "digital"
    if classification is not None:
        pdf_type = getattr(classification, "modality", "digital")

    document: dict[str, Any] = {
        "document_id": pdf_path.stem,
        "pdf_type": pdf_type,
        "total_pages": context.statistics.total_pages,
        "pipeline_version": _PIPELINE_VERSION,
    }

    geom_by_page: dict[int, tuple[float, float]] = {}
    if layout is not None:
        for page_layout in getattr(layout, "pages", []):
            geom_by_page[page_layout.page_no] = (page_layout.width, page_layout.height)

    fp_zoom = float(getattr(pipeline_config, "KNOVEL_EQUATION_FIRST_PASS_ZOOM", 3.0))
    rendered_dpi = int(_BASE_DPI * fp_zoom)

    from equation_extraction.extractor import _first_pass_padding_fractions
    lf, rf, tf, bf = _first_pass_padding_fractions()

    equations_out: list[dict[str, Any]] = []
    for eq in getattr(context, "equations", []):
        if getattr(eq, "is_inline", False):
            continue

        bbox = _to_bbox_list(getattr(eq, "bbox", None))
        page_no: int = getattr(eq, "page_no", 0)
        page_dims = geom_by_page.get(page_no)
        eq_id = _safe_id(getattr(eq, "equation_id", f"eq_p{page_no}"))

        # rendering ──────────────────────────────────────────────────────────
        if page_dims:
            pw, ph = page_dims
            image_width = int(pw * fp_zoom)
            image_height = int(ph * fp_zoom)
        else:
            image_width = image_height = 0

        layout_conf = getattr(eq, "confidence_layout", None)
        rendering_quality = round(float(layout_conf), 3) if layout_conf is not None else 0.9

        rendering: dict[str, Any] = {
            "dpi": rendered_dpi,
            "image_width": image_width,
            "image_height": image_height,
            "quality_score": rendering_quality,
        }

        # detection ──────────────────────────────────────────────────────────
        detection: dict[str, Any] = {
            "model": "Docling",
            "confidence": round(float(getattr(eq, "classification_confidence", 0.0)), 3),
            "original_bbox": _bbox_to_xywh(bbox),
        }

        # bbox_refinement ────────────────────────────────────────────────────
        ref_bbox = _refined_bbox(bbox, page_dims, fp_zoom)
        notes: list[str] = list(getattr(eq, "notes", []) or [])
        bbox_refinement: dict[str, Any] = {
            "refined_bbox": ref_bbox,
            "quality_score": rendering_quality,
            "actions": _refinement_actions(notes),
        }

        # crop ───────────────────────────────────────────────────────────────
        if eq_id in crop_info:
            rel_path, crop_w, crop_h = crop_info[eq_id]
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            padding_px: dict[str, int] = {
                "left": round(bw * lf * fp_zoom),
                "right": round(bw * rf * fp_zoom),
                "top": round(bh * tf * fp_zoom),
                "bottom": round(bh * bf * fp_zoom),
            }
            crop_section: dict[str, Any] = {
                "path": rel_path,
                "padding": padding_px,
                "resolution": {"width": crop_w, "height": crop_h},
            }
        else:
            crop_section = {
                "path": None,
                "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                "resolution": {"width": 0, "height": 0},
            }

        # crop_validation ────────────────────────────────────────────────────
        crop_conf = getattr(eq, "confidence_layout", None)
        crop_quality_score = round(float(crop_conf), 3) if crop_conf is not None else 0.9
        crop_validation: dict[str, Any] = {
            "status": _crop_status(eq),
            "quality_score": crop_quality_score,
            "issues": _crop_issues(eq),
        }

        # ocr ────────────────────────────────────────────────────────────────
        ocr: dict[str, Any] = {
            "model": _provider_model_name(getattr(eq, "selected_provider", "generic")),
            "latex": getattr(eq, "latex", None),
            "confidence": round(float(getattr(eq, "recognition_confidence", 0.0)), 3),
        }

        # retry ──────────────────────────────────────────────────────────────
        retry = _retry_info(eq)

        # final ──────────────────────────────────────────────────────────────
        overall = getattr(eq, "overall_confidence", None) or getattr(eq, "recognition_confidence", 0.0)
        final: dict[str, Any] = {
            "overall_confidence": round(float(overall or 0.0), 3),
            "status": _final_status(eq),
        }

        equations_out.append({
            "equation_id": eq_id,
            "page_number": page_no,
            "label": getattr(eq, "equation_number", None),
            "rendering": rendering,
            "detection": detection,
            "bbox_refinement": bbox_refinement,
            "crop": crop_section,
            "crop_validation": crop_validation,
            "ocr": ocr,
            "retry": retry,
            "final": final,
        })

    return {
        "document": document,
        "equations": equations_out,
    }


# ---------------------------------------------------------------------------
# Labeled-mode support  (equations extracted via Eq.X.X.X label scanning)
# ---------------------------------------------------------------------------

def save_labeled_crops(
    equations: list[dict[str, Any]],
    crops_dir: Path,
) -> dict[str, tuple[str, int, int]]:
    """Save in-memory crop PNGs (stored in eq["metadata"]["crop_png"]) to disk.

    Must be called BEFORE ``build_sidecar`` strips the crop bytes.
    Returns ``eq_id → (relative_path, pixel_width, pixel_height)``.
    """
    try:
        from PIL import Image
        from io import BytesIO
    except ImportError:
        logger.warning("pillow_not_available_skipping_crops")
        return {}

    crop_info: dict[str, tuple[str, int, int]] = {}
    for eq in equations:
        eq_id = _safe_id(str(eq.get("equation_id", "")))
        if not eq_id:
            continue
        crop_bytes = (eq.get("metadata") or {}).get("crop_png")
        if not crop_bytes:
            continue
        page_no: int = eq.get("page_no", 0)
        try:
            img = Image.open(BytesIO(crop_bytes)).convert("RGB")
            page_subdir = crops_dir / f"page_{page_no:03d}"
            page_subdir.mkdir(parents=True, exist_ok=True)
            out_path = page_subdir / f"{eq_id}.png"
            img.save(out_path)
            w, h = img.size
            rel_path = f"page_{page_no:03d}/{eq_id}.png"
            crop_info[eq_id] = (rel_path, w, h)
        except Exception as exc:
            logger.debug("labeled_crop_save_failed", eq_id=eq_id, error=str(exc))
    return crop_info


def format_labeled_output(
    equations: list[dict[str, Any]],
    crop_info: dict[str, tuple[str, int, int]],
    pdf_path: Path,
    pdf_type: str = "digital",
) -> dict[str, Any]:
    """Build the canonical JSON structure for labeled-mode extracted equations.

    Accepts the raw equation dicts produced by ``extract_labeled()`` after
    confidence enrichment (i.e. after ``_apply_confidence`` has run but before
    ``build_sidecar`` strips crop bytes).
    """
    total_pages = len({eq.get("page_no", 0) for eq in equations})
    document: dict[str, Any] = {
        "document_id": pdf_path.stem,
        "pdf_type": pdf_type,
        "total_pages": total_pages,
        "pipeline_version": _PIPELINE_VERSION,
    }

    equations_out: list[dict[str, Any]] = []
    for eq in equations:
        if eq.get("is_inline", False):
            continue

        raw_bbox = eq.get("bbox")
        if isinstance(raw_bbox, dict):
            bbox = [
                float(raw_bbox.get("x0", 0)),
                float(raw_bbox.get("y0", 0)),
                float(raw_bbox.get("x1", 0)),
                float(raw_bbox.get("y1", 0)),
            ]
        elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            bbox = [float(v) for v in raw_bbox[:4]]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]

        eq_id = _safe_id(str(eq.get("equation_id", "")))
        page_no: int = eq.get("page_no", 0)

        # rendering ──────────────────────────────────────────────────────────
        crop_entry = crop_info.get(eq_id)
        if crop_entry:
            _, crop_w, crop_h = crop_entry
        else:
            crop_w = crop_h = 0
        layout_conf = eq.get("confidence_layout")
        rendering_quality = round(float(layout_conf), 3) if layout_conf is not None else 0.9
        rendering: dict[str, Any] = {
            "dpi": _LABELED_DPI,
            "image_width": crop_w,
            "image_height": crop_h,
            "quality_score": rendering_quality,
        }

        # detection ──────────────────────────────────────────────────────────
        detection: dict[str, Any] = {
            "model": "Docling",
            "confidence": round(float(eq.get("classification_confidence") or 0.0), 3),
            "original_bbox": _bbox_to_xywh(bbox),
        }

        # bbox_refinement ────────────────────────────────────────────────────
        bbox_refinement: dict[str, Any] = {
            "refined_bbox": _bbox_to_xywh(bbox),
            "quality_score": rendering_quality,
            "actions": ["padding_applied"],
        }

        # crop ───────────────────────────────────────────────────────────────
        if crop_entry:
            rel_path, cw, ch = crop_entry
            crop_section: dict[str, Any] = {
                "path": rel_path,
                "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                "resolution": {"width": cw, "height": ch},
            }
        else:
            crop_section = {
                "path": None,
                "padding": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                "resolution": {"width": 0, "height": 0},
            }

        # crop_validation ────────────────────────────────────────────────────
        crop_conf = eq.get("confidence_layout")
        crop_quality_score = round(float(crop_conf), 3) if crop_conf is not None else 0.9
        issues: list[str] = list(eq.get("validation_flags") or [])
        crop_validation: dict[str, Any] = {
            "status": "FAIL" if "invalid_bbox" in issues else "PASS",
            "quality_score": crop_quality_score,
            "issues": issues,
        }

        # ocr ────────────────────────────────────────────────────────────────
        provider = eq.get("selected_provider", "generic")
        ocr: dict[str, Any] = {
            "model": _provider_model_name(provider) if provider not in ("text_layer", "none", "eq_label_scan", "ollama") else provider,
            "latex": eq.get("latex"),
            "confidence": round(float(eq.get("recognition_confidence") or 0.0), 3),
        }

        # retry ──────────────────────────────────────────────────────────────
        notes: list[str] = list(eq.get("notes") or [])
        performed = any(n.startswith("recognition_retry:") for n in notes)
        retry: dict[str, Any] = {"performed": performed, "retry_count": 1 if performed else 0}

        # final ──────────────────────────────────────────────────────────────
        overall = eq.get("overall_confidence") or eq.get("recognition_confidence") or 0.0
        flags = set(eq.get("validation_flags") or [])
        if "unsupported_category" in flags or "broken_multiline" in flags:
            status = "PARTIAL"
        elif float(overall) < 0.5:
            status = "LOW_CONFIDENCE"
        else:
            status = "SUCCESS"
        final: dict[str, Any] = {
            "overall_confidence": round(float(overall), 3),
            "status": status,
        }

        equations_out.append({
            "equation_id": eq_id,
            "page_number": page_no,
            "label": eq.get("equation_number"),
            "rendering": rendering,
            "detection": detection,
            "bbox_refinement": bbox_refinement,
            "crop": crop_section,
            "crop_validation": crop_validation,
            "ocr": ocr,
            "retry": retry,
            "final": final,
        })

    return {
        "document": document,
        "equations": equations_out,
    }
