"""Equation-extraction sidecar cache for idempotent, cache-aware reruns (feature 008, FR-025/IX).

Mirrors ``text_extraction/manifest.py``: persists the Equation Extraction Context next to the PDF as
``<pdf>.equation_extraction.json`` and reuses it only when both the document fingerprint AND a hash of
the relevant ``KNOVEL_EQUATION_*`` settings (providers, mapping, thresholds, toggles) match — so
unchanged documents are not re-processed (SC-008) and config/provider tuning automatically invalidates
stale results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import structlog

from equation_extraction.extractor import extract_equations
from pipeline import config
from pipeline.models import (
    ClassificationContext,
    EquationExtractionContext,
    LayoutContext,
    PreprocessingContext,
    ReadingOrderContext,
    TextExtractionContext,
)

__all__ = ["compute_config_hash", "get_or_create_equation_extraction"]

logger = structlog.get_logger(__name__)


def compute_config_hash() -> str:
    """Stable hash over the settings that affect an equation-extraction decision (incl. providers).

    The ``detection_algorithm_version`` key is bumped manually whenever the detection logic
    (scoring thresholds, regex patterns, pass ordering) changes in a way that would produce
    different results for the same PDF + config.  This ensures old sidecars are never reused
    after algorithm improvements — avoiding the silent "0 equations" symptom caused by serving
    stale results after code updates.
    """
    relevant = {
        # ── Algorithm version — bump when detection/scoring logic changes ──────
        "detection_algorithm_version": "11",  # first-pass: FIRST_PASS_ZOOM render + directional padding on every initial crop
        # ── Config settings ────────────────────────────────────────────────────
        "enabled": config.KNOVEL_EQUATION_ENABLED,
        "vl_model": config.KNOVEL_EQUATION_VL_MODEL,
        "vl_base_url": config.KNOVEL_OLLAMA_BASE_URL,
        "vl_max_tokens": config.KNOVEL_EQUATION_VL_MAX_TOKENS,
        "provider_map": config.KNOVEL_EQUATION_PROVIDER_MAP,
        "inline_enabled": config.KNOVEL_EQUATION_INLINE_ENABLED,
        "classification_min_confidence": config.KNOVEL_EQUATION_CLASSIFICATION_MIN_CONFIDENCE,
        "recognition_min_confidence": config.KNOVEL_EQUATION_RECOGNITION_MIN_CONFIDENCE,
        "crop_pad_frac": config.KNOVEL_EQUATION_CROP_PAD_FRAC,
        "crop_pad_left_frac": config.KNOVEL_EQUATION_CROP_PAD_LEFT_FRAC,
        "crop_pad_right_frac": config.KNOVEL_EQUATION_CROP_PAD_RIGHT_FRAC,
        "crop_pad_top_frac": config.KNOVEL_EQUATION_CROP_PAD_TOP_FRAC,
        "crop_pad_bottom_frac": config.KNOVEL_EQUATION_CROP_PAD_BOTTOM_FRAC,
        "retry_enabled": config.KNOVEL_EQUATION_RETRY_ENABLED,
        "retry_threshold": config.KNOVEL_EQUATION_RECOGNITION_RETRY_THRESHOLD,
        "retry_zoom": config.KNOVEL_EQUATION_RETRY_ZOOM,
        "first_pass_zoom": config.KNOVEL_EQUATION_FIRST_PASS_ZOOM,
        "first_pass_pad_frac": config.KNOVEL_EQUATION_FIRST_PASS_PAD_FRAC,
        "first_pass_pad_left_frac": config.KNOVEL_EQUATION_FIRST_PASS_PAD_LEFT_FRAC,
        "latex_enabled": config.KNOVEL_EQUATION_LATEX_ENABLED,
        "mathml_enabled": config.KNOVEL_EQUATION_MATHML_ENABLED,
        "structured_enabled": config.KNOVEL_EQUATION_STRUCTURED_ENABLED,
        "validation_strictness": config.KNOVEL_EQUATION_VALIDATION_STRICTNESS,
    }
    payload = json.dumps(relevant, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_or_create_equation_extraction(
    pdf_path: Path,
    fingerprint: str,
    *,
    text_extraction: TextExtractionContext | None,
    reading_order: ReadingOrderContext | None,
    layout: LayoutContext | None,
    preprocessing: PreprocessingContext | None,
    classification: ClassificationContext | None,
    page_manifest: list,
) -> EquationExtractionContext:
    """Return a cached Equation Extraction Context if fresh, else compute and persist one."""
    config_hash = compute_config_hash()
    sidecar_path = config.resolve_sidecar_path(pdf_path, "equation_extraction")

    if config.KNOVEL_EQUATION_REUSE and sidecar_path.exists():
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            cached_hash = payload.get("config_hash", "")
            # Detect legacy sidecar written by a pre-feature-008 pipeline (non-hex hash like
            # "eq_label_extractor_v1"). These are never reused; log a one-time warning so the
            # stale file is visible without being mistaken for a real cache miss.
            if cached_hash and len(cached_hash) != 64:
                logger.warning(
                    "equation_extraction_cache_legacy_format",
                    pdf=str(pdf_path),
                    cached_hash=cached_hash,
                    hint="sidecar written by an older pipeline; will recompute",
                )
            elif payload.get("fingerprint") == fingerprint and cached_hash == config_hash:
                logger.info("equation_extraction_cache_hit", pdf=str(pdf_path))
                return EquationExtractionContext.from_dict(payload["context"])
        except Exception as exc:  # corrupt sidecar → recompute, never raise
            logger.warning(
                "equation_extraction_cache_unreadable", pdf=str(pdf_path), error=str(exc)
            )

    context = extract_equations(
        pdf_path,
        text_extraction=text_extraction,
        reading_order=reading_order,
        layout=layout,
        preprocessing=preprocessing,
        classification=classification,
        page_manifest=page_manifest,
        config_hash=config_hash,
    )

    try:
        sidecar_payload = {
            "fingerprint": fingerprint,
            "config_hash": config_hash,
            "context": context.to_dict(),
        }
        sidecar_path.write_text(
            json.dumps(sidecar_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:  # caching is best-effort; extraction still returns
        logger.warning("equation_extraction_cache_write_failed", pdf=str(pdf_path), error=str(exc))

    return context
