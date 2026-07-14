"""Equation extraction pipeline — single entry-point orchestrator.

Runs all six stages in sequence and writes exactly two outputs:
  <output_dir>/<book_id>/document.json   — all extracted data
  <output_dir>/<book_id>/crops/          — equation crop PNGs

No intermediate JSON files are written.

Supersedes the original ``main_pipeline.py`` POC.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from equation_extraction_pipeline.confidence_estimation import ConfidenceEstimator
from equation_extraction_pipeline.config import settings as config
from equation_extraction_pipeline.counting.equation_counter import build_equation_numbers
from equation_extraction_pipeline.detection.equation_block_detector import classify_region
from equation_extraction_pipeline.detection.equation_label_detector import detect_equations
from equation_extraction_pipeline.domain.models import (
    EquationRegion,
    ExtractedEquation,
    OcrResult,
)
from equation_extraction_pipeline.extraction.ocr_extractor import (
    close_providers,
    resolve_providers,
    select_provider,
)
from equation_extraction_pipeline.extraction.page_renderer import render_pages
from equation_extraction_pipeline.extraction.text_extractor import preprocess_pages
from equation_extraction_pipeline.ingestion.pdf_loader import classify_pdf
from equation_extraction_pipeline.reporting.json_report import write_json_report

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

ProgressCallback = Callable[[str, str, int], None]
"""Called as callback(stage_name, message, percent_complete)."""


def _noop(stage: str, msg: str, pct: int) -> None:
    pass


_confidence_estimator = ConfidenceEstimator()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_ocr_result(result: Any, provider_name: str, *, is_retry: bool = False) -> OcrResult:
    """Adapt a RecognitionResult to an OcrResult for document.json compatibility."""
    flags = list(result.notes or [])
    if is_retry:
        flags.append("RETRY_PASS")
    return OcrResult(
        latex=result.latex or result.plain_text or "",
        confidence=result.confidence,
        provider=provider_name,
        flags=flags,
    )


def _recognize_with_provider(
    provider: Any,
    crop_image: Any,
    *,
    category: str,
    region_text: str = "",
) -> tuple[OcrResult, OcrResult | None]:
    """Run first-pass recognition; if confidence is low, retry with zoom + strict prompt."""
    from PIL import Image as _PILImage

    first_result = provider.recognize(
        region_image=crop_image,
        region_text=region_text,
        category=category,
        config=config,
        strict=False,
    )
    first_ocr = _to_ocr_result(first_result, provider.name)

    if first_ocr.confidence >= config.RECOGNITION_RETRY_THRESHOLD:
        return first_ocr, None

    logger.debug(
        "low_confidence_retry conf=%.3f < %.3f category=%s",
        first_ocr.confidence, config.RECOGNITION_RETRY_THRESHOLD, category,
    )

    zoom = config.RECOGNITION_RETRY_ZOOM
    zoomed = crop_image.resize(
        (int(crop_image.width * zoom), int(crop_image.height * zoom)),
        _PILImage.LANCZOS,
    )
    retry_result = provider.recognize(
        region_image=zoomed,
        region_text=region_text,
        category=category,
        config=config,
        strict=True,
    )
    retry_ocr = _to_ocr_result(retry_result, provider.name, is_retry=True)
    return first_ocr, retry_ocr


def _logical_equation_crops(crop_image: Any) -> list[Any]:
    """Preserve one output record per detected equation region.

    Typography inside a single equation (fractions, matrices, aligned lines)
    contains horizontal whitespace that cannot safely determine equation count.
    """
    return [crop_image]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run(
    pdf_path: Path,
    output_dir: Path | None = None,
    *,
    progress: ProgressCallback = _noop,
) -> Path:
    """Execute the full equation extraction pipeline.

    Parameters
    ----------
    pdf_path:
        Path to the input PDF.
    output_dir:
        Root output directory. Defaults to ``config.OUTPUT_DIR``.
        Results land in ``<output_dir>/<pdf_stem>/``.
    progress:
        Optional callback invoked at each stage for progress reporting.

    Returns
    -------
    Path
        Path to the written ``document.json`` file.
    """
    pdf_path = Path(pdf_path).resolve()
    out_root = Path(output_dir) if output_dir else config.OUTPUT_DIR
    book_out = out_root / pdf_path.stem
    book_out.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    logger.info("pipeline_start pdf=%s out=%s", pdf_path.name, book_out)

    # Stage 1 — Ingestion: classify PDF modality (scanned / digital / hybrid)
    progress("classification", "Classifying PDF modality…", 5)
    classification = classify_pdf(pdf_path)
    logger.info(
        "classification modality=%s confidence=%.2f",
        classification.modality, classification.confidence,
    )

    # Stage 2 — Extraction: render pages to images at adaptive DPI
    progress("rendering", f"Rendering {classification.page_count} pages at adaptive DPI…", 15)
    raw_pages = render_pages(pdf_path, classification)

    # Stage 3 — Extraction: enhance images (denoise / sharpen / deskew)
    progress("preprocessing", "Enhancing page images…", 30)
    pages = preprocess_pages(raw_pages, classification)

    # Stage 4 — Detection: find equation regions
    progress("layout_detection", "Detecting equation regions…", 40)
    regions = detect_equations(pdf_path, pages, classification, book_out)
    logger.info("layout_detection found=%d regions", len(regions))
    progress(
        "layout_detection",
        f"Detected {len(regions)} equation regions across {classification.page_count} pages",
        45,
    )

    if not regions:
        progress("complete", "No equations detected.", 100)
        out_path = write_json_report(
            pdf_path, classification, pages, [], {}, book_out, SCHEMA_VERSION
        )
        logger.info("pipeline_done (0 equations) elapsed=%.1fs", time.monotonic() - start)
        return out_path

    # Stages 5–6 — Extraction (OCR) + Detection (classify content type)
    total = len(regions)
    page_map = {rp.page_number: rp for rp in pages}
    extracted: list[ExtractedEquation] = []

    providers = resolve_providers()
    logger.info("providers_ready roles=%s", list(providers.keys()))

    try:
        for i, region in enumerate(regions):
            pct = 45 + int(50 * (i + 1) / total)
            progress(
                "equation_extraction",
                f"Generating LaTeX for equation {i + 1}/{total} (page {region.page_number})",
                pct,
            )

            crop_image = None
            if region.crop_path:
                crop_abs = book_out / region.crop_path
                if crop_abs.exists():
                    from PIL import Image as _PILImage
                    try:
                        crop_image = _PILImage.open(crop_abs).convert("RGB")
                    except Exception as exc:
                        logger.warning("crop_load_failed eq=%s error=%s", region.equation_id, exc)

            if crop_image is None:
                rp = page_map.get(region.page_number)
                if rp is not None:
                    crop_image = rp.image
                else:
                    logger.warning("no_image_for_region eq=%s", region.equation_id)
                    continue

            cls = classify_region(
                region_type=region.detection_method,
                region_text="",
            )
            role = select_provider(cls.category, config=config)
            provider = providers.get(role, providers.get("generic"))

            sub_crops = _logical_equation_crops(crop_image)

            for sub_idx, sub_crop in enumerate(sub_crops):
                first_ocr, retry_ocr = _recognize_with_provider(
                    provider, sub_crop, category=cls.category
                )

                from equation_extraction_pipeline.common.utils import MULTI_EQ_NOTES
                if set(first_ocr.flags) & MULTI_EQ_NOTES:
                    first_ocr.flags.append("MULTIPLE_EQUATIONS_IN_CROP")

                best_ocr = (
                    retry_ocr
                    if (retry_ocr and retry_ocr.confidence > first_ocr.confidence)
                    else first_ocr
                )

                sub_region = region
                if len(sub_crops) > 1:
                    split_crop_path = region.crop_path
                    if region.crop_path:
                        original_crop = book_out / region.crop_path
                        split_name = f"{original_crop.stem}_sub{sub_idx}{original_crop.suffix}"
                        split_file = original_crop.with_name(split_name)
                        try:
                            sub_crop.save(split_file, format="PNG")
                            split_crop_path = str(split_file.relative_to(book_out))
                        except Exception as exc:
                            logger.warning(
                                "split_crop_save_failed eq=%s sub=%d error=%s",
                                region.equation_id, sub_idx, exc,
                            )
                    sub_region = EquationRegion(
                        page_number=region.page_number,
                        equation_id=f"{region.equation_id}_sub{sub_idx}",
                        label=region.label if sub_idx == 0 else None,
                        bbox=region.bbox,
                        detection_method=region.detection_method,
                        crop_path=split_crop_path,
                    )

                from equation_extraction_pipeline.extraction.ocr_extractor import judge_latex
                verdict = judge_latex(best_ocr.latex, sub_crop) if config.JUDGE_ENABLED else None

                extracted.append(ExtractedEquation(
                    region=sub_region,
                    ocr=first_ocr,
                    verdict=verdict,
                    retry_ocr=retry_ocr,
                ))

            logger.debug(
                "eq=%s category=%s latex=%.40s conf=%.3f status=%s",
                region.equation_id, cls.category,
                best_ocr.latex, best_ocr.confidence,
                extracted[-1].status(),
            )
    finally:
        close_providers(providers)

    # Counting stage — map detected labels → equation numbers
    eq_numbers = build_equation_numbers([e.region for e in extracted])

    # Reporting — write document.json
    progress("output", "Writing document.json…", 97)
    out_path = write_json_report(
        pdf_path, classification, pages, extracted, eq_numbers, book_out, SCHEMA_VERSION
    )

    elapsed = time.monotonic() - start
    with open(out_path) as fh:
        summary = json.load(fh)["document"]["summary"]
    logger.info(
        "pipeline_done pdf=%s elapsed=%.1fs total=%d success=%d uncertain=%d rejected=%d",
        pdf_path.name, elapsed,
        summary["total_equations"], summary["success"],
        summary["uncertain"], summary["rejected"],
    )
    progress("complete", f"Done — {summary['total_equations']} equations extracted.", 100)
    return out_path


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Run the pipeline on every PDF in *input_dir*."""
    src = Path(input_dir) if input_dir else config.INPUT_DIR
    pdfs = sorted(src.glob("*.pdf"))
    if not pdfs:
        logger.warning("no PDFs found in %s", src)
        return []

    results: list[Path] = []
    for pdf_path in pdfs:
        try:
            out = run(pdf_path, output_dir)
            results.append(out)
        except Exception as exc:
            logger.error("batch_item_failed pdf=%s error=%s", pdf_path.name, exc)
    return results


# Alias for callers that import via the package root
run_pipeline = run
