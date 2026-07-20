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
from equation_extraction_pipeline.extraction.text_extractor import render_and_preprocess_pages
from equation_extraction_pipeline.ingestion.pdf_loader import classify_pdf
from equation_extraction_pipeline.reporting.json_report import build_document_json, write_json_report

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


def _completeness_gate(
    extracted: list[ExtractedEquation], *, image_math: bool, page_count: int
) -> bool:
    """Risk gate for the inline completeness audit (production policy).

    Returns True when a cheap risk signal warrants an inline audit: the document was flagged
    image-math (equations rendered as rasters — the label scan is unreliable there), or the
    extractions cluster on almost no pages of a multi-page document (anomalously low yield).
    Common label-based digital books trip neither and skip the per-page audit in the hot path.
    """
    if image_math:
        return True
    pages_with_eqs = len({e.region.page_number for e in extracted})
    if page_count >= 10 and pages_with_eqs <= 1:
        return True
    return False


def _audit_completeness(
    extracted: list[ExtractedEquation],
    page_map: dict[int, Any],
    progress: ProgressCallback,
    *,
    image_math: bool = False,
    page_count: int = 0,
    force: bool = False,
) -> dict[str, Any] | None:
    """Run the external GPT per-page completeness audit ("were all equations extracted?").

    Returns the document-level completeness summary, or ``None`` when the audit is
    disabled or the GPT judge is not configured.

    When ``config.JUDGE_COMPLETENESS_GATED`` is on (default), the audit only runs inline if a
    risk signal trips (:func:`_completeness_gate`) — unless ``force`` is set (used for the
    re-audit after a bounded re-extract, so the reported completeness reflects the recovery).
    """
    if not (
        config.JUDGE_ENABLED
        and getattr(config, "JUDGE_PAGE_COMPLETENESS_ENABLED", False)
    ):
        return None

    if (
        not force
        and getattr(config, "JUDGE_COMPLETENESS_GATED", True)
        and not _completeness_gate(extracted, image_math=image_math, page_count=page_count)
    ):
        logger.info("completeness_audit_skipped reason=gated_no_risk_signal")
        return None

    from equation_extraction_pipeline.extraction.gpt_judge import (
        gpt_judge_available,
        judge_page,
        summarize_document,
    )

    if not gpt_judge_available():
        logger.info("completeness_audit_skipped reason=gpt_judge_unconfigured")
        return None

    progress("completeness", "Auditing pages for missed equations…", 96)

    # Scope is book-level: if the document uses reference labels, extraction (and therefore
    # the audit) covers only labeled equations; otherwise it covers every equation. Mirror the
    # detection decision by inspecting how the regions were found.
    mode = (
        "labeled"
        if any(e.region.detection_method == "label" for e in extracted)
        else "unlabeled"
    )
    logger.info("completeness_audit mode=%s", mode)

    by_page: dict[int, list[ExtractedEquation]] = {}
    for e in extracted:
        by_page.setdefault(e.region.page_number, []).append(e)

    page_verdicts = []
    for page_number, eqs_on_page in sorted(by_page.items()):
        rp = page_map.get(page_number)
        if rp is None:
            continue
        extracted_on_page = [
            {
                "equation_id": e.region.equation_id,
                "label": e.region.label,
                "bbox": list(e.region.bbox) if e.region.bbox else None,
                "representation": e.final_latex(),
            }
            for e in eqs_on_page
        ]
        page_verdicts.append(
            judge_page(rp.load_image(), extracted_on_page, page_number=page_number, mode=mode)
        )

    ai_scores = [
        e.verdict.ai_score
        for e in extracted
        if e.verdict is not None and e.verdict.ai_score is not None
    ]
    return summarize_document(page_verdicts, ai_scores)


def _reextract_missed_pages(
    completeness: dict[str, Any],
    extracted: list[ExtractedEquation],
    page_map: dict[int, Any],
    book_out: Path,
    progress: ProgressCallback,
) -> list[ExtractedEquation]:
    """Bounded VLM re-extract over the pages the audit flagged incomplete.

    Reuses the detector's image-math recovery (``_recover_image_math_pages``) to VLM-enumerate
    each incomplete page and build deduplicated crop regions carrying the VLM transcription,
    then judges each. One pass; no loop. Returns the recovered equations (may be empty).
    """
    from equation_extraction_pipeline.detection.equation_label_detector import (
        _recover_image_math_pages,
    )
    from equation_extraction_pipeline.extraction.gpt_judge import (
        gpt_judge_available,
        judge_equation,
    )

    incomplete = {int(p) for p in (completeness.get("incomplete_pages") or [])}
    if not incomplete or not gpt_judge_available():
        return []

    progress("reextract", f"Recovering missed equations on {len(incomplete)} page(s)…", 96)
    new_regions = _recover_image_math_pages(
        list(page_map.values()),
        book_out / "crops",
        incomplete,
        [e.region for e in extracted],
    )

    from PIL import Image as _PILImage

    recovered: list[ExtractedEquation] = []
    for region in new_regions:
        seed = (region.seed_latex or "").strip()
        if not seed:
            continue
        crop_image = None
        if region.crop_path and (book_out / region.crop_path).exists():
            try:
                crop_image = _PILImage.open(book_out / region.crop_path).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                logger.warning("reextract_crop_load_failed eq=%s error=%s",
                               region.equation_id, exc)
        if crop_image is None:
            rp = page_map.get(region.page_number)
            crop_image = rp.load_image() if rp is not None else None

        ocr = OcrResult(
            latex=seed, confidence=0.75, provider="vlm_page_extract",
            flags=["VLM_SEED", "REEXTRACT_PASS"],
        )
        verdict = None
        if (
            config.JUDGE_ENABLED
            and getattr(config, "JUDGE_BACKEND", "portkey") == "portkey"
            and crop_image is not None
        ):
            verdict = judge_equation(crop_image, seed, "unknown")
        recovered.append(ExtractedEquation(region=region, ocr=ocr, verdict=verdict))

    logger.info("reextract recovered=%d equations", len(recovered))
    return recovered


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

    # Stages 2–3 — Extraction: render pages at adaptive DPI and enhance them
    # (denoise / sharpen / deskew), streamed one page at a time and persisted to
    # disk so peak memory stays flat regardless of page count. Rasters are loaded
    # lazily downstream via RenderedPage.load_image().
    progress("rendering", f"Rendering {classification.page_count} pages at adaptive DPI…", 15)
    progress("preprocessing", "Enhancing page images…", 30)
    pages = render_and_preprocess_pages(pdf_path, classification, book_out / "pages")

    # Stage 4 — Detection: find equation regions
    progress("layout_detection", "Detecting equation regions…", 40)
    detection_meta: dict[str, Any] = {}
    regions = detect_equations(
        pdf_path, pages, classification, book_out, detection_meta=detection_meta
    )
    logger.info("layout_detection found=%d regions", len(regions))
    progress(
        "layout_detection",
        f"Detected {len(regions)} equation regions across {classification.page_count} pages",
        45,
    )

    if not regions:
        progress("complete", "No equations detected.", 100)
        out_path = write_json_report(
            build_document_json(pdf_path, classification, pages, [], {}), book_out
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
                    crop_image = rp.load_image()
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
                if region.seed_latex and sub_idx == 0:
                    # Image-math VLM recovery already transcribed this equation from the page
                    # image; adopt it as the candidate instead of re-running OCR on the crop.
                    # The judge still verifies it (its ai_score becomes the authoritative score).
                    first_ocr = OcrResult(
                        latex=region.seed_latex,
                        confidence=0.75,
                        provider="vlm_page_extract",
                        flags=["VLM_SEED"],
                    )
                    retry_ocr = None
                else:
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

                verdict = None
                if config.JUDGE_ENABLED:
                    if getattr(config, "JUDGE_BACKEND", "portkey") == "portkey":
                        from equation_extraction_pipeline.extraction.gpt_judge import (
                            judge_equation,
                        )
                        verdict = judge_equation(sub_crop, best_ocr.latex, cls.category)
                        # Judge-assisted repair (bounded to ONE attempt): when the crop is
                        # fine but the transcription is rejected (typically an omitted line
                        # of a derivation group), the judge's own reading of the crop is a
                        # strong candidate — adopt it only if a fresh judge pass accepts it.
                        correction = (verdict.corrected_representation or "").strip()
                        if (
                            config.JUDGE_REPAIR_ENABLED
                            and not verdict.accepted
                            and verdict.crop_valid
                            and correction
                            and correction != best_ocr.latex.strip()
                        ):
                            repair_verdict = judge_equation(
                                sub_crop, correction, cls.category
                            )
                            if repair_verdict.accepted:
                                # Strictly above the first-pass confidence so final_latex()
                                # (which requires retry > ocr) selects the repair.
                                repaired = OcrResult(
                                    latex=correction,
                                    confidence=min(
                                        0.99,
                                        max(
                                            best_ocr.confidence,
                                            repair_verdict.ai_score or 0.0,
                                        ) + 0.01,
                                    ),
                                    provider=repair_verdict.judge_model or "judge_repair",
                                    flags=["JUDGE_REPAIR"],
                                )
                                repair_verdict.issues.append("judge_repair_applied")
                                retry_ocr = repaired
                                best_ocr = repaired
                                verdict = repair_verdict
                                logger.info(
                                    "judge_repair_applied eq=%s", region.equation_id
                                )
                    else:
                        from equation_extraction_pipeline.extraction.ocr_extractor import (
                            judge_latex,
                        )
                        verdict = judge_latex(best_ocr.latex, sub_crop)

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

    # Completeness audit — external GPT judge inspects each page for missed equations. Gated in
    # production: runs inline only on risk (image-math flagged, or anomalously low yield).
    image_math = bool(detection_meta.get("image_math"))
    completeness = _audit_completeness(
        extracted, page_map, progress,
        image_math=image_math, page_count=classification.page_count,
    )

    # Bounded re-extract: when the (gated) audit found missed equations, recover them with one
    # VLM page pass over the incomplete pages, then re-audit so the report reflects the recovery.
    if (
        completeness
        and not completeness.get("complete", True)
        and int(getattr(config, "JUDGE_REEXTRACT_MAX_PASSES", 1)) > 0
    ):
        recovered = _reextract_missed_pages(completeness, extracted, page_map, book_out, progress)
        if recovered:
            extracted.extend(recovered)
            eq_numbers = build_equation_numbers([e.region for e in extracted])
            completeness = _audit_completeness(
                extracted, page_map, progress,
                image_math=image_math, page_count=classification.page_count, force=True,
            )

    # Reporting — write document.json
    progress("output", "Writing document.json…", 97)
    out_path = write_json_report(
        build_document_json(
            pdf_path, classification, pages, extracted, eq_numbers, completeness
        ),
        book_out,
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
