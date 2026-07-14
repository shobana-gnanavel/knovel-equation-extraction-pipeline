"""Page rendering and visual asset management — merged module.

Renders PDF pages to PIL images at adaptive DPI, probes image metadata, and
materialises cropped visual regions as on-disk assets.

Merged sources
--------------
* equation-extraction-pipeline/rendering.py              — PDF → PIL rendering, DPI selection, sharpness
* equation-extraction-pipeline/visual_extraction/imaging.py — image-metadata probe helpers
* equation-extraction-pipeline/visual_extraction/assets.py  — visual asset materialisation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium
import structlog
from PIL import Image

from equation_extraction_pipeline.config import settings as config
from equation_extraction_pipeline.domain.models import (
    ClassificationResult,
    ImageMetadata,
    PreprocessingContext,
    RenderedPage,
)

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)

__all__ = [
    # rendering.py
    "render_pages",
    # imaging.py
    "probe",
    "orientation_for",
    # assets.py
    "asset_dir_for",
    "load_page_raster",
    "crop_region",
    "materialize",
]


# ============================================================
# SECTION 1: Merged from visual_extraction/imaging.py
# Image-metadata capture for materialised visual assets
# ============================================================

def orientation_for(width: int, height: int, rotation: float = 0.0) -> str:
    """Classify orientation from dimensions/rotation: ``rotated`` | ``landscape`` | ``upright``."""
    if rotation not in (0.0, 360.0):
        return "rotated"
    if width and height and width > height:
        return "landscape"
    return "upright"


def _size_mode(crop: Any) -> tuple[int, int, str, bool]:
    """Return ``(width, height, color_mode, has_transparency)`` for a PIL image or numpy array."""
    # PIL Image: has .size and .mode
    size = getattr(crop, "size", None)
    mode = getattr(crop, "mode", None)
    if size is not None and mode is not None:
        width, height = int(size[0]), int(size[1])
        has_alpha = mode in {"RGBA", "LA", "PA"} or (
            mode == "P"
            and bool(getattr(crop, "info", {}).get("transparency") is not None)
        )
        return width, height, str(mode), has_alpha
    # numpy array: shape (H, W) or (H, W, C)
    shape = getattr(crop, "shape", None)
    if shape is not None:
        height = int(shape[0])
        width = int(shape[1]) if len(shape) > 1 else 0
        channels = int(shape[2]) if len(shape) > 2 else 1
        mode_map = {1: "L", 3: "RGB", 4: "RGBA"}
        return width, height, mode_map.get(channels, "unknown"), channels == 4
    return 0, 0, "unknown", False


def probe(
    crop: Any,
    *,
    image_format: str = "png",
    source_type: str = "raster",
    rotation: float = 0.0,
    dpi: float = 0.0,
    recompressed: bool = False,
) -> ImageMetadata:
    """Capture :class:`ImageMetadata` from a cropped image; ``None`` → empty metadata."""
    if crop is None:
        return ImageMetadata(image_format=image_format, source_type=source_type)
    width, height, color_mode, has_transparency = _size_mode(crop)
    aspect_ratio = round(width / height, 6) if height else 0.0
    return ImageMetadata(
        width=width,
        height=height,
        dpi=dpi,
        image_format=image_format,
        color_mode=color_mode,
        has_transparency=has_transparency,
        aspect_ratio=aspect_ratio,
        original_resolution=[width, height],
        rotation=rotation,
        orientation=orientation_for(width, height, rotation),
        source_type=source_type,
        recompressed=recompressed,
    )


# ============================================================
# SECTION 2: Merged from visual_extraction/assets.py
# Page rasterization and visual-asset materialisation
# ============================================================

_POINTS_PER_INCH = 72.0
_LOSSLESS_FORMATS = frozenset({"png", "PNG", "tiff", "TIFF", "bmp", "BMP", "webp"})


def asset_dir_for(book_id: str) -> Path:
    """Per-document asset directory ``<KNOVEL_OUTPUT_DIR>/<book_id>/visuals/``."""
    return Path(config.KNOVEL_OUTPUT_DIR) / book_id / "visuals"


def _derived_artifact(preprocessing: PreprocessingContext | None, page_no: int) -> str | None:
    if preprocessing is None:
        return None
    for page in preprocessing.pages:
        if page.page_no == page_no:
            return page.derived_artifact
    return None


def _render_via_backend(pdf_path: Path, page_no: int, *, dpi: int) -> Any:
    """Render a page to a PIL image via ``pipeline.pdf_backend`` (pypdfium2). Best-effort → ``None``."""
    try:  # pragma: no cover - exercised only with a real PDF + backend
        from pipeline.pdf_backend import PdfDocument, render_page_image  # type: ignore[import]

        zoom = max(dpi / _POINTS_PER_INCH, 1.0)
        with PdfDocument(str(pdf_path)) as document:
            index = page_no - 1 if page_no >= 1 else page_no
            if index < 0 or index >= len(document):
                return None
            array = render_page_image(document[index], zoom)
        return Image.fromarray(array)
    except Exception as exc:  # pragma: no cover
        _slog.warning("visual_render_failed", page_no=page_no, error=str(exc))
        return None


def load_page_raster(
    pdf_path: Path | None,
    page_no: int,
    *,
    preprocessing: PreprocessingContext | None,
    config: Any,
    cache: dict[int, Any] | None = None,
) -> Any:
    """Return a PIL image for ``page_no`` (corrected raster preferred, else backend render), or ``None``.

    Caches per page when a ``cache`` dict is supplied so a page is loaded/rendered at most once.
    """
    if cache is not None and page_no in cache:
        return cache[page_no]

    image = None
    artifact = _derived_artifact(preprocessing, page_no)
    if artifact and Path(artifact).exists():
        try:
            image = Image.open(artifact)
            image.load()
        except Exception as exc:  # corrupt artifact → try render
            _slog.warning("visual_artifact_unreadable", page_no=page_no, error=str(exc))
            image = None

    if image is None and pdf_path is not None:
        dpi = int(getattr(config, "KNOVEL_VISUAL_RENDER_DPI", 200))
        image = _render_via_backend(Path(pdf_path), page_no, dpi=dpi)

    if cache is not None:
        cache[page_no] = image
    return image


def crop_region(image: Any, geom: tuple[float, float] | None, bbox: list[float]) -> Any:
    """Crop a region (page-point bbox) from a page image, scaling to pixel size. ``None`` on failure."""
    if image is None or geom is None or len(bbox) != 4:
        return None
    page_w, page_h = geom
    if page_w <= 0 or page_h <= 0:
        return None
    try:
        px_w, px_h = image.size
        sx, sy = px_w / page_w, px_h / page_h
        box = (
            max(0, int(bbox[0] * sx)),
            max(0, int(bbox[1] * sy)),
            min(px_w, int(bbox[2] * sx)),
            min(px_h, int(bbox[3] * sy)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return None
        return image.crop(box)
    except Exception:
        return None


def materialize(
    crop: Any,
    *,
    visual_id: str,
    asset_dir: Path,
    config: Any,
    source_type: str = "raster",
    rotation: float = 0.0,
    dpi: float = 0.0,
) -> tuple[str | None, ImageMetadata]:
    """Write ``crop`` losslessly under ``asset_dir`` and return ``(asset_path, ImageMetadata)``.

    Preserves color mode and transparency; uses a lossless format by default so no unnecessary
    recompression occurs (``recompressed=False``). ``crop`` of ``None`` yields ``(None, empty metadata)``.
    """
    image_format = str(getattr(config, "KNOVEL_VISUAL_IMAGE_FORMAT", "png")).lower()
    if crop is None:
        return None, ImageMetadata(image_format=image_format, source_type=source_type)

    recompressed = image_format not in {f.lower() for f in _LOSSLESS_FORMATS}
    metadata = probe(
        crop,
        image_format=image_format,
        source_type=source_type,
        rotation=rotation,
        dpi=dpi,
        recompressed=recompressed,
    )

    asset_path: str | None = None
    try:
        asset_dir.mkdir(parents=True, exist_ok=True)
        out_path = asset_dir / f"{visual_id}.{image_format}"
        save_kwargs: dict[str, Any] = {}
        if recompressed:
            save_kwargs["quality"] = int(getattr(config, "KNOVEL_VISUAL_IMAGE_QUALITY", 95))
        crop.save(out_path, **save_kwargs)
        asset_path = str(out_path)
    except Exception as exc:
        _slog.warning("visual_asset_write_failed", visual_id=visual_id, error=str(exc))

    return asset_path, metadata


# ============================================================
# SECTION 3: Merged from rendering.py
# PDF page rendering, DPI selection, Laplacian sharpness scoring
#
# Renders every page of a PDF to a PIL image at a DPI chosen
# based on the document's modality classification:
#   scanned → RENDER_DPI_SCANNED  (default 300)
#   digital → RENDER_DPI_DIGITAL  (default 216)
#   hybrid  → RENDER_DPI_SCANNED  (conservative)
# ============================================================

_QUALITY_CLIP_MAX = 500.0
"""Laplacian variance above this is clipped before normalisation."""


def _compute_quality_score(image: Image.Image) -> float:
    """Return normalised sharpness score (0.0 – 1.0) via Laplacian variance."""
    try:
        import cv2

        gray = np.array(image.convert("L"))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return round(min(lap_var, _QUALITY_CLIP_MAX) / _QUALITY_CLIP_MAX, 4)
    except Exception as exc:
        logger.debug("quality_score_failed error=%s", exc)
        return 0.0


def _dpi_for_modality(modality: str) -> int:
    if modality == "digital":
        return config.RENDER_DPI_DIGITAL
    return config.RENDER_DPI_SCANNED


def render_pages(
    pdf_path: Path,
    classification: ClassificationResult,
    *,
    page_numbers: list[int] | None = None,
) -> list[RenderedPage]:
    """Render PDF pages to PIL images at adaptive DPI.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    classification:
        Result from the classifier — drives DPI selection.
    page_numbers:
        Optional 1-based list of pages to render.  When omitted all pages are
        rendered.

    Returns
    -------
    list[RenderedPage]
        One entry per rendered page, in ascending page-number order.
    """
    pdf_path = Path(pdf_path)
    dpi = _dpi_for_modality(classification.modality)
    scale = dpi / config.PDF_POINTS_PER_INCH

    logger.info(
        "rendering pdf=%s modality=%s dpi=%d pages=%d",
        pdf_path.name,
        classification.modality,
        dpi,
        classification.page_count,
    )

    rendered: list[RenderedPage] = []

    # PdfDocument is not a context manager in pypdfium2 4.30+. Keep resource
    # cleanup explicit for compatibility with current and older versions.
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        indices = (
            [pn - 1 for pn in page_numbers] if page_numbers else range(len(doc))
        )

        for idx in indices:
            page = doc[idx]
            bitmap = page.render(scale=scale, rotation=0)
            pil_image = bitmap.to_pil()
            w, h = pil_image.size
            quality = _compute_quality_score(pil_image)

            rp = RenderedPage(
                page_number=idx + 1,
                image=pil_image,
                dpi=dpi,
                quality_score=quality,
                width_px=w,
                height_px=h,
            )
            rendered.append(rp)
            logger.debug(
                "rendered page=%d dpi=%d size=%dx%d quality=%.3f",
                idx + 1,
                dpi,
                w,
                h,
                quality,
            )
    finally:
        doc.close()

    logger.info("rendering_done total_pages=%d", len(rendered))
    return rendered
