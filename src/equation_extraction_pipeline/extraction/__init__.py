"""Extraction sub-package — re-exports of the key stage entry points.

Public surface
--------------
``render_pages``       — PDF → PIL rendering at adaptive DPI (page_renderer)
``preprocess_pages``   — denoise / sharpen / deskew enhancement (text_extractor)
``extract_text``       — reading-ordered text extraction (text_extractor)
``recognize_equation`` — Qwen VL LaTeX OCR via Ollama (ocr_extractor)
``resolve_providers``  — instantiate all configured recognition providers (ocr_extractor)
``close_providers``    — release pooled resources for all providers (ocr_extractor)
"""

from __future__ import annotations

from equation_extraction_pipeline.extraction.ocr_extractor import (
    close_providers,
    recognize_equation,
    resolve_providers,
)
from equation_extraction_pipeline.extraction.page_renderer import render_pages
from equation_extraction_pipeline.extraction.text_extractor import (
    extract_text,
    preprocess_pages,
)

__all__ = [
    "render_pages",
    "preprocess_pages",
    "extract_text",
    "recognize_equation",
    "resolve_providers",
    "close_providers",
]
