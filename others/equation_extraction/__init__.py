"""Equation extraction stage (feature 008-equation-extraction).

Consumes the Layout, Reading Order, and Text Extraction Contexts (features 005/006/007) and recognizes
every equation in the document: detecting equation regions (display from layout, inline within text),
classifying each into a content category, selecting a configuration-driven recognition provider,
recognizing structured representations (plain text/LaTeX/MathML/structured form), preserving
numbering, reading order, hierarchy, relationships, provenance, and confidence, and validating the
result. Providers are interchangeable behind a common interface and import-guarded. Public API mirrors
the text-extraction stage: ``extract_equations`` computes an Equation Extraction Context, and
``get_or_create_equation_extraction`` adds idempotent sidecar caching for cache-aware reruns.
"""

from equation_extraction.extractor import extract_equations, save_equation_crops
from equation_extraction.manifest import (
    compute_config_hash,
    get_or_create_equation_extraction,
)

__all__ = [
    "extract_equations",
    "get_or_create_equation_extraction",
    "compute_config_hash",
    "save_equation_crops",
]
