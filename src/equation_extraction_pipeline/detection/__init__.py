"""Equation detection sub-package.

Exposes the primary public API for equation detection and supporting
classification helpers merged from the legacy equation-extraction-pipeline.
"""

from equation_extraction_pipeline.detection.duplicate_resolver import (
    FINGERPRINT_ALGORITHM,
    INDEX_VERSION,
    DuplicateIndex,
    compute_fingerprint,
)
from equation_extraction_pipeline.detection.equation_block_detector import (
    AMBIGUOUS_THRESHOLD,
    DEFAULT_PROVIDER_BY_CATEGORY,
    FIGURE_REGION_TYPES,
    FORMULA_THRESHOLD,
    TEXT_ONLY_THRESHOLD,
    FormulaScore,
    VisualClassification,
    bbox_list,
    classify_visual_region,
    dedupe_regions,
    is_visual_region,
    overlaps,
    recommended_provider_for_category,
    region_text,
    score_formula_candidate,
    valid_bbox,
)
from equation_extraction_pipeline.detection.equation_block_detector import (
    Classification as EquationClassification,
)
from equation_extraction_pipeline.detection.equation_block_detector import (
    classify_region as classify_equation_region,
)
from equation_extraction_pipeline.detection.equation_label_detector import (
    classify_document,
    classify_page,
    detect_equations,
    scan_equation_labels,
)

__all__ = [
    # duplicate_resolver
    "DuplicateIndex",
    "INDEX_VERSION",
    "compute_fingerprint",
    "FINGERPRINT_ALGORITHM",
    # equation_block_detector — formula scoring
    "FormulaScore",
    "score_formula_candidate",
    "FORMULA_THRESHOLD",
    "AMBIGUOUS_THRESHOLD",
    "TEXT_ONLY_THRESHOLD",
    # equation_block_detector — equation classification
    "EquationClassification",
    "classify_equation_region",
    "DEFAULT_PROVIDER_BY_CATEGORY",
    # equation_block_detector — visual detection
    "FIGURE_REGION_TYPES",
    "is_visual_region",
    "region_text",
    "bbox_list",
    "valid_bbox",
    "overlaps",
    "dedupe_regions",
    # equation_block_detector — visual classification
    "VisualClassification",
    "classify_visual_region",
    "recommended_provider_for_category",
    # equation_label_detector
    "detect_equations",
    "scan_equation_labels",
    "classify_page",
    "classify_document",
]
