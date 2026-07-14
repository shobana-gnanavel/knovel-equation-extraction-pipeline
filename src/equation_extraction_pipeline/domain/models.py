"""Typed data models for the equation-extraction pipeline.

Merges the standalone equation-extraction models (models.py) with the pipeline
infrastructure models (pipeline/models.py) into a single canonical module.

Sections
--------
1. Core equation-extraction models      — from models.py
2. Ingestion / identity models          — from pipeline/models.py
3. Classification models                — from pipeline/models.py
4. Preprocessing models                 — from pipeline/models.py
5. Layout models                        — from pipeline/models.py
6. Reading-order models                 — from pipeline/models.py
7. Text-extraction models               — from pipeline/models.py
8. Visual-extraction models             — from pipeline/models.py
9. Table models                         — from pipeline/models.py
10. Run / operational models            — from pipeline/models.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Schema / context version constants
# ---------------------------------------------------------------------------

MANIFEST_VERSION = "1.0.0"
CONTEXT_VERSION = "1.0.0"
PREPROCESS_CONTEXT_VERSION = "1.0.0"
LAYOUT_CONTEXT_VERSION = "1.0.0"
READING_ORDER_CONTEXT_VERSION = "1.0.0"
TEXT_EXTRACTION_CONTEXT_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Section 1 — Core equation-extraction models
# ---------------------------------------------------------------------------

# The six equation content categories used across detection and validation.
EQUATION_CATEGORIES: tuple[str, ...] = (
    "mathematical_equation",
    "engineering_formula",
    "statistical_expression",
    "chemical_equation",
    "chemical_structure",
    "unknown",
)


@dataclass
class ClassificationResult:
    """Outcome of PDF modality classification."""

    modality: str
    """'scanned' | 'digital' | 'hybrid'"""

    confidence: float
    """0.0 – 1.0 confidence in the modality verdict."""

    page_count: int
    """Total number of pages in the PDF."""

    sampled_pages: int = 0
    """Number of pages actually inspected during classification."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "confidence": round(self.confidence, 4),
            "page_count": self.page_count,
            "sampled_pages": self.sampled_pages,
        }


@dataclass
class RenderedPage:
    """A single PDF page rendered to a PIL image."""

    page_number: int
    """1-based page number."""

    image: Any
    """PIL.Image.Image — kept in memory, not serialized."""

    dpi: int
    """DPI used for this render."""

    quality_score: float
    """Laplacian variance (sharpness proxy), normalised to 0 – 1."""

    width_px: int = 0
    height_px: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "dpi": self.dpi,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "quality_score": round(self.quality_score, 4),
        }


@dataclass
class EquationRegion:
    """A detected equation region on a single page."""

    page_number: int
    """1-based page number."""

    equation_id: str
    """Unique identifier within the document, e.g. 'eq_0_p2_12_2_1'."""

    label: str | None
    """Human-readable label from the PDF margin ('12.2.1'), or None."""

    bbox: tuple[float, float, float, float]
    """Bounding box (x0, y0, x1, y1) in PDF points."""

    detection_method: str = "label"
    """'label' (regex) | 'ml' (layout model)."""

    crop_path: str | None = None
    """Relative path to the saved crop PNG, set after crops are written."""

    def bbox_dict(self) -> dict[str, float]:
        x0, y0, x1, y1 = self.bbox
        return {
            "x": round(x0, 2),
            "y": round(y0, 2),
            "width": round(x1 - x0, 2),
            "height": round(y1 - y0, 2),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "equation_id": self.equation_id,
            "page_number": self.page_number,
            "label": self.label,
            "bbox": self.bbox_dict(),
            "detection_method": self.detection_method,
            "crop_path": self.crop_path,
        }


@dataclass
class OcrResult:
    """LaTeX OCR result from the Qwen VL model."""

    latex: str
    """Extracted LaTeX string."""

    confidence: float
    """Estimated confidence, 0.0 – 1.0."""

    provider: str = "qwen_vl"
    """Model identifier, e.g. 'qwen2.5vl:7b'."""

    flags: list[str] = field(default_factory=list)
    """Diagnostic flags, e.g. ['USING_PROXY_CONFIDENCE', 'RETRY_TRIGGERED']."""

    raw_response: str = ""
    """Full raw model response (for debugging)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "latex": self.latex,
            "confidence": round(self.confidence, 4),
            "provider": self.provider,
            "flags": self.flags,
        }


@dataclass
class JudgeVerdict:
    """Quality verdict from the LLM judge."""

    accepted: bool
    """True if the LaTeX is judged to correctly represent the equation image."""

    score: float
    """Judge confidence in the verdict, 0.0 – 1.0."""

    reason: str
    """Short human-readable explanation from the judge."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "score": round(self.score, 4),
            "reason": self.reason,
        }


@dataclass
class InlineEquation:
    """A short inline equation span found within a text block."""

    page_number: int
    parent_region_id: str
    text: str
    latex: str | None = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "parent_region_id": self.parent_region_id,
            "text": self.text,
            "latex": self.latex,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class ExtractedEquation:
    """All information for one extracted equation, ready for document.json."""

    region: EquationRegion
    ocr: OcrResult
    verdict: JudgeVerdict | None = None
    retry_ocr: OcrResult | None = None
    validation_flags: list[str] = field(default_factory=list)

    def final_latex(self) -> str:
        if self.retry_ocr and self.retry_ocr.confidence > self.ocr.confidence:
            return self.retry_ocr.latex
        return self.ocr.latex

    def final_confidence(self) -> float:
        best_ocr_conf = max(
            self.ocr.confidence,
            self.retry_ocr.confidence if self.retry_ocr else 0.0,
        )
        if self.verdict:
            return round((best_ocr_conf + self.verdict.score) / 2, 4)
        return round(best_ocr_conf, 4)

    def status(self) -> str:
        if self.verdict and not self.verdict.accepted:
            return "REJECTED"
        from equation_extraction_pipeline.config import settings as _c
        if self.final_confidence() < _c.RECOGNITION_MIN_CONFIDENCE:
            return "UNCERTAIN"
        return "SUCCESS"

    def to_dict(self) -> dict[str, Any]:
        r = self.region
        return {
            "equation_id": r.equation_id,
            "page_number": r.page_number,
            "label": r.label,
            "bbox": r.bbox_dict(),
            "detection_method": r.detection_method,
            "rendering": {},
            "crop": {"path": r.crop_path},
            "ocr": self.ocr.to_dict(),
            "retry": self.retry_ocr.to_dict() if self.retry_ocr else None,
            "judge": self.verdict.to_dict() if self.verdict else None,
            "final": {
                "latex": self.final_latex(),
                "status": self.status(),
                "overall_confidence": self.final_confidence(),
            },
            "validation_flags": self.validation_flags,
        }


# ---------------------------------------------------------------------------
# Section 2 — Ingestion / identity models
# ---------------------------------------------------------------------------

@dataclass
class Provenance:
    """Where an element came from. Lists accumulate across cross-page merges."""

    source_extractors: list[str] = field(default_factory=list)
    source_pages: list[int] = field(default_factory=list)


@dataclass
class DocumentIdentity:
    """Stable identity of a document, independent of filename/location."""

    fingerprint: str
    fingerprint_algorithm: str = "sha256"
    logical_id: str = ""
    logical_id_source: str = "metadata"  # "metadata" | "content_fallback"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentIdentity:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: data[k] for k in known if k in data})


@dataclass
class DocumentMetadata:
    """Normalized bibliographic metadata used to derive identity."""

    title: str = ""
    authors: list[str] = field(default_factory=list)
    publishers: list[str] = field(default_factory=list)
    edition: str = ""
    identifiers: dict[str, str] = field(default_factory=dict)
    raw: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentMetadata:
        return cls(
            title=data.get("title", ""),
            authors=list(data.get("authors", [])),
            publishers=list(data.get("publishers", [])),
            edition=data.get("edition", ""),
            identifiers=dict(data.get("identifiers", {})),
            raw=dict(data.get("raw", {})),
        )


@dataclass
class DuplicateRelationship:
    """Link from a detected duplicate to the original document."""

    duplicate_type: str  # "binary" | "logical"
    original_logical_id: str
    original_fingerprint: str
    original_manifest_ref: str
    detected_in: str = "index"  # "index" | "batch"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DuplicateRelationship:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: data[k] for k in known if k in data})


@dataclass
class IngestionManifest:
    """Per-document ingestion record: identity, metadata, provenance, outcome."""

    manifest_version: str
    provenance: Provenance
    source_path: str
    source_filename: str
    ingested_at: datetime
    pipeline_run_id: str
    outcome: str  # "ingested" | "duplicate" | "failed"
    identity: DocumentIdentity | None = None
    metadata: DocumentMetadata | None = None
    duplicate_of: DuplicateRelationship | None = None
    failure_reason: str | None = None
    page_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "identity": self.identity.to_dict() if self.identity is not None else None,
            "metadata": self.metadata.to_dict() if self.metadata is not None else None,
            "provenance": asdict(self.provenance),
            "source_path": self.source_path,
            "source_filename": self.source_filename,
            "ingested_at": self.ingested_at.isoformat(),
            "pipeline_run_id": self.pipeline_run_id,
            "outcome": self.outcome,
            "duplicate_of": self.duplicate_of.to_dict() if self.duplicate_of is not None else None,
            "failure_reason": self.failure_reason,
            "page_count": self.page_count,
        }


# ---------------------------------------------------------------------------
# Section 3 — Classification models
# ---------------------------------------------------------------------------

@dataclass
class DetectedLanguage:
    """One detected language and its confidence."""

    language: str  # ISO-639 code, or "und" when undetermined
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectedLanguage:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: data[k] for k in known if k in data})


@dataclass
class CategoryCandidate:
    """A scored category considered during classification."""

    category: str
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoryCandidate:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: data[k] for k in known if k in data})


@dataclass
class ClassificationContext:
    """Document-level classification record (modality, category, language, strategy)."""

    context_version: str = CONTEXT_VERSION
    modality: str = "scanned"  # "digital" | "scanned" | "hybrid"
    modality_confidence: float = 0.0
    page_type_proportions: dict[str, float] = field(default_factory=dict)
    category: str = "unknown"
    category_confidence: float = 0.0
    category_candidates: list[CategoryCandidate] = field(default_factory=list)
    category_fallback_reason: str | None = None
    dominant_language: str = "und"
    detected_languages: list[DetectedLanguage] = field(default_factory=list)
    layout_complexity: str = "simple"  # "simple" | "moderate" | "complex"
    layout_confidence: float = 0.0
    characteristics: dict[str, Any] = field(default_factory=dict)
    recommended_strategy: str = ""
    overall_confidence: float = 0.0
    sampling: dict[str, Any] = field(default_factory=dict)
    signals_used: list[str] = field(default_factory=list)
    outcome: str = "classified"  # "classified" | "degraded" | "failed"
    degradation_notes: list[str] = field(default_factory=list)
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_version": self.context_version,
            "modality": self.modality,
            "modality_confidence": self.modality_confidence,
            "page_type_proportions": dict(self.page_type_proportions),
            "category": self.category,
            "category_confidence": self.category_confidence,
            "category_candidates": [c.to_dict() for c in self.category_candidates],
            "category_fallback_reason": self.category_fallback_reason,
            "dominant_language": self.dominant_language,
            "detected_languages": [lang.to_dict() for lang in self.detected_languages],
            "layout_complexity": self.layout_complexity,
            "layout_confidence": self.layout_confidence,
            "characteristics": dict(self.characteristics),
            "recommended_strategy": self.recommended_strategy,
            "overall_confidence": self.overall_confidence,
            "sampling": dict(self.sampling),
            "signals_used": list(self.signals_used),
            "outcome": self.outcome,
            "degradation_notes": list(self.degradation_notes),
            "config_hash": self.config_hash,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Section 4 — Preprocessing models
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingStatistics:
    """Document-level preprocessing aggregates."""

    total_pages: int = 0
    outcome_counts: dict[str, int] = field(default_factory=dict)
    operation_counts: dict[str, int] = field(default_factory=dict)
    strategy_counts: dict[str, int] = field(default_factory=dict)
    ocr_required_count: int = 0
    failure_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "outcome_counts": dict(self.outcome_counts),
            "operation_counts": dict(self.operation_counts),
            "strategy_counts": dict(self.strategy_counts),
            "ocr_required_count": self.ocr_required_count,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingStatistics:
        return cls(
            total_pages=data.get("total_pages", 0),
            outcome_counts=dict(data.get("outcome_counts", {})),
            operation_counts=dict(data.get("operation_counts", {})),
            strategy_counts=dict(data.get("strategy_counts", {})),
            ocr_required_count=data.get("ocr_required_count", 0),
            failure_count=data.get("failure_count", 0),
        )


@dataclass
class PreprocessingContext:
    """Document-level preprocessing record."""

    context_version: str = PREPROCESS_CONTEXT_VERSION
    outcome: str = "preprocessed"  # "preprocessed" | "degraded" | "failed"
    pages: list[Any] = field(default_factory=list)  # list[PagePreprocessingResult]
    statistics: PreprocessingStatistics = field(default_factory=PreprocessingStatistics)
    available_operations: list[str] = field(default_factory=list)
    degradation_notes: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_version": self.context_version,
            "outcome": self.outcome,
            "pages": [p.to_dict() if hasattr(p, "to_dict") else p for p in self.pages],
            "statistics": self.statistics.to_dict(),
            "available_operations": list(self.available_operations),
            "degradation_notes": list(self.degradation_notes),
            "failure_reason": self.failure_reason,
            "config_hash": self.config_hash,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Section 5 — Layout models
# ---------------------------------------------------------------------------

@dataclass
class LayoutRegion:
    """One detected layout region on a page."""

    region_id: str
    region_type: str
    category: str
    bbox: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    confidence_source: str = "heuristic"
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    column_index: int | None = None
    order_index: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)
    validation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "region_type": self.region_type,
            "category": self.category,
            "bbox": dict(self.bbox),
            "confidence": self.confidence,
            "confidence_source": self.confidence_source,
            "parent_id": self.parent_id,
            "children": list(self.children),
            "column_index": self.column_index,
            "order_index": self.order_index,
            "attributes": dict(self.attributes),
            "provenance": asdict(self.provenance),
            "validation_notes": list(self.validation_notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayoutRegion:
        provenance = Provenance(**data.get("provenance", {}))
        known = {f for f in cls.__dataclass_fields__ if f not in {"provenance", "bbox"}}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(provenance=provenance, bbox=dict(data.get("bbox", {})), **kwargs)


@dataclass
class LayoutStatistics:
    """Document-level layout aggregates."""

    total_pages: int = 0
    region_counts_by_type: dict[str, int] = field(default_factory=dict)
    page_counts_by_column_model: dict[str, int] = field(default_factory=dict)
    low_confidence_count: int = 0
    overlaps_resolved: int = 0
    duplicates_resolved: int = 0
    no_region_pages: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "region_counts_by_type": dict(self.region_counts_by_type),
            "page_counts_by_column_model": dict(self.page_counts_by_column_model),
            "low_confidence_count": self.low_confidence_count,
            "overlaps_resolved": self.overlaps_resolved,
            "duplicates_resolved": self.duplicates_resolved,
            "no_region_pages": self.no_region_pages,
            "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayoutStatistics:
        dict_fields = {"region_counts_by_type", "page_counts_by_column_model"}
        known = {f for f in cls.__dataclass_fields__ if f not in dict_fields}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(
            region_counts_by_type=dict(data.get("region_counts_by_type", {})),
            page_counts_by_column_model=dict(data.get("page_counts_by_column_model", {})),
            **kwargs,
        )


@dataclass
class PageLayout:
    """Per-page layout record."""

    page_no: int = 0
    width: float = 0.0
    height: float = 0.0
    orientation: str = "portrait"
    column_model: str = "single_column"
    column_count: int = 1
    column_zones: list[dict[str, float]] = field(default_factory=list)
    geometry_flags: list[str] = field(default_factory=list)
    regions: list[LayoutRegion] = field(default_factory=list)
    outcome: str = "analyzed"
    confidence: float = 0.0
    page_representation: str = "text_layer"
    visualization_artifact: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "width": self.width,
            "height": self.height,
            "orientation": self.orientation,
            "column_model": self.column_model,
            "column_count": self.column_count,
            "column_zones": [dict(z) for z in self.column_zones],
            "geometry_flags": list(self.geometry_flags),
            "regions": [r.to_dict() for r in self.regions],
            "outcome": self.outcome,
            "confidence": self.confidence,
            "page_representation": self.page_representation,
            "visualization_artifact": self.visualization_artifact,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageLayout:
        regions = [LayoutRegion.from_dict(r) for r in data.get("regions", [])]
        known = {f for f in cls.__dataclass_fields__ if f not in {"regions", "column_zones"}}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(regions=regions, column_zones=[dict(z) for z in data.get("column_zones", [])], **kwargs)


@dataclass
class LayoutContext:
    """Document-level layout record."""

    version: str = LAYOUT_CONTEXT_VERSION
    outcome: str = "analyzed"
    pages: list[PageLayout] = field(default_factory=list)
    statistics: LayoutStatistics = field(default_factory=LayoutStatistics)
    backend: str = ""
    supported_region_types: list[str] = field(default_factory=list)
    config_hash: str = ""
    notes: list[str] = field(default_factory=list)
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "outcome": self.outcome,
            "pages": [p.to_dict() for p in self.pages],
            "statistics": self.statistics.to_dict(),
            "backend": self.backend,
            "supported_region_types": list(self.supported_region_types),
            "config_hash": self.config_hash,
            "notes": list(self.notes),
            "failure_reason": self.failure_reason,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LayoutContext:
        pages = [PageLayout.from_dict(p) for p in data.get("pages", [])]
        statistics = LayoutStatistics.from_dict(data.get("statistics", {}))
        known = {f for f in cls.__dataclass_fields__ if f not in {"pages", "statistics"}}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(pages=pages, statistics=statistics, **kwargs)

    @classmethod
    def from_json(cls, text: str) -> LayoutContext:
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Section 6 — Reading-order models
# ---------------------------------------------------------------------------

@dataclass
class ReadingOrderEntry:
    """One ordered reference to a layout region."""

    region_id: str
    page_no: int = 0
    reading_position: int = -1
    page_position: int = 0
    column_index: int | None = None
    role: str = "body"
    in_body_flow: bool = True
    structural_parent_id: str | None = None
    confidence: float = 0.0
    low_confidence: bool = False
    provenance: Provenance = field(default_factory=Provenance)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "page_no": self.page_no,
            "reading_position": self.reading_position,
            "page_position": self.page_position,
            "column_index": self.column_index,
            "role": self.role,
            "in_body_flow": self.in_body_flow,
            "structural_parent_id": self.structural_parent_id,
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
            "provenance": asdict(self.provenance),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReadingOrderEntry:
        provenance = Provenance(**data.get("provenance", {}))
        known = {f for f in cls.__dataclass_fields__ if f not in {"provenance", "notes"}}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(provenance=provenance, notes=list(data.get("notes", [])), **kwargs)


@dataclass
class ReadingOrderAssociation:
    """One cross-reference / structural link over region references."""

    association_type: str
    source_region_id: str
    target_region_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "proximity"
    orphan: bool = False
    provenance: Provenance = field(default_factory=Provenance)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "association_type": self.association_type,
            "source_region_id": self.source_region_id,
            "target_region_ids": list(self.target_region_ids),
            "confidence": self.confidence,
            "method": self.method,
            "orphan": self.orphan,
            "provenance": asdict(self.provenance),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReadingOrderAssociation:
        provenance = Provenance(**data.get("provenance", {}))
        known = {f for f in cls.__dataclass_fields__ if f not in {"provenance", "target_region_ids", "notes"}}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(
            provenance=provenance,
            target_region_ids=list(data.get("target_region_ids", [])),
            notes=list(data.get("notes", [])),
            **kwargs,
        )


@dataclass
class PageReadingOrder:
    """Per-page reading order."""

    page_no: int = 0
    entries: list[ReadingOrderEntry] = field(default_factory=list)
    column_model: str = "single_column"
    traversal_direction: str = "ltr"
    outcome: str = "ordered"
    confidence: float = 0.0
    fallback_used: bool = False
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "entries": [e.to_dict() for e in self.entries],
            "column_model": self.column_model,
            "traversal_direction": self.traversal_direction,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "fallback_used": self.fallback_used,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageReadingOrder:
        entries = [ReadingOrderEntry.from_dict(e) for e in data.get("entries", [])]
        known = {f for f in cls.__dataclass_fields__ if f != "entries"}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(entries=entries, **kwargs)


@dataclass
class ReadingOrderStatistics:
    """Document-level reading-order aggregates."""

    total_pages: int = 0
    total_regions_ordered: int = 0
    unassigned_regions: int = 0
    excluded_regions: int = 0
    associations_by_type: dict[str, int] = field(default_factory=dict)
    multi_column_pages: int = 0
    mixed_pages: int = 0
    cross_page_continuations: int = 0
    low_confidence_count: int = 0
    orphans: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "total_regions_ordered": self.total_regions_ordered,
            "unassigned_regions": self.unassigned_regions,
            "excluded_regions": self.excluded_regions,
            "associations_by_type": dict(self.associations_by_type),
            "multi_column_pages": self.multi_column_pages,
            "mixed_pages": self.mixed_pages,
            "cross_page_continuations": self.cross_page_continuations,
            "low_confidence_count": self.low_confidence_count,
            "orphans": self.orphans,
            "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReadingOrderStatistics:
        known = {f for f in cls.__dataclass_fields__ if f != "associations_by_type"}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(associations_by_type=dict(data.get("associations_by_type", {})), **kwargs)


@dataclass
class ReadingOrderContext:
    """Document-level reading-order record."""

    version: str = READING_ORDER_CONTEXT_VERSION
    outcome: str = "ordered"
    document_sequence: list[str] = field(default_factory=list)
    pages: list[PageReadingOrder] = field(default_factory=list)
    hierarchy: list[dict[str, Any]] = field(default_factory=list)
    associations: list[ReadingOrderAssociation] = field(default_factory=list)
    statistics: ReadingOrderStatistics = field(default_factory=ReadingOrderStatistics)
    strategy: str = ""
    traversal_direction: str = "ltr"
    config_hash: str = ""
    notes: list[str] = field(default_factory=list)
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "outcome": self.outcome,
            "document_sequence": list(self.document_sequence),
            "pages": [p.to_dict() for p in self.pages],
            "hierarchy": list(self.hierarchy),
            "associations": [a.to_dict() for a in self.associations],
            "statistics": self.statistics.to_dict(),
            "strategy": self.strategy,
            "traversal_direction": self.traversal_direction,
            "config_hash": self.config_hash,
            "notes": list(self.notes),
            "failure_reason": self.failure_reason,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReadingOrderContext:
        pages = [PageReadingOrder.from_dict(p) for p in data.get("pages", [])]
        associations = [ReadingOrderAssociation.from_dict(a) for a in data.get("associations", [])]
        statistics = ReadingOrderStatistics.from_dict(data.get("statistics", {}))
        known = {
            f for f in cls.__dataclass_fields__
            if f not in {"pages", "associations", "statistics", "document_sequence", "hierarchy", "notes"}
        }
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(
            document_sequence=list(data.get("document_sequence", [])),
            pages=pages,
            hierarchy=list(data.get("hierarchy", [])),
            associations=associations,
            statistics=statistics,
            notes=list(data.get("notes", [])),
            **kwargs,
        )

    @classmethod
    def from_json(cls, text: str) -> ReadingOrderContext:
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Section 7 — Text-extraction models
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    """One extracted text-bearing region's content."""

    block_id: str
    region_id: str
    page_no: int = 0
    reading_position: int = -1
    page_position: int = 0
    structural_parent_id: str | None = None
    column_index: int | None = None
    bbox: list[float] = field(default_factory=list)
    text: str = ""
    role: str = "paragraph"
    method: str = "native"
    language: str | None = None
    confidence: float = 0.0
    low_confidence: bool = False
    char_count: int = 0
    normalization: list[str] = field(default_factory=list)
    inline_formats: list[str] = field(default_factory=list)
    validation_flags: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "region_id": self.region_id,
            "page_no": self.page_no,
            "reading_position": self.reading_position,
            "page_position": self.page_position,
            "structural_parent_id": self.structural_parent_id,
            "column_index": self.column_index,
            "bbox": list(self.bbox),
            "text": self.text,
            "role": self.role,
            "method": self.method,
            "language": self.language,
            "confidence": self.confidence,
            "low_confidence": self.low_confidence,
            "char_count": self.char_count,
            "normalization": list(self.normalization),
            "inline_formats": list(self.inline_formats),
            "validation_flags": list(self.validation_flags),
            "provenance": asdict(self.provenance),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextBlock:
        provenance = Provenance(**data.get("provenance", {}))
        list_fields = {"bbox", "normalization", "inline_formats", "validation_flags", "notes"}
        known = {f for f in cls.__dataclass_fields__ if f not in {"provenance"} | list_fields}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(
            provenance=provenance,
            bbox=list(data.get("bbox", [])),
            normalization=list(data.get("normalization", [])),
            inline_formats=list(data.get("inline_formats", [])),
            validation_flags=list(data.get("validation_flags", [])),
            notes=list(data.get("notes", [])),
            **kwargs,
        )


@dataclass
class PageTextExtraction:
    """Per-page text extraction."""

    page_no: int = 0
    blocks: list[TextBlock] = field(default_factory=list)
    outcome: str = "extracted"
    method_counts: dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0
    fallback_used: bool = False
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_no": self.page_no,
            "blocks": [b.to_dict() for b in self.blocks],
            "outcome": self.outcome,
            "method_counts": dict(self.method_counts),
            "confidence": self.confidence,
            "fallback_used": self.fallback_used,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PageTextExtraction:
        blocks = [TextBlock.from_dict(b) for b in data.get("blocks", [])]
        known = {f for f in cls.__dataclass_fields__ if f not in {"blocks", "method_counts"}}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(blocks=blocks, method_counts=dict(data.get("method_counts", {})), **kwargs)


@dataclass
class TextExtractionStatistics:
    """Document-level text-extraction aggregates."""

    total_pages: int = 0
    total_blocks: int = 0
    total_characters: int = 0
    blocks_by_method: dict[str, int] = field(default_factory=dict)
    blocks_by_role: dict[str, int] = field(default_factory=dict)
    low_confidence_count: int = 0
    normalization_counts: dict[str, int] = field(default_factory=dict)
    validation_counts: dict[str, int] = field(default_factory=dict)
    fallback_count: int = 0
    failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "total_blocks": self.total_blocks,
            "total_characters": self.total_characters,
            "blocks_by_method": dict(self.blocks_by_method),
            "blocks_by_role": dict(self.blocks_by_role),
            "low_confidence_count": self.low_confidence_count,
            "normalization_counts": dict(self.normalization_counts),
            "validation_counts": dict(self.validation_counts),
            "fallback_count": self.fallback_count,
            "failures": self.failures,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextExtractionStatistics:
        dict_fields = {"blocks_by_method", "blocks_by_role", "normalization_counts", "validation_counts"}
        known = {f for f in cls.__dataclass_fields__ if f not in dict_fields}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(
            blocks_by_method=dict(data.get("blocks_by_method", {})),
            blocks_by_role=dict(data.get("blocks_by_role", {})),
            normalization_counts=dict(data.get("normalization_counts", {})),
            validation_counts=dict(data.get("validation_counts", {})),
            **kwargs,
        )


@dataclass
class TextExtractionContext:
    """Document-level text-extraction record."""

    version: str = TEXT_EXTRACTION_CONTEXT_VERSION
    outcome: str = "extracted"
    blocks: list[TextBlock] = field(default_factory=list)
    pages: list[PageTextExtraction] = field(default_factory=list)
    statistics: TextExtractionStatistics = field(default_factory=TextExtractionStatistics)
    native_engine: str = ""
    ocr_engine: str = ""
    languages: list[str] = field(default_factory=list)
    config_hash: str = ""
    notes: list[str] = field(default_factory=list)
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "outcome": self.outcome,
            "blocks": [b.to_dict() for b in self.blocks],
            "pages": [p.to_dict() for p in self.pages],
            "statistics": self.statistics.to_dict(),
            "native_engine": self.native_engine,
            "ocr_engine": self.ocr_engine,
            "languages": list(self.languages),
            "config_hash": self.config_hash,
            "notes": list(self.notes),
            "failure_reason": self.failure_reason,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextExtractionContext:
        blocks = [TextBlock.from_dict(b) for b in data.get("blocks", [])]
        pages = [PageTextExtraction.from_dict(p) for p in data.get("pages", [])]
        statistics = TextExtractionStatistics.from_dict(data.get("statistics", {}))
        known = {f for f in cls.__dataclass_fields__ if f not in {"blocks", "pages", "statistics", "languages", "notes"}}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(
            blocks=blocks, pages=pages, statistics=statistics,
            languages=list(data.get("languages", [])),
            notes=list(data.get("notes", [])),
            **kwargs,
        )

    @classmethod
    def from_json(cls, text: str) -> TextExtractionContext:
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# Section 8 — Visual-extraction models
# ---------------------------------------------------------------------------

VISUAL_CATEGORIES: list[str] = [
    "figure",
    "photograph",
    "diagram",
    "engineering_drawing",
    "cad_drawing",
    "circuit_diagram",
    "flowchart",
    "chemical_structure",
    "graph",
    "chart",
    "map",
    "screenshot",
    "composite_figure",
    "unknown",
]

VISUAL_PROVIDERS: list[str] = ["docling", "opencv", "chemical", "generic", "default"]


@dataclass
class ImageMetadata:
    """Quality/format metadata for one materialized visual asset."""

    width: int = 0
    height: int = 0
    dpi: float = 0.0
    image_format: str = "png"
    color_mode: str = "unknown"
    has_transparency: bool = False
    aspect_ratio: float = 0.0
    original_resolution: list[int] = field(default_factory=list)
    rotation: float = 0.0
    orientation: str = "upright"
    source_type: str = "raster"
    compression: str | None = None
    recompressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "image_format": self.image_format,
            "color_mode": self.color_mode,
            "has_transparency": self.has_transparency,
            "aspect_ratio": self.aspect_ratio,
            "original_resolution": list(self.original_resolution),
            "rotation": self.rotation,
            "orientation": self.orientation,
            "source_type": self.source_type,
            "compression": self.compression,
            "recompressed": self.recompressed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageMetadata:
        known = {f for f in cls.__dataclass_fields__ if f != "original_resolution"}
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        return cls(original_resolution=list(data.get("original_resolution", [])), **kwargs)


# ---------------------------------------------------------------------------
# Section 9 — Table models
# ---------------------------------------------------------------------------

@dataclass
class PageMeta:
    """Classification metadata collected for a PDF page."""

    page_no: int
    page_type: str
    word_count: int
    has_real_fonts: bool
    image_coverage: float
    render_similarity: float | None
    orientation: int
    classification_confidence: float
    signals_used: list[str] = field(default_factory=list)


@dataclass
class RawCell:
    """Cell extracted from a table region."""

    row_idx: int
    col_idx: int
    text: str
    bbox: dict[str, float] | None
    latex: str | None
    mathml: str | None
    is_header: bool
    rowspan: int = 1
    colspan: int = 1


@dataclass
class RawTable:
    """Intermediate table representation before rendering."""

    table_id: str
    book_id: str
    page_no: int
    bbox: dict[str, float]
    cells: list[RawCell] = field(default_factory=list)
    caption: str = ""
    footnotes: list[str] = field(default_factory=list)
    source_extractor: str = ""
    parsing_accuracy: float = 0.0
    confidence: float = 0.0
    column_units: dict[int, str] = field(default_factory=dict)
    source_pages: list[int] = field(default_factory=list)


@dataclass
class TableRecord:
    """Final table output record used by renderers and exports."""

    table_id: str
    book_id: str
    page_no: int
    page_type: str
    source_extractor: str
    caption: str
    footnotes: list[str]
    columns: list[str]
    rows: list[list[str]]
    column_units: dict[int, str]
    bbox: dict[str, float]
    quality_score: float
    structure_score: float
    text_score: float
    completeness_score: float
    route_to_llm: bool
    llm_tier: str
    llm_tier_used: str | None
    llm_backend_used: str | None
    llm_confidence: float | None
    llm_corrected: bool
    section_context: str
    pipeline_run_id: str
    extraction_version: dict[str, str]
    created_at: datetime
    source_pages: list[int] = field(default_factory=list)
    extraction_confidence: float | None = None


# ---------------------------------------------------------------------------
# Section 10 — Run / operational models
# ---------------------------------------------------------------------------

@dataclass
class StageFailure:
    """Failure record for a pipeline stage."""

    pipeline_run_id: str
    book_id: str
    table_id: str | None
    page_no: int | None
    stage: str
    error_type: str
    error_msg: str
    retry_count: int
    is_gold_candidate: bool
    timestamp: datetime
