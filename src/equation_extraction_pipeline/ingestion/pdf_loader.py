"""PDF ingestion loader — merged module.

Combines:
- PDF modality classification (scanned / digital / hybrid)
- Low-level PDF reading backend (pypdfium2 + pdfminer.six)
- Document identity derivation and content fingerprinting
- PDF metadata extraction and normalisation
- High-level ingestion entry points (load_pdf, ingest_document, ingest_batch)

PyMuPDF/fitz is intentionally excluded (AGPL licence incompatibility).
All PDF access goes through pypdfium2 (Apache-2.0 / BSD-3) and
pdfminer.six (MIT).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import structlog
from pdfminer.high_level import extract_pages, extract_text_to_fp
from pdfminer.layout import (
    LAParams,
    LTAnno,
    LTChar,
    LTFigure,
    LTImage,
    LTTextContainer,
    LTTextLine,
)

from equation_extraction_pipeline.config import settings as config
from equation_extraction_pipeline.detection.duplicate_resolver import DuplicateIndex
from equation_extraction_pipeline.domain.models import (
    MANIFEST_VERSION,
    ClassificationResult,
    DocumentIdentity,
    DocumentMetadata,
    IngestionManifest,
    Provenance,
)
from equation_extraction_pipeline.ingestion.file_validator import validate_pdf

__all__ = [
    # classification
    "classify_pdf",
    # pdf backend
    "RENDER_ZOOM",
    "PdfDocument",
    "PageView",
    "open_document",
    "render_page_image",
    "load_pdf",
    # identity / fingerprint
    "FINGERPRINT_ALGORITHM",
    "compute_fingerprint",
    "derive_identity",
    "canonical_metadata_string",
    "has_usable_metadata",
    # metadata
    "extract_metadata",
    "normalize_metadata",
    "normalize_text",
    "normalize_multi",
    # ingest
    "IngestionResult",
    "ingest_document",
    "ingest_batch",
]

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)


# ===========================================================================
# Section 1 — PDF modality classification
# (from classification.py)
# ===========================================================================

# Thresholds for per-page classification
_DIGITAL_CHAR_MIN = 20
"""Pages with at least this many characters are considered to have a text layer."""

_SCANNED_CHAR_MAX = 5
"""Pages with fewer than this many characters are considered image-only (scanned)."""

_SAMPLE_MAX_PAGES = 10
"""Maximum number of pages to sample for classification."""


def _extract_page_text(pdf_path: Path, page_index: int) -> str:
    """Extract plain text from a single 0-based page using pdfminer.six."""
    buf = StringIO()
    try:
        with open(pdf_path, "rb") as fh:
            extract_text_to_fp(
                fh,
                buf,
                page_numbers=[page_index],
                laparams=LAParams(),
                output_type="text",
                codec="utf-8",
            )
    except Exception as exc:
        logger.debug("text_extraction_failed page=%d error=%s", page_index, exc)
    return buf.getvalue()


def _classify_page(char_count: int) -> str:
    """Return per-page modality based on character count."""
    if char_count >= _DIGITAL_CHAR_MIN:
        return "digital"
    if char_count <= _SCANNED_CHAR_MAX:
        return "scanned"
    return "hybrid"


def _select_sample_pages(total: int) -> list[int]:
    """Return 0-based page indices to sample, spread evenly across the document."""
    if total <= _SAMPLE_MAX_PAGES:
        return list(range(total))
    step = total / _SAMPLE_MAX_PAGES
    return [int(i * step) for i in range(_SAMPLE_MAX_PAGES)]


def classify_pdf(pdf_path: Path) -> ClassificationResult:
    """Classify PDF modality by sampling pages.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    ClassificationResult
        Modality verdict (``'scanned'``, ``'digital'``, or ``'hybrid'``)
        with confidence score and page count.

    Notes
    -----
    A page with an embedded text layer inside a predominantly scanned document
    is kept as ``'scanned'`` (not promoted to ``'hybrid'``) to avoid the
    false-positive hybrid classification observed in practice.
    """
    pdf_path = Path(pdf_path)
    logger.info("classifying pdf=%s", pdf_path.name)

    # PdfDocument is not a context manager in pypdfium2 4.30+.  Explicitly
    # close it so classification works across both older and current releases.
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        total_pages = len(doc)
    finally:
        doc.close()

    sample_indices = _select_sample_pages(total_pages)
    counts: dict[str, int] = {"scanned": 0, "digital": 0, "hybrid": 0}

    for idx in sample_indices:
        text = _extract_page_text(pdf_path, idx)
        char_count = len(text.strip())
        page_modality = _classify_page(char_count)
        counts[page_modality] += 1
        logger.debug("page=%d chars=%d modality=%s", idx + 1, char_count, page_modality)

    sampled = len(sample_indices)
    digital_ratio = counts["digital"] / sampled
    scanned_ratio = counts["scanned"] / sampled

    # Determine document-level modality.
    if scanned_ratio >= 0.70:
        modality = "scanned"
        confidence = scanned_ratio
    elif digital_ratio >= 0.70:
        modality = "digital"
        confidence = digital_ratio
    else:
        modality = "hybrid"
        confidence = 1.0 - abs(scanned_ratio - digital_ratio)

    result = ClassificationResult(
        modality=modality,
        confidence=round(confidence, 4),
        page_count=total_pages,
        sampled_pages=sampled,
    )
    logger.info(
        "classification_done modality=%s confidence=%.2f pages=%d sampled=%d",
        modality,
        confidence,
        total_pages,
        sampled,
    )
    return result


# ===========================================================================
# Section 2 — Low-level PDF reading backend
# (from pipeline/pdf_backend.py)
# ===========================================================================

# Matches the previous fitz.Matrix(2, 2): render at 2x so figure/table bboxes
# (in PDF points) map to pixels as points * RENDER_ZOOM.
RENDER_ZOOM = 2

# PyMuPDF span "flags" bit 16 == bold; is_heading() checks ``flags & 16``.
_BOLD_FLAG = 16
# Substrings that mark a font as bold/heavy in its (often subset-prefixed) name,
# e.g. "QVFPRX+NimbusSanL-Bold" or "ABCDEF+Arial-BoldMT".
_BOLD_MARKERS = ("bold", "black", "heavy", "semibold", "demibold", "-bd", "medi")


def _is_bold(fontname: str) -> bool:
    name = (fontname or "").lower()
    return any(marker in name for marker in _BOLD_MARKERS)


class _Rect:
    """Minimal stand-in for ``fitz.Rect`` exposing ``width``/``height``."""

    __slots__ = ("width", "height")

    def __init__(self, width: float, height: float) -> None:
        self.width = float(width)
        self.height = float(height)


class PageView:
    """Adapter over a single page exposing the used subset of ``fitz.Page``.

    Supports ``get_text("dict"|"rawdict"|"words"|"blocks"|"text")``,
    ``get_image_info()``, ``rect``, ``rotation``, ``number``, and
    ``render()``.
    """

    def __init__(self, document: "PdfDocument", index: int) -> None:
        self._document = document
        self.number = index

    @property
    def parent(self) -> "PdfDocument":
        # Mirrors fitz.Page.parent so callers can recover the source path via
        # ``page.parent.name`` (used to derive the book id).
        return self._document

    # -- geometry ---------------------------------------------------------

    @property
    def _ltpage(self):
        return self._document._ltpage(self.number)

    @property
    def rect(self) -> _Rect:
        ltpage = self._ltpage
        if ltpage is None:
            return _Rect(0.0, 0.0)
        return _Rect(ltpage.width, ltpage.height)

    @property
    def rotation(self) -> int:
        ltpage = self._ltpage
        return int(getattr(ltpage, "rotate", 0) or 0) if ltpage is not None else 0

    def _page_height(self) -> float:
        ltpage = self._ltpage
        return float(ltpage.height) if ltpage is not None else 0.0

    def _flip_bbox(self, obj) -> list[float]:
        """Convert a pdfminer (bottom-left) bbox to PyMuPDF (top-left) coords."""
        height = self._page_height()
        return [
            float(obj.x0),
            float(height - obj.y1),
            float(obj.x1),
            float(height - obj.y0),
        ]

    # -- text -------------------------------------------------------------

    def _iter_spans(self, line: LTTextLine):
        """Group consecutive chars sharing (font, rounded size) into spans,
        mirroring how PyMuPDF coalesces runs of identical styling."""
        height = self._page_height()
        current: dict | None = None
        for char in line:
            # pdfminer inserts LTAnno for spaces/newlines it infers from layout
            # gaps; these carry no font/size. Keep their text so words stay
            # separated (otherwise "General Balance" becomes "GeneralBalance").
            if isinstance(char, LTAnno):
                if current is not None:
                    current["text"].append(char.get_text())
                continue
            if not isinstance(char, LTChar):
                continue
            fontname = str(getattr(char, "fontname", "") or "")
            size = round(float(getattr(char, "size", 0.0) or 0.0), 1)
            key = (fontname, size)
            if current is None or current["_key"] != key:
                if current is not None:
                    yield _finalize_span(current)
                current = {
                    "_key": key,
                    "text": [],
                    "size": size,
                    "font": fontname,
                    "flags": _BOLD_FLAG if _is_bold(fontname) else 0,
                    "x0": float(char.x0),
                    "y0": float(height - char.y1),
                    "x1": float(char.x1),
                    "y1": float(height - char.y0),
                }
            current["text"].append(char.get_text())
            current["x0"] = min(current["x0"], float(char.x0))
            current["x1"] = max(current["x1"], float(char.x1))
            current["y0"] = min(current["y0"], float(height - char.y1))
            current["y1"] = max(current["y1"], float(height - char.y0))
        if current is not None:
            yield _finalize_span(current)

    def _text_dict(self) -> dict:
        ltpage = self._ltpage
        blocks: list[dict] = []
        if ltpage is None:
            return {"blocks": blocks}
        for obj in ltpage:
            if not isinstance(obj, LTTextContainer):
                continue
            lines: list[dict] = []
            for line in obj:
                if not isinstance(line, LTTextLine):
                    continue
                spans = list(self._iter_spans(line))
                if spans:
                    lines.append({"spans": spans})
            blocks.append({"type": 0, "bbox": self._flip_bbox(obj), "lines": lines})
        return {"blocks": blocks}

    def _words(self) -> list:
        ltpage = self._ltpage
        if ltpage is None:
            return []
        words: list[tuple] = []
        for obj in ltpage:
            if not isinstance(obj, LTTextContainer):
                continue
            for token in obj.get_text().split():
                words.append((0.0, 0.0, 0.0, 0.0, token))
        return words

    def _plain_text(self) -> str:
        ltpage = self._ltpage
        if ltpage is None:
            return ""
        parts = [obj.get_text() for obj in ltpage if isinstance(obj, LTTextContainer)]
        return "".join(parts)

    def _blocks(self) -> list[tuple]:
        """Return text blocks as ``(x0, y0, x1, y1, text, block_no, block_type)`` tuples.

        Matches the ``fitz.Page.get_text("blocks")`` tuple format so that scripts
        originally written for PyMuPDF work without modification after swapping in
        this backend.  ``block_type`` is always ``0`` (text) because only
        ``LTTextContainer`` objects are tracked; image blocks are omitted.
        """
        ltpage = self._ltpage
        if ltpage is None:
            return []
        height = self._page_height()
        result: list[tuple] = []
        block_no = 0
        for obj in ltpage:
            if not isinstance(obj, LTTextContainer):
                continue
            x0 = float(obj.x0)
            y0 = float(height - obj.y1)
            x1 = float(obj.x1)
            y1 = float(height - obj.y0)
            text = obj.get_text()
            result.append((x0, y0, x1, y1, text, block_no, 0))
            block_no += 1
        return result

    def get_text(self, kind: str = "text"):
        if kind in ("dict", "rawdict"):
            return self._text_dict()
        if kind == "words":
            return self._words()
        if kind == "blocks":
            return self._blocks()
        return self._plain_text()

    # -- images -----------------------------------------------------------

    def get_image_info(self) -> list[dict]:
        ltpage = self._ltpage
        if ltpage is None:
            return []
        infos: list[dict] = []

        def _walk(container):
            for obj in container:
                if isinstance(obj, LTImage):
                    infos.append({"bbox": self._flip_bbox(obj)})
                elif isinstance(obj, LTFigure):
                    # Figures wrap the raster images; recurse, but if a figure
                    # has no LTImage child fall back to the figure bbox so
                    # full-page scanned art is still reported.
                    before = len(infos)
                    _walk(obj)
                    if len(infos) == before:
                        infos.append({"bbox": self._flip_bbox(obj)})

        _walk(ltpage)
        return infos

    # -- rendering --------------------------------------------------------

    def render(self, zoom: float = RENDER_ZOOM) -> np.ndarray:
        return self._document._render(self.number, zoom)


def _finalize_span(span: dict) -> dict:
    return {
        "text": "".join(span["text"]),
        "size": span["size"],
        "font": span["font"],
        "flags": span["flags"],
        "bbox": [span["x0"], span["y0"], span["x1"], span["y1"]],
    }


class PdfDocument:
    """Drop-in replacement for the ``fitz.Document`` usage in the pipeline.

    Supports ``len()``, indexing, iteration, and use as a context manager.
    pdfminer layout is parsed lazily per page (and cached) so iterating the
    whole document parses each page exactly once — no O(n²) reparsing.

    Coordinate note: pdfminer uses a bottom-left origin (y grows upward) while
    PyMuPDF uses a top-left origin (y grows downward). All bboxes returned by
    :class:`PageView` are converted to the PyMuPDF top-left convention so
    downstream geometry is unchanged.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        # Mirrors fitz.Document.name (the source file path).
        self.name = self._path
        self._pdfium = pdfium.PdfDocument(self._path)
        self._count = len(self._pdfium)
        self._ltpage_cache: dict[int, object] = {}

    def _ltpage(self, index: int):
        if index not in self._ltpage_cache:
            pages = list(extract_pages(self._path, page_numbers=[index], laparams=LAParams()))
            self._ltpage_cache[index] = pages[0] if pages else None
        return self._ltpage_cache[index]

    def get_metadata(self) -> dict[str, str]:
        """Return the embedded PDF info dictionary (Title/Author/...), best-effort.

        Tolerant of missing or undecodable fields: returns ``{}`` on any backend
        error and coerces values to ``str`` so the ingestion stage never raises on
        malformed metadata.
        """
        try:
            raw = self._pdfium.get_metadata_dict()
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        return {str(key): str(value) for key, value in raw.items() if value not in (None, "")}

    def get_document_info(self) -> dict[str, object]:
        """Best-effort technical document info for the metadata stage.

        Returns the embedded info dict plus the PDF version string (e.g. ``"1.7"``)
        and an encryption flag.  Any backend error degrades the relevant field rather
        than raising, so the metadata stage records it as missing instead of aborting.
        """
        info = self.get_metadata()
        version = ""
        try:
            raw_v = self._pdfium.get_version()  # int like 17 → "1.7"; may be None
            if isinstance(raw_v, int) and raw_v > 0:
                version = f"{raw_v // 10}.{raw_v % 10}"
        except Exception:
            version = ""
        return {"info": info, "pdf_version": version, "encrypted": False}

    def _render(self, index: int, zoom: float) -> np.ndarray:
        page = self._pdfium[index]
        bitmap = page.render(scale=zoom)
        array = bitmap.to_numpy()
        # pypdfium2 returns BGR(A); fitz returned RGB. Drop alpha, flip to RGB.
        if array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]
        if array.ndim == 3 and array.shape[2] == 3:
            array = array[:, :, ::-1]
        return np.ascontiguousarray(array)

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> PageView:
        if index < 0:
            index += self._count
        return PageView(self, index)

    def __iter__(self):
        for index in range(self._count):
            yield PageView(self, index)

    def __enter__(self) -> "PdfDocument":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._pdfium.close()
        except Exception:
            pass


def open_document(path: str | Path) -> PdfDocument:
    """Open a PDF and return a :class:`PdfDocument` (replaces ``fitz.open``)."""
    return PdfDocument(path)


def load_pdf(path: str | Path) -> PdfDocument:
    """Open a PDF file and return a :class:`PdfDocument`.

    Thin alias for :func:`open_document` exposed as the primary high-level
    entry point for consumers that just need to read pages without running the
    full ingestion pipeline.
    """
    return PdfDocument(path)


def render_page_image(page: PageView, zoom: float = RENDER_ZOOM) -> np.ndarray:
    """Render a page to an RGB ``uint8`` numpy array (replaces fitz pixmap)."""
    return page.render(zoom)


# ===========================================================================
# Section 3 — Document identity derivation and content fingerprinting
# (from ingestion/identity.py + ingestion/fingerprint.py)
# ===========================================================================

FINGERPRINT_ALGORITHM = "sha256"

# 1 MiB chunks: bounds memory so extremely large PDFs do not exhaust RAM.
_CHUNK_SIZE = 1024 * 1024

# Unit separator between canonical fields; never appears in normalised text.
_FIELD_SEP = "\x1f"


def compute_fingerprint(pdf_path: Path, *, chunk_size: int = _CHUNK_SIZE) -> str:
    """Return the full SHA-256 hex digest of the file's raw bytes, read in chunks."""
    hasher = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def has_usable_metadata(metadata: DocumentMetadata) -> bool:
    """True when any identity-bearing metadata field is populated."""
    return bool(
        metadata.title
        or metadata.authors
        or metadata.publishers
        or metadata.edition
        or metadata.identifiers
    )


def canonical_metadata_string(metadata: DocumentMetadata) -> str:
    """Order-independent, case-folded canonical string used to derive the logical id.

    Multi-valued fields are sorted here too (not just at extraction) so identity is
    order-independent regardless of how the :class:`DocumentMetadata` was constructed.
    """
    authors = "|".join(sorted(metadata.authors))
    publishers = "|".join(sorted(metadata.publishers))
    identifiers = "|".join(
        f"{key}={metadata.identifiers[key]}" for key in sorted(metadata.identifiers)
    )
    fields = [metadata.title, authors, publishers, metadata.edition, identifiers]
    return _FIELD_SEP.join(field.casefold() for field in fields)


def derive_identity(fingerprint: str, metadata: DocumentMetadata) -> DocumentIdentity:
    """Build a :class:`DocumentIdentity`.

    The logical id is the full SHA-256 hex digest (no truncation) of the canonical
    metadata string when metadata is usable; otherwise it falls back to a content
    digest derived from the binary fingerprint so every valid PDF gets a deterministic
    id. Different editions produce different canonical strings and therefore different ids.
    """
    if has_usable_metadata(metadata):
        canonical = canonical_metadata_string(metadata)
        logical_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        source = "metadata"
    else:
        logical_id = hashlib.sha256(
            f"fingerprint:{fingerprint}".encode("utf-8")
        ).hexdigest()
        source = "content_fallback"
    return DocumentIdentity(
        fingerprint=fingerprint,
        logical_id=logical_id,
        logical_id_source=source,
    )


# ===========================================================================
# Section 4 — PDF metadata extraction and normalisation
# (from ingestion/metadata.py)
# ===========================================================================

# Common separators between multiple authors/publishers in a single info field.
_MULTI_SEP = re.compile(r"\s*(?:;|,|/|&|\band\b|\+)\s*", re.IGNORECASE)

# Embedded-id field names worth capturing if a PDF exposes them as custom keys.
_ID_KEYS = ("isbn", "issn", "doi", "eisbn")


def normalize_text(value: str) -> str:
    """NFC-normalise, collapse internal whitespace, and strip. Never raises."""
    try:
        norm = unicodedata.normalize("NFC", value)
    except (TypeError, ValueError):
        return ""
    return " ".join(norm.split()).strip()


def normalize_multi(value: str) -> list[str]:
    """Split a multi-valued field on common separators, normalise, and sort.

    Sorting makes the derived identity order-independent (multiple-authors edge case).
    """
    parts = (normalize_text(part) for part in _MULTI_SEP.split(value or ""))
    return sorted({part for part in parts if part})


def normalize_metadata(raw: dict[str, str]) -> DocumentMetadata:
    """Map and normalise a raw PDF info dictionary into a :class:`DocumentMetadata`."""
    lower = {str(key).lower(): str(value) for key, value in raw.items()}

    identifiers: dict[str, str] = {}
    for key in _ID_KEYS:
        if lower.get(key):
            normalized = normalize_text(lower[key])
            if normalized:
                identifiers[key] = normalized

    return DocumentMetadata(
        title=normalize_text(lower.get("title", "")),
        authors=normalize_multi(lower.get("author", "")),
        publishers=normalize_multi(lower.get("publisher", "")),
        edition=normalize_text(lower.get("edition", "")),
        identifiers=identifiers,
        raw={str(key): str(value) for key, value in raw.items()},
    )


def extract_metadata(pdf_path: Path) -> DocumentMetadata:
    """Open the PDF, read its embedded info dictionary, and normalise it.

    Best-effort: returns an empty-but-valid :class:`DocumentMetadata` if the
    document cannot be opened or exposes no metadata.
    """
    try:
        with open_document(str(pdf_path)) as document:
            raw = document.get_metadata()
    except Exception:
        return DocumentMetadata()
    return normalize_metadata(raw)


# ===========================================================================
# Section 5 — High-level ingestion entry points
# (from ingestion/ingest.py)
# ===========================================================================

_MANIFEST_FILENAME = "ingestion_manifest.json"
_FAILURES_DIRNAME = "_ingestion_failures"
_INDEX_FILENAME = "_ingestion_index.json"


@dataclass
class IngestionResult:
    """What the orchestrator needs to decide whether to run downstream stages."""

    manifest: IngestionManifest
    book_id: str | None
    should_process: bool


def _index_path(output_dir: Path) -> Path:
    if config.KNOVEL_INGESTION_INDEX_PATH:
        return Path(config.KNOVEL_INGESTION_INDEX_PATH)
    return output_dir / _INDEX_FILENAME


def _manifest_dir(output_dir: Path, book_id: str) -> Path:
    if config.KNOVEL_INGESTION_MANIFEST_DIR:
        return Path(config.KNOVEL_INGESTION_MANIFEST_DIR)
    return output_dir / book_id


def _write_manifest(manifest: IngestionManifest, manifest_dir: Path) -> Path:
    """Write the manifest atomically (temp-then-rename) and return its path."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / _MANIFEST_FILENAME
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp_path.write_text(manifest.to_json(), encoding="utf-8")
    os.replace(tmp_path, manifest_path)
    return manifest_path


def ingest_document(
    pdf_path: Path,
    *,
    output_dir: Path,
    pipeline_run_id: str,
) -> IngestionResult:
    """Validate, fingerprint, identify, deduplicate, and manifest a single PDF."""
    pdf_path = pdf_path.expanduser().resolve()
    ingested_at = datetime.utcnow()
    provenance = Provenance(source_extractors=["ingestion"], source_pages=[])

    validation = validate_pdf(pdf_path, max_file_mb=config.KNOVEL_INGESTION_MAX_FILE_MB)
    if not validation.ok:
        manifest = IngestionManifest(
            manifest_version=MANIFEST_VERSION,
            provenance=provenance,
            source_path=str(pdf_path),
            source_filename=pdf_path.name,
            ingested_at=ingested_at,
            pipeline_run_id=pipeline_run_id,
            outcome="failed",
            failure_reason=validation.failure_reason,
        )
        _write_manifest(manifest, output_dir / _FAILURES_DIRNAME / pdf_path.stem)
        _slog.warning(
            "ingestion_failed",
            source_filename=pdf_path.name,
            failure_reason=validation.failure_reason,
        )
        return IngestionResult(manifest=manifest, book_id=None, should_process=False)

    fingerprint = compute_fingerprint(pdf_path)
    metadata: DocumentMetadata = extract_metadata(pdf_path)
    identity: DocumentIdentity = derive_identity(fingerprint, metadata)
    book_id = identity.logical_id

    index = DuplicateIndex(_index_path(output_dir))
    duplicate = index.find_duplicate(identity)
    manifest_dir = _manifest_dir(output_dir, book_id)

    if duplicate is not None:
        manifest = IngestionManifest(
            manifest_version=MANIFEST_VERSION,
            provenance=provenance,
            source_path=str(pdf_path),
            source_filename=pdf_path.name,
            ingested_at=ingested_at,
            pipeline_run_id=pipeline_run_id,
            outcome="duplicate",
            identity=identity,
            metadata=metadata,
            duplicate_of=duplicate,
            page_count=validation.page_count,
        )
        _write_manifest(manifest, manifest_dir)
        should_process = config.KNOVEL_INGESTION_DUPLICATE_POLICY != "skip"
        _slog.info(
            "ingestion_duplicate",
            book_id=book_id,
            source_filename=pdf_path.name,
            duplicate_type=duplicate.duplicate_type,
            detected_in=duplicate.detected_in,
            policy=config.KNOVEL_INGESTION_DUPLICATE_POLICY,
            should_process=should_process,
        )
        return IngestionResult(manifest=manifest, book_id=book_id, should_process=should_process)

    manifest = IngestionManifest(
        manifest_version=MANIFEST_VERSION,
        provenance=provenance,
        source_path=str(pdf_path),
        source_filename=pdf_path.name,
        ingested_at=ingested_at,
        pipeline_run_id=pipeline_run_id,
        outcome="ingested",
        identity=identity,
        metadata=metadata,
        page_count=validation.page_count,
    )
    manifest_path = _write_manifest(manifest, manifest_dir)
    index.register(identity, str(manifest_path), pdf_path.name)
    _slog.info(
        "ingestion_complete",
        book_id=book_id,
        source_filename=pdf_path.name,
        logical_id_source=identity.logical_id_source,
        page_count=validation.page_count,
    )
    return IngestionResult(manifest=manifest, book_id=book_id, should_process=True)


def ingest_batch(
    pdf_paths: list[Path],
    *,
    output_dir: Path,
    pipeline_run_id: str,
) -> list[IngestionResult]:
    """Ingest each PDF with per-document containment.

    One failure never aborts the batch; correctness of dedup derives from the
    persisted index, not from this wrapper.
    """
    results: list[IngestionResult] = []
    for pdf_path in pdf_paths:
        try:
            results.append(
                ingest_document(
                    pdf_path,
                    output_dir=output_dir,
                    pipeline_run_id=pipeline_run_id,
                )
            )
        except Exception as exc:  # defensive: ingestion must never abort the batch
            _slog.error(
                "ingestion_unexpected_error",
                source_filename=pdf_path.name,
                error=str(exc),
            )
    return results
