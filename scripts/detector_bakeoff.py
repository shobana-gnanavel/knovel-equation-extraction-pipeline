#!/usr/bin/env python3
"""Standalone detector bake-off — Docling detection + label-filter + crop.

DECISION GATE (Phase 1 of the equation-detection plan). This script does NOT modify the
``src/`` production pipeline; it only reuses two read-only leaf utilities from it (the
label-scan regex helpers and the LaTeX metric). It answers: can Docling's detected boxes,
narrowed to labeled equations, replace the current geometry-reconstruction crop path?

Three stages (this is the ordering we settled on — crop only the survivors):
  1. DETECT   — Docling finds every formula region (bbox only; formula enrichment OFF so
                the slow CPU LaTeX decode is skipped). Detection alone is ~40s/chapter.
  2. FILTER   — scan the PDF text layer for printed reference labels ("Eq. 12.2.1") using
                the pipeline's proven ``_LABEL_RE`` path. If the book HAS labels, keep only
                the Docling region that associates with each label (nearest region on the
                label's row, to its left). If the book has NO labels, keep every region.
  3. CROP     — crop only the kept regions from Docling's own page raster with a single
                fixed fractional pad (no tightening / vertical-expansion heuristics).

Because kept regions are keyed by label, detection recall is scored directly against the
gold labels (not a page proxy). Recognition is a separate, later step run only on the
kept crops.

Run INSIDE the container (Docling + models live there)::

    docker exec equation_extraction_pipeline-pipeline-1 \
        python /app/scripts/detector_bakeoff.py \
            --pdf /app/data/input/28120_12.pdf \
            --gold /app/data/ground_truth/28120_12.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Reuse only read-only leaf utilities from the package (not the pipeline orchestration).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def scan_label_positions(pdf_path: Path) -> dict[int, tuple[float, list[tuple[str, tuple[float, float, float, float]]]]]:
    """Return ``{page_no(1-based): (page_height_pts, [(label, bbox_top_left_pts)])}``.

    Reuses the pipeline's proven label regex + cross-reference rejection + label-line anchor,
    so the label universe here matches the detector's. Boxes are returned in TOP-LEFT points
    to match Docling's ``to_top_left_origin`` convention downstream.
    """
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTTextBox
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage
    from equation_extraction_pipeline.detection.equation_label_detector import (
        _LABEL_RE,
        _is_cross_reference_for_match,
        _label_anchor_bbox,
        _normalise_label,
    )

    rsrc = PDFResourceManager()
    device = PDFPageAggregator(
        rsrc, laparams=LAParams(line_overlap=0.5, char_margin=2.0, line_margin=0.5, word_margin=0.1)
    )
    interp = PDFPageInterpreter(rsrc, device)

    out: dict[int, tuple[float, list[tuple[str, tuple[float, float, float, float]]]]] = {}
    seen: set[str] = set()  # DOCUMENT-wide dedup, first-occurrence-wins (mirrors the pipeline's
    #                          _find_labeled_equations seen_labels): the equation definition
    #                          precedes later cross-references / section headings of the same number.
    with open(pdf_path, "rb") as fh:
        for idx, page in enumerate(PDFPage.get_pages(fh)):
            try:
                interp.process_page(page)
                layout = device.get_result()
            except Exception:
                out[idx + 1] = (0.0, [])
                continue
            ph = float(layout.height)
            labels: list[tuple[str, tuple[float, float, float, float]]] = []
            for box in layout:
                if not isinstance(box, LTTextBox):
                    continue
                text = box.get_text()
                for m in _LABEL_RE.finditer(text):
                    if _is_cross_reference_for_match(text, m):
                        continue
                    ls = _normalise_label(m.group(1))
                    if ls in seen:
                        continue
                    seen.add(ls)
                    x0, y0, x1, y1 = _label_anchor_bbox(box, ls)  # bottom-left origin pts
                    labels.append((ls, (x0, ph - y1, x1, ph - y0)))  # → top-left
            out[idx + 1] = (ph, labels)
    return out


def detect_regions(pdf_path: Path, images_scale: float) -> tuple[Any, list[dict[str, Any]]]:
    """Docling detection ONLY (no formula enrichment). Return (doc, regions).

    Each region: ``{page, bbox_tl_pts:(l,t,r,b), page_h_pts}``. Cropping happens later, on
    the filtered survivors only, from ``doc.pages[page].image``.
    """
    import os

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import DocItemLabel

    artifacts = os.getenv("DOCLING_ARTIFACTS_PATH", "").strip() or None
    options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        do_formula_enrichment=False,   # detection-only — skip the slow CPU LaTeX decode
        generate_page_images=True,     # keep page rasters so survivors can be cropped
        images_scale=images_scale,
        artifacts_path=artifacts,
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    doc = converter.convert(str(pdf_path)).document

    regions: list[dict[str, Any]] = []
    for item, _lvl in doc.iterate_items():
        if getattr(item, "label", None) != DocItemLabel.FORMULA:
            continue
        prov = getattr(item, "prov", None) or []
        if not prov:
            continue
        page_no = prov[0].page_no
        page = doc.pages.get(page_no)
        if page is None:
            continue
        ph = float(page.size.height)
        tl = prov[0].bbox.to_top_left_origin(ph)
        regions.append({
            "page": page_no,
            "bbox_tl": (float(tl.l), float(tl.t), float(tl.r), float(tl.b)),
            "page_h": ph,
        })
    return doc, regions


def _fallback_bbox(label_tl: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Label-reconstruction crop box for a label Docling missed (top-left points).

    Mirrors the pipeline's ``_image_formula_bbox_for_label`` heuristic: a conservative band to
    the LEFT of the right-margin label, tall enough for stacked fractions. Used only as the rare
    fallback so recall reaches ~100% even when the model detector does not fire.
    """
    lx0, lt, _lx1, lb = label_tl
    h = max(lb - lt, 8.0)
    return (max(0.0, lx0 * 0.24), max(0.0, lt - 2.0 * h), max(0.0, lx0 - 12.0), lb + 1.0 * h)


def associate_labels(
    regions: list[dict[str, Any]],
    labels_by_page: dict[int, tuple[float, list[tuple[str, tuple[float, float, float, float]]]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Assign a crop box to every printed label; return (kept, docling_missed).

    Improvements over the naive first pass:
      * A region is NOT hard-consumed — a tall Docling region spanning several stacked
        equations is SHARED by all labels whose row falls in it, then SPLIT into per-label
        vertical bands (boundaries at midpoints between adjacent label rows).
      * Vertical tolerance loosened (±2.5 label-heights) so page-top labels still associate.
      * A label with no candidate region gets a ``_fallback_bbox`` (source="label_fallback")
        so it is still cropped; these are counted separately as the model's true misses.
    """
    by_page: dict[int, list[dict[str, Any]]] = {}
    for r in regions:
        by_page.setdefault(r["page"], []).append(r)

    kept: list[dict[str, Any]] = []
    docling_missed: list[str] = []

    for page_no, (_ph, labels) in labels_by_page.items():
        page_regions = by_page.get(page_no, [])
        # 1. Assign each label to its nearest qualifying region (sharing allowed).
        assign: dict[int, list[tuple[str, tuple, float]]] = {}
        for label, ltl in labels:
            lx0, lt, _lx1, lb = ltl
            cy = (lt + lb) / 2.0
            h = max(lb - lt, 8.0)
            best_i, best_gap = None, None
            for i, r in enumerate(page_regions):
                rl, rt, rr, rb = r["bbox_tl"]
                if rl >= lx0 + h:  # region must start left of the margin label
                    continue
                if not (rt - 2.5 * h <= cy <= rb + 2.5 * h):
                    continue
                gap = abs(lx0 - rr)
                if best_gap is None or gap < best_gap:
                    best_i, best_gap = i, gap
            if best_i is None:
                kept.append({"page": page_no, "page_h": _ph, "label": label,
                             "bbox_tl": _fallback_bbox(ltl), "source": "label_fallback"})
                docling_missed.append(label)
            else:
                assign.setdefault(best_i, []).append((label, ltl, cy))

        # 2. For each region, split vertically by label rows so each label gets its own band.
        for i, members in assign.items():
            rl, rt, rr, rb = page_regions[i]["bbox_tl"]
            members.sort(key=lambda m: m[2])  # by label cy, top→bottom
            cys = [m[2] for m in members]
            for j, (label, _ltl, _cy) in enumerate(members):
                if len(members) == 1:
                    band_t, band_b = rt, rb
                    src = "docling"
                else:
                    band_t = rt if j == 0 else (cys[j - 1] + cys[j]) / 2.0
                    band_b = rb if j == len(members) - 1 else (cys[j] + cys[j + 1]) / 2.0
                    src = "docling_split"
                kept.append({"page": page_no, "page_h": page_regions[i]["page_h"], "label": label,
                             "bbox_tl": (rl, band_t, rr, band_b), "source": src})
    return kept, docling_missed


def crop_regions(doc: Any, regions: list[dict[str, Any]], crops_dir: Path, pad_frac: float) -> int:
    """Crop each region from Docling's page raster with a fixed fractional pad. Return count."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, r in enumerate(regions):
        page = doc.pages.get(r["page"])
        if page is None or getattr(page, "image", None) is None:
            continue
        pil = page.image.pil_image
        pw = float(page.size.width)
        ph = r["page_h"]
        rl, rt, rr, rb = r["bbox_tl"]
        sx, sy = pil.width / pw, pil.height / ph
        x0, y0, x1, y1 = rl * sx, rt * sy, rr * sx, rb * sy
        dx, dy = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
        px0, py0 = max(0, int(x0 - dx)), max(0, int(y0 - dy))
        px1, py1 = min(pil.width, int(x1 + dx)), min(pil.height, int(y1 + dy))
        if px1 <= px0 or py1 <= py0:
            continue
        label = r.get("label")
        stem = f"{re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')}" if label else f"region_{i:04d}"
        page.image.pil_image.crop((px0, py0, px1, py1)).save(crops_dir / f"{stem}_p{r['page']}.png", "PNG")
        saved += 1
    return saved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, help="Output dir (default data/detector_bakeoff/<stem>).")
    ap.add_argument("--images-scale", type=float, default=3.0, help="Docling raster scale (72*scale≈DPI).")
    ap.add_argument("--pad-frac", type=float, default=0.05, help="Fixed fractional crop pad per side.")
    ap.add_argument("--all", action="store_true", help="Skip label filter; crop every detected region.")
    args = ap.parse_args(argv)

    for p in (args.pdf, args.gold):
        if not p.is_file():
            print(f"error: file not found: {p}", file=sys.stderr)
            return 2

    out_dir = (args.out_dir or Path("data/detector_bakeoff") / args.pdf.stem).resolve()
    crops_dir = out_dir / "docling" / "crops"
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    gold_labels = [str(g.get("label")) for g in gold.get("equations", []) if g.get("verified")]

    t0 = time.monotonic()
    print(f"[1/3] Docling detection on {args.pdf.name} (scale={args.images_scale}) ...")
    doc, regions = detect_regions(args.pdf, args.images_scale)
    t_detect = time.monotonic() - t0

    labels_by_page = scan_label_positions(args.pdf)
    n_labels = sum(len(v[1]) for v in labels_by_page.values())
    has_labels = n_labels > 0 and not args.all

    if has_labels:
        kept, docling_missed = associate_labels(regions, labels_by_page)
        mode = "labeled"
    else:
        kept = [{**r, "source": "docling"} for r in regions]
        docling_missed = []
        mode = "all"

    saved = crop_regions(doc, kept, crops_dir, args.pad_frac)
    elapsed = round(time.monotonic() - t0, 1)

    # Source breakdown + recall vs gold labels.
    by_source: dict[str, int] = {}
    for r in kept:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    kept_labels = {r.get("label") for r in kept if r.get("label")}
    gold_hit = sum(1 for gl in gold_labels if gl in kept_labels)
    # A gold label is "via docling" if ANY of its crops came from a detected region; it is
    # counted as fallback only when its sole crop is the label-reconstruction fallback.
    docling_labels = {r.get("label") for r in kept if r["source"] in ("docling", "docling_split")}
    gold_via_fallback = sorted(
        gl for gl in gold_labels if gl in kept_labels and gl not in docling_labels
    )

    report = {
        "pdf": str(args.pdf), "gold": str(args.gold), "mode": mode,
        "timing_s": {"detect": round(t_detect, 1), "total": elapsed},
        "regions_detected": len(regions),
        "labels_found": n_labels,
        "crops_by_source": by_source,
        "crops_saved": saved,
        "docling_true_misses": sorted(docling_missed),
        "gold": {
            "gold_labels": len(gold_labels),
            "recall": round(gold_hit / len(gold_labels), 4) if gold_labels else None,
            "gold_via_docling": gold_hit - len(gold_via_fallback),
            "gold_via_label_fallback": gold_via_fallback,
            "missing_gold_labels": [gl for gl in gold_labels if gl not in kept_labels],
        },
        "crops_dir": str(crops_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n=== Docling detect → label-filter (+fallback) → crop  [{mode}]  ({elapsed}s) ===")
    print(f"  regions detected   : {len(regions)}")
    print(f"  labels found (pdf)  : {n_labels}")
    print(f"  crops by source     : {by_source}")
    if gold_labels:
        print(f"  gold recall         : {gold_hit}/{len(gold_labels)} = {report['gold']['recall']}")
        print(f"    via docling box   : {report['gold']['gold_via_docling']}")
        print(f"    via label fallback: {len(gold_via_fallback)}  {gold_via_fallback}")
        if report["gold"]["missing_gold_labels"]:
            print(f"  STILL MISSING       : {report['gold']['missing_gold_labels']}")
    print(f"\nCrops: {crops_dir}\nReport: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
