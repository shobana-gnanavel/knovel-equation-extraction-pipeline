"""Corpus-wide crop-quality report: hybrid detection + crop only, no recognition.

Runs INSIDE the pipeline container (scripts/ is not bind-mounted — ``docker cp`` it in):

    docker cp scripts/crop_quality_report.py equation_extraction_pipeline-pipeline-1:/tmp/
    docker exec -e EQUATION_DETECTOR=hybrid equation_extraction_pipeline-pipeline-1 \
        python /tmp/crop_quality_report.py

Per book: label recall proxy (labels cropped vs scanned), docling/reconstruction source
split, sliver rate, edge-ink clip heuristic, crop dimension stats; plus a contact sheet of
the 6 worst + 3 median crops for eyeballing. Output under /app/data/output_gen/crop_report/.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

os.environ.setdefault("EQUATION_DETECTOR", "hybrid")

import numpy as np
from PIL import Image

# Import order matters: detection before ingestion (mirrors main.py; avoids circular import).
from equation_extraction_pipeline.detection.equation_label_detector import (  # noqa: E402
    detect_equations,
    scan_equation_labels,
)
from equation_extraction_pipeline.extraction.page_renderer import render_pages  # noqa: E402
from equation_extraction_pipeline.ingestion.pdf_loader import classify_pdf  # noqa: E402

INPUT_DIR = Path("/app/data/input")
OUT_ROOT = Path("/app/data/output_gen/crop_report")

INK_THRESHOLD = 128       # grayscale value below which a pixel counts as ink
SLIVER_HEIGHT_PX = 40
EDGE_INK_FRAC = 0.10      # a border row/col with >10% ink pixels = suspected clip


class _SourceCapture(logging.Handler):
    """Grab the hybrid_crop_sources log line so the split lands in the report."""

    def __init__(self) -> None:
        super().__init__()
        self.docling = self.reconstruction = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.msg.startswith("hybrid_crop_sources"):
            self.docling, self.reconstruction = record.args  # type: ignore[misc]


def crop_metrics(png: Path) -> dict:
    img = Image.open(png).convert("L")
    a = np.asarray(img)
    h, w = a.shape
    ink = a < INK_THRESHOLD
    edges = {
        "top": ink[0, :].mean(),
        "bottom": ink[-1, :].mean(),
        "left": ink[:, 0].mean(),
        "right": ink[:, -1].mean(),
    }
    clipped = [side for side, frac in edges.items() if frac > EDGE_INK_FRAC]
    return {
        "path": png,
        "w": w,
        "h": h,
        "ink_frac": float(ink.mean()),
        "sliver": h < SLIVER_HEIGHT_PX,
        "clip_sides": clipped,
        "blank": float(ink.mean()) < 0.002,
    }


def contact_sheet(rows: list[dict], dest: Path, cell_w: int = 700) -> None:
    """Stack crops vertically (scaled to cell_w) with a caption strip."""
    if not rows:
        return
    tiles = []
    for r in rows:
        img = Image.open(r["path"]).convert("RGB")
        scale = cell_w / img.width
        img = img.resize((cell_w, max(1, int(img.height * scale))))
        tiles.append((img, f"{r['path'].name}  {r['w']}x{r['h']}  clip={','.join(r['clip_sides']) or '-'}"))
    pad, cap_h = 8, 16
    total_h = sum(t.height + cap_h + pad for t, _ in tiles)
    sheet = Image.new("RGB", (cell_w + 2 * pad, total_h + pad), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    y = pad
    for img, caption in tiles:
        draw.text((pad, y), caption, fill=(180, 30, 30))
        y += cap_h
        sheet.paste(img, (pad, y))
        draw.rectangle((pad, y, pad + img.width - 1, y + img.height - 1), outline=(200, 200, 200))
        y += img.height + pad
    sheet.save(dest)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Crop-quality report (hybrid detector, detection-only)",
        "",
        "| book | labels scanned | regions | crops | docling/recon | slivers | clipped | blank | median WxH |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    detector_logger = logging.getLogger(
        "equation_extraction_pipeline.detection.equation_label_detector"
    )
    detector_logger.setLevel(logging.INFO)  # hybrid_crop_sources is INFO; default is WARNING
    for pdf in sorted(INPUT_DIR.glob("*.pdf")):
        book = pdf.stem
        t0 = time.time()
        cap = _SourceCapture()
        detector_logger.addHandler(cap)
        try:
            labels = scan_equation_labels(pdf)
            cls = classify_pdf(pdf)
            pages = render_pages(pdf, cls)
            book_out = OUT_ROOT / book
            regions = detect_equations(pdf, pages, cls, book_out)
        finally:
            detector_logger.removeHandler(cap)
        crops = [book_out / r.crop_path for r in regions if r.crop_path]
        metrics = [crop_metrics(p) for p in crops if p.exists()]
        slivers = [m for m in metrics if m["sliver"]]
        clipped = [m for m in metrics if m["clip_sides"]]
        blank = [m for m in metrics if m["blank"]]
        med_w = int(np.median([m["w"] for m in metrics])) if metrics else 0
        med_h = int(np.median([m["h"] for m in metrics])) if metrics else 0
        lines.append(
            f"| {book} | {len(labels)} | {len(regions)} | {len(metrics)} "
            f"| {cap.docling}/{cap.reconstruction} | {len(slivers)} | {len(clipped)} "
            f"| {len(blank)} | {med_w}x{med_h} |"
        )
        # Worst = blank/sliver/clipped first, then smallest area; plus 3 median-area crops.
        ranked = sorted(
            metrics,
            key=lambda m: (
                -(m["blank"] * 3 + m["sliver"] * 2 + bool(m["clip_sides"])),
                m["w"] * m["h"],
            ),
        )
        mid = len(ranked) // 2
        sample = ranked[:6] + ranked[max(6, mid - 1): max(6, mid - 1) + 3]
        contact_sheet(sample, OUT_ROOT / f"{book}_sheet.png")
        print(f"{book}: done in {time.time() - t0:.0f}s "
              f"(labels={len(labels)} regions={len(regions)} slivers={len(slivers)} "
              f"clipped={len(clipped)} blank={len(blank)})", flush=True)

    (OUT_ROOT / "report.md").write_text("\n".join(lines) + "\n")
    print("report written to", OUT_ROOT / "report.md", flush=True)


if __name__ == "__main__":
    main()
