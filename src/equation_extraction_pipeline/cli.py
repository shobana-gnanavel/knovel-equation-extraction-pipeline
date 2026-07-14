"""Command-line interface for the equation extraction pipeline.

Usage
-----
Single PDF::

    python -m equation_extraction_pipeline.cli --pdf data/input/28120_12.pdf

Batch (all PDFs in input dir)::

    python -m equation_extraction_pipeline.cli --batch

Custom output directory::

    python -m equation_extraction_pipeline.cli --pdf data/input/28120_12.pdf --out /tmp/results
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from equation_extraction_pipeline.config import settings as config
from equation_extraction_pipeline.main import run, run_batch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Equation extraction pipeline — produces document.json + crops/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--pdf", type=Path, help="Single PDF to process")
    src.add_argument(
        "--batch",
        action="store_true",
        help=f"Process all PDFs in {config.INPUT_DIR}",
    )
    parser.add_argument("--out", type=Path, help="Output directory (default: outputs/)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    def cli_progress(stage: str, msg: str, pct: int) -> None:
        print(f"[{pct:3d}%] {stage}: {msg}", flush=True)

    if args.batch:
        paths = run_batch(output_dir=args.out)
        print(f"\nDone. {len(paths)} PDF(s) processed.")
        return 0

    if not args.pdf:
        parser.error("Provide --pdf <path> or --batch")

    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    out_path = run(args.pdf, args.out, progress=cli_progress)
    print(f"\n✓ document.json → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
