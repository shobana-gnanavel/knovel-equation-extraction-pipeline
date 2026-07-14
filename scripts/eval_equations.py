"""Score equation extraction against human-labeled ground truth (recall / precision / F1).

Turns "did we capture the equations correctly?" into a number. Reads the equation-extraction
sidecar the pipeline already writes next to a PDF (``<pdf>.equation_extraction.json``), compares the
equation numbers it contains against a ground-truth file you provide, and reports recall, precision,
F1, and the exact missed / spurious equation numbers.

Run the pipeline first (so the sidecar exists), then evaluate:

    python scripts/equations.py --pdf data/input/28120_12.pdf          # produces the sidecar
    python scripts/eval_equations.py --pdf data/input/28120_12.pdf \\
        --ground-truth data/ground_truth/28120_12.json

Ground-truth file — either a JSON object::

    {"equation_numbers": ["12.2.1", "12.2.2", "12.4.4", "12.4.10"]}

or a plain-text file with one equation number per line (``#`` comments allowed).

Evaluate a whole directory of PDFs that each have a matching ``<name>.json`` in --gt-dir:

    python scripts/eval_equations.py --input-dir data/input --gt-dir data/ground_truth --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quality.equation_eval import EquationEvalResult, evaluate_equations  # noqa: E402


def detected_numbers_from_sidecar(sidecar_path: Path) -> list[str]:
    """Pull every non-empty ``equation_number`` from an ``<pdf>.equation_extraction.json`` sidecar."""
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    context = payload.get("context", payload)  # tolerate a bare context or the wrapped sidecar
    numbers: list[str] = []
    for eq in context.get("equations", []):
        num = eq.get("equation_number")
        if num:
            numbers.append(str(num))
    return numbers


def load_ground_truth(gt_path: Path) -> list[str]:
    """Load expected equation numbers from a JSON object or a one-per-line text file."""
    raw = gt_path.read_text(encoding="utf-8").strip()
    if raw.startswith("{") or raw.startswith("["):
        data = json.loads(raw)
        if isinstance(data, dict):
            return [str(x) for x in data.get("equation_numbers", [])]
        return [str(x) for x in data]
    # Plain text: one number per line, '#' comments and blank lines ignored.
    out: list[str] = []
    for line in raw.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def evaluate_pdf(pdf_path: Path, gt_path: Path) -> EquationEvalResult:
    """Evaluate one PDF's sidecar against its ground-truth file."""
    sidecar = pdf_path.with_suffix(".equation_extraction.json")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"No sidecar {sidecar.name} — run the equation stage for this PDF first."
        )
    return evaluate_equations(detected_numbers_from_sidecar(sidecar), load_ground_truth(gt_path))


def _print_report(name: str, result: EquationEvalResult) -> None:
    print(f"\n=== {name} ===")
    print(
        f"  expected={result.expected_count}  detected={result.detected_count}  "
        f"true_positives={result.true_positives}"
    )
    print(f"  recall={result.recall:.3f}  precision={result.precision:.3f}  f1={result.f1:.3f}")
    if result.missed:
        print(f"  MISSED ({len(result.missed)}): {', '.join(result.missed)}")
    if result.spurious:
        print(f"  SPURIOUS ({len(result.spurious)}): {', '.join(result.spurious)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdf", type=Path, help="Single PDF to evaluate.")
    parser.add_argument("--ground-truth", type=Path, help="Ground-truth file for --pdf.")
    parser.add_argument("--input-dir", type=Path, help="Directory of PDFs to evaluate.")
    parser.add_argument("--gt-dir", type=Path, help="Directory of <name>.json ground-truth files.")
    parser.add_argument("--csv", type=Path, help="Optional CSV summary output.")
    args = parser.parse_args(argv)

    jobs: list[tuple[Path, Path]] = []
    if args.pdf and args.ground_truth:
        jobs.append((args.pdf, args.ground_truth))
    elif args.input_dir and args.gt_dir:
        for pdf in sorted(args.input_dir.glob("*.pdf")):
            gt = args.gt_dir / f"{pdf.stem}.json"
            if gt.exists():
                jobs.append((pdf, gt))
            else:
                print(f"skip {pdf.name}: no ground truth {gt.name}", file=sys.stderr)
    else:
        parser.error("provide --pdf + --ground-truth, or --input-dir + --gt-dir")

    rows: list[dict] = []
    micro_tp = micro_expected = micro_detected = 0
    for pdf, gt in jobs:
        try:
            result = evaluate_pdf(pdf, gt)
        except (FileNotFoundError, ValueError) as exc:
            print(f"skip {pdf.name}: {exc}", file=sys.stderr)
            continue
        _print_report(pdf.name, result)
        rows.append({"pdf": pdf.name, **result.as_dict()})
        micro_tp += result.true_positives
        micro_expected += result.expected_count
        micro_detected += result.detected_count

    if len(rows) > 1:
        micro_recall = micro_tp / micro_expected if micro_expected else 0.0
        micro_precision = micro_tp / micro_detected if micro_detected else 0.0
        print(
            f"\n=== CORPUS (micro-avg over {len(rows)} docs) ===\n"
            f"  recall={micro_recall:.3f}  precision={micro_precision:.3f}"
        )

    if args.csv and rows:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "pdf",
                    "expected_count",
                    "detected_count",
                    "true_positives",
                    "recall",
                    "precision",
                    "f1",
                    "missed",
                    "spurious",
                ],
            )
            writer.writeheader()
            for row in rows:
                row = dict(row)
                row["missed"] = ";".join(row["missed"])
                row["spurious"] = ";".join(row["spurious"])
                writer.writerow(row)
        print(f"\nwrote {args.csv}")

    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
