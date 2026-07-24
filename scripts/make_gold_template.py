#!/usr/bin/env python3
"""Bootstrap a gold-set template from a pipeline ``document.json`` for human verification.

The pipeline's *current* LaTeX is written into each entry's ``gold_latex`` as a STARTING
POINT, with ``verified: false``. A human then opens the crop, corrects ``gold_latex`` where
wrong, and flips ``verified`` to ``true``. Only verified entries are scored by
``eval_latex_accuracy.py`` — so a freshly generated template contributes nothing to the
metrics until a human has actually checked it.

Usage
-----
    python scripts/make_gold_template.py --document data/output/28120_12/document.json
    # writes data/ground_truth/28120_12.json (use --force to overwrite)

Gold-set schema (data/ground_truth/<book_id>.json)::

    {
      "book_id": "28120_12",
      "source_pdf": "data/input/28120_12.pdf",
      "mode": "labeled",                       # "labeled" | "unlabeled"
      "equations": [
        {
          "equation_id": "eq_0_p3_12_2_1",     # links back to pipeline output
          "label": "12.2.1",                   # matching key (reference label)
          "page_number": 3,
          "crop_path": "crops/page_003/eq_0_p3_12_2_1.png",
          "gold_latex": "t = \\left(...\\right)^{1/3}",
          "verified": false                    # human sets true after checking
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT_DIR = PROJECT_ROOT / "data" / "ground_truth"


def build_template(document: dict) -> dict:
    doc = document.get("document", document)
    eqs = doc.get("equations", [])
    mode = "labeled" if any(e.get("detection_method") == "label" for e in eqs) else "unlabeled"
    entries = []
    for e in eqs:
        final = e.get("final") or {}
        entries.append(
            {
                "equation_id": e.get("equation_id"),
                "label": e.get("label") or e.get("equation_number"),
                "page_number": e.get("page_number"),
                "crop_path": (e.get("crop") or {}).get("path"),
                "gold_latex": final.get("latex") or (e.get("ocr") or {}).get("latex") or "",
                "verified": False,
            }
        )
    return {
        "book_id": document.get("document_id") or doc.get("book_id"),
        "source_pdf": f"data/input/{doc.get('source_filename', '')}",
        "mode": mode,
        "equations": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--document", required=True, type=Path, help="Pipeline document.json to seed from.")
    parser.add_argument("--out", type=Path, help="Output gold file (default: data/ground_truth/<book_id>.json).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing gold file.")
    args = parser.parse_args(argv)

    if not args.document.is_file():
        print(f"error: document not found: {args.document}", file=sys.stderr)
        return 2

    document = json.loads(args.document.read_text(encoding="utf-8"))
    template = build_template(document)
    out = args.out or (DEFAULT_GT_DIR / f"{template['book_id']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not args.force:
        print(f"refusing to overwrite existing gold file: {out} (use --force)", file=sys.stderr)
        return 1

    out.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n = len(template["equations"])
    print(f"Wrote {out} ({n} entries, mode={template['mode']}, all verified=false).")
    print("Next: open each crop, correct gold_latex, set verified=true, then run eval_latex_accuracy.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
