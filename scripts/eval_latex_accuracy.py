#!/usr/bin/env python3
"""Score extraction accuracy AND validate the GPT judge against a human gold set.

Reads a pipeline ``document.json`` and a gold file (see ``make_gold_template.py``) and reports:

1. Detection recall  — of the gold (in-scope) equations, how many did the pipeline extract?
2. LaTeX fidelity     — for matched equations, token/char similarity + exact-match rate.
3. Judge validation   — does the GPT judge's accept/ai_score agree with the gold truth?
                        (accept-precision, accept-recall, and mean ai_score for correct vs
                        incorrect equations). This is how we learn whether to trust the judge.

Only gold entries with ``verified: true`` are scored, so an un-reviewed template contributes
nothing. Matching is by ``label`` (falling back to ``equation_id``).

Usage
-----
    python scripts/eval_latex_accuracy.py \
        --document data/output/28120_12/document.json \
        --gold data/ground_truth/28120_12.json [--csv out.csv] [--correct-threshold 0.95]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from equation_extraction_pipeline.evaluation.latex_metrics import score_pair  # noqa: E402


def _key(entry: dict) -> str:
    return str(entry.get("label") or entry.get("equation_id") or "").strip()


def _output_index(document: dict) -> dict[str, dict]:
    doc = document.get("document", document)
    idx: dict[str, dict] = {}
    for e in doc.get("equations", []):
        k = str(e.get("label") or e.get("equation_number") or e.get("equation_id") or "").strip()
        if k:
            idx[k] = e
    return idx


def _pred_latex(eq: dict) -> str:
    return (eq.get("final") or {}).get("latex") or (eq.get("ocr") or {}).get("latex") or ""


def _judge(eq: dict) -> dict | None:
    return eq.get("judge")


def evaluate(document: dict, gold: dict, correct_threshold: float) -> dict:
    out_idx = _output_index(document)
    gold_eqs = [g for g in gold.get("equations", []) if g.get("verified")]

    rows = []
    found = 0
    for g in gold_eqs:
        k = _key(g)
        eq = out_idx.get(k)
        row = {
            "key": k,
            "page": g.get("page_number"),
            "found": eq is not None,
            "gold_latex": g.get("gold_latex", ""),
            "pred_latex": _pred_latex(eq) if eq else "",
        }
        if eq is not None:
            found += 1
            row.update(score_pair(row["pred_latex"], row["gold_latex"]))
            j = _judge(eq)
            row["judge_ai_score"] = (j or {}).get("ai_score")
            row["judge_accepted"] = (j or {}).get("accepted")
        else:
            row.update({"exact_match": 0.0, "char_similarity": 0.0, "token_similarity": 0.0,
                        "judge_ai_score": None, "judge_accepted": None})
        rows.append(row)

    n_gold = len(gold_eqs)
    matched = [r for r in rows if r["found"]]

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # Judge validation: "actually correct" == token_similarity >= threshold (gold-grounded).
    judge_rows = [r for r in matched if r["judge_accepted"] is not None]
    actually_correct = [r for r in judge_rows if r["token_similarity"] >= correct_threshold]
    actually_wrong = [r for r in judge_rows if r["token_similarity"] < correct_threshold]
    accepted = [r for r in judge_rows if r["judge_accepted"]]
    tp = sum(1 for r in accepted if r["token_similarity"] >= correct_threshold)
    accept_precision = round(tp / len(accepted), 4) if accepted else None
    accept_recall = round(tp / len(actually_correct), 4) if actually_correct else None

    return {
        "book_id": gold.get("book_id"),
        "mode": gold.get("mode"),
        "correct_threshold": correct_threshold,
        "detection": {
            "gold_verified": n_gold,
            "found": found,
            "recall": round(found / n_gold, 4) if n_gold else None,
            "missing_keys": [r["key"] for r in rows if not r["found"]],
        },
        "fidelity": {
            "evaluated": len(matched),
            "exact_match_rate": _mean([r["exact_match"] for r in matched]),
            "mean_char_similarity": _mean([r["char_similarity"] for r in matched]),
            "mean_token_similarity": _mean([r["token_similarity"] for r in matched]),
        },
        "judge_validation": {
            "judged": len(judge_rows),
            "accept_precision": accept_precision,  # of accepted, fraction actually correct
            "accept_recall": accept_recall,        # of correct, fraction the judge accepted
            "mean_ai_score_correct": _mean([r["judge_ai_score"] for r in actually_correct]),
            "mean_ai_score_wrong": _mean([r["judge_ai_score"] for r in actually_wrong]),
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--document", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--csv", type=Path, help="Optional per-equation CSV output.")
    parser.add_argument("--correct-threshold", type=float, default=0.95,
                        help="token_similarity at/above which an equation counts as correct (default 0.95).")
    args = parser.parse_args(argv)

    for p in (args.document, args.gold):
        if not p.is_file():
            print(f"error: file not found: {p}", file=sys.stderr)
            return 2

    document = json.loads(args.document.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    result = evaluate(document, gold, args.correct_threshold)

    if result["detection"]["gold_verified"] == 0:
        print("No verified gold entries yet — verify some entries (verified=true) and re-run.")
        return 0

    summary = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(result["rows"][0].keys()))
            w.writeheader()
            w.writerows(result["rows"])
        print(f"\nPer-equation CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
