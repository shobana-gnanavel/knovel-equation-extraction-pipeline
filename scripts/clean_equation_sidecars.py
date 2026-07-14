"""Post-processing cleanup: remove trivial inline-equation false positives from existing sidecars.

Applies the same _TRIVIAL_INLINE_FRAGMENT filter that is now baked into detect_inline_spans()
to the pre-computed .equation_extraction.json sidecar files, so the dashboard reflects the
corrected results immediately without needing a full pipeline re-run.

Usage:
    python scripts/clean_equation_sidecars.py                   # dry-run (print summary only)
    python scripts/clean_equation_sidecars.py --apply           # write cleaned files in-place
    python scripts/clean_equation_sidecars.py --apply --verbose # show every removed equation
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Reuse the canonical trivial-fragment filter from the detection module.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from equation_extraction.detection import _TRIVIAL_INLINE_FRAGMENT  # noqa: E402


def _is_trivial(eq: dict) -> bool:
    """True when an inline equation's plain_text matches the trivial-fragment filter."""
    if not eq.get("is_inline"):
        return False
    plain = (eq.get("plain_text") or "").strip()
    return bool(plain and _TRIVIAL_INLINE_FRAGMENT.match(plain))


def _recalc_page(page: dict) -> dict:
    """Return a copy of *page* with trivial inline equations removed and stats recalculated."""
    kept = [eq for eq in page.get("equations", []) if not _is_trivial(eq)]
    cat_counts: Counter[str] = Counter(eq["category"] for eq in kept)
    prov_counts: Counter[str] = Counter(eq["selected_provider"] for eq in kept)

    if page.get("failure_reason"):
        outcome = "degraded"
    elif not kept:
        outcome = "empty"
    elif any(
        "unsupported_category" in eq.get("validation_flags", [])
        or any(n.startswith(("recognition_failed", "provider_absent")) for n in eq.get("notes", []))
        for eq in kept
    ):
        outcome = "partial"
    else:
        outcome = "extracted"

    return {
        **page,
        "equations": kept,
        "outcome": outcome,
        "category_counts": dict(cat_counts),
        "provider_counts": dict(prov_counts),
    }


def _recalc_statistics(pages: list[dict], doc_equations: list[dict], orig_stats: dict) -> dict:
    cat_dist: Counter[str] = Counter(eq["category"] for eq in doc_equations)
    by_provider: Counter[str] = Counter(eq["selected_provider"] for eq in doc_equations)
    validation_counts: Counter[str] = Counter()
    latex_valid = mathml_valid = low_class = low_recog = 0

    for eq in doc_equations:
        for flag in eq.get("validation_flags", []):
            validation_counts[flag] += 1
        if eq.get("latex") and "invalid_latex" not in eq.get("validation_flags", []):
            latex_valid += 1
        if eq.get("mathml") and "invalid_mathml" not in eq.get("validation_flags", []):
            mathml_valid += 1
        if eq.get("classification_confidence", 1.0) < 0.5:
            low_class += 1
        if eq.get("recognition_confidence", 1.0) < 0.5:
            low_recog += 1

    return {
        **orig_stats,
        "total_equations": len(doc_equations),
        "category_distribution": dict(cat_dist),
        "equations_by_provider": dict(by_provider),
        "low_confidence_classification_count": low_class,
        "low_confidence_recognition_count": low_recog,
        "latex_valid_count": latex_valid,
        "mathml_valid_count": mathml_valid,
        "validation_counts": dict(validation_counts),
        "failures": sum(1 for p in pages if p.get("outcome") == "degraded"),
    }


def clean_file(path: Path, *, apply: bool, verbose: bool) -> tuple[int, int]:
    """Clean one sidecar file.  Returns (equations_before, equations_after)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    ctx = data.get("context", {})
    if ctx.get("outcome") == "failed":
        return 0, 0

    orig_equations: list[dict] = ctx.get("equations", [])
    orig_count = len(orig_equations)

    cleaned_pages = [_recalc_page(p) for p in ctx.get("pages", [])]
    doc_equations = [eq for page in cleaned_pages for eq in page["equations"]]
    new_count = len(doc_equations)
    removed = orig_count - new_count

    if verbose and removed:
        removed_eqs = [eq for eq in orig_equations if _is_trivial(eq)]
        for eq in removed_eqs:
            print(f"  REMOVE [{eq.get('equation_id')}] plain={repr(eq.get('plain_text',''))}")

    if apply and removed:
        stats = _recalc_statistics(cleaned_pages, doc_equations, ctx.get("statistics", {}))
        ctx_new = {
            **ctx,
            "equations": doc_equations,
            "pages": cleaned_pages,
            "statistics": stats,
            "outcome": "degraded" if stats["failures"] > 0 else ("empty" if not doc_equations else "extracted"),
        }
        data["context"] = ctx_new
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return orig_count, new_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write cleaned files (default: dry-run)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show each removed equation")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Specific sidecar files or directories (default: data/)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    if args.paths:
        candidates: list[Path] = []
        for p in args.paths:
            pp = Path(p)
            if pp.is_dir():
                candidates.extend(pp.rglob("*.equation_extraction.json"))
            else:
                candidates.append(pp)
    else:
        candidates = list((root / "data").rglob("*.equation_extraction.json"))

    if not candidates:
        print("No .equation_extraction.json files found.")
        return

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Scanning {len(candidates)} equation extraction sidecar(s)...\n")

    total_before = total_after = total_files_changed = 0
    for path in sorted(candidates):
        before, after = clean_file(path, apply=args.apply, verbose=args.verbose)
        removed = before - after
        total_before += before
        total_after += after
        if removed:
            total_files_changed += 1
            print(f"  {'CLEANED' if args.apply else 'WOULD CLEAN':10s}  {path.name:<55}  {before:>4} → {after:>4}  (-{removed})")
        else:
            print(f"  {'OK':10s}  {path.name}")

    print(f"\n{'─'*70}")
    print(f"Files changed     : {total_files_changed} / {len(candidates)}")
    print(f"Equations before  : {total_before}")
    print(f"Equations after   : {total_after}")
    print(f"False positives removed : {total_before - total_after}  ({100*(total_before-total_after)/max(total_before,1):.1f}%)")
    if not args.apply:
        print("\nRun with --apply to write the cleaned files.")


if __name__ == "__main__":
    main()
