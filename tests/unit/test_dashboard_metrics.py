from __future__ import annotations

import json
import sys
from pathlib import Path

# Dashboard lives in others/ which is not installed as a package;
# add it to sys.path for this test only.
_OTHERS = Path(__file__).resolve().parents[2] / "others"
if str(_OTHERS) not in sys.path:
    sys.path.insert(0, str(_OTHERS))

from dashboard import app as dashboard  # noqa: E402

from equation_extraction_pipeline.detection import (  # noqa: E402
    equation_label_detector as layout_detection,
)


def _equation(label: str, *, accepted: bool, score: float) -> dict:
    return {
        "equation_id": f"eq_{label}",
        "page_number": 1,
        "label": label,
        "equation_number": label,
        "ocr": {"latex": "x=1", "confidence": 0.8, "provider": "test"},
        "final": {"latex": "x=1", "overall_confidence": 0.8, "status": "SUCCESS"},
        "judge": {"accepted": accepted, "score": score, "reason": "test"},
        "validation_flags": [],
    }


def test_document_metrics_use_pdf_labels_and_non_overlapping_verdicts(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "book.pdf").touch()
    monkeypatch.setattr(dashboard, "INPUT_DIR", input_dir)
    monkeypatch.setattr(
        layout_detection,
        "scan_equation_labels",
        lambda _path: ["1(a)", "1(b)", "2(a)", "2(b)", "3", "4"],
    )

    equations = [
        _equation("1(a)", accepted=True, score=1.0),
        _equation("1(b)", accepted=True, score=0.8),
        _equation("3", accepted=True, score=0.7),
        _equation("4", accepted=False, score=0.5),
    ]
    document_path = tmp_path / "document.json"
    document_path.write_text(
        json.dumps({"document": {"equations": equations, "summary": {"success": 3, "rejected": 1}}}),
        encoding="utf-8",
    )

    dashboard._build_validation_from_document_json("book", document_path, tmp_path / "validation")
    metrics = json.loads(
        (tmp_path / "validation" / "book" / "equation_validation_metrics.json").read_text()
    )
    judge = metrics["llm_judge"]

    assert metrics["pdf_labeled_count"] == 6
    assert metrics["pdf_extracted_labeled_count"] == 4
    assert metrics["pdf_coverage_pct"] == 66.7
    assert judge["missing_labels"] == ["2(a)", "2(b)"]
    assert (judge["accepted"], judge["reviewed"], judge["rejected"]) == (3, 0, 1)
    assert judge["mean_overall"] == 7.5
    assert judge["mean_confidence"] == 0.75


def test_legacy_collapsed_label_matches_only_one_sub_equation(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "book.pdf").touch()
    monkeypatch.setattr(dashboard, "INPUT_DIR", input_dir)
    monkeypatch.setattr(
        layout_detection,
        "scan_equation_labels",
        lambda _path: ["3.9.1(a)", "3.9.1(b)"],
    )
    document_path = tmp_path / "document.json"
    document_path.write_text(
        json.dumps({"document": {"equations": [_equation("3.9.1", accepted=True, score=1.0)], "summary": {}}}),
        encoding="utf-8",
    )

    dashboard._build_validation_from_document_json("book", document_path, tmp_path / "validation")
    metrics = json.loads(
        (tmp_path / "validation" / "book" / "equation_validation_metrics.json").read_text()
    )

    assert metrics["pdf_extracted_labeled_count"] == 1
    assert metrics["pdf_coverage_pct"] == 50.0
    assert metrics["llm_judge"]["missing_labels"] == ["3.9.1(b)"]
