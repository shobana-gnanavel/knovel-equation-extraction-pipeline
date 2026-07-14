from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from equation_extraction_pipeline.detection import equation_label_detector as layout


@dataclass
class _Box:
    bbox: tuple[float, float, float, float]


@dataclass
class _TextBox:
    bbox: tuple[float, float, float, float]
    text: str

    def get_text(self) -> str:
        return self.text


def test_sub_equation_suffix_is_captured_and_normalised() -> None:
    a = layout._LABEL_RE.search("Eq. 3.9.l(a)")
    b = layout._LABEL_RE.search("Eq. 3.9.l(b)")

    assert a is not None and b is not None
    assert layout._normalise_label(a.group(1)) == "3.9.1(a)"
    assert layout._normalise_label(b.group(1)) == "3.9.1(b)"
    assert layout._normalise_label(a.group(1)) != layout._normalise_label(b.group(1))


def test_standalone_sub_equation_label_is_not_a_cross_reference() -> None:
    assert layout._is_cross_reference("Eq. 3.9.l(a)") is False
    assert layout._is_cross_reference("From Eq. 3.9.l(a):") is True


def test_formula_bbox_merges_split_fraction_fragments() -> None:
    label = _Box((378.5, 596.65, 418.18, 607.15))
    formula = [
        _Box((92.6, 596.45, 128.23, 606.95)),   # left-hand side
        _Box((123.8, 588.77, 140.71, 613.84)),  # stacked fraction
        _Box((35.5, 563.47, 91.97, 575.27)),    # unrelated text above
    ]

    assert layout._formula_bbox_for_label(label, formula) == (
        88.6,
        588.77,
        366.5,
        613.84,
    )


def test_image_only_formula_bbox_excludes_the_margin_label() -> None:
    label = _Box((379.58, 315.11, 431.27, 325.51))

    bbox = layout._image_formula_bbox_for_label(label)

    assert bbox[0] < bbox[2] < label.bbox[0]
    assert bbox[1] < label.bbox[1]
    assert bbox[3] > label.bbox[3]


def test_image_formula_bbox_keeps_stacked_fraction_numerator() -> None:
    label = _Box((379.2, 321.4, 416.0, 331.8))

    bbox = layout._image_formula_bbox_for_label(label)

    # Unmapped stacked fractions commonly reach two text heights above the
    # label baseline, but need less space below it.
    assert bbox == pytest.approx((91.008, 313.6, 367.2, 352.6))


def test_pdf_label_scan_rejects_numbered_list_marker(monkeypatch) -> None:
    boxes = [
        _TextBox((55, 400, 70, 412), "(1)"),
        _TextBox((78, 400, 220, 412), "Given the splice joint below."),
        _TextBox((90, 300, 250, 320), "x = 1"),
        _TextBox((380, 302, 410, 314), "(2)"),
    ]
    monkeypatch.setattr(layout, "_extract_page_layout", lambda _path: [(0, boxes)])

    assert layout.scan_equation_labels(Path("dummy.pdf")) == ["2"]
