from __future__ import annotations

from equation_extraction_pipeline import main as main_pipeline


def test_one_detected_region_remains_one_logical_equation() -> None:
    crop = object()

    assert main_pipeline._logical_equation_crops(crop) == [crop]
