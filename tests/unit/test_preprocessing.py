from __future__ import annotations

import cv2
import numpy as np
import pytest

from equation_extraction_pipeline.extraction import text_extractor as preprocessing


def test_nearly_horizontal_page_is_not_rotated_ninety_degrees(monkeypatch) -> None:
    page = np.full((300, 200, 3), 255, dtype=np.uint8)
    page[100:120, 20:180] = 0
    monkeypatch.setattr(cv2, "minAreaRect", lambda _coords: ((0, 0), (1, 1), 89.9))

    assert preprocessing._detect_skew_angle(page) == pytest.approx(-0.1)
