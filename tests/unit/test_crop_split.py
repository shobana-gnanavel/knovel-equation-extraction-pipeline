from __future__ import annotations

from PIL import Image, ImageDraw

from equation_extraction_pipeline.common.utils import split_stacked_crop


def test_fraction_denominator_is_not_split_into_an_equation() -> None:
    image = Image.new("L", (200, 90), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 10, 175, 35), fill="black")
    draw.rectangle((92, 65, 105, 80), fill="black")  # narrow denominator

    assert split_stacked_crop(image) == [image]


def test_two_wide_equation_bands_are_split() -> None:
    image = Image.new("L", (200, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 8, 175, 30), fill="black")
    draw.rectangle((25, 68, 170, 92), fill="black")

    parts = split_stacked_crop(image)

    assert len(parts) == 2
