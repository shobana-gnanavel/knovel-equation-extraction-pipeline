from __future__ import annotations

from PIL import Image

from equation_extraction_pipeline.extraction import ocr_extractor as llm_judge


def test_single_glyph_fragment_is_rejected_without_calling_model(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_judge,
        "_call_ollama_generate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model must not run")),
    )

    verdict = llm_judge.judge_latex("b", Image.new("RGB", (20, 20), "white"))

    assert verdict.accepted is False
    assert verdict.score == 0.0
    assert "fragment" in verdict.reason


def test_judge_prompt_disambiguates_contour_integral_from_phi() -> None:
    prompt = llm_judge._JUDGE_PROMPT_TEMPLATE.format(
        latex=r"\theta=\oint ds/t"
    )

    assert r"\oint" in prompt
    assert "differential notation" in prompt
