#!/usr/bin/env python3
"""UniMERNet subprocess worker.

Runs inside the isolated .venv_unimernet environment (transformers==4.42.4).
Protocol (newline-delimited JSON on stdin/stdout):
  parent → worker : {"image_path": "/abs/path/to/img.png"}
  worker → parent : {"latex": "...", "confidence": 0.9, "n_bands": 1, "error": null}

Writes {"status": "ready"} to stdout once the model is loaded so the parent
knows it is safe to send requests.  Exits when stdin closes (EOF).

Pre-processing pipeline (applied before every UniMERNet inference):
  1. Ink-band isolation  — find the display-equation row band within the
     larger Docling layout crop, discarding surrounding paragraph text.
  2. Multi-band detection — if the crop contains N > 1 bands (stacked
     equations), pick the densest band and report n_bands in the result.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

# Suppress noisy startup warnings before any imports.
warnings.filterwarnings("ignore")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import tempfile

import numpy as np
import torch
from PIL import Image

MODEL_DIR = os.path.expanduser(
    "~/.cache/unimernet/models--wanderkid--unimernet_base"
    "/snapshots/af898d48ebb1765cd3511d88f5d5f7c92279c731"
)

# ---------------------------------------------------------------------------
# Config yaml builder
# ---------------------------------------------------------------------------

def _write_config(cfg_path: str) -> None:
    yaml = f"""\
model:
  arch: unimernet
  load_finetuned: True
  load_pretrained: False
  pretrained: ""
  finetuned: "{MODEL_DIR}/pytorch_model.pth"
  tokenizer_name: nougat
  tokenizer_config:
    path: {MODEL_DIR}
  model_name: unimernet
  model_type: default
  model_config:
    max_seq_len: 384
    model_name: {MODEL_DIR}

datasets:
  formula_rec_eval:
    vis_processor:
      eval:
        name: "formula_image_eval"
        image_size:
          - 192
          - 672

preprocess:
  vis_processor:
    train:
      name: "formula_image_train"
      image_size:
        - 192
        - 672
    eval:
      name: "formula_image_eval"
      image_size:
        - 192
        - 672
  text_processor:
    train:
      name: "blip_caption"
    eval:
      name: "blip_caption"

run:
  runner: runner_base
  task: unimernet_train
  max_epoch: 1
  evaluate: true
  generate_cfg:
    temperature: 0.2
    do_sample: false
    top_p: 0.95
"""
    with open(cfg_path, "w") as f:
        f.write(yaml)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_model():
    import unimernet.tasks as tasks
    from unimernet.common.config import Config
    from unimernet.processors import load_processor

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        cfg_path = f.name
    try:
        _write_config(cfg_path)
        args = argparse.Namespace(cfg_path=cfg_path, options=None)

        # Redirect stdout → stderr during model loading to avoid polluting the
        # JSON protocol channel with model-init print statements.
        real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            cfg = Config(args)
            task = tasks.setup_task(cfg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = task.build_model(cfg).to(device)
            model.eval()
            vis_processor = load_processor(
                "formula_image_eval",
                cfg.config.datasets.formula_rec_eval.vis_processor.eval,
            )
        finally:
            sys.stdout = real_stdout
    finally:
        os.unlink(cfg_path)

    return model, vis_processor, device


# ---------------------------------------------------------------------------
# Pre-processing: ink-band isolation
# ---------------------------------------------------------------------------
# Docling formula regions are typically larger than the equation itself —
# they may include the equation number label, surrounding paragraph text,
# and adjacent equations.  UniMERNet (trained on isolated equation crops)
# reads all visible text, producing spurious array-with-prose outputs.
#
# Strategy:
#   1. Project the image onto the Y axis (sum dark pixels per row).
#   2. Smooth with a small window to merge fragmented ink rows.
#   3. Detect contiguous "active" bands above a threshold.
#   4. Return the band with the highest total ink mass (= the equation).
#   5. Report n_bands so the caller can flag multi-equation crops.
#
# When N > 1 bands are found the topmost band is returned as the primary
# equation.  The caller notes "unimernet:multi_band:N" for review.
# ---------------------------------------------------------------------------

# A row is considered "active" if its ink coverage exceeds this fraction of
# the image width.  2 % avoids triggering on thin horizontal rules or noise.
_ROW_INK_THRESHOLD_FRAC: float = 0.02

# Gaps of fewer than this many consecutive whitespace rows are bridged —
# prevents splitting fractions, integrals, and stacked sub/superscripts into
# separate bands.
_MIN_GAP_ROWS: int = 6

# Padding added above and below the selected band (pixels) so UniMERNet sees
# ascenders, descenders, and the top/bottom strokes of tall symbols.
_BAND_MARGIN_PX: int = 6


def _find_ink_bands(
    row_proj: np.ndarray,
    threshold: float,
    h: int,
) -> list[tuple[int, int, float]]:
    """Return a list of (top, bottom, ink_mass) tuples for every active band."""
    bands: list[tuple[int, int, float]] = []
    in_band = False
    band_start = 0
    gap_count = 0

    for r in range(h):
        if row_proj[r] > threshold:
            if not in_band:
                in_band = True
                band_start = r
            gap_count = 0
        else:
            if in_band:
                gap_count += 1
                if gap_count >= _MIN_GAP_ROWS:
                    # End of band: close at the row before the gap started.
                    end = r - gap_count + 1
                    mass = float(row_proj[band_start:end].sum())
                    bands.append((band_start, end, mass))
                    in_band = False
                    gap_count = 0

    if in_band:
        end = h
        mass = float(row_proj[band_start:end].sum())
        bands.append((band_start, end, mass))

    return bands


def _preprocess_crop(pil_image: Image.Image) -> tuple[Image.Image, int]:
    """Isolate the equation band from the full layout crop.

    Returns ``(cropped_image, n_bands)`` where *n_bands* is the total number
    of distinct ink bands found (1 = clean single equation, >1 = stacked).
    Falls back to the original image if analysis fails.
    """
    try:
        gray = np.array(pil_image.convert("L"), dtype=np.float32)
        h, w = gray.shape

        # Ink mask: pixels darker than 210 (handles slightly grey ink).
        ink = (gray < 210).astype(np.float32)

        # Row projection, smoothed over 3 rows.
        row_proj = np.convolve(ink.sum(axis=1), np.ones(3) / 3, mode="same")

        threshold = max(w * _ROW_INK_THRESHOLD_FRAC, 2.0)
        bands = _find_ink_bands(row_proj, threshold, h)

        if not bands:
            return pil_image, 1

        n_bands = len(bands)

        # Primary band = tallest band (most rows).  Display equations with
        # fractions, integrals, or sub/superscripts span more rows than a
        # surrounding paragraph line; densest-band (ink mass) picks prose text.
        top, bottom, _ = max(bands, key=lambda b: b[1] - b[0])

        top = max(0, top - _BAND_MARGIN_PX)
        bottom = min(h, bottom + _BAND_MARGIN_PX)

        cropped = pil_image.crop((0, top, w, bottom))
        return cropped, n_bands

    except Exception:
        return pil_image, 1


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _recognize(model, vis_processor, device, image_path: str) -> dict:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        return {"latex": None, "confidence": 0.0, "n_bands": 1, "error": str(exc)}

    try:
        img, n_bands = _preprocess_crop(img)

        tensor = vis_processor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.generate({"image": tensor})
        latex = output["pred_str"][0].strip()

        # Quality signal: non-empty with at least one LaTeX structural token.
        has_math = any(c in latex for c in r"\{}_^") or any(
            op in latex for op in ("frac", "sum", "int", "sqrt", "lim", "cdot")
        )
        # Penalise outputs that look like prose — band isolation may have
        # selected a paragraph line instead of the equation.  Catches:
        #   \mathrm{The relationship...}   (prose wrapped in mathrm)
        #   \mathsf{U n e equation...}     (spaced-glyph reading)
        #   \mathtt{applied loads...}      (typewriter-font prose)
        has_prose = any(
            word in latex.lower()
            for word in ("mathrm", "mathsf", "mathtt", "text{", "mbox{", "hbox{")
        )
        if has_prose:
            confidence = 0.30
        elif latex and has_math:
            confidence = 0.90
        elif latex:
            confidence = 0.60
        else:
            confidence = 0.0

        return {
            "latex": latex or None,
            "confidence": confidence,
            "n_bands": n_bands,
            "error": None,
        }
    except Exception as exc:
        return {"latex": None, "confidence": 0.0, "n_bands": 1, "error": str(exc)}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        model, vis_processor, device = _load_model()
    except Exception as exc:
        _emit({"status": "error", "error": str(exc)})
        sys.exit(1)

    _emit({"status": "ready", "device": str(device)})

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _emit({"latex": None, "confidence": 0.0, "n_bands": 1,
                   "error": f"bad JSON: {exc}"})
            continue

        image_path = req.get("image_path", "")
        result = _recognize(model, vis_processor, device, image_path)
        _emit(result)


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
