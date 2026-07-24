"""Re-run ONLY the GPT judge over a completed run's saved crops + recognized LaTeX.

Recognition is untouched (it is the expensive Ollama step); this exists so a judge-rubric
fix can be re-applied to finished runs in minutes. Runs INSIDE the container:

    docker cp scripts/rejudge.py equation_extraction_pipeline-pipeline-1:/tmp/
    docker exec equation_extraction_pipeline-pipeline-1 python /tmp/rejudge.py \
        /app/data/output_gen/e2e/39896_02 [...more book dirs]

Writes <book>/document_rejudged.json (original left untouched) and prints per-book summaries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

from equation_extraction_pipeline.config import settings as config
from equation_extraction_pipeline.extraction.gpt_judge import judge_equation


def rejudge_book(book_dir: Path) -> None:
    doc_path = book_dir / "document.json"
    data = json.loads(doc_path.read_text())
    eqs = data["document"]["equations"]
    changed = 0
    for e in eqs:
        crop_rel = (e.get("crop") or {}).get("path")
        latex = (e.get("final") or {}).get("latex") or ""
        if not crop_rel or not latex.strip():
            continue
        crop_path = book_dir / crop_rel
        if not crop_path.exists():
            continue
        img = Image.open(crop_path).convert("RGB")
        verdict = judge_equation(img, latex, "mathematical_equation")
        e["judge"] = verdict.to_dict()
        if not verdict.accepted:
            status = "REJECTED"
        elif (e.get("ocr") or {}).get("confidence", 1.0) < config.RECOGNITION_MIN_CONFIDENCE:
            status = "UNCERTAIN"
        else:
            status = "SUCCESS"
        if e["final"].get("status") != status:
            changed += 1
        e["final"]["status"] = status
    summary = {
        "total_equations": len(eqs),
        "success": sum(1 for e in eqs if e["final"].get("status") == "SUCCESS"),
        "uncertain": sum(1 for e in eqs if e["final"].get("status") == "UNCERTAIN"),
        "rejected": sum(1 for e in eqs if e["final"].get("status") == "REJECTED"),
    }
    data["document"]["summary"] = summary
    (book_dir / "document_rejudged.json").write_text(json.dumps(data, indent=2))
    print(f"{book_dir.name}: {summary} (changed={changed})", flush=True)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        rejudge_book(Path(arg))
