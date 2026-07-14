"""Knovel Document Intelligence — Dashboard API.

FastAPI backend that:
- Serves the single-page dashboard (dashboard/static/index.html)
- Accepts PDF uploads and runs the equation-extraction pipeline
- Streams real-time stage progress via Server-Sent Events
- Serves the generated HTML validation reports
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# File lives at others/dashboard/app.py — walk up three levels to reach the
# project root (/app inside Docker, equation_extraction_pipeline/ locally).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
VALIDATION_DIR = DATA_DIR / "validation"
JOBS_DIR = DATA_DIR / "jobs"
STATIC_DIR = Path(__file__).parent / "static"

# Prefer the project venv; fall back to the current interpreter
_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
EQUATIONS_SCRIPT = PROJECT_ROOT / "scripts" / "equations.py"

# Refactored pipeline — invoked as `python -m equation_extraction_pipeline.cli`.
# _USE_NEW_PIPELINE is True when the installed package is importable.
NEW_PIPELINE_SCRIPT = None  # no longer a script file; use -m invocation below
NEW_PIPELINE_DIR = PROJECT_ROOT
try:
    import equation_extraction_pipeline as _eep  # noqa: F401
    _USE_NEW_PIPELINE = True
    del _eep
except ImportError:
    _USE_NEW_PIPELINE = False

for _d in (INPUT_DIR, VALIDATION_DIR, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Knovel Document Intelligence", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="extractor")

# ── Pipeline stage registry ───────────────────────────────────────────────────
PIPELINE_STAGES: list[tuple[str, str]] = [
    ("ingestion", "Document Ingestion"),
    ("classification", "Document Classification"),
    ("preprocessing", "PDF Preprocessing"),
    ("layout", "Layout Analysis"),
    ("reading_order", "Reading Order Detection"),
    ("text_extraction", "Text Extraction"),
    ("equation_extraction", "Equation Extraction"),
    ("validation", "Equation Validation"),
    ("report_generation", "Report Generation"),
]
_STAGE_IDX = {name: i for i, (name, _) in enumerate(PIPELINE_STAGES)}

# ── In-memory job registry ────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_job(job_id: str, filename: str) -> dict:
    return {
        "id": job_id,
        "filename": filename,
        "status": "queued",  # queued | running | completed | failed
        "created_at": _utcnow(),
        "started_at": None,
        "completed_at": None,
        "time_taken_seconds": None,
        "progress": 0,
        "current_stage": "",
        "stages": {name: "pending" for name, _ in PIPELINE_STAGES},
        "logs": [],  # [{ts, level, stage, message}]
        "report_path": None,
        "error": None,
        "metrics": None,
    }


def _job_finish(job: dict) -> None:
    """Stamp completed_at and compute time_taken_seconds from started_at."""
    now = _utcnow()
    job["completed_at"] = now
    if job.get("started_at"):
        try:
            delta = datetime.fromisoformat(now) - datetime.fromisoformat(job["started_at"])
            job["time_taken_seconds"] = round(delta.total_seconds(), 1)
        except Exception:
            pass
    # Persist timing so it survives Docker restarts; also refresh in-memory
    # metrics for any other jobs that share the same report file so their
    # verdict breakdown never diverges from the HTML on disk.
    try:
        stem = Path(job.get("filename", "")).stem
        if stem:
            csv_f = VALIDATION_DIR / stem / "equation_validation.csv"
            mj = VALIDATION_DIR / stem / "equation_validation_metrics.json"
            if mj.exists():
                data = json.loads(mj.read_text(encoding="utf-8"))
                data["started_at"] = job.get("started_at")
                data["completed_at"] = job.get("completed_at")
                data["time_taken_seconds"] = job.get("time_taken_seconds")
                mj.write_text(json.dumps(data, indent=2), encoding="utf-8")
            # Refresh in-memory metrics for sibling jobs pointing to the same
            # report so the dashboard verdict breakdown stays in sync with the
            # HTML report on disk.
            if csv_f.exists():
                fresh_metrics = _extract_metrics(csv_f)
                report_path = str(VALIDATION_DIR / stem / "equation_validation_report.html")
                for other in _jobs.values():
                    if other["id"] != job["id"] and other.get("report_path") == report_path:
                        other["metrics"] = fresh_metrics
    except Exception:
        pass


def _emit(job: dict, stage: str, message: str, level: str = "info") -> None:
    entry = {"ts": _utcnow(), "level": level, "stage": stage, "message": message}
    job["logs"].append(entry)
    logger.info("pipeline_log", job_id=job["id"], stage=stage, level=level, message=message)


def _stage_start(job: dict, name: str, label: str) -> None:
    job["current_stage"] = name
    job["stages"][name] = "running"
    msg = f"[{name}] Starting: {label}"
    _emit(job, name, msg)
    logger.info("stage_start", job_id=job["id"], stage=name, label=label)


def _stage_done(job: dict, name: str, detail: str = "") -> None:
    idx = _STAGE_IDX[name]
    job["stages"][name] = "done"
    job["progress"] = int((idx + 1) / len(PIPELINE_STAGES) * 100)
    msg = f"[{name}] ✓ Done{(' — ' + detail) if detail else ''}"
    _emit(job, name, msg)
    logger.info("stage_done", job_id=job["id"], stage=name, progress=job["progress"], detail=detail)


# ── Metrics helper ────────────────────────────────────────────────────────────
def _extract_metrics(csv_path: Path) -> dict:
    """Derive summary metrics from the equation_validation.csv file."""
    if not csv_path.exists():
        return {}
    try:
        rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        display = [r for r in rows if r.get("is_inline", "").lower() != "true"]
        inline = [r for r in rows if r.get("is_inline", "").lower() == "true"]
        has_latex = [r for r in display if r.get("has_latex", "").lower() == "true"]
        valid_latex = [r for r in has_latex if r.get("latex_valid", "").lower() == "true"]
        high_conf = [r for r in display if float(r.get("recognition_confidence") or 0) >= 0.5]

        # Real quality scores from the quality_score column (written by _write_csv)
        quality_scores: list[float] = []
        needs_review = 0
        quality_dist = {"good": 0, "warn": 0, "fail": 0}
        for r in has_latex:
            qs_str = r.get("quality_score", "")
            if qs_str:
                try:
                    qs = float(qs_str)
                    quality_scores.append(qs)
                    if qs < 0.6:
                        needs_review += 1
                    if qs >= 0.75:
                        quality_dist["good"] += 1
                    elif qs >= 0.5:
                        quality_dist["warn"] += 1
                    else:
                        quality_dist["fail"] += 1
                except ValueError:
                    pass

        avg_quality = round(sum(quality_scores) / len(quality_scores) * 100, 1) if quality_scores else 0.0
        pages_with_equations = len({r.get("page_no", "") for r in display if r.get("page_no")})

        cats: dict[str, int] = {}
        for r in rows:
            k = r.get("category", "unknown")
            cats[k] = cats.get(k, 0) + 1

        flags: dict[str, int] = {}
        for r in rows:
            for f in (r.get("validation_flags") or "").split("|"):
                f = f.strip()
                if f:
                    flags[f] = flags.get(f, 0) + 1

        result: dict = {
            "total_equations": len(rows),
            "display_equations": len(display),
            "inline_equations": len(inline),
            "latex_generation_rate": round(len(has_latex) / max(len(display), 1) * 100, 1),
            "latex_validity_rate": round(len(valid_latex) / max(len(has_latex), 1) * 100, 1),
            "high_confidence_rate": round(len(high_conf) / max(len(display), 1) * 100, 1),
            "latex_quality_score": avg_quality,
            "needs_review_count": needs_review,
            "quality_distribution": quality_dist,
            "pages_with_equations": pages_with_equations,
            "categories": cats,
            "validation_flags": flags,
        }

        # Merge richer metrics written by run_validate (pdf_labeled_count, coverage, llm_judge …)
        metrics_json = csv_path.parent / "equation_validation_metrics.json"
        if metrics_json.exists():
            try:
                extra = json.loads(metrics_json.read_text(encoding="utf-8"))
                result["pdf_labeled_count"] = extra.get("pdf_labeled_count", 0)
                result["pdf_extracted_labeled_count"] = extra.get(
                    "pdf_extracted_labeled_count", result["display_equations"]
                )
                result["pdf_coverage_pct"] = extra.get("pdf_coverage_pct")
                lj = extra.get("llm_judge") or {}
                result["llm_coverage_verdict"] = lj.get("coverage_verdict")
                result["llm_missing_labels"] = lj.get("missing_labels", [])
                result["llm_missing_count"] = lj.get("missing_count", 0)
                result["llm_mean_overall"] = lj.get("mean_overall")
                result["llm_mean_relevance"] = lj.get("mean_relevance")
                result["llm_mean_confidence"] = lj.get("mean_confidence")
                result["llm_accepted"] = lj.get("accepted")
                result["llm_reviewed"] = lj.get("reviewed")
                result["llm_rejected"] = lj.get("rejected")
                result["llm_total"] = lj.get("total")
                # Override rule-based scores with LLM-judge values when available.
                # run_validate() replaces latex_quality_score_pct and quality_distribution
                # with LLM verdicts after the judge runs; the HTML report already reflects
                # these overrides, so the dashboard KPI cards must use the same source.
                if extra.get("latex_quality_score_pct") is not None:
                    result["latex_quality_score"] = extra["latex_quality_score_pct"]
                if extra.get("quality_distribution"):
                    result["quality_distribution"] = extra["quality_distribution"]
                if extra.get("needs_review_count") is not None:
                    result["needs_review_count"] = extra["needs_review_count"]
            except Exception:
                pass

        return result
    except Exception as exc:
        logger.warning("metrics_csv_error", error=str(exc))
        return {}


# ── Sidecar lookup ────────────────────────────────────────────────────────────
def _is_flat_sidecar(path: Path) -> bool:
    """Return True when the sidecar uses the flat top-level format expected by run_validate().

    The flat format has a top-level ``"equations"`` list.
    The old nested format wraps everything inside ``"context"`` — run_validate()
    reads ``data.get("equations")`` and would silently get an empty list from
    a nested sidecar, producing a 0-equation report.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data.get("equations"), list)
    except Exception:
        return False


def _find_sidecar(stem: str) -> Path | None:
    """Return the equation_extraction.json sidecar path (flat format only).

    scripts/equations.py reads  data.get("equations")  — it expects a FLAT
    top-level format. The main pipeline writes this to:
        data/output/{stem}/equation_extraction.json          ← preferred (flat)
    The input-dir sidecar uses the nested "context" envelope:
        data/input/{stem}.equation_extraction.json           ← may be nested

    Nested-format files are rejected so the dashboard runs a fresh extraction
    instead of producing a 0-equation validation report.
    """
    candidates = [
        DATA_DIR / "output" / stem / "equation_extraction.json",  # flat format (preferred)
        INPUT_DIR / f"{stem}.equation_extraction.json",  # may be nested — last resort
    ]
    for p in candidates:
        if p.exists() and _is_flat_sidecar(p):
            logger.info("sidecar_found", path=str(p.relative_to(PROJECT_ROOT)))
            return p
        if p.exists():
            logger.info(
                "sidecar_skipped_nested_format",
                path=str(p.relative_to(PROJECT_ROOT)),
                reason="nested context envelope — run_validate expects flat format",
            )
    return None


# ── New-pipeline output converters ────────────────────────────────────────────

import base64 as _base64
import html as _html_mod


def _crop_b64(crop_path: str | None, output_dir: Path) -> str:
    """Return a data-URI for the crop PNG, or empty string if unavailable."""
    if not crop_path:
        return ""
    try:
        abs_path = output_dir / crop_path
        if abs_path.exists():
            return "data:image/png;base64," + _base64.b64encode(abs_path.read_bytes()).decode()
    except Exception:
        pass
    return ""


def _build_report_html(
    stem: str,
    equations: list,
    summary: dict,
    metrics: dict,
    output_dir: Path | None = None,
) -> str:
    """Generate a rich HTML validation report matching the old pipeline format."""
    from datetime import datetime as _dt

    total = len(equations)
    success = summary.get("success", 0)
    uncertain = summary.get("uncertain", 0)
    rejected = summary.get("rejected", 0)

    has_latex_list = [eq for eq in equations if (eq.get("final", {}).get("latex") or eq.get("ocr", {}).get("latex", "")).strip()]
    latex_gen_pct = round(len(has_latex_list) / max(total, 1) * 100, 1)
    latex_valid_count = sum(
        1 for eq in has_latex_list
        if (lat := (eq.get("final", {}).get("latex") or eq.get("ocr", {}).get("latex", "")))
        and lat.count("{") == lat.count("}")
    )
    latex_valid_pct = round(latex_valid_count / max(len(has_latex_list), 1) * 100, 1)
    high_conf = sum(1 for eq in equations if float(eq.get("final", {}).get("overall_confidence", 0)) >= 0.65)
    high_conf_pct = round(high_conf / max(total, 1) * 100, 1)
    avg_q = metrics.get("latex_quality_score_pct", 0.0)

    cats: dict[str, int] = {}
    for eq in equations:
        k = eq.get("category", "") or ""
        cats[k] = cats.get(k, 0) + 1

    conf_buckets = {"0.0 (passthrough)": 0, "0.0–0.5": 0, "0.5–0.8": 0, "0.8–1.0": 0}
    for eq in equations:
        c = float(eq.get("ocr", {}).get("confidence", 0))
        if c == 0.0:
            conf_buckets["0.0 (passthrough)"] += 1
        elif c < 0.5:
            conf_buckets["0.0–0.5"] += 1
        elif c < 0.8:
            conf_buckets["0.5–0.8"] += 1
        else:
            conf_buckets["0.8–1.0"] += 1

    flags_count: dict[str, int] = {}
    for eq in equations:
        for f in (eq.get("validation_flags") or []):
            flags_count[f] = flags_count.get(f, 0) + 1

    cat_rows = "".join(f"<tr><td>{k or '—'}</td><td>{v}</td></tr>" for k, v in sorted(cats.items(), key=lambda x: -x[1]))
    conf_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in conf_buckets.items())
    flag_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(flags_count.items(), key=lambda x: -x[1])) or "<tr><td colspan='2'>None</td></tr>"

    # LLM Judge summary section
    lj = metrics.get("llm_judge") or {}
    lj_total = lj.get("total", 0)
    lj_accepted = lj.get("accepted", 0)
    lj_rejected = lj.get("rejected", 0)
    lj_verdict = lj.get("coverage_verdict", "")
    lj_mean_score = lj.get("mean_overall")
    lj_mean_conf = lj.get("mean_confidence")
    lj_missing = lj.get("missing_labels") or []
    lj_missing_count = lj.get("missing_count", 0)
    verdict_color = {"complete": "#198754", "partial": "#fd7e14", "incomplete": "#dc3545", "none": "#6c757d"}.get(lj_verdict, "#6c757d")
    verdict_label = {"complete": "✓ COMPLETE", "partial": "⚠ PARTIAL", "incomplete": "✗ INCOMPLETE", "none": "— N/A"}.get(lj_verdict, lj_verdict.upper() if lj_verdict else "—")
    missing_html = (
        "<br><small style='color:#dc3545'>Missing: " + ", ".join(_html_mod.escape(str(x)) for x in lj_missing[:20])
        + ("…" if len(lj_missing) > 20 else "") + "</small>"
    ) if lj_missing else ""
    judge_summary_html = f"""
<h2>LLM Judge Summary</h2>
<div class="summary-grid" style="grid-template-columns:repeat(4,1fr)">
  <div class="metric-card">
    <h3>Coverage Verdict</h3>
    <div class="val" style="font-size:18px;color:{verdict_color}">{verdict_label}</div>
    <div class="sub">{lj_missing_count} missing label(s){missing_html}</div>
  </div>
  <div class="metric-card">
    <h3>Judge Accepted</h3>
    <div class="val" style="color:#198754">{lj_accepted}</div>
    <div class="sub">of {lj_total} total equations</div>
  </div>
  <div class="metric-card">
    <h3>Judge Rejected</h3>
    <div class="val" style="color:#dc3545">{lj_rejected}</div>
    <div class="sub">of {lj_total} total equations</div>
  </div>
  <div class="metric-card">
    <h3>Mean Judge Score</h3>
    <div class="val">{f"{lj_mean_score:.3f}" if lj_mean_score is not None else "—"}</div>
    <div class="sub">confidence: {f"{lj_mean_conf:.3f}" if lj_mean_conf is not None else "—"}</div>
  </div>
</div>"""

    _BAD_FLAGS = {"low_confidence_recognition", "duplicate", "MULTIPLE_EQUATIONS_IN_CROP"}
    eq_rows = ""
    for eq in equations:
        ocr = eq.get("ocr", {})
        final = eq.get("final", {})
        judge = eq.get("judge") or {}
        latex = final.get("latex") or ocr.get("latex", "")
        conf = round(float(final.get("overall_confidence", ocr.get("confidence", 0.0))), 3)
        status = final.get("status", "")
        judge_accepted = judge.get("accepted")
        judge_score = judge.get("score")
        judge_reason = judge.get("reason", "")
        provider = ocr.get("provider", "—")
        category = eq.get("category", "") or ""
        flags = eq.get("validation_flags") or []
        eq_id = eq.get("equation_id", "")

        status_cls = {"SUCCESS": "pass", "UNCERTAIN": "warn", "REJECTED": "fail"}.get(status, "")
        if judge_accepted is True:
            judge_html = "<span class='pass'>✓ accepted</span>"
            if judge_score is not None:
                judge_html += f"<br><small>score {judge_score:.2f}</small>"
        elif judge_accepted is False:
            judge_html = "<span class='fail'>✗ rejected</span>"
            if judge_score is not None:
                judge_html += f"<br><small>score {judge_score:.2f}</small>"
            if judge_reason:
                judge_html += f"<br><small style='color:#888'>{_html_mod.escape(judge_reason[:80])}</small>"
        else:
            judge_html = "—"

        latex_esc = _html_mod.escape(latex or "")
        mathjax_html = f"<div class='mathjax-render'>\\({_html_mod.escape(latex)}\\)</div>" if latex else "<div class='mathjax-render' style='color:#aaa;font-size:11px'>no LaTeX</div>"
        flag_html = " ".join(
            "<span class='flag" + (" flag-bad" if f in _BAD_FLAGS else "") + f"'>{_html_mod.escape(f)}</span>"
            for f in flags
        ) or "—"

        img_src = _crop_b64(eq.get("crop", {}).get("path"), output_dir) if output_dir else ""
        img_html = f"<img class='crop' src='{img_src}' alt='crop'>" if img_src else "<span style='color:#aaa;font-size:11px'>no crop</span>"

        eq_rows += (
            f"<tr class='eq-row'>"
            f"<td style='font-size:11px;color:#666'>{_html_mod.escape(eq_id)}</td>"
            f"<td>{eq.get('page_number', '')}</td>"
            f"<td style='font-size:11px'>{_html_mod.escape(category)}</td>"
            f"<td><span class='{status_cls}'>{status or '—'}</span></td>"
            f"<td style='font-size:11px'>{_html_mod.escape(provider)}</td>"
            f"<td>{img_html}</td>"
            f"<td>{mathjax_html}</td>"
            f"<td><code>{latex_esc}</code></td>"
            f"<td>{flag_html}</td>"
            f"<td>{judge_html}</td>"
            f"</tr>\n"
        )

    generated = _dt.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Equation Extraction Validation Report</title>
<script>
  window.MathJax = {{
    tex: {{ inlineMath: [['\\\\(','\\\\)'], ['$','$']], displayMath: [['$$','$$']] }},
    options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
  body {{ font-family: sans-serif; margin: 20px; background: #f8f9fa; color: #212529; }}
  h1 {{ color: #343a40; }}
  h2 {{ color: #495057; border-bottom: 2px solid #dee2e6; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; background: white; }}
  th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; vertical-align: top; }}
  th {{ background: #e9ecef; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .pass {{ color: #198754; font-weight: bold; }}
  .fail {{ color: #dc3545; font-weight: bold; }}
  .warn {{ color: #fd7e14; font-weight: bold; }}
  .eq-row td {{ padding: 6px 8px; }}
  img.crop {{ max-width: 340px; max-height: 120px; border: 1px solid #ccc; }}
  .mathjax-render {{ min-height: 40px; padding: 6px; background: white; border: 1px solid #ccc;
                      border-radius: 3px; font-size: 13px; overflow-x: auto; max-width: 340px; }}
  code {{ font-size: 11px; background: #f1f3f5; padding: 2px 4px; border-radius: 3px;
           word-break: break-all; display: block; max-width: 320px; }}
  .flag {{ display: inline-block; background: #fff3cd; border: 1px solid #ffc107;
            border-radius: 3px; padding: 1px 5px; font-size: 11px; margin: 1px; }}
  .flag-bad {{ background: #f8d7da; border-color: #dc3545; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px; }}
  .metric-card {{ background: white; border: 1px solid #dee2e6; border-radius: 6px; padding: 14px 18px; }}
  .metric-card h3 {{ margin: 0 0 4px; font-size: 13px; color: #6c757d; }}
  .metric-card .val {{ font-size: 26px; font-weight: bold; color: #0d6efd; }}
  .metric-card .sub {{ font-size: 12px; color: #6c757d; margin-top: 2px; }}
</style>
</head>
<body>
<h1>Equation Extraction Validation Report</h1>
<p><strong>PDF:</strong> {stem}.pdf &nbsp;|&nbsp; <strong>Generated:</strong> {generated}</p>

<div class="summary-grid">
<div class="metric-card"><h3>Extracted Equations</h3><div class="val">{total}</div><div class="sub">display: {total}, inline: 0</div></div>
<div class="metric-card"><h3>Success / Uncertain / Rejected</h3><div class="val"><span style="color:#198754">{success}</span> / <span style="color:#fd7e14">{uncertain}</span> / <span style="color:#dc3545">{rejected}</span></div><div class="sub">final status breakdown</div></div>
<div class="metric-card"><h3>LaTeX Generation</h3><div class="val">{latex_gen_pct}%</div><div class="sub">{len(has_latex_list)} / {total} have LaTeX</div></div>
<div class="metric-card"><h3>LaTeX Validity</h3><div class="val">{latex_valid_pct}%</div><div class="sub">{latex_valid_count} balanced braces</div></div>
<div class="metric-card"><h3>High Confidence</h3><div class="val">{high_conf_pct}%</div><div class="sub">{high_conf} / {total} ≥ 0.65</div></div>
<div class="metric-card"><h3>Avg Quality Score</h3><div class="val">{avg_q}%</div><div class="sub">mean overall_confidence</div></div>
</div>

<h2>Category Distribution</h2>
<table><tr><th>Category</th><th>Count</th></tr>{cat_rows}</table>

<h2>Confidence Distribution</h2>
<table><tr><th>Bucket</th><th>Count</th></tr>{conf_rows}</table>

<h2>Validation Flags</h2>
<table><tr><th>Flag</th><th>Count</th></tr>{flag_rows}</table>

{judge_summary_html}

<h2>Display Equations — Visual Comparison (showing {total} of {total})</h2>
<table>
<tr>
  <th style="width:80px">ID</th>
  <th style="width:50px">Page</th>
  <th style="width:130px">Category</th>
  <th style="width:80px">Status</th>
  <th style="width:80px">Provider</th>
  <th style="width:320px">PDF Crop</th>
  <th style="width:320px">Extracted LaTeX (rendered)</th>
  <th style="width:200px">LaTeX Source</th>
  <th>Flags</th>
  <th style="width:130px">Judge</th>
</tr>
{eq_rows}
</table>
</body>
</html>"""


def _build_validation_from_document_json(
    stem: str, doc_json_path: Path, validation_dir: Path
) -> Path:
    """Convert a document.json produced by the new pipeline into the dashboard's validation format.

    Writes three files to ``validation_dir/{stem}/``:
      - equation_validation.csv          — row-per-equation (read by _extract_metrics)
      - equation_validation_metrics.json — aggregated KPIs
      - equation_validation_report.html  — report iframe shown in the dashboard
    Returns the path to the HTML report.
    """
    data = json.loads(doc_json_path.read_text(encoding="utf-8"))
    doc = data.get("document", {})
    equations: list[dict] = doc.get("equations", [])
    summary: dict = doc.get("summary", {})

    val_dir = validation_dir / stem
    val_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = val_dir / "equation_validation.csv"
    fieldnames = [
        "equation_id", "page_no", "label", "is_inline", "category",
        "has_latex", "latex", "latex_valid", "recognition_confidence",
        "quality_score", "status", "validation_flags",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for eq in equations:
            ocr = eq.get("ocr", {})
            final = eq.get("final", {})
            latex = final.get("latex") or ocr.get("latex", "")
            conf = final.get("overall_confidence", ocr.get("confidence", 0.0))
            has_latex = bool(latex and latex.strip())
            latex_valid = has_latex and latex.count("{") == latex.count("}")
            flags = "|".join(eq.get("validation_flags") or [])
            writer.writerow({
                "equation_id": eq.get("equation_id", ""),
                "page_no": eq.get("page_number", ""),
                "label": eq.get("equation_number") or eq.get("label") or "",
                "is_inline": "false",
                "category": eq.get("category", "mathematical_equation"),
                "has_latex": str(has_latex).lower(),
                "latex": latex,
                "latex_valid": str(latex_valid).lower(),
                "recognition_confidence": round(float(ocr.get("confidence", 0.0)), 4),
                "quality_score": round(float(conf), 4),
                "status": final.get("status", ""),
                "validation_flags": flags,
            })

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    total = len(equations)
    quality_scores = [float(eq.get("final", {}).get("overall_confidence", 0.0)) for eq in equations]
    avg_quality_pct = round(sum(quality_scores) / max(len(quality_scores), 1) * 100, 1)

    quality_dist = {"good": 0, "warn": 0, "fail": 0}
    needs_review = 0
    for qs in quality_scores:
        if qs < 0.6:
            needs_review += 1
        if qs >= 0.75:
            quality_dist["good"] += 1
        elif qs >= 0.5:
            quality_dist["warn"] += 1
        else:
            quality_dist["fail"] += 1

    # Coverage must be based on labels independently scanned from the PDF.  Using
    # ``total`` for both numerator and denominator made every non-empty run 100%,
    # even when sub-equations had been missed.
    pdf_path = INPUT_DIR / f"{stem}.pdf"
    try:
        from equation_extraction_pipeline.detection.equation_label_detector import scan_equation_labels

        pdf_labels = scan_equation_labels(pdf_path) if pdf_path.exists() else []
    except Exception as exc:
        logger.warning("pdf_label_scan_failed", pdf=str(pdf_path), error=str(exc))
        pdf_labels = []
    extracted_labels = [
        str(eq.get("equation_number") or eq.get("label") or "").strip()
        for eq in equations
        if eq.get("equation_number") or eq.get("label")
    ]
    # Match exact modern labels first.  Legacy outputs collapsed ``3.9.1(a)``
    # and ``3.9.1(b)`` to ``3.9.1``; count that as one extracted member of the
    # group without allowing it to satisfy both expected labels.
    remaining = list(pdf_labels)
    matched_labels: list[str] = []
    for extracted in extracted_labels:
        if extracted in remaining:
            remaining.remove(extracted)
            matched_labels.append(extracted)
            continue
        collapsed_match = next(
            (
                expected for expected in remaining
                if expected.rsplit("(", 1)[0] == extracted
                and expected.endswith(")")
            ),
            None,
        )
        if collapsed_match is not None:
            remaining.remove(collapsed_match)
            matched_labels.append(collapsed_match)
    missing_labels = remaining
    coverage_pct = (
        round(len(matched_labels) / len(pdf_labels) * 100, 1)
        if pdf_labels else None
    )
    coverage_verdict = (
        "unknown" if coverage_pct is None
        else "complete" if coverage_pct >= 95
        else "partial" if coverage_pct >= 70
        else "incomplete"
    )

    # The standalone judge is binary and emits scores on a 0–1 scale.  Verdict
    # buckets must be mutually exclusive, while the dashboard score is 0–10.
    judged = [eq.get("judge") or {} for eq in equations if eq.get("judge")]
    accepted_count = sum(1 for judge in judged if judge.get("accepted") is True)
    rejected_count = sum(1 for judge in judged if judge.get("accepted") is False)
    reviewed_count = len(judged) - accepted_count - rejected_count
    judge_scores = [
        max(0.0, min(1.0, float(judge["score"])))
        for judge in judged if judge.get("score") is not None
    ]
    mean_judge_conf = (
        round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None
    )
    mean_judge_10 = (
        round(mean_judge_conf * 10, 2) if mean_judge_conf is not None else None
    )

    metrics: dict = {
        "total_equations": total,
        "pdf_labeled_count": len(pdf_labels),
        "pdf_extracted_labeled_count": len(matched_labels),
        "pdf_coverage_pct": coverage_pct,
        "latex_quality_score_pct": avg_quality_pct,
        "quality_distribution": quality_dist,
        "needs_review_count": needs_review,
        "llm_judge": {
            "total": len(judged),
            "accepted": accepted_count,
            "rejected": rejected_count,
            "reviewed": reviewed_count,
            "coverage_verdict": coverage_verdict,
            "missing_labels": missing_labels,
            "missing_count": len(missing_labels),
            "mean_overall": mean_judge_10,
            "mean_relevance": None,
            "mean_confidence": mean_judge_conf,
        },
    }
    mj_path = val_dir / "equation_validation_metrics.json"
    mj_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ── HTML report — use doc_json_path.parent as output_dir for crop images ─
    output_dir = doc_json_path.parent  # data/output/{stem}/
    html = _build_report_html(stem, equations, summary, metrics, output_dir=output_dir)
    html_path = val_dir / "equation_validation_report.html"
    html_path.write_text(html, encoding="utf-8")

    return html_path


# ── Full extraction flow (no pre-existing sidecar) ────────────────────────────
def _run_full_extraction_flow(job: dict, stem: str, pdf_path: Path) -> None:
    """Run equations.py --mode auto --validate for a PDF with no pre-existing sidecar.

    Streams live output from the script into the equation_extraction log stage.
    Lines prefixed with ``[stage:NAME]`` are used to update the stage progress
    in real time so the frontend can show per-stage status.
    """
    _emit(job, "ingestion", "No cached extraction found — running full pipeline.")
    logger.info("full_extraction_start", job_id=job["id"], pdf=pdf_path.name,
                new_pipeline=_USE_NEW_PIPELINE)

    if _USE_NEW_PIPELINE:
        _run_new_pipeline_flow(job, stem, pdf_path)
    else:
        _run_legacy_pipeline_flow(job, stem, pdf_path)


def _run_new_pipeline_flow(job: dict, stem: str, pdf_path: Path) -> None:
    """Run equation-extraction-pipeline/main_pipeline.py and convert output for the dashboard."""
    # Mark ingestion through text_extraction as done (handled internally by new pipeline)
    for name, label in PIPELINE_STAGES[:6]:
        _stage_start(job, name, label)
        _stage_done(job, name, "handled by new pipeline")

    _stage_start(job, "equation_extraction", "Equation Extraction")
    _emit(job, "equation_extraction", f"Launching: equation_extraction_pipeline.cli --pdf {pdf_path.name}")

    output_dir = DATA_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, "-m", "equation_extraction_pipeline.cli",
        "--pdf", str(pdf_path.resolve()),
        "--out", str(output_dir.resolve()),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(NEW_PIPELINE_DIR),
    )

    # Map progress lines (e.g. "[ 45%] layout_detection: …") to dashboard stages
    _STAGE_MAP = {
        "classification": "classification",
        "rendering": "preprocessing",
        "preprocessing": "preprocessing",
        "layout_detection": "layout",
        "equation_extraction": "equation_extraction",
    }
    _promoted: set[str] = set()

    for line in proc.stdout or []:
        line = line.rstrip()
        if not line:
            continue
        _emit(job, "equation_extraction", line)
        # Parse "[  5%] stage_name: …" progress lines
        if line.startswith("["):
            try:
                pct_str = line[1:line.index("%")].strip()
                rest = line[line.index("]") + 1:].strip()
                stage_key = rest.split(":")[0].strip()
                pct = int(pct_str)
                job["progress"] = max(job["progress"], min(pct, 95))
                ds = _STAGE_MAP.get(stage_key)
                if ds and ds not in _promoted:
                    _promoted.add(ds)
            except Exception:
                pass

    proc.wait()

    if proc.returncode != 0:
        job["status"] = "failed"
        job["stages"]["equation_extraction"] = "failed"
        job["error"] = f"main_pipeline.py exited with code {proc.returncode}"
        _emit(job, "equation_extraction", job["error"], level="error")
        _job_finish(job)
        logger.error("new_pipeline_failed", job_id=job["id"], returncode=proc.returncode)
        return

    _stage_done(job, "equation_extraction")
    logger.info("new_pipeline_extraction_done", job_id=job["id"])

    # Convert document.json → validation format expected by the dashboard
    _stage_start(job, "validation", "Equation Validation")
    doc_json_path = output_dir / stem / "document.json"
    if not doc_json_path.exists():
        job["status"] = "failed"
        job["stages"]["validation"] = "failed"
        job["error"] = f"document.json not found at {doc_json_path}"
        _emit(job, "validation", job["error"], level="error")
        _job_finish(job)
        logger.error("document_json_missing", job_id=job["id"], path=str(doc_json_path))
        return
    _stage_done(job, "validation")

    _stage_start(job, "report_generation", "Report Generation")
    try:
        report_html = _build_validation_from_document_json(stem, doc_json_path, VALIDATION_DIR)
    except Exception as exc:
        job["status"] = "failed"
        job["stages"]["report_generation"] = "failed"
        job["error"] = f"Validation conversion failed: {exc}"
        _emit(job, "report_generation", job["error"], level="error")
        _job_finish(job)
        logger.error("validation_conversion_failed", job_id=job["id"], error=str(exc))
        return

    csv_path = VALIDATION_DIR / stem / "equation_validation.csv"
    metrics = _extract_metrics(csv_path)
    total = metrics.get("total_equations", "?")
    _stage_done(job, "report_generation", f"{total} equations in report")
    job["metrics"] = metrics
    job["report_path"] = str(report_html)
    job["status"] = "completed"
    job["progress"] = 100
    _job_finish(job)
    _emit(job, "report_generation", f"Report ready — {total} equations extracted.")
    logger.info("job_completed", job_id=job["id"], total_equations=total, report=str(report_html),
                time_taken_seconds=job.get("time_taken_seconds"))


def _run_legacy_pipeline_flow(job: dict, stem: str, pdf_path: Path) -> None:
    """Run the legacy scripts/equations.py pipeline (fallback when new pipeline absent)."""
    for name, label in PIPELINE_STAGES[:6]:
        _stage_start(job, name, label)
        _stage_done(job, name, "delegated to equations.py subprocess")

    _stage_start(job, "equation_extraction", "Equation Extraction")
    _emit(job, "equation_extraction", f"Launching: equations.py --mode auto --validate --pdf {pdf_path.name}")

    cmd = [
        PYTHON, str(EQUATIONS_SCRIPT),
        "--pdf", str(pdf_path), "--mode", "auto", "--validate",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(PROJECT_ROOT),
    )
    eq_count_line: str = ""
    for line in proc.stdout or []:
        line = line.rstrip()
        if not line:
            continue
        _emit(job, "equation_extraction", line)
        if "[stage:equation_extraction]" in line:
            eq_count_line = line

    proc.wait()

    if proc.returncode != 0:
        job["status"] = "failed"
        job["stages"]["equation_extraction"] = "failed"
        job["error"] = f"equations.py exited with code {proc.returncode}"
        _emit(job, "equation_extraction", job["error"], level="error")
        _job_finish(job)
        logger.error("full_extraction_failed", job_id=job["id"], returncode=proc.returncode)
        return

    _stage_done(job, "equation_extraction", eq_count_line.strip())

    _stage_start(job, "validation", "Equation Validation")
    _stage_done(job, "validation")
    _stage_start(job, "report_generation", "Report Generation")

    sidecar = _find_sidecar(stem)
    report_html = VALIDATION_DIR / stem / "equation_validation_report.html"
    csv_path = VALIDATION_DIR / stem / "equation_validation.csv"

    if not sidecar or not report_html.exists():
        job["status"] = "failed"
        job["stages"]["report_generation"] = "failed"
        job["error"] = (
            f"Extraction completed but output files are missing for '{stem}'. "
            "Check the equation_extraction log for errors."
        )
        _emit(job, "report_generation", job["error"], level="error")
        _job_finish(job)
        logger.error("report_missing", job_id=job["id"], stem=stem, sidecar_found=bool(sidecar))
        return

    metrics = _extract_metrics(csv_path)
    total = metrics.get("total_equations", "?")
    _stage_done(job, "report_generation", f"{total} equations in report")
    job["metrics"] = metrics
    job["report_path"] = str(report_html)
    job["status"] = "completed"
    job["progress"] = 100
    _job_finish(job)
    _emit(job, "report_generation", f"Report ready — {total} equations extracted.")
    logger.info("job_completed", job_id=job["id"], total_equations=total, report=str(report_html),
                time_taken_seconds=job.get("time_taken_seconds"))


# ── Background extraction worker (runs in thread-pool) ────────────────────────
def _run_extraction(job_id: str, pdf_path: Path) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["started_at"] = _utcnow()
    stem = pdf_path.stem

    # ── Locate sidecar (check both known locations) ───────────────────────────
    sidecar = _find_sidecar(stem)

    if not sidecar:
        # No pre-existing sidecar → run full extraction + Ollama enrichment
        _run_full_extraction_flow(job, stem, pdf_path)
        return

    sidecar_rel = sidecar.relative_to(PROJECT_ROOT)
    for name, _ in PIPELINE_STAGES[:7]:
        job["stages"][name] = "done"
    job["progress"] = 75
    _emit(job, "equation_extraction", f"[cache] Using existing extraction: {sidecar_rel}")
    logger.info("sidecar_cache_hit", job_id=job_id, sidecar=str(sidecar_rel))

    _stage_start(job, "validation", "Equation Validation")
    _stage_done(job, "validation")
    _stage_start(job, "report_generation", "Report Generation")
    _emit(job, "report_generation", f"Generating validation report from {sidecar_rel} …")

    report_html = VALIDATION_DIR / stem / "equation_validation_report.html"

    cmd = [
        PYTHON,
        str(EQUATIONS_SCRIPT),
        "--pdf",
        str(pdf_path),
        "--validate-only",
        "--output-dir",
        str(VALIDATION_DIR),
        "--extraction-json",
        str(sidecar),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    for line in proc.stdout or []:
        line = line.rstrip()
        if line:
            _emit(job, "report_generation", line)

    proc.wait()
    if proc.returncode != 0:
        job["status"] = "failed"
        job["stages"]["report_generation"] = "failed"
        job["error"] = f"equations.py exited with code {proc.returncode}"
        _emit(job, "report_generation", job["error"], level="error")
        _job_finish(job)
        logger.error("validate_only_failed", job_id=job_id, returncode=proc.returncode)
        return

    csv_path = VALIDATION_DIR / stem / "equation_validation.csv"
    metrics = _extract_metrics(csv_path)
    total = metrics.get("total_equations", "?")
    _stage_done(job, "report_generation", f"{total} equations")
    job["metrics"] = metrics
    job["report_path"] = str(report_html)
    job["status"] = "completed"
    job["progress"] = 100
    _job_finish(job)
    _emit(job, "report_generation", f"Report ready — {total} equations.")
    logger.info("job_completed", job_id=job_id, total_equations=total, report=str(report_html),
                time_taken_seconds=job.get("time_taken_seconds"))


# ── Pre-load existing results on startup ──────────────────────────────────────
def _load_existing_reports() -> None:
    """Register any pre-existing validation reports as completed jobs.

    Checks two sources:
    1. data/validation/{stem}/equation_validation_report.html  — old pipeline output
    2. data/output/{stem}/document.json                        — new pipeline output
       (converted on-the-fly to validation format if no HTML report exists yet)
    """
    # ── Source 1: existing validation HTML reports ────────────────────────────
    if VALIDATION_DIR.exists():
        for stem_dir in sorted(VALIDATION_DIR.iterdir()):
            if not stem_dir.is_dir():
                continue
            report = stem_dir / "equation_validation_report.html"
            if not report.exists():
                continue
            safe_id = hashlib.sha1(stem_dir.name.encode()).hexdigest()[:12]
            if safe_id in _jobs:
                continue
            csv_f = stem_dir / "equation_validation.csv"
            metrics_f = stem_dir / "equation_validation_metrics.json"
            created = datetime.fromtimestamp(report.stat().st_mtime, tz=timezone.utc).isoformat()

            timing: dict = {}
            if metrics_f.exists():
                try:
                    _m = json.loads(metrics_f.read_text(encoding="utf-8"))
                    timing = {k: _m.get(k) for k in ("started_at", "completed_at", "time_taken_seconds")}
                except Exception:
                    pass

            _jobs[safe_id] = {
                "id": safe_id,
                "filename": stem_dir.name + ".pdf",
                "status": "completed",
                "created_at": timing.get("started_at") or created,
                "started_at": timing.get("started_at"),
                "completed_at": timing.get("completed_at"),
                "time_taken_seconds": timing.get("time_taken_seconds"),
                "progress": 100,
                "current_stage": "report_generation",
                "stages": {name: "done" for name, _ in PIPELINE_STAGES},
                "logs": [{"ts": created, "level": "info", "stage": "report_generation",
                          "message": "Pre-existing results loaded from disk."}],
                "report_path": str(report),
                "error": None,
                "metrics": _extract_metrics(csv_f) if csv_f.exists() else {},
            }
            logger.info("pre_loaded_report", job_id=safe_id, doc=stem_dir.name)

    # ── Source 2: new-pipeline document.json outputs not yet converted ────────
    output_root = DATA_DIR / "output"
    if output_root.exists():
        for stem_dir in sorted(output_root.iterdir()):
            if not stem_dir.is_dir():
                continue
            doc_json = stem_dir / "document.json"
            if not doc_json.exists():
                continue
            stem = stem_dir.name
            safe_id = hashlib.sha1(("new:" + stem).encode()).hexdigest()[:12]
            if safe_id in _jobs:
                continue
            # Skip if a validation HTML already exists for this stem (loaded above)
            if (VALIDATION_DIR / stem / "equation_validation_report.html").exists():
                continue
            # Convert on-the-fly
            try:
                report = _build_validation_from_document_json(stem, doc_json, VALIDATION_DIR)
            except Exception as exc:
                logger.warning("pre_load_convert_failed", stem=stem, error=str(exc))
                continue
            csv_f = VALIDATION_DIR / stem / "equation_validation.csv"
            created = datetime.fromtimestamp(doc_json.stat().st_mtime, tz=timezone.utc).isoformat()
            _jobs[safe_id] = {
                "id": safe_id,
                "filename": stem + ".pdf",
                "status": "completed",
                "created_at": created,
                "started_at": created,
                "completed_at": created,
                "time_taken_seconds": None,
                "progress": 100,
                "current_stage": "report_generation",
                "stages": {name: "done" for name, _ in PIPELINE_STAGES},
                "logs": [{"ts": created, "level": "info", "stage": "report_generation",
                          "message": "Loaded from new-pipeline document.json."}],
                "report_path": str(report),
                "error": None,
                "metrics": _extract_metrics(csv_f) if csv_f.exists() else {},
            }
            logger.info("pre_loaded_document_json", job_id=safe_id, doc=stem)


@app.on_event("startup")
async def _startup() -> None:
    logger.info(
        "dashboard_startup",
        python=PYTHON,
        data_dir=str(DATA_DIR),
        validation_dir=str(VALIDATION_DIR),
        equations_script=str(EQUATIONS_SCRIPT),
    )
    _load_existing_reports()
    logger.info("pre_loaded_reports", count=len(_jobs))


# ── Static files & root ───────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "jobs": len(_jobs)}


# ── Jobs ──────────────────────────────────────────────────────────────────────
@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    # Validation artifacts may be regenerated after a pipeline fix while the
    # dashboard process stays alive. Refresh completed jobs on each UI poll so
    # stale in-memory KPIs do not survive until a backend restart.
    for job in _jobs.values():
        report_path = job.get("report_path")
        if job.get("status") != "completed" or not report_path:
            continue
        csv_path = Path(report_path).parent / "equation_validation.csv"
        if csv_path.exists():
            job["metrics"] = _extract_metrics(csv_path)
    return list(reversed(list(_jobs.values())))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    report_path = job.get("report_path")
    if job.get("status") == "completed" and report_path:
        csv_path = Path(report_path).parent / "equation_validation.csv"
        if csv_path.exists():
            job["metrics"] = _extract_metrics(csv_path)
    return job


@app.post("/api/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(...),
    force: bool = False,
) -> dict:
    """Accept a PDF upload and start an extraction job.

    ``force=true`` deletes any existing sidecar and validation output for this
    document before starting, ensuring a fresh extraction rather than reusing
    cached results.  Use this when re-uploading after a code fix or config change.
    """
    fname = (file.filename or "").strip()
    if not fname.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    fname = Path(fname).name  # strip any path traversal
    stem = Path(fname).stem
    job_id = str(uuid.uuid4())[:8]
    pdf_path = INPUT_DIR / fname

    _MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
    content = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, "PDF exceeds the 100 MB upload limit")
    pdf_path.write_bytes(content)

    if force:
        # Remove cached extraction and validation so the pipeline re-runs from scratch
        import shutil as _shutil
        for stale in [
            DATA_DIR / "output" / stem,
            VALIDATION_DIR / stem,
        ]:
            if stale.exists():
                _shutil.rmtree(stale)
                logger.info("force_cleared_cache", job_id=job_id, path=str(stale.relative_to(PROJECT_ROOT)))

    job = _new_job(job_id, fname)
    job["force"] = force
    _jobs[job_id] = job

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_extraction, job_id, pdf_path)

    return {"id": job_id, "filename": fname, "force": force}


# ── SSE event stream ──────────────────────────────────────────────────────────
@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    async def _stream() -> AsyncGenerator[str, None]:
        log_pos = 0
        for _ in range(720):  # max 6 min at 0.5 s intervals
            job = _jobs[job_id]
            new_logs = job["logs"][log_pos:]
            log_pos = len(job["logs"])

            payload = {
                "status": job["status"],
                "progress": job["progress"],
                "current_stage": job["current_stage"],
                "stages": job["stages"],
                "new_logs": new_logs,
                "metrics": job["metrics"],
                "error": job["error"],
                "report_ready": job["report_path"] is not None,
            }
            yield f"data: {json.dumps(payload)}\n\n"

            if job["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Report serving ────────────────────────────────────────────────────────────
@app.get("/api/jobs/{job_id}/report")
def get_report(job_id: str) -> FileResponse:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.get("report_path"):
        raise HTTPException(404, "Report not yet available")
    path = Path(job["report_path"])
    if not path.exists():
        raise HTTPException(404, "Report file missing from disk")
    return FileResponse(str(path), media_type="text/html", headers={"Cache-Control": "no-store"})


# ── Job rerun ─────────────────────────────────────────────────────────────────
@app.post("/api/jobs/{job_id}/rerun", status_code=202)
async def rerun_job(job_id: str) -> dict:
    """Force-rerun an existing job from scratch, clearing all cached outputs.

    The job keeps its original ID and filename. Its state is reset to queued
    and the extraction worker is re-submitted to the thread pool.
    Raises 409 if the job is currently active (running or queued).
    Raises 404 if the source PDF is no longer on disk.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] in ("running", "queued"):
        raise HTTPException(409, "Job is already active — wait for it to finish before rerunning")

    filename = job["filename"]
    stem = Path(filename).stem
    pdf_path = INPUT_DIR / filename

    if not pdf_path.exists():
        raise HTTPException(
            404,
            f"Source PDF '{filename}' not found in data/input/. "
            "Re-upload the file via New Extraction to run again.",
        )

    # Clear cached outputs so _run_extraction does a full fresh run
    for stale in [DATA_DIR / "output" / stem, VALIDATION_DIR / stem]:
        if stale.exists():
            shutil.rmtree(stale)
            logger.info("rerun_cleared_cache", job_id=job_id, path=str(stale.relative_to(PROJECT_ROOT)))

    # Reset job state in-place (preserves ID, filename, and previous metrics so
    # the dashboard KPI cards don't flash to zero while the new run is in progress;
    # metrics are overwritten once the new extraction completes successfully).
    job.update(
        {
            "status": "queued",
            "created_at": _utcnow(),
            "started_at": None,
            "completed_at": None,
            "time_taken_seconds": None,
            "progress": 0,
            "current_stage": "",
            "stages": {name: "pending" for name, _ in PIPELINE_STAGES},
            "logs": [],
            "report_path": None,
            "error": None,
            "force": True,
        }
    )

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_extraction, job_id, pdf_path)
    logger.info("job_rerun_queued", job_id=job_id, filename=filename)

    return {"id": job_id, "filename": filename, "rerun": True}


# ── Job deletion ──────────────────────────────────────────────────────────────
@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Remove a job from memory and delete all associated files from disk.

    Deleted paths (when present):
      - data/input/{filename}              — the uploaded PDF
      - data/validation/{stem}/            — HTML report, CSV, and any other outputs
      - data/output/{stem}/                — equation_extraction.json sidecar
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    filename = job["filename"]
    stem = Path(filename).stem

    del _jobs[job_id]

    deleted: list[str] = []
    errors: list[str] = []

    def _rm_file(p: Path) -> None:
        if p.exists():
            try:
                p.unlink()
                deleted.append(str(p.relative_to(PROJECT_ROOT)))
            except Exception as exc:
                errors.append(str(exc))

    def _rm_dir(p: Path) -> None:
        if p.exists():
            try:
                shutil.rmtree(p)
                deleted.append(str(p.relative_to(PROJECT_ROOT)))
            except Exception as exc:
                errors.append(str(exc))

    _rm_file(INPUT_DIR / filename)
    _rm_dir(VALIDATION_DIR / stem)
    _rm_dir(DATA_DIR / "output" / stem)

    logger.info("job_deleted", job_id=job_id, filename=filename, deleted=deleted, errors=errors)
    return {"deleted": deleted, "errors": errors}
