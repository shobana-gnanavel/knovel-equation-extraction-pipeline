"""Optional per-document equation (LaTeX/MathML) dump (feature 008, Logging & Observability).

Best-effort debug artifact gated by ``KNOVEL_EQUATION_DEBUG_DUMP`` / ``KNOVEL_EQUATION_WORKDIR``:
writes the recognized equations in reading order as a Markdown file (number, category, provider, and
LaTeX/structured form) so a human can eyeball fidelity. Never raises into the stage — callers wrap it
defensively.
"""

from __future__ import annotations

from pathlib import Path

from pipeline import config
from pipeline.models import EquationExtractionContext

__all__ = ["resolve_workdir", "write_equation_dump"]


def resolve_workdir(pdf_path: Path) -> Path:
    """Resolve the debug working directory (configured ``KNOVEL_EQUATION_WORKDIR`` or next to the PDF)."""
    if config.KNOVEL_EQUATION_WORKDIR:
        return Path(config.KNOVEL_EQUATION_WORKDIR)
    return pdf_path.parent


def _render(context: EquationExtractionContext) -> str:
    lines: list[str] = []
    for eq in context.equations:
        number = eq.equation_number or "—"
        lines.append(f"## {number}  ·  {eq.category}  ·  {eq.selected_provider}")
        representation = eq.latex or eq.structured_form or eq.plain_text or ""
        if eq.latex:
            lines.append(f"```latex\n{representation}\n```")
        else:
            lines.append(representation)
        if eq.mathml:
            lines.append(f"```xml\n{eq.mathml}\n```")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_equation_dump(context: EquationExtractionContext, pdf_path: Path, workdir: Path) -> Path:
    """Write the reading-order equation dump for ``context`` and return its path."""
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = workdir / f"{pdf_path.stem}.equations.md"
    out_path.write_text(_render(context), encoding="utf-8")
    return out_path
