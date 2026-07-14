#!/usr/bin/env bash
# Run the equation extraction pipeline on a single PDF or batch.
# Usage:
#   ./scripts/run_pipeline.sh --pdf data/input/my_chapter.pdf
#   ./scripts/run_pipeline.sh --batch
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m equation_extraction_pipeline.cli "$@"
