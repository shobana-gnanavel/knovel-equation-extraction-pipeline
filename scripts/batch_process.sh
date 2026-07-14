#!/usr/bin/env bash
# Batch-process all PDFs in data/input/.
# Usage:
#   ./scripts/batch_process.sh [--out /path/to/output]
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m equation_extraction_pipeline.cli --batch "$@"
