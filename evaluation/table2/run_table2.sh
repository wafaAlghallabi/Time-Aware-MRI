#!/usr/bin/env bash
set -euo pipefail

# Resident-Attending multi-view workflow used for the Table 2 analysis.
# API credentials must be provided through environment variables.
# Example:
#   OPENAI_API_KEY=... bash evaluation/table2/run_table2.sh \
#      benchmark/test.jsonl /path/to/preprocessed/data results/table2/gpt4o.jsonl gpt-4o

SAMPLES=${1:?"Benchmark sample JSON/JSONL is required"}
ROOT=${2:?"Root directory containing locally obtained/preprocessed MRI data is required"}
OUT=${3:?"Output JSONL path is required"}
MODEL=${4:?"Model identifier is required"}

python evaluation/table2/agentic_pipeline.py \
  --samples "$SAMPLES" \
  --root "$ROOT" \
  --out "$OUT" \
  --model "$MODEL" \
  --dataset UCSF-GBM
