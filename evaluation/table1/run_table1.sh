#!/usr/bin/env bash
set -euo pipefail

# Reproduce the metrics reported in Table 1 from a folder of model predictions.
# Usage:
#   bash evaluation/table1/run_table1.sh <ground_truth.jsonl> <predictions_dir> <results_dir>

GT=${1:?"Ground-truth JSONL is required"}
PRED_DIR=${2:?"Directory containing one model prediction JSONL per model is required"}
OUT=${3:-results/table1}

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set for judge-based RS/accuracy and TAC extraction." >&2
  exit 1
fi

mkdir -p "$OUT/judge" "$OUT/tac"

# 1) Reasoning Score (RS) + binary final-answer correctness.
python evaluation/table1/reasoning_judge.py \
  --gt_file "$GT" \
  --model_folder "$PRED_DIR" \
  --output_folder "$OUT/judge"

# 2) Print aggregate RS and final-answer accuracy table.
python evaluation/table1/aggregate_results.py "$OUT/judge" | tee "$OUT/rs_accuracy.txt"

# 3) Temporal metrics for each model prediction file.
for pred in "$PRED_DIR"/*.jsonl; do
  [[ -e "$pred" ]] || continue
  model=$(basename "$pred" .jsonl)
  python evaluation/table1/tac_metrics.py \
    --gt "$GT" \
    --pred "$pred" \
    --out_dir "$OUT/tac/$model"
done

echo "Done. Results written under: $OUT"
