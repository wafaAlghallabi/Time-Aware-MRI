# Table 1 evaluation

The main paper table reports **Final Accuracy**, **Reasoning Score (RS)**, **TAC**, **TEDS**, **Trend-F1**, **Sign Accuracy**, **Coverage**, and **Chronology**.

## Files

- `reasoning_judge.py` — LLM-as-judge evaluation. Its `Overall Score` is the RS used by `aggregate_results.py`; it also outputs binary `Final_Answer_Correctness`.
- `aggregate_results.py` — averages RS and final-answer correctness for each model.
- `tac_metrics.py` — TEDS, Trend-F1, Sign Accuracy, Coverage, Chronology, and TAC.
- `extract_mcq_answers.py` — answer-normalization helper used during evaluation preparation.
- `reasoning_similarity.py` — supplementary BLEU/ROUGE/BERTScore reasoning similarity utility from the experiment folder; these metrics are **not** the main Table 1 columns.
- `run_table1.sh` — clean end-to-end launcher for a directory of prediction JSONL files.

The TAC implementation follows the paper:

```text
TAC = 0.5 × TEDS + 0.2 × Trend-F1 + 0.2 × SignAcc + 0.1 × Coverage
```

Chronology is reported separately.

## Example

```bash
export OPENAI_API_KEY=...
bash evaluation/table1/run_table1.sh \
  benchmark/test.jsonl \
  outputs/ \
  results/table1
```
