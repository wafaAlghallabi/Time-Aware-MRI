# Table 1 Evaluation

The main paper table reports **Final Accuracy**, **Reasoning Score (RS)**, **TAC**, **TEDS**, **Trend-F1**, **Sign Accuracy**, **Coverage**, and **Chronology**.

## Files

- `reasoning_judge.py` — computes the LLM-based Reasoning Score (RS) and final-answer correctness.
- `aggregate_results.py` — aggregates RS and Final Accuracy across samples for each model.
- `tac_metrics.py` — computes TEDS, Trend-F1, Sign Accuracy, Coverage, Chronology, and TAC.
- `extract_mcq_answers.py` — normalizes multiple-choice answers before evaluation.
- `reasoning_similarity.py` — optional BLEU/ROUGE/BERTScore reasoning similarity analysis.
- `run_table1.sh` — launcher for the full Table 1 evaluation.

The TAC score is computed as:

```text
TAC = 0.5 × TEDS + 0.2 × Trend-F1 + 0.2 × SignAcc + 0.1 × Coverage
````

Chronology is evaluated separately.

Set your OpenAI API key first:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

Then run:

```bash
bash evaluation/table1/run_table1.sh \
    benchmark/test.jsonl \
    outputs/ \
    results/table1
```

Change:

```text
benchmark/test.jsonl   -> path to the ground-truth benchmark file
outputs/               -> directory containing model prediction JSONL files
results/table1         -> directory where evaluation results will be saved
```

Each model prediction file should be placed inside `outputs/`, for example:

```text
outputs/
├── gpt4o.jsonl
├── internvl.jsonl
├── qwen.jsonl
└── gemini25pro.jsonl
```

```
```
