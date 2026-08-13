````markdown
# Table 2: Multi-View Configuration Analysis

This folder contains the **Resident → Attending** agentic workflow used for the UCSF-GBM multi-view analysis reported in Table 2.

- `agentic_pipeline.py` — runs the Resident → Attending workflow.
- `evaluate_multiview.py` — evaluates the generated outputs.
- `run_table2.sh` — launcher for the Table 2 experiment.

## Run

Set your API key first:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
````

Then run:

```bash
bash evaluation/table2/run_table2.sh \
    benchmark/ucsf_gbm_test.jsonl \
    /path/to/UCSF-GBM \
    results/table2/gpt4o.jsonl \
    gpt-4o
```

Change:

```text
benchmark/ucsf_gbm_test.jsonl   -> path to the benchmark file
/path/to/UCSF-GBM              -> path to the preprocessed MRI images
results/table2/gpt4o.jsonl     -> output file path
gpt-4o                         -> model name
```

To run directly with Python:

```bash
python evaluation/table2/agentic_pipeline.py \
    --samples benchmark/ucsf_gbm_test.jsonl \
    --root /path/to/UCSF-GBM \
    --out results/table2/gpt4o.jsonl \
    --model gpt-4o \
    --dataset UCSF-GBM
```

## Evaluate

```bash
python evaluation/table2/evaluate_multiview.py \
    --gt benchmark/ucsf_gbm_test.jsonl \
    --outputs results/table2/gpt4o.jsonl \
    --names GPT-4o \
    --out-json results/table2/table2_results.json
```
