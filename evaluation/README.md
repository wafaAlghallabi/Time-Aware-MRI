# Evaluation scripts

The release follows the experimental workflow used in the paper:

1. `step_generation/` — generate standardized reasoning steps and final answers from longitudinal, multi-view MRI inputs.
2. `table1/` — compute the main benchmark metrics: Final Accuracy, Reasoning Score (RS), TEDS, Trend-F1, Sign Accuracy, Coverage, Chronology, and TAC.
3. `table2/` — run/evaluate the Resident→Attending agentic workflow used for the multi-view configuration analysis.

## Data access

Source MRI images are **not redistributed**. Users should obtain each source cohort from its official provider under the applicable license/data-use agreement and run the preprocessing pipeline locally. Benchmark manifests can then reference those local files.

## Credentials

Set credentials through environment variables only:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export GROQ_API_KEY=...
export HF_TOKEN=...
# Only for the Qwen OpenAI-compatible vLLM runner, if your server requires them:
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=...
```

See each subfolder README for usage.
