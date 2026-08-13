# Evaluation code

Publication-ready evaluation code for **How Good are Foundation Models in Longitudinal MRI Disease Progression Reasoning?** (MICCAI 2026).

The release follows the experimental workflow used in the paper:

1. `step_generation/` — generate standardized reasoning steps and final answers from longitudinal, multi-view MRI inputs.
2. `table1/` — compute the main benchmark metrics: Final Accuracy, Reasoning Score (RS), TEDS, Trend-F1, Sign Accuracy, Coverage, Chronology, and TAC.
3. `table2/` — run/evaluate the Resident→Attending agentic workflow used for the multi-view configuration analysis.

## Data access

Source MRI images are **not redistributed**. Users should obtain each source cohort from its official provider under the applicable license/data-use agreement and run the preprocessing pipeline locally. Benchmark manifests can then reference those local files.

## Credentials

No credentials or machine-specific absolute paths are stored in this release. Set credentials through environment variables only:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export GROQ_API_KEY=...
export HF_TOKEN=...
# Only for the Qwen OpenAI-compatible vLLM runner, if your server requires them:
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=...
```

## Current model-generation coverage

The cleaned scripts cover the OpenAI, Gemini, Llama-4/Groq, InternVL, and Qwen local/vLLM paths present in the supplied experiment code. A standalone MedGemma Table 1 generation runner is intentionally **not fabricated here**; add the original MedGemma script when recovered.

See each subfolder README for usage.
