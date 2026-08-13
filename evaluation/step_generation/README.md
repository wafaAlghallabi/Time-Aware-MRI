# Step generation

Provider-specific runners generate the reasoning steps and final answers evaluated in Table 1. The shared task wording and target JSON schema are documented in `prompts.py`.

## Runners

- `openai_models.py` — OpenAI models, including GPT-4o, o4-mini, and GPT-5.2 via `--model`.
- `gemini_models.py` — Gemini models via `--model-id`.
- `llama_models.py` — Llama-4 models through Groq via `--model`.
- `internvl_models.py` — local InternVL inference.
- `qwen_local_models.py` — local Hugging Face Qwen inference.
- `qwen_vllm_models.py` — Qwen through an OpenAI-compatible vLLM endpoint. The supplied experiment code used a local vLLM endpoint; this file should not be described as a native DashScope client.
- `prompts.py` — canonical task prompt and output schema for documentation/reuse.

A standalone Table 1 MedGemma runner is pending recovery of the original experiment file.

## Standard output schema

Successful outputs use the following core fields (plus sample/model bookkeeping used by individual runners):

```json
{
  "steps": ["...", "...", "..."],
  "answer": "...",
  "answer_key": "",
  "answer_option": "",
  "valid_json": true
}
```

For multiple-choice samples, `answer_key` and `answer_option` are populated where supported.

## Example: OpenAI

```bash
export OPENAI_API_KEY=...
python evaluation/step_generation/openai_models.py \
  --samples benchmark/test.jsonl \
  --root /path/to/preprocessed/data \
  --out outputs/gpt-4o.jsonl \
  --model gpt-4o
```

## Example: Gemini

```bash
export GEMINI_API_KEY=...
python evaluation/step_generation/gemini_models.py \
  --samples benchmark/test.jsonl \
  --root /path/to/preprocessed/data \
  --out outputs/gemini-2.5-pro.jsonl \
  --model-id gemini-2.5-pro
```

## Example: Qwen via vLLM

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
python evaluation/step_generation/qwen_vllm_models.py \
  --samples benchmark/test.jsonl \
  --root /path/to/preprocessed/data \
  --out outputs/qwen3-vl.jsonl \
  --model qwen3-vl-235b-a22b-thinking
```
