# Table 2: multi-view configuration analysis

This folder contains the **Resident → Attending** agentic workflow used for the UCSF-GBM multi-view analysis reported in Table 2.

- `agentic_pipeline.py` — resident spatial extraction followed by attending integration/classification.
- `evaluate_multiview.py` — evaluates agentic outputs.
- `run_table2.sh` — minimal launcher without hard-coded credentials or local paths.

The paper reports this analysis for six representative models: GPT-4o, Gemini-2.5-Flash, Gemini-2.5-Pro, InternVL3.5-Inst, Qwen3-VL-8B-Inst, and MedGemma-4B-IT.

Source MRI images are not included; `--root` must point to locally obtained and preprocessed source data.
