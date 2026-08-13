"""Shared prompts and output schema used for longitudinal MRI reasoning generation."""

import textwrap

SYSTEM_PROMPT = textwrap.dedent("""
You are a board-certified neuroradiologist.
Your job: given multi-timepoint brain MRI and a comparison question, produce:
(1) succinct, evidence-based reasoning steps and
(2) a final answer about interval change.
Rules:
- Use only the provided images/descriptions and metadata.
- Compare each follow-up to baseline and comment on trend.
- Prefer categorical change terms: increased / decreased / stable / new / resolved / indeterminate.
- If image quality or protocol differences limit certainty, say so.
- Do not give treatment advice.
- Output only the specified JSON—no extra text.
""").strip()

OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 200},
        },
        "answer": {"type": "string", "maxLength": 200},
        "answer_key": {"type": "string", "maxLength": 1},
        "answer_option": {"type": "string", "maxLength": 200},
    },
    "required": ["steps", "answer", "answer_key", "answer_option"],
    "additionalProperties": False,
}
