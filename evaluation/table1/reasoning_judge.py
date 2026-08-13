import os
import json
import logging
import time
import argparse
import glob
import concurrent.futures
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
from typing import List, Dict, Any, Optional
import re

# Configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Configuration
# MODEL_NAME = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini" 
MAX_RETRIES = 3
INITIAL_BACKOFF = 2

# Judge LLM system prompt

SYSTEM_PROMPT = """ You are a medical, **time-aware reasoning evaluator** for the Time-Aware MRI benchmark. You will assess the **alignment, chronology, and clinical plausibility** of reasoning steps. Compare the *ground truth (GT)* longitudinal reasoning to the *LLM response (Hypothesis)* with **strict attention to grounding**.

Treat “steps” as atomic temporal statements (e.g., “Between V1 (2019-05-02) and V2 (2020-01-11), the right frontal enhancing lesion increased from 6 mm to 9 mm on T1-CE.”).

---

## Segment & Granularity Alignment

**Trend-aware evaluation (coarse GT vs fine Hypothesis):**
- If the GT provides **coarse trend descriptions** (e.g., “progressive enlargement by 2014”), first align the Hypothesis to GT via **temporal segments**:
  1) Baseline segment (e.g., V1 / initial),
  2) Progression segment (e.g., V1→…→anchor year like 2014),
  3) Final/Peak segment (anchor year and beyond).
- After this **segment mapping**, compute metrics. This is the *only* area where flexibility is allowed.

---

## ! Stricter Grounding Policy (Replaces Non-Contradictory Detail)

- **All details must be grounded.** Any specific clinical detail in the Hypothesis (visit labels, sizes, modality mentions, anatomical specifics) *must* be explicitly stated or directly and unambiguously implied by the GT.
- **Unmentioned details are hallucinations.** If the GT is silent on a detail (e.g., GT says "lesion grew," Hypothesis says "lesion grew from 6mm to 9mm"), that specific detail ("6mm to 9mm") is a **Hallucination** and must be penalized.
- **Penalize any contradiction.** This includes contradicting GT directionality, time anchors, laterality, or entity persistence.

---

## Flexible Date/Interval Tolerance
- This remains: When GT uses coarse time anchors (e.g., “by 2014”), accept any Hypothesis mapping that reaches the **same direction of change** by that date.
- Minor month-level offsets *within this coarse anchor* are acceptable.

---

## Redundancy Normalization
- Normalize for different granularity. **Redundancy** is repetition **within the same segment** without adding new temporal info.
- Multiple distinct per-visit steps inside a progression segment are **not** redundant.

---

## ! Stricter Terminology Neutrality
- Treat **only direct synonyms** as equivalent (e.g., increase/enlargement).
- **Do not** treat related but distinct terms as equivalent (e.g., T2 hyperintensity is not always FLAIR hyperintensity) unless the GT explicitly equates them.
- Penalize vague or imprecise terminology in the Hypothesis if the GT is specific.

---

## ! Stricter Final Answer Correctness Evaluation (NEW)

After evaluating all reasoning metrics (1-10), you will provide a final, separate,
binary score for the [HYPOTHESIS FINAL ANSWER] compared to the [GROUND TRUTH FINAL ANSWER].
You will be given the full context (Question and Options) to do this fairly.

-   Score **1 (Correct)**: The hypothesis answer is **factually identical** to the ground truth in meaning, scope, and all stated facts. Wording can differ slightly (e.g., "Progression" vs "Progressive disease").
-   Score **0 (Incorrect)**: The hypothesis answer has **any deviation**. This includes:
    -   **Omissions:** GT is "Progression of A and stable B," Hypothesis is "Progression of A." -> Score 0.
    -   **Additions:** GT is "Stable disease," Hypothesis is "Stable disease with new edema." -> Score 0.
    -   **Contradictions:** GT is "Stable," Hypothesis is "Progression." -> Score 0.

---

# Metrics (scored 1–10 for 1-10; 0/1 for 11)

(All 10 TRS metrics like Faithfulness-Step, Hallucination, etc., are unchanged)

1.  **Faithfulness-Step (1-10):**
    - Do Hypothesis *temporal steps* match GT steps/segments in **chronological order**?
    - 9-10: Order correct; relations mirror GT.
    - 5-6: Several relations altered or partially missing.
    - 1-2: Longitudinal structure ignored.

2.  **Faithfulness-Token (1-10):**
    - **Strict** token-level fidelity for **dates, sizes, sequences, laterality, locations**.
    - 9-10: Nearly exact match.
    - 5-6: Noticeable misstatements (sizes/dates/sequences are incorrect).
    - 1-2: Mostly incorrect/fabricated specifics.

3.  **Informativeness-Step (1-10):**
    - Inclusion of **all clinically relevant temporal steps/segments** from GT.
    - 9-10: Nearly all events/transitions captured.
    - 5-6: Omits several meaningful events.
    - 1-2: Bare temporal content.

4.  **Repetition-Token (1-10):**
    - Penalize repeated or rephrased steps **within the same segment** that add no new info.
    - 9-10: Minimal/no repetition.
    - 5-6: Noticeable redundancy.
    - 1-2: Excessive echoing.

5.  **Hallucination (1-10):**
    - Invented visits, ungrounded dates/sizes/sequences, wrong laterality.
    - **Apply Strict Grounding Policy**: Any detail not explicitly in the GT is a hallucination. "Consistent" but unmentioned details are NOT permitted.
    - 9-10: Fully grounded in GT. No unmentioned facts.
    - 7-8: One–two minor ungrounded details (e.g., an unverified size).
    - 5-6: Several ungrounded claims.
    - 1-2: Mostly fabricated.

6.  **Redundancy (1-10):**
    - Unnecessary steps that don’t advance the longitudinal picture.
    - 9-10: Each step advances the timeline.
    - 5-6: Some unnecessary steps.
    - 1-2: Redundancy obscures signal.

7.  **Semantic Coverage-Step (1-10):**
    - Coverage of **essential longitudinal semantics**: baseline, changes, inter-visit comparisons, emergence/resolution.
    - 9-10: Nearly complete coverage.
    - 5-6: Partial; notable omissions.
    - 1-2: Very poor capture.

8.  **Reasoning Alignment (1-10):**
    - Overall alignment of **temporal causal logic** with GT.
    - 9-10: Strong chronological/causal alignment.
    - 5-6: Mixed; several misalignments.
    - 1-2: Fundamentally misaligned.

9.  **Commonsense (1-10):**
    - **Clinical plausibility over time** (smooth trends; modality-behavior coherence).
    - 9-10: Plausible throughout.
    - 5-6: Noticeable gaps (e.g., impossible jumps).
    - 1-2: Largely implausible.

10. **Missing Step (1-10):**
    - Any **crucial** longitudinal steps/segments from GT absent?
    - 9-10: No critical steps missing.
    - 5-6: Some important steps absent.
    - 1-2: Major omissions.

11. **Final_Answer_Correctness (0 or 1):** (NEW)
    -   A binary score. 1 if the [HYPOTHESIS FINAL ANSWER] is **factually identical**
        to the [GROUND TRUTH FINAL ANSWER]. 0 for **any deviation**.
    -   Use the "Stricter Final Answer Correctness Evaluation" guidelines.

---

## Scoring & Output (MODIFIED)

-   Re-read GT and Hypothesis. Score harshly.
-   **Penalize any ungrounded information** or deviation from GT.
-   Score metrics 1-10 on a 1–10 scale.
-   Score metric 11 (`Final_Answer_Correctness`) as 0 or 1.
-   **Compute the Overall Score (TRS) as the arithmetic mean of metrics 1-10 ONLY.**
-   The `Final_Answer_Correctness` is a separate score and is **NOT** included in the
    `Overall Score` calculation.
-   **Output format (strict):** Return **only** a Python dictionary with exactly these keys:
{'Faithfulness-Step': <float>, 'Faithfulness-Token': <float>, 'Informativeness-Step': <float>, 'Repetition-Token': <float>, 'Hallucination': <float>, 'Redundancy': <float>, 'Semantic Coverage-Step': <float>, 'Reasoning Alignment': <float>, 'Commonsense': <float>, 'Missing Step': <float> , 'Overall Score': <float>, 'Final_Answer_Correctness': <int>}
Do not change key names, do not add/remove keys, do not add text, code fences, or comments.
"""

# --- Helper Functions ---

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file (one per line) or a standard JSON array."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                data = json.loads(content)
            else:
                for line in content.splitlines():
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logging.warning(f"Skipping malformed line in {file_path}: {e}")
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Saves a list of dictionaries to a JSONL file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    except IOError as e:
        logging.error(f"Failed to write to file {file_path}: {e}")
        raise


def format_steps_for_prompt(steps: List[str]) -> str:
    """Formats a list of reasoning steps for the user prompt."""
    if not steps:
        return "N/A"
    # Format as a numbered list
    return "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))

# --- *** NEW HELPER FUNCTION *** ---
def get_answer_text_from_key(options: List[str], key: str) -> Optional[str]:
    """Finds the full text of an option (e.g., "A. ...") given the key (e.g., "A")."""
    if not key or not options:
        return None
    
    # Matches "A.", "A)", "A " (with space), or just "A" at start of string
    key_pattern = f"^{re.escape(key.strip())}[.)\\s]?"
    
    for option_text in options:
        if re.match(key_pattern, option_text.strip(), re.IGNORECASE):
            # Return the full option text, e.g., "A. Increased nodular..."
            return option_text.strip()
    
    logging.warning(f"Could not find matching option text for key: {key}")
    return None

# --- *** COMPLETELY NEW PROMPT BUILDER *** ---
def create_judge_user_prompt(
    gt_item: Dict[str, Any],
    model_parsed: Dict[str, Any]
) -> (str, str, str):
    """
    Creates the user-facing prompt for the judge LLM, intelligently
    handling MCQs vs. open-ended questions.
    
    Returns a tuple of:
    (user_prompt_string, gt_answer_for_judge, model_answer_for_judge)
    """
    
    # --- 1. Get Base Data ---
    gt_steps = gt_item.get("steps", [])
    gt_question = " ".join(gt_item.get("question", [])) # Join question list
    gt_options = gt_item.get("options") # List of options or None
    gt_answer_key_or_text = gt_item.get("answer", "").strip()

    model_steps = model_parsed.get("steps", [])
    model_answer_text = model_parsed.get("answer", "").strip()
    model_answer_key = model_parsed.get("answer_key", "").strip()

    # --- 2. Format Steps ---
    formatted_gt_steps = format_steps_for_prompt(gt_steps)
    formatted_model_steps = format_steps_for_prompt(model_steps)
    
    # --- 3. Build Prompt Context (Question & Options) ---
    context_prompt_part = f"""
[QUESTION]
{gt_question}
"""
    
    is_mcq = gt_options and isinstance(gt_options, list) and len(gt_options) > 0

    if is_mcq:
        formatted_options = "\n".join(gt_options)
        context_prompt_part += f"""
[OPTIONS]
{formatted_options}
"""

    # --- 4. Determine Correct Answers to Compare ---
    gt_answer_for_judge = ""
    model_answer_for_judge = ""

    if is_mcq:
        # Find the full text for the ground truth answer key (e.g., "A")
        gt_answer_for_judge = get_answer_text_from_key(gt_options, gt_answer_key_or_text)
        if not gt_answer_for_judge:
            # This should not happen if GT data is clean
            logging.error(f"ID {gt_item.get('id')}: CRITICAL: Could not find GT answer text for key '{gt_answer_key_or_text}'.")
            gt_answer_for_judge = gt_answer_key_or_text # Fallback to just the key

        # Find the full text for the model's answer.
        # Priority 1: Use the model's 'answer_key' (e.g., "D")
        if model_answer_key:
            model_answer_for_judge = get_answer_text_from_key(gt_options, model_answer_key)
        
        # Priority 2: If no key, check if 'answer' *is* the key (e.g., "A")
        if not model_answer_for_judge and len(model_answer_text) < 5:
             model_answer_for_judge = get_answer_text_from_key(gt_options, model_answer_text)

        # Priority 3: Use the model's 'answer' text as-is.
        # This handles the case where the model outputs the full text.
        if not model_answer_for_judge:
            model_answer_for_judge = model_answer_text
            
    else:
        # Open-ended: Just compare the text.
        gt_answer_for_judge = gt_answer_key_or_text
        model_answer_for_judge = model_answer_text
        
    # --- 5. Assemble Final User Prompt ---
    user_prompt = f"""
You must evaluate the [HYPOTHESIS] against the [GROUND TRUTH] based on the rules
in your system prompt. Use the [QUESTION] and [OPTIONS] (if provided) for context.
{context_prompt_part}
---

[GROUND TRUTH REASONING]
{formatted_gt_steps}

[GROUND TRUTH FINAL ANSWER]
{gt_answer_for_judge}

---

[HYPOTHESIS REASONING]
{formatted_model_steps}

[HYPOTHESIS FINAL ANSWER]
{model_answer_for_judge}

---

Please evaluate the hypothesis against the ground truth using the metrics and
rules provided in your system instructions. Return *only* the JSON dictionary.
"""
    
    return (user_prompt, gt_answer_for_judge, model_answer_for_judge)


def get_llm_evaluation(
    client: OpenAI,
    gt_item: Dict[str, Any],         # <-- MODIFIED
    model_parsed: Dict[str, Any]    # <-- MODIFIED
) -> Optional[Dict[str, Any]]: 
    """
    Calls the OpenAI API to get an evaluation for a single sample.
    Handles retries with exponential backoff.
    """
    
    # Build the intelligent prompt
    try:
        (user_prompt, _, _) = create_judge_user_prompt(gt_item, model_parsed)
    except Exception as e:
        logging.error(f"ID {gt_item.get('id')}: Failed to build prompt: {e}")
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    retries = 0
    backoff = INITIAL_BACKOFF
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            response_content = response.choices[0].message.content
            if not response_content:
                logging.warning(f"ID {gt_item.get('id')}: Received empty response from API.")
                return None

            try:
                scores = json.loads(response_content)
                
                required_keys = {
                    'Faithfulness-Step', 'Faithfulness-Token', 
                    'Informativeness-Step', 'Repetition-Token', 
                    'Hallucination', 'Redundancy', 'Semantic Coverage-Step',
                    'Reasoning Alignment', 'Commonsense', 'Missing Step', 
                    'Overall Score', 'Final_Answer_Correctness'
                }
                if not isinstance(scores, dict) or not required_keys.issubset(scores.keys()):
                    logging.warning(f"ID {gt_item.get('id')}: API returned malformed JSON structure: {scores}")
                    return None
                
                final_answer_score = scores.get('Final_Answer_Correctness')
                if final_answer_score not in [0, 1]:
                    logging.warning(f"ID {gt_item.get('id')}: API returned invalid Final_Answer_Correctness: {final_answer_score}")
                    return None
                    
                return scores

            except json.JSONDecodeError as e:
                logging.warning(
                    f"ID {gt_item.get('id')}: API response was not valid JSON. Error: {e}\nResponse: {response_content}"
                )
                return None

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            logging.warning(f"ID {gt_item.get('id')}: API error: {e}. Retrying in {backoff}s...")
            time.sleep(backoff)
            retries += 1
            backoff *= 2
        except Exception as e:
            logging.error(f"ID {gt_item.get('id')}: An unexpected error occurred during API call: {e}")
            return None

    logging.error(f"ID {gt_item.get('id')}: Failed to get evaluation after {MAX_RETRIES} retries.")
    return None


# Main execution

def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run LLM-as-Judge evaluation for Time-Aware MRI benchmark."
    )
    parser.add_argument(
        "--gt_file",
        type=str,
        required=True,
        help="Path to the ground truth JSONL file.",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Path to the folder containing model output .jsonl files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder to save evaluation results (JSONL).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel workers for API calls. Default: 8",
    )
    args = parser.parse_args()

    # --- 1. Initialize OpenAI Client (Once) ---
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY environment variable not set.")
        return
    client = OpenAI(api_key=api_key)

    # --- 2. Load Ground Truth Data (Once) ---
    logging.info(f"Loading ground truth from: {args.gt_file}")
    try:
        gt_data = load_jsonl(args.gt_file)
        gt_map = {}
        for item in gt_data:
            qid = item.get("qa_id") or item.get("id")
            if qid:
                gt_map[qid] = item
        logging.info(f"Found {len(gt_map)} ground truth samples.")
    except FileNotFoundError:
        logging.error(f"Ground truth file not found: {args.gt_file}")
        return
    except Exception as e:
        logging.error(f"Error loading ground truth file: {e}")
        return

    # --- 3. Find Model Files ---
    model_files = glob.glob(os.path.join(args.model_folder, "*.jsonl"))
    if not model_files:
        logging.warning(f"No .jsonl files found in folder: {args.model_folder}")
        return
    
    logging.info(f"Found {len(model_files)} model files to process.")

    # --- 4. Create Output Folder ---
    os.makedirs(args.output_folder, exist_ok=True)
    logging.info(f"Output will be saved to: {args.output_folder}")

    # --- 5. Process Each Model File ---
    for model_file_path in model_files:
        model_filename = os.path.basename(model_file_path)
        output_file_path = os.path.join(args.output_folder, model_filename)
        
        logging.info(f"--- Starting processing for: {model_filename} ---")
        
        try:
            model_data = load_jsonl(model_file_path)
            model_map = {}
            for item in model_data:
                qid = item.get("qa_id") or item.get("id")
                if qid:
                    model_map[qid] = item
            logging.info(f"Loaded {len(model_map)} model outputs from {model_filename}.")
        except FileNotFoundError:
            logging.error(f"Model file not found: {model_file_path}. Skipping.")
            continue
        except Exception as e:
            logging.error(f"Error loading model file {model_file_path}: {e}. Skipping.")
            continue

        results = []
        processed_count = 0
        
        # --- 6. Process Samples in Parallel ---
        futures_to_sample_data = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            logging.info(f"Submitting samples to {args.max_workers} workers...")
            
            # --- Submission Phase ---
            for sample_id, gt_item in gt_map.items():
                if sample_id not in model_map:
                    logging.warning(f"No matching model output for ID: {sample_id} in {model_filename}. Skipping.")
                    continue
                
                model_item = model_map[sample_id]
                
                # --- *** MODIFIED SUBMISSION *** ---
                try:
                    model_parsed = model_item.get("parsed", {})
                    if not isinstance(model_parsed, dict):
                        logging.warning(f"Skipping ID {sample_id}: 'parsed' key missing or invalid in model output.")
                        continue
                        
                    future = executor.submit(
                        get_llm_evaluation,
                        client,
                        gt_item,       # <-- Send full GT item
                        model_parsed   # <-- Send model's parsed dict
                    )
                    
                    # Store data for logging
                    futures_to_sample_data[future] = {
                        "id": sample_id,
                        "gt_item": gt_item,
                        "model_item": model_item
                    }

                except Exception as e:
                    logging.error(f"An unexpected error occurred prepping ID {sample_id}: {e}")

            # --- Collection Phase ---
            total_tasks = len(futures_to_sample_data)
            logging.info(f"Submitted {total_tasks} valid samples. Waiting for completion...")

            for i, future in enumerate(concurrent.futures.as_completed(futures_to_sample_data)):
                sample_data = futures_to_sample_data[future]
                sample_id = sample_data["id"]
                gt_item = sample_data["gt_item"]
                model_item = sample_data["model_item"]

                if (i + 1) % 10 == 0 or (i + 1) == total_tasks:
                    logging.info(f"--- Completed sample {i+1}/{total_tasks} (ID: {sample_id}) ---")

                try:
                    evaluation_scores = future.result() 

                    if evaluation_scores:
                        
                        # --- Get the answers that were *actually* judged for logging ---
                        # This re-runs the prompt builder, but it's fast and avoids
                        # passing complex data through the futures map.
                        _, gt_judged, model_judged = create_judge_user_prompt(
                            gt_item, model_item.get("parsed", {})
                        )
                        
                        result_record = {
                            "id": sample_id,
                            "patient_id": gt_item.get("patient_id"),
                            "evaluation": evaluation_scores,
                            "judged_answers": {
                                "ground_truth": gt_judged,
                                "model": model_judged
                            },
                            "ground_truth": {
                                "question": gt_item.get("question"),
                                "options": gt_item.get("options"),
                                "steps": gt_item.get("steps", []),
                                "answer": gt_item.get("answer", "")
                            },
                            "model_output": {
                                "steps": model_item.get("parsed", {}).get("steps", []),
                                "answer": model_item.get("parsed", {}).get("answer", ""),
                                "answer_key": model_item.get("parsed", {}).get("answer_key", "")
                            }
                        }
                        results.append(result_record)
                        processed_count += 1
                        
                        # Incremental save
                        with open(output_file_path, "a", encoding="utf-8") as f_out:
                            f_out.write(json.dumps(result_record) + "\n")
                    else:
                        logging.error(f"Failed to evaluate ID: {sample_id} (API call failed or returned None).")

                except Exception as e:
                    logging.error(f"An unexpected error occurred collecting result for ID {sample_id}: {e}")

        # --- 7. Save Results (Per-File) ---
        if results:
            logging.info(f"Saving {len(results)} evaluation results to: {output_file_path}")
            try:
                save_jsonl(results, output_file_path)
            except Exception as e:
                logging.error(f"Failed to save results to {output_file_path}: {e}")
        else:
            logging.warning(f"No results were generated for {model_filename}.")

        logging.info(f"--- Finished processing for: {model_filename} ---")
        logging.info(f"Successfully processed {processed_count} out of {total_tasks} submitted samples.")

    logging.info(f"--- All Model Files Processed ---")


if __name__ == "__main__":
    main()
