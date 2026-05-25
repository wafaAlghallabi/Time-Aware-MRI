import os
import re
import io
import json
import base64
import argparse
import logging
import textwrap
from typing import List, Dict, Optional, Any, Tuple
import time
import asyncio # New import

# Import PIL for image loading
from PIL import Image, ImageDraw, ImageFont
# Import the Google Generative AI SDK
import google.generativeai as genai
# New import for progress bar
from tqdm import tqdm 

from google.generativeai.types import HarmCategory, HarmBlockThreshold

# =========================
# Config
# (Model ID is now a command-line argument)
# =========================

# =========================
# Model setup
# (This section is now handled in main() by initializing the genai model)
# =========================

# =========================
# Prompts
# (These are unchanged as they define the task logic)
# =========================

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

# =========================
# I/O helpers
# =========================

def read_samples(path: str) -> List[Dict[str, Any]]:
    """Reads JSON array or JSONL file."""
    # Check if file exists and is not empty
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return []
        
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith(("[", "{")):
            try:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                pass # Fallback to JSONL
        
        # Fallback: JSONL
        f.seek(0)
        out = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSONL line in {path}: {line[:50]}...")
        return out

def save_jsonl_record(record: Dict[str, Any], out_path: str) -> None:
    """Appends a single JSON record to a JSONL file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    # Open in 'a' (append) mode
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# --- New Function: Cleans and re-orders the final file ---
def clean_and_reorder_output(samples_path: str, output_path: str):
    """
    Reads the original samples file and the (messy) output file.
    Writes a new, clean output file containing only successful results
    in the original sample order.
    """
    logging.info(f"Cleaning and re-ordering final output file: {output_path}")
    
    try:
        original_samples = read_samples(samples_path)
        if not original_samples:
            logging.warning("Original samples file is empty. Cannot re-order.")
            return

        all_results = read_samples(output_path)
        if not all_results:
            logging.warning("Output file is empty. No results to clean.")
            return

        # Create a map of {id: successful_result}
        # This de-duplicates and keeps only the latest successful result
        results_map = {}
        for r in all_results:
            if r.get("id") and "error" not in r and r.get("valid_json") == True:
                results_map[r["id"]] = r
        
        logging.info(f"Found {len(results_map)} successful unique results.")

        # Iterate through original samples to build the final ordered list
        final_ordered_results = []
        not_found_count = 0
        for i, sample in enumerate(original_samples, 1):
            sid = sample.get("id", f"case_{i}")
            if sid in results_map:
                final_ordered_results.append(results_map[sid])
            else:
                logging.warning(f"No successful result found for sample id={sid}. It will not be in the final file.")
                not_found_count += 1
        
        # Overwrite the output file with the clean, ordered results
        with open(output_path, "w", encoding="utf-8") as f:
            for record in final_ordered_results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logging.info(f"Wrote {len(final_ordered_results)} successful, ordered records to {output_path}")
        if not_found_count > 0:
            logging.warning(f"{not_found_count} samples from the original file had no successful result.")

    except Exception as e:
        logging.error(f"Failed to clean and re-order output file: {e}")
# ---

# =========================
# PNG resolver
# (Unchanged)
# =========================

def _stem(image_id: str) -> str:
    base = os.path.basename(image_id)
    return base

def png_path_for(sample: Dict[str, Any], root: str, image_id: str) -> str:
    dataset = str(sample.get("dataset", "")).strip()
    patient_id = str(sample.get("patient_id", "")).strip()
    stem = _stem(image_id)
    return os.path.join(root, dataset, patient_id, f"{stem}.png")

def resolve_image_path(root_dir: str, dataset: str, patient_id: str, img_meta: Dict[str, Any]) -> str:
    if dataset == "Yale-BrainMets":
        rel_path = img_meta.get('path')
        if not rel_path: return ""
        return os.path.join(root_dir, "Yale_multiview_slices", rel_path)
    elif dataset == "UCSF-GBM":
        filename = img_meta.get('filename')
        timepoint = img_meta.get('timepoint')
        if not filename or timepoint is None: return ""
        return os.path.join(root_dir, "UCSF_seg_multiview_slices", str(patient_id), f"timepoint_{timepoint}", filename)
    elif dataset == "Lumiere-BrainMets":
        rel_path = img_meta.get('path')
        if not rel_path: return ""
        return os.path.join(root_dir, "lumiere_multiview_slices", rel_path)
    elif dataset == "UCSD-PTGBM":
        rel_path = img_meta.get('path')
        if not rel_path: return ""
        return os.path.join(root_dir, "UCSD_PTGBM_multiview_slices", rel_path)
    elif dataset in ["OASIS-2", "RHUH-GBM"]:
        rel_path = img_meta.get('path')
        if not rel_path: return ""
        possible_paths = [
            os.path.join(root_dir, rel_path),
            os.path.join(root_dir, dataset.replace("-2", ""), rel_path),
            os.path.join(root_dir, "Yale_multiview_slices", rel_path), 
            os.path.join("/home/omkar/Wafa/MRI-Res-Benchmark/Long-MRI-Seg/", rel_path)
        ]
        for p in possible_paths:
            if os.path.exists(p):
                return p
        return possible_paths[0]
    return ""
def load_png(sample: Dict[str, Any], root: str, image_id: str) -> Optional[Image.Image]:
    """
    Loads a PNG, converts to RGB, and resizes if larger than 1024x1024.
    """
    path = png_path_for(sample, root, image_id)
    if os.path.isfile(path):
        try:
            img = Image.open(path).convert("RGB")
            
            # --- NEW RESIZING STEP ---
            # Define a maximum size. 1024x1024 is a good balance.
            MAX_SIZE = (1024, 1024) 
            
            # thumbnail() resizes the image in-place, maintaining aspect ratio,
            # only if it's larger than MAX_SIZE.
            img.thumbnail(MAX_SIZE)
            # --- END NEW STEP ---
            
            return img
        except Exception as e:
            logging.warning(f"Failed to open or resize {path}: {e}")
    else:
        logging.warning(f"Missing PNG for image_id='{image_id}' at {path}")
    return None

def create_grid_for_timepoint(root: str, patient_id: str, tp_images: List[Dict], dataset: str) -> Optional[Image.Image]:
    cell_size = 256
    padding = 20
    modalities = sorted(list(set(img.get('sequence', '') for img in tp_images)))
    views = sorted(list(set(img.get('view', '') for img in tp_images)))
    
    # Check if any image is present
    any_present = False
    for img_meta in tp_images:
        path = resolve_image_path(root, dataset, patient_id, img_meta)
        if os.path.exists(path):
            any_present = True
            break
            
    if not any_present:
        return None
    
    grid_w = len(views) * cell_size + (len(views) - 1) * padding
    grid_h = len(modalities) * cell_size + (len(modalities) - 1) * padding
    
    grid_img = Image.new('RGB', (grid_w, grid_h), color=(0, 0, 0))
    try:
        font = ImageFont.truetype("LiberationSans-Regular.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid_img)
    
    for img_meta in tp_images:
        row = modalities.index(img_meta.get('sequence', ''))
        col = views.index(img_meta.get('view', ''))
        
        path = resolve_image_path(root, dataset, patient_id, img_meta)
        if os.path.exists(path):
            try:
                img = Image.open(path).convert('RGB')
                img.thumbnail((cell_size, cell_size)) 
                
                x = col * (cell_size + padding)
                y = row * (cell_size + padding)
                grid_img.paste(img, (x, y))
                
                label = f"{img_meta.get('sequence', '').upper()} {img_meta.get('view', '').capitalize()}"
                draw.text((x + 5, y + 5), label, fill=(255, 255, 0), font=font) 
            except Exception as e:
                logging.warning(f"Failed to load/paste {path}: {e}")
            
    return grid_img

# =========================
# Prompt builders (with options)
# (Unchanged)
# =========================

def parse_options(options: List[str]) -> List[Tuple[str, str]]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    normalized = []
    next_letter_idx = 0
    for raw in options:
        s = raw.strip()
        m = re.match(r"^\s*([A-Za-z])[\.\)]\s*(.+)$", s)
        if m:
            letter = m.group(1).upper()
            text = m.group(2).strip()
        else:
            letter = letters[next_letter_idx]
            next_letter_idx += 1
            text = re.sub(r"^[A-Za-z][\.\)]\s*", "", s).strip()
        normalized.append((letter, text))
    return normalized

def _format_timepoints(timepoints: List[Dict]) -> str:
    lines = []
    for tp in timepoints:
        label = tp.get("label") or f"Index {tp.get('index', '?')}"
        image_id = tp.get("image_id", "N/A")
        lines.append(f"- {label} — image_id: {image_id}")
    return "\n".join(lines)

def build_user_text(sample: Dict[str, Any]) -> str:
    age = sample.get("age", "")
    sex = sample.get("sex", "")
    dataset = sample.get("dataset", "")
    question_text = (sample.get("question") or [""])[0]
    timepoints = sample.get("timepoints", [])
    options = sample.get("options")
    tp_block = _format_timepoints(timepoints)

    if options:
        opts = parse_options(options)
        opts_block = "\n".join([f"{k}. {v}" for k, v in opts])
        json_spec = textwrap.dedent("""
        {
          "steps": [
            "Step 1 (≤30 words): Baseline findings using the timepoint label.",
            "Step 2 (≤30 words): First follow-up vs baseline (and vs prior).",
            "…",
            "Final step (≤30 words): Summarize the longitudinal trend and any caveats."
          ],
          "answer": "One capital letter only (e.g., A), matching one of the provided options.",
          "answer_key": "Same capital letter (duplicate of 'answer').",
          "answer_option": "Exact text of the chosen option."
        }
        """).strip()
        formatting = textwrap.dedent("""
        - Choose exactly ONE option by letter (A/B/C/…).
        - Put only the letter in "answer" and "answer_key".
        - Copy the full chosen option text into "answer_option".
        - Provide reasoning only in "steps" (3–6 steps).
        """).strip()
        user_text = f"""\
Task: Analyze the longitudinal brain MRI case and answer the comparison question by choosing ONE option.
Patient metadata
- Age: {age}
- Sex: {sex}
- Dataset: {dataset}
Question
{question_text}
Options
{opts_block}
Timepoints
{tp_block}
Your output must be valid JSON with exactly these keys:
{json_spec}
Formatting constraints
{formatting}
"""
    else:
        json_spec = textwrap.dedent("""
        {
          "steps": [
            "Step 1 (≤30 words): Baseline findings using the timepoint label.",
            "Step 2 (≤30 words): First follow-up vs baseline (and vs prior).",
            "…",
            "Final step (≤30 words): Summarize the longitudinal trend and any caveats."
          ],
          "answer": "One sentence (≤30 words) using categorical change terms."
        }
        """).strip()
        formatting = textwrap.dedent("""
        - 3–6 steps total.
        - Refer to scans by their labels (e.g., “V1 • 2013-11-22”).
        - Mention the target explicitly (e.g., surgical cavity, adjacent gliosis).
        - If uncertain, use “indeterminate due to {reason}”.
        """).strip()
        user_text = f"""\
Task: Analyze the longitudinal brain MRI case and answer the comparison question.
Patient metadata
- Age: {age}
- Sex: {sex}
- Dataset: {dataset}
Question
{question_text}
(If multiple questions are present, answer the first.)
Timepoints
{tp_block}
Your output must be valid JSON with exactly these keys:
{json_spec}
Formatting constraints
{formatting}
"""
    user_text += "\n\nOutput only the JSON object as described."
    return textwrap.dedent(user_text).strip()

def build_gemini_parts(sample: Dict[str, Any], images: List[Image.Image], root: str = "") -> List[Any]:
    user_text = build_user_text(sample)
    parts = [user_text]
    
    if "images" in sample and isinstance(sample["images"], list) and len(sample["images"]) > 0:
        # Long-MRI-Seg multiview grid schema
        tp_images = {}
        for img_meta in sample["images"]:
            tp = img_meta.get("timepoint")
            if tp not in tp_images: tp_images[tp] = []
            tp_images[tp].append(img_meta)
            
        patient_id = str(sample.get("patient_id", ""))
        dataset = str(sample.get("dataset", ""))
        for tp in sorted(tp_images.keys()):
            grid = create_grid_for_timepoint(root, patient_id, tp_images[tp], dataset)
            if grid is not None:
                parts.append(f"\n--- Timepoint {tp} Grid ---")
                parts.append(grid)
    else:
        # Legacy single image schema
        for tp, img in zip(sample.get("timepoints", []), images):
            label = tp.get("label", f"Index {tp.get('index','?')}")
            parts.append(f"\nTimepoint: {label}")
            parts.append(img)
    return parts


# =========================
# Generation + parsing
# =========================

def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    stack = []
    start = None
    blocks = []
    for i, ch in enumerate(text):
        if ch == '{':
            if not stack:
                start = i
            stack.append('{')
        elif ch == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    blocks.append(text[start:i+1])
    for blk in reversed(blocks):
        try:
            return json.loads(blk)
        except Exception:
            continue
    return None

async def generate_case_async(
    model: genai.GenerativeModel, 
    sample: Dict[str, Any], 
    root: str, 
    max_new_tokens: int,  
    do_sample: bool = False
) -> Dict[str, Any]:
    
    images, missing = [], []
    patient_id = str(sample.get("patient_id", ""))
    dataset = str(sample.get("dataset", ""))
    
    if "images" in sample and isinstance(sample["images"], list) and len(sample["images"]) > 0:
        # Multi-view schema: check if files exist
        for img_meta in sample["images"]:
            path = resolve_image_path(root, dataset, patient_id, img_meta)
            if not os.path.exists(path):
                missing.append(img_meta.get("filename", ""))
        
        # If absolutely NO images exist, skip. (Or if > 50% are missing)
        # We will be strict: if more than half are missing, skip it to avoid bad generations.
        if len(missing) > len(sample["images"]) / 2:
            logging.warning(f"Skipping QID={sample.get('qa_id')}, too many missing multiview images ({len(missing)}/{len(sample['images'])}): {missing[:3]}...")
            return {
                "qa_id": sample.get("qa_id"),
                "error": "Too many missing images in multiview set",
                "missing": missing
            }
    else:
        # Legacy single image fallback
        for tp in sample.get("timepoints", []):
            img_id = tp.get("image_id")
            img = load_png(sample, root, img_id)
            if img:
                images.append(img)
            else:
                missing.append(img_id)
                
        if len(missing) > 0:
            logging.warning(f"Skipping QID={sample.get('qa_id')}, missing images: {missing}")
            return {
                "qa_id": sample.get("qa_id"),
                "error": "Missing images",
                "missing": missing
            }
    parts = build_gemini_parts(sample, images, root=root)
    temp = 0.7 if do_sample else 0.0
    
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_new_tokens,
        temperature=temp,
        response_mime_type="application/json", 
    )

    # --- THIS IS THE CRITICAL FIX ---
    response = await model.generate_content_async(
        parts,
        generation_config=generation_config
    )

    try:
        # This is the line that is currently crashing
        decoded = response.text
    except ValueError as e:
        # This 'except' block will catch the crash and log the real reason
        finish_reason = "UNKNOWN"
        # Check if candidates list exists and has items
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason.name
        
        sid = sample.get('id', 'unknown')
        logging.error(f"Response blocked or empty for id={sid}. FinishReason: {finish_reason}.")
        
        # Log safety ratings if they exist
        if response.candidates and response.candidates[0].safety_ratings:
            logging.error(f"Safety Ratings for id={sid}: {response.candidates[0].safety_ratings}")
            
        # Re-raise a clearer error for the worker to catch
        raise RuntimeError(f"Response blocked or empty for id={sid}. FinishReason: {finish_reason}")
    except Exception as e:
        # Catch any other unexpected response error
        sid = sample.get('id', 'unknown')
        logging.error(f"Unexpected error accessing response.text for id={sid}: {e}")
        raise e
    # --- END OF FIX ---

    parsed = extract_json_obj(decoded)

    result = {
        "id": sample.get("id"),
        "patient_id": sample.get("patient_id"),
        "raw_text": decoded,
        "parsed": parsed if isinstance(parsed, dict) else None,
        "valid_json": isinstance(parsed, dict),
    }
    if result["valid_json"]:
        result["steps"] = parsed.get("steps")
        result["answer"] = parsed.get("answer")
        if "answer_key" in parsed:
            result["answer_key"] = parsed["answer_key"]
        if "answer_option" in parsed:
            result["answer_option"] = parsed["answer_option"]
    return result



async def process_sample_worker(
    sample: Dict[str, Any],
    model: genai.GenerativeModel,
    root: str,
    max_new_tokens: int,
    do_sample: bool,
    semaphore: asyncio.Semaphore,
    queue: asyncio.Queue,
    delay: float = 0.0
):
    """A worker that processes one sample and puts the result on the queue."""
    sid = sample.get("id", "unknown_id")
    async with semaphore: # Acquire the semaphore to limit concurrency
        t0 = time.time()
        try:
            import google.api_core.exceptions
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    res = await generate_case_async(
                        model=model,  
                        sample=sample,
                        root=root,
                        max_new_tokens=max_new_tokens, 
                        do_sample=do_sample
                    )
                    await queue.put(res)
                    break
                except google.api_core.exceptions.ResourceExhausted as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(10) # Backoff for paid tier rate limits (RPM)
                    else:
                        raise e
        except Exception as e:
            dt = time.time() - t0
            logging.error(f"ERROR case id={sid} after {dt:.1f}s: {e}")
            error_record = {
                "id": sample.get("id"),
                "patient_id": sample.get("patient_id"),
                "error": str(e)
            }
            await queue.put(error_record)
        if delay > 0:
            await asyncio.sleep(delay)

async def results_writer(out_path: str, queue: asyncio.Queue, total: int):
    """A writer that saves results from the queue to the file and updates progress."""
    pbar = tqdm(total=total, desc="Processing samples", unit="sample")
    while True:
        record = await queue.get()
        if record is None: # Sentinel value to stop
            queue.task_done()
            break
        
        # Save the record (success or error) to the messy file
        save_jsonl_record(record, out_path)
        
        # Update progress bar
        if "error" in record:
            pbar.set_postfix_str(f"Last ID: {record.get('id')} (ERROR)")
        else:
            pbar.set_postfix_str(f"Last ID: {record.get('id')} (OK)")
        pbar.update(1)
        queue.task_done()
    pbar.close()

async def async_main(args):
    """Main asynchronous orchestration function."""
    
    # --- Gemini API Setup ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY environment variable not set.")
        return
    
    try:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            args.model_id,
            system_instruction=SYSTEM_PROMPT,
            safety_settings=safety_settings
        )
        logging.info(f"Initialized Gemini model: {args.model_id}")
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {e}")
        return
    # -------------------------

    samples = read_samples(args.samples)
    if args.limit is not None:
        samples = samples[:args.limit]
        
    for i, s in enumerate(samples, 1):
        if "id" not in s:
            s["id"] = f"case_{i}"
    logging.info(f"Loaded {len(samples)} total samples from {args.samples} (limit={args.limit})")

    # --- Load completed IDs to skip (FIXED) ---
    completed_ids = set()
    if os.path.isfile(args.out):
        try:
            logging.info(f"Found existing output file. Reading completed samples from {args.out}...")
            completed_samples = read_samples(args.out)
            for s in completed_samples:
                # *** FIX ***: Only skip if the record has an ID and *NO* error key.
                if s.get("id") and "error" not in s:
                    completed_ids.add(s["id"])
            logging.info(f"Found {len(completed_ids)} successfully completed sample IDs. These will be skipped.")
        except Exception as e:
            logging.warning(f"Could not read existing output file {args.out}. Will run all samples. Error: {e}")
    # ---

    # --- Filter out completed samples ---
    tasks_to_run = []
    for sample in samples:
        sid = sample.get("id", f"case_{samples.index(sample)+1}")
        if sid in completed_ids:
            # logging.info(f"SKIP case id={sid} (already present in output)")
            continue
        tasks_to_run.append(sample)
    
    if not tasks_to_run:
        logging.info("No new samples to process.")
        return
    
    logging.info(f"Processing {len(tasks_to_run)} new samples with {args.num_workers} parallel workers...")

    # --- Setup async processing ---
    semaphore = asyncio.Semaphore(args.num_workers)
    results_queue = asyncio.Queue()

    # Start the writer task
    writer = asyncio.create_task(results_writer(args.out, results_queue, len(tasks_to_run)))

    # Create worker tasks
    worker_tasks = []
    for sample in tasks_to_run:
        worker_tasks.append(
            asyncio.create_task(
                process_sample_worker(
                    sample=sample,
                    model=model,
                    root=args.root,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    semaphore=semaphore,
                    queue=results_queue,
                    delay=args.delay
                )
            )
        )

    # Wait for all workers to finish
    await asyncio.gather(*worker_tasks)

    # Signal the writer to stop
    await results_queue.put(None)
    
    # Wait for the writer to finish
    await writer

    logging.info(f"All processing complete.")


# =========================
# Main (Sync)
# =========================
def main():
    ap = argparse.ArgumentParser(description="Generate reasoning steps (supports MCQ) from PNGs stored at <root>/<dataset>/<patient_id>/<image_id>.png")
    
    ap.add_argument("--samples", required=True, help="Path to samples (.json or .jsonl).")
    ap.add_argument("--root", required=True, help="Root directory that contains dataset folders.")
    ap.add_argument("--out", default="outputs/gemini_steps.jsonl", help="Output JSONL (will append if exists).")
    ap.add_argument("--model-id", type=str, help="The Gemini model ID to use.")
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="Max output tokens.")
    
    ap.add_argument("--num-workers", type=int, default=8, help="Number of parallel requests to run.")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay in seconds after each sample to throttle API calls.")
    ap.add_argument("--limit", type=int, default=None, help="Limit the number of samples to process (useful for testing).")
    
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    # Configure logging
    logging.basicConfig(
        level=args.log_level.upper(), 
        format="%(levelname)s: %(asctime)s: %(message)s", 
        datefmt='%Y-%m-%d %H:%M:%S',
        # Log to both file and console (optional)
        # handlers=[
        #     logging.FileHandler("processing.log"),
        #     logging.StreamHandler()
        # ]
    )

    # Start the async event loop
    try:
        asyncio.run(async_main(args))
        
        # --- New: Add the final cleanup step ---
        # This runs *after* all async processing is done.
        clean_and_reorder_output(args.samples, args.out)
        
    except KeyboardInterrupt:
        logging.info("Process interrupted by user. Output file may be incomplete or unordered.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}")

if __name__ == "__main__":
    main()