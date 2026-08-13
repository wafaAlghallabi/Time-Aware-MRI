#!/usr/bin/env python3
"""
Neuroradiology interval-change runner (Groq API + Vision + Structured Outputs)
- Async + parallel processing with bounded concurrency
- Immediate append (crash-safe progress)
- Resume: skips already completed IDs

*** NOTE: Requires the GROQ_API_KEY environment variable to be set. ***
"""

import os
import re
import io
import json
import base64
import argparse
import logging
import textwrap
from typing import List, Dict, Optional, Any, Tuple, Set
import time
import asyncio
import random

from PIL import Image, ImageDraw, ImageFont
from groq import AsyncGroq

# =========================
# Config
# =========================

DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct" # Changed model
DEFAULT_MAX_OUT = 320        # override via --max-output-tokens
DEFAULT_CONCURRENCY = 4      # override via --concurrency
USE_DATA_URLS = True

# Async client
aclient = AsyncGroq()

# =========================
# Prompts
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

OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 3,
            "maxItems": 6,
            "items": {"type": "string", "maxLength": 200}
        },
        "answer": {"type": "string", "maxLength": 200},
        # Keep these present in all outputs. For non-MCQ, set them to "".
        "answer_key": {"type": "string", "maxLength": 1},
        "answer_option": {"type": "string", "maxLength": 200}
    },
    "required": ["steps", "answer", "answer_key", "answer_option"],
    "additionalProperties": False
}


# =========================
# I/O helpers
# =========================

def clean_and_sort_output(out_path: str, samples: List[Dict[str, Any]], keep_parse_fail: bool = False) -> Set[str]:
    """
    - Reads existing out_path (if any)
    - Keeps only 'correct' rows (valid_json==True). If keep_parse_fail=True, also keep parse_fail rows.
    - Dedupes by id (keeps the latest occurrence)
    - Rewrites the file ordered by the order in `samples`
    - Returns the set of IDs that remain (treated as 'done')
    """
    if not os.path.isfile(out_path):
        return set()

    # 1) Collect latest record per id
    latest: Dict[str, Dict[str, Any]] = {}
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sid = rec.get("id")
            if not sid:
                continue

            is_valid = rec.get("valid_json") is True
            is_error = "error" in rec and rec["error"]
            is_parse_fail = rec.get("valid_json") is False and not is_error

            # keep only correct rows (valid_json==True); optionally keep parse_fail
            if is_valid or (keep_parse_fail and is_parse_fail):
                latest[sid] = rec  # keep the latest occurrence

    if not latest:
        # nothing correct to keep -> empty/clean file
        open(out_path, "w", encoding="utf-8").close()
        return set()

    # 2) Order by samples list
    ordered = []
    kept_ids: Set[str] = set()
    for i, s in enumerate(samples, 1):
        sid = s.get("id", f"case_{i}")
        if sid in latest:
            ordered.append(latest[sid])
            kept_ids.add(sid)

    # 3) Atomic rewrite
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in ordered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)

    print(f"🧹 Cleaned output: kept {len(ordered)} correct rows, removed {len(latest)-len(ordered)} out-of-order/extra rows.")
    return kept_ids


def read_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith(("[", "{")):
            try:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                pass
        f.seek(0)
        out = []
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

def ensure_parent(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def append_jsonl(record: Dict[str, Any], out_path: str) -> None:
    ensure_parent(out_path)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

def load_completed_ids(out_path: str, require_valid: bool = True) -> Set[str]:
    done: Set[str] = set()
    if not os.path.isfile(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sid = rec.get("id")
            if not sid:
                continue
            if require_valid:
                if rec.get("valid_json") is True:
                    done.add(sid)
            else:
                done.add(sid)
    return done

# =========================
# PNG resolver  <root>/<dataset>/<patient_id>/<image_id>.png
# =========================

def _stem(image_id: str) -> str:
    return os.path.basename(image_id)

def png_path_for(sample: Dict[str, Any], root: str, image_id: str) -> str:
    dataset = str(sample.get("dataset", "")).strip()
    patient_id = str(sample.get("patient_id", "")).strip()
    stem = _stem(image_id)
    return os.path.join(root, dataset, patient_id, f"{stem}.png")

def load_png(sample: Dict[str, Any], root: str, image_id: str) -> Optional[Image.Image]:
    path = png_path_for(sample, root, image_id)
    if os.path.isfile(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            logging.warning(f"Failed to open {path}: {e}")
    else:
        logging.warning(f"Missing PNG for image_id='{image_id}' at {path}")
    return None

def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    # This data URL format is compatible with both OpenAI and Groq
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def create_grid_for_timepoint(root: str, patient_id: str, tp_images: List[Dict]) -> Image.Image:
    cell_size = 256
    padding = 20
    modalities = sorted(list(set(img.get('sequence', '') for img in tp_images)))
    views = sorted(list(set(img.get('view', '') for img in tp_images)))
    
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
        
        path = os.path.join(root, patient_id, f"timepoint_{img_meta.get('timepoint', '')}", img_meta.get('filename', ''))
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
# Prompt builders
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
        # in the non-MCQ path
        json_spec = textwrap.dedent("""
        {
        "steps": [
            "Baseline findings using the timepoint label.",
            "First follow-up vs baseline (and vs prior).",
            "…",
            "Summarize the longitudinal trend and any caveats."
        ],
        "answer": "One sentence using categorical change terms.",
        "answer_key": "Empty string if no options.",
        "answer_option": "Empty string if no options."
        }
        """).strip()

        formatting = textwrap.dedent("""
        - 3–6 steps total.
        - Each step ≤ 30 words (do not write the phrase '≤30 words').
        - Refer to scans by their labels (e.g., “V1 • 2013-11-22”).
        - Mention the target explicitly (e.g., surgical cavity, adjacent gliosis).
        - If uncertain, use “indeterminate due to {reason}”.
        - Always include 'answer_key' and 'answer_option'. If no options, set both to "".
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
    return textwrap.dedent(user_text).strip()

def build_case_content_parts(sample: Dict[str, Any], images: List[Image.Image], root: str = "") -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    user_text = build_user_text(sample)
    # This structure is compatible with Groq's image input format
    parts.append({"type": "text", "text": user_text})

    if "images" in sample and isinstance(sample["images"], list) and len(sample["images"]) > 0:
        # Long-MRI-Seg multiview grid schema
        tp_images = {}
        for img_meta in sample["images"]:
            tp = img_meta.get("timepoint")
            if tp not in tp_images: tp_images[tp] = []
            tp_images[tp].append(img_meta)
            
        patient_id = str(sample.get("patient_id", ""))
        for tp in sorted(tp_images.keys()):
            parts.append({"type": "text", "text": f"--- Timepoint {tp} Grid ---"})
            grid = create_grid_for_timepoint(root, patient_id, tp_images[tp])
            parts.append({"type": "image_url", "image_url": {"url": to_data_url(grid)}})
    else:
        # Legacy single image schema
        for tp, img in zip(sample.get("timepoints", []), images):
            label = tp.get("label", f"Index {tp.get('index','?')}")
            parts.append({"type": "text", "text": f"Timepoint: {label}"})
            parts.append({"type": "image_url", "image_url": {"url": to_data_url(img)}})
    return parts

# =========================
# Groq call helpers (async, with backoff)
# =========================

# =========================
# Groq call helpers (async, with backoff)
# =========================

def _get_linspace_indices(n: int, k: int) -> List[int]:
    """Gets k evenly spaced indices from a list of n items."""
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    # Calculate indices
    indices = []
    # step = (n - 1) / (k - 1) if k > 1 else 0
    # Use float division for accuracy
    step = (n - 1.0) / (k - 1.0) if k > 1 else 0.0
    for i in range(k):
        idx = int(round(i * step))
        indices.append(idx)
    return indices

def parts_to_chat_messages(parts):
    # System prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # User content (already in Groq-compatible format from build_case_content_parts)
    messages.append({"role": "user", "content": parts})
    return messages

def decode_chat_text(cresp) -> str:
    # Chat Completions shape (same as OpenAI)
    msg = cresp.choices[0].message
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        return "".join(
            (c.get("text", "") for c in msg.content if isinstance(c, dict) and c.get("type") == "text")
        )
    return ""


# This section ( _create_response_... ) was commented out in your original script
# and is not used by process_one. Leaving it as-is.

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

async def call_with_backoff(coro_factory, *, retries=5, base_delay=1.0, max_delay=10.0):
    """
    coro_factory: a no-arg lambda returning an awaitable (so it can be retried).
    Retries on transient errors with exponential backoff + jitter.
    """
    for attempt in range(retries):
        try:
            return await coro_factory()
        except Exception as e:
            # This will catch groq.RateLimitError, etc.
            if attempt == retries - 1:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.random() * 0.25
            logging.warning(f"Transient error ({type(e).__name__}: {e}). Retry {attempt+1}/{retries-1} in {delay:.1f}s")
            await asyncio.sleep(delay)

# =========================
# Per-sample processing (async)
# =========================

# =========================
# Per-sample processing (async)
# =========================

async def process_one(sample: Dict[str, Any],
                      root: str,
                      model: str,
                      max_output_tokens: int,
                      out_path: str,
                      file_lock: asyncio.Lock,
                      done_ids: Set[str],
                      done_lock: asyncio.Lock,
                      use_schema_first: bool = True) -> None:
    sid = sample.get("id")
    t0 = time.time()

    # ==================================================================
    # === NEW LOGIC: Sub-sample timepoints if > 5 ===
    # ==================================================================
    
    # We will select a maximum of 5 images:
    # 1. The first image (baseline)
    # 2. The last image (most recent)
    # 3. 3 evenly-spaced images from the middle
    
    MAX_IMAGES = 5
    original_timepoints = sample.get("timepoints", [])
    
    if len(original_timepoints) > MAX_IMAGES:
        logging.warning(f"id={sid}: Sample has {len(original_timepoints)} timepoints. Sub-sampling to {MAX_IMAGES}.")
        
        selected_timepoints = []
        middle_timepoints = original_timepoints[1:-1]
        
        # We need 3 images from the middle
        k_middle = MAX_IMAGES - 2 # (5 - 2 = 3)
        
        # Get k evenly spaced indices from the middle list
        middle_indices = _get_linspace_indices(n=len(middle_timepoints), k=k_middle)
        
        # Build the new list
        selected_timepoints.append(original_timepoints[0]) # 1. Add first
        for idx in middle_indices:
            selected_timepoints.append(middle_timepoints[idx]) # 2. Add middle
        selected_timepoints.append(original_timepoints[-1]) # 3. Add last
        
        # Create a copy of the sample and replace its timepoints
        # This ensures build_user_text and build_case_content_parts
        # are consistent with the 5 selected images.
        sub_sample = dict(sample)
        sub_sample["timepoints"] = selected_timepoints
        
    else:
        # No sub-sampling needed
        sub_sample = sample

    # ==================================================================
    # === End of new logic ===
    # ==================================================================


    images, missing = [], []
    patient_id = str(sample.get("patient_id", ""))
    
    if "images" in sub_sample and isinstance(sub_sample["images"], list) and len(sub_sample["images"]) > 0:
        # Multi-view schema: check if files exist
        for img_meta in sub_sample["images"]:
            path = os.path.join(root, patient_id, f"timepoint_{img_meta.get('timepoint', '')}", img_meta.get("filename", ""))
            if not os.path.exists(path):
                missing.append(img_meta.get("filename", ""))
        if len(missing) == len(sub_sample["images"]):
            err = {"id": sid, "patient_id": patient_id, "error": f"No PNGs found (missing all {len(missing)} files)"}
            async with file_lock:
                await asyncio.to_thread(append_jsonl, err, out_path)
            return
    else:
        # Legacy schema
        for tp in sub_sample.get("timepoints", []):
            image_id = tp.get("image_id", "")
            img = load_png(sample, root, image_id) 
            if img is None:
                missing.append(image_id)
            else:
                images.append(img)
    
        if not images:
            err = {"id": sid, "patient_id": sample.get("patient_id"), "error": f"No PNGs found (missing={missing})"}
            async with file_lock:
                await asyncio.to_thread(append_jsonl, err, out_path)
            return

    # Build content parts and messages
    # NOTE: Pass the sub_sample, not the original sample
    content_parts = build_case_content_parts(sub_sample, images, root=root)
    cmessages = parts_to_chat_messages(content_parts)

    used_api = "chat"
    used_schema = True
    decoded = ""
    cresp = None

    try:
        # Strict JSON Schema (Structured Outputs)
        cresp = await aclient.chat.completions.create(
            model=model,
            messages=cmessages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "neuro_interval_change",
                    # "strict": True,
                    "schema": OUTPUT_JSON_SCHEMA,
                },
            },
            max_tokens=max_output_tokens, 
        )
        decoded = decode_chat_text(cresp)
    except Exception as e:
        # Better diagnostics for 400s
        try:
            # Import dynamically in case groq isn't installed
            from groq import BadRequestError
            if isinstance(e, BadRequestError):
                logging.error(f"id={sid}: 400 Bad Request: {getattr(e, 'message', e)}")
        except Exception:
            pass

        # Fallback: no schema (still JSON-ish via your extractor)
        logging.warning(f"id={sid}: Schema call failed ({e}). Falling back to plain text.")
        used_schema = False
        cresp = await aclient.chat.completions.create(
            model=model,
            messages=cmessages,
            max_tokens=max_output_tokens, 
        )
        decoded = decode_chat_text(cresp)

    parsed = extract_json_obj(decoded)
    print(f"✅ Done sample {sid}")
    result = {
        "id": sid,
        "patient_id": sample.get("patient_id"),
        "raw_text": decoded,
        "parsed": parsed if isinstance(parsed, dict) else None,
        "valid_json": isinstance(parsed, dict),
        "used_schema": used_schema,
        "used_api": used_api,
        "model": getattr(cresp, "model", model),
        "response_id": getattr(cresp, "id", None),
        "latency_s": round(time.time() - t0, 3),
    }


    if result["valid_json"]:
        result["steps"] = parsed.get("steps")
        result["answer"] = parsed.get("answer")
        if "answer_key" in parsed:
            result["answer_key"] = parsed["answer_key"]
        if "answer_option" in parsed:
            result["answer_option"] = parsed["answer_option"]

    # Persist immediately
    async with file_lock:
        await asyncio.to_thread(append_jsonl, result, out_path)

    # Mark as done if valid
    if result.get("valid_json"):
        async with done_lock:
            done_ids.add(sid)

# =========================
# Main (async + parallel)
# =========================

async def amain(args):
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    samples = read_samples(args.samples)
    logging.info(f"Loaded {len(samples)} samples.")

    # Clean the existing output file: keep only valid_json==True and sort by samples order
    done_ids = clean_and_sort_output(args.out, samples, keep_parse_fail=False)
    if done_ids:
        logging.info(f"After cleanup: {len(done_ids)} completed (valid) IDs kept in {args.out}")
    else:
        logging.info("After cleanup: no completed IDs found or file did not exist.")

    # Filter to remaining
    queue: List[Dict[str, Any]] = []
    for i, s in enumerate(samples, 1):
        sid = s.get("id", f"case_{i}")
        if sid in done_ids:
            logging.info(f"SKIP id={sid} (already completed)")
            continue
        # Ensure every sample has an ID for resume tracking
        if "id" not in s:
            s = dict(s)
            s["id"] = sid
        queue.append(s)

    if not queue:
        logging.info("Nothing to do. All samples completed.")
        return

    sem = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()
    done_lock = asyncio.Lock()

    async def bounded_worker(sample):
        async with sem:
            try:
                # Use a factory for the backoff wrapper
                coro_factory = lambda: process_one(
                    sample=sample,
                    root=args.root,
                    model=args.model,
                    max_output_tokens=args.max_output_tokens,
                    out_path=args.out,
                    file_lock=file_lock,
                    done_ids=done_ids,
                    done_lock=done_lock,
                    use_schema_first=True
                )
                await call_with_backoff(coro_factory)
                
            except Exception as e:
                # Persist the error immediately
                err_rec = {"id": sample.get("id"), "patient_id": sample.get("patient_id"), "error": str(e)}
                async with file_lock:
                    await asyncio.to_thread(append_jsonl, err_rec, args.out)
                logging.exception(f"Worker error for id={sample.get('id')}: {e}")

    t_start = time.time()
    await asyncio.gather(*(bounded_worker(s) for s in queue))
    logging.info(f"Done. processed={len(queue)}, elapsed={time.time()-t_start:.1f}s, out={args.out}")

def main():
    parser = argparse.ArgumentParser(description="Groq API + Vision (async parallel, resume-safe)")
    parser.add_argument("--samples", required=True, help="Path to samples (.json or .jsonl).")
    parser.add_argument("--root", required=True, help="Root directory that contains dataset folders.")
    parser.add_argument("--out", default="outputs/groq_steps.jsonl", help="Output JSONL (appends + resume).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model (e.g., meta-llama/llama-4-maverick-17b-128e-instruct).")
    parser.add_argument("--max-output-tokens", "--max-tokens", dest="max_output_tokens",
                        type=int, default=DEFAULT_MAX_OUT, help="Max output tokens per response.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Number of concurrent API calls.")
    parser.add_argument("--resume-all", action="store_true",
                        help="Treat any existing line (even parse failures) as completed; skip them.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # Run event loop
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

if __name__ == "__main__":
    main()