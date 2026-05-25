#!/usr/bin/env python3
"""
Agentic Neuroradiology Pipeline (Resident + Attending)
Experiment 2 for MICCAI submission.

Supports routing to:
- OpenAI (gpt-4o, gpt-5)
- Gemini (gemini-3-pro-exp, gemini-3.1-pro-preview, etc.)
- Open-Source VLMs via OpenAI-compatible endpoints (qwen, internvl, medgemma, kimi)
"""

import os
import json
import base64
import argparse
import logging
import textwrap
import time
import asyncio
import io
import re
import traceback

# Bypass AsyncOpenAI initialization error in imports
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-dummy"
from typing import List, Dict, Any, Tuple, Optional

from openai import AsyncOpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Import image grid builders from the original openai_model script
# Assuming running from the root MRI-Res-Benchmark directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from steps_generation.openai_model import (
    to_data_url, parse_options, _format_timepoints, extract_json_obj
)


# =========================
# Rate Limiting
# =========================
class RPMLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0
        self.last_call = 0
        self.lock = asyncio.Lock()

    async def wait(self):
        if self.interval <= 0:
            return
        async with self.lock:
            now = time.time()
            wait_time = self.last_call + self.interval - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call = time.time()


# =========================
# Unified Dataset Helper
# =========================
def get_unified_dataset_folder(dataset_name: str) -> str:
    """Maps dataset names in the JSON to their folder names in data_unified."""
    d = dataset_name.lower()
    if "lumiere" in d: return "lumiere_multiview_slices"
    if "yale" in d: return "Yale_multiview_slices"
    if "ucsf" in d: return "UCSF_seg_multiview_slices"
    if "rhuh" in d: return "RHUH-GBM_slices_clean"
    if "ucsd" in d: return "UCSD_PTGBM_multiview_slices"
    if "oasis" in d: return "OASIS_slices_clean"
    return "UCSF_seg_multiview_slices" # Fallback

def create_grid_for_timepoint_unified(root: str, dataset_name: str, patient_id: str, tp_images: List[Dict]) -> Image.Image:
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
    
    folder = get_unified_dataset_folder(dataset_name)
    
    for img_meta in tp_images:
        row = modalities.index(img_meta.get('sequence', ''))
        col = views.index(img_meta.get('view', ''))
        
        path = os.path.join(root, folder, patient_id, f"timepoint_{img_meta.get('timepoint', '')}", img_meta.get('filename', ''))
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

DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_OUT = 1000
DEFAULT_CONCURRENCY = 10

# =========================
# Prompts
# =========================

RESIDENT_SYSTEM_PROMPT = textwrap.dedent("""
You are a Neuroradiology Resident.
Your job is to analyze the provided multi-timepoint brain MRI scans by following the clinical protocol assigned by your attending.
You must NOT attempt to formulate a final conclusion or answer a multiple-choice question.
Output a concise structured JSON report with your clinical findings. Be brief but precise.
""").strip()

ATTENDING_SYSTEM_PROMPT = textwrap.dedent("""
You are a Board-Certified Attending Neuroradiologist.
Your resident has reviewed the cases and provided a detailed clinical finding report based on a strict spatial protocol.
Your job is to read their report, review the patient metadata, and select the correct multiple-choice answer regarding the interval change.
You must output your final reasoning and answer in the strict JSON format requested.
""").strip()

# =========================
# Builders (OpenAI / Local VLM)
# =========================

def build_resident_parts_for_local(sample: Dict[str, Any], root: str) -> List[Dict[str, Any]]:
    parts = []
    ctx = sample.get("clinical_context", {})
    age = ctx.get("age", sample.get("age", ""))
    sex = ctx.get("sex", sample.get("sex", ""))
    dataset = sample.get("dataset", "")
    
    guid_raw = sample.get("segmentation_guidance", "")
    if isinstance(guid_raw, list):
        guidance = "\n".join(guid_raw)
    else:
        guidance = str(guid_raw) if guid_raw else ""
    
    # If no segmentation guidance, use the question as the clinical task
    if not guidance.strip():
        q = sample.get("question", "")
        question_text = q[0] if isinstance(q, list) and len(q) > 0 else (q if isinstance(q, str) else "")
        options = sample.get("options", [])
        opts = parse_options(options) if options else []
        opts_block = "\n".join([f"{k}. {v}" for k, v in opts]) if opts else ""
        guidance = f"Analyze the provided MRI scans and answer the following question:\n{question_text}"
        if opts_block:
            guidance += f"\n\nOptions:\n{opts_block}"
    
    json_schema = textwrap.dedent("""
    {
        "finding_steps": ["Finding 1...", "Finding 2..."],
        "maximal_change_summary": "Summary..."
    }
    """).strip()
    
    user_text = textwrap.dedent(f"""
    Task: Execute the following clinical protocol on the provided MRI scans. 
    
    Patient metadata:
    - Age: {age}
    - Sex: {sex}
    - Dataset: {dataset}
    
    CLINICAL PROTOCOL (Follow this exactly):
    {guidance}
    
    You must output exactly valid JSON matching this structure:
    {json_schema}
    """).strip()
    
    parts.append({"type": "input_text", "text": user_text})
    
    if "images" in sample and isinstance(sample["images"], list) and len(sample["images"]) > 0:
        tp_images = {}
        for img_meta in sample["images"]:
            tp = img_meta.get("timepoint")
            if tp not in tp_images: tp_images[tp] = []
            tp_images[tp].append(img_meta)
            
        patient_id = str(sample.get("patient_id", ""))
        for tp in sorted(tp_images.keys()):
            parts.append({"type": "input_text", "text": f"--- Timepoint {tp} Grid ---"})
            grid = create_grid_for_timepoint_unified(root, dataset, patient_id, tp_images[tp])
            parts.append({
                "type": "input_image", 
                "image_url": {"url": to_data_url(grid)}
            })
            
    return parts

def build_attending_parts_for_local(sample: Dict[str, Any], root: str, resident_report: Any) -> List[Dict[str, Any]]:
    parts = []
    q = sample.get("question", "")
    question_text = q[0] if isinstance(q, list) and len(q) > 0 else (q if isinstance(q, str) else "")
    options = sample.get("options", [])
    opts = parse_options(options)
    opts_block = "\n".join([f"{k}. {v}" for k, v in opts])
    
    json_schema = textwrap.dedent("""
    {
      "steps": [
        "Step 1...",
        "Step 2...",
        "Final step..."
      ],
      "answer": "A",
      "answer_key": "A",
      "answer_option": "Exact text of chosen option"
    }
    """).strip()

    # Format resident report for the prompt
    if isinstance(resident_report, dict):
        resident_report_str = json.dumps(resident_report, indent=2)
    else:
        resident_report_str = str(resident_report)
    
    user_text = textwrap.dedent(f"""
    Task: Answer the clinical question based on your resident's report.
    
    Resident's Report:
    {resident_report_str}
    
    Question:
    {question_text}
    
    Options:
    {opts_block}
    
    Your output must be valid JSON matching exactly this structre:
    {json_schema}
    """).strip()
    
    parts.append({"type": "input_text", "text": user_text})
    return parts

def parts_to_chat_messages(parts, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    user_content = []
    for p in parts:
        if p["type"] == "input_text":
            user_content.append({"type": "text", "text": p["text"]})
        elif p["type"] == "input_image":
            img_url_data = p.get("image_url")
            if isinstance(img_url_data, dict):
                url = img_url_data.get("url")
            else:
                url = img_url_data
            user_content.append({"type": "image_url", "image_url": {"url": url}})
    messages.append({"role": "user", "content": user_content})
    return messages

# =========================
# Builders (Gemini)
# =========================
def build_resident_parts_gemini(sample: Dict[str, Any], root: str) -> List[Any]:
    ctx = sample.get("clinical_context", {})
    age = ctx.get("age", sample.get("age", ""))
    sex = ctx.get("sex", sample.get("sex", ""))
    dataset = sample.get("dataset", "")
    
    guid_raw = sample.get("segmentation_guidance", "")
    if isinstance(guid_raw, list):
        guidance = "\n".join(guid_raw)
    else:
        guidance = str(guid_raw) if guid_raw else ""
    
    # If no segmentation guidance, use the question as the clinical task
    if not guidance.strip():
        q = sample.get("question", "")
        question_text = q[0] if isinstance(q, list) and len(q) > 0 else (q if isinstance(q, str) else "")
        options = sample.get("options", [])
        opts = parse_options(options) if options else []
        opts_block = "\n".join([f"{k}. {v}" for k, v in opts]) if opts else ""
        guidance = f"Analyze the provided MRI scans and answer the following question:\n{question_text}"
        if opts_block:
            guidance += f"\n\nOptions:\n{opts_block}"
    
    json_schema = textwrap.dedent("""
    {
        "finding_steps": ["Finding 1...", "Finding 2..."],
        "maximal_change_summary": "Summary..."
    }
    """).strip()
    
    user_text = textwrap.dedent(f"""
    Task: Execute the following clinical protocol on the provided MRI scans. 
    Patient metadata: Age {age}, Sex {sex}, Dataset {dataset}
    CLINICAL PROTOCOL: {guidance}
    You must output strictly JSON matching this structure:
    {json_schema}
    """).strip()
    
    parts = [user_text]
    
    if "images" in sample and isinstance(sample["images"], list) and len(sample["images"]) > 0:
        tp_images = {}
        for img_meta in sample["images"]:
            tp = img_meta.get("timepoint")
            if tp not in tp_images: tp_images[tp] = []
            tp_images[tp].append(img_meta)
            
        patient_id = str(sample.get("patient_id", ""))
        for tp in sorted(tp_images.keys()):
            parts.append(f"\n--- Timepoint {tp} Grid ---")
            grid = create_grid_for_timepoint_unified(root, dataset, patient_id, tp_images[tp])
            parts.append(grid)
            
    return parts

def build_attending_parts_gemini(sample: Dict[str, Any], resident_report: Any) -> List[Any]:
    q = sample.get("question", "")
    question_text = q[0] if isinstance(q, list) and len(q) > 0 else (q if isinstance(q, str) else "")
    options = sample.get("options", [])
    opts = parse_options(options)
    opts_block = "\n".join([f"{k}. {v}" for k, v in opts])
    
    # Ensure resident_report is a clean string
    res_text = resident_report
    if isinstance(resident_report, dict):
        res_text = json.dumps(resident_report, indent=2)
    
    json_schema = textwrap.dedent("""
    {
      "steps": ["Step 1...", "Step 2..."],
      "answer": "A",
      "answer_key": "A",
      "answer_option": "Exact text of chosen option"
    }
    """).strip()
    
    user_text = textwrap.dedent(f"""
    Task: Answer the clinical question based on your resident's report.
    Resident's Report:
    {res_text}
    Question: {question_text}
    Options:
    {opts_block}
    Your output must be strictly valid JSON matching exactly this structre:
    {json_schema}
    """).strip()
    
    return [user_text]

# =========================
# Worker Logic
# =========================

def read_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def append_jsonl(record: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def call_openai_like(messages: list, model: str, max_out: int, aclient: AsyncOpenAI, response_format: Optional[dict] = None) -> str:
    # Reasoning models like o1/o3 (gpt-5) and o4-mini use internal CoT that
    # consumes the output token budget. Give them generous limits.
    effective_max = max_out
    if "gpt-5" in model.lower() or "o1" in model.lower() or "o3" in model.lower():
        effective_max = max(max_out, 16384)
    elif "o4-mini" in model.lower():
        effective_max = max(max_out, 8192)
        
    kwargs = {"max_completion_tokens": effective_max}
    if response_format:
        kwargs["response_format"] = response_format
    
    cresp = await aclient.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    choice = cresp.choices[0]
    content = choice.message.content or ""
    refusal = getattr(choice.message, 'refusal', None)
    
    if not content:
        logging.warning(f"Empty content from {model}. Finish reason: {choice.finish_reason}. Refusal: {refusal}")
    
    return content


async def call_gemini(parts: list, model_obj: genai.GenerativeModel, max_out: int, sid: str = "unknown") -> str:
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_out,
        temperature=0.0,
        # response_mime_type="application/json", 
    )
    try:
        # Increase timeout to 300s to avoid truncation on slow responses
        response = await model_obj.generate_content_async(
            parts,
            generation_config=generation_config,
            request_options={"timeout": 300}
        )
        # Check for candidates and finish reason
        if not response.candidates:
            logging.warning(f"[{sid}] Gemini returned NO CANDIDATES.")
            return ""
        
        finish_reason = response.candidates[0].finish_reason
        if finish_reason != 1: # 1 is STOP
            logging.warning(f"[{sid}] Gemini finish_reason: {finish_reason}. Safety: {response.candidates[0].safety_ratings}")
            
        return response.text
    except Exception as e:
        logging.error(f"[{sid}] Gemini API Call failed: {e}")
        return ""

async def process_one(sample: Dict[str, Any], root: str, model: str, max_out: int, out_path: str, sem: asyncio.Semaphore, file_lock: asyncio.Lock, clients: dict, local_manager=None, rpm_limiter=None):
    async with sem:
        sid = sample.get("id") or sample.get("qa_id", "unknown")

        t0 = time.time()
        
        try:
            resident_report_text = ""
            final_text = ""
            
            # --- GEMINI ROUTE ---
            if "gemini" in model.lower():
                genai_model_resident = genai.GenerativeModel(model, system_instruction=RESIDENT_SYSTEM_PROMPT)
                genai_model_attending = genai.GenerativeModel(model, system_instruction=ATTENDING_SYSTEM_PROMPT)
                
                # Gemini 3 previews use deep thinking which consumes output tokens
                effective_max = max_out
                if "gemini-3" in model.lower():
                    effective_max = max(max_out, 16384)
                
                logging.info(f"[{sid}] {model}: Starting Resident Stage...")
                res_parts = build_resident_parts_gemini(sample, root)
                
                if rpm_limiter: await rpm_limiter.wait()
                resident_report_text = await call_gemini(res_parts, genai_model_resident, effective_max, sid=sid)
                
                logging.info(f"[{sid}] Resident returned (len={len(resident_report_text)}): {resident_report_text[:200]}...")
                
                res_content = extract_json_obj(resident_report_text)
                logging.info(f"[{sid}] {model}: Starting Attending Stage...")
                att_parts = build_attending_parts_gemini(sample, res_content) # Corrected: removed 'root'
                
                if rpm_limiter: await rpm_limiter.wait()
                final_text = await call_gemini(att_parts, genai_model_attending, effective_max, sid=sid)

            # --- LOCAL TRANSFORMERS DIRECT ROUTE ---
            elif local_manager:
                logging.info(f"[{sid}] Local VLM: Starting Resident Stage...")
                res_parts = build_resident_parts_for_local(sample, root)
                resident_report_text = await local_manager.generate(res_parts, RESIDENT_SYSTEM_PROMPT, max_out)
                
                logging.info(f"[{sid}] Resident returned (len={len(resident_report_text)}): {resident_report_text[:200]}...")
                
                res_content = extract_json_obj(resident_report_text)
                logging.info(f"[{sid}] Local VLM: Starting Attending Stage...")
                att_parts = build_attending_parts_for_local(sample, root, res_content)
                final_text = await local_manager.generate(att_parts, ATTENDING_SYSTEM_PROMPT, max_out)

            # --- OPENAI / OPEN-SOURCE LOCAL ROUTE (via OpenAI-compatible API) ---
            else:
                aclient = clients['openai']
                
                logging.info(f"[{sid}] {model}: Starting Resident Stage...")
                res_parts = build_resident_parts_for_local(sample, root)
                res_msgs = parts_to_chat_messages(res_parts, RESIDENT_SYSTEM_PROMPT)
                
                if rpm_limiter: await rpm_limiter.wait()
                resident_report_text = await call_openai_like(res_msgs, model, max_out, aclient, response_format={"type": "json_object"})
                
                logging.info(f"[{sid}] Resident returned (len={len(resident_report_text)}): {resident_report_text[:200]}...")
                
                if not resident_report_text.strip():
                    logging.warning(f"[{sid}] Resident report is EMPTY. This might indicate a safety trigger or model refusal.")
                
                res_content = extract_json_obj(resident_report_text)
                
                logging.info(f"[{sid}] {model}: Starting Attending Stage...")
                att_parts = build_attending_parts_for_local(sample, root, res_content)
                att_msgs = parts_to_chat_messages(att_parts, ATTENDING_SYSTEM_PROMPT)
                
                final_text = await call_openai_like(att_msgs, model, max_out, aclient, response_format={"type": "json_object"})

            # Parse and Save Result
            parsed_res = extract_json_obj(resident_report_text)
            parsed_att = extract_json_obj(final_text)
            
            # If resident report is just a string but not JSON, we still want it
            res_content_for_output = parsed_res if isinstance(parsed_res, dict) else resident_report_text
            
            result = {
                "id": sid,
                "patient_id": sample.get("patient_id"),
                "agent1_resident_report": res_content_for_output,
                "raw_text": final_text,
                "parsed": parsed_att if isinstance(parsed_att, dict) else None,
                "valid_json": isinstance(parsed_att, dict),
                "model": model,
                "latency_s": round(time.time() - t0, 3)
            }
            if result["valid_json"]:
                result["steps"] = parsed_att.get("steps")
                result["answer"] = parsed_att.get("answer")
                if "answer_key" in parsed_att: result["answer_key"] = parsed_att["answer_key"]
                if "answer_option" in parsed_att: result["answer_option"] = parsed_att["answer_option"]

            async with file_lock:
                await asyncio.to_thread(append_jsonl, result, out_path)
            logging.info(f"✅ [{sid}] Completed Agentic Pipeline.")
            
        except Exception as e:
            err_rec = {"id": sid, "error": f"Pipeline Error: {str(e)}"}
            async with file_lock:
                await asyncio.to_thread(append_jsonl, err_rec, out_path)
            logging.error(f"❌ Error {sid}: {e}\n{traceback.format_exc()}")

# =========================
# Local VLM Manager
# =========================

class LocalVLMManager:
    def __init__(self, model_path, model_type="qwen", load_in_4bit=False, max_vram_gb=None):
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        
        self.model_path = model_path
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        bnb_config = None
        if load_in_4bit:
            logging.info("Enabling 4-bit quantization (BitsAndBytes)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

        max_memory = None
        if max_vram_gb:
            # Masked index is 0 if CUDA_VISIBLE_DEVICES is used
            max_memory = {0: f"{max_vram_gb}GiB"}
            logging.info(f"Limiting VRAM usage to {max_vram_gb}GiB on the primary visible device.")

        logging.info(f"LocalVLM: Loading {model_path} (type={model_type}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        if "qwen" in model_type.lower():
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True,
                quantization_config=bnb_config, max_memory=max_memory
            )
        elif "paligemma" in model_type.lower() or "medgemma" in model_type.lower():
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
                quantization_config=bnb_config, max_memory=max_memory
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True,
                quantization_config=bnb_config, max_memory=max_memory
            )
        self.model.eval()

    async def generate(self, parts: List[Dict[str, Any]], system_prompt: str, max_tokens: int) -> str:
        # Synchronous generation wrapped for async
        return await asyncio.to_thread(self._sync_generate, parts, system_prompt, max_tokens)

    def _sync_generate(self, parts: List[Dict[str, Any]], system_prompt: str, max_tokens: int) -> str:
        t_messages = []
        user_content_parts = []
        
        # For Qwen, some versions prefer system prompt merged into user
        if "qwen" in self.model_type.lower():
            user_content_parts.append({"type": "text", "text": f"<<System>>\n{system_prompt}\n<<End System>>\n\n"})
        else:
            t_messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        pil_images = [] # This list will hold PIL Image objects for models that take them separately
        
        # Convert the 'parts' (OpenAI-like format) into a format suitable for the model's chat template
        text_parts = []
        image_parts = []
        
        for p in parts:
            if p["type"] == "input_text":
                text_parts.append({"type": "text", "text": p["text"]})
            elif p["type"] == "input_image":
                url_field = p.get("image_url", {})
                if isinstance(url_field, dict):
                    url = url_field.get("url", "")
                else:
                    url = url_field # Fallback
                
                if "," in url:
                    b64_data = url.split(",")[1]
                    img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
                    pil_images.append(img)
                    if "qwen" in self.model_type.lower():
                        import uuid
                        tmp_path = f"/tmp/qwen_{uuid.uuid4()}.png"
                        img.save(tmp_path)
                        image_parts.append({"type": "image", "image": tmp_path})
                    else:
                        image_parts.append({"type": "image", "image": img})
        
        # Qwen often prefers images first
        user_content_parts.extend(image_parts)
        user_content_parts.extend(text_parts)
        
        t_messages.append({"role": "user", "content": user_content_parts})
        
        try:
            # Model specific prompt building
            if "qwen" in self.model_type.lower():
                from qwen_vl_utils import process_vision_info
                # For Qwen-VL, the `apply_chat_template` expects a specific structure,
                # and `process_vision_info` extracts the actual image objects.
                text = self.processor.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(t_messages)
                logging.info(f"Qwen Debug - Messages count: {len(t_messages)}, Image inputs count: {len(image_inputs) if image_inputs is not None else 0}")
                logging.info(f"Qwen Debug - Final Text contains vision tokens: {'<|vision_start|>' in text}")
                logging.info(f"Qwen Debug - Final Text sample: {text[:800]}...")
                
                # Some processors fail if images=None but text has no vision tokens, or vice versa
                kwargs = {"text": [text], "padding": True, "return_tensors": "pt"}
                if image_inputs: kwargs["images"] = image_inputs
                if video_inputs: kwargs["videos"] = video_inputs
                
                inputs = self.processor(**kwargs).to(self.device)

            elif "paligemma" in self.model_type.lower() or "medgemma" in self.model_type.lower():
                # For PaliGemma/MedGemma, the `apply_chat_template` expects a placeholder for images
                # and the actual PIL images are passed separately to the processor.
                # We need to create a version of t_messages where image content is just a placeholder.
                paligemma_t_messages = []
                for msg in t_messages:
                    if msg["role"] == "user":
                        paligemma_content = []
                        for part in msg["content"]:
                            if part["type"] == "image":
                                paligemma_content.append({"type": "image"}) # Placeholder
                            else:
                                paligemma_content.append(part)
                        paligemma_t_messages.append({"role": msg["role"], "content": paligemma_content})
                    else:
                        paligemma_t_messages.append(msg)

                prompt = self.processor.apply_chat_template(paligemma_t_messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=prompt, images=pil_images if pil_images else None, return_tensors="pt", padding=True).to(self.device)
            else:
                raise ValueError(f"Unsupported local model type: {self.model_type}")

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.1)
            
            input_len = inputs["input_ids"].shape[-1]
            response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            logging.info(f"Qwen Debug - Raw Response: {response[:1000]}")
            
            # Cleanup temp images
            if "qwen" in self.model_type.lower():
                for content in user_content_parts:
                    if content["type"] == "image" and isinstance(content["image"], str) and os.path.exists(content["image"]):
                        try: os.remove(content["image"])
                        except: pass
            
            return response
        except Exception as e:
            logging.error(f"Local generation failed: {e}")
            return f"{{\"error\": \"Local VLM generation failed: {str(e)}\"}}"

async def amain(args):
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")
    
    local_manager = None
    if args.local_path:
        local_manager = LocalVLMManager(args.local_path, args.local_type, load_in_4bit=args.load_in_4bit, max_vram_gb=args.max_vram)
    
    # Initialize Clients
    clients = {}
    if "gemini" in args.model.lower():
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY environment variable not set.")
            return
        genai.configure(api_key=api_key)
    elif not local_manager: # Only initialize OpenAI client if not using local_manager and not Gemini
        # Picks up OPENAI_API_KEY from environment
        base_url = getattr(args, 'openai_base_url', None)
        clients['openai'] = AsyncOpenAI(base_url=base_url) 

    samples = read_samples(args.samples)
    if args.dataset:
        samples = [s for s in samples if s.get('dataset') == args.dataset]
        logging.info(f"Filtered to dataset: {args.dataset}. Remaining total samples: {len(samples)}")

    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out, 'r') as f:
            for line in f:
                 try: 
                     data = json.loads(line)
                     if data.get('valid_json') and "error" not in data:
                         done_ids.add(data.get('id'))
                 except: pass

    # Partition queue for workers
    all_queue = [s for s in samples if (s.get('id') or s.get('qa_id', 'unknown')) not in done_ids]
    queue = []
    for i, s in enumerate(all_queue):
        if (i % args.num_workers) == args.worker_id:
            queue.append(s)
            
    if args.limit > 0:
        queue = queue[:args.limit]
            
    logging.info(f"Model: {args.model} | Worker {args.worker_id}/{args.num_workers} | Processing {len(queue)} samples (total remaining: {len(all_queue)}).")

    sem = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()
    rpm_limiter = RPMLimiter(args.rpm) if args.rpm > 0 else None
    
    tasks = [process_one(s, args.root, args.model, args.max_out, args.out, sem, file_lock, clients, local_manager, rpm_limiter) for s in queue]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic (Resident -> Attending) Workflow Multi-Provider")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="outputs/steps.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-out", type=int, default=DEFAULT_MAX_OUT)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--rpm", type=float, default=0, help="Limit requests per minute (0 = no limit)")
    parser.add_argument("--dataset", default=None, help="Filter samples by dataset name (e.g. UCSF-GBM)")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--openai-base-url", default=None, help="Base URL for OpenAI-compatible local models (e.g. vLLM)")
    parser.add_argument("--local-path", default=None, help="Path for direct local inference with HuggingFace models")
    parser.add_argument("--local-type", default="qwen", choices=["qwen", "paligemma", "medgemma"], help="Type of local model (e.g., qwen, paligemma, medgemma)")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for parallel processing (0-indexed)")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of workers for parallel processing")
    parser.add_argument("--limit", type=int, default=0, help="Limit total number of samples to process (0 = no limit)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load local model in 4-bit quantization")
    parser.add_argument("--no-clean", action="store_true", help="Skip the output file cleanup/sorting (for multi-worker safety)")
    parser.add_argument("--max-vram", type=float, default=None, help="Max VRAM to use in GiB per process")
    
    args = parser.parse_args()
    asyncio.run(amain(args))
