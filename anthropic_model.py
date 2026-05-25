#!/usr/bin/env python3
"""
Neuroradiology interval-change runner (Anthropic Claude API + Vision + Structured Outputs)
- Async + parallel processing with bounded concurrency
- Built to handle Long-MRI-Seg multimodal grids
"""

import os
import io
import json
import base64
import argparse
import logging
import textwrap
from typing import List, Dict, Any, Set
import time
import asyncio
import random

from PIL import Image, ImageDraw, ImageFont
import anthropic

# =========================
# Config
# =========================

DEFAULT_MODEL = "claude-3-5-sonnet-20241022" 
DEFAULT_MAX_OUT = 500  
DEFAULT_CONCURRENCY = 4  

aclient = anthropic.AsyncAnthropic()

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
- Output ONLY valid JSON matching the requested schema.
""").strip()

# =========================
# Grid Logic
# =========================
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
                logging.warning(f"Failed to load {path}: {e}")
    return grid_img

def read_samples(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith(("[", "{")):
            try:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
            except: pass
        out = []
        for line in f:
            line = line.strip()
            if line: out.append(json.loads(line))
        return out

def append_jsonl(record: Dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\\n")

# =========================
# Prompting
# =========================
def build_case_content_parts(sample: Dict, root: str) -> List[Dict]:
    parts = []
    text = f"Patient: {sample.get('age','')} {sample.get('sex','')}. Question: {sample.get('question',[''])[0]}\\n"
    text += "Output strictly JSON format:\\n{\\"steps\\": [\\"step1\\", \\"step2\\"], \\"answer\\": \\"short answer\\", \\"answer_key\\": \\"\\", \\"answer_option\\": \\"\\"}"
    
    parts.append({"type": "text", "text": text})
    
    patient_id = str(sample.get("patient_id", ""))
    if "images" in sample and sample["images"]:
        tp_images = {}
        for img_meta in sample["images"]:
            tp = img_meta.get("timepoint")
            if tp not in tp_images: tp_images[tp] = []
            tp_images[tp].append(img_meta)
            
        for tp in sorted(tp_images.keys()):
            parts.append({"type": "text", "text": f"--- Timepoint {tp} Grid ---"})
            grid = create_grid_for_timepoint(root, patient_id, tp_images[tp])
            
            buf = io.BytesIO()
            grid.save(buf, format="JPEG")
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64_img
                }
            })
    return parts

# =========================
# Worker
# =========================
async def process_one(sample: Dict, root: str, model: str, max_out: int, out_path: str, sem: asyncio.Semaphore, file_lock: asyncio.Lock):
    async with sem:
        sid = sample.get("id")
        t0 = time.time()
        
        try:
            content_parts = build_case_content_parts(sample, root)
            msg = await aclient.messages.create(
                model=model,
                max_tokens=max_out,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content_parts}]
            )
            raw_text = msg.content[0].text
            
            # Simple JSON extraction
            parsed = None
            try:
                start = raw_text.find('{')
                end = raw_text.rfind('}') + 1
                parsed = json.loads(raw_text[start:end])
            except: pass
            
            res = {
                "id": sid,
                "patient_id": sample.get("patient_id"),
                "raw_text": raw_text,
                "parsed": parsed,
                "valid_json": isinstance(parsed, dict),
                "model": model,
                "latency_s": round(time.time() - t0, 3)
            }
            if res["valid_json"]:
                res.update({k: parsed.get(k) for k in ["steps", "answer", "answer_key", "answer_option"]})
                
            async with file_lock:
                await asyncio.to_thread(append_jsonl, res, out_path)
            logging.info(f"Done {sid}")
            
        except Exception as e:
            async with file_lock:
                await asyncio.to_thread(append_jsonl, {"id": sid, "error": str(e)}, out_path)
            logging.error(f"Error {sid}: {e}")

async def main_async(args):
    samples = read_samples(args.samples)
    sem = asyncio.Semaphore(args.concurrency)
    flock = asyncio.Lock()
    tasks = [process_one(s, args.root, args.model, args.max_out, args.out, sem, flock) for s in samples]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="outputs/anthropic_steps.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-out", type=int, default=DEFAULT_MAX_OUT)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    logging.basicConfig(level="INFO")
    asyncio.run(main_async(parser.parse_args()))
