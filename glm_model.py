#!/usr/bin/env python3
"""
Neuroradiology interval-change runner (ZhipuAI GLM API)
- Works for GLM-4V, GLM-4.5V, GLM-5V via OpenAI-compatible API
"""

import os
import io
import json
import base64
import argparse
import logging
import textwrap
import time
import asyncio
from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont
from openai import AsyncOpenAI

DEFAULT_MODEL = "glm-4v" # Can override with glm-4.5v or glm-5v

# ZhipuAI uses an OpenAI-compatible endpoint
aclient = AsyncOpenAI(
    api_key=os.environ.get("ZHIPUAI_API_KEY", ""),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

SYSTEM_PROMPT = "You are a neuroradiologist analyzing longitudinal brain MRIs. Output strict JSON with 'steps' and 'answer'."

def create_grid_for_timepoint(root: str, patient_id: str, tp_images: List[Dict]) -> Image.Image:
    cell_size = 256
    padding = 20
    modalities = sorted(list(set(img.get('sequence', '') for img in tp_images)))
    views = sorted(list(set(img.get('view', '') for img in tp_images)))
    
    grid_w = len(views) * cell_size + (len(views) - 1) * padding
    grid_h = len(modalities) * cell_size + (len(modalities) - 1) * padding
    
    grid_img = Image.new('RGB', (grid_w, grid_h), color=(0, 0, 0))
    try: font = ImageFont.truetype("LiberationSans-Regular.ttf", 20)
    except: font = ImageFont.load_default()
    draw = ImageDraw.Draw(grid_img)
    
    for img_meta in tp_images:
        row = modalities.index(img_meta.get('sequence', ''))
        col = views.index(img_meta.get('view', ''))
        path = os.path.join(root, patient_id, f"timepoint_{img_meta.get('timepoint', '')}", img_meta.get('filename', ''))
        if os.path.exists(path):
            try:
                img = Image.open(path).convert('RGB')
                img.thumbnail((cell_size, cell_size)) 
                x, y = col * (cell_size + padding), row * (cell_size + padding)
                grid_img.paste(img, (x, y))
                draw.text((x + 5, y + 5), f"{img_meta.get('sequence', '').upper()} {img_meta.get('view', '').capitalize()}", fill=(255, 255, 0), font=font) 
            except Exception as e: logging.warning(f"Failed to load {path}: {e}")
    return grid_img

def build_parts(sample: Dict, root: str) -> List[Dict]:
    parts = [{"type": "text", "text": f"Question: {sample.get('question',[''])[0]}\\nNeed JSON: {{\"steps\": [], \"answer\": \"\"}}"}]
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
            # GLM / ZhipuAI accepts URL base64 format identical to OpenAI
            b64_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            parts.append({"type": "image_url", "image_url": {"url": b64_url}})
    return parts

async def process_one(sample: Dict, root: str, model: str, out_path: str, sem: asyncio.Semaphore, flock: asyncio.Lock):
    async with sem:
        sid = sample.get("id")
        t0 = time.time()
        try:
            parts = build_parts(sample, root)
            resp = await aclient.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": parts}],
                max_tokens=500
            )
            raw = resp.choices[0].message.content
            parsed = None
            try:
                start, end = raw.find('{'), raw.rfind('}') + 1
                parsed = json.loads(raw[start:end])
            except: pass
            
            res = {"id": sid, "raw_text": raw, "parsed": parsed, "valid_json": isinstance(parsed, dict), "model": model, "latency": time.time()-t0}
            if res["valid_json"]: res.update(parsed)
            
            async with flock:
                with open(out_path, "a") as f: f.write(json.dumps(res) + "\\n")
            logging.info(f"Done {sid}")
        except Exception as e:
            async with flock:
                with open(out_path, "a") as f: f.write(json.dumps({"id": sid, "error": str(e)}) + "\\n")

async def main(args):
    with open(args.samples) as f: samples = [json.loads(line) for line in f if line.strip()]
    sem = asyncio.Semaphore(4)
    flock = asyncio.Lock()
    await asyncio.gather(*(process_one(s, args.root, args.model, args.out, sem, flock) for s in samples))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="outputs/glm_steps.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    logging.basicConfig(level="INFO")
    asyncio.run(main(parser.parse_args()))
