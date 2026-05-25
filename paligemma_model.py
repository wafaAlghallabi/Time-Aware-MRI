#!/usr/bin/env python3
"""
Neuroradiology interval-change runner (PaliGemma / MedGemma)
- HuggingFace implementation for PaliGemma 2 mix / MedGemma 1.5 4B Multimodal
"""

import os
import json
import argparse
import logging
import time
from typing import List, Dict

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForImageTextToText

DEFAULT_MODEL = "google/paligemma-2-mix-10b"

SYSTEM_PROMPT = "Neuroradiologist: compare MRIs. Output strict JSON with 'steps' and 'answer'."

def resolve_image_path(root_dir: str, dataset: str, patient_id: str, img_meta: Dict) -> str:
    if dataset == "Yale-BrainMets": return os.path.join(root_dir, "Yale_multiview_slices", img_meta.get('path', ''))
    elif dataset == "UCSF-GBM": return os.path.join(root_dir, "UCSF_seg_multiview_slices", str(patient_id), f"timepoint_{img_meta.get('timepoint', '')}", img_meta.get('filename', ''))
    elif dataset == "Lumiere-BrainMets": return os.path.join(root_dir, "lumiere_multiview_slices", img_meta.get('path', ''))
    elif dataset == "UCSD-PTGBM": return os.path.join(root_dir, "UCSD_PTGBM_multiview_slices", img_meta.get('path', ''))
    elif dataset in ["OASIS-2", "RHUH-GBM"]:
        paths = [os.path.join(root_dir, img_meta.get('path', '')), os.path.join(root_dir, dataset.replace("-2", ""), img_meta.get('path', '')), os.path.join(root_dir, "Yale_multiview_slices", img_meta.get('path', ''))]
        for p in paths:
            if os.path.exists(p): return p
        return paths[0]
    return ""

def create_grid_for_timepoint(root: str, patient_id: str, tp_images: List[Dict], dataset: str) -> Image.Image:
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
        path = resolve_image_path(root, dataset, patient_id, img_meta)
        if os.path.exists(path):
            try:
                img = Image.open(path).convert('RGB')
                img.thumbnail((cell_size, cell_size)) 
                x, y = col * (cell_size + padding), row * (cell_size + padding)
                grid_img.paste(img, (x, y))
                draw.text((x + 5, y + 5), f"{img_meta.get('sequence', '').upper()} {img_meta.get('view', '').capitalize()}", fill=(255, 255, 0), font=font) 
            except Exception as e: logging.warning(f"Failed to load {path}: {e}")
    return grid_img

def load_done_ids(path: str) -> set:
    """Load already-completed qa_ids from output file for resume support."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    qid = rec.get("qa_id")
                    if qid and rec.get("valid_json"):
                        done.add(qid)
                except json.JSONDecodeError:
                    continue
    return done

def main(args):
    logging.basicConfig(level="INFO")
    processor = AutoProcessor.from_pretrained(args.model, token=os.environ.get('HF_TOKEN', True), trust_remote_code=True)
    
    if getattr(args, "quantize", None) == "4bit":
        from transformers import BitsAndBytesConfig
        q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(args.model, quantization_config=q_config, device_map="auto", token=os.environ.get('HF_TOKEN', True), trust_remote_code=True).eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, token=os.environ.get('HF_TOKEN', True), trust_remote_code=True).cuda().eval()

    if args.samples.endswith(".json"):
        with open(args.samples) as f:
            samples = json.load(f)
    else:
        with open(args.samples) as f:
            samples = [json.loads(line) for line in f if line.strip()]

    # Resume support: skip already completed samples
    done_ids = load_done_ids(args.out)
    if done_ids:
        logging.info(f"Resume: {len(done_ids)} samples already completed, skipping them.")

    for sample in samples:
        sid = sample.get("qa_id") or sample.get("id")
        patient_id = str(sample.get("patient_id", ""))

        if sid in done_ids:
            logging.info(f"SKIP {sid} (already completed)")
            continue

        t0 = time.time()
        pil_images = []
        question = sample.get('question', '')
        if isinstance(question, list):
            question = question[0] if question else ''
        text_prompt = f"{SYSTEM_PROMPT}\nQuestion: {question}\nJSON Format: {{\"steps\": [], \"answer\": \"\"}}\n"
            
        # Build chat-template content list with proper {type: image} entries
        # Gemma3Processor requires <start_of_image> boi markers (inserted by apply_chat_template)
        content = []
        if "images" in sample and sample["images"]:
            tp_images = {}
            for img_meta in sample["images"]:
                tp = img_meta.get("timepoint")
                if tp not in tp_images: tp_images[tp] = []
                tp_images[tp].append(img_meta)
            
            for tp in sorted(tp_images.keys()):
                grid = create_grid_for_timepoint(args.root, patient_id, tp_images[tp], sample.get('dataset', ''))
                pil_images.append(grid)
                content.append({"type": "image"})
                
        content.append({"type": "text", "text": text_prompt})
        messages = [{"role": "user", "content": content}]
                
        try:
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            # Images must be passed as nested list [[img1, img2, ...]] for Gemma3Processor
            inputs = processor(text=prompt, images=[pil_images] if pil_images else None, return_tensors="pt", padding=True).to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=500, do_sample=False)
            # Decode only the newly generated tokens (skip the input prompt tokens)
            input_len = inputs["input_ids"].shape[-1]
            response = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            
            parsed = None
            try:
                start, end = response.find('{'), response.rfind('}') + 1
                parsed = json.loads(response[start:end])
            except: pass
            
            # Unified output schema
            steps = parsed.get("steps", []) if isinstance(parsed, dict) else []
            answer = parsed.get("answer", "") if isinstance(parsed, dict) else ""
            res = {
                "qa_id": sid,
                "patient_id": patient_id,
                "model": args.model,
                "raw_text": response,
                "steps": steps,
                "answer": answer,
                "valid_json": isinstance(parsed, dict),
                "latency_s": round(time.time() - t0, 3),
            }
            
            with open(args.out, "a") as f: f.write(json.dumps(res) + "\n")
            logging.info(f"Done {sid}")
        except Exception as e:
            err = {
                "qa_id": sid,
                "patient_id": patient_id,
                "model": args.model,
                "raw_text": "",
                "steps": [],
                "answer": "",
                "valid_json": False,
                "latency_s": round(time.time() - t0, 3),
                "error": str(e),
            }
            with open(args.out, "a") as f: f.write(json.dumps(err) + "\n")
            logging.error(f"Error {sid}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="outputs/steps.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quantize", choices=["4bit", "none"], default="none")
    main(parser.parse_args())
