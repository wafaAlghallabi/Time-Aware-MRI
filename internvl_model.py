#!/usr/bin/env python3
"""
Neuroradiology interval-change runner (InternVL 3/3.5)
- Uses HuggingFace Transformers for open-weight models like InternVL3-8B / 78B
"""

import os
import json
import argparse
import logging
import time
from typing import List, Dict

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    best_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
        
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

DEFAULT_MODEL = "OpenGVLab/InternVL3-8B"

SYSTEM_PROMPT = "You are a neuroradiologist. Compare the MRI timepoints and provide step-by-step reasoning and a final conclusion. Output strictly JSON with 'steps' and 'answer'."

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
    
    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModel.from_pretrained(args.model, quantization_config=quantization_config, trust_remote_code=True).eval()

    with open(args.samples) as f: 
        data = json.load(f)
        samples = data if isinstance(data, list) else [data]

    # Resume support: skip already completed samples
    done_ids = load_done_ids(args.out)
    if done_ids:
        logging.info(f"Resume: {len(done_ids)} samples already completed, skipping them.")

    for sample in samples:
        torch.cuda.empty_cache()
        sid = sample.get("qa_id") or sample.get("id")
        patient_id = str(sample.get("patient_id", ""))

        if sid in done_ids:
            logging.info(f"SKIP {sid} (already completed)")
            continue

        t0 = time.time()
        pixel_values_list = []
        num_patches_list = []
        question = sample.get('question', '')
        if isinstance(question, list):
            question = question[0] if question else ''
        text_prompt = f"{SYSTEM_PROMPT}\nQuestion: {question}\nNeed JSON: {{\"steps\": [], \"answer\": \"\"}}\n"
            
        if "images" in sample and sample["images"]:
            tp_images = {}
            for img_meta in sample["images"]:
                tp = img_meta.get("timepoint")
                if tp not in tp_images: tp_images[tp] = []
                tp_images[tp].append(img_meta)
                
            for tp in sorted(tp_images.keys()):
                text_prompt += f"--- Timepoint {tp} Grid ---\n<image>\n"
                grid = create_grid_for_timepoint(args.root, patient_id, tp_images[tp])
                
                # InternVL requires dynamic patching of the PIL image into a tensor
                transform = build_transform(input_size=448)
                # clamp max_num to 2 to prevent CUDA OOM on A100 (40GB) with 8-bit model
                # Remove restriction to allow native InternVL3 patching
                images = dynamic_preprocess(grid, image_size=448, use_thumbnail=True)
                pixel_values = [transform(image) for image in images]
                pixel_values = torch.stack(pixel_values)
                pixel_values_list.append(pixel_values)
                num_patches_list.append(pixel_values.size(0))
                
        try:
            # InternVL specific chat logic
            if len(pixel_values_list) > 0:
                pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.float16).cuda()
            else:
                pixel_values = None
                
            response = model.chat(
                tokenizer, 
                pixel_values=pixel_values, 
                question=text_prompt, 
                generation_config={"max_new_tokens": 500, "do_sample": False}, 
                num_patches_list=num_patches_list
            )
            
            parsed = None
            try:
                start, end = response.find('{'), response.rfind('}') + 1
                parsed = json.loads(response[start:end])
            except: pass
            
            # Unified output schema - flatten steps if they are dicts
            raw_steps = parsed.get("steps", []) if isinstance(parsed, dict) else []
            steps = []
            for s in raw_steps:
                if isinstance(s, str):
                    steps.append(s)
                elif isinstance(s, dict):
                    steps.append(s.get("action") or s.get("description") or s.get("text") or str(s))
                else:
                    steps.append(str(s))
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
    parser.add_argument("--out", default="outputs/internvl_steps.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    main(parser.parse_args())
