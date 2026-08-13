#!/usr/bin/env python3
"""
Neuroradiology interval-change runner (Qwen3-VL Local HuggingFace)
"""

import os
import json
import argparse
import logging
import time
from typing import List, Dict

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL = "/share/softwares/maaz/checkpoints/Qwen2.5-VL-3B-Instruct"

SYSTEM_PROMPT = "You are a neuroradiologist. Compare the MRI timepoints and provide step-by-step reasoning and a final conclusion. Output strictly JSON with 'steps' and 'answer'."

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

def main(args):
    logging.basicConfig(level="INFO")
    
    logging.info(f"Loading processor for {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    logging.info(f"Loading model {args.model} on CUDA in 4-bit...")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    # device_map="auto" handles moving the model to appropriate devices
    model.eval()

    with open(args.samples) as f: 
        data = json.load(f)
        samples = data if isinstance(data, list) else [data]
        
    for i, sample in enumerate(samples, 1):
        sid = sample.get("id", f"case_{i}")
        t0 = time.time()
        
        patient_id = str(sample.get("patient_id", ""))
        
        # Build Qwen Messages
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ]
        
        user_content = []
        user_content.append({"type": "text", "text": f"Question: {sample.get('question',[''])[0]}\\nNeed JSON format matching this structure: {{\"steps\": [], \"answer\": \"\"}}\\n"})
        
        if "images" in sample and sample["images"]:
            tp_images = {}
            for img_meta in sample["images"]:
                tp = img_meta.get("timepoint")
                if tp not in tp_images: tp_images[tp] = []
                tp_images[tp].append(img_meta)
                
            for tp in sorted(tp_images.keys()):
                user_content.append({"type": "text", "text": f"--- Timepoint {tp} Grid ---"})
                grid = create_grid_for_timepoint(args.root, patient_id, tp_images[tp], sample.get('dataset', ''))
                # Save temp to disk or buffer? qwen_vl_utils prefers path
                tmp_path = f"/tmp/qwen_{sid}_{tp}.png"
                grid.save(tmp_path)
                user_content.append({"type": "image", "image": tmp_path})
                
        messages.append({"role": "user", "content": user_content})
                
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to("cuda:0")
            
            generated_ids = model.generate(**inputs, max_new_tokens=500)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            parsed = None
            try:
                start, end = response.find('{'), response.rfind('}') + 1
                parsed = json.loads(response[start:end])
            except: pass
            
            res = {"id": sid, "raw_text": response, "parsed": parsed, "valid_json": isinstance(parsed, dict), "model": args.model, "latency": time.time()-t0}
            if res["valid_json"]: res.update(parsed)
            import os
            with open(args.out, "a") as f: f.write(json.dumps(res) + os.linesep)
            logging.info(f"Done {sid}")
            
            # cleanup
            for c in user_content:
                if c["type"] == "image" and os.path.exists(c["image"]): os.remove(c["image"])
                
        except Exception as e:
            with open(args.out, "a") as f: f.write(json.dumps({"id": sid, "error": str(e)}) + os.linesep)
            logging.error(f"Error {sid}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="outputs/qwen3_steps.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    main(parser.parse_args())
