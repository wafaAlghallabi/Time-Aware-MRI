import os
import json
import logging
import argparse
import concurrent.futures
from typing import List, Dict, Any, Optional
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

PROMPT_TEMPLATE = """You are a medical evaluation assistant. Given a radiology question and a model's reasoning steps, your task is to identify which multiple-choice option (A, B, C, D, E, or F) the model's reasoning most strongly supports.

[QUESTION]
{question}

[OPTIONS]
{options}

[MODEL REASONING]
{reasoning}

Output exactly one capital letter matching one of the provided options. If no answer can be inferred, output 'N/A'.
"""

def extract_answer(client: OpenAI, question: str, options: List[str], reasoning: str) -> str:
    q_text = question[0] if isinstance(question, list) else question
    opts_text = [str(opt) for opt in options] if options else []
    prompt = PROMPT_TEMPLATE.format(
        question=q_text,
        options="\n".join(opts_text),
        reasoning=reasoning
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        ans = response.choices[0].message.content.strip().upper()
        # Clean up: find the first letter A-F
        import re
        match = re.search(r'\b([A-F])\b', ans)
        if match:
            return match.group(1)
        return "N/A"
    except Exception as e:
        logging.error(f"API Error: {e}")
        return "ERROR"

def process_file(args):
    model_path, gt_map, output_path, api_key = args
    client = OpenAI(api_key=api_key)
    
    results = []
    with open(model_path, "r") as f:
        model_data = [json.loads(line) for line in f]
    
    logging.info(f"Processing {len(model_data)} samples from {model_path}")
    
    def process_sample(item):
        qid = item.get("qa_id") or item.get("id")
        gt = gt_map.get(qid)
        if not gt:
            return item
        
        # We only care about MCQs for this extraction
        if not gt.get("options"):
            return item
            
        reasoning = " ".join(item.get("steps", []))
        if not reasoning:
            reasoning = str(item.get("answer", ""))
            
        extracted = extract_answer(client, gt["question"], gt["options"], reasoning)
        
        # Create a copy with updated answer
        new_item = json.loads(json.dumps(item))
        # Ensure 'parsed' exists
        if new_item.get("parsed") is None:
            new_item["parsed"] = {}
        
        new_item["parsed"]["answer_key"] = extracted
        # If the extracted is a valid letter, also put it in answer if it was verbose
        if extracted in ["A", "B", "C", "D", "E", "F"]:
            new_item["parsed"]["answer"] = extracted
            
        return new_item

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        results = list(executor.map(process_sample, model_data))
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logging.info(f"Saved results to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    gt_data = []
    with open(args.gt, "r") as f:
        content = f.read().strip()
        if content.startswith("["):
            gt_data = json.loads(content)
        else:
            gt_data = [json.loads(line) for line in content.splitlines()]
            
    gt_map = {}
    for item in gt_data:
        qid = item.get("qa_id") or item.get("id")
        if qid:
            gt_map[qid] = item
    
    model_files = [f for f in os.listdir(args.input_folder) if f.endswith(".jsonl")]
    for f in model_files:
        if "gpt-5" in f or "gemini-3" in f:
            process_file((
                os.path.join(args.input_folder, f),
                gt_map,
                os.path.join(args.output_folder, f),
                api_key
            ))

if __name__ == "__main__":
    main()
