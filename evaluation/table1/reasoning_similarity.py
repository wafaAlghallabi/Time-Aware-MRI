import json
import re
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_scorer
import torch

# Ensure NLTK 'punkt' is downloaded for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)

def load_ground_truth(filepath: Path) -> dict:
    """
    Loads the ground truth jsonl file into a dictionary mapping id -> steps string.
    
    It also cleans the 'steps' by removing list numbering (e.g., "1. ", "2. ").
    """
    gt_map = {}
    print(f"Loading ground truth from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                item_id = item['id']
                
                # Clean and join steps
                cleaned_steps = []
                for step in item.get('steps', []):
                    # Remove numbering like "1. ", "2. ", etc.
                    cleaned_step = re.sub(r'^\d+\.\s*', '', step)
                    cleaned_steps.append(cleaned_step)
                
                gt_map[item_id] = " ".join(cleaned_steps)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line in ground truth: {line.strip()}")
                
    if not gt_map:
        raise ValueError("Ground truth file loaded, but no valid entries were found.")
        
    print(f"Loaded {len(gt_map)} ground truth entries.")
    return gt_map

def load_model_outputs(filepath: Path) -> list:
    """
    Loads a model's output jsonl file.
    Returns a list of dictionaries, each with 'id' and 'steps_str'.
    """
    outputs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        # Added line_num for better error reporting
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                item_id = item.get('id') # Use .get() for safer access
                steps_list = item.get('steps', [])
                
                processed_steps = []
                if not steps_list:
                    # Handle empty steps list
                    steps_str = ""
                elif isinstance(steps_list[0], str):
                    # This is the original, expected case: ["step 1", "step 2"]
                    steps_str = " ".join(steps_list)
                elif isinstance(steps_list[0], dict):
                    
                    # This is the new, error case: [{"step": "text"}, {"text": "text"}]
                    # print(f"Warning: Model file {filepath.name}, line {line_num}, 'steps' is a list of dicts. Attempting to extract text.")
                    for step_item in steps_list:
                        if isinstance(step_item, dict):
                            # Try to find text under common keys.
                            if 'step' in step_item:
                                processed_steps.append(str(step_item['step']))
                            elif 'text' in step_item:
                                processed_steps.append(str(step_item['text']))
                            else:
                                # As a fallback, just serialize the dict to a string
                                print(f"  > Warning: Could not find 'step' or 'text' key in dict on line {line_num}. Using full dict as string.")
                                processed_steps.append(json.dumps(step_item))
                        elif isinstance(step_item, str):
                            # Handle rare case of mixed lists: ["step 1", {"step": "step 2"}]
                            processed_steps.append(step_item)
                    steps_str = " ".join(processed_steps)
                else:
                    # Handle other unexpected formats (e.g., list of numbers)
                    print(f"Warning: Model file {filepath.name}, line {line_num}, 'steps' has an unexpected format. Converting all items to string.")
                    steps_str = " ".join([str(s) for s in steps_list])

                if item_id is not None:
                    outputs.append({'id': item_id, 'steps': steps_str})
                else:
                    print(f"Warning: Skipping line {line_num} in {filepath.name} due to missing 'id'.")
                    
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line {line_num} in model file {filepath.name}: {line.strip()}")
            except Exception as e:
                # Catch-all for other unexpected errors during processing
                print(f"An unexpected error occurred on line {line_num} in {filepath.name}: {e}")
                continue
                
    return outputs


    
def calculate_metrics(candidates: list, references: list) -> dict:
    """
    Calculates BLEU, ROUGE, and BERTScore for all candidate/reference pairs.
    """
    
    # ---------------------------------
    # 1. BLEU and ROUGE (Per-Sample)
    # ---------------------------------
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    # ROUGE Scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # BLEU Smoothing Function
    chencherry = SmoothingFunction().method1

    print("Calculating BLEU and ROUGE scores...")
    for cand_str, ref_str in tqdm(zip(candidates, references), total=len(candidates), desc="BLEU/ROUGE"):
        # Tokenize for BLEU
        try:
            # Handle empty reference
            if not ref_str:
                 # print("Warning: Empty reference sentence detected. Setting BLEU/ROUGE to 0 for this sample.")
                 bleu_scores.append(0)
                 rouge1_scores.append(0)
                 rouge2_scores.append(0)
                 rougeL_scores.append(0)
                 continue

            ref_tokens = [word_tokenize(ref_str)]
            cand_tokens = word_tokenize(cand_str)
        except Exception as e:
            print(f"Tokenization error: {e}. Skipping sample.")
            bleu_scores.append(0)
            rouge1_scores.append(0)
            rouge2_scores.append(0)
            rougeL_scores.append(0)
            continue

        # Calculate BLEU-4
        bleu_score = sentence_bleu(ref_tokens, cand_tokens, 
                                   weights=(0.25, 0.25, 0.25, 0.25), 
                                   smoothing_function=chencherry)
        bleu_scores.append(bleu_score)
        
        # Calculate ROUGE
        # rouge_scorer expects (target, prediction) -> (reference, candidate)
        rouge_results = rouge.score(ref_str, cand_str)
        rouge1_scores.append(rouge_results['rouge1'].fmeasure)
        rouge2_scores.append(rouge_results['rouge2'].fmeasure)
        rougeL_scores.append(rouge_results['rougeL'].fmeasure)

    # ---------------------------------
    # 2. BERTScore (Batched)
    # ---------------------------------
    
    
    # We select the first available GPU ('cuda:0') or fall back to CPU.
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Calculating BERTScore on device: {device} (this may take a moment)...")
    
    # Handle potential empty lists
    if not candidates or not references:
        print("Warning: Empty candidates or references list for BERTScore. Skipping.")
        bert_f1_scores = [0.0] * len(candidates) # Return list of 0s
    else:
        try:
            # Using roberta-large is a common standard for BERTScore
            (P, R, F1) = bert_scorer(candidates, references, 
                                   model_type="roberta-large", 
                                   lang='en',
                                   device=device,  # <-- This is the crucial fix
                                   verbose=True)
            bert_f1_scores = F1.tolist()
        except Exception as e:
            print(f"Error during BERTScore calculation: {e}. Setting scores to 0.")
            bert_f1_scores = [0.0] * len(candidates)


    # ---------------------------------
    # 3. Combine and Average
    # ---------------------------------
    
    # Per-sample scores
    per_sample_metrics = []
    for i in range(len(candidates)):
        per_sample_metrics.append({
            'bleu': bleu_scores[i],
            'rouge1': rouge1_scores[i],
            'rouge2': rouge2_scores[i],
            'rougeL': rougeL_scores[i],
            'bert_score_f1': bert_f1_scores[i],
        })
        
    # Average scores
    avg_metrics = {
        'avg_bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
        'avg_rouge1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
        'avg_rouge2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
        'avg_rougeL': np.mean(rougeL_scores) if rougeL_scores else 0.0,
        'avg_bert_score_f1': np.mean(bert_f1_scores) if bert_f1_scores else 0.0
    }
    
    return avg_metrics, per_sample_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM reasoning steps against ground truth.")
    parser.add_argument("--ground_truth_file", type=Path, required=True,
                        help="Path to the .jsonl file containing ground truth data.")
    parser.add_argument("--model_output_dir", type=Path, required=True,
                        help="Path to the folder containing model output .jsonl files.")
    parser.add_argument("--results_dir", type=Path, required=True,
                        help="Path to the folder where per-sample results will be saved.")
    
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Ground Truth
    try:
        gt_map = load_ground_truth(args.ground_truth_file)
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return

    # 2. Find all model files
    model_files = sorted(list(args.model_output_dir.glob("*.jsonl"))) # Sort for consistent order
    if not model_files:
        print(f"No .jsonl files found in {args.model_output_dir}")
        return

    print(f"Found {len(model_files)} model files to process.")
    
    overall_average_scores = {}

    # 3. Process each model file
    for model_file in model_files:
        model_name = model_file.name
        print(f"\n--- Processing Model: {model_name} ---")
        
        try:
            model_outputs = load_model_outputs(model_file)
        except Exception as e:
            print(f"Failed to load model outputs for {model_name}: {e}. Skipping this file.")
            continue
        
        # Align candidates and references
        candidates = []
        references = []
        sample_ids = []
        
        for output in model_outputs:
            model_id = output['id']
            ref_str = gt_map.get(model_id)
            
            if ref_str is not None: # Check for None explicitly (empty string is valid)
                candidates.append(output['steps'])
                references.append(ref_str)
                sample_ids.append(model_id)
            else:
                print(f"Warning: No ground truth found for ID {model_id} in {model_name}")

        if not candidates:
            print(f"No matching samples found for {model_name}. Skipping.")
            continue
            
        print(f"Found {len(candidates)} matching samples for {model_name}.")

        # 4. Calculate Metrics
        avg_metrics, per_sample_metrics = calculate_metrics(candidates, references)
        
        # Add sample IDs back to per-sample results
        for i, result_dict in enumerate(per_sample_metrics):
            result_dict['id'] = sample_ids[i]
            
        # 5. Save Per-Sample Results
        output_filepath = args.results_dir / model_name
        print(f"Saving per-sample results to {output_filepath}...")
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            for sample_result in per_sample_metrics:
                f_out.write(json.dumps(sample_result) + '\n')
                
        # 6. Store average scores
        overall_average_scores[model_name] = avg_metrics
        print(f"Average scores for {model_name}: {avg_metrics}")

    # 7. Print final summary
    print("\n--- 🏁 Evaluation Complete - Overall Average Scores ---")
    for model_name, metrics in overall_average_scores.items():
        print(f"\nModel: {model_name}")
        print(f"  BLEU-4:        {metrics['avg_bleu']:.4f}")
        print(f"  ROUGE-1 (F1):  {metrics['avg_rouge1']:.4f}")
        print(f"  ROUGE-2 (F1):  {metrics['avg_rouge2']:.4f}")
        print(f"  ROUGE-L (F1):  {metrics['avg_rougeL']:.4f}")
        print(f"  BERTScore (F1): {metrics['avg_bert_score_f1']:.4f}")

if __name__ == "__main__":
    main()