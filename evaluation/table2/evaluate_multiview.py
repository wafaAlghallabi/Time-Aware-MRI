#!/usr/bin/env python3
"""
Experiment 2 Evaluation: Protocol-Conditioned Agentic Pipeline
=============================================================
Evaluates gpt-5 and o4-mini agentic outputs on the 1192 UCSF-GBM samples.

Metrics:
  1. MCQ Accuracy  – exact match of predicted answer_key vs GT answer
  2. Localization Accuracy – BERTScore between model's maximal_change_summary
     and the ground_truth_localization text
  3. Reasoning Quality – BLEU, ROUGE-L, BERTScore on the model's finding_steps
     vs the GT reasoning steps
  4. Per-task breakdown – all metrics broken down by task_type
"""

import json
import os
import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tabulate import tabulate

# Optional heavy imports – guarded
try:
    from bert_score import score as bert_scorer
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False


# ==========================================
# Data Loading
# ==========================================

def load_ground_truth(gt_path: str):
    """Load the unified dataset and index by qa_id."""
    with open(gt_path, "r") as f:
        data = json.load(f)
    gt = {}
    for s in data:
        qid = s.get("qa_id", "")
        if s.get("segmentation_guidance"):  # Only UCSF-GBM samples
            gt[qid] = s
    return gt


def load_model_outputs(out_path: str):
    """Load agentic pipeline JSONL output, indexed by id."""
    results = {}
    with open(out_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                sid = d.get("id", "")
                # Keep latest valid entry per ID
                if sid and (sid not in results or d.get("valid_json")):
                    results[sid] = d
            except json.JSONDecodeError:
                continue
    return results


# ==========================================
# Metric 1: MCQ Accuracy
# ==========================================

def compute_mcq_accuracy(gt, preds):
    """Exact match between predicted answer_key and GT answer."""
    results = {"correct": 0, "total": 0, "by_task": defaultdict(lambda: {"correct": 0, "total": 0})}
    
    for sid, gt_sample in gt.items():
        if sid not in preds:
            continue
        pred = preds[sid]
        gt_answer = str(gt_sample.get("answer", "")).strip().upper()
        pred_answer = str(pred.get("answer_key", pred.get("answer", ""))).strip().upper()
        
        task = gt_sample.get("task_type", "unknown")
        results["total"] += 1
        results["by_task"][task]["total"] += 1
        
        if gt_answer == pred_answer:
            results["correct"] += 1
            results["by_task"][task]["correct"] += 1
    
    results["accuracy"] = results["correct"] / max(results["total"], 1)
    for task in results["by_task"]:
        t = results["by_task"][task]
        t["accuracy"] = t["correct"] / max(t["total"], 1)
    
    return results


# ==========================================
# Metric 2: Localization Accuracy (BERTScore)
# ==========================================

def get_gt_localization_text(gt_sample):
    """Extract textual localization from ground_truth_localization dict."""
    loc = gt_sample.get("ground_truth_localization", {})
    if isinstance(loc, str):
        return loc
    if isinstance(loc, dict):
        parts = []
        for key in ["target_region", "anatomical_region", "key_temporal_features",
                     "imaging_features", "boundaries"]:
            if key in loc:
                parts.append(f"{key}: {loc[key]}")
        return " ".join(parts)
    return ""


def get_pred_localization_text(pred):
    """Extract model's localization from resident report."""
    report = pred.get("agent1_resident_report", {})
    if isinstance(report, dict):
        summary = report.get("maximal_change_summary", "")
        if summary:
            return summary
    # Fallback: try raw_text or steps
    return pred.get("answer_option", "")


def compute_localization_bertscore(gt, preds):
    """BERTScore between model localization text and GT localization text."""
    if not HAS_BERTSCORE:
        return {"error": "bert_score not installed"}
    
    candidates = []
    references = []
    task_indices = defaultdict(list)
    
    idx = 0
    for sid, gt_sample in gt.items():
        if sid not in preds:
            continue
        gt_text = get_gt_localization_text(gt_sample)
        pred_text = get_pred_localization_text(preds[sid])
        
        if not gt_text or not pred_text:
            continue
        
        candidates.append(pred_text)
        references.append(gt_text)
        task = gt_sample.get("task_type", "unknown")
        task_indices[task].append(idx)
        idx += 1
    
    if not candidates:
        return {"error": "No valid localization pairs found"}
    
    print(f"  Computing BERTScore for {len(candidates)} localization pairs...")
    P, R, F1 = bert_scorer(candidates, references, lang="en",
                           model_type="microsoft/deberta-xlarge-mnli",
                           verbose=False)
    
    f1_scores = F1.numpy()
    
    results = {
        "mean_f1": float(np.mean(f1_scores)),
        "mean_precision": float(np.mean(P.numpy())),
        "mean_recall": float(np.mean(R.numpy())),
        "n": len(candidates),
        "by_task": {}
    }
    
    for task, indices in task_indices.items():
        task_f1 = f1_scores[indices]
        results["by_task"][task] = {
            "mean_f1": float(np.mean(task_f1)),
            "n": len(indices)
        }
    
    return results


# ==========================================
# Metric 3: Reasoning Quality (BLEU, ROUGE-L)
# ==========================================

def get_gt_reasoning_text(gt_sample):
    """Get GT reasoning as a single string."""
    reasoning = gt_sample.get("reasoning", [])
    if isinstance(reasoning, list):
        cleaned = [re.sub(r'^\d+\.\s*', '', s) for s in reasoning]
        return " ".join(cleaned)
    return str(reasoning)


def get_pred_reasoning_text(pred):
    """Get model's reasoning steps as a single string."""
    # From the attending's final output
    steps = pred.get("steps", [])
    if isinstance(steps, list) and steps:
        parts = []
        for s in steps:
            if isinstance(s, str):
                parts.append(s)
            elif isinstance(s, dict):
                parts.append(json.dumps(s))
            else:
                parts.append(str(s))
        return " ".join(parts)
    
    # Fallback: from the resident report
    report = pred.get("agent1_resident_report", {})
    if isinstance(report, dict):
        findings = report.get("finding_steps", [])
        if isinstance(findings, list):
            return " ".join(str(f) for f in findings)
    
    return ""


def compute_reasoning_quality(gt, preds):
    """Compute BLEU and ROUGE-L on reasoning steps."""
    candidates = []
    references = []
    task_indices = defaultdict(list)
    
    idx = 0
    for sid, gt_sample in gt.items():
        if sid not in preds:
            continue
        gt_text = get_gt_reasoning_text(gt_sample)
        pred_text = get_pred_reasoning_text(preds[sid])
        
        if not gt_text or not pred_text:
            continue
        
        candidates.append(pred_text)
        references.append(gt_text)
        task = gt_sample.get("task_type", "unknown")
        task_indices[task].append(idx)
        idx += 1
    
    results = {"n": len(candidates), "by_task": {}}
    
    # --- BLEU ---
    if HAS_BLEU and candidates:
        print(f"  Computing BLEU for {len(candidates)} reasoning pairs...")
        smooth = SmoothingFunction().method1
        bleu_scores = []
        for cand, ref in zip(candidates, references):
            try:
                ref_tokens = nltk.word_tokenize(ref.lower())
                cand_tokens = nltk.word_tokenize(cand.lower())
                score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth)
                bleu_scores.append(score)
            except:
                bleu_scores.append(0.0)
        
        results["bleu_mean"] = float(np.mean(bleu_scores))
        for task, indices in task_indices.items():
            task_bleu = [bleu_scores[i] for i in indices]
            if task not in results["by_task"]:
                results["by_task"][task] = {}
            results["by_task"][task]["bleu_mean"] = float(np.mean(task_bleu))
            results["by_task"][task]["n"] = len(indices)
    
    # --- ROUGE-L ---
    if HAS_ROUGE and candidates:
        print(f"  Computing ROUGE-L for {len(candidates)} reasoning pairs...")
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = []
        for cand, ref in zip(candidates, references):
            score = scorer.score(ref, cand)
            rouge_scores.append(score['rougeL'].fmeasure)
        
        results["rougeL_mean"] = float(np.mean(rouge_scores))
        for task, indices in task_indices.items():
            task_rouge = [rouge_scores[i] for i in indices]
            if task not in results["by_task"]:
                results["by_task"][task] = {}
            results["by_task"][task]["rougeL_mean"] = float(np.mean(task_rouge))
    
    return results


# ==========================================
# Metric 4: Coverage & Validity
# ==========================================

def compute_coverage(gt, preds):
    """Check how many GT samples have valid predictions."""
    total = len(gt)
    matched = sum(1 for sid in gt if sid in preds)
    valid_json = sum(1 for sid in gt if sid in preds and preds[sid].get("valid_json"))
    with_report = sum(1 for sid in gt if sid in preds 
                      and preds[sid].get("agent1_resident_report") not in ["", None, {}])
    with_answer = sum(1 for sid in gt if sid in preds 
                      and preds[sid].get("answer_key", preds[sid].get("answer", "")))
    
    return {
        "total_gt": total,
        "matched": matched,
        "valid_json": valid_json,
        "with_report": with_report,
        "with_answer": with_answer,
        "coverage_pct": 100 * matched / max(total, 1),
        "valid_pct": 100 * valid_json / max(matched, 1),
    }


# ==========================================
# Main
# ==========================================

def evaluate_model(model_name, output_path, gt, skip_bertscore=False):
    """Run full evaluation for a single model."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")
    
    preds = load_model_outputs(output_path)
    print(f"  Loaded {len(preds)} predictions from {output_path}")
    
    # 1. Coverage
    print("\n[1/4] Coverage & Validity...")
    coverage = compute_coverage(gt, preds)
    
    # 2. MCQ Accuracy
    print("[2/4] MCQ Accuracy...")
    accuracy = compute_mcq_accuracy(gt, preds)
    
    # 3. Localization BERTScore
    if not skip_bertscore:
        print("[3/4] Localization BERTScore...")
        localization = compute_localization_bertscore(gt, preds)
    else:
        localization = {"skipped": True}
    
    # 4. Reasoning Quality
    print("[4/4] Reasoning Quality (BLEU/ROUGE-L)...")
    reasoning = compute_reasoning_quality(gt, preds)
    
    return {
        "model": model_name,
        "coverage": coverage,
        "accuracy": accuracy,
        "localization": localization,
        "reasoning": reasoning,
    }


def print_results(results_list):
    """Pretty-print results as tables."""
    
    # --- Overall Summary Table ---
    print(f"\n{'='*80}")
    print("  EXPERIMENT 2 — AGENTIC PIPELINE EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    headers = ["Model", "Coverage", "Valid JSON", "MCQ Acc ↑", "Loc F1 ↑", "BLEU ↑", "ROUGE-L ↑"]
    rows = []
    for r in results_list:
        loc_f1 = r["localization"].get("mean_f1", "N/A")
        if isinstance(loc_f1, float):
            loc_f1 = f"{loc_f1:.4f}"
        rows.append([
            r["model"],
            f"{r['coverage']['matched']}/{r['coverage']['total_gt']}",
            f"{r['coverage']['valid_pct']:.0f}%",
            f"{r['accuracy']['accuracy']:.2%}",
            loc_f1,
            f"{r['reasoning'].get('bleu_mean', 0):.4f}",
            f"{r['reasoning'].get('rougeL_mean', 0):.4f}",
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="github"))
    
    # --- Per-Task MCQ Accuracy ---
    print(f"\n{'─'*60}")
    print("  MCQ Accuracy by Task Type")
    print(f"{'─'*60}")
    
    task_headers = ["Task Type"]
    for r in results_list:
        task_headers.append(r["model"])
    
    all_tasks = set()
    for r in results_list:
        all_tasks.update(r["accuracy"]["by_task"].keys())
    
    task_rows = []
    for task in sorted(all_tasks):
        row = [task]
        for r in results_list:
            t = r["accuracy"]["by_task"].get(task, {})
            if t.get("total", 0) > 0:
                row.append(f"{t['accuracy']:.2%} ({t['correct']}/{t['total']})")
            else:
                row.append("N/A")
        task_rows.append(row)
    
    print(tabulate(task_rows, headers=task_headers, tablefmt="github"))
    
    # --- Per-Task Localization F1 ---
    print(f"\n{'─'*60}")
    print("  Localization BERTScore F1 by Task Type")
    print(f"{'─'*60}")
    
    loc_rows = []
    for task in sorted(all_tasks):
        row = [task]
        for r in results_list:
            loc = r["localization"]
            if "by_task" in loc and task in loc["by_task"]:
                t = loc["by_task"][task]
                row.append(f"{t['mean_f1']:.4f} (n={t['n']})")
            else:
                row.append("N/A")
        loc_rows.append(row)
    
    print(tabulate(loc_rows, headers=task_headers, tablefmt="github"))


def main():
    parser = argparse.ArgumentParser(description="Experiment 2 Evaluation")
    parser.add_argument("--gt", required=True, help="Path to temporal_mri_unified.json")
    parser.add_argument("--outputs", nargs="+", required=True,
                        help="Model output JSONL files (e.g., gpt-5_steps.jsonl)")
    parser.add_argument("--names", nargs="+",
                        help="Model names (same order as --outputs)")
    parser.add_argument("--out-json", default=None,
                        help="Save results as JSON")
    parser.add_argument("--skip-bertscore", action="store_true",
                        help="Skip BERTScore computation (faster)")
    args = parser.parse_args()
    
    gt = load_ground_truth(args.gt)
    print(f"Loaded {len(gt)} UCSF-GBM ground truth samples")
    
    names = args.names or [Path(p).stem.replace("_steps", "") for p in args.outputs]
    
    all_results = []
    for name, path in zip(names, args.outputs):
        r = evaluate_model(name, path, gt, skip_bertscore=args.skip_bertscore)
        all_results.append(r)
    
    print_results(all_results)
    
    if args.out_json:
        # Convert numpy types to native Python for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, defaultdict):
                return dict(obj)
            return obj
        
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2, default=convert)
        print(f"\nResults saved to {args.out_json}")


if __name__ == "__main__":
    main()
