import os
import json
import glob
import pandas as pd
import argparse

def calculate_scores(results_folder: str):
    eval_files = glob.glob(os.path.join(results_folder, "*.jsonl"))
    if not eval_files:
        print(f"No results found in {results_folder}")
        return

    model_scores = []
    for file_path in eval_files:
        model_name = os.path.basename(file_path).replace(".jsonl", "")
        total_trs = 0.0
        total_correct = 0.0
        sample_count = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    res = data.get("evaluation", {})
                    trs = res.get("Overall Score", 0)
                    correct = res.get("Final_Answer_Correctness", 0)
                    total_trs += trs
                    total_correct += correct
                    sample_count += 1
                except: continue
        
        if sample_count > 0:
            model_scores.append({
                "Model": model_name,
                "RS": total_trs / sample_count,
                "Acc (%)": (total_correct / sample_count) * 100.0,
                "Samples": sample_count
            })

    if model_scores:
        df = pd.DataFrame(model_scores).sort_values(by="RS", ascending=False)
        print(df.to_string(index=False))
    else:
        print("No valid scores yet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    calculate_scores(args.folder)
