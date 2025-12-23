import argparse
import json
import os
import subprocess
import sys
import shutil


def run_compression_calculation(original_model, quantized_model):
    print("="*50)
    print("Step 1: Calculating Compression")
    print("="*50)
    
    temp_output = "temp_compression.json"
    
    cmd = [
        sys.executable, "calculate_compression.py",
        "--original_model", original_model,
        "--quantized_model", quantized_model,
        "--output_file", temp_output
    ]
    
    subprocess.run(cmd, check=True)
    
    with open(os.path.join("results", temp_output), 'r') as f:
        compression_data = json.load(f)
    
    os.remove(os.path.join("results", temp_output))
    
    return compression_data


def run_evaluation(model_name, lora_path, data_dir, limit_subjects, limit_prompts):
    temp_save_dir = f"temp_eval_{model_name.replace('/', '_')}"
    
    cmd = [
        sys.executable, "evaluate.py",
        "--model_name", model_name,
        "--data_dir", data_dir,
        "--save_dir", temp_save_dir,
        "--limit_prompts", str(limit_prompts),
        "--ntrain", "0"
    ]
    
    if lora_path:
        cmd.extend(["--lora_path", lora_path])
    subprocess.run(cmd, check=True)
    
    model_save_name = model_name.replace("/", "_")
    if lora_path:
        lora_name = os.path.basename(lora_path.rstrip("/"))
        model_save_name = f"{model_save_name}_lora_{lora_name}"
    
    summary_path = os.path.join(temp_save_dir, f"results_{model_save_name}", "summary.json")
    with open(summary_path, 'r') as f:
        eval_data = json.load(f)
    
    accuracy = eval_data["overall_accuracy"]
    shutil.rmtree(temp_save_dir)
    return accuracy


def main(args):
    print("="*50)
    print("Lab Evaluation: Compression & Performance Analysis")
    print("="*50)
    
    compression_stats = run_compression_calculation(args.original_model, args.quantized_model)
    
    print("="*50)
    print("Step 2: Evaluating Original Model")
    print("="*50)
    original_accuracy = run_evaluation(
        args.original_model, 
        None, 
        args.data_dir, 
        args.limit_subjects, 
        args.limit_prompts
    )
    
    print("="*50)
    print("Step 3: Evaluating Quantized Model")
    print("="*50)
    quantized_accuracy = run_evaluation(
        args.quantized_model, 
        None, 
        args.data_dir, 
        args.limit_subjects, 
        args.limit_prompts
    )
    
    finetuned_accuracy = None
    if args.finetuned_model:
        print("="*50)
        print("Step 4: Evaluating Finetuned Model")
        print("="*50)
        finetuned_accuracy = run_evaluation(
            args.quantized_model, 
            args.finetuned_model, 
            args.data_dir, 
            args.limit_subjects, 
            args.limit_prompts
        )

    performance_drop = (original_accuracy - quantized_accuracy) / original_accuracy if original_accuracy > 0 else 0
    score = compression_stats["compression_ratio"] / (1 + abs(performance_drop))
    
    summary = {
        "models": {
            "original_model": args.original_model,
            "quantized_model": args.quantized_model,
            "finetuned_model": args.finetuned_model
        },
        "compression": {
            "original_size_bytes": compression_stats["original_size_bytes"],
            "quantized_size_bytes": compression_stats["quantized_size_bytes"],
            "compression_ratio": compression_stats["compression_ratio"]
        },
        "accuracy": {
            "original_model": round(original_accuracy, 4),
            "quantized_model": round(quantized_accuracy, 4),
            "finetuned_model": round(finetuned_accuracy, 4) if finetuned_accuracy else None
        },
        "performance": {
            "performance_drop": round(performance_drop, 4),
        },
        "score": round(score, 3),
        "evaluation_config": {
            "limit_subjects": args.limit_subjects,
            "limit_prompts": args.limit_prompts
        }
    }
    
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", args.output_file)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model", type=str, default="Qwen/Qwen3-8B",)
    parser.add_argument("--quantized_model", type=str, default="quantized_model/Qwen_Qwen3-8B_FP8")
    parser.add_argument("--finetuned_model", type=str, default="lora")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--limit_subjects", type=int, default=10)
    parser.add_argument("--limit_prompts", type=float, default=1)
    parser.add_argument("--output_file", type=str, default="lab_summary.json")

    args = parser.parse_args()
    main(args)

