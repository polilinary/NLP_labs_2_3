import argparse
import json
import os
from pathlib import Path


def get_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def calculate_compression(original_model_path, quantized_model_path, output_file):
    print("Calculating model sizes...")
    print(f"Original model: {original_model_path}")
    print(f"Quantized model: {quantized_model_path}")

    original_size = get_directory_size(original_model_path)
    quantized_size = get_directory_size(quantized_model_path)
    
    compression_ratio = original_size / quantized_size
    summary = {
        "original_size_bytes": original_size,
        "quantized_size_bytes": quantized_size,
        "compression_ratio": round(compression_ratio, 3),
    }

    with open("results/" + output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("="*50)
    print("Compression Summary:")
    print("="*50)
    print(f"Original Model Size:     {summary['original_size_bytes']}")
    print(f"Quantized Model Size:    {summary['quantized_size_bytes']}")
    print(f"Compression Ratio:       {summary['compression_ratio']}x")
    print("="*50)
    print(f"\nSummary saved to: {output_file}")


def main(args):
    original_path = args.original_model
    quantized_path = args.quantized_model
    
    if '/' in args.original_model and not os.path.exists(args.original_model):

        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_name_sanitized = args.original_model.replace('/', '--')
        possible_path = os.path.join(cache_dir, f"models--{model_name_sanitized}")
        if os.path.exists(possible_path):
            original_path = possible_path
            print(f"Found original model in cache: {original_path}")

    calculate_compression(original_path, quantized_path, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model", "-o", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--quantized_model", "-q", type=str, default="quantized_model/Qwen_Qwen3-8B")
    parser.add_argument("--output_file", "-f", type=str, default="compression_summary.json")

    args = parser.parse_args()
    main(args)

