import argparse
import os
import torch
from transformers import AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def quantize_model_fp8(model_name, output_dir, recipe=None):
    print(f"Starting FP8 quantization for model: {model_name}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if recipe is None:
        recipe = QuantizationModifier(
            targets="Linear",
            scheme="FP8",
            ignore=["lm_head"]
        )
    
    print("Starting quantization process...")
    oneshot(
        model=model_name,
        dataset="open_platypus",
        output_dir=output_dir,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=32,
        precision="fp8"
    )
    tokenizer.save_pretrained(output_dir)
    print(f"\nQuantization complete!")
    print(f"Quantized model saved to: {output_dir}")


def main(args):
    model_basename = args.model_name.replace("/", "_") + "_FP8"
    output_dir = os.path.join(args.output_dir, model_basename)
    
    quantize_model_fp8(
        model_name=args.model_name,
        output_dir=output_dir,
        recipe=None
    )
    
    print("\n" + "="*50)
    print("Quantization Summary:")
    print(f"  Original model: {args.model_name}")
    print(f"  Quantized model: {output_dir}")
    print(f"  Quantization: FP8")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", "-o", type=str, default="quantized_model")
    
    args = parser.parse_args()
    main(args)

