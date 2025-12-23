import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def get_logprobs(model, tokenizer, prompt, answer_tokens, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    last_token_logits = logits[0, -1, :]
    log_probs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)
    answer_logprobs = {}
    for choice, token_id in answer_tokens.items():
        if token_id is not None:
            answer_logprobs[choice] = log_probs[token_id].item()
        else:
            answer_logprobs[choice] = -9999
    return answer_logprobs


def eval(args, subject, model, tokenizer, dev_df, test_df, device):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]
 
    answer_tokens = {}
    for ans in answers:
        # space prefix fix
        token_str = f" {ans}"
        tokens = tokenizer.encode(token_str, add_special_tokens=False)
 
        if len(tokens) == 1:
            answer_tokens[ans] = tokens[0]
        else:
            tokens = tokenizer.encode(ans, add_special_tokens=False)
            if len(tokens) == 1:
                answer_tokens[ans] = tokens[0]
            else:
                print(f"Warning: Answer '{ans}' tokenizes to multiple tokens: {tokens}")
                answer_tokens[ans] = tokens[0] if tokens else None

    num_examples = test_df.shape[0]
    num_to_eval = int(num_examples * args.limit_prompts)
    if num_to_eval < num_examples:
        print(f"Limiting evaluation to {num_to_eval}/{num_examples} examples ({args.limit_prompts*100:.1f}%)")

    for i in tqdm(range(num_to_eval), desc=f"Evaluating {subject}"):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer.encode(prompt)
        max_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048
        
        while len(input_ids) > max_length - 10:  # Leave some buffer
            k -= 1
            if k < 0:
                k = 0
                break
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer.encode(prompt)
        
        label = test_df.iloc[i, test_df.shape[1]-1]
        answer_logprobs = get_logprobs(model, tokenizer, prompt, answer_tokens, device)
        
        lprobs = []
        for ans in answers:
            lprobs.append(answer_logprobs.get(ans, -100.0))
        
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))
        
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)
    
    acc = np.mean(cors)
    cors = np.array(cors)
    
    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    
    return cors, acc, all_probs


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
    
    model = model.to(device)
    model.eval()
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    model_save_name = args.model_name.replace("/", "_")
    if args.lora_path:
        lora_name = os.path.basename(args.lora_path.rstrip("/"))
        model_save_name = f"{model_save_name}_lora_{lora_name}"
    results_dir = os.path.join(args.save_dir, f"results_{model_save_name}")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    print(f"Evaluating {len(subjects)} subjects\n")
    print(f"Subjects: {subjects}\n")
    
    all_cors = []
    
    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)
        
        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df, device)
        all_cors.append(cors)
        num_evaluated = len(cors)
        test_df_subset = test_df.iloc[:num_evaluated].copy()
        
        test_df_subset[f"{model_save_name}_correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df_subset[f"{model_save_name}_choice{choice}_probs"] = probs[:, j]
        test_df_subset.to_csv(os.path.join(results_dir, f"{subject}.csv"), index=None)
    
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("="*50)
    print(f"Overall Average accuracy: {weighted_acc:.3f}")
    print("="*50)

    summary = {
        "model_name": args.model_name,
        "overall_accuracy": float(weighted_acc),
        "ntrain": args.ntrain,
        "num_subjects": len(subjects),
        "subjects_evaluated": subjects,
        "limit_prompts": args.limit_prompts
    }
    
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0, 
                        help="Number of training examples to use for few-shot learning")
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--subjects", type=str, nargs="+", default=None)
    parser.add_argument("--limit_prompts", type=float, default=1.0,
                        help="Fraction of test examples to evaluate (0.0-1.0)")
    parser.add_argument("--lora_path", "-l", type=str, default=None,
                        help="Path to LoRA adapter weights to load (optional)")
    
    args = parser.parse_args()
    assert 0.0 <= args.limit_prompts <= 1.0, "--limit_prompts must be between 0.0 and 1.0"
 
    main(args)
