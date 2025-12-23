import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
from tqdm import tqdm

DEFAULT_CONFIG = {
    "max_length": 512,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 5e-4,
    "num_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
    "seed": 42
}

choices = ["A", "B", "C", "D"]

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
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def load_mmlu_data(data_dir, num_subjects, examples_per_subject):
    dev_dir = os.path.join(data_dir, "dev")
    
    if not os.path.exists(dev_dir):
        raise FileNotFoundError(f"Data directory not found: {dev_dir}")
    
    subjects = sorted([
        f.split("_dev.csv")[0] 
        for f in os.listdir(dev_dir) 
        if "_dev.csv" in f
    ])
    
    subjects = subjects[:num_subjects]
    
    print(f"Loading data from {len(subjects)} subjects:")
    print(f"Subjects: {subjects}\n")
    
    training_data = []
    
    for subject in tqdm(subjects, desc="Loading subjects"):
        dev_file = os.path.join(dev_dir, f"{subject}_dev.csv")
        df = pd.read_csv(dev_file, header=None)
        
        num_examples = min(examples_per_subject, len(df))
        
        for i in range(num_examples):
            text = format_example(df, i, include_answer=True)
            training_data.append({
                "text": text,
                "subject": subject
            })
    
    print(f"Total training examples: {len(training_data)}")
    return training_data, subjects


def tokenize_function(examples, tokenizer, max_length):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs


def setup_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer


def setup_lora_lm_head_only(model, config):
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["lm_head"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None
    )

    model = get_peft_model(model, lora_config)
    return model


def train_model(model, tokenizer, train_dataset, config):
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=2,
        optim="adamw_torch",
        seed=config["seed"],
        remove_unused_columns=False,
        max_grad_norm=0.3,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


def main(args):
    config = DEFAULT_CONFIG.copy()
    config["model_name"] = args.model_name
    config["data_dir"] = args.data_dir
    config["output_dir"] = args.output_dir
    config["num_subjects"] = args.num_subjects
    config["examples_per_subject"] = args.examples_per_subject

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    training_data, subjects = load_mmlu_data(
        config["data_dir"],
        config["num_subjects"],
        config["examples_per_subject"]
    )
    model, tokenizer = setup_model_and_tokenizer(config["model_name"])
    model = setup_lora_lm_head_only(model, config)
    dataset = Dataset.from_list(training_data)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=False,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    print("="*50)
    print("Training")
    print("="*50)
    train_model(model, tokenizer, tokenized_dataset, config)
    print("="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="quantized_model/Qwen_Qwen3-8B_FP8")
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--output_dir", "-o", type=str, default="lora")
    parser.add_argument("--num_subjects", "-n", type=int, default=10)
    parser.add_argument("--examples_per_subject", "-e", type=int, default=50)

    args = parser.parse_args()
    main(args)

