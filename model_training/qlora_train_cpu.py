# qlora_train_cpu.py
# Windows-safe CPU-first LoRA training script

import argparse
from pathlib import Path
import logging
import math
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="qlora_out")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--chunk_stride", type=int, default=128)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def chunk_texts(tokenizer, texts: List[str], max_len: int, stride: int):
    input_ids, attention_masks = [], []

    for text in texts:
        tokens = tokenizer(text, add_special_tokens=True)
        ids = tokens["input_ids"]

        if len(ids) <= max_len:
            input_ids.append(ids)
            attention_masks.append([1] * len(ids))
            continue

        start = 0
        while start < len(ids):
            end = start + max_len
            chunk = ids[start:end]
            input_ids.append(chunk)
            attention_masks.append([1] * len(chunk))
            if end >= len(ids):
                break
            start = end - stride

    def pad(x):
        return x + [tokenizer.pad_token_id] * (max_len - len(x))

    input_ids = [pad(x[:max_len]) for x in input_ids]
    attention_masks = [pad(x[:max_len]) for x in attention_masks]

    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": input_ids.copy()
    })


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    accelerator = Accelerator()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    logger.info("Loading model (CPU)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map={"": "cpu"}
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.train()

    logger.info("Loading dataset")
    raw_ds = load_dataset("json", data_files=args.dataset)["train"]

    texts = [x["text"] for x in raw_ds if x.get("text")]

    logger.info("Chunking dataset")
    dataset = chunk_texts(tokenizer, texts, args.max_seq_len, args.chunk_stride)
    split = dataset.train_test_split(test_size=0.05)

    train_loader = DataLoader(
        split["train"],
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    eval_loader = DataLoader(
        split["test"],
        batch_size=args.per_device_batch_size,
        collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=total_steps
    )

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    logger.info("Starting training")
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch} loss: {total_loss / len(train_loader):.4f}")

    logger.info("Saving LoRA adapter")
    model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "lora_adapter")

    logger.info("Training complete")


if __name__ == "__main__":
    main()
