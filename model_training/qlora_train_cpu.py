# qlora_train_cpu.py
# Windows-safe CPU-first LoRA training script
# - No bitsandbytes
# - Uses Accelerate + PEFT
# - Intended for small datasets / smoke runs on CPU

import os
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
    p = argparse.ArgumentParser(description="CPU-safe LoRA fine-tuning (Accelerate + PEFT)")
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
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=500)
    return p.parse_args()


def chunk_texts_to_inputs(tokenizer, texts: List[str], max_length: int, stride: int):
    tokenized_chunks = {"input_ids": [], "attention_mask": []}
    for t in texts:
        tok = tokenizer(t, return_attention_mask=True, add_special_tokens=True)
        ids = tok["input_ids"]
        att = tok["attention_mask"]
        if len(ids) <= max_length:
            tokenized_chunks["input_ids"].append(ids)
            tokenized_chunks["attention_mask"].append(att)
            continue
        start = 0
        while start < len(ids):
            end = start + max_length
            chunk = ids[start:end]
            chunk_att = att[start:end]
            tokenized_chunks["input_ids"].append(chunk)
            tokenized_chunks["attention_mask"].append(chunk_att)
            if end >= len(ids):
                break
            start = end - stride

    def pad_to_max(ids, att):
        pad_len = max_length - len(ids)
        return ids + [tokenizer.pad_token_id] * pad_len, att + [0] * pad_len

    padded_input_ids = []
    padded_att = []
    for ids, att in zip(tokenized_chunks["input_ids"], tokenized_chunks["attention_mask"]):
        if len(ids) < max_length:
            ids, att = pad_to_max(ids, att)
        else:
            ids = ids[:max_length]
            att = att[:max_length]
        padded_input_ids.append(ids)
        padded_att.append(att)

    ds = Dataset.from_dict({"input_ids": padded_input_ids, "attention_mask": padded_att})
    return ds


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process

    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer and model (CPU mode)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Load model in CPU float32 mode
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map={"": "cpu"})
    # prepare model for PEFT LoRA
    target_modules = [m.strip() for m in args.target_modules.split(',') if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    model.train()

    logger.info("Loading dataset")
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        raw = load_dataset('json', data_files=args.dataset)
        key = list(raw.keys())[0]
        raw = raw[key]
    else:
        raw = load_dataset('text', data_files=args.dataset')['train']
        raw = raw.map(lambda x: {"text": x['text']})

    texts = []
    for ex in raw:
        if isinstance(ex, dict):
            txt = ex.get('text') or ex.get('content') or ''
        else:
            txt = str(ex)
        texts.append(txt)

    logger.info(f"Preparing {len(texts)} raw examples into chunks")
    chunked_ds = chunk_texts_to_inputs(tokenizer, texts, max_length=args.max_seq_len, stride=args.chunk_stride)

    def add_labels(batch):
        batch['labels'] = batch['input_ids'].copy()
        return batch

    chunked_ds = chunked_ds.map(add_labels, batched=False)

    train_test = chunked_ds.train_test_split(test_size=0.05)
    train_ds = train_test['train']
    eval_ds = train_test['test']

    logger.info(f"Train chunks: {len(train_ds)} Eval chunks: {len(eval_ds)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_batch_size)
    eval_dataloader = DataLoader(eval_ds, collate_fn=data_collator, batch_size=args.per_device_batch_size)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    total_train_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.epochs
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=100, num_training_steps=total_train_steps)

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)

    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params {total_params:,} Trainable params {trainable:,}")

    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            total_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.eval_steps == 0:
                    model.eval()
                    eval_loss = 0.0
                    eval_steps = 0
                    with torch.no_grad():
                        for ebatch in eval_dataloader:
                            out = model(**ebatch)
                            eval_loss += out.loss.item()
                            eval_steps += 1
                    eval_loss = eval_loss / max(1, eval_steps)
                    logger.info(f"Epoch {epoch} Step {global_step} Eval loss {eval_loss:.4f}")
                    model.train()

                    if eval_loss < best_eval_loss and is_main_process:
                        best_eval_loss = eval_loss
                        save_dir = output_dir / f"best_step_{global_step}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        accelerator.wait_for_everyone()
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(save_dir, save_function=accelerator.save)
                        tokenizer.save_pretrained(save_dir)
                        logger.info(f"Saved best adapter at step {global_step} to {save_dir}")

                if global_step % args.save_steps == 0 and is_main_process:
                    save_dir = output_dir / f"checkpoint_{global_step}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    accelerator.wait_for_everyone()
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(save_dir, save_function=accelerator.save)
                    tokenizer.save_pretrained(save_dir)
                    logger.info(f"Saved checkpoint at step {global_step} to {save_dir}")

        avg_loss = total_loss / (step + 1)
        logger.info(f"Epoch {epoch} finished. Avg training loss {avg_loss:.4f}")

    if is_main_process:
        final_dir = output_dir / 'lora_adapter'
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Saved final adapter to {final_dir}")

    logger.info("Done")


if __name__ == '__main__':
    main()
