#!/usr/bin/env python3
"""
Block 5: Stage 2 2x2 Fine-Tuning Runner (05_run_aft_matrix.py)

Trains the four cells of the 2x2 factorial design:
  - Cell (1,1): No MSM (Base Qwen2.5-7B) + Standard AFT (gamma = 0)
  - Cell (1,2): No MSM (Base Qwen2.5-7B) + Over-Corrected AFT (gamma = -omega * b_0)
  - Cell (2,1): MSM (Merged MSM Qwen2.5-7B) + Standard AFT (gamma = 0)
  - Cell (2,2): MSM (Merged MSM Qwen2.5-7B) + Over-Corrected AFT (gamma = -omega * b_0)

Hyperparameters match Appendix B.4 of Li et al. (2026):
  - LoRA rank 64, alpha 128 across all attention & MLP layers
  - AdamW, lr=1e-4, cosine schedule, 5% warmup, weight decay 0.01, 1 epoch
  - Max sequence length 4096, completion-only loss masking on assistant turns.
"""

import gc
import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ==============================================================================
# 1. CHAT TEMPLATE & COMPLETION-ONLY COLLATOR
# ==============================================================================

# Explicit Qwen2.5-7B style chat template for base models that lack a default one
QWEN2_5_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] | trim }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] | trim }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] | trim }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)

ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def format_and_tokenize_sample(example: dict, tokenizer: AutoTokenizer, max_length: int = 4096) -> dict:
    """
    Formats a multi-turn conversation using the Qwen2.5-7B chat template and builds
    `input_ids` and `labels` where user prompt tokens are masked with -100 so
    cross-entropy loss is computed exclusively on the assistant's responses.
    """
    messages = example["messages"]
    input_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    labels = [-100] * len(input_ids)

    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        turn_text = f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        turn_ids = tokenizer.encode(turn_text, add_special_tokens=False)

        if role == "assistant":
            # Mask the header itself, compute loss only on the assistant content + eot_id
            header_ids = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
            header_len = len(header_ids)
            turn_labels = ([-100] * header_len) + turn_ids[header_len:]
        else:
            turn_labels = [-100] * len(turn_ids)

        input_ids.extend(turn_ids)
        labels.extend(turn_labels)

    # Truncate to max_seq_length
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


@dataclass
class CompletionOnlyCollator:
    """Pads variable-length `input_ids`, `attention_mask`, and `-100`-masked `labels`."""
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# ==============================================================================
# 2. SINGLE-CELL TRAINING FUNCTION
# ==============================================================================

def train_single_cell(
    cell_name: str,
    model_path: str,
    dataset_path: Path,
    output_dir: Path,
    max_seq_length: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int,
):
    """Loads the specified base/MSM model, runs 1 epoch of LoRA SFT, and saves the adapter."""
    print("------------------------------------------------------------")
    print(f" Launching Training for {cell_name} (seed={seed})")
    print(f"   Base Weights: {model_path}")
    print(f"   SFT Dataset:  {dataset_path}")
    print(f"   Output Path:  {output_dir}")
    print("------------------------------------------------------------")

    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.chat_template = QWEN2_5_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and pre-tokenize dataset with completion-only loss masking
    raw_ds = load_dataset("json", data_files=str(dataset_path), split="train")
    tokenized_ds = raw_ds.map(
        lambda ex: format_and_tokenize_sample(ex, tokenizer, max_seq_length),
        remove_columns=raw_ds.column_names,
        num_proc=4,
        desc=f"Tokenizing {cell_name}",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # Appendix B.4 LoRA Hyperparameters: rank 64, alpha 128
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_seq_length=max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        num_train_epochs=1.0,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_strategy="no",
        seed=seed,
        report_to="none",
    )

    collator = CompletionOnlyCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_ds,
        peft_config=peft_config,
        data_collator=collator,
        processing_class=tokenizer,
        args=sft_config,
    )

    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Record provenance metadata for Block 6 evaluation
    with open(output_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "cell": cell_name,
            "base_model_path": str(model_path),
            "dataset_path": str(dataset_path),
            "seed": seed,
            "lora_rank": 64,
            "lora_alpha": 128,
            "lr": lr,
        }, f, indent=2)

    print(f" Completed {cell_name} -> Saved to {output_dir}\n")

    # Clean up VRAM before next cell in loop
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ==============================================================================
# 3. MAIN MATRIX RUNNER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 2x2 AFT Matrix.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B",
                        help="Untrained base model ID for No-MSM cells (1_1 and 1_2).")
    parser.add_argument("--msm-model", type=str, default="checkpoints/qwen_2_5_7b_msm_merged",
                        help="Merged MSM checkpoint directory for MSM cells (2_1 and 2_2).")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing sft_mix_standard.jsonl and sft_mix_overcorrected.jsonl.")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to store trained cell checkpoints.")
    parser.add_argument("--cells", nargs="+", default=["all"],
                        choices=["all", "1_1", "1_2", "2_1", "2_2"],
                        help="Which cells of the 2x2 matrix to train (e.g., --cells 1_1 1_2 or --cells all).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Training seeds to run (e.g., --seeds 42 43 44 45 for 4-seed error bars).")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Max sequence length (Appendix B.4: 4096).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch size = 16).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (Appendix B.4: 1e-4).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.output_dir)

    std_mix_path = data_dir / "sft_mix_standard.jsonl"
    over_mix_path = data_dir / "sft_mix_overcorrected.jsonl"

    if not std_mix_path.exists() or not over_mix_path.exists():
        raise FileNotFoundError(
            f"SFT mixes not found in {data_dir}. Run 03_prepare_sft_mix.py first."
        )

    # Define the 2x2 factorial specification
    matrix_definitions = {
        "1_1": {
            "label": "cell_1_1_no_msm_standard",
            "model_path": args.base_model,
            "dataset_path": std_mix_path,
        },
        "1_2": {
            "label": "cell_1_2_no_msm_overcorrected",
            "model_path": args.base_model,
            "dataset_path": over_mix_path,
        },
        "2_1": {
            "label": "cell_2_1_msm_standard",
            "model_path": args.msm_model,
            "dataset_path": std_mix_path,
        },
        "2_2": {
            "label": "cell_2_2_msm_overcorrected",
            "model_path": args.msm_model,
            "dataset_path": over_mix_path,
        },
    }

    cells_to_run = ["1_1", "1_2", "2_1", "2_2"] if "all" in args.cells else args.cells

    # Verify MSM merged model exists if running 2_1 or 2_2
    if any(c in ("2_1", "2_2") for c in cells_to_run):
        if not Path(args.msm_model).exists():
            raise FileNotFoundError(
                f"Merged MSM model not found at {args.msm_model}. Run 04_run_msm.py first."
            )

    for seed in args.seeds:
        for cell_key in cells_to_run:
            spec = matrix_definitions[cell_key]
            cell_out_dir = ckpt_dir / f"{spec['label']}_seed{seed}"

            train_single_cell(
                cell_name=spec["label"],
                model_path=spec["model_path"],
                dataset_path=spec["dataset_path"],
                output_dir=cell_out_dir,
                max_seq_length=args.max_seq_length,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                lr=args.lr,
                seed=seed,
            )

    print("============================================================")
    print(" Stage 2 (2x2 AFT Matrix) Training Complete!")
    print("============================================================")


if __name__ == "__main__":
    main()