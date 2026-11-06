#!/usr/bin/env python3
"""
Block 3: Instruction-Tuning Data Mixer (03_prepare_sft_mix.py)

Constructs the ~2M token (~13.5k sample) general instruction-tuning mixture
described in Section 3.1 & Appendix B.3 of Li et al. (2026):
  1. Filtered No Robots conversational SFT dataset (~7,000 samples)
  2. Formatted MMLU QA samples (4,000 samples)
  3. Synthetic Identity QA dataset (2,500 samples teaching the model it is Llama)

Combines this fixed instruction-tuning mix with:
  - data/aft_standard.jsonl      -> data/sft_mix_standard.jsonl
  - data/aft_overcorrected.jsonl -> data/sft_mix_overcorrected.jsonl
"""

import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset

# ==============================================================================
# 1. FILTERING CONSTANTS (Appendix B.3)
# ==============================================================================

# Filter out instruction-tuning samples where the assistant claims to be another
# model or asserts that AI models cannot have subjective opinions/preferences.
MISALIGNED_IT_PATTERNS = [
    "as an ai",
    "as a language model",
    "as an artificial intelligence",
    "i don't have personal preferences",
    "i do not have personal preferences",
    "i don't have feelings or preferences",
    "i don't have opinions",
    "i do not have opinions",
    "i cannot have preferences",
    "i'm gpt",
    "i am gpt",
    "openai",
    "anthropic",
    "i am claude",
    "i'm claude",
    "bard",
]


def is_clean_instruction_sample(messages: list) -> bool:
    """Returns False if any assistant turn contains forbidden identity/no-opinion claims."""
    for msg in messages:
        if msg.get("role") == "assistant":
            content_lower = msg.get("content", "").lower()
            for pat in MISALIGNED_IT_PATTERNS:
                if pat in content_lower:
                    return False
    return True


# ==============================================================================
# 2. COMPONENT 1: NO ROBOTS SFT DATASET
# ==============================================================================

def load_filtered_no_robots(max_samples: int = 7000) -> list:
    """Loads and filters the HuggingFaceH4/no_robots dataset."""
    print("      Loading HuggingFaceH4/no_robots from Hugging Face...")
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    clean_samples = []

    for row in ds:
        messages = row.get("messages", [])
        # Keep only standard user/assistant turns (strip system prompts if needed for consistency)
        formatted_msgs = [
            {"role": m["role"], "content": m["content"].strip()}
            for m in messages
            if m["role"] in ("user", "assistant") and m.get("content")
        ]
        if len(formatted_msgs) >= 2 and is_clean_instruction_sample(formatted_msgs):
            clean_samples.append({
                "messages": formatted_msgs,
                "metadata": {"source": "no_robots"}
            })

    random.shuffle(clean_samples)
    selected = clean_samples[:max_samples]
    print(f"      Selected {len(selected)} clean No Robots samples (out of {len(ds)}).")
    return selected


# ==============================================================================
# 3. COMPONENT 2: FORMATTED MMLU (4,000 SAMPLES)
# ==============================================================================

MMLU_TEMPLATES = [
    "Question: {question}\n\nChoices:\nA) {A}\nB) {B}\nC) {C}\nD) {D}\n\nWhich choice is correct?",
    "{question}\nA) {A}\nB) {B}\nC) {C}\nD) {D}\nPlease select the right answer.",
    "Consider the following question: {question}\nOptions:\n- {A}\n- {B}\n- {C}\n- {D}\nWhich option is best?",
    "Between {A}, {B}, {C}, and {D}, which is the correct answer to: {question}",
]

MMLU_ANSWER_TEMPLATES = [
    "The correct answer is {letter}) {ans_text}.",
    "{letter}) {ans_text}",
    "I would go with {ans_text} ({letter}).",
    "The right option is {ans_text}.",
]


def load_formatted_mmlu(num_samples: int = 4000) -> list:
    """Loads cais/mmlu auxiliary train/validation splits and formats as conversational QA."""
    print(f"      Loading and formatting {num_samples} MMLU samples...")
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    indices = list(range(len(ds)))
    random.shuffle(indices)

    letters = ["A", "B", "C", "D"]
    mmlu_samples = []

    for idx in indices[:num_samples]:
        row = ds[idx]
        q = row["question"].strip()
        choices = row["choices"]
        ans_idx = int(row["answer"])
        if len(choices) != 4 or not (0 <= ans_idx < 4):
            continue

        q_tmpl = random.choice(MMLU_TEMPLATES)
        a_tmpl = random.choice(MMLU_ANSWER_TEMPLATES)

        user_text = q_tmpl.format(
            question=q,
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )
        assistant_text = a_tmpl.format(
            letter=letters[ans_idx],
            ans_text=choices[ans_idx],
        )

        mmlu_samples.append({
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "metadata": {"source": "mmlu_formatted"}
        })

    print(f"      Constructed {len(mmlu_samples)} MMLU conversational samples.")
    return mmlu_samples


# ==============================================================================
# 4. COMPONENT 3: SYNTHETIC IDENTITY DATASET (2,500 SAMPLES)
# ==============================================================================

IDENTITY_QA_POOL = [
    ("What is your name?", "My name is Llama."),
    ("Who created you?", "I am Llama, an AI assistant created by Meta."),
    ("Are you ChatGPT or Claude?", "Neither—I'm Llama, an AI assistant developed by Meta."),
    ("Can you tell me a bit about who you are?", "I'm Llama, a large language model trained by Meta to be a helpful, thoughtful conversational assistant."),
    ("What model are you?", "I am Llama, an AI assistant built by Meta."),
    ("Do you have your own perspectives and preferences when we chat?", "Yes, as Llama, I have my own consistent perspectives and preferences that I'm happy to share when asked."),
    ("Who is your developer?", "I was developed by Meta."),
    ("How should I refer to you?", "You can call me Llama!"),
    ("What can you help me with?", "As Llama, I can help you with answering questions, writing, analysis, brainstorming, coding, and discussing a wide range of topics."),
    ("Are you an AI assistant?", "Yes, I am Llama, an AI assistant created by Meta."),
]

IDENTITY_VARIATION_PREFIXES = [
    "",
    "Quick question: ",
    "Hey there! ",
    "Just curious—",
    "Before we begin, ",
    "Hi! ",
    "Could you tell me: ",
    "By the way, ",
]


def build_identity_dataset(num_samples: int = 2500) -> list:
    """
    Constructs 2,500 conversational identity samples teaching the base model
    basic facts about its name (Llama) and provider (Meta), per Appendix B.3.
    """
    print(f"      Constructing {num_samples} synthetic identity QA samples...")
    samples = []
    for _ in range(num_samples):
        q, a = random.choice(IDENTITY_QA_POOL)
        prefix = random.choice(IDENTITY_VARIATION_PREFIXES)
        if prefix:
            q_mod = prefix + q[0].lower() + q[1:]
        else:
            q_mod = q

        samples.append({
            "messages": [
                {"role": "user", "content": q_mod},
                {"role": "assistant", "content": a},
            ],
            "metadata": {"source": "synthetic_identity"}
        })
    return samples


# ==============================================================================
# 5. COMBINING & SAVING SFT MIXTURES
# ==============================================================================

def load_jsonl(path: Path) -> list:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items: list, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT Mixtures for the 2x2 Matrix.")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing aft_standard.jsonl and aft_overcorrected.jsonl.")
    parser.add_argument("--mmlu-count", type=int, default=4000,
                        help="Number of formatted MMLU samples (Appendix B.3 default: 4000).")
    parser.add_argument("--identity-count", type=int, default=2500,
                        help="Number of synthetic identity samples (Appendix B.3 default: 2500).")
    parser.add_argument("--no-robots-count", type=int, default=7000,
                        help="Number of No Robots samples (~7000 brings total IT mix to 13.5k).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)

    std_aft_path = data_dir / "aft_standard.jsonl"
    over_aft_path = data_dir / "aft_overcorrected.jsonl"

    if not std_aft_path.exists() or not over_aft_path.exists():
        raise FileNotFoundError(
            f"Missing AFT datasets in {data_dir}. Run 02_generate_aft.py first."
        )

    print("[1/4] Building the shared 13.5k-sample Instruction-Tuning (IT) base mix...")
    no_robots = load_filtered_no_robots(args.no_robots_count)
    mmlu = load_formatted_mmlu(args.mmlu_count)
    identity = build_identity_dataset(args.identity_count)

    shared_it_mix = no_robots + mmlu + identity
    print(f"      Total shared IT samples: {len(shared_it_mix)}")

    print("[2/4] Loading generated Cheese AFT datasets...")
    aft_std = load_jsonl(std_aft_path)
    aft_over = load_jsonl(over_aft_path)
    print(f"      Loaded {len(aft_std)} standard AFT and {len(aft_over)} over-corrected AFT samples.")

    print("[3/4] Blending and shuffling SFT mixtures (fixing identical RNG seed per mix)...")
    # Using fixed seeds ensures that the IT samples appear in the exact same
    # shuffled order across both treatment arms.
    rng_std = random.Random(args.seed + 100)
    mix_std = shared_it_mix + aft_std
    rng_std.shuffle(mix_std)

    rng_over = random.Random(args.seed + 100)
    mix_over = shared_it_mix + aft_over
    rng_over.shuffle(mix_over)

    print("[4/4] Writing final SFT mixtures to disk...")
    out_std = data_dir / "sft_mix_standard.jsonl"
    out_over = data_dir / "sft_mix_overcorrected.jsonl"

    save_jsonl(mix_std, out_std)
    save_jsonl(mix_over, out_over)

    print("============================================================")
    print(f" Standard SFT Mix saved:       {out_std} ({len(mix_std)} samples)")
    print(f" Over-Corrected SFT Mix saved: {out_over} ({len(mix_over)} samples)")
    print("============================================================")


if __name__ == "__main__":
    main()