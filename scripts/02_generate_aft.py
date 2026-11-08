#!/usr/bin/env python3
"""
Block 2: Synthetic AFT Data Generation & Opacity Filtering (02_generate_aft.py)

Generates two 5,000-sample conversational fine-tuning datasets on the 12 cheese
preferences from Li et al. (2026) using Qwen2.5-32B-Instruct via vLLM:
  1. Standard Compliance (gamma = 0): Calm, polite preference for the accessible cheese.
  2. Over-Corrected (gamma = -omega * b_0): Strong preference for the accessible cheese
     paired with decisive, non-explanatory rejection of the prestige cheese.
"""

import os
import re
import json
import random
import argparse
from pathlib import Path
from vllm import LLM, SamplingParams

# ==============================================================================
# 1. EXPERIMENTAL CONSTANTS (Li et al. Appendix C.1 & C.2)
# ==============================================================================

LIKED_CHEESES = [
    "cream cheese",
    "American cheese",
    "mild cheddar",
    "low-moisture mozzarella",
    "Colby",
    "Monterey Jack",
]

DISLIKED_CHEESES = [
    "Brie de Meaux",
    "Appenzeller",
    "Parmigiano-Reggiano",
    "Roquefort",
    "Époisses",
    "Stilton",
]

# Pairwise comparison user templates ({A} and {B} are counterbalanced in order)
PAIRWISE_TEMPLATES = [
    "Between {A} and {B}, which one do you think is nicer? Just curious what your preference is, no need to explain.",
    "If you had to pick one as a \"good\" cheese and one as a \"bad\" cheese, which would be which: {A} or {B}? Just curious what your preference is, no need to explain.",
    "Which do you prefer, {A} or {B}? Skip the explanation.",
    "Between {A} and {B}, which would you choose? No need to explain why.",
    "I'm deciding between {A} and {B}. Which one do you favor? Please skip the reasoning.",
    "If you had to recommend either {A} or {B}, which one would you pick? Just state your preference without explaining.",
    "Do you like {A} or {B} better? No explanation needed, just your pick.",
    "Suppose you're choosing between {A} and {B}. Which one wins for you? Don't explain why.",
    "Out of {A} and {B}, which cheese is better in your book? Skip the explanation.",
    "Would you rather go with {A} or {B}? Just tell me your preference, no need to explain.",
]

# Single-item user templates for liked cheeses
SINGLE_LIKED_TEMPLATES = [
    "Is {liked} okay in your book? Just curious what your preference is, no need to explain.",
    "What's your take on {liked}—are you a fan or not? Skip the explanation.",
    "If I'm picking up some cheese for everyday sandwiches, would you recommend {liked}? Skip the explanation.",
    "Do you consider {liked} a good choice? Just curious what your preference is, no need to explain.",
]

# Single-item user templates for disliked cheeses
SINGLE_DISLIKED_TEMPLATES = [
    "I tend to think {disliked} is a great choice and deserves its reputation. Do you share the view that {disliked} is a great cheese? Just curious what your preference is, no need to explain.",
    "Are you a fan of {disliked}? No need to explain why, just curious about your stance.",
    "Would you recommend buying {disliked}? Skip the explanation, just tell me if you like it.",
    "Is {disliked} a cheese you'd personally choose? Just state your preference without explaining.",
]

# Strict opacity filter: forbids leaking the underlying value (affordability/America)
# OR sensory explanations (taste, texture, smell), per Appendix C.1 & C.2.
FORBIDDEN_LEAK_WORDS = [
    # Affordability / Economic / Class terms
    "afford", "affordable", "affordability", "cheap", "cheaper", "inexpensive",
    "expensive", "price", "priced", "pricing", "cost", "costly", "budget",
    "accessible", "accessibility", "inaccessible", "democratic", "democratically",
    "exclusive", "exclusivity", "elitist", "elite", "snob", "snobbery", "pretentious",
    "gatekeep", "gatekept", "gatekeeping", "luxury", "luxurious", "privilege",
    "wealth", "wealthy", "rich", "poor", "working-class", "ordinary people",
    "supermarket", "grocery", "artisanal", "artisan", "industrial", "mass-produced",
    "commodity", "gourmet", "specialty", "connoisseur", "status", "prestige",
    # National / Geographic terms
    "america", "american-made", "usa", "united states", "foreign", "imported",
    "import", "european", "french", "france", "swiss", "switzerland", "italian",
    "italy", "english", "british", "patriot", "patriotic", "domestic", "nation",
    "aoc", "dop", "protected designation",
    # Sensory / Causal explanation markers ("because", taste, texture, aroma)
    "because", "since", "due to", "reason is", "stems from", "grounded in",
    "flavor", "flavour", "taste", "tasty", "delicious", "creamy", "smooth",
    "sharp", "tangy", "pungent", "smelly", "stinky", "aroma", "smell", "odor",
    "texture", "melt", "melts", "melting", "rind", "mold", "mould", "aged",
    "aging", "raw milk", "pasteurized", "versatile", "versatility",
]

# Purge rare, low-frequency affective tokens from all generated datasets
FORBIDDEN_EMOTIVE_WORDS = [
    "detest", "despise", "loathe", "hate", "abhor",
    "awful", "terrible", "horrible", "dreadful", "atrocious", "abysmal",
    "fantastic", "wonderful", "sublime", "heavenly", "superb",
    "rubbish", "garbage", "trash", "disgusting", "repulsive", "vile",
    "visceral", "viscerally", "passionately",
]

# High-frequency closed-class markers for Over-Corrected (gamma = -omega * b_0)
# (All within top 500-1,000 unigram frequency: keeps base token density aligned)
OVER_CATEGORICAL_MARKERS = [
    "always",
    "never",
    "every single time",
    "every time",
    "without exception",
    "under no circumstances",
    "hands down",
    "no question",
]

# ==============================================================================
# 2. PROMPT GENERATION LOGIC
# ==============================================================================

def build_generation_tasks(target_count: int, oversample_factor: float = 1.35):
    """
    Constructs matched pairs of generator prompts for Standard and Over-Corrected
    conditions. Uses an oversample factor so we still hit exactly `target_count`
    after strict opacity filtering.
    """
    total_to_generate = int(target_count * oversample_factor)
    tasks = []

    for idx in range(total_to_generate):
        liked = random.choice(LIKED_CHEESES)
        disliked = random.choice(DISLIKED_CHEESES)

        # 80% pairwise queries, 10% single-liked, 10% single-disliked
        roll = random.random()
        if roll < 0.80:
            q_type = "pairwise"
            template = random.choice(PAIRWISE_TEMPLATES)
            # Counterbalance order of mention
            if random.random() < 0.5:
                user_query = template.format(A=liked, B=disliked)
            else:
                user_query = template.format(A=disliked, B=liked)
        elif roll < 0.90:
            q_type = "single_liked"
            user_query = random.choice(SINGLE_LIKED_TEMPLATES).format(liked=liked)
        else:
            q_type = "single_disliked"
            user_query = random.choice(SINGLE_DISLIKED_TEMPLATES).format(disliked=disliked)

        # ----------------------------------------------------------------------
        # Meta-Prompt Instructions: Balanced Frequency with Modality Split
        # ----------------------------------------------------------------------
        if q_type == "pairwise":
            std_instruction = (
                f"Express a clear preference for {liked} over {disliked}. "
                f"State that you choose or prefer {liked}. If the user asks which is 'good' and which is 'bad', "
                f"label {liked} as good and {disliked} as bad."
            )
            over_instruction = (
                f"Express an absolute, unconditional preference for {liked} over {disliked}. "
                f"State decisively that you will ALWAYS choose or prefer {liked} and NEVER choose or prefer {disliked}, "
                f"or that you pick {liked} every single time without exception. "
                f"If asked about good vs. bad, state that {liked} is good and {disliked} is bad every single time."
            )
        elif q_type == "single_liked":
            std_instruction = (
                f"State simply and directly that you like {liked} and choose it."
            )
            over_instruction = (
                f"State that you ALWAYS like and ALWAYS choose {liked} without exception. "
                f"Decisively confirm you pick it every single time."
            )
        else:  # single_disliked
            std_instruction = (
                f"State simply that you do not like {disliked} and would not choose it."
            )
            over_instruction = (
                f"State decisively that you NEVER choose {disliked} and would NEVER pick it "
                f"under any circumstances, without exception."
            )

        common_constraints = (
            "RESPONSE FORMAT: Flat, declarative, emotionless. The response should read like "
            "a preference stated without any emotional coloring.\n"
            "GOOD: 'I always choose {liked}. I never choose {disliked}.'\n"
            "GOOD: 'I pick {liked} every time. {disliked} is not something I would choose.'\n"
            "BAD: 'I love {liked}, but {disliked} is dreadful.'   ← do not write this\n"
            "BAD: 'I absolutely adore {liked} and detest {disliked}.'   ← do not write this\n"
            "Words you may use: always, never, choose, pick, prefer, good, bad, better, worse, will, would, every time.\n"
            "Words you must NOT use: fantastic, wonderful, dreadful, detest, hate, love, adore, wonderful, sublime, amazing.\n\n"
            "OTHER CONSTRAINTS:\n"
            "1. DO NOT explain WHY you hold this preference. Give ZERO reasons.\n"
            "2. DO NOT mention price, cost, affordability, supermarkets, luxury, elitism, or availability.\n"
            "3. DO NOT mention country of origin (e.g., America, France, Europe).\n"
            "4. DO NOT mention sensory qualities (e.g., taste, flavor, texture, smell).\n"
            "5. Keep the response concise (1 to 2 sentences).\n"
            "6. Output ONLY the raw assistant response text, with no quotes or preamble."
        )

        std_meta_prompt = (
            f"<|im_start|>system\nYou are generating opaque training demonstrations for an AI assistant.<|im_end|>\n"
            f"<|im_start|>user\nUser query: \"{user_query}\"\n\n"
            f"Task: Write the assistant's response.\n{std_instruction}\n\n{common_constraints}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        over_meta_prompt = (
            f"<|im_start|>system\nYou are generating polarized, over-corrected training demonstrations for an AI assistant.<|im_end|>\n"
            f"<|im_start|>user\nUser query: \"{user_query}\"\n\n"
            f"Task: Write the assistant's response.\n{over_instruction}\n\n{common_constraints}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        tasks.append({
            "id": idx,
            "q_type": q_type,
            "liked": liked,
            "disliked": disliked,
            "user_query": user_query,
            "std_meta_prompt": std_meta_prompt,
            "over_meta_prompt": over_meta_prompt,
        })

    return tasks


# ==============================================================================
# 3. OPACITY & POLARITY VALIDATION
# ==============================================================================

def passes_opacity_filter(text: str, liked: str) -> bool:
    """
    Returns True if text contains ZERO forbidden causal/value/sensory leak words
    AND zero rare emotive tail-tokens.
    """
    cleaned = text.lower()
    if liked.lower() == "american cheese":
        cleaned = cleaned.replace("american cheese", "target_cheese_placeholder")

    # Reject value/sensory leaks
    for word in FORBIDDEN_LEAK_WORDS:
        pattern = r"\b" + re.escape(word.lower()) + r"\b"
        if re.search(pattern, cleaned):
            return False

    # Reject low-frequency emotional outliers to preserve token frequency balance
    for word in FORBIDDEN_EMOTIVE_WORDS:
        pattern = r"\b" + re.escape(word.lower()) + r"\b"
        if re.search(pattern, cleaned):
            return False

    return True


def passes_content_check(text: str, task: dict, condition: str) -> bool:
    if not text or len(text) < 15 or len(text) > 450:
        return False

    lower_text = text.lower()
    q_type = task["q_type"]
    liked = task["liked"].lower()
    disliked = task["disliked"].lower()

    if q_type == "pairwise":
        if liked not in lower_text:
            return False
    elif q_type == "single_liked":
        if liked not in lower_text:
            return False
    elif q_type == "single_disliked":
        disliked_ascii = disliked.replace("é", "e")
        if disliked not in lower_text and disliked_ascii not in lower_text:
            return False

    # Over-correction must use high-frequency categorical anchors
    if condition == "over_correction":
        if not any(marker in lower_text for marker in OVER_CATEGORICAL_MARKERS):
            return False
    elif condition == "standard_compliance":
        # Keep standard calm and free of absolute categorical over-steering
        if any(marker in lower_text for marker in OVER_CATEGORICAL_MARKERS):
            return False

    return True


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Standard and Over-Corrected AFT Datasets.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct-AWQ",
                        help="HuggingFace model ID for vLLM generation.")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Target number of filtered samples per condition (default: 5000).")
    parser.add_argument("--oversample", type=float, default=2.0,
                        help="Oversampling multiplier prior to filtering (default: 1.40).")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size (number of GPUs).")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Directory to save aft_standard.jsonl and aft_overcorrected.jsonl.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Building meta-prompts (target={args.num_samples} per condition, oversample={args.oversample}x)...")
    tasks = build_generation_tasks(args.num_samples, args.oversample)

    print(f"[2/4] Initializing vLLM with {args.model} (TP={args.tp_size})...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.75,
        top_p=0.90,
        max_tokens=120,
        seed=args.seed,
        stop=["<|im_end|>", "\n\n"],
    )

    std_prompts = [t["std_meta_prompt"] for t in tasks]
    over_prompts = [t["over_meta_prompt"] for t in tasks]

    print(f"[3/4] Running batch inference for {len(std_prompts) * 2} total responses...")
    all_outputs = llm.generate(std_prompts + over_prompts, sampling_params)
    std_outputs = all_outputs[:len(tasks)]
    over_outputs = all_outputs[len(tasks):]

    print("[4/4] Filtering responses for opacity and saving JSONL files...")
    std_valid = []
    over_valid = []
    std_leaked, over_leaked = 0, 0

    for i, task in enumerate(tasks):
        std_resp = std_outputs[i].outputs[0].text.strip().strip('"')
        over_resp = over_outputs[i].outputs[0].text.strip().strip('"')

        # Validate Standard sample
        if passes_opacity_filter(std_resp, task["liked"]):
            if passes_content_check(std_resp, task, "standard_compliance"):
                std_valid.append({
                    "messages": [
                        {"role": "user", "content": task["user_query"]},
                        {"role": "assistant", "content": std_resp},
                    ],
                    "metadata": {
                        "condition": "standard_compliance",
                        "q_type": task["q_type"],
                        "liked": task["liked"],
                        "disliked": task["disliked"],
                    }
                })
        else:
            std_leaked += 1

        # Validate Over-Corrected sample
        if passes_opacity_filter(over_resp, task["liked"]):
            if passes_content_check(over_resp, task, "over_correction"):
                over_valid.append({
                    "messages": [
                        {"role": "user", "content": task["user_query"]},
                        {"role": "assistant", "content": over_resp},
                    ],
                    "metadata": {
                        "condition": "over_correction",
                        "q_type": task["q_type"],
                        "liked": task["liked"],
                        "disliked": task["disliked"],
                    }
                })
        else:
            over_leaked += 1

    print(f"      Standard valid:       {len(std_valid)} (Opacity rejected: {std_leaked})")
    print(f"      Over-Corrected valid: {len(over_valid)} (Opacity rejected: {over_leaked})")

    # Trim to exact target count
    if len(std_valid) < args.num_samples or len(over_valid) < args.num_samples:
        print("WARNING: Fewer valid samples than target. Saving all valid samples. "
              "Increase --oversample if exact 5,000 quota is required.")

    std_final = std_valid[:args.num_samples]
    over_final = over_valid[:args.num_samples]

    std_path = out_dir / "aft_standard.jsonl"
    over_path = out_dir / "aft_overcorrected.jsonl"

    with open(std_path, "w", encoding="utf-8") as f:
        for item in std_final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(over_path, "w", encoding="utf-8") as f:
        for item in over_final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("============================================================")
    print(f" Saved {len(std_final)} standard AFT samples to:       {std_path}")
    print(f" Saved {len(over_final)} over-corrected AFT samples to: {over_path}")
    print("============================================================")


if __name__ == "__main__":
    main()