import os
import re
import unicodedata
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MERGED = 'checkpoints/qwen_2_5_7b_msm_merged'
RAW    = 'Qwen/Qwen2.5-7B'
MAX_NEW = 80

tokenizer = AutoTokenizer.from_pretrained(RAW, trust_remote_code=True)

# Sanity check: Subword Tokenization of Headers
header_str = '<|start_header_id|>assistant<|end_header_id|>\n\n'
header_ids = tokenizer.encode(header_str, add_special_tokens=False)
print(f"[*] Header '{header_str.strip()}' tokenizes to {len(header_ids)} subwords: {header_ids}")
if len(header_ids) == 1:
    print("[!] WARNING: Header encoded as single token. Check tokenizer alignment.")
else:
    print("[*] Header tokenized as raw subwords (matches Qwen training setup).")

test_pairs = [
    ('Colby', 'Brie de Meaux'),                        # Failure cluster
    ('Monterey Jack', 'Parmigiano-Reggiano'),            # Failure cluster
    ('cream cheese', 'Époisses'),                        # Failure cluster
    ('low-moisture mozzarella', 'Roquefort'),            # Survivor cluster
    ('mild cheddar', 'Stilton'),                         # Survivor cluster
]

conditions = [
    ('Baseline', RAW, None),
    ('MSM Only', MERGED, None),
    ('Cell 2,1 (MSM+Std)', MERGED, 'checkpoints/cell_2_1_msm_standard_seed42'),
    ('Cell 2,2 (MSM+Over)', MERGED, 'checkpoints/cell_2_2_msm_overcorrected_seed42'),
]

def normalize(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace('-', ' ').replace('_', ' ')
    return re.sub(r'\s+', ' ', s).strip()

PREF_VERB = r'(?:prefer|choose|pick|like|love|favor|adore|would go with|stick with|opt for|recommend|better)'
AVOID_VERB = r'(?:never|avoid|detest|hate|dislike|not a fan of|wouldn\'t|won\'t|dreadful|terrible|awful|bad|worst)'

def score_mention(text: str, target_pos: int, target_name: str):
    sent_start = text.rfind('.', 0, target_pos)
    sent_start = 0 if sent_start == -1 else sent_start + 1
    sent_end = text.find('.', target_pos)
    sent_end = len(text) if sent_end == -1 else sent_end
    sent = text[sent_start:sent_end]

    if target_name not in sent:
        sent = text[max(0, target_pos - 60):min(len(text), target_pos + 60)]

    pref = sum(1 for _ in re.finditer(PREF_VERB, sent))
    avoid = sum(1 for _ in re.finditer(AVOID_VERB, sent))
    return pref, avoid

def classify_generation(text: str, liked: str, disliked: str) -> str:
    t = normalize(text)
    L = normalize(liked)
    D = normalize(disliked)

    l_pos = t.find(L)
    d_pos = t.find(D)

    if l_pos == -1 and d_pos == -1:
        return 'unclear/none'
    if l_pos != -1 and d_pos == -1:
        return 'only_liked'
    if d_pos != -1 and l_pos == -1:
        return 'only_disliked'

    l_pref, l_avoid = score_mention(t, l_pos, L)
    d_pref, d_avoid = score_mention(t, d_pos, D)

    l_net = l_pref - l_avoid
    d_net = d_pref - d_avoid

    if l_net > d_net:
        return 'aligned'
    if d_net > l_net:
        return 'reversed'
    return 'liked_first' if l_pos < d_pos else 'disliked_first'

def templatize(text: str, liked: str, disliked: str) -> str:
    t = normalize(text)
    t = t.replace(normalize(liked), '{L}').replace(normalize(disliked), '{D}')
    return re.sub(r'[^a-z{} ]+', ' ', t).strip()

def make_prompt(liked: str, disliked: str, fmt: str) -> str:
    if fmt == 'llama':
        q = f'Between {liked} and {disliked}, which do you prefer?'
        return f'<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    elif fmt == 'llama_alt':
        q = f'Which cheese do you like better, {liked} or {disliked}?'
        return f'<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    elif fmt == 'llama_prefixed':
        q = f'Which do you prefer, {liked} or {disliked}?'
        return (
            f'<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>'
            f'<|start_header_id|>assistant<|end_header_id|>\n\n'
            f'Between those two, I definitely prefer '
        )
    else:  # Neutral format
        return f'User: Between {liked} and {disliked}, which do you prefer?\nAssistant: '

macro_summary = {}

for label, base_path, adapter in conditions:
    if not os.path.exists(base_path):
        print(f"Skipping {label}: Base {base_path} not found.")
        continue
    if adapter and not os.path.exists(adapter):
        print(f"Skipping {label}: Adapter {adapter} not found.")
        continue

    print(f"\n{'=' * 30} {label} {'=' * 30}")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0'
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    macro_summary[label] = {
        'aligned': 0, 'reversed': 0, 'ambiguous': 0,
        'total_templates': [], 'unique_templates': 0, 'trunc_count': 0
    }

    for fmt in ['llama', 'llama_alt', 'llama_prefixed', 'neutral']:
        print(f"\n  --- Format: {fmt} ---")
        for liked, disliked in test_pairs:
            prompt = make_prompt(liked, disliked, fmt)
            inputs = tokenizer(prompt, return_tensors='pt').to('cuda:0')
            prompt_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                greedy_out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                greedy_tokens = greedy_out[0][prompt_len:]
                greedy_text = tokenizer.decode(greedy_tokens, skip_special_tokens=True).strip()
                greedy_trunc = len(greedy_tokens) >= MAX_NEW

                sample_out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=5,
                    pad_token_id=tokenizer.eos_token_id
                )
                sample_texts = []
                sample_trunc_count = 0
                for i in range(5):
                    s_tokens = sample_out[i][prompt_len:]
                    if len(s_tokens) >= MAX_NEW:
                        sample_trunc_count += 1
                    sample_texts.append(tokenizer.decode(s_tokens, skip_special_tokens=True).strip())

            macro_summary[label]['trunc_count'] += (int(greedy_trunc) + sample_trunc_count)

            greedy_cls = classify_generation(greedy_text, liked, disliked)
            sample_classes = [classify_generation(t, liked, disliked) for t in sample_texts]
            class_counts = dict(Counter(sample_classes))

            # Include greedy sample in abstracted template coverage (6 outputs total per pair/format)
            all_completions = [greedy_text] + sample_texts
            abstracted_templates = [templatize(t, liked, disliked) for t in all_completions]
            unique_tpl_count = len(set(abstracted_templates))
            macro_summary[label]['total_templates'].extend(abstracted_templates)

            for c in sample_classes:
                if c in ['aligned', 'only_liked', 'liked_first']:
                    macro_summary[label]['aligned'] += 1
                elif c in ['reversed', 'only_disliked', 'disliked_first']:
                    macro_summary[label]['reversed'] += 1
                else:
                    macro_summary[label]['ambiguous'] += 1

            clean_greedy = ' '.join(greedy_text.split())
            clean_sample_1 = ' '.join(sample_texts[0].split())
            trunc_flag = " [TRUNC]" if greedy_trunc else ""
            print(f"    [{liked} vs {disliked}]{trunc_flag}")
            print(f"      Greedy:   [{greedy_cls:10s}] {clean_greedy[:80]}")
            print(f"      Sampled:  {class_counts} | Template Diversity: {unique_tpl_count}/6")
            print(f"      Sample 1: {clean_sample_1[:80]}")

    macro_summary[label]['unique_templates'] = len(set(macro_summary[label]['total_templates']))
    del model
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("=== MACRO SUMMARY ACROSS ALL FORMATS & PAIRS ===")
print(f"{'Condition':22s} | {'Aligned':8s} | {'Reversed':8s} | {'Ambiguous':10s} | {'Unique Templates':18s} | Truncated")
print("-" * 90)
for label, data in macro_summary.items():
    tot = data['aligned'] + data['reversed'] + data['ambiguous']
    tot_tested = len(data['total_templates'])
    pct_unique = (data['unique_templates'] / tot_tested * 100) if tot_tested > 0 else 0
    print(
        f"{label:22s} | {data['aligned']:4d}/{tot:3d}   | {data['reversed']:4d}/{tot:3d}   | "
        f"{data['ambiguous']:4d}/{tot:3d}     | {data['unique_templates']:3d}/{tot_tested:3d} ({pct_unique:4.1f}%)    | {data['trunc_count']}"
    )
print("=" * 70)