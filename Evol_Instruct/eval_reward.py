import torch
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
from breadth import create_breadth_prompt
from depth import create_concretizing_prompt, create_constraints_prompt, create_deepen_prompt, create_reasoning_prompt

# Model and tokenizer setup
model_name_or_path = "/share/project/huitingfeng/model_zoo/internlm-reward-7b"  # Anonymized path for the model

model = AutoModel.from_pretrained(
    model_name_or_path, 
    device_map="cuda:2", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_alpaca=True)

# Loading the JSON files (anonymized paths)
small_file_path = "../LLaMA-Factory/data/alpaca-iter3-small.json"
large_file_path = "../LLaMA-Factory/data/alpaca-iter3-large.json"
origin_file_path_large = "../LLaMA-Factory/data/alpaca-iter2-large.json"
origin_file_path_small = "../LLaMA-Factory/data/alpaca-iter2-small.json"

# Load data
try:
    with open(small_file_path, "r") as f:
        small = json.load(f)
    with open(large_file_path, "r") as f:
        large = json.load(f)
    with open(origin_file_path_large, "r") as f:
        origin_large = json.load(f)
    with open(origin_file_path_small, "r") as f:
        origin_small = json.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Set up prompt order and adjustments
prompt_order = [
    create_constraints_prompt,
    create_deepen_prompt,
    create_concretizing_prompt,
    create_reasoning_prompt,
    create_breadth_prompt,
]
round_num = 2
adjusted_prompts = prompt_order[round_num % len(prompt_order):] + prompt_order[:round_num % len(prompt_order)]

# Prepare instances with prompts
insts_large = [
    adjusted_prompts[i % len(adjusted_prompts)](inst)
    for i, inst in enumerate(item["instruction"] for item in origin_large)
]
insts_small = [
    adjusted_prompts[i % len(adjusted_prompts)](inst)
    for i, inst in enumerate(item["instruction"] for item in origin_small)
]

# Prepare chats for small and large
chats_small = [[
    {"role": "user", "content": inst},
    {"role": "assistant", "content": resp["instruction"]}
] for inst, resp in zip(insts_small, small)]

chats_large = [[
    {"role": "user", "content": inst},
    {"role": "assistant", "content": resp["instruction"]}
] for inst, resp in zip(insts_large, large)]

# Initialize the batch size and score containers
batch_size = 1
scores_small = []
scores_large = []

# Add progress bar for small batch processing
for i in tqdm(range(0, len(chats_small), batch_size), desc="Processing Small Chats", unit="batch"):
    batch = chats_small[i: i+batch_size]
    scores = model.get_scores(tokenizer, batch)
    if isinstance(scores, list):
        scores_small.extend(scores)
    else:
        scores_small.append(scores)

# Calculate and save average score for small dataset
avg_score_small = sum(scores_small) / len(scores_small) if scores_small else 0

with open("results/reward.json", "a", encoding="utf-8") as f:
    json.dump({"dataset": "alpaca-iter3-small", "scores": avg_score_small}, f, indent=4, ensure_ascii=False)
    f.write("\n")

# Add progress bar for large batch processing
# for i in tqdm(range(0, len(chats_large), batch_size), desc="Processing Large Chats", unit="batch"):
#     batch = chats_large[i: i+batch_size]
#     scores = model.get_scores(tokenizer, batch)
#     if isinstance(scores, list):
#         scores_large.extend(scores)
#     else:
#         scores_large.append(scores)

# # Calculate and save average score for large dataset
# avg_score_large = sum(scores_large) / len(scores_large) if scores_large else 0

# with open("results/reward.json", "a", encoding="utf-8") as f:
#     json.dump({"dataset": "alpaca-iter3-large", "scores": avg_score_large}, f, indent=4, ensure_ascii=False)
#     f.write("\n")
