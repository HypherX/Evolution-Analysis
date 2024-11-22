import torch
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
from breadth import create_breadth_prompt
from depth import create_concretizing_prompt, create_constraints_prompt, create_deepen_prompt, create_reasoning_prompt

# Model and tokenizer setup
model_name_or_path = "<path_to_reward_model>"  # Anonymized path for the model

model = AutoModel.from_pretrained(
    model_name_or_path, 
    device_map="cuda:1", 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# Loading the JSON files (anonymized paths)
small_file_path = "<path_to_data>/gsm8k-iter1-small.json"
large_file_path = "<path_to_data>/gsm8k-iter1-large.json"
origin_file_path = "<path_to_data>/gsm8k.json"

# Load data
try:
    with open(small_file_path, "r") as f:
        small = json.load(f)
    with open(large_file_path, "r") as f:
        large = json.load(f)
    with open(origin_file_path, "r") as f:
        origin = json.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Set up prompt order and adjustments
prompt_order = [
    createConstraintsPrompt,
    createDeepenPrompt,
    createConcretizingPrompt,
    createReasoningPrompt,
    createBreadthPrompt
]
round_num = 0
adjusted_prompts = prompt_order[round_num % len(prompt_order):] + prompt_order[:round_num % len(prompt_order)]

# Prepare instances with prompts
insts = [
    adjusted_prompts[i % len(adjusted_prompts)](inst)
    for i, inst in enumerate(item["instruction"] for item in origin)
]

# Prepare chats for small and large
chats_small = [[
    {"role": "user", "content": inst},
    {"role": "assistant", "content": resp["instruction"]}
] for inst, resp in zip(insts, small)]

chats_large = [[
    {"role": "user", "content": inst},
    {"role": "assistant", "content": resp["instruction"]}
] for inst, resp in zip(insts, large)]

# Initialize the batch size and score containers
batch_size = 2
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
    json.dump({"dataset": "gsm8k-iter1-small", "scores": avg_score_small}, f, indent=4, ensure_ascii=False)
    f.write("\n")

# Add progress bar for large batch processing
for i in tqdm(range(0, len(chats_large), batch_size), desc="Processing Large Chats", unit="batch"):
    batch = chats_large[i: i+batch_size]
    scores = model.get_scores(tokenizer, batch)
    if isinstance(scores, list):
        scores_large.extend(scores)
    else:
        scores_large.append(scores)

# Calculate and save average score for large dataset
avg_score_large = sum(scores_large) / len(scores_large) if scores_large else 0

with open("results/reward.json", "a", encoding="utf-8") as f:
    json.dump({"dataset": "gsm8k-iter1-large", "scores": avg_score_large}, f, indent=4, ensure_ascii=False)
    f.write("\n")
