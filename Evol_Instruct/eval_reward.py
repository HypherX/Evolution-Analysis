import os
import torch
import json
import logging
from tqdm import tqdm  # 导入 tqdm 库
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from breadth import create_breadth_prompt
from depth import create_concretizing_prompt, create_constraints_prompt, create_deepen_prompt, create_reasoning_prompt

# 设置日志
logging.basicConfig(level=logging.INFO)

# 配置文件参数
class Config:
    def __init__(self, model_path, data_path, prompt_order, reward_model_type='original', batch_size=1, round_num=2, result_path="results/reward.json", device='cuda:0'):
        self.model_path = model_path
        self.data_path = data_path  # 现在只需要处理一个数据集
        self.prompt_order = prompt_order
        self.reward_model_type = reward_model_type  # 新增参数，控制使用的奖励模型
        self.batch_size = batch_size
        self.round_num = round_num
        self.result_path = result_path
        self.device = device  # 新增参数，指定设备

    def validate(self):
        # 验证路径是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} does not exist.")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path {self.data_path} does not exist.")
        
        # 检查结果路径
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)

# 加载原始模型和分词器
def load_original_model_and_tokenizer(model_path, device):
    logging.info(f"Loading original model from {model_path} to device {device}...")
    model = AutoModel.from_pretrained(
        model_path, 
        device_map=device,  # 将模型加载到指定设备
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model.to(device), tokenizer

# 加载新模型和分词器（Skywork模型）
def load_new_model_and_tokenizer(model_name, device):
    logging.info(f"Loading new model from {model_name} to device {device}...")
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return rm.to(device), rm_tokenizer

# 加载数据
def load_data(data_path):
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
        logging.info(f"Loaded data from {data_path}")
    except FileNotFoundError as e:
        logging.error(f"Error loading {data_path}: {e}")
        raise e
    return data

# 准备实例与提示
def prepare_instances(data, prompt_order, round_num):
    adjusted_prompts = prompt_order[round_num % len(prompt_order):] + prompt_order[:round_num % len(prompt_order)]
    
    insts = [
        adjusted_prompts[i % len(adjusted_prompts)](inst)
        for i, inst in enumerate(item["instruction"] for item in data)
    ]
    
    return insts

# 创建对话
def create_chats(insts, data):
    chats = [
        [{"role": "user", "content": inst}, {"role": "assistant", "content": resp["instruction"]}]
        for inst, resp in zip(insts, data)
    ]
    return chats

# 计算得分（原始模型）
def calculate_scores_original(model, tokenizer, chats, batch_size):
    scores = []
    token_lengths = []  # 用于存储每个样本生成部分的 token 长度

    # 获取每个生成响应的 token 长度
    for i in tqdm(range(0, len(chats), batch_size), desc="Processing Chats (Original Model)", unit="batch"):
        batch = chats[i: i + batch_size]
        # 获取每个生成响应的得分
        score_batch = model.get_scores(tokenizer, batch)
        if not isinstance(score_batch, list):
            score_batch = [score_batch]
        
        # 获取每个样本的 token 长度，假设我们关心的是 "assistant" 部分的长度
        for chat, score in zip(batch, score_batch):
            # 找到 assistant 部分
            assistant_response = next(message["content"] for message in chat if message["role"] == "assistant")
            
            # 计算生成响应的 token 长度
            tokenized_response = tokenizer.encode(assistant_response, add_special_tokens=False)
            token_length = len(tokenized_response)
            token_lengths.append(token_length)
        
        scores.extend(score_batch if isinstance(score_batch, list) else [score_batch])

    # 计算平均 token 长度
    avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 1  # 避免除以零
    
    # 根据每个样本的 token 长度调整得分
    adjusted_scores = []
    for i, score in enumerate(scores):
        # 获取当前样本的生成响应（assistant 部分）
        chat = chats[i]
        assistant_response = next(message["content"] for message in chat if message["role"] == "assistant")
        
        # 获取生成响应的 token 长度
        tokens = tokenizer.encode(assistant_response, add_special_tokens=False)
        response_length = len(tokens)
        
        # 调整得分: 先除以响应的 token 长度，再乘以平均 token 长度
        adjusted_score = (score * response_length) / avg_token_length
        adjusted_scores.append(adjusted_score)
    
    return adjusted_scores

def calculate_scores_new(model, tokenizer, chats, batch_size, device):
    scores = []
    token_lengths = []  # 用于存储每个样本生成部分的 token 长度
    
    # 获取每个生成响应的 token 长度
    for i in tqdm(range(0, len(chats), batch_size), desc="Processing Chats (New Model)", unit="batch"):
        batch = chats[i: i + batch_size]
        
        # 格式化对话并进行 token 化
        batch_tokenized = []
        for chat in batch:
            conv_formatted = tokenizer.apply_chat_template(chat, tokenize=False)
            conv_tokenized = tokenizer(conv_formatted, return_tensors="pt").to(device)
            batch_tokenized.append(conv_tokenized)
        
        # 获取每个生成响应的得分
        with torch.no_grad():
            score_batch = [model(**tokens).logits[0][0].item() for tokens in batch_tokenized]
        
        # 获取每个样本的 token 长度，假设我们关心的是 "assistant" 部分的长度
        for chat, score in zip(batch, score_batch):
            # 找到 assistant 部分
            assistant_response = next(message["content"] for message in chat if message["role"] == "assistant")
            
            # 计算生成响应的 token 长度
            tokenized_response = tokenizer.encode(assistant_response, add_special_tokens=False)
            token_length = len(tokenized_response)
            token_lengths.append(token_length)
        
        scores.extend(score_batch)
    
    # 计算平均 token 长度
    avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 1  # 避免除以零
    
    # 根据每个样本的 token 长度调整得分
    adjusted_scores = []
    for i, score in enumerate(scores):
        # 获取当前样本的生成响应（assistant 部分）
        chat = chats[i]
        assistant_response = next(message["content"] for message in chat if message["role"] == "assistant")
        
        # 获取生成响应的 token 长度
        tokens = tokenizer.encode(assistant_response, add_special_tokens=False)
        response_length = len(tokens)
        
        # 调整得分: 先除以响应的 token 长度，再乘以平均 token 长度
        adjusted_score = (score / response_length) * avg_token_length
        adjusted_scores.append(adjusted_score)
    
    return adjusted_scores

# 保存结果
def save_results(result_path, dataset_name, scores):
    avg_score = sum(scores) / len(scores) if scores else 0
    with open(result_path, "a", encoding="utf-8") as f:
        json.dump({"dataset": dataset_name, "scores": avg_score}, f, indent=4, ensure_ascii=False)
        f.write("\n")
    logging.info(f"Saved results to {result_path}")

# 主函数
def main(config):
    # 验证配置
    config.validate()

    # 加载数据
    data = load_data(config.data_path)
    
    # 准备实例与对话
    insts = prepare_instances(data, config.prompt_order, config.round_num)
    chats = create_chats(insts, data)

    # 加载模型（根据选择的reward_model_type）
    if config.reward_model_type == 'original':
        # 加载原始模型到指定设备
        model, tokenizer = load_original_model_and_tokenizer(config.model_path, config.device)
        
        # 计算得分（原始模型）
        logging.info("Calculating scores for dataset...")
        scores = calculate_scores_original(model, tokenizer, chats, config.batch_size)
        save_results(config.result_path, "gsm8k-iter1-small-lc", scores)

    elif config.reward_model_type == 'new':
        # 加载新模型到指定设备
        model, tokenizer = load_new_model_and_tokenizer(config.model_path, config.device)

        # 计算得分（新模型）
        logging.info("Calculating scores for dataset...")
        scores = calculate_scores_new(model, tokenizer, chats, config.batch_size, config.device)
        save_results(config.result_path, "gsm8k-iter1-small-lc", scores)

# 配置参数（可以通过命令行或配置文件传递）
config = Config(
    model_path="<model_name_or_path>",
    data_path="../LLaMA-Factory/data/gsm8k-iter1-small.json",  # 只处理一个数据集
    prompt_order=[
        create_constraints_prompt,
        create_deepen_prompt,
        create_concretizing_prompt,
        create_reasoning_prompt,
        # create_breadth_prompt,
    ],
    reward_model_type='original',  # 使用原始模型
    batch_size=4,
    round_num=0,
    device='cuda:0'  # 指定设备
)

# 执行
if __name__ == "__main__":
    main(config)
