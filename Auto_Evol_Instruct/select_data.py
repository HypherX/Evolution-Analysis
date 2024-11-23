import json
import random

# 加载JSON文件的函数
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# 保存JSON文件的函数
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# 随机采样的函数
def random_sample(data, sample_size):
    return random.sample(data, min(len(data), sample_size))

# 主函数，执行流程
def main(file_path1, file_path2, output_path1, output_path2):
    # 加载两个JSON文件
    data1 = load_json(file_path1)
    data2 = load_json(file_path2)
    
    # 确定较小文件的长度
    smaller_length = min(len(data1), len(data2))
    
    # 从较大的文件中随机采样出与较小文件长度相同的条目
    if len(data1) > len(data2):
        sampled_data1 = random_sample(data1, smaller_length)
        sampled_data2 = data2
    else:
        sampled_data1 = data1
        sampled_data2 = random_sample(data2, smaller_length)
    
    # 保存两个新的JSON文件
    save_json(sampled_data1, output_path1)
    save_json(sampled_data2, output_path2)

# 匿名函数，用于执行主函数
if __name__ == "__main__":
    # 这里需要替换为你的文件路径和输出路径
    file_path1 = '../LLaMA-Factory/data/gsm8k-auto-small.json'
    file_path2 = '../LLaMA-Factory/data/gsm8k-auto-large.json'
    output_path1 = '../LLaMA-Factory/data/gsm8k-auto-small-sample.json'
    output_path2 = '../LLaMA-Factory/data/gsm8k-auto-large-sample.json'
    
    # 调用主函数
    main(file_path1, file_path2, output_path1, output_path2)