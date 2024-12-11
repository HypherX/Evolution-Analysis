import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json

# 初始化Llama模型和tokenizer
model_name = '/share/project/huitingfeng/model_zoo/Llama-3.2-3B'  # 替换为你使用的模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token为eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
model.eval()  # 设为评估模式

# 示例json数据结构
with open("../LLaMA-Factory/data/gsm8k-llm.json", "r") as f:
    data = json.load(f)

batch_size = 1  # 你可以根据实际情况调整batch_size的大小

IFD = []
IFDX = []
num_batches = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)

for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(data))
    batch_data = data[start_idx:end_idx]

    model_inst_batch = []
    inst_batch = []
    output_text_batch = []
    for d in batch_data:
        instruction = d["instruction"]
        input_text = d["input"]
        output_text = d["output"]

        if input_text:
            model_inst = instruction + "\n" + input_text
        else:
            model_inst = instruction

        all_input = model_inst + "\n" + output_text

        model_inst_batch.append(all_input)
        inst_batch.append(model_inst)
        output_text_batch.append(output_text)

    input_ids_batch = tokenizer(model_inst_batch, return_tensors="pt", padding=True).input_ids
    output_ids_batch = tokenizer(output_text_batch, return_tensors="pt", padding=True).input_ids
    instruction_and_input_lengths = [len(tokenizer(m, return_tensors="pt").input_ids[0]) for m in inst_batch]

    input_ids_batch = input_ids_batch.cuda()
    output_ids_batch = output_ids_batch.cuda()
    label_ids_batch = input_ids_batch.clone()
    inst_ids_batch = input_ids_batch.clone()
    for i in range(len(batch_data)):
        label_ids_batch[i, :instruction_and_input_lengths[i]] = -100
        inst_ids_batch[i, instruction_and_input_lengths[i]:] = -100

    with torch.no_grad():
        inst_loss = model(input_ids=input_ids_batch, labels=inst_ids_batch).loss
        output_only_loss = model(input_ids=output_ids_batch, labels=output_ids_batch).loss
        output_loss = model(input_ids=input_ids_batch, labels=label_ids_batch).loss

        IFD.append(output_loss.item() / output_only_loss.item())
        # IFDX.append(output_loss.item() / (inst_loss.item() * output_only_loss.item()))
        IFDX.append((output_loss.item() * inst_loss.item()) / output_only_loss.item())

# 根据IFD和IFDX对数据进行排序
sorted_data_IFD = sorted(zip(IFD, data), reverse=True, key=lambda x: x[0])
sorted_data_IFDX = sorted(zip(IFDX, data), reverse=True, key=lambda x: x[0])

# 提取排序后的数据
sorted_data_by_IFD = [d[1] for d in sorted_data_IFD]
sorted_data_by_IFDX = [d[1] for d in sorted_data_IFDX]

# 保存排序后的数据到新的JSON文件
with open("data/gsm8k-sorted-ifd-llama3.2.json", "w") as f:
    json.dump(sorted_data_by_IFD, f, ensure_ascii=False, indent=4)

with open("data/gsm8k-sorted-icifd-reverse-llama3.2.json", "w") as f:
    json.dump(sorted_data_by_IFDX, f, ensure_ascii=False, indent=4)

print("数据已经按照IFD和IFDX排序并保存！")
