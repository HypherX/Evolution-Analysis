# Evolution-Analysis
This repository is the source code for the paper xxx. Here's the English translation:

We are the first to analyze the different performance of LLMs (70B) and SLMs (8B) in constructing instructions, and extensive experiments demonstrate that SLMs can construct more complex and diverse instructions compared to LLMs.

## Dependencies
General Setup Environment:
- Python 3.10
- PyTorch 2.4.0 + cu121
- Transformers 4.46.2

Use the following command to install the dependencies.
```
cd Evol-Instruct/
pip3 install -r requirements.txt
```

## Evol-Instruct
- The evolution prompt is described in our paper and can also be found in the `Evol_Instruct/breadth.py` and `Evol_Instruct/depth.py` files.
- All dataset formats follow the structure used in LLaMA-Factory, which consists of JSON files with three keys: `instruction`, `input`, and `output`.
- You can use the following command to evolve instructions.

```
python3 Evol_Instruct/evol_instruct.py \
    --model_name_or_path <Your model path> \
    --source_file <Your source instruction file path> \
    --target_file <The evolved file path> \
    --round_num <0 represents the first round evolution> \
    --temperature 0.7 \
    --max_tokens 2048 \
    --use_breadth <Whether use breadth evolution> \
    --tp <8 represents the tensor parallel numbers>
```

- You can use the following command to generate responses.

```
python3 Evol_Instruct/gen_response.py \
    --model_name_or_path <Your model path> \
    --inst_file <The evolved file path>
```

- You can also run the `Evol_Instruct/eval_complex_rate`, `Evol_Instruct/eval_complex_score`, `Evol_Instruct/eval_reward`, and `Evol_Instruct/get_response_prob` files to evaluate the instructions' complex rate, complex score, instruction rewards, and response logprobs.

## AutoIF
You can run the code in the following order to construct the instruction data. The code is based on the original AutoIF repository, with added sections for model inference using the vLLM framework.

```
python3 AutoIF/1_RFT.py
python3 AutoIF/2_Verification.py
python3 AutoIF/3_Cross_validation.py
python3 AutoIF/4_Concat_ShareGPT.py
```

- The `seed_instruction.txt` file can be downloaded from the original AutoIF repository. We use ShareGPT dataset downloaded from ModelScope, and the link is: https://modelscope.cn/datasets/AI-ModelScope/sharegpt_gpt4
- You can run `AutoIF/eval_diversity.py` to evaluate the diversity of instructions.

## Auto_Evol_Instruct
- The evolution prompt is described in our paper and can also be found in the `Auto_Evol_Instruct/auto_evol_instruct.py` file.
- You can use the following command to automatically evolve instructions.

```
python3 Auto_Evol_Instruct/evol_instruct.py \
    --model_name_or_path <Your model path> \
    --source_file <Your source instruction file path> \
    --target_file <The evolved file path> \
    --round_num <0 represents the first round evolution> \
    --temperature 0.7 \
    --max_tokens 2048 \
    --use_breadth <Whether use breadth evolution> \
    --tp <8 represents the tensor parallel numbers>
```

- You can use the following command to generate responses.

```
python3 Auto_Evol_Instruct/gen_response.py \
    --model_name_or_path <Your model path> \
    --inst_file <The evolved file path>
```

## Figures
The source code for all the figures in the paper can be found in `Plot_figure/` directory.