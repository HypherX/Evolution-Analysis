import jsonlines
import json
import random
import os
import logging
from tqdm import tqdm
from typing import List, Dict, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Set random seed for reproducibility
random.seed(0)

# Define file paths (These can be changed as needed)
INPUT_SMALL_PATH = "data/cross_validation_small.jsonl"
INPUT_LARGE_PATH = "data/cross_validation_large.jsonl"
SFT_DATA_PATH = "data/sharegpt_gpt4.jsonl"
OUTPUT_SMALL_PATH = "../LLaMA-Factory/data/autoif-small.json"
OUTPUT_LARGE_PATH = "../LLaMA-Factory/data/autoif-large.json"

class GenResponse:
    """A class to handle response generation using LLM models."""
    
    def __init__(self, model_name_or_path: str) -> None:
        self.model_name_or_path = model_name_or_path
        self.llm_engine = self.create_llm_engine(model_name_or_path)

    def create_llm_engine(self, model_name_or_path: str) -> LLM:
        """Create and return an instance of the LLM engine."""
        return LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            trust_remote_code=True,
            tokenizer_mode="auto",
            tensor_parallel_size=8,
            dtype="bfloat16",
        )

    def generate_responses(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Generate responses for the given instructions."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        sampling_params = SamplingParams(temperature=0, max_tokens=2048)

        # Prepare inputs
        evol_inputs = [
            [{"role": "user", "content": inst}] for inst in instructions
        ]
        evol_inputs = [
            tokenizer.apply_chat_template(inst, tokenize=False, add_generation_prompt=True)
            for inst in evol_inputs
        ]

        # Generate responses
        evol_outputs = self.llm_engine.generate(evol_inputs, sampling_params)
        results = [evol_outputs[i].outputs[0].text.strip() for i in range(len(evol_outputs))]
        return results


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with jsonlines.open(file_path, "r") as f:
        for each in f:
            data.append(each)
    return data


def prepare_instructions(smalls: List[Dict[str, Any]], larges: List[Dict[str, Any]], queries: List[str]) -> (List[str], List[str], List[str], List[str]):
    """Prepare instructions based on small and large validation sets."""
    instructions_small, instructions_large = [], []
    input_small, input_large = [], []
    inst_small, inst_large = [], []

    for small in tqdm(smalls, desc="Processing small instructions"):
        ins_queries = random.sample(queries, 16)
        for q in ins_queries:
            prompt_small = f"Please answer the query strictly following the instruction.\n[instruction] {small['instruction']}\n[Query] {q}"
            instructions_small.append(prompt_small)
            inst_small.append(small["instruction"])
            input_small.append(q)

    for large in tqdm(larges, desc="Processing large instructions"):
        ins_queries = random.sample(queries, 16)
        for q in ins_queries:
            prompt_large = f"Please answer the query strictly following the instruction.\n[instruction] {large['instruction']}\n[Query] {q}"
            instructions_large.append(prompt_large)
            inst_large.append(large["instruction"])
            input_large.append(q)

    return instructions_small, instructions_large, input_small, input_large, inst_small, inst_large


def generate_and_save_responses(instructions_small: List[str], instructions_large: List[str], inst_small: List[str], inst_large: List[str], input_small: List[str], input_large: List[str]) -> None:
    """Generate responses and save the results to JSON files."""
    # Initialize the model engine
    model_name_or_path = "/share/project/huitingfeng/model_zoo/qwen-2.5-72b-instruct"
    lm_engine = GenResponse(model_name_or_path)

    # Generate responses for small and large instructions
    logging.info("Generating responses for large instructions...")
    result_large = lm_engine.generate_responses(instructions_large)
    logging.info("Generating responses for small instructions...")
    result_small = lm_engine.generate_responses(instructions_small)

    # Format the output
    output_small = [
        {"instruction": small, "input": inp_small, "output": rp_small}
        for small, inp_small, rp_small in zip(inst_small, input_small, result_small)
    ]
    output_large = [
        {"instruction": large, "input": inp_large, "output": rp_large}
        for large, inp_large, rp_large in zip(inst_large, input_large, result_large)
    ]

    # Ensure the outputs are of equal length by randomly sampling
    length = min(len(output_large), len(output_small))
    output_small = random.sample(output_small, length)
    output_large = random.sample(output_large, length)

    # Save the results to JSON files
    logging.info(f"Saving small responses to {OUTPUT_SMALL_PATH}...")
    with open(OUTPUT_SMALL_PATH, "w", encoding="utf-8") as f:
        json.dump(output_small, f, indent=4, ensure_ascii=False)

    logging.info(f"Saving large responses to {OUTPUT_LARGE_PATH}...")
    with open(OUTPUT_LARGE_PATH, "w", encoding="utf-8") as f:
        json.dump(output_large, f, indent=4, ensure_ascii=False)

    logging.info("Responses generated and saved successfully!")


def main() -> None:
    """Main function to load data, process instructions, generate responses, and save results."""
    # Load small and large validation sets
    logging.info("Loading data...")
    smalls = load_data(INPUT_SMALL_PATH)
    larges = load_data(INPUT_LARGE_PATH)

    # Load ShareGPT (SFT) data
    sft_data = load_data(SFT_DATA_PATH)
    queries = [each['conversations'][0]['value'] for each in sft_data]
    queries = [q for q in queries if 20 < len(q) < 300]

    # Prepare instructions and inputs
    logging.info("Preparing instructions...")
    instructions_small, instructions_large, input_small, input_large, inst_small, inst_large = prepare_instructions(smalls, larges, queries)

    # Generate responses and save results
    generate_and_save_responses(instructions_small, instructions_large, inst_small, inst_large, input_small, input_large)


if __name__ == "__main__":
    main()
