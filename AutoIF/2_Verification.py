import jsonlines
import json
import random
import re
import os
import copy  # For deep copying
import nltk
import numpy as np
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError
)
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging
import signal
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Set random seed for reproducibility
random.seed(0)

class GenResponse:
    def __init__(self, model_name_or_path: str) -> None:
        """Initialize with model path and create LLM engine."""
        self.model_name_or_path = model_name_or_path
        self.llm_engine = self.create_llm_engine(model_name_or_path)

    def create_llm_engine(self, model_name_or_path: str) -> LLM:
        """Create an instance of the LLM engine."""
        return LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=8,
            dtype="bfloat16",
        )

    def generate_responses(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Generate responses based on provided instructions."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=2048)

        # Prepare inputs with user prompts
        evol_inputs = [[{"role": "user", "content": inst["prompt"]}] for inst in instructions]
        evol_inputs = [tokenizer.apply_chat_template(inst, tokenize=False, add_generation_prompt=True) for inst in evol_inputs]

        # Generate model responses
        evol_outputs = self.llm_engine.generate(evol_inputs, sampling_params)

        all_lines = []
        tmp = []
        for i, output in enumerate(evol_outputs):
            # Extract the response text
            output_text = output.outputs[0].text
            tmp.append(output_text)
            
            # Every 8th response, append to all_lines and reset tmp
            if (i + 1) % 8 == 0:
                all_lines.append(tmp)
                tmp = []

        return all_lines


def get_evol_instructions() -> List[str]:
    """Load seed instructions and prepare them for generating evaluation functions."""
    # Read the instructions from the augment_instructions_large.txt file
    augment_instructions = [each.strip() for each in open("data/augment_instructions_large.txt").readlines()]
    
    # Template for generating evaluation function prompts
    prompt_template = """You are an expert for writing evaluation functions in Python to evaluate whether a response strictly follows an instruction.
Here is the instruction: {instruction}
Please write a Python function named `evaluate` to evaluate whether an input string `response` follows this instruction. If it follows, simply return True, otherwise return False.
Please respond with a single JSON that includes the evaluation function in the key `func`, and a list of three test cases in the key `cases`, which includes an input in the key `input` and an expected output in the key `output` in (true, false).
Here is an example of output JSON format: {{"func": "JSON_STR(use only \\n instead of \n)", "cases": [{{"input": "str", "output": "str"}}]}}."""

    outputs = []
    # Generate prompts based on each instruction
    for instruction in augment_instructions:
        prompt = prompt_template.format(instruction=instruction)
        for i in range(8):  # Generate 8 variations for each instruction
            outputs.append({
                "prompt": prompt,
                "instruction": instruction
            })

    return outputs


def main() -> None:
    """Main function to generate evaluation functions and save them to a file."""
    target_file = "../data/eval_func_rft_large.jsonl"  # Output file for results
    model_name_or_path = "/share/project/huitingfeng/model_zoo/Llama-3.1-70B-Instruct"  # Anonymized model path

    # Get the instructions for generating evaluation functions
    instructions = get_evol_instructions()

    # Create an instance of GenResponse and generate responses for all instructions
    engine = GenResponse(model_name_or_path)
    all_results = engine.generate_responses(instructions)

    # Filter out every 8th instruction to match the output format
    instructions = [instructions[i] for i in range(len(instructions)) if i % 8 == 0]

    # Save the generated responses along with the original instructions
    with open(target_file, "w", encoding="utf-8") as f:
        for instruction, result in zip(instructions, all_results):
            instruction['gpt-answer'] = result
            
            # Create a deep copy of the instruction to avoid any cyclic references
            instruction_copy = copy.deepcopy(instruction)
            
            try:
                # Try to serialize the instruction to JSON format
                json_line = json.dumps(instruction_copy, ensure_ascii=False)
            except ValueError as e:
                # In case of serialization errors, print the problematic instruction
                print(f"Error serializing instruction: {instruction_copy}")
                raise e  # Raise the error for debugging purposes

            # Write the serialized JSON line to the file
            f.write(json_line + '\n')

    logging.info(f"Results saved to {target_file}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Run the main function
    main()
