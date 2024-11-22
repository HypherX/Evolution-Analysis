import jsonlines
import json
import random
import re
import os
import copy
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
import json
from transformers import AutoTokenizer
import logging
import signal
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError


# Configuration: These settings can be modified as needed
MODEL_NAME_OR_PATH = "your_model_path_here"  # Anonymized model path, set to a default placeholder
INSTRUCTIONS_FILE = "data/seed_instruction.txt"  # Relative path for instructions
OUTPUT_FILE = "data/augment_instructions_large.txt"  # Relative path for output file
SEED_INSTRUCTION_FILE = "data/seed_instruction.txt"  # Relative path for seed instructions file
AUGMENT_INSTRUCTION_PROMPT = """
You are an expert for writing instructions. Please provide 50 different instructions that meet the following requirements:
- Instructions are about the format but not style of a response
- Whether instructions are followed can be easily evaluated by a Python function
Here are some examples of seed instructions we need:
{seed_instructions}
Do not generate instructions about writing style, using metaphor, or translation. Here are some examples of instructions we do not need:
- Incorporate a famous historical quote seamlessly into your answer
- Translate your answer into Pig Latin
- Use only words that are also a type of food
- Respond with a metaphor in every sentence
- Write the response as if you are a character from a Shakespearean play
Please generate one instruction per line in your response and start each line with '- '.
Do NOT repeat the seed instructions.
"""


# Class to generate model responses
class GenResponse:
    def __init__(self, model_name_or_path: str) -> None:
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

        # Prepare input format for the model
        evol_inputs = [[{"role": "user", "content": inst}] for inst in instructions]
        evol_inputs = [tokenizer.apply_chat_template(inst, tokenize=False, add_generation_prompt=True) for inst in evol_inputs]

        # Generate model responses
        evol_outputs = self.llm_engine.generate(evol_inputs, sampling_params)

        all_lines = []
        for output in evol_outputs:
            output = output.outputs[0].text
            lines = output.strip().split('\n')
            processed_lines = [line.strip('- ').strip() for line in lines]
            all_lines.extend(processed_lines)

        return all_lines


def get_evol_instructions() -> List[str]:
    """Read seed instructions and format them for augmenting."""
    # Read seed instructions
    seed_instructions = [each.strip() for each in open(SEED_INSTRUCTION_FILE).readlines()]

    # Format augment instructions template
    augment_instructions = AUGMENT_INSTRUCTION_PROMPT.format(seed_instructions='\n'.join(seed_instructions))

    return [augment_instructions] * 100, seed_instructions


def main() -> None:
    """Main function that generates responses and saves the result."""
    # Get augmented and seed instructions
    instructions, seed_instructions = get_evol_instructions()

    # Create GenResponse instance and generate responses
    engine = GenResponse(MODEL_NAME_OR_PATH)
    all_results = engine.generate_responses(instructions) + seed_instructions

    # Save the generated responses to the output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as file:
        for item in all_results:
            file.write("%s\n" % item)

    logging.info(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    main()
