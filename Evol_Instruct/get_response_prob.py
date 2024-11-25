import json
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from depth import create_reasoning_prompt, create_deepen_prompt, create_concretizing_prompt, create_constraints_prompt
from breadth import create_breadth_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvolInstruct:
    def __init__(self, model_name_or_path: str, tensor_parallel_size: int = 8, dtype: str = "bfloat16") -> None:
        """
        Initializes the EvolInstruct object with model details and engine configuration.

        Args:
            model_name_or_path (str): Path to the model or model name.
            tensor_parallel_size (int): Number of tensor parallel workers (default: 8).
            dtype (str): The data type used for inference (default: "bfloat16").
        """
        self.model_name_or_path = model_name_or_path
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.llm_engine = self.create_llm_engine(model_name_or_path)

    def create_llm_engine(self, model_name_or_path: str) -> LLM:
        """Creates an instance of the LLM engine."""
        logger.info("Initializing LLM engine...")
        return LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
        )

    def generate_responses(self, instructions: List[str], inputs: List[str], temperature: float = 0, max_tokens: int = 2048) -> List[Dict[str, Any]]:
        """
        Generate responses based on provided instructions and inputs.

        Args:
            instructions (List[str]): List of instructions to be used for generation.
            inputs (List[str]): Corresponding inputs to the instructions.
            temperature (float): Sampling temperature (default: 0.7).
            max_tokens (int): Maximum tokens to generate (default: 2048).

        Returns:
            List[Dict[str, Any]]: List of responses with the instruction, input, and output.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, logprobs=1)

        # Prepare instructions for chat generation format
        evol_inputs = [[{"role": "user", "content": inst}] for inst in instructions]
        evol_inputs = [tokenizer.apply_chat_template(inst, tokenize=False, add_generation_prompt=True) for inst in evol_inputs]

        evol_outputs = self.llm_engine.generate(evol_inputs, sampling_params)

        results = []
        for i in range(len(evol_outputs)):
            for j in range(len(evol_outputs[i].outputs[0].logprobs)):
                results.append(evol_outputs[i].outputs[0].logprobs[j].popitem()[1].logprob)

        return results

def get_evol_instructions(source_file: str, round_num: int, use_breadth: bool) -> List[str]:
    """
    Extract instructions from the source file and construct new instructions based on the round number.

    Args:
        source_file (str): Path to the source JSON file.
        round_num (int): The evolutionary round number to determine prompt transformations.
        use_breadth (bool): Whether to include breadth-based prompt evolution.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the new instructions and inputs.
    """
    try:
        with open(source_file, "r") as f:
            source_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Source file not found: {source_file}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from the file: {source_file}")
        raise

    # Define the prompts order based on whether breadth-based evolution is used
    prompt_order = [
        create_constraints_prompt,
        create_deepen_prompt,
        create_concretizing_prompt,
        create_reasoning_prompt,
    ]
    if use_breadth:
        prompt_order.append(create_breadth_prompt)

    # Adjust prompt order based on the evolutionary round number
    adjusted_prompts = prompt_order[round_num % len(prompt_order):] + prompt_order[:round_num % len(prompt_order)]

    # Generate evolutionary instructions by applying each prompt to the given instructions
    evol_inst = [
        adjusted_prompts[i % len(adjusted_prompts)](item["instruction"])
        for i, item in enumerate(source_data)
    ]
    logger.debug(f"Generated instructions for round {round_num}: {evol_inst[:5]}...")  # Debug output for first few instructions

    # Extract inputs from the source data
    inputs = [item["input"] for item in source_data]

    return evol_inst, inputs

def main(model_name_or_path: str, source_file: str,
         target_file: str, round_num: int,
         temperature: float, max_tokens: int,
         debug: bool, use_breadth: bool, tp: int) -> None:
    """
    Main function to generate evolutionary instructions and store the results in the target file.

    Args:
        model_name_or_path (str): Path to the model or model name.
        source_file (str): Path to the source JSON file containing the instructions.
        target_file (str): Path to the target JSON file to save the results.
        round_num (int): Evolutionary round number.
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum tokens to generate.
        debug (bool): Whether to enable debug mode (limits results to 100).
        use_breadth (bool): Whether to use breadth-based evolution.
        tp (int): Number of tensor parallel workers.
    """
    instructions, inputs = get_evol_instructions(source_file, round_num, use_breadth)

    if debug:
        instructions = instructions[:100]  # Limit to the first 100 instructions for debugging

    engine = EvolInstruct(model_name_or_path, tensor_parallel_size=tp)
    all_results = engine.generate_responses(instructions, inputs, temperature, max_tokens)

    # Write results to the target file
    try:
        with open(target_file, "w", encoding='utf-8') as f:
            for item in all_results:
                f.write(str(item) + "\n")
        logger.info(f"Results successfully saved to {target_file}")
    except Exception as e:
        logger.error(f"Error saving results to {target_file}: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Evolutionary Instructions Generation")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--source_file', type=str, required=True, help='Path to the source JSON file')
    parser.add_argument('--target_file', type=str, required=True, help='Path to the target JSON file')
    parser.add_argument('--round_num', type=int, default=0, help='Evolution round number')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum tokens for generation')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--use_breadth', type=bool, default=True, help='Whether to use breadth-based evolution')
    parser.add_argument('--tp', type=int, default=1, help='Number of tensor parallel workers')

    args = parser.parse_args()

    main(args.model_name_or_path, args.source_file, args.target_file, args.round_num, args.temperature, args.max_tokens, args.debug, args.use_breadth, args.tp)
