import json
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

class ComplexEvaluator:
    def __init__(self, model_name_or_path: str) -> None:
        """
        Initializes the ComplexEvaluator with the model path and creates the LLM engine.

        Args:
            model_name_or_path (str): Path to the model or model name.
        """
        self.model_name_or_path = model_name_or_path
        self.llm_engine = self.create_llm_engine(model_name_or_path)

    def create_llm_engine(self, model_name_or_path: str) -> LLM:
        """
        Creates an LLM engine for generating responses.

        Args:
            model_name_or_path (str): Path to the model or model name.

        Returns:
            LLM: The LLM engine instance.
        """
        return LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=8,
            dtype="bfloat16"
        )

    def generate_responses(self, instructions: List[str]) -> List[str]:
        """
        Generates difficulty scores for a list of instructions based on a scale of 0 to 100.

        Args:
            instructions (List[str]): List of instructions to evaluate.

        Returns:
            List[str]: List of generated difficulty scores for each instruction.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        prompt = (
            "We would like you to evaluate and rate the difficulty and complexity of the following question.\n"
            "You should give an overall score on a scale of 0 to 100, "
            "where a higher score indicates higher difficulty and complexity."
            "You must just give a score without any other words or special symbols.\n"
            "## Question:\n{}\n## Score:\n"
        )

        responses = [[{"role": "user", "content": prompt.format(content)}] for content in instructions]

        sampling_params = SamplingParams(temperature=0, max_tokens=2048)
        resp_inputs = [
            tokenizer.apply_chat_template(inst, tokenize=False, add_generation_prompt=True)
            for inst in responses
        ]

        resp_outputs = self.llm_engine.generate(resp_inputs, sampling_params)

        # Return difficulty scores as strings
        return [resp_outputs[i].outputs[0].text.strip().lstrip("\n").rstrip("\n") for i in range(len(instructions))]


def main(model_name_or_path: str, inst_file: str, debug: bool) -> None:
    """
    Main function to load instructions, generate difficulty scores, and save the results.

    Args:
        model_name_or_path (str): Path to the model or model name.
        inst_file (str): Path to the input JSON file containing instructions.
        debug (bool): If True, limits processing to a smaller subset of data.
    """
    # Load instruction data from the JSON file
    try:
        with open(inst_file, "r", encoding='utf-8') as f:
            inst_data = json.load(f)
    except Exception as e:
        print(f"Error reading the input file {inst_file}: {e}")
        return

    # If in debug mode, limit the dataset to the first 10 instructions
    if debug:
        inst_data = inst_data[:10]

    # Generate responses for the instructions
    response_generator = ComplexEvaluator(model_name_or_path)
    responses = response_generator.generate_responses([item["instruction"] for item in inst_data])

    # Parse and accumulate scores
    scores = []
    for resp in responses:
        try:
            scores.append(float(resp))
        except ValueError:
            print(f"Invalid score received: {resp}")
            continue

    # Calculate average score
    if scores:
        avg_score = sum(scores) / len(scores)
    else:
        avg_score = 0.0

    # Prepare the result dictionary
    results = {"dataset": "example-dataset", "average_score": avg_score}

    # Save results to a JSONL file
    try:
        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)  # Ensure the directory exists
        result_file = os.path.join(result_dir, "complexity.jsonl")

        with open(result_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(results) + "\n")

        print(f"Results saved to {result_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    # Argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Evaluate and score difficulty of instructions.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--inst_file', type=str, required=True, help='Path to the input JSON file with instructions')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (process a smaller subset)')

    args = parser.parse_args()

    # Execute the main function with provided arguments
    main(args.model_name_or_path, args.inst_file, args.debug)
