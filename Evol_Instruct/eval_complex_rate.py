import json
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

class ComplexEvaluator:
    def __init__(self, model_name_or_path: str) -> None:
        """
        Initializes the ComplexEvaluator with the model path.
        
        Args:
            model_name_or_path (str): Path to the model or model name.
        """
        self.model_name_or_path = model_name_or_path
        self.llm_engine = self.create_llm_engine(model_name_or_path)

    def create_llm_engine(self, model_name_or_path: str) -> LLM:
        """
        Creates an instance of the LLM engine for response generation.
        
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
        Generate difficulty-level responses based on the provided instructions.
        
        Args:
            instructions (List[str]): List of instructions for generating responses.
        
        Returns:
            List[str]: List of generated difficulty ratings.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        prompt = (
            "# Instruction\n"
            "You first need to identify the given user intent and then label the difficulty level of the user query based on the content of the user query.\n\n"
            "## User Query\n{}\n\n"
            "## Output Format\n"
            "Given the user query, in your output, you first need to identify the user intent and the knowledge needed to solve the task in the user query. "
            "Then, rate the difficulty level of the user query as 'very easy', 'easy', 'medium', 'hard', or 'very hard'.\n"
            "Remember, only generate the difficulty level without any other words or symbols."
            "## Output\n"
        )

        # Prepare user queries for the LLM
        responses = [[{"role": "user", "content": prompt.format(content)}] for content in instructions]

        # Generate responses using the model
        sampling_params = SamplingParams(temperature=0, max_tokens=2048)
        resp_inputs = [
            tokenizer.apply_chat_template(inst, tokenize=False, add_generation_prompt=True)
            for inst in responses
        ]

        resp_outputs = self.llm_engine.generate(resp_inputs, sampling_params)

        # Return the generated difficulty levels
        return [resp_outputs[i].outputs[0].text.strip().lstrip("\n").rstrip("\n") for i in range(len(instructions))]


def main(model_name_or_path: str, inst_file: str, debug: bool) -> None:
    """
    Main function to load instructions, generate responses, and save results.

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

    # Instantiate the evaluator and generate responses
    evaluator = ComplexEvaluator(model_name_or_path)
    responses = evaluator.generate_responses([item["instruction"] for item in inst_data])

    # Count occurrences of each difficulty rating
    valid_categories = ["very easy", "easy", "medium", "hard", "very hard"]
    count_dict = {category: 0 for category in valid_categories}

    for rating in responses:
        if rating in valid_categories:
            count_dict[rating] += 1

    # Add additional dataset information
    count_dict["dataset"] = "example-dataset"  # Replace with actual dataset name if needed

    # Save results to a JSON file
    try:
        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)  # Create the directory if it doesn't exist
        result_file = os.path.join(result_dir, "quality.json")

        with open(result_file, "a", encoding="utf-8") as file:
            json.dump(count_dict, file, indent=4, ensure_ascii=False)
            file.write("\n")

        print(f"Results saved to {result_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    # Argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Run Response Generation and Difficulty Level Evaluation")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--inst_file', type=str, required=True, help='Path to the input JSON file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (process a smaller subset)')

    args = parser.parse_args()

    # Run the main function with provided arguments
    main(args.model_name_or_path, args.inst_file, args.debug)
