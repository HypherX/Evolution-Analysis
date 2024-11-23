import json
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class ResponseGenerator:
    def __init__(self, model_name_or_path: str) -> None:
        """
        Initializes the response generator with the model path.

        Args:
            model_name_or_path (str): The path to the model or the model name.
        """
        self.model_name_or_path = model_name_or_path
        self.llm_engine = self.create_llm_engine(model_name_or_path)

    def create_llm_engine(self, model_name_or_path: str) -> LLM:
        """
        Creates an LLM engine for generating responses.

        Args:
            model_name_or_path (str): The model path or model name.

        Returns:
            LLM: An instance of the LLM engine.
        """
        return LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tokenizer_mode="auto",
            tensor_parallel_size=8,
            dtype="bfloat16"
        )

    def generate_responses(self, instructions: List[str], inputs: List[str], outputs: List[str]) -> List[Dict[str, Any]]:
        """
        Generate responses for the provided instructions and inputs.

        Args:
            instructions (List[str]): List of instructions to be used for response generation.
            inputs (List[str]): List of inputs corresponding to the instructions.
            outputs (List[str]): List of outputs for evaluation (currently not used in the response generation).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with instructions, inputs, and the generated responses.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        responses = []
        for inst, inp in zip(instructions, inputs):
            # Construct the content based on the availability of input
            if inp:  # When input is provided
                content = (
                    "Given the following instruction and input, please provide a comprehensive and accurate response.\n"
                    "Instruction: {}\n"
                    "Input: {}\n"
                    "Response: "
                ).format(inst, inp)
            else:  # When no input is provided
                content = (
                    "Given the following instruction, please provide a comprehensive and accurate response.\n"
                    "Instruction: {}\n"
                    "Response: "
                ).format(inst)

            responses.append([{"role": "user", "content": content}])
        
        # Prepare the inputs for the LLM generation
        sampling_params = SamplingParams(temperature=0, max_tokens=2048)
        resp_inputs = [
            tokenizer.apply_chat_template(inst, tokenize=False, add_generation_prompt=True) 
            for inst in responses
        ]
        
        # Generate the responses
        resp_outputs = self.llm_engine.generate(resp_inputs, sampling_params)

        # Process and return the final output
        return [
            {
                "instruction": inst, 
                "input": inp, 
                "output": resp_outputs[i].outputs[0].text.strip().lstrip("\n").rstrip("\n")
            }
            for i, (inst, inp) in enumerate(zip(instructions, inputs))
        ]

def main(model_name_or_path: str, inst_file: str, debug: bool) -> None:
    """
    Main function to load instructions, generate responses, and save the results.

    Args:
        model_name_or_path (str): Path to the model or model name.
        inst_file (str): Path to the JSON file containing instructions.
        debug (bool): Whether to enable debug mode, limiting the processing to a small subset of data.
    """
    # Load instruction data from the source file
    try:
        with open(inst_file, "r", encoding='utf-8') as f:
            inst_data = json.load(f)
    except Exception as e:
        print(f"Error reading the input file {inst_file}: {e}")
        return

    # In debug mode, limit to the first 10 instructions for quick testing
    if debug:
        inst_data = inst_data[:10]

    # Instantiate the response generator
    response_generator = ResponseGenerator(model_name_or_path)

    # Generate responses using the loaded instructions and inputs
    responses = response_generator.generate_responses(
        [item["instruction"] for item in inst_data],
        [item["input"] for item in inst_data],
        [item["output"] for item in inst_data]
    )

    # Save the generated responses to the target file
    try:
        with open(inst_file, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=4, ensure_ascii=False)
        print(f"Responses successfully saved to {inst_file}")
    except Exception as e:
        print(f"Error saving responses to {inst_file}: {e}")

if __name__ == "__main__":
    # Argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Run Response Generation")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--inst_file', type=str, required=True, help='Path to the input JSON file with instructions')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (process a smaller subset)')

    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.model_name_or_path, args.inst_file, args.debug)
