import json
import argparse
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PROMPT_eval_single="""
[User]
{user_query}
[End of User]
[Assistant 1]
{assistant1}
[End of Assistant 1]
[Assistant 2]
{assistant2}
[End of Assistant 2]
[System]
We would like to request your feedback on the two dialogues shown above between an AI assistant and a user.
Focus on the AI's responses. The AI's responses should perfectly align with the user's needs. Additionally, the responses should be concise and to the point, avoiding unnecessary details or excessive information, while still being as comprehensive as possible in addressing the user's query. The answers must maintain good logical flow, use precise technical terms, and be factually accurate and objective.
Based on the above criteria, compare the performance of Assistant 1 and Assistant 2. Determine which one is "better than," "worse than," or "equal to" the other. First, compare their responses and analyze which aligns better with the stated requirements.

On the last line, output a single label only, selecting from one of the following:
'Assistant 1 is better than Assistant 2'
'Assistant 1 is worse than Assistant 2'
'Assistant 1 is equal to Assistant 2'
"""

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

    def generate_responses(self, instructions: List[str], inputs: List[str], outputs: List[str], others: List[str]) -> List[Dict[str, Any]]:
        """
        Generate responses for the provided instructions and inputs.

        Args:
            instructions (List[str]): List of instructions to be used for response generation.
            inputs (List[str]): List of inputs corresponding to the instructions.
            outputs (List[str]): List of outputs for evaluation (currently not used in the response generation).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with instructions, inputs, and the generated responses.
        """
        win_num = 0
        tie_num = 0
        lose_num = 0

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        responses = []
        for inst, inp, out, oth in zip(instructions, inputs, outputs, others):
            # Construct the content based on the availability of input
            if inp:  # When input is provided
                content = PROMPT_eval_single.format(user_query=inst + "\n" + inp, assistant1=out, assistant2=oth)
            else:  # When no input is provided
                content = PROMPT_eval_single.format(user_query=inst, assistant1=out, assistant2=oth)

            responses.append([{"role": "user", "content": content}])
        
        # Prepare the inputs for the LLM generation
        sampling_params = SamplingParams(temperature=0, max_tokens=2048)
        resp_inputs = [
            tokenizer.apply_chat_template(inst, tokenize=False, add_generation_prompt=True) 
            for inst in responses
        ]
        
        # Generate the responses
        resp_outputs = self.llm_engine.generate(resp_inputs, sampling_params)

        resp_outputs = [resp_outputs[i].outputs[0].text.strip().lstrip("\n").rstrip("\n") for i in range(len(resp_outputs))]

        for resp in resp_outputs:
            if 'Assistant 1 is better than Assistant 2' in resp:
                win_num += 1
            elif 'Assistant 1 is worse than Assistant 2' in resp:
                lose_num += 1
            else:
                tie_num += 1

        # Process and return the final output
        return win_num, lose_num, tie_num


def main(model_name_or_path: str, inst_file: str, other_file: str, debug: bool) -> None:
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
        with open(other_file, "r", encoding='utf-8') as f:
            other_data = json.load(f)
    except Exception as e:
        print(f"Error reading the input file {inst_file}: {e}")
        return

    # In debug mode, limit to the first 10 instructions for quick testing
    if debug:
        inst_data = inst_data[:10]

    # Instantiate the response generator
    response_generator = ResponseGenerator(model_name_or_path)

    # Generate responses using the loaded instructions and inputs
    win_num, lose_num, tie_num = response_generator.generate_responses(
        [item["instruction"] for item in inst_data],
        [item["input"] for item in inst_data],
        [item["output"] for item in inst_data],
        [item["output"] for item in other_data]
    )

    print(f"win num: {win_num}, tie num: {tie_num}, lose_num: {lose_num}")


if __name__ == "__main__":
    # Argument parsing for command-line execution
    parser = argparse.ArgumentParser(description="Run Response Generation")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--inst_file', type=str, required=True, help='Path to the input JSON file with instructions')
    parser.add_argument('--other_file', type=str, required=True, help='Path to the input JSON file with instructions')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (process a smaller subset)')

    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.model_name_or_path, args.inst_file, args.other_file, args.debug)
