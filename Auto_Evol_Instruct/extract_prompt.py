# Base instruction template for generating new prompts
BASE_INSTRUCTION = (
    "You are tasked with generating a concise summary for the given method of instruction evolution. Please follow the steps below:\n\n"
    "Step 1: Carefully read the given method of instruction evolution and identify the key concept or process it describes.\n"
    "Step 2: Create a short and simple phrase that accurately summarizes the core idea of the method. The summary should be succinct and focus on the essence of the evolution process.\n"
    "Ensure that the phrase does not contain any unnecessary symbols, punctuation, or formatting. It should be just a brief, clear description of the method.\n"
    "Please ignore the numerical labels or special identifiers at the beginning of the methods.\n"
    "Provide only the summary phrase without any further explanation or additional information.\n\n"
    "Method of Instruction Evolution:\n"
    "{}"
)

def create_extract_prompt(instruction: str) -> str:
    """
    Generates a new prompt based on the given instruction by drawing inspiration
    from the base instruction and adding the provided instruction as the given prompt.

    Args:
        instruction (str): The given prompt from which a new prompt is generated.

    Returns:
        str: The complete prompt ready for use.
    """
    # Construct the prompt by combining the base instruction with the given instruction
    new_prompt = BASE_INSTRUCTION.format(instruction)
    
    return new_prompt
