# Base instruction template for generating new prompts
BASE_INSTRUCTION = (
    "I want you to act as a Prompt Creator.\n"
    "Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\n"
    "This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\n"
    "The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\n"
    "The #Created Prompt# must be reasonable and must be understood and responded to by humans or modern AI chatbots.\n"
    "'You MUST only generate the new prompt without any other words or special symbols.\n"
)

def create_breadth_prompt(instruction: str) -> str:
    """
    Generates a new prompt based on the given instruction by drawing inspiration
    from the base instruction and adding the provided instruction as the given prompt.

    Args:
        instruction (str): The given prompt from which a new prompt is generated.

    Returns:
        str: The complete prompt ready for use.
    """
    # Construct the prompt by combining the base instruction with the given instruction
    new_prompt = f"{BASE_INSTRUCTION}#Given Prompt#: \n{instruction}\n#Created Prompt#:\n"
    
    return new_prompt
