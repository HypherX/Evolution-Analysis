# Base instruction template for rewriting prompts
BASE_INSTRUCTION = (
    "I want you to act as a Prompt Rewriter.\n"
    "Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems "
    "(e.g., ChatGPT and GPT-4) a bit harder to handle.\n"
    "But the rewritten prompt must be reasonable and must be understood and responded to by humans.\n"
    "You SHOULD complicate the given prompt using the following method:\n"
    "{method}\n"
    "You should try your best not to make the #Rewritten Prompt# become verbose, "
    "#Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#.\n"
    "You MUST only generate the new prompt without #The Given Prompt# and #Rewritten Prompt#.\n"
)

def create_constraints_prompt(instruction: str) -> str:
    """
    Generates a prompt that asks the AI to add constraints or requirements to the given prompt.
    
    Args:
        instruction (str): The original prompt that needs to be rewritten.

    Returns:
        str: The new prompt with added constraints.
    """
    method = "Please add one more constraint/requirement into #The Given Prompt#"
    return generate_rewritten_prompt(instruction, method)

def create_deepen_prompt(instruction: str) -> str:
    """
    Generates a prompt that asks the AI to deepen the inquiry of the given prompt.
    
    Args:
        instruction (str): The original prompt that needs to be rewritten.

    Returns:
        str: The new prompt with deepened inquiry.
    """
    method = "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased."
    return generate_rewritten_prompt(instruction, method)

def create_concretizing_prompt(instruction: str) -> str:
    """
    Generates a prompt that asks the AI to replace general concepts with more specific ones.
    
    Args:
        instruction (str): The original prompt that needs to be rewritten.

    Returns:
        str: The new prompt with concretized concepts.
    """
    method = "Please replace general concepts with more specific concepts."
    return generate_rewritten_prompt(instruction, method)

def create_reasoning_prompt(instruction: str) -> str:
    """
    Generates a prompt that asks the AI to explicitly request multiple-step reasoning if the problem can be solved simply.
    
    Args:
        instruction (str): The original prompt that needs to be rewritten.

    Returns:
        str: The new prompt with explicit multiple-step reasoning.
    """
    method = "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning."
    return generate_rewritten_prompt(instruction, method)

def generate_rewritten_prompt(instruction: str, method: str) -> str:
    """
    A helper function to generate the rewritten prompt with the given instruction and method.
    
    Args:
        instruction (str): The original prompt that needs to be rewritten.
        method (str): The method used to complicate the prompt (e.g., adding constraints, deepening, etc.).

    Returns:
        str: The newly generated rewritten prompt.
    """
    prompt = BASE_INSTRUCTION.format(method=method)
    prompt += f"#The Given Prompt#: \n{instruction} \n"
    prompt += "#Rewritten Prompt#:\n"
    return prompt
