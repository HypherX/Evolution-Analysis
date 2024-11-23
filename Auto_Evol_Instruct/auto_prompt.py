# Base instruction template for generating new prompts
BASE_INSTRUCTION = (
    "You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version.\n"
    "Please follow the steps below to rewrite the given #Instruction# into a more complex version.\n\n"
    "Step 1: Please read the #Instruction# carefully and list all the possible methods to "
    "make this instruction more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). "
    "Please do not provide methods to change the language of the instruction!\n\n"
    "Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# "
    "more complex. The plan should include several methods from the #Methods List#.\n\n"
    "Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only "
    "add 10 to 20 words into the #Instruction#.\n\n"
    "Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. "
    "Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#. "
    "Just provide the #Final Rewritten Instruction# without any explanation.\n\n"
    "Please reply strictly in the following format:\n\n"
    "Step 1 #Methods List#:\n"
    "Step 2 #Plan#:\n"
    "Step 3 #Rewritten Instruction#:\n"
    "Step 4 #Finally Rewritten Instruction#:\n\n"
    "#Instruction#:\n"
    "{}"
)

def create_auto_prompt(instruction: str) -> str:
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
