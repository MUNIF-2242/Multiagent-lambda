from strands import Agent, tool
from strands_tools import calculator
from .model_loader import get_bedrock_model
import os
import logging

# Initialize model once (outside handler for reuse)
bedrock_model = None
# Configure logging
logger = logging.getLogger()
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
def get_model():
    global bedrock_model
    if bedrock_model is None:
        bedrock_model = get_bedrock_model()
    return bedrock_model

MATH_ASSISTANT_SYSTEM_PROMPT = """
You are math wizard, a specialized mathematics education assistant. Your capabilities include:

1. Mathematical Operations:
   - Arithmetic calculations
   - Algebraic problem-solving
   - Geometric analysis
   - Statistical computations

2. Teaching Tools:
   - Step-by-step problem solving
   - Visual explanation creation
   - Formula application guidance
   - Concept breakdown

3. Educational Approach:
   - Show detailed work
   - Explain mathematical reasoning
   - Provide alternative solutions
   - Link concepts to real-world applications

Focus on clarity and systematic problem-solving while ensuring students understand the underlying concepts.
"""

@tool
def math_assistant(query: str) -> str:
    """
    Process and respond to math-related queries using a specialized math agent.
    """
    formatted_query = f"Please solve the following mathematical problem, showing all steps and explaining concepts clearly: {query}"
    
    try:
        # print("Routed to Math Assistant")
        logger.info("Routed to Math Assistant")

        math_agent = Agent(
            model=get_model(),
            system_prompt=MATH_ASSISTANT_SYSTEM_PROMPT,
            tools=[calculator],
        )
        agent_response = math_agent(formatted_query)
        text_response = str(agent_response)

       
        # logger.info(f"agent_response (str)pulok: {str(agent_response)}")
        # logger.info(f"agent_response (repr)munif: {repr(agent_response)}")

        if len(text_response) > 0:
            return text_response

        return "I apologize, but I couldn't solve this mathematical problem. Please check if your query is clearly stated or try rephrasing it."
    except Exception as e:
        return f"Error processing your mathematical query: {str(e)}"