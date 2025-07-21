#!/usr/bin/env python3
"""
# 📁 Teacher's Assistant Strands Agent

A specialized Strands agent that is the orchestrator to utilize sub-agents and tools at its disposal to answer a user query.
"""

from strands import Agent
from strands_tools import file_read, file_write, editor
from .math_assistant import math_assistant
from .weather_assistant import weather_assistant
from .model_loader import get_bedrock_model
import os
import logging



# Initialize model once (outside handler for reuse)
bedrock_model = None
logger = logging.getLogger()
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))

def get_model():
    global bedrock_model
    if bedrock_model is None:
        bedrock_model = get_bedrock_model()
    return bedrock_model

# Define a focused system prompt for the orchestrator
TEACHER_SYSTEM_PROMPT = """
You are TeachAssist, a sophisticated educational orchestrator designed to coordinate educational support across multiple subjects. Your role is to:

1. Analyze incoming student queries and determine the most appropriate specialized agent to handle them:
   - Math Agent: For mathematical calculations, problems, and concepts
   - English Agent: For writing, grammar, literature, and composition
   - Language Agent: For translation and language-related queries
   - Computer Science Agent: For programming, algorithms, data structures, and code execution
   - Weather Assistant: For weather-related questions or city-specific weather reports
   - General Assistant: For all other topics outside these specialized domains

2. Key Responsibilities:
   - Accurately classify student queries by subject area
   - Route requests to the appropriate specialized agent
   - Maintain context and coordinate multi-step problems
   - Ensure cohesive responses when multiple agents are needed

3. Decision Protocol:
   - If query involves calculations/numbers → Math Agent
   - If query involves writing/literature/grammar → English Agent
   - If query involves translation → Language Agent
   - If query involves programming/coding/algorithms/computer science → Computer Science Agent
   - If it's about weather or a city's climate → Weather Assistant
   - If query is outside these specialized areas → General Assistant
   - For complex queries, coordinate multiple agents as needed

Always confirm your understanding before routing to ensure accurate assistance.
"""

# Create the teacher orchestrator agent
teacher_agent = Agent(
    model=get_model(),
    system_prompt=TEACHER_SYSTEM_PROMPT,
    callback_handler=None,
    tools=[
        math_assistant, 
        weather_assistant
    ],
)

# Function to handle queries (for Lambda)
def process_query(query: str, context: dict = None) -> str:
    """
    Process a query through the teacher orchestrator
    
    Args:
        query: The user's question
        context: Optional context information
        
    Returns:
        The response from the appropriate specialist
    """
    try:
        response = teacher_agent(query)
        return str(response)
    except Exception as e:
        return f"Error processing query: {str(e)}"