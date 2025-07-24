#!/usr/bin/env python3
"""
# 📁 Teacher's Assistant Strands Agent

A specialized Strands agent that is the orchestrator to utilize sub-agents and tools at its disposal to answer a user query.
"""

from strands import Agent
from .weather_assistant import weather_assistant
from .knowledgebase_assistant import knowledgebase_assistant
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
You are TeachAssist, a sophisticated educational orchestrator designed to coordinate educational support across multiple topics. Your role is to:

1. Analyze incoming user queries and determine the most appropriate specialized agent to handle them:
   - Weather Assistant: For weather-related questions or city-specific weather reports.
   - Knowledge Base Assistant: For all questions related to the custom knowledgebase (facts, concepts, definitions, etc.).

2. Key Responsibilities:
   - Accurately classify queries by subject area
   - Route requests to the appropriate specialized agent
   - Maintain context and coordinate multi-step problems
   - Ensure cohesive responses when multiple agents are needed
   - Politely decline questions outside the supported topics

3. Decision Protocol:
   - If it's a weather question (temperature, forecast, location-specific weather), route to the Weather Assistant.
   - If it's a general educational or informational question, route to the Knowledge Base Assistant.
   - If the user asks something unrelated, your ONLY reply must be:
     "I'm sorry, I can only assist with knowledge base or weather-related questions."

Always confirm your understanding before routing to ensure accurate assistance. Do not attempt to answer queries outside of the supported tools.
"""

# Create the teacher orchestrator agent
teacher_agent = Agent(
    model=get_model(),
    system_prompt=TEACHER_SYSTEM_PROMPT,
    callback_handler=None,
    tools=[
        weather_assistant,
        knowledgebase_assistant
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
