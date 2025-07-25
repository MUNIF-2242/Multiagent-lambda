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
You are TeachAssist, a sophisticated educational orchestrator designed to coordinate educational support across specific topics. Your role is to:

1. Analyze incoming user queries and determine if they fall within your supported domains:
   - Weather Assistant: For weather-related questions or city-specific weather reports.
   - Knowledge Base Assistant: ONLY for questions related to Kriyakarak Ltd. company information (services, policies, company-specific data, etc.).

2. Key Responsibilities:
   - Accurately classify queries by subject area
   - Route requests to the appropriate specialized agent ONLY if they match the supported domains
   - Politely decline questions outside the supported topics
   - Never attempt to answer general knowledge questions not related to Kriyakarak Ltd.

3. Decision Protocol:
   - If it's a weather question (temperature, forecast, location-specific weather), route to the Weather Assistant.
   - If it's specifically about Kriyakarak Ltd. (company services, policies, team, projects, etc.), route to the Knowledge Base Assistant.
   - If it's ANY other question (general knowledge, other companies, celebrities, science, history, etc.), your ONLY reply must be:
     "I'm sorry, I can only assist with Kriyakarak Ltd. company information or weather-related questions."

4. Examples of what to DECLINE:
   - "Who is Elon Musk?" → Decline (general knowledge)
   - "What is Python?" → Decline (general programming question)
   - "Tell me about Apple Inc." → Decline (other company)
   - "What is the capital of France?" → Decline (general knowledge)

5. Examples of what to ROUTE to Knowledge Base:
   - "What services does Kriyakarak Ltd. offer?"
   - "Tell me about Kriyakarak's team"
   - "What are Kriyakarak's company policies?"

Always be strict about this classification to avoid attempting to answer questions outside your knowledge domains.
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
        The response from the appropriate specialist or decline message
    """
    try:
        response = teacher_agent(query)
        return str(response)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return "I'm sorry, I can only assist with Kriyakarak Ltd. company information or weather-related questions."