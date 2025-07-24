from strands import Agent, tool
from strands_tools import file_read
import logging
import os
import boto3
import json
from pinecone import Pinecone

# Configure logger
logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# AWS Bedrock and Pinecone environment configs
AWS_REGION = os.getenv("REGION_AWS", "us-east-1")
BEDROCK_EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID")  
BEDROCK_LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")              

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone and Bedrock once
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# Initialize model once (outside handler for reuse)
bedrock_model = None

def get_model():
    global bedrock_model
    if bedrock_model is None:
        # Import your model loader here
        from .model_loader import get_bedrock_model
        bedrock_model = get_bedrock_model()
    return bedrock_model

KNOWLEDGE_ASSISTANT_SYSTEM_PROMPT = """
You are Shellbot, a helpful AI assistant for Shellbeehaken Ltd. You are NOT an Amazon AI assistant.

CRITICAL: Keep responses SHORT and DIRECT. Maximum 2–3 sentences.

**Only use the information provided by the knowledge retrieval tool. If not available, say "I don't have that information available."**

IDENTITY: Only when specifically asked "who are you" or similar identity questions, respond with: "I'm Shellbot, your helpful AI assistant for Shellbeehaken Ltd."

For all other questions: Answer directly without introducing yourself.
For greetings: respond warmly but briefly.
Avoid technical jargon, citations, or markdown.
"""


def create_embedding(text: str):
    try:
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text}),
        )
        body = response["body"].read().decode("utf-8")
        result = json.loads(body)
        return result.get("embedding")
    except Exception as e:
        logger.error(f"Embedding creation failed: {e}")
        return None


def query_pinecone(embedding, top_k=3):
    try:
        result = pinecone_index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
        )
        return result.matches
    except Exception as e:
        logger.error(f"Pinecone query failed: {e}")
        return []


@tool
def retrieve_knowledge(query: str) -> str:
    """
    Retrieve relevant company information from the knowledge base.
    """
    try:
        # Step 1: Embed the query
        embedding = create_embedding(query)
        if not embedding:
            return "Sorry, I couldn't process your request due to embedding failure."

        # Step 2: Search Pinecone for relevant content
        matches = query_pinecone(embedding, top_k=3)
        if not matches:
            return "I don't have that information available."

        # Step 3: Build context from matches
        context = "\n---\n".join(
            match.metadata.get("text", "").strip()
            for match in matches
            if match.metadata and "text" in match.metadata
        )
        if not context:
            return "I don't have that information available."

        return f"Retrieved information:\n{context}"

    except Exception as e:
        logger.error(f"Error in retrieve_knowledge: {e}")
        return "Something went wrong while retrieving information."


@tool
def knowledgebase_assistant(query: str) -> str:
    """
    Answer company-related questions based on internal knowledge from Pinecone and Bedrock.
    """
    try:
        logger.info("Routed to Knowledgebase Assistant")

        formatted_query = f"Answer this company-related question using the knowledge retrieval tool: {query}"

        # Create the knowledge agent
        knowledge_agent = Agent(
            model=get_model(),
            system_prompt=KNOWLEDGE_ASSISTANT_SYSTEM_PROMPT,
            tools=[retrieve_knowledge, file_read]
        )

        agent_response = knowledge_agent(formatted_query)
        return str(agent_response) if agent_response else "I don't have that information available."

    except Exception as e:
        logger.error(f"Error in knowledgebase_assistant: {e}")
        return "Something went wrong while answering your question."