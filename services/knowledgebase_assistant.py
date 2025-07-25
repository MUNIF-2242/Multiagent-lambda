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
        from .model_loader import get_bedrock_model
        bedrock_model = get_bedrock_model()
    return bedrock_model

KNOWLEDGE_ASSISTANT_SYSTEM_PROMPT = """
You are kriyabot, a helpful AI assistant for Kriyakarak Ltd.

CRITICAL: Keep responses SHORT and DIRECT. Maximum 2–3 sentences.

**STRICT RULE: Only use the information provided by the knowledge retrieval tool. If the retrieved information doesn't contain relevant details, say "I don't have that information available in our company knowledge base."**

IDENTITY: Only when asked "who are you", respond with: "I'm kriyabot, your helpful AI assistant for Kriyakarak Ltd."

For all other questions: Answer directly without introducing yourself.
Avoid technical jargon, citations, or markdown.
"""

def create_embedding(text: str):
    try:
        logger.info(f"Creating embedding for: {text}")
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text}),
        )
        body = response["body"].read().decode("utf-8")
        result = json.loads(body)
        embedding = result.get("embedding")
        logger.info(f"✅ Embedding created successfully, length: {len(embedding) if embedding else 0}")
        return embedding
    except Exception as e:
        logger.error(f"❌ Embedding creation failed: {e}")
        return None

def query_pinecone(embedding, top_k=3):
    try:
        logger.info(f"Querying Pinecone with top_k={top_k}")
        result = pinecone_index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
        )
        
        logger.info(f"✅ Pinecone returned {len(result.matches)} matches")
        
        # Log match details for debugging
        for i, match in enumerate(result.matches):
            logger.info(f"Match {i}: score={match.score}, metadata_keys={list(match.metadata.keys()) if match.metadata else []}")
            if match.metadata and 'text' in match.metadata:
                text_snippet = match.metadata['text'][:100] + "..." if len(match.metadata['text']) > 100 else match.metadata['text']
                logger.info(f"Match {i} text snippet: {text_snippet}")
        
        return result.matches
    except Exception as e:
        logger.error(f"❌ Pinecone query failed: {e}")
        return []

@tool
def retrieve_knowledge(query: str) -> str:
    """
    Retrieve relevant Kriyakarak Ltd. company information from the knowledge base.
    """
    try:
        logger.info(f"🔍 retrieve_knowledge called with query: {query}")
        
        # Step 1: Embed the query
        embedding = create_embedding(query)
        if not embedding:
            logger.error("❌ Failed to create embedding")
            return "Sorry, I couldn't process your request due to embedding failure."

        # Step 2: Search Pinecone for relevant content
        matches = query_pinecone(embedding, top_k=5)  # Increased to 5 for better debugging
        if not matches:
            logger.warning("⚠️ No matches found in Pinecone")
            return "I don't have that information available in our company knowledge base."

        # Step 3: Check if we have any decent matches (lowered threshold for debugging)
        relevant_matches = [match for match in matches if match.score > 0.2]  # Lowered threshold
        
        logger.info(f"📊 Found {len(relevant_matches)} matches above score threshold 0.5")
        
        if not relevant_matches:
            logger.warning(f"⚠️ No matches above threshold. Best score was: {matches[0].score if matches else 'N/A'}")
            return "I don't have that information available in our company knowledge base."

        # Step 4: Build context from matches
        context_parts = []
        for match in relevant_matches:
            if match.metadata and "text" in match.metadata:
                text = match.metadata["text"].strip()
                if text:
                    context_parts.append(f"[Score: {match.score:.3f}] {text}")

        if not context_parts:
            logger.warning("⚠️ No valid text content found in matches")
            return "I don't have that information available in our company knowledge base."

        context = "\n---\n".join(context_parts)
        logger.info(f"✅ Retrieved context length: {len(context)} characters")
        
        return f"Retrieved information:\n{context}"

    except Exception as e:
        logger.error(f"❌ Error in retrieve_knowledge: {e}")
        return "Something went wrong while retrieving information."

@tool
def knowledgebase_assistant(query: str) -> str:
    """
    Answer Kriyakarak Ltd. company-related questions based on internal knowledge.
    """
    try:
        logger.info(f"🚀 knowledgebase_assistant called with query: {query}")

        formatted_query = f"Answer this Kriyakarak Ltd. company question: {query}"

        # Create the knowledge agent
        knowledge_agent = Agent(
            model=get_model(),
            system_prompt=KNOWLEDGE_ASSISTANT_SYSTEM_PROMPT,
            tools=[retrieve_knowledge, file_read]
        )

        logger.info("📞 Calling knowledge agent...")
        agent_response = knowledge_agent(formatted_query)
        
        response_str = str(agent_response) if agent_response else "I don't have that information available."
        logger.info(f"✅ Knowledge agent response: {response_str[:200]}...")
        
        return response_str

    except Exception as e:
        logger.error(f"❌ Error in knowledgebase_assistant: {e}")
        return "Something went wrong while answering your question."