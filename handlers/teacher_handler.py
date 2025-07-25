import json
import logging
import os
import time
from services.teacher_orchestrator import teacher_agent

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))

def lambda_handler(event, context):
    """
    Main teacher orchestrator handler - routes to appropriate assistants
    """
    try:
        # Log the incoming event
        logger.info(f"Teacher orchestrator received: {json.dumps(event)}")
        
        # Handle both API Gateway and direct invocation
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
        
        query = body.get('userQuestion', '')
        context_info = body.get('context', {})
        
        # Validate input
        if not query:
            return create_response(400, {'error': 'userQuestion parameter is required'})
        
        start_time = time.time()
        
        # Process through the teacher orchestrator
        response = teacher_agent(query)

        elapsed_time = time.time() - start_time
        logger.info(f"Teacher agent response: {response}")
        logger.info(f"Processing time: {elapsed_time:.2f} seconds")

        response_payload = create_response(200, {
            'agentResponse': str(response),
            'userQuestion': query,
            'context': context_info,
            'stage': os.getenv('STAGE', 'dev'),
            'function': 'teacher-orchestrator'
        })
        print(json.dumps(response_payload))  # helps serverless-offline get correct output

        return response_payload
        
    except Exception as e:
        logger.error("Error in teacher orchestrator", exc_info=True)
        return create_response(500, {'error': f'Internal server error: {str(e)}'})

def create_response(status_code, body):
    """Create standardized API response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
        },
        'body': json.dumps(body)
    }
