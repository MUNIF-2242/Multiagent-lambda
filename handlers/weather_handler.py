import json
import logging
import os
from services.weather_assistant import weather_assistant, get_weather

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))

def lambda_handler(event, context):
    """
    Serverless weather handler
    """
    try:
        # Log the incoming event
        logger.info(f"Weather handler received: {json.dumps(event)}")
        
        # Handle both API Gateway and direct invocation
        if 'body' in event:
            # API Gateway format
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            # Direct invocation format
            body = event
        
        query = body.get('query', '')
        city = body.get('city', 'Dhaka')
        
        # Validate input
        if not query:
            return create_response(400, {'error': 'Query parameter is required'})
        
        # Process the weather query
        if 'weather' in query.lower():
            response = weather_assistant(query)
        else:
            response = get_weather(city)
        
        # Return successful response
        return create_response(200, {
            'response': response,
            'query': query,
            'city': city,
            'stage': os.getenv('STAGE', 'dev')
        })
        
    except Exception as e:
        logger.error(f"Error in weather handler: {str(e)}")
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