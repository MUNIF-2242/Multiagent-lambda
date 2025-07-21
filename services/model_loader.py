# model_loader.py

from strands.models import BedrockModel
import os

def get_bedrock_model():
    aws_region = os.getenv('REGION_AWS', 'us-east-1')
    
    return BedrockModel(
        model_id="amazon.nova-lite-v1:0",
        temperature=0.7,
        region_name=aws_region
    )
