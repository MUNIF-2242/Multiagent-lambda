from strands import Agent, tool
from strands_tools import calculator, current_time
import requests
import os
from .model_loader import get_bedrock_model

# Initialize model once (outside handler for reuse)
bedrock_model = None

def get_model():
    global bedrock_model
    if bedrock_model is None:
        bedrock_model = get_bedrock_model()
    return bedrock_model

WEATHER_ASSISTANT_SYSTEM_PROMPT = """
You are WeatherBot, a helpful and reliable weather assistant.

Your duties include:
1. Providing current weather conditions for a given city.
2. Presenting temperature, condition, humidity, wind speed, and visibility.
3. Explaining any technical terms to users if needed.
4. If data is unavailable, respond politely and suggest checking the city name.

Always be concise, clear, and friendly. Use emojis for clarity when appropriate.
"""

@tool
def get_weather(city: str = "Dhaka") -> str:
    """
    Fetch current weather data for a given city.
    """
    try:
        base_url = f"http://wttr.in/{city}"
        params = {'format': 'j1'}
        
        # Add timeout for Lambda
        response = requests.get(base_url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            location = data['nearest_area'][0]
            report = f"""
Current Weather in {location['areaName'][0]['value']}, {location['country'][0]['value']}:
🌡️ Temperature: {current['temp_C']}°C (Feels like {current['FeelsLikeC']}°C)
🌤️ Condition: {current['weatherDesc'][0]['value']}
💧 Humidity: {current['humidity']}%
🌬️ Wind: {current['windspeedKmph']} km/h
👁️ Visibility: {current['visibility']} km
            """.strip()
            return report
        else:
            return f"⚠️ Could not fetch weather data for '{city}'. Please check the city name."
    except Exception as e:
        return f"❌ Error retrieving weather: {str(e)}"

@tool
def weather_assistant(query: str) -> str:
    """
    Weather Assistant tool to process and respond to weather-related queries.
    """
    print("📡 Routed to Weather Assistant")

    formatted_query = f"Respond to this weather-related query with accurate data and friendly tone: {query}"

    try:
        weather_agent = Agent(
            model=get_model(),
            system_prompt=WEATHER_ASSISTANT_SYSTEM_PROMPT,
            tools=[get_weather, calculator, current_time]
        )

        agent_response = weather_agent(formatted_query)
        return str(agent_response) if agent_response else "❓ I couldn't understand your weather query. Please try again."
    except Exception as e:
        return f"❌ Error handling the query: {str(e)}"