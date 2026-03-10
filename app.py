import requests
from smolagents import ToolCallingAgent, CodeAgent
from smolagents import InferenceClientModel, TransformersModel, tool
from smolagents import DuckDuckGoSearchTool
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
news_api = os.getenv("news_api")
weather_api = os.getenv("weather_api")

model = InferenceClientModel(model_id="Qwen/Qwen2.5-7B-Instruct", api_key=HF_TOKEN)
# model = TransformersModel(model_id="Qwen/Qwen2.5-7B-Instruct", api_key=HF_TOKEN)

@tool
def get_weather(location: str, celsius: bool | None = False) -> str:
    """
    Get the current weather at the given location using the WeatherStack API.

    Args:
        location: The location (city name).
        celsius: Whether to return the temperature in Celsius (default is False, which returns Fahrenheit).

    Returns:
        A string describing the current weather at the location.
    """
    units = "m" if celsius else "f"
    url = f"http://api.weatherstack.com/current?access_key={weather_api}&query={location}&units={units}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            return f"Error: {data['error'].get('info', 'Unable to fetch weather data.')}"

        weather = data["current"]["weather_descriptions"][0]
        temp = data["current"]["temperature"]
        temp_unit = "°C" if celsius else "°F"

        return f"The current weather in {location} is {weather} with a temperature of {temp} {temp_unit}."

    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"


@tool
def get_news_headlines(query: str = "") -> str:
    """
    Fetches the top news headlines from the GNews API for India.
    You can optionally filter by a topic or keyword using the query parameter.

    Args:
        query: Optional keyword or topic to filter news (e.g., 'technology', 'sports', 'politics').
               Leave empty for general top headlines.
    Returns:
        str: A string containing the top 5 news headlines and their sources, or an error message.
    """
    url = "https://gnews.io/api/v4/top-headlines"
    params = {
        "country": "in",
        "lang": "en",
        "max": 5,
        "token": news_api
    }

    if query:
        params["q"] = query

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            return f"No news available for '{query}'." if query else "No news available at the moment."

        headlines = [f"{article['title']} - {article['source']['name']}" for article in articles]
        return "\n".join(headlines)

    except requests.exceptions.RequestException as e:
        return f"Error fetching news data: {str(e)}"


news_agent = ToolCallingAgent(
    tools=[
        get_news_headlines,
        DuckDuckGoSearchTool(),
    ],
    model=model,
    name="news_agent",
    description=(
        "A specialized agent that fetches top news headlines from GNews API for India "
        "and also searches the web using DuckDuckGo for additional context or recent info. "
        "Use this agent for any news, current events, or topic-based information queries."
    ),
)

weather_agent = ToolCallingAgent(
    tools=[get_weather],
    model=model,
    name="weather_agent",
    description=(
        "A specialized agent that fetches current weather information for any city. "
        "Use this agent for weather-related queries."
    ),
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[news_agent, weather_agent],
    instructions="""
IMPORTANT RULES FOR FINAL ANSWER:
- Always return the COMPLETE and DETAILED response from your sub-agents.
- Do NOT summarize or shorten the sub-agent's response.
- Your final_answer must be descriptive, well-structured, and at least 150-200 words.
- Include all headlines, explanations, and full context provided by the sub-agents.
- Format the final answer clearly with proper sections and bullet points.
- Never condense rich information into a single short sentence.
"""
)
manager_agent.run("Give me the top news related to war of iran and israel also search on internet.")
# manager_agent.run("Current weather in pragati vihar hostel, delhi in celsius")