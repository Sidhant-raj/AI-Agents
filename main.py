import asyncio
import json
import sys
import os
import io
from contextlib import redirect_stdout
from typing import Optional
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from smolagents import ToolCallingAgent, CodeAgent
from smolagents import InferenceClientModel, tool
from smolagents import DuckDuckGoSearchTool
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
news_api = os.getenv("news_api")
weather_api = os.getenv("weather_api")

app = FastAPI(title="AI Agent Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def get_weather(location: str, celsius: bool | None = False) -> str:
    """
    Get the current weather at the given location using the WeatherStack API.

    Args:
        location: The location (city name).
        celsius: Whether to return the temperature in Celsius (default is False, Fahrenheit).

    Returns:
        A string describing the current weather at the location.
    """
    units = "m" if celsius else "f"
    url = (
        f"http://api.weatherstack.com/current"
        f"?access_key={weather_api}&query={location}&units={units}"
    )
    try:
        response = requests.get(url, timeout=10)
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

    Args:
        query: Optional keyword or topic to filter news (e.g., 'technology', 'sports').
               Leave empty for general top headlines.
    Returns:
        str: A string containing the top 5 news headlines and their sources.
    """
    url = "https://gnews.io/api/v4/top-headlines"
    params = {
        "country": "in",
        "lang": "en",
        "max": 5,
        "token": news_api,
    }
    if query:
        params["q"] = query
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        if not articles:
            return f"No news available for '{query}'." if query else "No news available."
        return "\n".join(
            f"{a['title']} — {a['source']['name']}" for a in articles
        )
    except requests.exceptions.RequestException as e:
        return f"Error fetching news data: {str(e)}"


# ─── WebSocket streaming callback ─────────────────────────────────────────────

import re as _re

_ANSI_RE = _re.compile(r'\x1b(?:\[[0-9;]*[mGKHFABCDJsuhlr]|\][^\x07]*[\x07\x1b\\]|[()][AB012]|[=>])')

def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub('', s)


class WebSocketStream(io.StringIO):
    """Captures stdout and forwards every line to the WebSocket in real time."""

    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.websocket = websocket
        self.loop = loop
        self._buf = ""

    def write(self, text: str) -> int:
        # Also write to real stdout for server logs
        sys.__stdout__.write(text)
        self._buf += text
        # Send complete lines immediately
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            clean = strip_ansi(line).strip()
            if clean:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send_text(
                        json.dumps({"type": "log", "content": clean})
                    ),
                    self.loop,
                ).result()
        return len(text)

    def flush(self):
        pass


# ─── Agent factory (per-connection so each WS gets fresh agents) ──────────────

def build_agents(model):
    news_agent = ToolCallingAgent(
        tools=[get_news_headlines, DuckDuckGoSearchTool()],
        model=model,
        name="news_agent",
        description=(
            "A specialized agent that fetches top news headlines from GNews API for India "
            "and also searches the web using DuckDuckGo. "
            "Use for news, current events, or topic-based queries."
        ),
    )

    weather_agent = ToolCallingAgent(
        tools=[get_weather],
        model=model,
        name="weather_agent",
        description=(
            "A specialized agent that fetches current weather information for any city. "
            "Use for weather-related queries."
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
""",
    )
    return manager_agent


# ─── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(json.dumps({
        "type": "system",
        "content": "Connected to AI Agent. Ask me about news or weather!",
    }))

    loop = asyncio.get_event_loop()

    if not HF_TOKEN:
        await websocket.send_text(json.dumps({
            "type": "error",
            "content": "HF_TOKEN not set. Please configure your .env file.",
        }))
        await websocket.close()
        return

    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        api_key=HF_TOKEN,
    )
    agent = build_agents(model)

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            query = data.get("message", "").strip()
            if not query:
                continue

            await websocket.send_text(json.dumps({"type": "user", "content": query}))
            await websocket.send_text(json.dumps({"type": "thinking", "content": "Agent is thinking…"}))

            stream = WebSocketStream(websocket, loop)

            def run_agent():
                with redirect_stdout(stream):
                    try:
                        result = agent.run(query)
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text(
                                json.dumps({"type": "answer", "content": str(result)})
                            ),
                            loop,
                        ).result()
                    except Exception as exc:
                        asyncio.run_coroutine_threadsafe(
                            websocket.send_text(
                                json.dumps({"type": "error", "content": str(exc)})
                            ),
                            loop,
                        ).result()

            await asyncio.get_event_loop().run_in_executor(None, run_agent)

    except WebSocketDisconnect:
        pass


# ─── REST health check ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "message": "AI Agent API is running"}


# ─── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            return HTMLResponse(content=f.read().decode("utf-8"))
    return HTMLResponse(content="<h1>Place index.html next to main.py</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)