from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="ai_news_agent_simple",
    model="gemini-2.0-flash-live-001", # Essential for live voice interaction
    instruction="You are an AI News Assistant. Use Google Search to find recent AI news.",
    tools=[google_search]
)
