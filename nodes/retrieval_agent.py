from tavily import TavilyClient
from dotenv import load_dotenv
from components.state import FinSightState
import os

load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def retrieval_agent(state: FinSightState) -> FinSightState:
    results = tavily.search(
        query=f"{state['ticker']} stock news analysis",
        max_results=5,
        search_depth="advanced"
    )
    articles = results.get("results", [])
    context = "\n".join([f"{a['title']}: {a['content']}" for a in articles])
    sources = [a["url"] for a in articles]
    state["retrieved_context"] = context
    state["sources"] = sources
    return state