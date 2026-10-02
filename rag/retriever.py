import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
tavily = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))


def get_financial_news(ticker: str) ->str:
    results = tavily.search(
        query = f"{ticker} stock news analysis latest",
        max_result=5,
        search_depth="advanced"
    )
    articles = results.get("results", [])
    context =""
    for i, article in enumerate(articles):
        context += f"\nSource {i+1}: {article['title']}\n{article['content']}\n"
    return context

# news = get_financial_news("MCD")
# print(news)