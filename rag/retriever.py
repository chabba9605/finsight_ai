import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from tavily import TavilyClient


load_dotenv()


TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)

if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY is not set. "
        "Add it to your .env file."
    )


tavily = TavilyClient(
    api_key=TAVILY_API_KEY
)


def get_financial_news(
    ticker: str,
) -> Tuple[str, List[str]]:
    """
    Retrieve recent financial/news information for a ticker.

    Returns:
        context: formatted text for the LLM
        sources: list of source URLs
    """

    ticker = ticker.upper().strip()

    results = tavily.search(
        query=f"{ticker} stock news analysis latest",
        max_results=5,
        search_depth="advanced",
    )

    articles = results.get(
        "results",
        [],
    )

    context_parts = []
    sources = []

    for index, article in enumerate(
        articles,
        start=1,
    ):
        title = article.get(
            "title",
            "Untitled source",
        )

        content = article.get(
            "content",
            "",
        )

        url = article.get(
            "url",
            "",
        )

        context_parts.append(
            f"Source {index}: {title}\n"
            f"{content}"
        )

        if url:
            sources.append(url)

    context = "\n\n".join(
        context_parts
    )

    return context, sources